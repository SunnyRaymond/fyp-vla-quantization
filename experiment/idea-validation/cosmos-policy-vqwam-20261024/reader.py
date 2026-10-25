"""Frozen joint-DiT suffix reader with an isolated student-future cotangent."""
from __future__ import annotations
import inspect

from common import torch_module, endpoint, cpu_tree, check_tensor_parity


def _counter(function, seen=None, depth=0):
    """Locate the official endpoint's NFE counter through the sampler closures."""
    seen = set() if seen is None else seen
    if depth > 10 or id(function) in seen:
        return None
    seen.add(id(function))
    if not inspect.isfunction(function):
        return None
    for cell in function.__closure__ or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, dict) and 'sampler_evaluations' in value:
            return value
        if inspect.isfunction(value):
            result = _counter(value, seen, depth + 1)
            if result is not None:
                return result
    return None


def _clone(value):
    torch = torch_module()
    if torch.is_tensor(value):
        with torch.inference_mode(False):
            return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value, (tuple, list)):
        return type(value)(_clone(v) for v in value)
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    return value


class LoopCapture:
    """Capture an actual solver boundary, retaining its original continuation."""
    def __init__(self, cut=3, retain_graph=False):
        self.cut, self.retain_graph = int(cut), bool(retain_graph)
        self.body = self.state = self.live_x = self.start = self.upper = self.nonlocals = None

    def __enter__(self):
        from cosmos_policy._src.imaginaire.modules import res_sampler
        self.module = res_sampler
        self.original = res_sampler.fori_loop

        def loop(lower, upper, body_fun, init_val):
            counter = _counter(body_fun)
            if counter is None:
                raise RuntimeError('Cannot locate the original sampler evaluation counter')

            def wrapped(index, state):
                if self.body is None and int(counter['sampler_evaluations']) + 1 == self.cut:
                    self.body, self.start, self.upper = body_fun, int(index), int(upper)
                    self.nonlocals = inspect.getclosurevars(body_fun).nonlocals
                    self.live_x = state[0] if self.retain_graph else _clone(state[0])
                    self.state = _clone(state)
                return body_fun(index, state)
            return self.original(lower, upper, wrapped, init_val)

        res_sampler.fori_loop = loop
        return self

    def __exit__(self, *_):
        self.module.fori_loop = self.original

    def require(self):
        if self.body is None:
            raise RuntimeError('Requested reader cut is not an actual original solver boundary')
        if 'x0_fn' not in self.nonlocals or 'sigmas_L' not in self.nonlocals:
            raise RuntimeError('Original solver continuation lacks the denoiser or sigma state')
        return self


def future_mask(value):
    torch = torch_module()
    if value.ndim != 5 or value.shape[2] != 9:
        raise RuntimeError('The fixed two-camera Cosmos LIBERO latent layout must have nine slots')
    mask = torch.zeros((value.shape[0], 1, value.shape[2], 1, 1),
                       device=value.device, dtype=torch.bool)
    mask[:, :, 6:8] = True
    return mask


def action_from_latent(runtime, generated):
    torch = torch_module()
    if generated.ndim != 5 or generated.shape[2] != 9:
        raise RuntimeError('Unexpected frozen Cosmos latent-slot layout')
    size = generated.shape[0]
    width = int(runtime.protocol['inference']['chunk_size']) * 7
    slot = generated[:, :, 4].reshape(size, -1)
    chunks = slot.shape[-1] // width
    action = slot[:, :chunks * width].reshape(size, chunks, width // 7, 7).mean(1)
    if runtime.protocol['inference']['unnormalize_actions']:
        lo = torch.as_tensor(runtime.dataset_stats['actions_min'], device=action.device).float()
        hi = torch.as_tensor(runtime.dataset_stats['actions_max'], device=action.device).float()
        action = .5 * (action.float() + 1) * (hi - lo) + lo
    return action


def reader_kl(runtime, action, teacher_action, statistics):
    from softbin_loss import pointwise_kl
    torch = torch_module()
    cfg = runtime.protocol['training']['reader']
    config = {'bins': cfg['soft_bins'], 'interior_min': cfg['interior_min'],
              'interior_max': cfg['interior_max'], 'sigma': cfg['sigma'],
              'std_floor': cfg['std_floor']}
    lo = statistics['actions_min'].to(action.device)
    hi = statistics['actions_max'].to(action.device)
    denominator = (hi - lo).clamp_min(1e-12)
    student = (2 * (action.float() - lo) / denominator - 1)[..., :6]
    teacher = (2 * (teacher_action.to(action.device).float() - lo) / denominator - 1)[..., :6]
    return pointwise_kl(torch, student, teacher, statistics['mean'], statistics['scale'], -1, config)


class FrozenSuffixReader:
    """Teacher parameters must be restored/frozen before construction and reading."""
    def __init__(self, runtime, context):
        torch = torch_module()
        self.runtime = runtime
        if any(v.requires_grad for v in runtime.model.parameters()):
            raise RuntimeError('Reader requires frozen teacher parameters')
        with torch.inference_mode(False), torch.enable_grad():
            with LoopCapture(runtime.protocol['training']['reader']['cut_evaluation']) as capture:
                self.teacher = endpoint(runtime, context, grad_enabled=True, checkpoint_denoiser=False)
        self.capture = capture.require()
        # This independently validates the fixed action-slot extraction used after a suffix.
        extraction = check_tensor_parity(action_from_latent(runtime, self.teacher.generated_latent),
                                         self.teacher.action, atol=1e-6)
        if not extraction['passed']:
            raise RuntimeError('Reader action-slot extraction differs from the official endpoint')

    def read(self, future_input):
        torch = torch_module()
        capture = self.capture
        state = _clone(capture.state)
        teacher_x = state[0]
        if future_input.shape != teacher_x.shape:
            raise ValueError('Teacher/student noisy state shapes differ at the reader cut')
        hybrid = torch.where(future_mask(teacher_x), future_input, teacher_x)
        state = type(state)((hybrid, *state[1:]))
        for index in range(capture.start, capture.upper):
            state = capture.body(index, state)
        variables = capture.nonlocals
        x0_fn, sigmas = variables['x0_fn'], variables['sigmas_L']
        ones = variables.get('ones_B')
        if ones is None:
            ones = torch.ones(state[0].shape[0], device=state[0].device, dtype=torch.float64)
        generated = x0_fn(state[0], sigmas[-1] * ones).to(self.teacher.generated_latent.dtype)
        return action_from_latent(self.runtime, generated)

    def identity(self):
        torch = torch_module()
        with torch.inference_mode(False), torch.enable_grad():
            # Validate the same input-gradient path used to obtain the reader cotangent.
            leaf = _clone(self.capture.state[0]).requires_grad_(True)
            action = self.read(leaf)
        return {**check_tensor_parity(action, self.teacher.action, atol=1e-6),
                'input_gradient_path': True}

    def cotangent(self, student_state, statistics):
        torch = torch_module()
        with torch.inference_mode(False), torch.enable_grad():
            leaf = student_state.detach().to(self.capture.state[0].device).clone().requires_grad_(True)
            action = self.read(leaf)
            loss = reader_kl(self.runtime, action, self.teacher.action.detach(), statistics)
            gradient, = torch.autograd.grad(loss, leaf)
        mask = future_mask(gradient)
        if not bool(torch.isfinite(gradient).all()) or bool((gradient.masked_select(~mask.expand_as(gradient)) != 0).any()):
            raise RuntimeError('Reader gradient is non-finite or escaped the future-only intervention')
        diagnostics = {'reader_kl': float(loss.detach()), 'gradient_norm': float(gradient.norm()),
                       'finite': True, 'outside_future_gradient_max_abs': 0.,
                       'reader_motor_rmse': float((action[..., :6].float()
                             - self.teacher.action[..., :6].float()).square().mean().sqrt())}
        return cpu_tree(gradient), diagnostics, cpu_tree(action)
