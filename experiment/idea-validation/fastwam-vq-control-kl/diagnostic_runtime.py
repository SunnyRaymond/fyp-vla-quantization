"""Frozen multi-arm runtime for sensor same-state diagnostics.

This module imports the model stack only after construction, so its toy
self-checks remain CPU-only and do not initialize CUDA or LIBERO.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random


ARMS = ("bf16", "per_channel_clip", "softbin", "relational")
QUANTIZED_ARMS = ("per_channel_clip", "softbin", "relational")


def max_abs_difference(left, right):
    """Maximum absolute elementwise difference for nested numeric sequences."""
    def flatten(value):
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().tolist()
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from flatten(item)
        else:
            yield float(value)

    a, b = list(flatten(left)), list(flatten(right))
    if len(a) != len(b) or not a:
        raise ValueError("Compared outputs must have the same nonempty shape")
    return max(abs(x - y) for x, y in zip(a, b))


def require_exact_first_chunk(candidate, source):
    error = max_abs_difference(candidate, source)
    if error != 0.0:
        raise RuntimeError(f"Source first-chunk parity failed: max_abs={error}")
    return error


def validate_binding(saved, expected):
    """Reject saved outputs whose query identity does not match the request."""
    fields = ("protocol", "unit", "index", "arm", "variant", "seed",
              "description", "input_identity")
    mismatch = [key for key in fields if saved.get(key) != expected.get(key)]
    if mismatch:
        raise ValueError("Saved query binding differs: " + ", ".join(mismatch))
    return True


def tensor_identity(torch, value):
    tensor = value.detach().contiguous()
    raw = tensor.view(torch.uint8).cpu().numpy().tobytes()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(raw).hexdigest()}


def datum_identity(torch, datum):
    return {"image": tensor_identity(torch, datum["image"]),
            "proprio": tensor_identity(torch, datum["proprio"])}


def install_compensated_from_base(modules, decoded_base_weights, log2_scales,
                                  compensated_weight):
    """Assign D-compensated BF16 weights from immutable decoded base tensors."""
    if set(modules) != set(decoded_base_weights) or set(modules) != set(log2_scales):
        raise ValueError("Compensation maps must cover the same modules")
    for name, module in modules.items():
        # Always source from the unscaled base reference: re-activation cannot
        # compound a previously assigned D.
        log2 = log2_scales[name]
        base = decoded_base_weights[name].to(device=log2.device)
        compensated = compensated_weight(base, log2)
        module.weight.data = compensated
        del base, compensated


def _copy_image(image):
    copier = getattr(image, "copy", None)
    return copier() if callable(copier) else image


class NoiseTap:
    """Observe pre-transform PIL inputs without changing the actual transform."""

    FUNCTIONS = ("motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur")

    def __init__(self):
        self.calls = 0
        self.last_clean = None
        self.last_function = None
        self.last_severity = None
        self._module = None
        self._originals = {}

    def clear(self):
        self.calls = 0
        self.last_clean = None
        self.last_function = None
        self.last_severity = None

    def install(self, module):
        if self._module is not None:
            raise RuntimeError("NoiseTap is already installed")
        self._module = module
        for name in self.FUNCTIONS:
            original = getattr(module, name)
            self._originals[name] = original

            def tapped(image, *args, _name=name, _original=original, **kwargs):
                self.calls += 1
                self.last_clean = _copy_image(image)
                self.last_function = _name
                severity = kwargs.get("severity", args[0] if args else None)
                self.last_severity = int(severity) if severity is not None else None
                # Preserve the actual function's return object and call it once.
                return _original(image, *args, **kwargs)

            setattr(module, name, tapped)
        return self

    def uninstall(self):
        if self._module is None:
            return
        for name, original in self._originals.items():
            setattr(self._module, name, original)
        self._originals.clear()
        self._module = None

    def clean_array(self, np):
        if self.last_clean is None:
            raise RuntimeError("No pre-transform image was captured")
        return np.array(self.last_clean, copy=True)

    def clean_observation(self, np, noisy_observation):
        """Copy an obs dict and replace only agentview_image."""
        if "agentview_image" not in noisy_observation:
            raise KeyError("agentview_image is missing from the noisy observation")
        result = dict(noisy_observation)
        result["agentview_image"] = self.clean_array(np)
        return result


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_protocol_and_deployment(p, arm):
    import torch

    closed = p["source_closed_protocol"]
    if arm in ("softbin", "relational"):
        source = closed["arm_sources"][arm]
        source_protocol = _load_json(source["source_protocol_file"])
        deployment_path = source["deployment"]
        expected_unit = source["deployment_unit"]
        expected_protocol = source["training_protocol"]
    elif arm == "per_channel_clip":
        results = Path(closed["references"]["per_channel_results"])
        evaluation_protocol = _load_json(results.parent / "protocol.json")
        sources = evaluation_protocol["sources"]
        source_protocol = _load_json(Path(sources["training_root"]) / "protocol.json")
        deployment_path = sources["deployment"]
        expected_unit = None  # The per-channel training deployment omits a unit field.
        expected_protocol = sources["training_protocol"]
    else:
        raise ValueError(f"No deployment for arm {arm}")

    deployment = torch.load(deployment_path, map_location="cpu", weights_only=True)
    if (deployment.get("protocol") != source_protocol
            or (expected_unit is not None and deployment.get("unit") != expected_unit)
            or deployment.get("updates") != 80
            or deployment.get("baseline_checkpoint") != closed["sources"]["baseline_deployment"]):
        raise RuntimeError(f"Frozen deployment identity differs for {arm}")
    if source_protocol.get("protocol") != expected_protocol:
        raise RuntimeError(f"Frozen source protocol differs for {arm}")
    return deployment, source_protocol, str(deployment_path)


@contextmanager
def _capture_action_initial_noise(model, torch):
    """Tap the first action latent within the existing sampler call."""
    name = "_denoise_action_with_video_cache"
    namespace = getattr(model, "__dict__", {})
    existed, previous = name in namespace, namespace.get(name)
    original = getattr(model, name)
    captured = {"value": None}

    def tapped(*args, **kwargs):
        if captured["value"] is None:
            value = kwargs.get("latents_action")
            if value is None and args:
                value = args[0]
            if isinstance(value, torch.Tensor):
                captured["value"] = value.detach().to(device="cpu", dtype=torch.float32).clone()
        return original(*args, **kwargs)

    setattr(model, name, tapped)
    try:
        yield captured
    finally:
        if existed:
            setattr(model, name, previous)
        else:
            delattr(model, name)


class SwitchingRuntime:
    """One frozen Fast-WAM model with BF16 and three cached VQ/A4 arms."""

    def __init__(self, p, out):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        import runner
        import sampler
        import channel_banks
        from checkpointing import capture_rng, restore_rng
        from frozen_reader import capture_original_weights

        self.torch = torch
        self.p = _load_json(p) if isinstance(p, (str, Path)) else p
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self._capture_rng, self._restore_rng = capture_rng, restore_rng
        self.sampler = sampler
        self.compensated_weight = channel_banks.compensated_weight
        self.runtime = runner.Runtime(self.out)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.runtime.model.eval()
        for parameter in self.runtime.model.parameters():
            parameter.requires_grad_(False)
        self.runtime.arm("bf16")
        # Keep source BF16 tensors before ChannelBanks disables/wraps the model.
        self.original_weights = capture_original_weights(self.runtime)

        closed = self.p["source_closed_protocol"]
        baseline_path = closed["sources"]["baseline_deployment"]
        baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
        if (baseline.get("arm") != "rho1.5" or baseline.get("updates") != 8
                or baseline.get("target_linears") != 614
                or set(baseline.get("module_masters", {})) != set(self.runtime.q.modules)):
            raise RuntimeError("Frozen C8 rho1.5 baseline masters differ")
        self.banks = channel_banks.ChannelBanks(
            self.runtime, closed["calibration"]["bank_directory"],
            baseline["module_masters"], closed["clipping"], closed["scaling"])
        if len(self.runtime.q.modules) != 614:
            raise RuntimeError("Expected 614 target Linears")
        self.decoded_base_weights = {
            name: module.weight.detach().cpu().clone()
            for name, module in self.runtime.q.modules.items()
        }
        self.student_forwards = {
            name: module.forward for name, module in self.runtime.q.modules.items()
        }
        for values in (self.banks.books.values(), self.banks.logs.values(),
                       self.banks.ratios.values(), self.banks.shadows.values()):
            for parameter in values:
                parameter.requires_grad_(False)
        self.deployments = {}
        self.deployment_bindings = {}
        for arm in QUANTIZED_ARMS:
            deployment, source_protocol, path = _source_protocol_and_deployment(self.p, arm)
            if (set(deployment.get("ratios", {})) != set(self.runtime.q.modules)
                    or set(deployment.get("log2_scales", {})) != set(self.runtime.q.modules)):
                raise RuntimeError(f"Deployment does not cover all 614 Linears: {arm}")
            self.deployments[arm] = {
                "ratios": deployment["ratios"],
                "log2_scales": deployment["log2_scales"],
            }
            self.deployment_bindings[arm] = {
                "protocol": source_protocol.get("protocol"),
                "unit": deployment.get("unit"),
                "updates": deployment.get("updates"),
                "deployment": path,
            }
            del deployment
        del baseline
        torch.cuda.empty_cache()
        self.active_arm = None
        self.query_calls = 0
        self.activate("bf16")

    def _set_bf16_forwards(self):
        F = self.torch.nn.functional
        for name, module in self.runtime.q.modules.items():
            weight = self.original_weights[name]
            bias = module.bias

            def forward(value, _name=name, _weight=weight, _bias=bias):
                self._query_seen.add(_name)
                return F.linear(value, _weight, _bias)

            module.weight.data = weight
            module.forward = forward

    def activate(self, arm):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm: {arm}")
        modules = self.runtime.q.modules
        if arm == "bf16":
            self.banks.cached = False
            self._set_bf16_forwards()
        else:
            self.banks.cached = False
            for name, module in modules.items():
                module.forward = self.student_forwards[name]
            self.banks.restore_clips(self.deployments[arm]["ratios"])
            self.banks.restore_scales(self.deployments[arm]["log2_scales"])
            self.install_compensated_weights()
            self.banks.cached = True
        self.active_arm = arm
        self.runtime.current_arm = arm

    def install_compensated_weights(self):
        install_compensated_from_base(
            self.runtime.q.modules, self.decoded_base_weights,
            self.banks.shadows, self.compensated_weight)

    def datum(self, obs):
        return self.runtime.datum(obs)

    def query(self, datum, seed, description=None):
        if self.active_arm is None:
            raise RuntimeError("Activate an arm before querying")
        if description is not None:
            self.runtime.description = description
        if not isinstance(getattr(self.runtime, "description", None), str):
            raise ValueError("A fixed task description is required")

        torch = self.torch
        before_rng = self._capture_rng(torch)
        self._query_seen = set()
        if self.active_arm != "bf16":
            self.banks.seen.clear()
        query_started = False
        try:
            with _capture_action_initial_noise(self.runtime.model, torch) as noise, torch.no_grad():
                query_started = True
                action, counts, future = self.sampler.differentiable_action(
                    self.runtime, datum, int(seed), checkpointed=False, return_future=True)
            seen = self._query_seen if self.active_arm == "bf16" else self.banks.seen
            if (len(seen) != 614 or counts != {"video_steps": 10, "action_steps": 10}
                    or tuple(action.shape) != (32, 7)
                    or tuple(future.shape) != (1, 48, 3, 14, 28)
                    or not bool(torch.isfinite(action).all())
                    or not bool(torch.isfinite(future).all())
                    or noise["value"] is None
                    or tuple(noise["value"].shape) != (1, 32, 7)
                    or not bool(torch.isfinite(noise["value"]).all())):
                raise RuntimeError("Query coverage, 10+10 steps, outputs, or noise capture failed")
            return (action.detach().float().cpu(), future.detach().float().cpu(),
                    dict(counts), noise["value"])
        finally:
            if query_started:
                self.query_calls += 1
            # Fixed-input diagnostic queries must not advance environment RNGs.
            self._restore_rng(torch, before_rng)

    def close(self):
        env = getattr(self.runtime, "env", None)
        if env is not None:
            env.close()
            self.runtime.env = None

