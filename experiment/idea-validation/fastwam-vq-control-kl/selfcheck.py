"""Small CPU-only checks for frozen BF16 forwards and future injection."""
import inspect

import torch
from torch.utils.checkpoint import checkpoint

from frozen_reader import (
    INFERENCE_STEPS,
    _FutureInjector,
    _bf16_forward_context,
    _checkpoint_context_fn,
    _inference_kwargs,
    _raw_idm_infer,
)


class _ToyRawModel:
    def infer_action(self, prompt, input_image, proprio, action_horizon,
                     num_video_frames, num_inference_steps, sigma_shift, seed,
                     rand_device, tiled, compile_action_infer):
        raise AssertionError("Signature-only selfcheck must not run model inference")


def main():
    toy_runtime = type("Runtime", (), {
        "description": "toy task",
        "ev": type("Evaluation", (), {"DEFAULT_PROMPT": "Task: {task}"})(),
    })()
    kwargs = _inference_kwargs(toy_runtime, {"image": object(), "proprio": object()}, 7)
    raw_infer = _raw_idm_infer(_ToyRawModel())
    inspect.signature(raw_infer).bind(_ToyRawModel(), **kwargs)
    assert "action_infer_mode" not in kwargs

    module = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.zero_()
    module.weight.requires_grad_(False)
    student_forward = lambda x: torch.nn.functional.linear(x, module.weight, module.bias)
    module.forward = student_forward
    held_weight = torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16).detach()
    modules = {"toy": module}
    weights = {"toy": held_weight}
    recompute_state = {"recompute_depth": 0}
    x = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16, requires_grad=True)

    with _bf16_forward_context(modules, weights):
        y = checkpoint(
            lambda value: module(value),
            x,
            use_reentrant=False,
            context_fn=_checkpoint_context_fn(modules, weights, recompute_state),
        )
    assert module.forward is student_forward
    assert y.item() == 5.0
    y.float().sum().backward()
    assert torch.equal(x.grad, torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16))
    assert held_weight.grad is None
    assert recompute_state["recompute_depth"] == 0

    try:
        with _bf16_forward_context(modules, weights):
            raise LookupError("exercise cleanup")
    except LookupError:
        pass
    assert module.forward is student_forward

    future = torch.tensor(
        [[[[[90.0]], [[2.0]], [[4.0]]]]], dtype=torch.bfloat16, requires_grad=True
    )
    injector = _FutureInjector(future)
    outputs = []
    for _ in range(INFERENCE_STEPS):
        sample = torch.zeros_like(future)
        sample[:, :, 0:1] = 7.0
        outputs.append(injector(None, None, sample))
    assert injector.calls == INFERENCE_STEPS
    assert torch.equal(injector.last_output[:, :, 0:1], torch.full_like(future[:, :, 0:1], 7.0))
    assert torch.equal(injector.last_output[:, :, 1:], future[:, :, 1:])
    with _bf16_forward_context(modules, weights):
        toy_actions = [
            checkpoint(
                lambda latent: module(latent),
                output[:, :, 1:].reshape(1, 2),
                use_reentrant=False,
                context_fn=_checkpoint_context_fn(modules, weights, recompute_state),
            )
            for output in outputs
        ]
    assert all(action.item() == 16.0 for action in toy_actions)
    sum(action.float().sum() for action in toy_actions).backward()
    assert torch.equal(future.grad[:, :, 0:1], torch.zeros_like(future[:, :, 0:1]))
    expected_tail_grad = torch.tensor([[[[[20.0]], [[30.0]]]]])
    assert torch.equal(future.grad[:, :, 1:], expected_tail_grad)
    assert held_weight.grad is None
    assert recompute_state["recompute_depth"] == 0

    print("SELF_CHECK_PASSED: BF16 gradient, checkpoint context, cleanup, clamp, and 10-step counts")


if __name__ == "__main__":
    main()
