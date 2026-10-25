"""Numerical checks for FrozenSuffixReader using the pinned Cosmos sampler.

This is a CPU toy-denoiser check, not a Cosmos model forward or an ablation.
Run it only inside an approved PBS compute allocation; common.guard() is
required before torch is imported.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import tempfile
import types

import common


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent,
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _run_checks(protocol: dict, root: Path, checks: dict) -> dict:
    # Ensure the source is the pinned checkout, then import the actual solver.
    common.ensure_imports(protocol)
    torch = common.torch_module()
    from cosmos_policy._src.imaginaire.modules import res_sampler
    from reader import FrozenSuffixReader, LoopCapture, action_from_latent, reader_kl

    inference = protocol["inference"]
    nfe = int(inference["denoising_steps"])
    cut = int(protocol["training"]["reader"]["cut_evaluation"])
    if nfe != 5 or cut != 3:
        raise ValueError(f"Frozen test requires five NFE and cut=3; got nfe={nfe}, cut={cut}")
    if not callable(getattr(res_sampler, "Sampler", None)):
        raise RuntimeError("Pinned res_sampler module has no Sampler class")

    # Only the sampler is instantiated. No CosmosRuntime or model is loaded.
    sampler = res_sampler.Sampler()
    raw_forward = inspect.unwrap(type(sampler).forward)
    raw_impl = inspect.unwrap(type(sampler)._forward_impl)
    sampler.forward = types.MethodType(raw_forward, sampler)
    sampler._forward_impl = types.MethodType(raw_impl, sampler)

    shape = (1, 16, 9, 2, 4)
    initial = torch.linspace(-0.6, 0.6, 1 * 16 * 9 * 2 * 4, dtype=torch.float32).reshape(shape)
    fixed_theta = torch.tensor(0.7, dtype=torch.float32)
    counter = {"sampler_evaluations": 0}

    def toy_denoiser(x, sigma):
        del sigma
        # A smooth, differentiable future-slot (6/7) -> action-slot (4) path.
        future_mean = x[:, :, 6:8].mean(dim=2, keepdim=True)
        action_delta = torch.zeros_like(x)
        action_delta[:, :, 4:5] = fixed_theta * future_mean
        return 0.45 * x + action_delta

    def counted_x0_fn(x, sigma):
        counter["sampler_evaluations"] += 1
        return toy_denoiser(x, sigma)

    capture = None
    # Unwrap the pinned @no_grad wrappers for this mathematical gradient check;
    # the algorithm, coefficients, timestamps, and fori_loop remain original.
    with torch.inference_mode(False), torch.enable_grad():
        with LoopCapture(cut=cut, retain_graph=True) as capture_context:
            generated = sampler.forward(
                counted_x0_fn,
                initial,
                num_steps=nfe,
                sigma_min=float(inference["sigma_min"]),
                sigma_max=float(inference["sigma_max"]),
                solver_option="2mid",
            )
        capture = capture_context.require()

    actual_nfe = int(counter["sampler_evaluations"])
    checks["actual_solver_nfe"] = {
        "passed": actual_nfe == nfe,
        "sampler_evaluations": actual_nfe,
        "expected": nfe,
        "solver_option": "2mid",
    }
    if actual_nfe != nfe:
        raise RuntimeError(f"Pinned 2mid solver executed {actual_nfe} evaluations, expected {nfe}")

    toy_protocol = {
        "inference": {"chunk_size": int(inference["chunk_size"]), "unnormalize_actions": False},
        "training": {"reader": protocol["training"]["reader"]},
    }
    runtime = types.SimpleNamespace(
        protocol=toy_protocol,
        dataset_stats={},
        model=types.SimpleNamespace(parameters=lambda: iter(())),
    )
    teacher_action = action_from_latent(runtime, generated).detach()
    teacher = types.SimpleNamespace(generated_latent=generated.detach(), action=teacher_action)
    reader = FrozenSuffixReader.__new__(FrozenSuffixReader)
    reader.runtime, reader.teacher, reader.capture = runtime, teacher, capture

    identity = reader.identity()
    checks["cut3_original_suffix_identity"] = {
        **identity,
        "passed": bool(identity["passed"] and identity["max_abs"] <= 1e-6),
        "cut": cut,
        "capture_start": int(capture.start),
        "capture_upper": int(capture.upper),
        "latent_shape": list(shape),
    }
    if not checks["cut3_original_suffix_identity"]["passed"]:
        raise RuntimeError("Replaying the original solver suffix changed the action by more than 1e-6")

    statistics = {
        "actions_min": torch.full((7,), -1.0, dtype=torch.float32),
        "actions_max": torch.full((7,), 1.0, dtype=torch.float32),
        "mean": torch.zeros(6, dtype=torch.float32),
        "scale": torch.full((6,), 0.5, dtype=torch.float32),
    }
    base = capture.state[0].detach().clone()
    future_selector = torch.zeros_like(base)
    future_selector[:, :, 6:8] = 1.0
    phi_value = 0.2
    student_state = base + phi_value * future_selector
    cotangent_cpu, cotangent_info, _ = reader.cotangent(student_state, statistics)
    cotangent = cotangent_cpu.to(device=base.device, dtype=base.dtype)
    future_region = torch.zeros_like(cotangent, dtype=torch.bool)
    future_region[:, :, 6:8] = True
    outside = cotangent.masked_select(~future_region)
    future_grad_norm = float(cotangent.masked_select(future_region).norm())
    cotangent_passed = (
        bool(torch.isfinite(cotangent).all())
        and future_grad_norm > 0.0
        and bool((outside == 0).all())
    )
    checks["future_only_cotangent"] = {
        "passed": cotangent_passed,
        "finite": bool(torch.isfinite(cotangent).all()),
        "future_gradient_norm": future_grad_norm,
        "outside_future_max_abs": float(outside.abs().max()) if outside.numel() else 0.0,
        "outside_future_nonzero_count": int((outside != 0).sum()),
        "reader_kl": cotangent_info["reader_kl"],
        "student_state_shape": list(student_state.shape),
    }
    if not cotangent_passed:
        raise RuntimeError("Future cotangent was zero/non-finite or escaped slots 6/7")

    # Chain the staged reader VJP through a scalar future-input parameter.
    phi_staged = torch.tensor(phi_value, dtype=torch.float32, requires_grad=True)
    staged_input = base + phi_staged * future_selector
    staged_gradient, = torch.autograd.grad(staged_input, phi_staged, grad_outputs=cotangent)

    # Compare with differentiating the same hybrid KL directly through the
    # captured original suffix and the same scalar future intervention.
    phi_direct = torch.tensor(phi_value, dtype=torch.float32, requires_grad=True)
    direct_action = reader.read(base + phi_direct * future_selector)
    direct_kl = reader_kl(runtime, direct_action, teacher.action, statistics)
    direct_gradient, = torch.autograd.grad(direct_kl, phi_direct)
    staged_value, direct_value = float(staged_gradient), float(direct_gradient)
    grad_delta = abs(staged_value - direct_value)
    grad_passed = (
        bool(torch.isfinite(staged_gradient)) and bool(torch.isfinite(direct_gradient))
        and grad_delta <= 1e-5 + 1e-4 * abs(direct_value)
    )
    checks["staged_vjp_vs_direct_hybrid_kl"] = {
        "passed": grad_passed,
        "staged_gradient": staged_value,
        "direct_gradient": direct_value,
        "absolute_error": grad_delta,
        "atol": 1e-5,
        "rtol": 1e-4,
        "phi": phi_value,
        "direct_kl": float(direct_kl.detach()),
    }
    if not grad_passed:
        raise RuntimeError("Staged VJP gradient differs from direct hybrid-KL gradient")

    return {
        "solver": "cosmos_policy.res_sampler.Sampler",
        "solver_source": str(Path(res_sampler.__file__).resolve()),
        "solver_option": "2mid",
        "nfe": actual_nfe,
        "cut": cut,
        "latent_shape": list(shape),
        "execution_scope": "CPU toy denoiser and original pinned sampler only; no Cosmos model/runtime forward",
        "root": str(root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True, help="Output directory or JSON file")
    args = parser.parse_args()

    out = Path(args.out)
    summary_path = out if out.suffix.lower() == ".json" else out / "reader_math.json"
    summary = {
        "protocol": "cosmos-policy-vqwam-reader-math-v1",
        "status": "failed",
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "checks": {},
    }
    try:
        common.guard()
        protocol = common.read_json(args.protocol)
        info = _run_checks(protocol, Path(args.root).resolve(), summary["checks"])
        summary.update(info)
        summary["status"] = "passed" if all(
            row.get("passed") is True for row in summary["checks"].values()
        ) else "failed"
        if summary["status"] != "passed":
            raise RuntimeError("One or more numerical reader checks failed")
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _write_json(summary_path, summary)
    print(json.dumps({"status": summary["status"], "summary": str(summary_path),
                      "checks": summary["checks"], "error": summary.get("error")},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
