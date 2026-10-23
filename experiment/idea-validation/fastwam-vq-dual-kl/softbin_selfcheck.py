"""PBS-only numerical checks for the frozen soft-bin dual-KL loss module."""
import json
import os
import socket
from pathlib import Path


def _require_compute_allocation():
    job_id = os.environ.get("PBS_JOBID")
    hostname = socket.gethostname()
    if not job_id:
        raise RuntimeError("soft-bin selfcheck requires a PBS compute allocation")
    if "login" in hostname.lower():
        raise RuntimeError(f"soft-bin selfcheck is forbidden on login hosts: {hostname}")
    nodefile = os.environ.get("PBS_NODEFILE")
    if not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("PBS_NODEFILE must identify the active compute allocation")
    short_host = hostname.split(".", 1)[0].lower()
    allocated_hosts = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if short_host not in allocated_hosts:
        raise RuntimeError(f"hostname {hostname} is absent from PBS_NODEFILE")
    artifact_dir = os.environ.get("ARTIFACTS")
    if not artifact_dir:
        raise RuntimeError("ARTIFACTS must be set for the selfcheck report")
    return job_id, hostname, Path(artifact_dir)


def main():
    job_id, hostname, artifact_dir = _require_compute_allocation()

    import torch
    from softbin_loss import losses, pointwise_kl, softbin_log_probs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "bins": 16,
        "interior_min": -3.5,
        "interior_max": 3.5,
        "sigma": 0.375,
        "std_floor": 0.02,
    }

    edges = torch.linspace(-3.5, 3.5, 15, device=device, dtype=torch.float32)
    tail_values = torch.tensor(
        [-150.0, -35.0, -2.0, 0.0, 2.0, 35.0, 150.0], device=device
    )
    log_probs = softbin_log_probs(torch, tail_values, edges, 0.375)
    probabilities = log_probs.exp()
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones_like(tail_values), rtol=2e-6, atol=2e-6
    )
    assert bool(torch.isfinite(log_probs).all())

    zero_mean = [0.0, 0.0]
    unit_scale = [1.0, 1.0]
    same = torch.tensor([[-1.0, 1.0]], device=device)
    same_kl = pointwise_kl(torch, same, same, zero_mean, unit_scale, -1, config)
    assert float(same_kl) <= 1e-7

    shifted = torch.tensor([[1.25]], device=device, requires_grad=True)
    zero = torch.zeros_like(shifted)
    shifted_kl = pointwise_kl(torch, shifted, zero, [0.0], [1.0], -1, config)
    shifted_grad, = torch.autograd.grad(shifted_kl, shifted)
    assert float(shifted_kl) > 0 and bool(torch.isfinite(shifted_grad).all())
    assert float(shifted_grad.abs().max()) > 0

    tail_student = torch.tensor([[25.0]], device=device, requires_grad=True)
    tail_teacher = torch.zeros_like(tail_student)
    tail_kl = pointwise_kl(torch, tail_student, tail_teacher, [0.0], [1.0], -1, config)
    tail_grad, = torch.autograd.grad(tail_kl, tail_student)
    assert bool(torch.isfinite(tail_kl)) and bool(torch.isfinite(tail_grad).all())
    assert float(tail_kl) > 0 and float(tail_grad.abs().max()) > 0

    extreme_student = torch.tensor([[-150.0], [150.0]], device=device, requires_grad=True)
    extreme_teacher = torch.zeros_like(extreme_student)
    extreme_kl = pointwise_kl(
        torch, extreme_student, extreme_teacher, [0.0], [1.0], -1, config
    )
    extreme_grad, = torch.autograd.grad(extreme_kl, extreme_student)
    assert bool(torch.isfinite(extreme_kl)) and bool(torch.isfinite(extreme_grad).all())
    assert bool((extreme_grad.abs() > 0).all())

    scaled_student = torch.tensor([[1.0, 15.0]], device=device)
    scaled_teacher = torch.tensor([[0.0, 5.0]], device=device)
    scaled_kl = pointwise_kl(
        torch, scaled_student, scaled_teacher, [0.0, 5.0], [1.0, 10.0], -1, config
    )
    one_channel_kl = pointwise_kl(
        torch, scaled_student[:, :1], scaled_teacher[:, :1], [0.0], [1.0], -1, config
    )
    torch.testing.assert_close(scaled_kl, one_channel_kl, rtol=1e-6, atol=1e-7)

    permuted_teacher = torch.tensor([[-1.0, 1.0]], device=device)
    permuted_student = permuted_teacher.flip(dims=(-1,))
    permutation_kl = pointwise_kl(
        torch, permuted_student, permuted_teacher, zero_mean, unit_scale, -1, config
    )
    assert float(permutation_kl) > 0

    grad_teacher = torch.zeros((1, 1), device=device, requires_grad=True)
    grad_student = torch.ones((1, 1), device=device, requires_grad=True)
    teacher_detach_kl = pointwise_kl(
        torch, grad_student, grad_teacher, [0.0], [1.0], -1, config
    )
    teacher_detach_kl.backward()
    assert grad_student.grad is not None and bool(torch.isfinite(grad_student.grad).all())
    assert grad_teacher.grad is None

    fd_x = 1.1
    fd_eps = 0.002
    fd_student = torch.tensor([[fd_x]], device=device, requires_grad=True)
    fd_loss = pointwise_kl(
        torch, fd_student, torch.zeros_like(fd_student), [0.0], [1.0], -1, config
    )
    fd_grad, = torch.autograd.grad(fd_loss, fd_student)
    with torch.no_grad():
        fd_plus = pointwise_kl(
            torch, torch.tensor([[fd_x + fd_eps]], device=device),
            torch.zeros((1, 1), device=device), [0.0], [1.0], -1, config,
        )
        fd_minus = pointwise_kl(
            torch, torch.tensor([[fd_x - fd_eps]], device=device),
            torch.zeros((1, 1), device=device), [0.0], [1.0], -1, config,
        )
        finite_difference = (fd_plus - fd_minus) / (2 * fd_eps)
    torch.testing.assert_close(fd_grad.reshape(()), finite_difference, rtol=3e-2, atol=3e-3)

    action = (torch.arange(32 * 7, device=device, dtype=torch.float32).reshape(32, 7) / 100)
    action.requires_grad_()
    target_action = action.detach() * 0.9
    future = (
        torch.arange(1 * 48 * 3 * 14 * 28, device=device, dtype=torch.float32)
        .reshape(1, 48, 3, 14, 28) / 10000
    )
    future.requires_grad_()
    target_future = future.detach() * 0.7 + 0.1
    stats = {
        "action": {"mean": [0.0] * 6, "scale": [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]},
        "latent": {
            "mean": [index / 100 for index in range(48)],
            "scale": [0.4 + 0.01 * (index % 10) for index in range(48)],
        },
        "config": config,
    }
    terms = losses(torch, action, future, target_action, target_future, stats)
    selected_error = future.detach()[:, :, 1:] - target_future[:, :, 1:]
    expected_nmse = selected_error.square().sum() / target_future[:, :, 1:].square().sum().clamp_min(1e-12)
    torch.testing.assert_close(terms["latent_nmse"], expected_nmse, rtol=0, atol=0)
    sum(terms.values()).backward()
    assert bool((action.grad[:, 6] == 0).all())
    assert bool((action.grad[10:] == 0).all())
    assert bool((future.grad[:, :, 0] == 0).all())

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "passed": True,
        "pbs_jobid": job_id,
        "hostname": hostname,
        "device": str(device),
        "checks": {
            "probability_normalization_and_tails": True,
            "same_input_kl_near_zero": True,
            "shifted_and_tail_gradients_finite_nonzero": True,
            "extreme_tail_over_100_gradients_finite_nonzero": True,
            "per_channel_scale_applied": True,
            "position_permutation_penalized": True,
            "teacher_detached": True,
            "finite_difference_gradient": True,
            "original_global_nmse_exact": True,
            "action_gripper_and_late22_gradients_zero": True,
            "future_frame0_gradient_zero": True,
        },
    }
    (artifact_dir / "softbin_selfcheck.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("SOFTBIN_SELFCHECK_PASSED", flush=True)


if __name__ == "__main__":
    main()
