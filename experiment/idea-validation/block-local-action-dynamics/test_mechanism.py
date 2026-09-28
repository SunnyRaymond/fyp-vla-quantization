"""PBS-only, small mechanism checks for the frozen controlled systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


def _run_checks(config: dict[str, Any], device: str) -> list[dict[str, Any]]:
    import torch

    from controlled_system import make_datasets
    from models import BLOCK_ARMS, build_model, matched_dense_hidden

    small_config = dict(config)
    small_config["data"] = {
        **config.get("data", {}),
        "train_episodes": 4,
        "dev_episodes": 2,
        "test_episodes": 2,
    }
    checks: list[tuple[str, Callable[[], dict[str, Any] | None]]] = []

    def check_coordinates_and_orthogonality() -> dict[str, Any]:
        # z=x@M.T, so multiplying a centered observation by M recovers centered x.
        x = torch.randn(5, 64, device=device)
        system, _, meta = make_datasets(small_config, "independent", device)
        z = system.observe(x)
        recovered = system.recover(z)
        torch.testing.assert_close(recovered, x, atol=2e-5, rtol=2e-5)
        action = torch.randn(5, 8, device=device)
        torch.testing.assert_close(
            system.step_z(z, action), system.observe(system.step_x(recovered, action)), atol=2e-5, rtol=2e-5
        )
        mean_z = meta["mean"]
        centered_s = (z - mean_z) @ system.M
        centered_x = x - mean_z @ system.M
        torch.testing.assert_close(centered_s, centered_x, atol=2e-5, rtol=2e-5)

        model = build_model("random_block", config, 1101, mean_z).to(device)
        q = model.orthogonal_matrix()
        identity = torch.eye(64, device=device)
        orthogonality_error = float((q.T @ q - identity).abs().max().item())
        torch.testing.assert_close(q.T @ q, identity, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close((z - mean_z) @ q @ q.T, z - mean_z, atol=3e-5, rtol=3e-5)
        return {"max_orthogonality_error": orthogonality_error}

    def check_independent_cross_block_jacobian() -> dict[str, Any]:
        system, _, _ = make_datasets(small_config, "independent", device)
        x = torch.randn(64, device=device, requires_grad=True)
        action = torch.randn(1, 8, device=device)
        jacobian = torch.autograd.functional.jacobian(
            lambda state: system.step_x(state.unsqueeze(0), action)[0], x
        )
        mask = torch.ones_like(jacobian)
        for block in range(4):
            sl = slice(block * 16, (block + 1) * 16)
            mask[sl, sl] = 0
        cross_max = float((jacobian * mask).abs().max().item())
        assert cross_max == 0.0, f"independent cross-block Jacobian max is {cross_max}"
        return {"cross_block_jacobian_max": cross_max}

    def check_lowrank_correction_rank() -> dict[str, Any]:
        system, _, _ = make_datasets(small_config, "lowrank_coupled", device)
        x = torch.randn(64, device=device, requires_grad=True)
        action = torch.randn(1, 8, device=device)
        jacobian = torch.autograd.functional.jacobian(
            lambda state: system.coupling_delta(state.unsqueeze(0), action)[0], x
        )
        rank = int(torch.linalg.matrix_rank(jacobian).item())
        assert rank == 4, f"true lowrank correction should have rank 4, got {rank}"
        return {"correction_jacobian_rank": rank}

    def check_projection_pairing_and_dense_budget() -> dict[str, Any]:
        mean = torch.zeros(64, device=device)
        global_model = build_model("learned_global4", config, 1101, mean).to(device)
        local_model = build_model("learned_local4", config, 1101, mean).to(device)
        random_model = build_model("random_block", config, 1101, mean).to(device)
        learned_model = build_model("learned_block", config, 1101, mean).to(device)
        global_weights = global_model.global_message
        local_weights = local_model.local_message
        dimensions = config.get("dimensions", config)
        message_dim = int(dimensions.get("message", 4))
        assert global_weights.numel() == local_weights.numel() == message_dim * 64
        torch.testing.assert_close(
            local_weights.permute(1, 0, 2).reshape_as(global_weights),
            global_weights,
            atol=0,
            rtol=0,
        )
        for name in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "out.weight", "out.bias"):
            left = dict(global_model.named_parameters())[name]
            right = dict(local_model.named_parameters())[name]
            torch.testing.assert_close(left, right, atol=0, rtol=0)
            torch.testing.assert_close(
                dict(random_model.named_parameters())[name],
                dict(learned_model.named_parameters())[name],
                atol=0,
                rtol=0,
            )
        torch.testing.assert_close(random_model.orthogonal_matrix(), learned_model.orthogonal_matrix(), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(global_model.orthogonal_matrix(), local_model.orthogonal_matrix(), atol=2e-5, rtol=2e-5)
        global_count = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
        local_count = sum(p.numel() for p in local_model.parameters() if p.requires_grad)
        assert global_count == local_count, f"global/local parameter counts differ: {global_count} vs {local_count}"
        dense = build_model("dense", config, 1101, mean)
        dense_params = sum(p.numel() for p in dense.parameters() if p.requires_grad)
        target = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
        relative_gap = abs(dense_params - target) / target
        assert relative_gap <= 0.05, f"dense budget gap {relative_gap:.4%} exceeds 5%"
        assert matched_dense_hidden(config, mean) == dense.net[0].out_features
        return {"global_local_projection_parameters": global_weights.numel(), "dense_parameters": dense_params,
                "global_model_parameters": target, "dense_budget_relative_gap": relative_gap}

    def check_paired_private_batch_generators() -> dict[str, Any]:
        def draw(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            episodes = torch.randint(512, (128,), generator=generator)
            offsets = torch.randint(36, (128,), generator=generator)
            return episodes, offsets

        first = draw(1101)
        paired = draw(1101)
        unpaired = draw(1102)
        assert torch.equal(first[0], paired[0]) and torch.equal(first[1], paired[1])
        assert not (torch.equal(first[0], unpaired[0]) and torch.equal(first[1], unpaired[1]))
        return {"paired_draws_identical": True, "different_seed_changes_draws": True}

    def check_step_rollout_and_causality() -> dict[str, Any]:
        mean = torch.randn(64, device=device)
        z0 = torch.randn(3, 64, device=device)
        actions = torch.randn(3, 6, 8, device=device)
        arms = list(config.get("arms", ["dense", *sorted(BLOCK_ARMS)]))
        for arm in arms:
            model = build_model(arm, config, 1101, mean).to(device).eval()
            predicted = model.rollout(z0, actions)
            assert predicted.shape == (3, 6, 64)
            torch.testing.assert_close(model.step(z0, actions[:, 0]), predicted[:, 0], atol=2e-5, rtol=2e-5)
            repeated = z0
            for t in range(actions.shape[1]):
                repeated = model.step(repeated, actions[:, t])
                torch.testing.assert_close(repeated, predicted[:, t], atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(model.rollout_final(z0, actions), predicted[:, -1], atol=3e-5, rtol=3e-5)
            altered = actions.clone()
            altered[:, 3:] += 10.0
            changed = model.rollout(z0, altered)
            torch.testing.assert_close(predicted[:, :3], changed[:, :3], atol=0, rtol=0)
            assert model.rollout(z0, actions[:, :0]).shape == (3, 0, 64)
            torch.testing.assert_close(model.rollout_final(z0, actions[:, :0]), z0, atol=0, rtol=0)
        return {"arms_checked": arms, "rollout_horizon": 6}

    def check_inference_cache_keeps_training_gradient() -> dict[str, Any]:
        mean = torch.randn(64, device=device)
        model = build_model("learned_block", config, 1101, mean).to(device)
        z, action = torch.randn(2, 64, device=device), torch.randn(2, 8, device=device)
        model.eval()
        before = model.step(z, action)
        model.prepare_inference()
        after = model.step(z, action)
        torch.testing.assert_close(before, after, atol=2e-5, rtol=2e-5)
        model.train()
        assert model._inference_q is None, "train() must clear the detached inference Q cache"
        model.step(z, action).square().mean().backward()
        grad = model.transform.parametrizations.weight.original.grad
        assert grad is not None and torch.isfinite(grad).all(), "live learned Q must retain training gradients"
        return {"cache_consistent": True, "training_q_gradient": True}

    def check_deterministic_paired_episode_splits() -> dict[str, Any]:
        sys_a, first, meta_a = make_datasets(small_config, "independent", device)
        sys_b, repeated, _ = make_datasets(small_config, "independent", device)
        sys_c, coupled, _ = make_datasets(small_config, "lowrank_coupled", device)
        torch.testing.assert_close(sys_a.M, sys_b.M, atol=0, rtol=0)
        torch.testing.assert_close(sys_a.M, sys_c.M, atol=0, rtol=0)
        for split in ("train", "dev", "test"):
            for key in ("z", "action"):
                torch.testing.assert_close(first[split][key], repeated[split][key], atol=0, rtol=0)
            torch.testing.assert_close(first[split]["action"], coupled[split]["action"], atol=0, rtol=0)
            torch.testing.assert_close(first[split]["z"][:, 0], coupled[split]["z"][:, 0], atol=0, rtol=0)

        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            for x in first[left]["action"]:
                assert not any(torch.equal(x, y) for y in first[right]["action"]), f"episode overlap: {left}/{right}"
            for x in first[left]["z"][:, 0]:
                assert not any(torch.equal(x, y) for y in first[right]["z"][:, 0]), f"initial-state overlap: {left}/{right}"
        expected_mean = first["train"]["z"].mean(dim=(0, 1))
        torch.testing.assert_close(meta_a["mean"], expected_mean, atol=1e-7, rtol=1e-7)
        delta = first["train"]["z"][:, 1:] - first["train"]["z"][:, :-1]
        expected_energy = float((delta - delta.mean()).square().mean().clamp_min(1e-6).item())
        assert abs(meta_a["delta_energy"] - expected_energy) < 1e-8
        return {"split_episode_counts": meta_a["episode_counts"], "paired_conditions": True,
                "split_seeds": meta_a["split_seeds"]}

    checks.extend(
        (
            ("known_coordinate_and_orthogonal_roundtrip", check_coordinates_and_orthogonality),
            ("independent_cross_block_jacobian_zero", check_independent_cross_block_jacobian),
            ("lowrank_correction_jacobian_rank_four", check_lowrank_correction_rank),
            ("global_local_projection_pairing_and_dense_budget", check_projection_pairing_and_dense_budget),
            ("paired_private_batch_generators", check_paired_private_batch_generators),
            ("step_rollout_and_future_action_causality", check_step_rollout_and_causality),
            ("inference_cache_and_training_gradient", check_inference_cache_keeps_training_gradient),
            ("deterministic_paired_episode_splits", check_deterministic_paired_episode_splits),
        )
    )

    results = []
    for name, check in checks:
        try:
            details = check() or {}
            results.append({"name": name, "status": "pass", "details": details})
        except Exception as exc:
            results.append({"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from allocation_guard import ensure_allocation

    allocation = ensure_allocation(require_gpu=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    results = _run_checks(config, "cuda")
    payload = {
        "status": "pass" if all(result["status"] == "pass" for result in results) else "fail",
        "device": "cuda",
        "allocation": allocation,
        "checks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
