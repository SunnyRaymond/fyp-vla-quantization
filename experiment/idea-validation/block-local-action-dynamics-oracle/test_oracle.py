"""Allocation-only mechanism checks for the known-coordinate oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def run_checks(config: dict[str, Any], global16_config: dict[str, Any], torch: Any) -> list[dict[str, Any]]:
    from controlled_system import ControlledSystem
    from models import build_model
    from oracle_models import ORACLE_TO_INITIALIZER, build_oracle_model, predictor_parameter_views

    device = torch.device("cuda")
    checks: list[dict[str, Any]] = []
    seed_list = [int(seed) for seed in config["training"]["seeds"]]
    mean = torch.linspace(-0.4, 0.4, int(config["dimensions"]["state"]), device=device)
    systems = [
        ControlledSystem(condition, int(config["data"]["system_seed"]), device)
        for condition in config["conditions"]
    ]
    for system in systems[1:]:
        torch.testing.assert_close(system.M, systems[0].M, atol=0, rtol=0)
    system = systems[0]
    x = torch.randn(7, 64, device=device)
    z = system.observe(x)
    recovered = z @ system.M
    torch.testing.assert_close(recovered, x, atol=2e-5, rtol=2e-5)
    s = (z - mean) @ system.M
    expected_s = x - mean @ system.M
    torch.testing.assert_close(s, expected_s, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(mean + s @ system.M.T, z, atol=3e-5, rtol=3e-5)
    checks.append({"name": "known_M_coordinate_map_and_inverse", "status": "pass"})

    model_configs = [
        (name, config, arm, seed)
        for name, arm in ORACLE_TO_INITIALIZER.items()
        if arm != "learned_global16"
        for seed in seed_list
    ] + [
        ("oracle_global16", global16_config, "learned_global16", seed)
        for seed in seed_list
    ]
    compared = 0
    for oracle_arm, model_config, base_arm, seed in model_configs:
        reference = build_model(base_arm, model_config, seed, mean).to(device)
        oracle = build_oracle_model(oracle_arm, model_config, seed, mean, system.M).to(device)
        left, right = predictor_parameter_views(reference), predictor_parameter_views(oracle)
        assert left.keys() == right.keys(), f"predictor state keys differ: {oracle_arm}/{seed}"
        for key in left:
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
        torch.testing.assert_close(oracle.orthogonal_matrix(), system.M, atol=0, rtol=0)
        assert not oracle.transform.weight.requires_grad
        assert "transform.weight" not in dict(oracle.named_parameters()) or not dict(oracle.named_parameters())["transform.weight"].requires_grad
        oracle.train()
        probe = torch.randn(2, 64, device=device)
        action = torch.randn(2, 8, device=device)
        oracle.step(probe, action).square().mean().backward()
        assert oracle.transform.weight.grad is None
        compared += 1
        del reference, oracle
    assert compared == 12
    checks.append({"name": "fixed_Q_and_exact_same_seed_predictor_initialization", "status": "pass", "model_seed_pairs": compared})

    independent = ControlledSystem("independent", int(config["data"]["system_seed"]), device)
    x0 = torch.randn(64, device=device, requires_grad=True)
    action0 = torch.randn(1, 8, device=device)
    true_jacobian = torch.autograd.functional.jacobian(
        lambda value: independent.step_x(value.unsqueeze(0), action0)[0], x0
    )
    cross = torch.ones_like(true_jacobian)
    for block in range(4):
        sl = slice(block * 16, (block + 1) * 16)
        cross[sl, sl] = 0
    true_cross_max = float((true_jacobian * cross).abs().max().item())
    assert true_cross_max == 0.0, f"independent true cross-block Jacobian={true_cross_max}"

    block_model = build_oracle_model("oracle_block", config, seed_list[0], mean, independent.M).to(device)
    state = torch.randn(1, 64, device=device, requires_grad=True)
    block_jacobian = torch.autograd.functional.jacobian(
        lambda value: block_model._advance(value.unsqueeze(0), action0)[0], state[0]
    )
    block_cross = torch.ones_like(block_jacobian)
    for block in range(4):
        sl = slice(block * 16, (block + 1) * 16)
        block_cross[sl, sl] = 0
    predictor_cross_max = float((block_jacobian * block_cross).abs().max().item())
    assert predictor_cross_max == 0.0, f"oracle_block predictor cross-block Jacobian={predictor_cross_max}"
    checks.append({
        "name": "independent_true_and_oracle_block_cross_block_jacobian_zero",
        "status": "pass",
        "true_cross_max": true_cross_max,
        "predictor_cross_max": predictor_cross_max,
    })

    global_model = build_oracle_model("oracle_global4", config, seed_list[0], mean, system.M).to(device)
    local_model = build_oracle_model("oracle_local4", config, seed_list[0], mean, system.M).to(device)
    global_weights = global_model.global_message.detach().reshape(4, 4, 16)
    global_column_norms = global_weights.square().sum(dim=(0, 2)).sqrt()
    assert bool((global_column_norms > 0).all()), "global message does not read every state block"
    state0 = torch.randn(64, device=device, requires_grad=True)
    local_jacobian = torch.autograd.functional.jacobian(
        lambda value: torch.einsum(
            "bkd,kmd->bkm", value.reshape(1, 4, 16), local_model.local_message
        )[0],
        state0,
    )
    for block in range(4):
        allowed = torch.zeros(64, dtype=torch.bool, device=device)
        allowed[block * 16 : (block + 1) * 16] = True
        assert float(local_jacobian[block, :, ~allowed].abs().max().item()) == 0.0
    checks.append({
        "name": "global4_reads_all_blocks_local4_reads_own_block_only",
        "status": "pass",
        "global_block_projection_norms": global_column_norms.detach().cpu().tolist(),
    })
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--global16-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from allocation_guard import ensure_allocation

    allocation = ensure_allocation(require_gpu=True)
    import torch

    from oracle_run import load_configs, prepare_reference_helpers

    config, global16_config = load_configs(args.config, args.global16_config)
    prepare_reference_helpers(config)
    checks = run_checks(config, global16_config, torch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"allocation": allocation, "checks": checks}, indent=2), encoding="utf-8")
    print(json.dumps({"checks": len(checks), "status": "pass"}))


if __name__ == "__main__":
    main()
