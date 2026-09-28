"""PBS-guarded mechanism checks for the LeWM block-local pilot."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path


def require_compute_allocation() -> dict[str, str]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    hostname = socket.gethostname()
    if not job_id:
        raise RuntimeError("PBS_JOBID is required")
    if not nodefile or not Path(nodefile).is_file() or Path(nodefile).stat().st_size == 0:
        raise RuntimeError("PBS_NODEFILE must name a non-empty allocation nodefile")
    short_host = hostname.split(".", 1)[0].lower()
    if any(word in short_host for word in ("login", "head", "submit")):
        raise RuntimeError(f"refusing numerical checks on login/head/submit host {hostname!r}")
    allocated_hosts = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if short_host not in allocated_hosts:
        raise RuntimeError(f"current host {short_host!r} is absent from PBS_NODEFILE")
    return {"pbs_job_id": job_id, "hostname": hostname, "pbs_nodefile": nodefile}


def run_checks(device: str | None = "cuda") -> dict:
    """Run the small numerical contract suite; callers must already be on PBS compute."""
    allocation = require_compute_allocation()
    import io

    import torch
    from torch.nn import functional as F

    from models import ARMS, LATENT_DIM, build_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(260926)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(260926)

    checks = []

    def run(name, fn):
        try:
            details = fn() or {}
            checks.append({"name": name, "passed": True, "details": details})
        except Exception as exc:  # Keep later independent checks useful on a failed arm.
            checks.append({"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"})

    def tensor(batch, steps, width, *, requires_grad=False):
        return torch.randn(batch, steps, width, device=device, requires_grad=requires_grad)

    def shape_and_history_contract():
        for arm in sorted(ARMS):
            model = build_model(arm).to(device).eval()
            model.prepare_for_inference()
            for t in (1, 2, 3):
                history = tensor(2, t, LATENT_DIM)
                past_n = min(t - 1, 2)
                past = tensor(2, past_n, 10)
                actions = tensor(2, 5, 10)
                with torch.inference_mode():
                    out = model(history, actions, past)
                    one = model.step(history, actions[:, 0, :], past)
                assert out.shape == (2, 5, LATENT_DIM), (arm, t, out.shape)
                assert one.shape == (2, LATENT_DIM), (arm, t, one.shape)
                assert torch.isfinite(out).all() and torch.isfinite(one).all(), arm
            del model
        return {"arms": sorted(ARMS), "history_lengths": [1, 2, 3], "rollout_horizon": 5}

    def count_contract():
        expected = {
            "block_adapter_global16": (773094, 773094, 0),
            "block_adapter_local16": (773094, 773094, 0),
            "block_identity_global16": (773094, 736230, 36864),
            "flat_adapter": (779556, 779556, 0),
        }
        observed = {}
        for arm, want in expected.items():
            counts = build_model(arm).parameter_counts()
            got = (counts["total"], counts["trainable"], counts["frozen"])
            assert got == want, (arm, got, want)
            observed[arm] = counts
        return observed

    def cayley_contract():
        results = {}
        eye = torch.eye(LATENT_DIM, device=device)
        z = tensor(3, 1, LATENT_DIM).squeeze(1)
        for arm in sorted(ARMS):
            model = build_model(arm).to(device).eval()
            model.prepare_for_inference()
            q = model.rotation._cached_q
            orth_error = (q.transpose(0, 1) @ q - eye).abs().max().item()
            recovered = (z @ q) @ q.transpose(0, 1)
            inverse_error = (recovered - z).abs().max().item()
            assert orth_error < 2e-4, (arm, orth_error)
            assert inverse_error < 2e-4, (arm, inverse_error)
            assert torch.isfinite(q).all(), arm
            results[arm] = {"max_orthogonality_error": orth_error, "max_inverse_error": inverse_error}
        return results

    def latent_history_causality():
        results = {}
        for arm in sorted(ARMS):
            model = build_model(arm).to(device).eval()
            model.prepare_for_inference()
            history_a = tensor(2, 5, LATENT_DIM)
            history_b = history_a.clone()
            history_b[:, :-3, :] += 100.0
            past = tensor(2, 2, 10)
            current = tensor(2, 1, 10).squeeze(1)
            with torch.inference_mode():
                a = model.step(history_a, current, past)
                b = model.step(history_b, current, past)
            torch.testing.assert_close(a, b, atol=0, rtol=0)

            first = tensor(2, 1, LATENT_DIM)
            repeated = first.expand(-1, 3, -1)
            zeros2 = first.new_zeros((2, 2, 10))
            short_actions = tensor(2, 1, 10).squeeze(1)
            with torch.inference_mode():
                short = model.step(first, short_actions, None)
                padded = model.step(repeated, short_actions, zeros2)
            torch.testing.assert_close(short, padded, atol=0, rtol=0)

            two = tensor(2, 2, LATENT_DIM)
            past_one = tensor(2, 1, 10)
            padded_two = torch.cat((two[:, :1, :], two), dim=1)
            padded_actions = torch.cat((past_one.new_zeros((2, 1, 10)), past_one), dim=1)
            with torch.inference_mode():
                short = model.step(two, current, past_one)
                padded = model.step(padded_two, current, padded_actions)
            torch.testing.assert_close(short, padded, atol=0, rtol=0)
            results[arm] = "older-than-three latent tokens ignored"
        return results

    def padding_and_action_window_contract():
        from models import _action_window, _previous_actions

        current = torch.arange(2 * 10, device=device, dtype=torch.float32).reshape(2, 10)
        empty = current.new_zeros((2, 0, 10))
        one = current.unsqueeze(1) + 20
        two = current.unsqueeze(1) + torch.tensor([30.0, 40.0], device=device).reshape(1, 2, 1)
        w0 = _action_window(current, empty)
        w1 = _action_window(current, one)
        w2 = _action_window(current, two)
        assert torch.equal(w0[:, 0, :], torch.zeros_like(current))
        assert torch.equal(w0[:, 1, :], torch.zeros_like(current))
        assert torch.equal(w0[:, 2, :], current)
        assert torch.equal(w1[:, 0, :], torch.zeros_like(current))
        assert torch.equal(w1[:, 1, :], one[:, 0, :])
        assert torch.equal(w1[:, 2, :], current)
        assert torch.equal(w2[:, 0, :], two[:, 0, :])
        assert torch.equal(w2[:, 1, :], two[:, 1, :])
        assert torch.equal(w2[:, 2, :], current)

        try:
            _previous_actions(tensor(2, 3, LATENT_DIM), one)
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched initial action history length was accepted")

        model = build_model("block_adapter_global16").to(device).eval()
        model.prepare_for_inference()
        seen = []
        original = model._predict_delta_rotated

        def capture(s_history, action_window):
            seen.append(action_window.detach().clone())
            return original(s_history, action_window)

        model._predict_delta_rotated = capture
        history = tensor(1, 3, LATENT_DIM)
        initial = tensor(1, 2, 10)
        actions = tensor(1, 4, 10)
        with torch.inference_mode():
            model.predict_rollout(history, actions, initial)
        expected = [
            torch.cat((initial, actions[:, 0:1, :]), dim=1),
            torch.cat((initial[:, 1:2, :], actions[:, 0:2, :]), dim=1),
            torch.cat((actions[:, 0:2, :], actions[:, 2:3, :]), dim=1),
            torch.cat((actions[:, 1:3, :], actions[:, 3:4, :]), dim=1),
        ]
        assert len(seen) == len(expected)
        for got, want in zip(seen, expected):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        return {"padding": "zero-left-pad to three actions", "rollout": "only consumed actions enter the next window"}

    def action_prefix_causality():
        results = {}
        for arm in sorted(ARMS):
            model = build_model(arm).to(device).eval()
            model.prepare_for_inference()
            history = tensor(2, 3, LATENT_DIM)
            past = tensor(2, 2, 10)
            actions_a = tensor(2, 5, 10)
            actions_b = actions_a.clone()
            actions_b[:, 2:, :] += 20.0
            with torch.inference_mode():
                a = model(history, actions_a, past)
                b = model(history, actions_b, past)
            torch.testing.assert_close(a[:, :2], b[:, :2], atol=0, rtol=0)
            results[arm] = "future action suffix does not alter earlier predictions"
        return results

    def paired_initialization():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(8841)
            global_model = build_model("block_adapter_global16")
            torch.manual_seed(8841)
            local_model = build_model("block_adapter_local16")
        for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
            assert torch.equal(getattr(global_model.functions, name), getattr(local_model.functions, name)), name
        recovered_global = (
            local_model.message_weight.reshape(6, 16, 3, 32)
            .permute(1, 2, 0, 3)
            .reshape(16, 3 * 192)
        )
        assert torch.equal(global_model.message_weight, recovered_global), "message projection pairing"
        return {"paired": "all six function tensors and history-major message projection exactly match"}

    def batched_mlp_equivalence_and_init_bounds():
        from models import _BatchedBlockMLP

        module = _BatchedBlockMLP().to(device)
        x = tensor(4, 6, 142)
        batched = module(x)
        manual = []
        for block in range(6):
            h = F.gelu(F.linear(x[:, block], module.w1[block], module.b1[block]))
            h = F.gelu(F.linear(h, module.w2[block], module.b2[block]))
            manual.append(F.linear(h, module.w3[block], module.b3[block]))
        manual = torch.stack(manual, dim=1)
        torch.testing.assert_close(batched, manual, atol=2e-6, rtol=2e-6)

        bound_details = {}
        for name, weight, bias in (
            ("w1", module.w1, module.b1),
            ("w2", module.w2, module.b2),
            ("w3", module.w3, module.b3),
        ):
            bound = weight.shape[-1] ** -0.5
            weight_max = weight.detach().abs().amax(dim=(1, 2))
            bias_max = bias.detach().abs().amax(dim=1)
            assert torch.all(weight_max <= bound + 1e-7), (name, weight_max.max().item(), bound)
            assert torch.all(weight_max > 0.5 * bound), (name, weight_max.min().item(), bound)
            assert torch.all(bias_max <= bound + 1e-7), (name, bias_max.max().item(), bound)
            assert torch.all(bias_max > 0.5 * bound), (name, bias_max.min().item(), bound)
            bound_details[name] = {
                "fan_in": weight.shape[-1],
                "expected_uniform_bound": bound,
                "max_abs_weight": weight_max.max().item(),
                "max_abs_bias": bias_max.max().item(),
            }
        return {
            "calculation": "batched einsums match six independent F.linear stacks",
            "init_bounds": bound_details,
        }

    def cross_block_jacobian():
        results = {}
        for arm in ("block_adapter_global16", "block_identity_global16"):
            model = build_model(arm).to(device).train()
            s_history = tensor(1, 3, LATENT_DIM, requires_grad=True)
            action_window = tensor(1, 3, 10)
            out = model._predict_delta_rotated(s_history, action_window).reshape(1, 6, 32)
            grad = torch.autograd.grad(out[:, 0, :].sum(), s_history)[0]
            cross = grad[:, :, 32:64]
            magnitude = cross.abs().max().item()
            assert magnitude > 1e-10, (arm, magnitude)
            results[arm] = {"max_cross_block_abs_jacobian": magnitude}

        model = build_model("block_adapter_local16").to(device).train()
        s_history = tensor(1, 3, LATENT_DIM, requires_grad=True)
        action_window = tensor(1, 3, 10)
        out = model._predict_delta_rotated(s_history, action_window).reshape(1, 6, 32)
        grad = torch.autograd.grad(out[:, 0, :].sum(), s_history)[0]
        cross = grad[:, :, 32:64]
        assert torch.count_nonzero(cross).item() == 0, "local arm acquired a cross-block path"
        results["block_adapter_local16"] = {"max_cross_block_abs_jacobian": cross.abs().max().item()}
        return results

    def gradient_and_update():
        model = build_model("block_adapter_local16").to(device).train()
        history = tensor(3, 3, LATENT_DIM)
        past = tensor(3, 2, 10)
        actions = tensor(3, 5, 10)
        target = tensor(3, 5, LATENT_DIM)
        before_q = model.rotation.raw.detach().clone()
        before_local = model.functions.w1.detach().clone()
        pred = model(history, actions, past)
        loss = (pred - target).square().mean()
        loss.backward()
        q_grad = model.rotation.raw.grad
        local_grad = model.functions.w1.grad
        assert q_grad is not None and torch.isfinite(q_grad).all(), "non-finite/missing Q gradient"
        assert local_grad is not None and torch.isfinite(local_grad).all(), "non-finite/missing local gradient"
        q_norm, local_norm = q_grad.norm().item(), local_grad.norm().item()
        assert q_norm > 0 and local_norm > 0, (q_norm, local_norm)
        torch.optim.SGD(model.parameters(), lr=1e-3).step()
        assert not torch.equal(model.rotation.raw, before_q), "Q did not update"
        assert not torch.equal(model.functions.w1, before_local), "local functions did not update"
        return {"loss": loss.item(), "q_grad_norm": q_norm, "local_grad_norm": local_norm}

    def cache_and_serialization():
        results = {}
        for arm in sorted(ARMS):
            model = build_model(arm).to(device).eval()
            model.prepare_for_inference()
            history = tensor(2, 3, LATENT_DIM)
            past = tensor(2, 2, 10)
            actions = tensor(2, 4, 10)
            with torch.inference_mode():
                cached_out = model(history, actions, past)
                fresh_q = model.rotation.compute_matrix()
            torch.testing.assert_close(model.rotation._cached_q, fresh_q, atol=0, rtol=0)
            torch.testing.assert_close(cached_out, model.predict_rollout(history, actions, past), atol=0, rtol=0)

            assert not any("_cached_q" in key for key in model.state_dict()), "cache leaked into state_dict"
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            try:
                state = torch.load(stream, map_location=device, weights_only=True)
            except TypeError:  # Older supported PyTorch versions lack weights_only.
                stream.seek(0)
                state = torch.load(stream, map_location=device)
            clone = build_model(arm).to(device)
            clone.load_state_dict(state)
            assert clone.rotation._cached_q is None, "loading state retained a stale Q cache"
            clone.eval().prepare_for_inference()
            with torch.inference_mode():
                clone_out = clone(history, actions, past)
            torch.testing.assert_close(cached_out, clone_out, atol=2e-5, rtol=2e-5)

            model.train()
            assert model.rotation._cached_q is None, "train() did not invalidate Q cache"
            results[arm] = "cached/fresh parity, non-persistent state serialization, and invalidation passed"
        return results

    for name, fn in (
        ("shapes_history_and_finite", shape_and_history_contract),
        ("parameter_budgets", count_contract),
        ("cayley_orthogonality_and_inverse", cayley_contract),
        ("latent_history_causality", latent_history_causality),
        ("action_window_padding_and_roll", padding_and_action_window_contract),
        ("future_action_prefix_causality", action_prefix_causality),
        ("global_local_paired_initialization", paired_initialization),
        ("batched_mlp_equivalence_and_init_bounds", batched_mlp_equivalence_and_init_bounds),
        ("rotated_coordinate_cross_block_jacobian", cross_block_jacobian),
        ("finite_gradients_and_parameter_updates", gradient_and_update),
        ("prepared_cache_and_state_serialization", cache_and_serialization),
    ):
        run(name, fn)

    passed = all(check["passed"] for check in checks)
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "device": str(device),
        "allocation": allocation,
        "checks": checks,
    }


def main() -> int:
    result = run_checks(device=None)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
