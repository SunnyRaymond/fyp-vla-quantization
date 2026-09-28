"""Train and evaluate the frozen C-SWM visual block-local pilot on PBS GPU."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
try:
    from allocation_guard import ensure_allocation
except ImportError:
    sys.path.insert(0, str(ROOT.parent / "block-local-action-dynamics"))
    from allocation_guard import ensure_allocation


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    identity = ensure_allocation(require_gpu=True)
    import numpy as np
    import torch
    from PIL import Image

    from model_checks import run_model_checks
    from prepare_data import BlockPushingCompat
    from visual_models import (
        make_encoder, make_predictor, parameter_budget, parameter_count,
    )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("frozen_config") != config:
        raise RuntimeError("Prepared data is incomplete or its frozen config differs from the GPU-run config")
    if len(config["stage_b_predictor_fit"]["arms"]) != 7:
        raise RuntimeError("Frozen protocol must contain exactly seven predictor arms")

    output = args.output.resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output / "FREEZE.json")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)

    model_checks = run_model_checks(config, device)
    checks = {
        "allocation": {"pass": True, "message": identity},
        "frozen_data_config_match": {"pass": True, "message": "prepared manifest embeds the same complete FREEZE.json"},
        "model_checks": model_checks,
    }
    write_json(output / "mechanism_tests.json", checks)
    failed = [name for name, value in model_checks.items() if not value.get("pass", False)]
    if failed:
        raise RuntimeError("Visual model API checks failed: " + ", ".join(failed))

    seeds = config["stage_a_encoder_fit"]["seeds"]
    arms = config["stage_b_predictor_fit"]["arms"]
    horizons = config["evaluation"]["horizons"]
    fit_updates = int(config["stage_a_encoder_fit"]["updates_per_seed"])
    updates = int(config["stage_b_predictor_fit"]["updates_per_arm_seed"])
    batch_size = int(config["stage_b_predictor_fit"]["batch_size"])
    ntrain = int(config["data"]["episodes"]["train"])
    ntest = int(config["data"]["episodes"]["test"])
    steps = int(config["data"]["steps_per_episode"])

    data_arrays = {}
    for split in ("train", "dev", "test"):
        data_arrays[split] = {
            "obs": np.load(args.data / f"{split}_obs.npy", mmap_mode="r"),
            "action": np.load(args.data / f"{split}_action.npy", mmap_mode="r"),
            "position": np.load(args.data / f"{split}_position.npy", mmap_mode="r"),
            "event": np.load(args.data / f"{split}_event.npy", mmap_mode="r"),
        }
        expected = config["data"]["episodes"][split]
        if data_arrays[split]["obs"].shape != (expected, steps + 1, 3, 50, 50):
            raise RuntimeError(f"{split} observation shape mismatch: {data_arrays[split]['obs'].shape}")
        if data_arrays[split]["action"].shape != (expected, steps):
            raise RuntimeError(f"{split} action shape mismatch: {data_arrays[split]['action'].shape}")

    def image_batch(array, flat_indices):
        view = array.reshape(-1, 3, 50, 50)
        host = np.asarray(view[flat_indices], dtype=np.uint8).copy()
        return torch.from_numpy(host).to(device=device, dtype=torch.float32).div_(255.0)

    @torch.no_grad()
    def encode_split(encoder, split):
        obs = data_arrays[split]["obs"]
        count = obs.shape[0] * obs.shape[1]
        latent = torch.empty(obs.shape[0], obs.shape[1], 5, 16, device=device, dtype=torch.float32)
        chunk = 256
        encoder.eval()
        flat = obs.reshape(-1, 3, 50, 50)
        for start in range(0, count, chunk):
            ids = np.arange(start, min(start + chunk, count), dtype=np.int64)
            x = image_batch(flat, ids)
            z = encoder(x)
            latent.reshape(count, 5, 16)[start:start + len(ids)] = z
        return latent

    def progress(stage, seed, arm=None, step=None, **extra):
        value = {"stage": stage, "seed": int(seed), "arm": arm, "step": step, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        value.update(extra)
        write_json(output / "progress.json", value)

    def save_summary_progress():
        completed_stage_a = list(results["stage_a"])
        completed_stage_b = list(results["stage_b"])
        write_json(output / "summary_progress.json", {
            "protocol_id": config["protocol_id"],
            "completed_stage_a_fits": completed_stage_a,
            "completed_stage_b_fits": completed_stage_b,
            "stage_a": results["stage_a"],
            "stage_b": results["stage_b"],
            "parameter_budget": results["parameter_budget"],
        })

    def initialize_official_conv(encoder, seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            for layer in encoder.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

    def fit_stage_a(seed):
        encoder = make_encoder(config, seed)
        initialize_official_conv(encoder, seed)
        encoder = encoder.to(device)
        transition = make_predictor("full_gnn", config, seed + 100_000).to(device)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(transition.parameters()),
            lr=float(config["stage_a_encoder_fit"]["learning_rate"]),
            weight_decay=float(config["stage_a_encoder_fit"]["weight_decay"]),
        )
        obs = data_arrays["train"]["obs"]
        actions = data_arrays["train"]["action"]
        rng = np.random.default_rng(seed + 731_001)
        negative_rng = torch.Generator(device="cpu").manual_seed(seed + 731_003)
        losses, started = [], time.perf_counter()
        encoder.train(); transition.train()
        for update in range(fit_updates):
            flat = rng.integers(0, ntrain * steps, size=batch_size, dtype=np.int64)
            episode, t = flat // steps, flat % steps
            x = image_batch(obs, episode * (steps + 1) + t)
            y = image_batch(obs, episode * (steps + 1) + t + 1)
            action = torch.as_tensor(np.asarray(actions[episode, t]).copy(), device=device, dtype=torch.long)
            state = encoder(x)
            target = encoder(y)
            prediction = transition(state, action)
            scale = 0.5 / (float(config["stage_a_encoder_fit"]["contrastive_sigma"]) ** 2)
            positive = (prediction - target).square().sum(-1).mean(-1) * scale
            permutation = torch.randperm(batch_size, generator=negative_rng).to(device)
            negative = (state - state.index_select(0, permutation)).square().sum(-1).mean(-1) * scale
            loss = positive.mean() + torch.relu(float(config["stage_a_encoder_fit"]["contrastive_hinge"]) - negative).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
            if (update + 1) % 500 == 0 or update + 1 == fit_updates:
                progress("stage_a", seed, step=update + 1, mean_loss=float(torch.stack(losses[-500:]).mean().item()), elapsed_s=round(time.perf_counter() - started, 2))
        encoder.eval()
        seed_dir = checkpoints / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), seed_dir / "encoder.pt")
        torch.save(transition.state_dict(), seed_dir / "stage_a_gnn.pt")
        return encoder, {
            "updates": fit_updates, "batch_size": batch_size,
            "train_loss_first_500": float(torch.stack(losses[:500]).mean().item()),
            "train_loss_last_500": float(torch.stack(losses[-500:]).mean().item()),
            "elapsed_s": time.perf_counter() - started,
            "encoder_params": parameter_count(encoder),
            "stage_a_gnn_params": parameter_count(transition),
        }

    def position_probe(train_z, train_pos, test_z, test_pos):
        """Train-only slot/object linear map and train-only assignment."""
        z = train_z.detach().cpu().numpy().reshape(-1, 5, 16)
        pos = np.asarray(train_pos).reshape(-1, 5, 2).astype(np.float64)
        test_lat = test_z.detach().cpu().numpy().reshape(-1, 5, 16)
        truth = np.asarray(test_pos).reshape(-1, 5, 2).astype(np.float64)
        design = np.concatenate([z, np.ones((len(z), 5, 1), dtype=np.float32)], axis=-1)
        test_design = np.concatenate([test_lat, np.ones((len(test_lat), 5, 1), dtype=np.float32)], axis=-1)
        coefficients, cost = {}, np.zeros((5, 5), dtype=np.float64)
        for slot in range(5):
            # Fit every possible object target from the same slot, only on train episodes.
            coef_by_object = []
            for obj in range(5):
                target = pos[:, obj]
                coef, _, _, _ = np.linalg.lstsq(design[:, slot], target, rcond=None)
                prediction = design[:, slot] @ coef
                cost[slot, obj] = np.mean(np.square(prediction - target))
                coef_by_object.append(coef)
            coefficients[slot] = coef_by_object
        best = min(itertools.permutations(range(5)), key=lambda p: sum(cost[s, p[s]] for s in range(5)))
        predictions = np.zeros_like(truth)
        for slot, obj in enumerate(best):
            predictions[:, obj] = test_design[:, slot] @ coefficients[slot][obj]
        return {
            "test_rmse_grid_cells": float(np.sqrt(np.mean(np.square(predictions - truth)))),
            "train_selected_slot_to_object": list(map(int, best)),
            "fit_samples_train_only": int(len(z)),
        }

    def save_slot_preview(encoder, seed, test_obs):
        from PIL import ImageDraw
        encoder.eval()
        frame = np.asarray(test_obs[0, 0]).copy()
        x = torch.from_numpy(frame[None]).to(device=device, dtype=torch.float32).div_(255.0)
        with torch.inference_mode():
            maps = torch.sigmoid(encoder.cnn2(torch.relu(encoder.bn1(encoder.cnn1(x)))))[0].detach().cpu().numpy()
        canvas = Image.new("RGB", (6 * 102, 122), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        canvas.paste(Image.fromarray(np.transpose(frame, (1, 2, 0)), mode="RGB").resize((100, 100), Image.Resampling.NEAREST), (1, 18))
        draw.text((2, 2), "input", fill=(0, 0, 0))
        for slot in range(5):
            mask = np.clip(maps[slot] * 255, 0, 255).astype(np.uint8)
            tile = Image.fromarray(mask, mode="L").resize((100, 100), Image.Resampling.NEAREST).convert("RGB")
            x0 = (slot + 1) * 102 + 1
            canvas.paste(tile, (x0, 18))
            draw.text((x0 + 2, 2), f"slot {slot}", fill=(0, 0, 0))
        path = output / f"encoder_slots_seed{seed}.png"
        canvas.save(path, format="PNG", optimize=True)
        return {"path": path.name, "bytes": path.stat().st_size}

    def make_draws(seed):
        rng = np.random.default_rng(seed + 910_009)
        episode = rng.integers(0, ntrain, size=(updates, batch_size), dtype=np.int32)
        start = rng.integers(0, steps - 5 + 1, size=(updates, batch_size), dtype=np.int16)
        return episode, start

    def fit_predictor(arm, seed, model_seed, train_z, train_actions, delta_energy, draws_ep, draws_t):
        model = make_predictor(arm, config, model_seed).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["stage_b_predictor_fit"]["learning_rate"]),
            weight_decay=float(config["stage_b_predictor_fit"]["weight_decay"]),
        )
        losses, started = [], time.perf_counter()
        model.train()
        for update in range(updates):
            ep = torch.as_tensor(draws_ep[update], device=device, dtype=torch.long)
            t = torch.as_tensor(draws_t[update], device=device, dtype=torch.long)
            batch = torch.arange(batch_size, device=device)
            state = train_z[ep, t]
            pred = model(state, train_actions[ep, t])
            errors = [(pred - train_z[ep, t + 1]).square().mean()]
            for h in range(1, 5):
                pred = model(pred, train_actions[ep, t + h])
                errors.append((pred - train_z[ep, t + h + 1]).square().mean())
            one_step = errors[0]
            rollout = torch.stack(errors).mean()
            loss = (one_step + 0.5 * rollout) / delta_energy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
            if (update + 1) % 500 == 0 or update + 1 == updates:
                progress("stage_b", seed, arm=arm, step=update + 1,
                         mean_loss=float(torch.stack(losses[-500:]).mean().item()), elapsed_s=round(time.perf_counter() - started, 2))
        model.eval()
        return model, {
            "updates": updates, "batch_size": batch_size,
            "train_loss_first_500": float(torch.stack(losses[:500]).mean().item()),
            "train_loss_last_500": float(torch.stack(losses[-500:]).mean().item()),
            "elapsed_s": time.perf_counter() - started,
            "trainable_params": parameter_count(model), "model_seed": int(model_seed),
        }

    @torch.inference_mode()
    def score_rollouts(model, test_z, test_actions):
        model.eval()
        n = test_z.shape[0]
        state = test_z[:, 0]
        predictions = {}
        max_h = max(horizons)
        for h in range(1, max_h + 1):
            state = model(state, test_actions[:, h - 1])
            if h in horizons:
                predictions[h] = state.clone()
        mse = np.zeros((n, len(horizons)), dtype=np.float32)
        h_at_1 = np.zeros_like(mse)
        mrr = np.zeros_like(mse)
        for col, h in enumerate(horizons):
            pred = predictions[h].flatten(1)
            targets = test_z[:, h].flatten(1)
            distance = torch.cdist(pred, targets, p=2).square()
            true_dist = distance.diagonal()
            # Count only strictly closer negatives: positive is the fixed first tie-break.
            rank = (distance < true_dist.unsqueeze(1)).sum(1) + 1
            mse[:, col] = (predictions[h] - test_z[:, h]).square().mean((1, 2)).cpu().numpy()
            h_at_1[:, col] = (rank == 1).to(torch.float32).cpu().numpy()
            mrr[:, col] = rank.float().reciprocal().cpu().numpy()
        return predictions, {"latent_mse": mse, "h_at_1": h_at_1, "mrr": mrr}

    @torch.inference_mode()
    def one_step_subsets(model, test_z, test_actions, event):
        model.eval()
        flat_z0 = test_z[:, :-1].reshape(-1, 5, 16)
        flat_zt = test_z[:, 1:].reshape(-1, 5, 16)
        flat_a = test_actions.reshape(-1)
        flat_event = np.asarray(event).reshape(-1)
        sums = np.zeros(3, dtype=np.float64)
        counts = np.zeros(3, dtype=np.int64)
        for start in range(0, len(flat_a), 256):
            end = min(start + 256, len(flat_a))
            pred = model(flat_z0[start:end], flat_a[start:end])
            error = (pred - flat_zt[start:end]).square().mean((1, 2)).cpu().numpy()
            categories = flat_event[start:end]
            for category in range(3):
                mask = categories == category
                sums[category] += float(error[mask].sum())
                counts[category] += int(mask.sum())
        return {
            EVENT_NAME: {"mean_latent_mse": float(sums[i] / max(1, counts[i])), "transitions": int(counts[i])}
            for i, EVENT_NAME in enumerate(("free", "other-object-blocked", "boundary-blocked"))
        }

    @torch.inference_mode()
    def make_counterfactual_latents(encoder, test_actions, test_positions):
        n = test_actions.shape[0]
        base_action = test_actions[:, 0]
        alt_action = (base_action // 4) * 4 + ((base_action % 4 + 2) % 4)
        alt_images = np.empty((n, 3, 50, 50), dtype=np.uint8)
        positions = np.asarray(test_positions[:, 0]).astype(np.int64)
        for i in range(n):
            env = BlockPushingCompat(np.random.default_rng(i), positions[i])
            env.step(int(alt_action[i].item()))
            alt_images[i] = env.render()
        z_alt = torch.empty(n, 5, 16, device=device)
        for start in range(0, n, 64):
            batch = torch.from_numpy(alt_images[start:start + 64].copy()).to(device=device, dtype=torch.float32).div_(255.0)
            z_alt[start:start + len(batch)] = encoder(batch)
        return alt_action, z_alt

    @torch.inference_mode()
    def counterfactual_response(model, test_z, test_actions, alt_action, z_alt, test_event):
        base_action = test_actions[:, 0]
        base_pred = model(test_z[:, 0], base_action)
        alt_pred = model(test_z[:, 0], alt_action)
        true_base = test_z[:, 1]
        predicted_response = base_pred - alt_pred
        true_response = true_base - z_alt
        per_episode = (predicted_response - true_response).square().mean((1, 2)).cpu().numpy().astype(np.float32)
        pred_magnitude = predicted_response.square().mean((1, 2)).sqrt().cpu().numpy().astype(np.float32)
        true_magnitude = true_response.square().mean((1, 2)).sqrt().cpu().numpy().astype(np.float32)
        categories = np.asarray(test_event[:, 0])
        grouped = {}
        for category, name in enumerate(("free", "other-object-blocked", "boundary-blocked")):
            keep = categories == category
            grouped[name] = {
                "episodes": int(keep.sum()),
                "response_mse": float(per_episode[keep].mean()) if keep.any() else None,
                "predicted_rms_delta": float(pred_magnitude[keep].mean()) if keep.any() else None,
                "true_rms_delta": float(true_magnitude[keep].mean()) if keep.any() else None,
            }
        return per_episode, grouped

    def latency(model, encoder, test_z, test_actions, test_obs):
        model.eval(); encoder.eval()
        timing = config["timing"]
        warmups = int(timing["warmup_iterations"])
        repeats = int(timing["timed_repeats"])
        max_h = int(timing["horizon"])
        sizes = {}

        @torch.inference_mode()
        def timed(call, count):
            for _ in range(warmups):
                call()
            torch.cuda.synchronize(device)
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                call()
                torch.cuda.synchronize(device)
                samples.append((time.perf_counter() - started) * 1000.0)
            return {
                "median_ms": float(np.median(samples)),
                "p90_ms": float(np.quantile(samples, 0.90)),
                "mean_ms": float(np.mean(samples)),
                "repeats": repeats,
            }

        for batch in timing["batch_sizes"]:
            idx = torch.arange(batch, device=device) % test_z.shape[0]
            states = test_z.index_select(0, idx)[:, 0].contiguous()
            actions = test_actions.index_select(0, idx)[:, :max_h].contiguous()
            def rollout_call():
                state = states
                for h in range(max_h):
                    state = model(state, actions[:, h])
            predictor_timing = timed(rollout_call, batch)
            sizes[str(batch)] = {
                "predictor_h10_ms": predictor_timing["median_ms"],
                "predictor_h10_median_ms": predictor_timing["median_ms"],
                "predictor_h10_p90_ms": predictor_timing["p90_ms"],
                "predictor_h10_mean_ms": predictor_timing["mean_ms"],
                "timed_repeats": predictor_timing["repeats"],
            }
            images = torch.from_numpy(np.asarray(test_obs.reshape(-1, 3, 50, 50)[:batch]).copy()).to(device=device, dtype=torch.float32).div_(255.0)
            encoder_timing = timed(lambda: encoder(images), batch)
            sizes[str(batch)]["encoder_initial_ms"] = encoder_timing["median_ms"]
            sizes[str(batch)]["encoder_initial_median_ms"] = encoder_timing["median_ms"]
            sizes[str(batch)]["encoder_initial_p90_ms"] = encoder_timing["p90_ms"]
            sizes[str(batch)]["encoder_initial_mean_ms"] = encoder_timing["mean_ms"]
        return sizes

    results = {"stage_a": {}, "stage_b": {}, "test": {}, "dev": {}, "latent_variance": {}, "latency": {}, "position_probe": {}, "slot_previews": {}, "one_step_subsets": {}, "rollout_subsets": {}, "action_response": {}, "parameter_budget": parameter_budget(config)}
    per_episode_by_seed = {}
    delta_energy_by_seed = {}
    model_order_by_seed = {}
    test_arrays = data_arrays["test"]
    test_actions_np = np.asarray(test_arrays["action"]).copy()
    test_positions = test_arrays["position"]
    test_events = test_arrays["event"]
    test_obs = test_arrays["obs"]

    for seed in seeds:
        print(f"STAGE_A seed={seed}", flush=True)
        encoder, stage_a_info = fit_stage_a(int(seed))
        encoder.requires_grad_(False)
        results["stage_a"][str(seed)] = stage_a_info
        save_summary_progress()
        train_z = encode_split(encoder, "train")
        dev_z = encode_split(encoder, "dev")
        test_z = encode_split(encoder, "test")
        if any(z.is_inference() for z in (train_z, dev_z, test_z)):
            raise RuntimeError("Feature cache contains inference tensors and cannot be saved for predictor backward")
        results["latent_variance"][str(seed)] = {}
        for split_name, split_z in (("train", train_z), ("test", test_z)):
            std = split_z.reshape(-1, 5, 16).std(dim=0, unbiased=False)
            results["latent_variance"][str(seed)][split_name] = {
                "std_per_slot": std.mean(-1).cpu().tolist(),
                "std_mean": float(std.mean().item()),
                "std_min": float(std.min().item()),
                "std_max": float(std.max().item()),
                "dimensions_below_1e-3": int((std < 1e-3).sum().item()),
                "dimension_count": int(std.numel()),
            }
        dev_actions = torch.from_numpy(np.asarray(data_arrays["dev"]["action"]).copy()).to(device=device, dtype=torch.long)
        results["position_probe"][str(seed)] = position_probe(
            train_z, data_arrays["train"]["position"], test_z, test_positions
        )
        results["slot_previews"][str(seed)] = save_slot_preview(encoder, int(seed), test_obs)
        train_actions = torch.from_numpy(np.asarray(data_arrays["train"]["action"]).copy()).to(device=device, dtype=torch.long)
        test_actions = torch.from_numpy(test_actions_np).to(device=device, dtype=torch.long)
        delta_energy = float((train_z[:, 1:] - train_z[:, :-1]).square().mean().item())
        if not math.isfinite(delta_energy) or delta_energy <= 1e-12:
            raise RuntimeError(f"Invalid train-only latent delta energy for seed {seed}: {delta_energy}")
        delta_energy_by_seed[str(seed)] = delta_energy
        results["stage_a"][str(seed)]["train_delta_energy"] = delta_energy
        draws_ep, draws_t = make_draws(int(seed))
        counterfactual_actions, counterfactual_z = make_counterfactual_latents(
            encoder, test_actions, test_positions
        )
        arm_order = list(arms)
        np.random.default_rng(int(seed) + 303_991).shuffle(arm_order)
        model_order_by_seed[str(seed)] = arm_order
        seed_dir = checkpoints / f"seed_{seed}"
        per_seed = {}
        saved_models = {}
        for arm_index, arm in enumerate(arm_order):
            model_seed = int(seed) * 100 + 1
            print(f"STAGE_B seed={seed} arm={arm} model_seed={model_seed}", flush=True)
            model, train_info = fit_predictor(
                arm, int(seed), model_seed, train_z, train_actions,
                delta_energy, draws_ep, draws_t,
            )
            torch.save(model.state_dict(), seed_dir / f"predictor_{arm}.pt")
            results["stage_b"][f"{seed}/{arm}"] = train_info
            save_summary_progress()
            predictions, metrics = score_rollouts(model, test_z, test_actions)
            per_seed[arm] = metrics
            _, dev_metrics = score_rollouts(model, dev_z, dev_actions)
            results["dev"][f"{seed}/{arm}"] = {
                metric: np.asarray(values).mean(axis=0).tolist()
                for metric, values in dev_metrics.items()
            }
            initial_categories = np.asarray(test_events[:, 0])
            results["rollout_subsets"][f"{seed}/{arm}"] = {}
            for category, name in enumerate(("free", "other-object-blocked", "boundary-blocked")):
                mask = initial_categories == category
                results["rollout_subsets"][f"{seed}/{arm}"][name] = {
                    "episodes": int(mask.sum()),
                    "by_horizon": {
                        str(h): {
                            "latent_mse": float(metrics["latent_mse"][mask, col].mean()) if mask.any() else None,
                            "h_at_1": float(metrics["h_at_1"][mask, col].mean()) if mask.any() else None,
                            "mrr": float(metrics["mrr"][mask, col].mean()) if mask.any() else None,
                        }
                        for col, h in enumerate(horizons)
                    },
                }
            event_summary = one_step_subsets(model, test_z, test_actions, test_events)
            results["one_step_subsets"][f"{seed}/{arm}"] = event_summary
            response_rows, response_summary = counterfactual_response(
                model, test_z, test_actions, counterfactual_actions, counterfactual_z, test_events
            )
            results["action_response"][f"{seed}/{arm}"] = response_summary
            results["latency"][f"{seed}/{arm}"] = latency(model, encoder, test_z, test_actions, test_obs)
            saved_models[arm] = {"episode": metrics, "response": response_rows}
            del model, predictions
            torch.cuda.empty_cache()
            progress("stage_b_complete", seed, arm=arm, step=updates)

        persistence = {"latent_mse": np.zeros((ntest, len(horizons)), dtype=np.float32),
                       "h_at_1": np.zeros((ntest, len(horizons)), dtype=np.float32),
                       "mrr": np.zeros((ntest, len(horizons)), dtype=np.float32)}
        for col, h in enumerate(horizons):
            pred = test_z[:, 0].flatten(1)
            target = test_z[:, h].flatten(1)
            distance = torch.cdist(pred, target, p=2).square()
            true_dist = distance.diagonal()
            rank = (distance < true_dist.unsqueeze(1)).sum(1) + 1
            persistence["latent_mse"][:, col] = (pred - target).square().mean(-1).cpu().numpy()
            persistence["h_at_1"][:, col] = (rank == 1).to(torch.float32).cpu().numpy()
            persistence["mrr"][:, col] = rank.float().reciprocal().cpu().numpy()
        per_seed["persistence"] = persistence
        initial_categories = np.asarray(test_events[:, 0])
        results["rollout_subsets"][f"{seed}/persistence"] = {}
        for category, name in enumerate(("free", "other-object-blocked", "boundary-blocked")):
            mask = initial_categories == category
            results["rollout_subsets"][f"{seed}/persistence"][name] = {
                "episodes": int(mask.sum()),
                "by_horizon": {
                    str(h): {
                        "latent_mse": float(persistence["latent_mse"][mask, col].mean()) if mask.any() else None,
                        "h_at_1": float(persistence["h_at_1"][mask, col].mean()) if mask.any() else None,
                        "mrr": float(persistence["mrr"][mask, col].mean()) if mask.any() else None,
                    }
                    for col, h in enumerate(horizons)
                },
            }
        per_episode_by_seed[str(seed)] = per_seed
        results["test"][str(seed)] = {
            arm: {metric: np.asarray(values).mean(axis=0).tolist() for metric, values in per_seed[arm].items()}
            for arm in arms
        }
        results["test"][str(seed)]["persistence"] = {
            metric: values.mean(axis=0).tolist() for metric, values in persistence.items()
        }
        names = list(arms) + ["persistence"]
        arrays = {}
        for metric in ("latent_mse", "h_at_1", "mrr"):
            arrays[metric] = np.stack([per_seed[name][metric] for name in names], axis=0).astype(np.float32)
        arrays["action_response_mse"] = np.stack([saved_models[arm]["response"] for arm in arms], axis=0).astype(np.float32)
        arrays["arm_names"] = np.asarray(names, dtype="U24")
        arrays["horizons"] = np.asarray(horizons, dtype=np.int16)
        path = output / f"episode_metrics_{seed}.npz"
        np.savez_compressed(path, **arrays)
        results["stage_a"][str(seed)]["episode_metrics_file"] = {"path": path.name, "bytes": path.stat().st_size}
        del encoder, train_z, dev_z, test_z, train_actions, dev_actions, test_actions, counterfactual_z
        torch.cuda.empty_cache()

    # Seed-level paired episode bootstrap for H@1 differences at the primary h10.
    bootstrap_seed = 810_026
    boot_rng = np.random.default_rng(bootstrap_seed)
    bootstrap_n = int(config["evaluation"].get("bootstrap_resamples", 1000))
    paired = {}
    h10_col = horizons.index(10)
    for seed in seeds:
        per_seed = per_episode_by_seed[str(seed)]
        base = per_seed["full_gnn"]["h_at_1"][:, h10_col]
        indices = boot_rng.integers(0, len(base), size=(bootstrap_n, len(base)))
        for arm in arms:
            delta = per_seed[arm]["h_at_1"][:, h10_col] - base
            boot = delta[indices].mean(axis=1)
            paired[f"{seed}/{arm}-full_gnn"] = {
                "mean_h10_h_at_1_difference": float(delta.mean()),
                "bootstrap_95ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                "episodes": int(len(delta)), "bootstrap_resamples": bootstrap_n,
            }
        communication_delta = per_seed["global4"]["h_at_1"][:, h10_col] - per_seed["local4"]["h_at_1"][:, h10_col]
        communication_boot = communication_delta[indices].mean(axis=1)
        paired[f"{seed}/global4-local4"] = {
            "mean_h10_h_at_1_difference": float(communication_delta.mean()),
            "bootstrap_95ci": [float(np.quantile(communication_boot, 0.025)), float(np.quantile(communication_boot, 0.975))],
            "episodes": int(len(communication_delta)), "bootstrap_resamples": bootstrap_n,
        }

    adequacy = {}
    decisions = {}
    for seed in seeds:
        seed = int(seed)
        test = results["test"][str(seed)]
        gnn = test["full_gnn"]["h_at_1"]
        persistence_h10 = test["persistence"]["h_at_1"][h10_col]
        passes = (gnn[horizons.index(1)] >= 0.80 and gnn[h10_col] >= 0.50
                  and gnn[h10_col] - persistence_h10 >= 0.05)
        adequacy[str(seed)] = {
            "pass": bool(passes), "full_gnn_h1_h_at_1": gnn[horizons.index(1)],
            "full_gnn_h10_h_at_1": gnn[h10_col],
            "persistence_h10_h_at_1": persistence_h10,
            "h10_gain_over_persistence": gnn[h10_col] - persistence_h10,
        }
        for arm in arms:
            quality_diff = test[arm]["h_at_1"][h10_col] - gnn[h10_col]
            quality_pass = quality_diff >= -0.03
            latency_row = results["latency"][f"{seed}/{arm}"]
            decisions[f"{seed}/{arm}"] = {"quality_h10_diff": quality_diff, "quality_pass": bool(quality_pass), "speed": {}}
            for batch in config["timing"]["batch_sizes"]:
                candidate = latency_row[str(batch)]["predictor_h10_ms"]
                gnn_ms = results["latency"][f"{seed}/full_gnn"][str(batch)]["predictor_h10_ms"] if arm != "full_gnn" else candidate
                trans_ms = results["latency"][f"{seed}/transformer6"][str(batch)]["predictor_h10_ms"] if arm != "transformer6" else candidate
                speed_gnn = 1.0 - candidate / gnn_ms if gnn_ms else 0.0
                speed_transformer = 1.0 - candidate / trans_ms if trans_ms else 0.0
                speed_pass = (speed_gnn >= 0.20 and speed_transformer >= 0.20)
                decisions[f"{seed}/{arm}"]["speed"][str(batch)] = {
                    "vs_full_gnn_fractional_reduction": speed_gnn,
                    "vs_transformer6_fractional_reduction": speed_transformer,
                    "pass_both": bool(speed_pass),
                    "quality_and_speed_promising": bool(quality_pass and speed_pass),
                }

    all_adequate = all(value["pass"] for value in adequacy.values())
    decision = {
        "protocol_id": config["protocol_id"],
        "all_24_fits_complete": len(results["stage_a"]) == 3 and len(results["stage_b"]) == 21,
        "visual_reference_adequate_all_seeds": bool(all_adequate),
        "interpretation": "visual reference adequate; assess predictor gates per seed and batch" if all_adequate else "inconclusive: at least one frozen visual reference adequacy criterion failed; all planned fits remain reported",
        "encoder_adequacy_by_seed": adequacy,
        "arm_gates_by_seed": decisions,
    }
    results["paired_episode_bootstrap"] = paired
    results["encoder_adequacy"] = adequacy
    results["arm_decisions"] = decisions
    results["experiment_scope"] = {
        "stage_a_fits": len(results["stage_a"]), "stage_b_fits": len(results["stage_b"]),
        "total_fits": len(results["stage_a"]) + len(results["stage_b"]),
        "seeds": list(map(int, seeds)), "arm_order_by_seed": model_order_by_seed,
        "train_only_delta_energy": delta_energy_by_seed,
        "all_training_steps_fixed": True,
    }
    results["runtime"] = {
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "precision": "float32, autocast off", "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "device": str(device), "execution_identity": identity,
    }

    write_json(output / "summary.json", results)
    write_json(output / "DECISION.json", decision)
    report = build_report(config, results, decision, paired, arms, seeds, horizons, parameter_budget(config))
    (output / "REPORT.zh.md").write_text(report, encoding="utf-8")
    done = {
        "status": "complete", "protocol_id": config["protocol_id"],
        "stage_a_fits": len(results["stage_a"]), "stage_b_fits": len(results["stage_b"]),
        "encoder_adequate_all_seeds": bool(all_adequate),
        "decision": decision["interpretation"],
    }
    write_json(output / "DONE.json", done)
    print(json.dumps(done), flush=True)


def build_report(config, results, decision, paired, arms, seeds, horizons, budget):
    primary = horizons.index(10)
    lines = [
        "# C-SWM 视觉分块动力学 pilot 结果", "",
        f"Protocol: `{config['protocol_id']}`；官方来源 commit `{config['source']['commit']}`。", "",
        f"**结果判定：{decision['interpretation']}。** 预定 fits：Stage A {decision['all_24_fits_complete'] and 3 or len(results['stage_a'])}/3，Stage B {len(results['stage_b'])}/21。没有因 dev/test 结果增加训练步数或追加 fits。", "",
        "## 先看结论", "",
        "该 pilot 比较 C-SWM-style object-slot predictor、局部模块、低维摘要、六层 Transformer 与参数匹配 flat MLP。每个 seed 的 encoder 与 full GNN 联合训练后冻结，因此这组表示对 GNN 有训练偏置；若局部模型成功，只能证明已有 C-SWM object-slot interface 对该环境有利，不能说明 LeWM latent 已可同样分解或已实现 LeWM 加速。", "",
        f"global4 的 flat 参数匹配对照预算为 {budget['global4_params']:,} 参数；flat MLP hidden={budget['flat_hidden']}，参数={budget['flat_params']:,}（差异 {budget['relative_difference'] * 100:.3f}%）。", "",
        "## Encoder 与参考模型是否可用", "",
        "| seed | full GNN H@1 h1 | full GNN H@1 h10 | persistence H@1 h10 | 提升 | adequate | test latent std | near-zero dims | position probe test RMSE |", "|---:|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for seed in seeds:
        row = decision["encoder_adequacy_by_seed"][str(seed)]
        probe = results["position_probe"][str(seed)]["test_rmse_grid_cells"]
        variance = results["latent_variance"][str(seed)]["test"]
        lines.append(f"| {seed} | {row['full_gnn_h1_h_at_1']:.3f} | {row['full_gnn_h10_h_at_1']:.3f} | {row['persistence_h10_h_at_1']:.3f} | {row['h10_gain_over_persistence']:+.3f} | {'yes' if row['pass'] else 'no'} | {variance['std_mean']:.4g} | {variance['dimensions_below_1e-3']}/{variance['dimension_count']} | {probe:.3f} |")
    lines += ["", "Adequacy 的预设门槛为每 seed h1≥0.80、h10≥0.50 且 h10 比 persistence 至少高 0.05。test latent std 和 near-zero dimensions 是坍塌诊断；position probe 是 train-fitted/test-evaluated 的辅助表示诊断。dev 指标仅用于补充描述，未用于选择模型或 checkpoint。", "", "## 各 seed 的多步预测与速度", ""]
    for seed in seeds:
        lines += [f"### Seed {seed}", "", "| arm | H@1 h1 | H@1 h5 | H@1 h10 | H@1 h20 | latent MSE h10 | params | B1 median/p90 ms | B300 median/p90 ms | quality | B1 speed vs GNN/T | B300 speed vs GNN/T |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|"]
        for arm in arms:
            test = results["test"][str(seed)][arm]
            gate = decision["arm_gates_by_seed"][f"{seed}/{arm}"]
            params = results["stage_b"][f"{seed}/{arm}"]["trainable_params"]
            lat = results["latency"][f"{seed}/{arm}"]
            b1 = gate["speed"]["1"]
            b300 = gate["speed"]["300"]
            speed1 = f"{b1['vs_full_gnn_fractional_reduction'] * 100:+.1f}%/{b1['vs_transformer6_fractional_reduction'] * 100:+.1f}%"
            speed300 = f"{b300['vs_full_gnn_fractional_reduction'] * 100:+.1f}%/{b300['vs_transformer6_fractional_reduction'] * 100:+.1f}%"
            lines.append(f"| {arm} | {test['h_at_1'][0]:.3f} | {test['h_at_1'][1]:.3f} | {test['h_at_1'][2]:.3f} | {test['h_at_1'][3]:.3f} | {test['latent_mse'][primary]:.5f} | {params:,} | {lat['1']['predictor_h10_median_ms']:.3f}/{lat['1']['predictor_h10_p90_ms']:.3f} | {lat['300']['predictor_h10_median_ms']:.3f}/{lat['300']['predictor_h10_p90_ms']:.3f} | {'pass' if gate['quality_pass'] else 'fail'} | {speed1} | {speed300} |")
        lines += ["", "| arm | one-step MSE free | one-step MSE object-blocked | one-step MSE boundary-blocked | rollout H@1 h10 free | rollout H@1 h10 object-blocked | rollout H@1 h10 boundary-blocked | counterfactual response MSE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in arms:
            one = results["one_step_subsets"][f"{seed}/{arm}"]
            multi = results["rollout_subsets"][f"{seed}/{arm}"]
            response = results["action_response"][f"{seed}/{arm}"]
            response_n = sum(response[name]["episodes"] for name in response)
            response_mse = sum((response[name]["response_mse"] or 0.0) * response[name]["episodes"] for name in response) / max(1, response_n)
            lines.append(f"| {arm} | {one['free']['mean_latent_mse']:.5f} | {one['other-object-blocked']['mean_latent_mse']:.5f} | {one['boundary-blocked']['mean_latent_mse']:.5f} | {multi['free']['by_horizon']['10']['h_at_1']:.3f} | {multi['other-object-blocked']['by_horizon']['10']['h_at_1']:.3f} | {multi['boundary-blocked']['by_horizon']['10']['h_at_1']:.3f} | {response_mse:.5f} |")
        lines.append("")
    lines += [
        "Speed 列格式为“相对 full_gnn / 相对 transformer6 的延迟降低比例”；门槛按 60 repeats 的 median 计算，`+20%` 才达到预设速度门槛；p90 作为尾延迟一并显示。quality 表示 h10 H@1 不比 full GNN 低超过 3 个百分点。B=1、B=300 单独判定。encoder image-forward 的 median/p90 初始成本在 summary 中另列；predictor latency 包含完整 10 步 rollout 的 message/token 运算。", "",
        "## 通信、动作响应与数据子集", "",
        "`global4` vs `local4` 是主要通信对比：两者使用相同局部函数，新增投影参数均为 320，初始化配对。`flat_mlp_matched` 用于辨别结构收益是否只是参数减少。Counterfactual action-response 从相同图像/latent 对比存储动作与同一 object 的反向方向动作；MSE 越低表示预测的动作响应变化越接近 frozen encoder 编码的真实后继差分。", "",
        "逐 arm 的 `free`、`other-object-blocked`、`boundary-blocked` 单步 latent MSE、counterfactual response MSE 和响应幅度保存在 `summary.json`。Test 状态类别由位置仅在评估时计算，训练过程不读取这些字段。每 seed 的 episode-level 误差、H@1、MRR 与 counterfactual response 数据保存在 `episode_metrics_*.npz`。", "",
        "## 配对不确定性与运行环境", "",
        "H@1 h10 的 paired episode bootstrap 95% CI；除 `global4-local4` 行外均相对 full GNN：", "",
        "| seed/arm | H@1 差异 | paired 95% CI |", "|---|---:|---:|"]
    for key, value in paired.items():
        lo, hi = value["bootstrap_95ci"]
        lines.append(f"| {key} | {value['mean_h10_h_at_1_difference']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    runtime = results["runtime"]
    lines += ["", f"实际环境：Torch `{runtime['torch']}`、CUDA `{runtime['cuda_runtime']}`、GPU `{runtime['gpu']}`、precision `{runtime['precision']}`、matmul TF32 `{runtime['matmul_tf32']}`、cuDNN TF32 `{runtime['cudnn_tf32']}`。GPU 利用率和显存记录在 job log。", "", "## 适用边界", "", "三 seeds 分别报告，bootstrap 以 held-out episode 为单位；这些区间不代表训练 seed population。视觉 reference 未达到任一 seed 的预设 adequacy 时，全部 predictor 对比仍完整报告，但整体标为 inconclusive。该环境是离散 5×5 grid，不能代表连续 PushT 接触动力学。六层 Transformer 是共用 slots 接口的新架构对照，不是已训练的 LeWM checkpoint。结果不支持 CEM candidate ranking 或闭环控制结论。", "", "## 设计来源", "", "官方机制来源固定到 C-SWM commit；本实验做了明确记录的现代兼容改动，不是论文配置复现。冻结设计按 episode 分割、seed 内配对 minibatches，并以 episode 作为 bootstrap 单位。实验设计来源：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
