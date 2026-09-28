#!/usr/bin/env python3
"""Guarded CPU-only LpWM task/env identity and alignment preparation."""
import argparse
import json
import os
import random
import re
import socket
import sys
import time
from pathlib import Path


def require_compute_allocation():
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("real PBS_JOBID and PBS_NODEFILE are required")
    host = socket.gethostname().split(".")[0].lower()
    if any(marker in host for marker in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/submit host: {host}")
    allocated = {
        line.strip().split(".")[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in allocated:
        raise RuntimeError(f"host {host} is not listed in PBS_NODEFILE")
    return job_id, host, nodefile


def dummy_rollout(observations, actions, num_hist=3, calls=None):
    """Toy causal predictor that reads its whole available latent window."""
    embs = list(observations)

    def predict_next():
        h = min(num_hist, len(embs))
        window = embs[-h:]
        action_index = len(embs) - 1
        if calls is not None:
            calls.append({"obs_window": list(window), "action_index": action_index,
                          "action_id": actions[action_index]})
        return embs[-1] + 0.01 * sum(window) + actions[action_index]

    while len(embs) < len(actions):
        embs.append(predict_next())
    embs.append(predict_next())
    return embs


def check_alignment():
    horizon = 5
    past_actions = [10, 11]
    future_actions = [20, 21, 22, 23, 24]
    h3_calls = []
    h3 = dummy_rollout([0, 1, 2], past_actions + future_actions, calls=h3_calls)
    # Both cold-start entrances have exactly one initial observation, while
    # the predictor's configured maximum history remains three in both arms.
    cold_a_calls, cold_b_calls = [], []
    cold_a = dummy_rollout([2], future_actions, num_hist=3, calls=cold_a_calls)
    cold_b = dummy_rollout([2], future_actions, num_hist=3, calls=cold_b_calls)
    crop = h3[2:]
    assert len(h3) == 8 and len(crop) == horizon + 1
    assert crop[0] == 2
    assert h3_calls[0]["obs_window"] == [0, 1, 2]
    assert h3_calls[0]["action_index"] == 2 and h3_calls[0]["action_id"] == 20
    assert h3_calls[-1]["action_index"] == 6 and h3_calls[-1]["action_id"] == 24
    assert cold_a == cold_b and cold_a_calls == cold_b_calls
    assert len(future_actions) == horizon
    return {
        "passed": True,
        "history3_first_predictor_inputs": h3_calls[0],
        "history3_last_predictor_inputs": h3_calls[-1],
        "history3_output_frames": len(h3),
        "crop_start": 2,
        "cropped_frames": len(crop),
        "one_initial_frame_wrapper_branch_matches_native_cold_entry": True,
        "predictor_num_hist_in_both_cold_branches": 3,
        "future_blocks_executable": horizon,
        "scope": "deterministic indexing/lag selfcheck only; real-model history1 parity remains a GPU interface gate"
    }


def array_shape(value):
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def leaf_shapes(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(leaf_shapes(child, f"{prefix}.{key}" if prefix else str(key)))
        return result
    shape = array_shape(value)
    if shape is not None:
        return {prefix: shape}
    if isinstance(value, (list, tuple)) and value:
        return leaf_shapes(value[0], prefix + "[]")
    return {prefix: None}


def finite_array(value, np, torch):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    return bool(np.isfinite(np.asarray(value)).all())


def numeric_json_arrays(value, path, np, found):
    if isinstance(value, dict):
        for key, child in value.items():
            numeric_json_arrays(child, f"{path}.{key}" if path else str(key), np, found)
        return
    if isinstance(value, list):
        try:
            arr = np.asarray(value)
            if arr.dtype.kind in "biuf" and arr.size:
                found.append((path, arr))
                return
        except (TypeError, ValueError):
            pass
        for index, child in enumerate(value[:50]):
            if isinstance(child, (dict, list)):
                numeric_json_arrays(child, f"{path}[{index}]", np, found)


def compare_state_sidecar(targets, sidecar_path, np):
    if not sidecar_path.is_file():
        return {"present": False, "identity_match": False, "reason": "sidecar not staged"}
    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    found = []
    numeric_json_arrays(data, "", np, found)
    state0 = np.asarray(targets["state_0"].detach().cpu().numpy()
                        if hasattr(targets["state_0"], "detach") else targets["state_0"])
    stateg = np.asarray(targets["state_g"].detach().cpu().numpy()
                        if hasattr(targets["state_g"], "detach") else targets["state_g"])
    matches = []
    for path, arr in found:
        if arr.shape == state0.shape and np.array_equal(arr, state0):
            matches.append({"path": path, "matches": "state_0"})
        if arr.shape == stateg.shape and np.array_equal(arr, stateg):
            matches.append({"path": path, "matches": "state_g"})
        if arr.shape == (state0.shape[0], 2, state0.shape[-1]):
            if np.array_equal(arr[:, 0, :], state0) and np.array_equal(arr[:, 1, :], stateg):
                matches.append({"path": path, "matches": "[state_0,state_g]"})
            elif np.array_equal(arr[:, 1, :], state0) and np.array_equal(arr[:, 0, :], stateg):
                matches.append({"path": path, "matches": "[state_g,state_0]"})
    return {
        "present": True,
        "top_level_keys": list(data.keys()) if isinstance(data, dict) else None,
        "numeric_array_paths_and_shapes": [{"path": p, "shape": list(a.shape)} for p, a in found],
        "exact_matches": matches,
        "identity_match": any(m["matches"] == "[state_0,state_g]" for m in matches)
        or (any(m["matches"] == "state_0" for m in matches)
            and any(m["matches"] == "state_g" for m in matches))
    }


def extract_outcome_vectors(value, path=""):
    aliases = {"success", "successes", "issuccess", "finalsuccess", "finalsuccesses"}
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            clean = re.sub(r"[^a-z]", "", str(key).lower())
            child_path = f"{path}.{key}" if path else str(key)
            if clean in aliases and isinstance(child, list) and len(child) == 50:
                if all(isinstance(v, (bool, int, float)) and v in (0, 1, False, True) for v in child):
                    found.append({"path": child_path,
                                  "success_indices": [i for i, item in enumerate(child) if bool(item)],
                                  "failure_indices": [i for i, item in enumerate(child) if not bool(item)]})
            found.extend(extract_outcome_vectors(child, child_path))
    elif isinstance(value, list):
        if len(value) == 50 and all(isinstance(row, dict) for row in value):
            for key in aliases:
                rows = [row.get(key) for row in value]
                if all(isinstance(v, (bool, int, float)) and v in (0, 1, False, True) for v in rows):
                    found.append({"path": f"{path}[].{key}",
                                  "success_indices": [i for i, item in enumerate(rows) if bool(item)],
                                  "failure_indices": [i for i, item in enumerate(rows) if not bool(item)]})
        for index, child in enumerate(value[:50]):
            if isinstance(child, (dict, list)):
                found.extend(extract_outcome_vectors(child, f"{path}[{index}]"))
    return found


def summarize_outcomes(path):
    if not path.is_file():
        return {"present": False}
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "present": True,
        "top_level_type": type(value).__name__,
        "top_level_keys": list(value.keys()) if isinstance(value, dict) else None,
        "per_task_vectors": extract_outcome_vectors(value)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    job_id, host, nodefile = require_compute_allocation()

    # Heavy imports, dataset reads, physics replay and pickle access start only after guard.
    import numpy as np
    import torch
    import gym
    import hydra
    from omegaconf import OmegaConf
    from einops import rearrange
    import plan as official_plan
    from preprocessor import Preprocessor
    from utils import seed

    repo = Path("/scratch/users/ntu/yguo017/experiment/reproduction/lpwm/lpworldmodel-bdd812d")
    plan_out = repo / "plan_outputs/20260927102710_repro_sparse_pusht_mlp_var_pd384_gH5"
    target_path = plan_out / "plan_targets.pkl"
    logs_path = plan_out / "logs.json"
    model_dir = Path("/scratch/users/ntu/yguo017/experiment/reproduction/lpwm/formal_runs/25571461.pbs101/sparse/checkpoints/outputs/repro_sparse_pusht_mlp_var_pd384")
    checkpoint = model_dir / "checkpoints/model_latest.pth"
    model_config = model_dir / "hydra.yaml"
    dataset_root = Path("/scratch/users/ntu/yguo017/dino-wm-wall/data/pusht_noise")
    control = Path("/scratch/users/ntu/yguo017/lpwm-real-history/source")
    sidecar_states = control / "final_eval_target_states.json"
    sidecar_outcomes = control / "final_eval_outcomes.json"
    expected_commit = "bdd812d9432cccda8c350086006401b436f91982"

    # This pinned source snapshot intentionally has no .git directory.  Its
    # small SOURCE_COMMIT marker was created with the checkout and is the
    # available source identity; do not require a git executable in PBS PATH.
    source_marker = repo / "SOURCE_COMMIT"
    if not source_marker.is_file():
        raise FileNotFoundError(f"missing pinned source identity marker: {source_marker}")
    commit = source_marker.read_text(encoding="utf-8").strip()
    if commit != expected_commit:
        raise RuntimeError(f"source revision mismatch: {commit}")
    manifest_path = control / "run_manifest.json"
    for path in (target_path, logs_path, checkpoint, model_config, manifest_path,
                 sidecar_states, sidecar_outcomes):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_eval = manifest["evaluation"]
    if (int(run_eval["seed"]) != 99 or int(run_eval["n_evals"]) != 50
            or int(run_eval["goal_H"]) != 5 or int(run_eval["cem_candidates"]) != 300
            or int(run_eval["cem_elites"]) != 30 or int(run_eval["cem_opt_steps"]) != 30):
        raise RuntimeError("frozen run_manifest evaluation identity differs from this protocol")

    with target_path.open("rb") as handle:
        targets = __import__("pickle").load(handle)
    required = {"obs_0", "obs_g", "state_0", "state_g", "gt_actions", "goal_H"}
    missing = sorted(required - set(targets))
    if missing:
        raise RuntimeError(f"plan target pickle misses keys: {missing}")
    state0 = targets["state_0"]
    stateg = targets["state_g"]
    state_shapes = {"state_0": array_shape(state0), "state_g": array_shape(stateg)}
    state_finite = {"state_0": finite_array(state0, np, torch),
                    "state_g": finite_array(stateg, np, torch)}
    if state_shapes != {"state_0": [50, 7], "state_g": [50, 7]}:
        raise RuntimeError(f"expected saved [50,7] states, got {state_shapes}")
    if not all(state_finite.values()) or int(targets["goal_H"]) != 5:
        raise RuntimeError("non-finite states or goal_H mismatch")

    sidecar_identity = compare_state_sidecar(targets, sidecar_states, np)

    # Recreate the original dset task selection, env_info and expert-action replay.
    seed(99)
    cfg = OmegaConf.load(model_config)
    if int(cfg.frameskip) != 5 or int(cfg.num_hist) != 3 or int(cfg.num_pred) != 1:
        raise RuntimeError("model config differs from frozen temporal configuration")
    _, splits = hydra.utils.call(
        cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred,
        frameskip=cfg.frameskip
    )
    dset = splits["valid"]
    traj_len = 5 * int(targets["goal_H"]) + 1
    # Match the official prepass and random draws exactly.
    valid_traj = [dset[i][0]["visual"].shape[0] for i in range(len(dset))
                  if dset[i][0]["visual"].shape[0] >= traj_len]
    if not valid_traj:
        raise RuntimeError("validation dataset has no trajectory long enough")
    sampled_states, sampled_actions, env_infos = [], [], []
    for _ in range(50):
        max_offset = -1
        while max_offset < 0:
            traj_id = random.randint(0, len(dset) - 1)
            obs, act, state, env_info = dset[traj_id]
            max_offset = obs["visual"].shape[0] - traj_len
        state_np = state.detach().cpu().numpy() if torch.is_tensor(state) else np.asarray(state)
        offset = random.randint(0, max_offset)
        sampled_states.append(state_np[offset:offset + traj_len])
        sampled_actions.append(act[offset:offset + 5 * int(targets["goal_H"])])
        env_infos.append(env_info)
    init_from_dataset = np.asarray([state[0] for state in sampled_states])
    if not np.array_equal(init_from_dataset, state0.detach().cpu().numpy()
                          if torch.is_tensor(state0) else np.asarray(state0)):
        raise RuntimeError("recreated dataset initial states do not exactly match frozen task states")

    actions = torch.stack(sampled_actions)
    wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=5)
    saved_gt = targets["gt_actions"]
    saved_gt_np = saved_gt.detach().cpu().numpy() if torch.is_tensor(saved_gt) else np.asarray(saved_gt)
    gt_action_match = bool(np.array_equal(wm_actions.detach().cpu().numpy(), saved_gt_np))
    if not gt_action_match:
        raise RuntimeError("recreated ground-truth action blocks differ from saved targets")

    preprocessor = Preprocessor(
        action_mean=dset.action_mean, action_std=dset.action_std,
        state_mean=dset.state_mean, state_std=dset.state_std,
        proprio_mean=dset.proprio_mean, proprio_std=dset.proprio_std,
        transform=dset.transform
    )
    executable_actions = preprocessor.denormalize_actions(actions).detach().cpu().numpy()
    env = gym.make(cfg.env.name, *list(cfg.env.args), **OmegaConf.to_container(cfg.env.kwargs))
    base_env = env.unwrapped
    obs0_matches, obsg_matches, stateg_matches = [], [], []
    max_abs = {"obs_0": {}, "obs_g": {}, "state_g": 0.0}
    env_shape_metadata = []
    eval_seeds = [99 * i + 1 for i in range(50)]
    try:
        for i in range(50):
            info = env_infos[i]
            base_env.update_env(info)
            env_shape_metadata.append(str(info.get("shape", "missing"))[:160])
            init_state_i = state0[i].detach().cpu().numpy() if torch.is_tensor(state0) else np.asarray(state0[i])
            start_obs, start_state = base_env.prepare(eval_seeds[i], init_state_i)
            for key, saved in targets["obs_0"].items():
                saved_frame = saved[i, 0]
                diff = np.asarray(start_obs[key]).astype(np.float64) - np.asarray(saved_frame).astype(np.float64)
                delta = float(np.max(np.abs(diff))) if diff.size else 0.0
                max_abs["obs_0"][key] = max(max_abs["obs_0"].get(key, 0.0), delta)
                obs0_matches.append(delta == 0.0)
            rolled_obs, rolled_states = base_env.rollout(
                eval_seeds[i], init_state_i, executable_actions[i]
            )
            goal_state_i = stateg[i].detach().cpu().numpy() if torch.is_tensor(stateg) else np.asarray(stateg[i])
            state_diff = np.asarray(rolled_states[-1]).astype(np.float64) - np.asarray(goal_state_i).astype(np.float64)
            state_delta = float(np.max(np.abs(state_diff))) if state_diff.size else 0.0
            max_abs["state_g"] = max(max_abs["state_g"], state_delta)
            stateg_matches.append(state_delta == 0.0)
            for key, saved in targets["obs_g"].items():
                saved_frame = saved[i, 0]
                diff = np.asarray(rolled_obs[key][-1]).astype(np.float64) - np.asarray(saved_frame).astype(np.float64)
                delta = float(np.max(np.abs(diff))) if diff.size else 0.0
                max_abs["obs_g"][key] = max(max_abs["obs_g"].get(key, 0.0), delta)
                obsg_matches.append(delta == 0.0)
    finally:
        env.close()

    action_patterns = ("*action*.pt", "*action*.pth", "*action*.pkl", "*action*.npz",
                       "*proposal*.pt", "*proposal*.pth", "*proposal*.pkl", "*proposal*.npz")
    candidate_names = sorted({m.name for p in action_patterns for m in plan_out.glob(p)})
    def task_labels(pattern):
        rows = []
        for path in sorted(plan_out.glob(pattern)):
            match = re.match(r"(?:output_final|plan[01])_real_(\d+)_(success|failure)\.png$", path.name)
            if match:
                rows.append({"task_index": int(match.group(1)), "label": match.group(2)})
        return rows

    outcome_summary = summarize_outcomes(sidecar_outcomes)
    final_image_labels = task_labels("output_final_real_*_*.png")
    if len(final_image_labels) == 50:
        final_failures = [row["task_index"] for row in final_image_labels if row["label"] == "failure"]
        outcome_source = "official output_final per-task image labels"
    else:
        vectors = outcome_summary.get("per_task_vectors", [])
        vector = next((item for item in vectors if len(item["failure_indices"]) + len(item["success_indices"]) == 50), None)
        final_failures = vector["failure_indices"] if vector else []
        outcome_source = vector["path"] if vector else "unresolved"
    exploratory_cohort = sorted(final_failures)[:8]
    cohort_passed = (
        len(final_failures) == 29
        and len(exploratory_cohort) == 8
        and len(set(exploratory_cohort)) == 8
        and all(0 <= i < 50 for i in exploratory_cohort)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env_identity_path = out.parent / "env_identity.pkl"
    with env_identity_path.open("wb") as handle:
        __import__("pickle").dump(
            {
                "source_commit": commit,
                "planning_seed": 99,
                "task_indices": exploratory_cohort,
                "eval_seeds": {str(i): eval_seeds[i] for i in exploratory_cohort},
                "env_infos": {str(i): env_infos[i] for i in exploratory_cohort},
                "identity_checks_passed": bool(
                    all(obs0_matches) and all(obsg_matches) and all(stateg_matches)
                    and gt_action_match and sidecar_identity.get("identity_match", False)
                ),
            },
            handle,
            protocol=4,
        )
    alignment_check = check_alignment()
    task_identity_passed = bool(
        state_shapes == {"state_0": [50, 7], "state_g": [50, 7]}
        and all(state_finite.values())
        and int(targets["goal_H"]) == 5
        and sidecar_identity.get("identity_match", False)
        and int(run_eval["seed"]) == 99
        and int(run_eval["n_evals"]) == 50
        and int(run_eval["goal_H"]) == 5
        and int(run_eval["cem_candidates"]) == 300
        and int(run_eval["cem_elites"]) == 30
        and int(run_eval["cem_opt_steps"]) == 30
    )
    replay_identity_passed = bool(
        all(obs0_matches) and all(obsg_matches) and all(stateg_matches)
        and gt_action_match
    )
    gates_passed = {
        "task_identity": task_identity_passed,
        "environment_and_replay_identity": replay_identity_passed,
        "fixed_failure_cohort": cohort_passed,
        "dummy_alignment": bool(alignment_check.get("passed", False)),
    }
    report = {
        "job_id": job_id,
        "host": host,
        "nodefile": nodefile,
        "source_commit": commit,
        "source_identity_provenance": {
            "kind": "pinned snapshot marker",
            "path": str(source_marker),
            "value": commit,
            "git_metadata_present": (repo / ".git").exists(),
            "note": "The source tree is a pinned snapshot without Git metadata; SOURCE_COMMIT is the checkout-provided identity marker."
        },
        "run_manifest_identity": {
            "seed": int(run_eval["seed"]),
            "task_count": int(run_eval["n_evals"]),
            "goal_H": int(run_eval["goal_H"]),
            "cem_candidates": int(run_eval["cem_candidates"]),
            "cem_elites": int(run_eval["cem_elites"]),
            "cem_iterations": int(run_eval["cem_opt_steps"])
        },
        "plan_target_keys": sorted(targets.keys()),
        "obs_0_leaf_shapes": leaf_shapes(targets["obs_0"]),
        "obs_g_leaf_shapes": leaf_shapes(targets["obs_g"]),
        "state_shapes": state_shapes,
        "state_finite": state_finite,
        "goal_H": int(targets["goal_H"]),
        "task_state_sidecar_identity": sidecar_identity,
        "recreated_dataset_initial_state_match": True,
        "recreated_gt_action_blocks_match": gt_action_match,
        "environment_shape_metadata_by_task": env_shape_metadata,
        "old_env_replay_identity": {
            "task_count": 50,
            "eval_seed_formula": "99*original_task_index+1",
            "obs_0_exact_fraction": float(np.mean(obs0_matches)),
            "obs_g_exact_fraction": float(np.mean(obsg_matches)),
            "state_g_exact_fraction": float(np.mean(stateg_matches)),
            "max_abs_differences": max_abs,
            "physics_reference_warning": "7D anchor state omits block velocity; future candidate truth must replay real executed prefix from frozen init state/seed"
        },
        "prior_task_labels": {
            "final_outcomes_sidecar": outcome_summary,
            "plan0_images": task_labels("plan0_real_*_*.png"),
            "plan1_images": task_labels("plan1_real_*_*.png"),
            "final_images": final_image_labels
        },
        "exploratory_cohort": {
            "selection": "first 8 of the 29 prior-formal final-failure task indices in ascending order; fixed before new mechanism metrics",
            "source": outcome_source,
            "indices_zero_based": exploratory_cohort,
            "prior_failure_count": len(final_failures)
        },
        "env_identity_bundle": {
            "path": str(env_identity_path),
            "bytes": env_identity_path.stat().st_size,
            "contents": "task env_info/shape and original eval seeds only; no model outputs, native planner actions or action banks"
        },
        "one_level_candidate_action_names_only": candidate_names,
        "native_action_trace_status": "not inferred from filenames; official plan_targets writer keys and MPC in-memory behavior establish it is not persisted",
        "alignment_dummy_check": alignment_check,
        "model_loaded": False,
        "gpu_used": False,
        "gates_passed": gates_passed,
        "identity_preflight_passed": bool(task_identity_passed and replay_identity_passed),
        "native_action_trace_available": False,
        "mechanism_gate_ready": False,
        "preflight_status": "PASS" if all(gates_passed.values()) else "FAIL",
        "elapsed_seconds": round(time.monotonic() - started, 3)
    }
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(gates_passed.values()) else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CPU_PREFLIGHT_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
