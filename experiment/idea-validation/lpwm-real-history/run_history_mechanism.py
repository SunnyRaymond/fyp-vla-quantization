#!/usr/bin/env python3
"""Guarded LpWM real-history mechanism capture and fixed-bank test.

This entry point is intentionally tied to the pinned PushT sparse run. It must
only execute inside its real PBS allocation; it never trains or changes module
mode, CEM RNG calls, or the fixed 300-candidate bank.
"""
import argparse
import json
import os
import pickle
import random
import re
import socket
import sys
import time
import traceback
from pathlib import Path


EXPECTED_COMMIT = "bdd812d9432cccda8c350086006401b436f91982"
EXPECTED_COHORT = [0, 2, 3, 4, 5, 8, 9, 12]
PLAN_OUTPUT = "plan_outputs/20260927102710_repro_sparse_pusht_mlp_var_pd384_gH5"
N_CANDIDATES = 300
N_ELITES = 30
N_ITERATIONS = 30
HORIZON = 5
FRAMESKIP = 5
N_TAKEN_BLOCKS = 5
PACKED_ACTION_DIM = 10  # pinned PushT: 2-D action repeated over frameskip=5
PRIMARY_FLOOR = 0.01
TRUE_DELTA_EPS = 1e-8
RESOURCE_CAP_SECONDS = 80 * 60


class MechanismGateError(RuntimeError):
    """A required numeric/interface gate failed; do not classify it as NO_GO."""


def require_compute_allocation():
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("real PBS_JOBID and PBS_NODEFILE are required")
    host = socket.gethostname().split(".")[0].lower()
    if any(marker in host for marker in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/submit host: {host}")
    nodes = {
        line.strip().split(".")[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in nodes:
        raise RuntimeError(f"host {host} is not listed in PBS_NODEFILE")
    return job_id, host, nodefile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--cpu-preflight", required=True)
    parser.add_argument("--env-identity", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def as_numpy(value, np):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def numpy_tree(value, np):
    if isinstance(value, dict):
        return {key: numpy_tree(child, np) for key, child in value.items()}
    return as_numpy(value, np)


def subset_tree(value, ids, np, torch):
    if isinstance(value, dict):
        return {key: subset_tree(child, ids, np, torch) for key, child in value.items()}
    if torch.is_tensor(value) and value.ndim > 0:
        index = torch.as_tensor(ids, dtype=torch.long, device=value.device)
        return value.index_select(0, index).clone()
    if torch.is_tensor(value):
        return value.clone()
    arr = np.asarray(value)
    return arr[ids].copy() if arr.ndim else value


def clone_first(value):
    return {key: child[:1].detach().clone() for key, child in value.items()}


def repeat_first(value, count, repeat):
    return {key: repeat(child, "1 ... -> n ...", n=count) for key, child in value.items()}


def named_buffer_snapshot(model):
    return {name: buf.detach().clone() for name, buf in model.named_buffers()}


def restore_buffers(model, snapshot):
    current = dict(model.named_buffers())
    if set(current) != set(snapshot):
        raise RuntimeError("model buffer names changed during paired evaluation")
    for name, value in snapshot.items():
        current[name].copy_(value)


def torch_rng_snapshot(torch):
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def restore_torch_rng(torch, state):
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def training_flags(model):
    result = {}
    for name in ("encoder", "projector", "link", "predictor", "action_encoder"):
        module = getattr(model, name, None)
        if module is None:
            result[name] = None
            continue
        result[name] = {
            "root_training": bool(module.training),
            "submodules": {path: bool(child.training)
                           for path, child in module.named_modules()},
        }
    return result


def close_environments(*envs):
    for env in envs:
        if env is None:
            continue
        try:
            env.close()
        except Exception:
            pass


class Capture:
    """Observe existing native CEM calls; do not replace its sampling loop."""
    def __init__(self, model, task_ids, frameskip, topk, samples, np, torch):
        self.model = model
        self.task_ids = list(task_ids)
        self.frameskip = frameskip
        self.topk = topk
        self.samples = samples
        self.np = np
        self.torch = torch
        self.last_exec_trace = None
        self.contexts = {}
        self.current_by_slot = {}
        self.parent = None
        self.anchor = -1
        self.iteration = 0
        self.slot = 0
        self.active = False
        self.last_forward = None

    def record_plan_eval(self, filename, actions, e_obses, e_states):
        match = re.fullmatch(r"plan(\d+)", str(filename))
        if not match:
            return
        next_anchor = int(match.group(1)) + 1
        if next_anchor not in (1, 2):
            self.last_exec_trace = None
            return
        actions_np = as_numpy(actions, self.np).copy()
        obs_np = numpy_tree(e_obses, self.np)
        states_np = as_numpy(e_states, self.np).copy()
        raw_end = next_anchor * N_TAKEN_BLOCKS * self.frameskip
        if actions_np.shape[1] != next_anchor * N_TAKEN_BLOCKS:
            raise RuntimeError(f"native executed prefix length mismatch at anchor {next_anchor}: {actions_np.shape}")
        if states_np.shape[1] <= raw_end:
            raise RuntimeError(f"native environment trace too short for anchor {next_anchor}: {states_np.shape}")
        history_indices = [raw_end - 2 * self.frameskip, raw_end - self.frameskip, raw_end]
        if history_indices[0] < 0:
            raise RuntimeError("history index precedes task initialization")
        compact_obs = {}
        for key, arr in obs_np.items():
            if arr.shape[1] <= raw_end:
                raise RuntimeError(f"native observation trace too short for {key} at anchor {next_anchor}")
            compact_obs[key] = arr[:, history_indices].copy()
        self.last_exec_trace = {
            "filename": str(filename),
            "next_anchor": next_anchor,
            "raw_end": raw_end,
            "history_indices": history_indices,
            "actions": actions_np,
            "observations": compact_obs,
            "states_at_anchor": states_np[:, raw_end].copy(),
        }

    def begin_cem(self, parent, obs_0):
        self.parent = parent
        self.anchor = int(parent.iter)
        self.iteration = 0
        self.slot = 0
        self.last_forward = None
        self.current_by_slot = {}
        self.active = self.anchor in (1, 2)
        if not self.active:
            return
        trace = self.last_exec_trace
        expected_filename = f"plan{self.anchor - 1}"
        if trace is None or trace["filename"] != expected_filename or trace["next_anchor"] != self.anchor:
            raise RuntimeError(f"missing executed-prefix capture for {expected_filename}")
        if len(self.task_ids) != len(parent.is_success):
            raise RuntimeError("task identity count differs from native MPC batch")
        obs_np = numpy_tree(obs_0, self.np)
        for slot, task_id in enumerate(self.task_ids):
            if bool(parent.is_success[slot]):
                continue
            for key, arr in obs_np.items():
                current = arr[slot, 0]
                traced = trace["observations"][key][slot, -1]
                if not self.np.array_equal(current, traced):
                    raise RuntimeError(f"native current observation disagrees with trace: task={task_id}, key={key}")
            context = {
                "task_index": int(task_id),
                "task_slot": int(slot),
                "anchor": int(self.anchor),
                "eval_seed": int(99 * int(task_id) + 1),
                "raw_end": int(trace["raw_end"]),
                "history_indices": list(trace["history_indices"]),
                "history_obs": {key: arr[slot].copy() for key, arr in trace["observations"].items()},
                "current_obs": {key: arr[slot, -1].copy() for key, arr in trace["observations"].items()},
                "current_state": trace["states_at_anchor"][slot].copy(),
                "prefix_actions": trace["actions"][slot].copy(),
                "past_action_blocks": trace["actions"][slot, -2:].copy(),
                "early": None,
                "late": None,
            }
            self.contexts[(int(task_id), int(self.anchor))] = context
            self.current_by_slot[slot] = context

    def before_rollout(self, obs_0, actions):
        self.last_forward = None
        if not self.active or int(actions.shape[0]) != self.samples:
            return
        if tuple(actions.shape[1:]) != (HORIZON, PACKED_ACTION_DIM):
            raise MechanismGateError(
                f"native CEM action shape mismatch: {tuple(actions.shape)}; expected (300,5,{PACKED_ACTION_DIM})")
        context = self.current_by_slot.get(self.slot)
        if context is None:
            return
        self.last_forward = {
            "actions": actions.detach(),
            "obs_single": clone_first(obs_0),
            "torch_rng": torch_rng_snapshot(self.torch),
            "buffers": named_buffer_snapshot(self.model),
        }

    def scored(self, loss, goal):
        if self.active:
            context = self.current_by_slot.get(self.slot)
            if context is not None:
                if self.last_forward is None:
                    raise RuntimeError("CEM objective has no paired candidate rollout capture")
                losses = loss.detach().reshape(-1)
                if int(losses.numel()) != self.samples:
                    raise RuntimeError(f"unexpected CEM objective shape: {tuple(loss.shape)}")
                elite_indices = self.torch.argsort(losses)[:self.topk]
                generation = {
                    "iteration": int(self.iteration),
                    "actions": self.last_forward["actions"].detach().cpu().numpy().copy(),
                    "losses": losses.detach().cpu().numpy().copy(),
                    "elite_indices": elite_indices.detach().cpu().numpy().astype("int64", copy=True),
                    "obs_single": {key: value.detach().clone() for key, value in self.last_forward["obs_single"].items()},
                    "goal_single": clone_first(goal),
                    "torch_rng": self.last_forward["torch_rng"],
                    "buffers": self.last_forward["buffers"],
                }
                if self.iteration == 0:
                    context["early"] = generation
                else:
                    # Retain the last actually evaluated population. Native all-success
                    # early stopping is preserved, so this may precede configured i=29.
                    context["late"] = generation
            self.slot += 1
            if self.slot >= len(self.task_ids):
                self.slot = 0
                self.iteration += 1
        self.last_forward = None


def make_candidate_bank(context, np):
    early = context.get("early")
    late = context.get("late")
    if early is None or late is None:
        return None
    early_actions = np.asarray(early["actions"])
    late_actions = np.asarray(late["actions"])
    if early_actions.shape != (N_CANDIDATES, HORIZON, PACKED_ACTION_DIM):
        raise RuntimeError(f"unexpected early candidate shape: {early_actions.shape}")
    if late_actions.shape != (N_CANDIDATES, HORIZON, PACKED_ACTION_DIM):
        raise RuntimeError(f"unexpected late candidate shape: {late_actions.shape}")
    early_ids = np.linspace(0, N_CANDIDATES - 1, 100, dtype=np.int64)
    elite_ids = np.asarray(late["elite_indices"], dtype=np.int64)
    elite_set = set(int(i) for i in elite_ids.tolist())
    non_elite_ids = np.asarray([i for i in range(N_CANDIDATES) if i not in elite_set], dtype=np.int64)
    if len(elite_ids) != N_ELITES or len(non_elite_ids) != 270:
        raise RuntimeError("late CEM elite/non-elite partition differs from 30/270")
    late_non_elite_ids = non_elite_ids[np.linspace(0, len(non_elite_ids) - 1, 170, dtype=np.int64)]
    bank = np.concatenate((early_actions[early_ids], late_actions[late_non_elite_ids], late_actions[elite_ids]), axis=0)
    sources = ([{"population": "early", "iteration": int(early["iteration"]), "candidate_index": int(i)}
                for i in early_ids]
               + [{"population": "late_non_elite", "iteration": int(late["iteration"]), "candidate_index": int(i)}
                  for i in late_non_elite_ids]
               + [{"population": "late_elite", "iteration": int(late["iteration"]), "candidate_index": int(i)}
                  for i in elite_ids])
    if bank.shape != (N_CANDIDATES, HORIZON, PACKED_ACTION_DIM) or len(sources) != N_CANDIDATES:
        raise RuntimeError(f"candidate bank composition error: {bank.shape}")
    if not np.isfinite(bank).all():
        raise MechanismGateError("fixed candidate bank contains non-finite actions")
    flat = bank.reshape(300, -1)
    _, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    groups = []
    for group in range(len(counts)):
        ids = np.flatnonzero(inverse == group).tolist()
        if len(ids) > 1:
            groups.append(ids)
    return {
        "actions": bank,
        "sources": sources,
        "duplicate_groups": groups,
        "early_indices": early_ids,
        "late_non_elite_indices": late_non_elite_ids,
        "late_elite_indices": elite_ids,
    }


def synthetic_bank_shape_selfcheck(np):
    actual_shape = (300, 5, 10)
    values = np.arange(np.prod(actual_shape), dtype=np.float32)
    actions = values.reshape(actual_shape)
    context = {
        "early": {"actions": actions, "iteration": 0},
        "late": {"actions": actions + values.size, "iteration": N_ITERATIONS - 1,
                 "elite_indices": np.arange(N_CANDIDATES - N_ELITES, N_CANDIDATES)},
    }
    bank = make_candidate_bank(context, np)
    if bank is None or bank["actions"].shape != (300, 5, 10):
        raise MechanismGateError("synthetic 300x5x10 candidate-bank selfcheck failed")
    return {"passed": True, "synthetic_source_shape": [300, 5, 10],
            "composed_bank_shape": list(bank["actions"].shape),
            "bank_size": int(len(bank["sources"]))}


def predict_arm(model, preprocessor, objective_fn, context, actions_np, history, device,
                repeat, move_to_device, torch, np, obs_override=None, goal_override=None):
    count = int(actions_np.shape[0])
    if obs_override is None:
        if history:
            raw = {key: torch.as_tensor(value[None], device="cpu")
                   for key, value in context["history_obs"].items()}
        else:
            raw = {key: torch.as_tensor(value[None, None], device="cpu")
                   for key, value in context["current_obs"].items()}
        obs_single = move_to_device(preprocessor.transform_obs(raw), device)
    else:
        obs_single = obs_override
    obs_batch = repeat_first(obs_single, count, repeat)
    action_batch = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
    if history:
        past = torch.as_tensor(context["past_action_blocks"], dtype=torch.float32, device=device)
        past = past.unsqueeze(0).expand(count, -1, -1)
        action_batch = torch.cat((past, action_batch), dim=1)
    with torch.no_grad():
        prediction, _ = model.rollout(obs_0=obs_batch, act=action_batch)
        if history:
            prediction = {key: value[:, 2:] for key, value in prediction.items()}
        goal_single = goal_override if goal_override is not None else context["late"]["goal_single"]
        goal_batch = repeat_first(goal_single, count, repeat)
        objective = objective_fn(prediction, goal_batch).reshape(-1)
        visual = prediction["visual"]
        if visual.ndim == 4:
            summary = visual.mean(dim=2)
        elif visual.ndim == 3:
            summary = visual
        else:
            raise RuntimeError(f"unexpected linked visual shape: {tuple(visual.shape)}")
    objective_np = objective.detach().cpu().numpy()
    summary_np = summary.detach().cpu().numpy()
    if objective_np.shape != (count,) or summary_np.shape[:2] != (count, HORIZON + 1):
        raise MechanismGateError(
            f"predictor output shape mismatch: objective={objective_np.shape}, latent={summary_np.shape}")
    if not np.isfinite(objective_np).all() or not np.isfinite(summary_np).all():
        raise MechanismGateError("predictor objective/latent contains non-finite values")
    return objective_np, summary_np


def paired_model_scores(model, preprocessor, objective_fn, context, bank, device,
                        repeat, move_to_device, torch, np):
    # Score both arms from the same model buffer and RNG state. No module mode is changed.
    live_buffers = named_buffer_snapshot(model)
    live_rng = torch_rng_snapshot(torch)
    paired_buffers = context["late"]["buffers"]
    paired_rng = context["late"]["torch_rng"]
    try:
        restore_buffers(model, paired_buffers)
        restore_torch_rng(torch, paired_rng)
        cold_loss, cold_latents = predict_arm(
            model, preprocessor, objective_fn, context, bank, False, device,
            repeat, move_to_device, torch, np,
            obs_override=context["late"]["obs_single"],
            goal_override=context["late"]["goal_single"])
        restore_buffers(model, paired_buffers)
        restore_torch_rng(torch, paired_rng)
        hist_loss, hist_latents = predict_arm(
            model, preprocessor, objective_fn, context, bank, True, device,
            repeat, move_to_device, torch, np,
            goal_override=context["late"]["goal_single"])
    finally:
        restore_buffers(model, live_buffers)
        restore_torch_rng(torch, live_rng)
    return cold_loss, hist_loss, cold_latents, hist_latents


def native_history1_parity(model, preprocessor, objective_fn, context, device,
                           repeat, move_to_device, torch, np):
    late = context["late"]
    live_buffers = named_buffer_snapshot(model)
    live_rng = torch_rng_snapshot(torch)
    try:
        restore_buffers(model, late["buffers"])
        restore_torch_rng(torch, late["torch_rng"])
        direct_loss, _ = predict_arm(
            model, preprocessor, objective_fn, context, late["actions"], False,
            device, repeat, move_to_device, torch, np,
            obs_override=late["obs_single"], goal_override=late["goal_single"])
        max_abs = float(np.max(np.abs(direct_loss - late["losses"])))
        passed = bool(np.allclose(direct_loss, late["losses"], rtol=1e-5, atol=1e-6))
    finally:
        restore_buffers(model, live_buffers)
        restore_torch_rng(torch, live_rng)
    return {"passed": passed, "max_abs_objective_difference": max_abs,
            "batch_size": N_CANDIDATES, "predictor_num_hist": 3,
            "native_candidate_source_iteration": int(late["iteration"])}


def cost_only_predictor_forwards(model, preprocessor, context, bank, device,
                                 repeat, move_to_device, torch, np, raw_truth):
    live_buffers = named_buffer_snapshot(model)
    live_rng = torch_rng_snapshot(torch)
    paired_buffers = context["late"]["buffers"]
    paired_rng = context["late"]["torch_rng"]
    try:
        restore_buffers(model, paired_buffers)
        restore_torch_rng(torch, paired_rng)
        start = time.monotonic()
        raw_cold = {key: torch.as_tensor(value[None, None], device="cpu")
                    for key, value in context["current_obs"].items()}
        cold_single = move_to_device(preprocessor.transform_obs(raw_cold), device)
        cold_input = repeat_first(cold_single, 300, repeat)
        cold_act = torch.as_tensor(bank, dtype=torch.float32, device=device)
        with torch.no_grad():
            model.rollout(obs_0=cold_input, act=cold_act)
        torch.cuda.synchronize(device)
        cold_secs = time.monotonic() - start
        restore_buffers(model, paired_buffers)
        restore_torch_rng(torch, paired_rng)
        start = time.monotonic()
        raw = {key: torch.as_tensor(value[None], device="cpu")
               for key, value in context["history_obs"].items()}
        hist_single = move_to_device(preprocessor.transform_obs(raw), device)
        hist_input = repeat_first(hist_single, 300, repeat)
        past = torch.as_tensor(context["past_action_blocks"], dtype=torch.float32, device=device)
        hist_act = torch.cat((past.unsqueeze(0).expand(300, -1, -1), cold_act), dim=1)
        with torch.no_grad():
            model.rollout(obs_0=hist_input, act=hist_act)
        torch.cuda.synchronize(device)
        hist_secs = time.monotonic() - start
        restore_buffers(model, paired_buffers)
        restore_torch_rng(torch, paired_rng)
        start = time.monotonic()
        true_future_summary(model, preprocessor, raw_truth, device, move_to_device, torch, np)
        torch.cuda.synchronize(device)
        truth_encoder_secs = time.monotonic() - start
    finally:
        restore_buffers(model, live_buffers)
        restore_torch_rng(torch, live_rng)
    return {"cold_history1_forward_seconds": cold_secs,
            "real_history3_forward_seconds": hist_secs,
            "true_future_encoder_seconds": truth_encoder_secs}


def make_physics_env(model_cfg, gym, OmegaConf):
    kwargs = OmegaConf.to_container(model_cfg.env.kwargs, resolve=True)
    args = list(OmegaConf.to_container(model_cfg.env.args, resolve=True))
    return gym.make(model_cfg.env.name, *args, **kwargs)


def normalized_blocks_to_primitives(blocks, preprocessor, torch, rearrange, frameskip):
    tensor = torch.as_tensor(blocks, dtype=torch.float32).unsqueeze(0)
    raw_blocks = rearrange(tensor.cpu(), "b t (f d) -> b (t f) d", f=frameskip)
    return preprocessor.denormalize_actions(raw_blocks).detach().cpu().numpy()[0]


def collect_raw_truth(env, env_info, context, bank, targets, original_slot,
                      preprocessor, torch, np, rearrange, frameskip):
    env.unwrapped.update_env(env_info)
    init_state = as_numpy(targets["state_0"][original_slot], np).copy()
    prefix = np.asarray(context["prefix_actions"], dtype=np.float32)
    prefix_primitives = prefix.shape[0] * frameskip
    anchor_obs_index = prefix_primitives
    # Verify the full hidden-state-preserving executed prefix before any candidate scoring.
    prefix_raw = normalized_blocks_to_primitives(prefix, preprocessor, torch, rearrange, frameskip)
    prefix_obs, prefix_states = env.unwrapped.rollout(context["eval_seed"], init_state, prefix_raw)
    prefix_state = as_numpy(prefix_states[-1], np)
    if not np.array_equal(prefix_state, np.asarray(context["current_state"])):
        delta = float(np.max(np.abs(prefix_state.astype(np.float64) - context["current_state"].astype(np.float64))))
        raise RuntimeError(f"prefix replay state mismatch task={context['task_index']} anchor={context['anchor']} max_abs={delta}")
    for key, saved_frames in context["history_obs"].items():
        if key not in prefix_obs:
            raise RuntimeError(f"prefix replay missing observation key {key}")
        if not np.array_equal(as_numpy(prefix_obs[key][anchor_obs_index], np), saved_frames[-1]):
            raise RuntimeError(f"prefix replay current observation mismatch task={context['task_index']} key={key}")

    endpoints = []
    future_visuals = []
    frame_indices = prefix_primitives + np.arange(HORIZON + 1) * frameskip
    for candidate in range(N_CANDIDATES):
        all_blocks = np.concatenate((prefix, bank[candidate]), axis=0)
        executable = normalized_blocks_to_primitives(all_blocks, preprocessor, torch, rearrange, frameskip)
        obs, states = env.unwrapped.rollout(context["eval_seed"], init_state, executable)
        end_state = as_numpy(states[-1], np).copy()
        if not np.isfinite(end_state).all():
            raise MechanismGateError(f"non-finite replay endpoint task={context['task_index']} candidate={candidate}")
        endpoints.append(end_state)
        if "visual" not in obs:
            raise RuntimeError("PushT replay did not return visual observations")
        frames = as_numpy(obs["visual"], np)[frame_indices].copy()
        if not np.isfinite(frames).all():
            raise MechanismGateError(f"non-finite replay visual task={context['task_index']} candidate={candidate}")
        future_visuals.append(frames)
    return {
        "endpoint_states": np.asarray(endpoints),
        "future_visuals": np.asarray(future_visuals),
        "initial_state": init_state,
    }


def derive_truth_metrics(raw_truth, goal_state, np):
    endpoints = raw_truth["endpoint_states"]
    goal_state = np.asarray(goal_state)
    if endpoints.shape != (N_CANDIDATES, 7) or goal_state.shape != (7,):
        raise MechanismGateError(
            f"physical truth state shape mismatch: endpoints={endpoints.shape}, goal={goal_state.shape}")
    if not np.isfinite(goal_state).all():
        raise MechanismGateError("goal state contains non-finite values")
    state_diff = goal_state[None, :4] - endpoints[:, :4]
    positions = np.linalg.norm(state_diff, axis=1)
    angles = (goal_state[4] - endpoints[:, 4] + np.pi) % (2 * np.pi) - np.pi
    costs = (positions / 20.0) ** 2 + (angles / (np.pi / 9.0)) ** 2
    if not all(np.isfinite(array).all() for array in (positions, angles, costs)):
        raise MechanismGateError("non-finite endpoint metric after resource gate")
    return {
        **raw_truth,
        "position_errors": positions,
        "wrapped_angle_errors": angles,
        "diagnostic_costs": costs,
        "official_success": (positions < 20) & (np.abs(angles) < np.pi / 9),
        "goal_state": goal_state.copy(),
        "object_only_position_errors": np.linalg.norm(
            goal_state[2:4][None, :] - endpoints[:, 2:4], axis=1),
    }


def true_future_summary(model, preprocessor, truth, device, move_to_device, torch, np):
    raw_visual = torch.as_tensor(truth["future_visuals"], device="cpu")
    trans = move_to_device(
        {"visual": preprocessor.transform_obs_visual(raw_visual)}, device)
    with torch.no_grad():
        linked = model.encode_obs_linked(trans)
        visual = linked["visual"]
        if visual.ndim == 4:
            summary = visual.mean(dim=2)
        elif visual.ndim == 3:
            summary = visual
        else:
            raise RuntimeError(f"unexpected real-future latent shape {tuple(visual.shape)}")
    result = summary.detach().cpu().numpy()
    if result.shape[:2] != (N_CANDIDATES, HORIZON + 1):
        raise MechanismGateError(f"real-future encoder shape mismatch: {result.shape}")
    if not np.isfinite(result).all():
        raise MechanismGateError("real-future encoder returned non-finite latent values")
    return result


def future_encoder_interface_selfcheck(model, preprocessor, obs0_targets, device,
                                       move_to_device, torch, np):
    if not isinstance(obs0_targets, dict) or "visual" not in obs0_targets:
        raise MechanismGateError("fixed task0 obs_0 target has no visual leaf")
    obs0_visual = torch.as_tensor(obs0_targets["visual"], device="cpu")
    if obs0_visual.ndim == 5:
        if int(obs0_visual.shape[1]) < 1:
            raise MechanismGateError(f"fixed task0 obs_0 has empty time dimension: {tuple(obs0_visual.shape)}")
        frame = obs0_visual[0, 0]
    elif obs0_visual.ndim == 4:
        frame = obs0_visual[0]
    elif obs0_visual.ndim == 3:
        frame = obs0_visual
    else:
        raise MechanismGateError(f"fixed task0 obs_0 visual shape is unsupported: {tuple(obs0_visual.shape)}")
    if frame.ndim != 3 or frame.numel() == 0 or not torch.isfinite(frame).all():
        raise MechanismGateError(f"fixed task0 obs_0 image is invalid: {tuple(frame.shape)}")
    repeated_future = frame.reshape(1, 1, *frame.shape).expand(
        N_CANDIDATES, HORIZON + 1, *frame.shape).contiguous()

    rng_before = torch_rng_snapshot(torch)
    buffers_before = named_buffer_snapshot(model)
    flags_before = training_flags(model)
    try:
        latents = true_future_summary(
            model, preprocessor, {"future_visuals": repeated_future},
            device, move_to_device, torch, np)
    finally:
        restore_buffers(model, buffers_before)
        restore_torch_rng(torch, rng_before)
    flags_after = training_flags(model)
    if flags_after != flags_before:
        raise MechanismGateError("future-encoder interface selfcheck changed model training flags")
    if latents.shape != (N_CANDIDATES, HORIZON + 1, 384) or not np.isfinite(latents).all():
        raise MechanismGateError(f"future-encoder interface selfcheck output invalid: {latents.shape}")
    return {
        "passed": True,
        "source": "fixed task0 obs_0 visual repeated over a 300x6 future batch",
        "input_shape": list(repeated_future.shape),
        "latent_shape": list(latents.shape),
        "finite": True,
        "training_flags_unchanged": True,
        "rng_restored": True,
        "buffers_restored": True,
    }


def relative_response(predicted, truth, np):
    delta_pred = predicted[:, -1] - predicted[0:1, -1]
    delta_true = truth[:, -1] - truth[0:1, -1]
    pred_norm = np.linalg.norm(delta_pred, axis=-1)
    true_norm = np.linalg.norm(delta_true, axis=-1)
    truth_valid = true_norm > TRUE_DELTA_EPS
    pred_valid = pred_norm > TRUE_DELTA_EPS
    cos_valid = truth_valid & pred_valid
    cosines = np.sum(delta_pred[cos_valid] * delta_true[cos_valid], axis=-1) / (
        pred_norm[cos_valid] * true_norm[cos_valid]
    )
    ratios = np.where(
        pred_valid[truth_valid],
        pred_norm[truth_valid] / true_norm[truth_valid],
        0.0,
    )
    return {
        "reference_bank_index": 0,
        "true_delta_degenerate_threshold": TRUE_DELTA_EPS,
        "true_delta_degenerate_count": int(np.count_nonzero(~truth_valid)),
        "predicted_zero_response_count_when_truth_nonzero": int(np.count_nonzero(truth_valid & ~pred_valid)),
        "valid_direction_count": int(np.count_nonzero(cos_valid)),
        "direction_cosine_mean": float(np.mean(cosines)) if cosines.size else None,
        "magnitude_ratio_mean": float(np.mean(ratios)) if ratios.size else None,
        "magnitude_ratio_median": float(np.median(ratios)) if ratios.size else None,
        "predicted_zero_direction_cosine": None,
    }


def candidate_provenance(bank_info, np):
    return {
        "candidate_sources": bank_info["sources"],
        "duplicate_groups_by_exact_sequence": bank_info["duplicate_groups"],
        "duplicate_instance_count": int(sum(len(group) for group in bank_info["duplicate_groups"])),
        "candidate_bank_size": int(len(bank_info["actions"])),
    }


def context_metrics(context, bank_info, truth, cold_loss, hist_loss,
                    cold_latents, hist_latents, true_latents, np, torch, device):
    costs = truth["diagnostic_costs"]
    if not all(np.isfinite(array).all() for array in
               (costs, cold_loss, hist_loss, cold_latents, hist_latents, true_latents)):
        raise MechanismGateError("non-finite inputs reached mechanism scoring")
    oracle = float(np.mean(np.sort(costs)[:N_ELITES]))
    cold_selected = torch.argsort(torch.as_tensor(cold_loss, device=device))[:N_ELITES].cpu().numpy()
    hist_selected = torch.argsort(torch.as_tensor(hist_loss, device=device))[:N_ELITES].cpu().numpy()
    cold_top = float(np.mean(costs[cold_selected]))
    hist_top = float(np.mean(costs[hist_selected]))
    cold_regret = float(cold_top - oracle)
    hist_regret = float(hist_top - oracle)
    cold_mse = float(np.mean((cold_latents[:, 1:] - true_latents[:, 1:]) ** 2))
    hist_mse = float(np.mean((hist_latents[:, 1:] - true_latents[:, 1:]) ** 2))
    return {
        "task_index": int(context["task_index"]),
        "anchor": int(context["anchor"]),
        "candidate_provenance": candidate_provenance(bank_info, np),
        "oracle_true_top30_cost": oracle,
        "cold_selected_true_top30_cost": cold_top,
        "history_selected_true_top30_cost": hist_top,
        "cold_elite_regret": cold_regret,
        "history_elite_regret": hist_regret,
        "cold_selected_official_success_fraction": float(np.mean(truth["official_success"][cold_selected])),
        "history_selected_official_success_fraction": float(np.mean(truth["official_success"][hist_selected])),
        "cold_true_latent_mse_future_frames_1_to_5": cold_mse,
        "history_true_latent_mse_future_frames_1_to_5": hist_mse,
        "cold_action_response": relative_response(cold_latents, true_latents, np),
        "history_action_response": relative_response(hist_latents, true_latents, np),
        "candidate_count": int(len(costs)),
    }


def aggregate_primary(context_results, task_ids, np, math):
    by_task = {}
    for row in context_results:
        if "cold_elite_regret" not in row:
            continue
        by_task.setdefault(row["task_index"], []).append(row)
    tasks = []
    for task_id in task_ids:
        rows = by_task.get(int(task_id), [])
        if not rows:
            continue
        cold = float(np.mean([row["cold_elite_regret"] for row in rows]))
        hist = float(np.mean([row["history_elite_regret"] for row in rows]))
        at_floor = cold < PRIMARY_FLOOR
        reduction = (cold - hist) / max(cold, PRIMARY_FLOOR)
        tasks.append({"task_index": int(task_id), "contexts": len(rows),
                      "cold_mean_anchor_regret": cold, "history_mean_anchor_regret": hist,
                      "at_floor": bool(at_floor), "relative_regret_reduction": float(reduction)})
    nonfloor = [row for row in tasks if not row["at_floor"]]
    reductions = [row["relative_regret_reduction"] for row in nonfloor]
    required_improved = int(math.ceil(5.0 / 8.0 * len(nonfloor)))
    improved = sum(value > 0 for value in reductions)
    measured_contexts = sum(len(rows) for rows in by_task.values())
    sample_ok = (measured_contexts >= 12 and len(tasks) >= 6 and len(nonfloor) >= 6)
    median_reduction = float(np.median(reductions)) if reductions else None
    supported = bool(sample_ok and median_reduction is not None and median_reduction >= 0.10
                     and improved >= required_improved)
    status = "INCOMPLETE" if not sample_ok else ("SUPPORTED_EXPLORATORY" if supported else "NO_GO")
    return {
        "status": status,
        "contexts": measured_contexts,
        "tasks_with_contexts": len(tasks),
        "non_at_floor_tasks": len(nonfloor),
        "minimums": {"contexts": 12, "tasks": 6, "non_at_floor_tasks": 6},
        "relative_regret_reduction_median_nonfloor_tasks": median_reduction,
        "improved_nonfloor_tasks": int(improved),
        "required_improved_nonfloor_tasks": required_improved,
        "tasks": tasks,
        "no_significance_test": True,
    }


def save_context_artifacts(out_dir, context, bank_info, truth, cold_loss,
                           hist_loss, cold_latents, hist_latents, true_latents, np):
    target = Path(out_dir) / "contexts" / f"task{context['task_index']:02d}_anchor{context['anchor']}"
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        task_index=np.int64(context["task_index"]),
        anchor=np.int64(context["anchor"]),
        eval_seed=np.int64(context["eval_seed"]),
        history_indices=np.asarray(context["history_indices"], dtype=np.int64),
        prefix_actions=context["prefix_actions"],
        past_action_blocks=context["past_action_blocks"],
        current_state=context["current_state"],
        candidate_actions=bank_info["actions"],
        candidate_sources_json=np.asarray(json.dumps(bank_info["sources"])),
        duplicate_groups_json=np.asarray(json.dumps(bank_info["duplicate_groups"])),
        truth_endpoint_states=truth["endpoint_states"],
        true_diagnostic_costs=truth["diagnostic_costs"],
        object_only_position_errors=truth["object_only_position_errors"],
        official_success=truth["official_success"],
        cold_model_objectives=cold_loss,
        history_model_objectives=hist_loss,
        cold_terminal_latent=cold_latents[:, -1],
        history_terminal_latent=hist_latents[:, -1],
        truth_terminal_latent=true_latents[:, -1],
    )
    for key, value in context["history_obs"].items():
        arrays[f"history_obs_{key}"] = value
    np.savez_compressed(target.with_suffix(".npz"), **arrays)


def run_pipeline(args, job_id, host, nodefile):
    import math
    import contextlib
    import io
    import numpy as np
    import torch
    import gym
    import hydra
    from einops import repeat, rearrange
    from omegaconf import OmegaConf
    from utils import seed, move_to_device
    import plan as official_plan
    from env.venv import SubprocVectorEnv

    started = time.monotonic()
    out_dir = Path(args.out).resolve()
    if not out_dir.is_dir():
        raise RuntimeError(f"PBS wrapper must create the isolated output directory first: {out_dir}")
    result_dir = out_dir / "results"
    result_dir.mkdir(exist_ok=False)
    synthetic_bank_gate = synthetic_bank_shape_selfcheck(np)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    cpu = json.loads(Path(args.cpu_preflight).read_text(encoding="utf-8"))
    gates = cpu.get("gates_passed", {})
    if (cpu.get("preflight_status") != "PASS"
            or cpu.get("source_commit") != EXPECTED_COMMIT
            or not cpu.get("identity_preflight_passed")
            or not all(gates.get(name) is True for name in
                       ("task_identity", "environment_and_replay_identity", "fixed_failure_cohort", "dummy_alignment"))):
        raise RuntimeError("CPU preflight report does not pass all required identity/cohort/alignment gates")
    if freeze.get("source", {}).get("commit") != EXPECTED_COMMIT:
        raise RuntimeError("freeze source commit mismatch")
    repo = Path(freeze["source"]["remote_checkout"])
    marker = repo / "SOURCE_COMMIT"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != EXPECTED_COMMIT:
        raise RuntimeError("pinned checkout SOURCE_COMMIT does not match frozen revision")
    if cpu.get("job_id") != freeze.get("cpu_preflight", {}).get("job_id"):
        raise RuntimeError("CPU preflight job differs from frozen preparation report")
    task_ids = cpu.get("exploratory_cohort", {}).get("indices_zero_based", [])
    if (task_ids != EXPECTED_COHORT
            or cpu.get("exploratory_cohort", {}).get("prior_failure_count") != 29
            or cpu.get("run_manifest_identity", {}).get("task_count") != 50
            or cpu.get("run_manifest_identity", {}).get("goal_H") != HORIZON):
        raise RuntimeError("CPU preflight cohort or native planning identity differs from the frozen protocol")
    with Path(args.env_identity).open("rb") as handle:
        env_identity = pickle.load(handle)
    if (not env_identity.get("identity_checks_passed")
            or env_identity.get("task_indices") != EXPECTED_COHORT):
        raise RuntimeError("environment identity bundle does not match the fixed 8-task cohort")
    identity_path = cpu.get("env_identity_bundle", {}).get("path")
    if identity_path and str(Path(args.env_identity)) != identity_path:
        raise RuntimeError("provided environment identity bundle path differs from CPU report")
    task_ids = [int(value) for value in task_ids]
    eval_seeds = [99 * value + 1 for value in task_ids]
    env_infos = [env_identity["env_infos"][str(value)] for value in task_ids]

    checkpoint = Path(freeze["source"]["checkpoint"])
    model_dir = checkpoint.parent.parent
    model_config_path = model_dir / "hydra.yaml"
    target_path = repo / PLAN_OUTPUT / "plan_targets.pkl"
    if not checkpoint.is_file() or not model_config_path.is_file() or not target_path.is_file():
        raise FileNotFoundError("pinned checkpoint/config/plan targets are unavailable")
    source_commit = marker.read_text(encoding="utf-8").strip()
    if source_commit != EXPECTED_COMMIT:
        raise RuntimeError("source marker changed after CPU preparation")

    seed(99)
    model_cfg = OmegaConf.load(model_config_path)
    if (int(model_cfg.num_hist) != 3 or int(model_cfg.num_pred) != 1
            or int(model_cfg.frameskip) != FRAMESKIP
            or int(model_cfg.embed_dim) != 384
            or str(model_cfg.predictor._target_).split(".")[-1] != "LinearDynamicsPredictor"
            or str(model_cfg.predictor.mode) != "mlp_var"
            or str(model_cfg.action_conditioning) != "adaln"):
        raise RuntimeError("sparse model configuration differs from frozen mlp_var D384 temporal/conditioning settings")
    _, splits = hydra.utils.call(
        model_cfg.env.dataset, num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred, frameskip=model_cfg.frameskip)
    dset = splits["valid"]
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation has no CUDA device")
    model = official_plan.load_model(
        checkpoint, model_cfg, int(model_cfg.num_action_repeat), device=device)
    flags_before = training_flags(model)

    with target_path.open("rb") as handle:
        all_targets = pickle.load(handle)
    subset_targets = {key: subset_tree(value, task_ids, np, torch)
                      for key, value in all_targets.items()
                      if key in {"obs_0", "obs_g", "state_0", "state_g", "gt_actions", "goal_H"}}
    goal_h_values = subset_targets["goal_H"]
    if torch.is_tensor(goal_h_values):
        goal_h_values = goal_h_values.detach().cpu().reshape(-1).tolist()
    elif np.isscalar(goal_h_values):
        goal_h_values = [goal_h_values]
    else:
        goal_h_values = np.asarray(goal_h_values).reshape(-1).tolist()
    if not goal_h_values or any(int(value) != HORIZON for value in goal_h_values):
        raise RuntimeError("subset target horizon mismatch")
    input_dir = out_dir / "inputs"
    input_dir.mkdir(exist_ok=False)
    subset_path = input_dir / "plan_targets_subset.pkl"
    with subset_path.open("wb") as handle:
        pickle.dump(subset_targets, handle, protocol=4)

    plan_cfg = OmegaConf.load(repo / "conf/plan.yaml")
    planner_cfg = OmegaConf.load(repo / "conf/planner/mpc_cem.yaml")
    cfg_dict = OmegaConf.to_container(plan_cfg, resolve=False)
    cfg_dict.update({
        "saved_folder": str(out_dir),
        "ckpt_base_path": str(model_dir.parent),
        "model_name": model_dir.name,
        "model_epoch": "latest",
        "seed": 99,
        "n_evals": len(task_ids),
        "goal_source": "file",
        "goal_file_path": str(subset_path),
        "goal_H": HORIZON,
        "n_plot_samples": 0,
        "debug_dset_init": False,
        "wandb_logging": False,
        "planner": OmegaConf.to_container(planner_cfg, resolve=False),
    })
    cfg_dict["planner"]["sub_planner"]["horizon"] = HORIZON
    cfg_dict["planner"]["sub_planner"]["num_samples"] = N_CANDIDATES
    cfg_dict["planner"]["sub_planner"]["topk"] = N_ELITES
    cfg_dict["planner"]["sub_planner"]["opt_steps"] = N_ITERATIONS
    cfg_dict["planner"]["sub_planner"]["eval_every"] = 1

    env_args = list(OmegaConf.to_container(model_cfg.env.args, resolve=True))
    env_kwargs = OmegaConf.to_container(model_cfg.env.kwargs, resolve=True)
    env_name = str(model_cfg.env.name)
    env = SubprocVectorEnv([
        lambda args=env_args, kwargs=env_kwargs, name=env_name: gym.make(name, *args, **kwargs)
        for _ in task_ids
    ])
    physics_env = None
    original_cwd = Path.cwd()
    try:
        env.update_env(env_infos)
        # PlanWorkspace.dump_targets() writes a cwd-relative plan_targets.pkl.
        # Keep that and all native logs inside this PBS job's isolated OUT.
        os.chdir(out_dir)
        workspace = official_plan.PlanWorkspace(
            cfg_dict=cfg_dict, wm=model, dset=dset, env=env,
            env_name=env_name, frameskip=FRAMESKIP, wandb_run=None)
        native_action_metadata = {
            "dataset_action_dim": int(dset.action_dim),
            "frameskip": int(FRAMESKIP),
            "packed_action_dim_expected": int(PACKED_ACTION_DIM),
            "workspace_action_dim": int(workspace.action_dim),
            "cem_action_dim": int(workspace.planner.sub_planner.action_dim),
        }
        if (native_action_metadata["dataset_action_dim"] != 2
                or native_action_metadata["workspace_action_dim"] != PACKED_ACTION_DIM
                or native_action_metadata["cem_action_dim"] != PACKED_ACTION_DIM):
            raise MechanismGateError(f"pinned native action-dimension gate failed: {native_action_metadata}")
        workspace.eval_seed = eval_seeds
        workspace.evaluator.seed = eval_seeds
        workspace.log_filename = str(out_dir / "native_logs.json")
        workspace.planner.log_filename = None
        workspace.planner.sub_planner.log_filename = None
        preprocessor = workspace.data_preprocessor
        future_encoder_gate = future_encoder_interface_selfcheck(
            model, preprocessor, subset_targets["obs_0"], device,
            move_to_device, torch, np)
        write_json(result_dir / "runner_preflight.json", {
            "job_id": job_id,
            "source_commit": source_commit,
            "synthetic_bank_shape_selfcheck": synthetic_bank_gate,
            "native_action_metadata": native_action_metadata,
            "future_encoder_interface_selfcheck": future_encoder_gate,
        })
        if len(workspace.planner.planned_actions) != 0:
            raise RuntimeError("native MPC planner unexpectedly has pre-existing actions")
        capture = Capture(model, task_ids, FRAMESKIP, N_ELITES, N_CANDIDATES, np, torch)

        original_rollout = model.rollout
        def rollout_hook(*args, **kwargs):
            obs_0 = kwargs.get("obs_0", args[0] if args else None)
            actions = kwargs.get("act", args[1] if len(args) > 1 else None)
            if actions is not None:
                capture.before_rollout(obs_0, actions)
            return original_rollout(*args, **kwargs)
        model.rollout = rollout_hook

        parent = workspace.planner
        subplanner = parent.sub_planner
        original_sub_plan = subplanner.plan
        original_objective = subplanner.objective_fn
        workspace_objective = parent.objective_fn
        def plan_hook(obs_0, obs_g, actions=None):
            capture.begin_cem(parent, obs_0)
            try:
                return original_sub_plan(obs_0=obs_0, obs_g=obs_g, actions=actions)
            finally:
                capture.active = False
                capture.current_by_slot = {}
        def objective_hook(prediction, goal):
            value = original_objective(prediction, goal)
            capture.scored(value, goal)
            return value
        subplanner.plan = plan_hook
        subplanner.objective_fn = objective_hook

        original_eval = workspace.evaluator.eval_actions
        def eval_hook(actions, action_len=None, filename="output", save_video=False):
            result = original_eval(actions, action_len, filename, save_video)
            if re.fullmatch(r"plan\d+", str(filename)):
                capture.record_plan_eval(filename, actions, result[2], result[3])
            return result
        workspace.evaluator.eval_actions = eval_hook

        print(json.dumps({"stage": "NATIVE_CAPTURE_STARTED", "job_id": job_id,
                          "task_ids": task_ids, "planning_seed": 99,
                          "eval_seeds": eval_seeds, "capture_anchors": [1, 2],
                          "cem": {"candidates": 300, "elites": 30, "iterations": 30,
                                  "eval_every": 1}, "model_training_flags": flags_before}))
        native_start = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            workspace.perform_planning()
        native_seconds = time.monotonic() - native_start
        flags_after_capture = training_flags(model)
        if flags_before != flags_after_capture:
            raise RuntimeError("model training flags changed during native capture")
        contexts = [capture.contexts[key] for key in sorted(capture.contexts)]
        valid_contexts = [context for context in contexts
                          if context.get("early") is not None and context.get("late") is not None]
        if not valid_contexts:
            raise RuntimeError("native capture produced no complete early/late candidate context")
        context_keys = {(c["task_index"], c["anchor"]) for c in contexts}
        valid_context_keys = {(c["task_index"], c["anchor"]) for c in valid_contexts}
        write_json(result_dir / "capture_manifest.json", {
            "source_commit": source_commit,
            "checkpoint": str(checkpoint),
            "planning_seed": 99,
            "task_ids_zero_based": task_ids,
            "eval_seeds_by_task": {str(t): 99 * t + 1 for t in task_ids},
            "active_contexts_captured": len(contexts),
            "complete_early_late_contexts": len(valid_contexts),
            "synthetic_bank_shape_selfcheck": synthetic_bank_gate,
            "native_action_metadata": native_action_metadata,
            "missing_contexts": [
                {"task_index": task_id, "anchor": anchor,
                 "reason": ("inactive_or_already_successful" if (task_id, anchor) not in context_keys
                            else "missing_early_or_late_population_native_early_stop")}
                for task_id in task_ids for anchor in (1, 2)
                if (task_id, anchor) not in valid_context_keys
            ],
            "model_training_flags_before_and_after": flags_before,
            "module_mode_changed": False,
            "native_rng_calls_modified": False,
            "native_eval_every_mean_environment_calls_preserved": True,
            "native_capture_seconds": native_seconds,
        })
        env_identity_map = env_identity["env_infos"]
        physics_env = make_physics_env(model_cfg, gym, OmegaConf)
        objective_fn = workspace_objective

        # Interface gate first: reproduce the late full CEM population's cold objective
        # using the exact pre-rollout RNG/buffers, batch size, current frame and goal.
        parity_records = []
        for context in valid_contexts:
            parity = native_history1_parity(
                model, preprocessor, objective_fn, context, device,
                repeat, move_to_device, torch, np)
            parity_records.append({"task_index": context["task_index"],
                                   "anchor": context["anchor"], **parity})
            if not parity["passed"]:
                write_json(result_dir / "run_summary.json", {
                    "status": "INCOMPLETE_INTERFACE_GATE_FAILURE",
                    "reason": "history1/cold branch did not reproduce native CEM objective",
                    "parity": parity_records,
                    "closed_loop_run": False,
                })
                return 3
        print(json.dumps({"stage": "HISTORY1_PARITY_PASSED", "contexts": len(parity_records)}))

        # Candidate bank construction uses only frozen indices and native CEM elites.
        for context in valid_contexts:
            context["bank_info"] = make_candidate_bank(context, np)
            if context["bank_info"] is None:
                raise RuntimeError("complete context lost its frozen candidate bank")

        first_context = valid_contexts[0]
        first_bank = first_context["bank_info"]["actions"]
        physics_env.unwrapped.update_env(env_identity_map[str(first_context["task_index"])])

        # Cost-only calibration: generate no task errors, ranking, regret or response
        # metrics. Keep raw physical truth in RAM so it can be reused after the gate.
        print(json.dumps({"stage": "COST_CALIBRATION_STARTED", "task_index": first_context["task_index"],
                          "anchor": first_context["anchor"], "bank_size": 300}))
        physical_start = time.monotonic()
        first_truth_raw = collect_raw_truth(
            physics_env, env_identity_map[str(first_context["task_index"])],
            first_context, first_bank, all_targets, first_context["task_index"],
            preprocessor, torch, np, rearrange, FRAMESKIP)
        physical_seconds = time.monotonic() - physical_start

        torch.cuda.synchronize(device)
        model_start = time.monotonic()
        model_costs = cost_only_predictor_forwards(
            model, preprocessor, first_context, first_bank, device,
            repeat, move_to_device, torch, np, first_truth_raw)
        torch.cuda.synchronize(device)
        model_forward_seconds = time.monotonic() - model_start
        before_quality = time.monotonic() - started
        anchor2_prefix_multiplier = 1.0 if first_context["anchor"] == 2 else 1.5
        estimated_total = 1.2 * (
            before_quality + 15 * physical_seconds * anchor2_prefix_multiplier
            + 16 * model_forward_seconds) + 300.0
        calibration = {
            "status": "COST_ONLY_COMPLETE",
            "cost_only_metric_values_emitted": False,
            "native_capture_seconds_all_8_tasks": native_seconds,
            "elapsed_through_calibration_seconds": before_quality,
            "first_context_full_300_candidate_prefix_replay_seconds": physical_seconds,
            "first_context_cold_predictor_seconds": model_costs["cold_history1_forward_seconds"],
            "first_context_real_history_predictor_seconds": model_costs["real_history3_forward_seconds"],
            "first_context_real_future_encoder_seconds": model_costs["true_future_encoder_seconds"],
            "first_context_paired_model_and_truth_encoder_seconds": model_forward_seconds,
            "anchor2_prefix_multiplier": anchor2_prefix_multiplier,
            "anchor2_prefix_basis": "prefix+5 future blocks: anchor1=10 blocks and anchor2=15 blocks, so 1.5x physical replay cost when calibration is anchor1",
            "conservative_estimate_formula": "1.2*(elapsed_through_calibration + 15*physical_context_seconds*anchor2_prefix_multiplier + 16*(cold_forward+history_forward+truth_encoder)) + 300s teardown/stageout reserve; the cost-only first forward is not retained for mechanism metrics",
            "projected_full_mechanism_seconds": estimated_total,
            "internal_cap_seconds": RESOURCE_CAP_SECONDS,
            "contexts_for_projection": 16,
            "bank_size": 300,
            "native_eval_every_1_environment_calls_included": True,
            "truth_replays_from_init_plus_full_prefix": True,
            "fixed_bank_unchanged": True,
        }
        write_json(result_dir / "cost_calibration.json", calibration)
        print(json.dumps({"stage": "COST_CALIBRATION_WRITTEN", **calibration}))
        if estimated_total > RESOURCE_CAP_SECONDS:
            result = {
                "status": "INCOMPLETE_RESOURCE_LIMIT",
                "reason": "conservative full 16-context projection exceeds the frozen 80-minute internal cap",
                "cost_calibration": calibration,
                "quality_metrics_computed": False,
                "closed_loop_run": False,
            }
            write_json(result_dir / "run_summary.json", result)
            return 4

        # The cost gate passed. The first context's raw truth is reused; only now derive
        # endpoint errors, model ranking, response diagnostics, and the primary gate.
        print(json.dumps({"stage": "MECHANISM_METRICS_STARTED", "reuse_first_context_raw_truth": True}))
        context_results = []
        parity_by_key = {(r["task_index"], r["anchor"]): r for r in parity_records}
        for context in valid_contexts:
            key = (context["task_index"], context["anchor"])
            bank_info = context["bank_info"]
            if context is first_context:
                goal_state = as_numpy(all_targets["state_g"][context["task_index"]], np).copy()
                truth = derive_truth_metrics(first_truth_raw, goal_state, np)
            else:
                env_info = env_identity_map[str(context["task_index"]) ]
                raw_truth = collect_raw_truth(
                    physics_env, env_info, context, bank_info["actions"],
                    all_targets, context["task_index"], preprocessor, torch, np,
                    rearrange, FRAMESKIP)
                goal_state = as_numpy(all_targets["state_g"][context["task_index"]], np).copy()
                truth = derive_truth_metrics(raw_truth, goal_state, np)
            cold_loss, hist_loss, cold_latents, hist_latents = paired_model_scores(
                model, preprocessor, objective_fn, context, bank_info["actions"],
                device, repeat, move_to_device, torch, np)
            live_buffers = named_buffer_snapshot(model)
            live_rng = torch_rng_snapshot(torch)
            try:
                restore_buffers(model, context["late"]["buffers"])
                restore_torch_rng(torch, context["late"]["torch_rng"])
                truth_latents = true_future_summary(
                    model, preprocessor, truth, device, move_to_device, torch, np)
            finally:
                restore_buffers(model, live_buffers)
                restore_torch_rng(torch, live_rng)
            row = context_metrics(context, bank_info, truth, cold_loss, hist_loss,
                                  cold_latents, hist_latents, truth_latents, np, torch, device)
            row["history1_native_parity"] = parity_by_key[key]
            context_results.append(row)
            save_context_artifacts(result_dir, context, bank_info, truth, cold_loss,
                                   hist_loss, cold_latents, hist_latents,
                                   truth_latents, np)
            print(json.dumps({"stage": "CONTEXT_COMPLETE", "task_index": context["task_index"],
                              "anchor": context["anchor"], "cold_regret": row["cold_elite_regret"],
                              "history_regret": row["history_elite_regret"]}))

        for task_id in task_ids:
            for anchor in (1, 2):
                key = (task_id, anchor)
                if key not in valid_context_keys:
                    context_results.append({"task_index": task_id, "anchor": anchor,
                                            "status": "MISSING_CEM_CONTEXT"})
        completed = [row for row in context_results if "cold_elite_regret" in row]
        primary = aggregate_primary(completed, task_ids, np, math)
        result = {
            "status": primary["status"],
            "job_id": job_id,
            "host": host,
            "source_commit": source_commit,
            "checkpoint": str(checkpoint),
            "cpu_preflight_job_id": cpu["job_id"],
            "task_ids_zero_based": task_ids,
            "eval_seeds_by_task": {str(t): 99 * t + 1 for t in task_ids},
            "native_capture_seconds": native_seconds,
            "cost_calibration": calibration,
            "history1_parity": parity_records,
            "model_training_flags_before_and_after_capture": flags_before,
            "module_mode_changed": False,
            "bank_definition": {
                "size": 300,
                "packed_action_dim": PACKED_ACTION_DIM,
                "dataset_action_dim": native_action_metadata["dataset_action_dim"],
                "frameskip": FRAMESKIP,
                "early": "iteration 0; indices linspace(0,299,100,dtype=int)",
                "late_non_elite": "last actually evaluated iteration; complement of native top30; indices linspace(0,269,170,dtype=int)",
                "late_elite": "all native top30 in argsort objective order",
                "duplicates_retained_and_reported": True,
            },
            "contexts": context_results,
            "primary_gate": primary,
            "closed_loop_run": False,
            "claims": {"task_success_claim": False, "speedup_claim": False,
                       "candidate_is_independent_sample": False,
                       "interpretation": "exploratory fixed-bank mechanism screen only"},
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        write_json(result_dir / "run_summary.json", result)
        return 0 if primary["status"] != "INCOMPLETE" else 5
    finally:
        close_environments(physics_env, env)
        os.chdir(original_cwd)


def main():
    args = parse_args()
    # This must be the first operational check; no model/data import or output before it.
    job_id, host, nodefile = require_compute_allocation()
    out_dir = Path(args.out).resolve()
    try:
        status = run_pipeline(args, job_id, host, nodefile)
        return status
    except Exception as exc:
        result_dir = out_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        gate_failure = isinstance(exc, MechanismGateError)
        error = {
            "status": "INCOMPLETE_INTERFACE_GATE_FAILURE" if gate_failure else "ERROR",
            "job_id": job_id,
            "host": host,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "closed_loop_run": False,
        }
        write_json(result_dir / "run_summary.json", error)
        print(json.dumps(error, indent=2))
        return 3 if gate_failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
