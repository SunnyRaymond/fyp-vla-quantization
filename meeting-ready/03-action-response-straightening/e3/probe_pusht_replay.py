#!/usr/bin/env python3
"""Probe exact PushT reset-and-replay reproducibility for E3 branches.

This tests seeded reset plus common-prefix replay. It does not claim that the
simulator exposes a full mid-episode snapshot/restore API.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any


def require_compute_allocation() -> dict[str, str]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    host = socket.gethostname().split(".", 1)[0].lower()
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("PBS_JOBID and a valid PBS_NODEFILE are required")
    if any(marker in host for marker in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login/submit host: {host}")
    nodes = {
        line.strip().split(".", 1)[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in nodes:
        raise RuntimeError(f"host {host} is not listed in PBS_NODEFILE")
    return {"pbs_job_id": job_id, "hostname": host}


def plain(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def tree_equal(left: Any, right: Any) -> bool:
    import numpy as np

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.shape == right.shape
            and left.dtype == right.dtype
        ):
            return False
        equal_nan = np.issubdtype(left.dtype, np.inexact)
        return bool(np.array_equal(left, right, equal_nan=equal_nan))
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, (tuple, list))
            and isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def pymunk_snapshot(core: Any) -> dict[str, Any]:
    """Capture body state for replay comparison, not a serializable Space snapshot."""
    bodies = []
    fields = (
        "body_type",
        "mass",
        "moment",
        "position",
        "velocity",
        "force",
        "angle",
        "angular_velocity",
        "torque",
        "center_of_gravity",
    )
    for body in core.space.bodies:
        bodies.append(
            {
                field: plain(getattr(body, field))
                for field in fields
                if hasattr(body, field)
            }
        )
    return {
        "bodies": bodies,
        "shape_count": len(core.space.shapes),
        "constraint_count": len(core.space.constraints),
        "n_contact_points": getattr(core, "n_contact_points", None),
        "latest_action": plain(getattr(core, "latest_action", None)),
        "goal_state": plain(getattr(core, "goal_state", None)),
        "environment_rng": plain(core.rng.bit_generator.state),
        "gym_rng": plain(core.np_random.bit_generator.state),
    }


def capture(core: Any, observation: Any, info: Any, reward: float | None = None,
            terminated: bool | None = None, truncated: bool | None = None) -> dict[str, Any]:
    return {
        "observation": observation,
        "info": info,
        "rgb": core.render(),
        "physics": pymunk_snapshot(core),
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
    }


def one_replay(gym: Any, seed: int, reset_state: list[float], goal_state: list[float],
               prefix: list[list[float]], branch: list[list[float]]) -> dict[str, Any]:
    import numpy as np

    env = gym.make("swm/PushT-v1", max_episode_steps=50, render_mode="rgb_array")
    core = env.unwrapped
    try:
        env.action_space.seed(seed)
        observation, info = env.reset(
            seed=seed,
            options={"state": reset_state, "goal_state": goal_state},
        )
        reset = capture(core, observation, info)
        prefix_trace = []
        for action in prefix:
            observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            prefix_trace.append(
                capture(core, observation, info, float(reward), bool(terminated), bool(truncated))
            )
        branch_start = capture(core, observation, info)
        branch_trace = []
        for action in branch:
            observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            branch_trace.append(
                capture(core, observation, info, float(reward), bool(terminated), bool(truncated))
            )
        return {
            "reset": reset,
            "prefix_trace": prefix_trace,
            "branch_start": branch_start,
            "branch_trace": branch_trace,
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stablewm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allocation = require_compute_allocation()
    sys.path.insert(0, str(args.stablewm_root.resolve()))

    import gymnasium as gym
    import stable_worldmodel as swm  # noqa: F401 - registers swm/PushT-v1

    seed = 4101
    initial_state = [100.0, 256.0, 220.0, 256.0, 0.0, 0.0, 0.0]
    goal_state = [400.0, 400.0, 400.0, 400.0, 0.0, 0.0, 0.0]
    prefix = [[1.0, 0.0] for _ in range(10)]
    branch_actions = {
        "minus": [[-0.2, 0.0] for _ in range(5)],
        "center": [[0.0, 0.0] for _ in range(5)],
        "plus": [[0.2, 0.0] for _ in range(5)],
    }

    probe_env = gym.make("swm/PushT-v1", max_episode_steps=50, render_mode="rgb_array")
    core = probe_env.unwrapped
    snapshot_pairs = (
        ("get_state", "set_state"),
        ("snapshot", "restore"),
        ("save_state", "load_state"),
        ("state_dict", "load_state_dict"),
    )
    api = {
        "env_id": "swm/PushT-v1",
        "wrapper_type": f"{type(probe_env).__module__}.{type(probe_env).__qualname__}",
        "base_type": f"{type(core).__module__}.{type(core).__qualname__}",
        "reset_signature": str(inspect.signature(core.reset)),
        "state_setter_signature": str(inspect.signature(core._set_state))
        if callable(getattr(core, "_set_state", None)) else None,
        "snapshot_restore_pairs": {
            f"{getter}/{setter}": callable(getattr(core, getter, None))
            and callable(getattr(core, setter, None))
            for getter, setter in snapshot_pairs
        },
        "action_space_shape": list(probe_env.action_space.shape),
        "action_low": probe_env.action_space.low.tolist(),
        "action_high": probe_env.action_space.high.tolist(),
    }
    probe_env.close()

    runs: dict[str, list[dict[str, Any]]] = {}
    for branch_name, actions in branch_actions.items():
        runs[branch_name] = [
            one_replay(gym, seed, initial_state, goal_state, prefix, actions)
            for _ in range(2)
        ]

    reset_equal = all(
        tree_equal(runs[name][0]["reset"], runs[name][1]["reset"])
        for name in branch_actions
    )
    prefix_equal = all(
        tree_equal(runs[name][0]["prefix_trace"], runs[name][1]["prefix_trace"])
        for name in branch_actions
    )
    branch_start_equal = all(
        tree_equal(runs["minus"][0]["branch_start"], runs[name][replicate]["branch_start"])
        for name in branch_actions
        for replicate in range(2)
    )
    branch_replay_equal = all(
        tree_equal(runs[name][0]["branch_trace"], runs[name][1]["branch_trace"])
        for name in branch_actions
    )
    contact_counts = {
        name: [
            step["physics"]["n_contact_points"]
            for step in runs[name][0]["prefix_trace"] + runs[name][0]["branch_trace"]
        ]
        for name in branch_actions
    }
    passed = reset_equal and prefix_equal and branch_start_equal and branch_replay_equal

    result = {
        "schema": "lewm-action-response-straightening.e3-pusht-replay-probe",
        "schema_version": 1,
        "status": "PASS_SEEDED_REPLAY" if passed else "FAIL_REPLAY_MISMATCH",
        "claim_boundary": (
            "Tests deterministic reset plus common-prefix replay and branch outcomes. "
            "Does not establish a full mid-episode snapshot/restore API."
        ),
        "allocation": allocation,
        "runtime": {
            "stable_worldmodel_file": str(Path(swm.__file__).resolve()),
            "gymnasium": gym.__version__,
            "python": platform.python_version(),
        },
        "environment_api": api,
        "design": {
            "seed": seed,
            "reset_state": initial_state,
            "goal_state": goal_state,
            "common_prefix_actions": prefix,
            "branch_actions": branch_actions,
            "replicates_per_branch": 2,
            "branch_states_are_reconstructed_by": "same seeded reset options plus exact common-prefix replay",
        },
        "checks": {
            "reset_observation_and_render_exact": reset_equal,
            "common_prefix_observation_reward_and_physics_exact": prefix_equal,
            "branch_start_public_body_snapshot_exact_across_paths": branch_start_equal,
            "same_branch_replay_observation_reward_and_render_exact": branch_replay_equal,
        },
        "n_contact_points_by_branch": contact_counts,
        "branch_end_state": {
            name: plain(runs[name][0]["branch_trace"][-1]["observation"]["state"])
            for name in branch_actions
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "checks": result["checks"]}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
