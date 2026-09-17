"""OTC-PTQ Phase 1 runner for the pinned Fast-WAM Optional IDM path.

The runner deliberately keeps the experiment small and explicit.  ``probe``
checks one replayable state, ``collect`` records native reference replans and
selects four states, and ``score`` replays those states with one W4 fake-quant
site at a time.  It is intended to run inside an approved PBS compute
allocation; it fails closed before importing/loading the model otherwise.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


EXPECTED_FASTWAM_REVISION = "7faa71108368fbb3b6885649f112af607427a2d4"
DEFAULT_BASE = "/scratch/users/ntu/yguo017/fastwam-smoke"
DEFAULT_CHECKPOINT_NAME = "libero_optional_idm_2cam224.clean.pt"
DEFAULT_STATS_NAME = "libero_optional_idm_2cam224_dataset_stats.json"
DEFAULT_TASK_CONFIG = "libero_optional_idm_2cam224_1e-4"
DEFAULT_SUITE = "libero_goal"
DEFAULT_DEPTHS = (0, 10, 20, 29)
DEFAULT_NUM_STATES = 4
DEFAULT_REPLAN_STEPS = 10
DEFAULT_NUM_STEPS_WAIT = 30
DEFAULT_MAX_STEPS = 400
EPS = 1.0e-3

LOGGER = logging.getLogger("otc_ptq_phase1")


class RunnerError(RuntimeError):
    """An experiment identity, replay, or measurement contract failed."""


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    # Large raw observations remain in the binary artifact. JSON is a compact
    # auditable control/result file suitable for bounded retrieval.
    def compact(value):
        if isinstance(value, Mapping):
            return {k: compact(v) for k, v in value.items()
                    if k not in {'raw', 'obs', 'images', 'env_mutable', 'rng',
                                 'next_obs', 'native_state', 'model_arrays'}
                    and not k.endswith('_images')}
        if isinstance(value, (list, tuple)):
            return [compact(v) for v in value]
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(compact(payload), indent=2, cls=NumpyEncoder, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _torch_save(path: Path, payload: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _expand_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(value)))).resolve()


def _require_pbs_allocation() -> dict[str, str]:
    """Refuse all model/env work on login nodes or outside PBS."""

    job_id = os.environ.get("PBS_JOBID", "").strip()
    host = os.environ.get("OTC_ALLOCATION_HOST", "").strip()
    if not job_id:
        raise RunnerError("PBS_JOBID is required; refusing to run outside a PBS allocation.")
    actual_host = subprocess.check_output(["hostname", "-s"], text=True).strip()
    if "login" in actual_host.lower() or "login" in host.lower():
        raise RunnerError(f"Login node detected ({actual_host!r}); refusing workload.")
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if nodefile and not Path(nodefile).is_file():
        raise RunnerError(f"PBS_NODEFILE is not readable: {nodefile}")
    if host and host.split(".", 1)[0] != actual_host.split(".", 1)[0]:
        raise RunnerError(f"Allocation host mismatch: declared={host!r}, actual={actual_host!r}")
    return {"PBS_JOBID": job_id, "hostname": actual_host, "PBS_NODEFILE": nodefile}


def _verify_gpu() -> dict[str, Any]:
    import torch

    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not mask or "," in mask:
        raise RunnerError(
            "CUDA_VISIBLE_DEVICES must expose exactly the allocated GPU; "
            f"got {mask!r}."
        )
    if not mask.upper().startswith("GPU-"):
        raise RunnerError(f"Expected a PBS GPU UUID in CUDA_VISIBLE_DEVICES, got {mask!r}.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RunnerError("Exactly one CUDA device is required for this runner.")
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    norm = lambda x: x.removeprefix("GPU-").lower()
    if norm(actual_uuid) != norm(mask):
        raise RunnerError(f"Allocated GPU UUID mismatch: mask={mask!r}, device={actual_uuid!r}")
    props = torch.cuda.get_device_properties(0)
    return {
        "visible_mask": mask,
        "uuid": actual_uuid,
        "name": str(props.name),
        "total_memory_bytes": int(props.total_memory),
        "torch_version": str(torch.__version__),
    }


def _git_revision(fastwam_root: Path, identity_gate: Path | None = None) -> str:
    declared = (
        os.environ.get("OTC_FASTWAM_REVISION", "").strip()
        or os.environ.get("FASTWAM_REVISION", "").strip()
    )
    revision = declared
    if not revision and (fastwam_root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(fastwam_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        revision = result.stdout.strip()
    if not revision and identity_gate is not None and identity_gate.is_file():
        gate_text = identity_gate.read_text(encoding="utf-8", errors="replace")
        if EXPECTED_FASTWAM_REVISION not in gate_text:
            raise RunnerError(
                f"Identity gate does not contain pinned Fast-WAM revision: {identity_gate}"
            )
        revision = EXPECTED_FASTWAM_REVISION
    if revision != EXPECTED_FASTWAM_REVISION:
        raise RunnerError(
            "Fast-WAM revision mismatch or unavailable: "
            f"expected={EXPECTED_FASTWAM_REVISION}, got={revision or '<missing>'}."
        )
    return revision


def _checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise RunnerError(f"Checkpoint is missing: {checkpoint}")
    marker = checkpoint.with_suffix(".VERIFIED")
    if not marker.is_file():
        raise RunnerError(f"Verified checkpoint marker is missing: {marker}")
    marker_text = marker.read_text(encoding="utf-8", errors="replace").strip()
    # The marker is the prior compute-node integrity gate.  This runner does
    # not hash the 12 GB checkpoint again; it records the gate and file stats.
    return {
        "path": str(checkpoint),
        "marker": str(marker),
        "marker_text": marker_text,
        "size_bytes": int(checkpoint.stat().st_size),
        "mtime_ns": int(checkpoint.stat().st_mtime_ns),
    }


def _capture_rng() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [x.clone() for x in torch.cuda.get_rng_state_all()],
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state["torch_cuda"]])


_CORE_STATE_ATTRS = ("cur_time", "timestep", "done", "deterministic_reset")
_ROBOT_STATE_ATTRS = ("torques",)
_CONTROLLER_STATE_ATTRS = (
    "new_update", "initial_joint", "initial_ee_pos", "initial_ee_ori_mat",
    "ee_pos", "ee_ori_mat", "ee_pos_vel", "ee_ori_vel", "joint_pos", "joint_vel",
    "J_pos", "J_ori", "J_full", "mass_matrix", "torques",
    "action_scale", "action_input_transform", "action_output_transform",
    "goal_ori", "goal_pos", "relative_ori", "ori_ref", "kp", "kd",
)
_GRIPPER_STATE_ATTRS = ("current_action",)
_ROBOT_BUFFER_ATTRS = (
    "recent_qpos", "recent_actions", "recent_torques", "recent_ee_forcetorques",
    "recent_ee_pose", "recent_ee_vel", "recent_ee_vel_buffer", "recent_ee_acc",
)
_OBSERVABLE_STATE_ATTRS = (
    "_time_since_last_sample", "_current_delay", "_current_observed_value",
    "_sampled", "_enabled", "_active",
)
_MODEL_STATE_ATTRS = (
    "body_pos", "body_quat", "site_pos", "site_quat", "site_rgba", "geom_rgba", "eq_active",
)
_PHYSICS_STATE_ATTRS = (
    "qpos", "qvel", "act", "ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied",
    "mocap_pos", "mocap_quat", "userdata",
)


def _copy_state_value(value: Any) -> Any:
    """Copy serializable leaves only; never deepcopy simulator-bearing objects."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().clone().cpu()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, Mapping):
        return {_copy_state_value(k): _copy_state_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_state_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_state_value(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return type(value)({_copy_state_value(v) for v in value})
    raise RunnerError(
        "Refusing to copy opaque mutable object (would risk disconnected references): "
        f"{type(value)!r}"
    )


def _restore_like(current: Any, saved: Any) -> Any:
    """Restore a leaf while retaining the live object's dtype and device."""
    import torch
    if isinstance(current, torch.Tensor):
        return torch.as_tensor(saved, dtype=current.dtype, device=current.device).clone()
    if isinstance(current, np.ndarray):
        value = np.asarray(saved, dtype=current.dtype)
        if value.shape != current.shape:
            raise RunnerError(f"Array shape changed: {current.shape} vs {value.shape}")
        return value.copy()
    if isinstance(current, tuple):
        return tuple(_restore_like(None, value) for value in saved)
    if isinstance(current, list):
        return [_restore_like(None, value) for value in saved]
    if isinstance(current, Mapping):
        return {_restore_like(None, k): _restore_like(None, v) for k, v in saved.items()}
    return _copy_state_value(saved)


def _capture_named_attrs(obj: Any, names: Sequence[str], label: str) -> dict[str, Any]:
    if obj is None:
        raise RunnerError(f"Required state object is missing: {label}")
    result: dict[str, Any] = {}
    for name in names:
        if not hasattr(obj, name):
            raise RunnerError(f"Required mutable attribute is missing: {label}.{name}")
        value = getattr(obj, name)
        if callable(value):
            raise RunnerError(f"Required mutable attribute is callable: {label}.{name}")
        result[name] = _copy_state_value(value)
    return result


def _restore_attrs(obj: Any, attrs: Mapping[str, Any], label: str = "state") -> None:
    if obj is None:
        raise RunnerError(f"Cannot restore state into missing object: {label}")
    failures: list[str] = []
    for name, value in attrs.items():
        if not hasattr(obj, name):
            failures.append(f"{label}.{name}: attribute missing")
            continue
        try:
            setattr(obj, name, _restore_like(getattr(obj, name), value))
        except Exception as exc:
            failures.append(f"{label}.{name}: {exc}")
    if failures:
        raise RunnerError("Mutable state restore failed: " + "; ".join(failures))


def _capture_state_cache(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RunnerError(f"Required observation cache is not a mapping: {label}")
    return {str(key): _copy_state_value(leaf) for key, leaf in value.items()}


def _capture_buffer(value: Any, label: str) -> dict[str, Any]:
    if value is None or not hasattr(value, "__dict__"):
        raise RunnerError(f"Required robot buffer is unavailable: {label}")
    attrs = {}
    for name, leaf in vars(value).items():
        if callable(leaf):
            raise RunnerError(f"Opaque callable in required buffer: {label}.{name}")
        attrs[name] = _copy_state_value(leaf)
    return {"class": f"{type(value).__module__}.{type(value).__qualname__}", "attrs": attrs}


def _restore_buffer(value: Any, saved: Mapping[str, Any], label: str) -> None:
    if value is None or not hasattr(value, "__dict__"):
        raise RunnerError(f"Required robot buffer is unavailable: {label}")
    actual = f"{type(value).__module__}.{type(value).__qualname__}"
    expected = str(saved.get("class", ""))
    if expected and expected != actual:
        raise RunnerError(f"Robot buffer class changed for {label}: {expected} vs {actual}")
    _restore_attrs(value, saved.get("attrs", {}), label)


def _capture_nested_object(value: Any, label: str) -> Any:
    if value is None:
        return None
    if not hasattr(value, "__dict__"):
        raise RunnerError(f"Required nested controller object has no state dictionary: {label}")
    attrs = {}
    for name, leaf in vars(value).items():
        if callable(leaf):
            raise RunnerError(f"Opaque callable state in {label}.{name}")
        attrs[name] = _copy_state_value(leaf)
    return {"class": f"{type(value).__module__}.{type(value).__qualname__}", "attrs": attrs}


def _restore_nested_object(value: Any, saved: Any, label: str) -> None:
    if saved is None:
        if value is not None:
            raise RunnerError(f"Nested controller object appeared during restore: {label}")
        return
    if value is None or not hasattr(value, "__dict__"):
        raise RunnerError(f"Nested controller object disappeared during restore: {label}")
    actual = f"{type(value).__module__}.{type(value).__qualname__}"
    expected = str(saved.get("class", ""))
    if expected and expected != actual:
        raise RunnerError(f"Nested controller class changed for {label}: {expected} vs {actual}")
    _restore_attrs(value, saved.get("attrs", {}), label)


def _env_core(env: Any) -> Any:
    return getattr(env, "env", env)


def _capture_env_mutable(env: Any) -> dict[str, Any]:
    """Capture known mutable leaves without copying live simulator references."""
    core = _env_core(env)
    robots = list(getattr(core, "robots", []) or [])
    observables = dict(getattr(core, "_observables", {}) or {})
    result: dict[str, Any] = {
        "schema": 3,
        "core": _capture_named_attrs(core, _CORE_STATE_ATTRS, "env.core"),
        "obs_cache": _capture_state_cache(getattr(core, "_obs_cache", None), "env.core._obs_cache"),
        "robots": [],
        "observables": {},
    }
    for index, robot in enumerate(robots):
        controller = getattr(robot, "controller", None)
        controller_state = _capture_named_attrs(controller, _CONTROLLER_STATE_ATTRS, f"robot{index}.controller")
        controller_state["nested"] = {
            name: _capture_nested_object(getattr(controller, name), f"robot{index}.controller.{name}")
            for name in ("interpolator_pos", "interpolator_ori")
        }
        result["robots"].append({
            "attrs": _capture_named_attrs(robot, _ROBOT_STATE_ATTRS, f"robot{index}"),
            "controller": controller_state,
            "gripper": _capture_named_attrs(getattr(robot, "gripper", None), _GRIPPER_STATE_ATTRS, f"robot{index}.gripper"),
            "buffers": {
                name: _capture_buffer(getattr(robot, name), f"robot{index}.{name}")
                for name in _ROBOT_BUFFER_ATTRS
            },
        })
    for name, observable in observables.items():
        result["observables"][str(name)] = _capture_named_attrs(
            observable, _OBSERVABLE_STATE_ATTRS, f"observable:{name}"
        )
    return result


def _restore_env_mutable(
    env: Any,
    state: Mapping[str, Any],
    *,
    restore_observables: bool = True,
    restore_obs_cache: bool = True,
) -> None:
    if int(state.get("schema", 0)) < 3:
        raise RunnerError("Snapshot mutable-state schema is stale; recollect with schema 3.")
    core = _env_core(env)
    _restore_attrs(core, state.get("core", {}), "env.core")
    if restore_obs_cache:
        _restore_attrs(core, {"_obs_cache": state.get("obs_cache", {})}, "env.core")
    else:
        if not hasattr(core, "_obs_cache"):
            raise RunnerError("Environment observation cache is missing")
        core._obs_cache = {}
    robots = list(getattr(core, "robots", []) or [])
    saved_robots = list(state.get("robots", []))
    if len(robots) != len(saved_robots):
        raise RunnerError(f"Robot count changed across state restore: {len(saved_robots)} vs {len(robots)}")
    for index, (robot, robot_state) in enumerate(zip(robots, saved_robots)):
        _restore_attrs(robot, robot_state.get("attrs", {}), f"robot{index}")
        controller = getattr(robot, "controller", None)
        controller_state = robot_state.get("controller", {})
        _restore_attrs(controller, {k: v for k, v in controller_state.items() if k != "nested"}, f"robot{index}.controller")
        nested = controller_state.get("nested", {})
        for name in ("interpolator_pos", "interpolator_ori"):
            _restore_nested_object(getattr(controller, name, None), nested.get(name), f"robot{index}.controller.{name}")
        _restore_attrs(getattr(robot, "gripper", None), robot_state.get("gripper", {}), f"robot{index}.gripper")
        for name, saved_buffer in robot_state.get("buffers", {}).items():
            _restore_buffer(getattr(robot, name, None), saved_buffer, f"robot{index}.{name}")
    observables = dict(getattr(core, "_observables", {}) or {})
    saved_observables = dict(state.get("observables", {}))
    if set(observables) != set(saved_observables):
        raise RunnerError("Observable registry changed across state restore.")
    if restore_observables:
        for name, observable_state in saved_observables.items():
            _restore_attrs(observables[name], observable_state, f"observable:{name}")


def _capture_contacts(env: Any) -> dict[str, Any]:
    sim = getattr(env, "sim", None)
    if sim is None or not hasattr(sim, "data"):
        return {"available": False, "count": None, "pairs": []}
    try:
        ncon = int(sim.data.ncon)
        pairs: list[list[str | int]] = []
        for idx in range(ncon):
            contact = sim.data.contact[idx]
            ids = (int(contact.geom1), int(contact.geom2))
            names: list[str | int] = []
            for geom_id in ids:
                try:
                    if hasattr(sim.model, "geom_id2name"):
                        name = sim.model.geom_id2name(geom_id)
                    else:
                        name = sim.model.id2name(geom_id, "geom")
                except Exception:
                    name = None
                names.append(str(name) if name is not None else geom_id)
            pairs.append(names)
        return {"available": True, "count": ncon, "raw_count": ncon, "pairs": sorted(pairs, key=str)}
    except Exception as exc:
        return {"available": False, "count": None, "pairs": [], "error": str(exc)}


def _capture_physics(env: Any) -> dict[str, Any]:
    sim = getattr(env, "sim", None)
    if sim is None or not hasattr(sim, "data"):
        return {}
    result: dict[str, Any] = {}
    for attr in ("qpos", "qvel", "act", "ctrl"):
        try:
            result[attr] = np.asarray(getattr(sim.data, attr), dtype=np.float64).copy()
        except Exception:
            pass
    try:
        body_positions: dict[str, np.ndarray] = {}
        body_quaternions: dict[str, np.ndarray] = {}
        for body_id in range(int(sim.model.nbody)):
            if hasattr(sim.model, "body_id2name"):
                name = sim.model.body_id2name(body_id)
            else:
                name = sim.model.id2name(body_id, "body")
            if not name:
                continue
            lower = str(name).lower()
            if any(word in lower for word in ("robot", "panda", "gripper", "finger", "table", "floor", "world", "mount", "support")):
                continue
            body_positions[str(name)] = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).copy()
            body_quaternions[str(name)] = np.asarray(sim.data.body_xquat[body_id], dtype=np.float64).copy()
        result["object_body_xpos"] = body_positions
        result["object_body_xquat"] = body_quaternions
        result["task_physics_scope"] = "non-robot, non-support MuJoCo bodies"
    except Exception:
        result["task_physics_scope"] = "all simulator qpos/qvel only; object body extraction unavailable"
    return result


def _copy_obs(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(key): np.array(value, copy=True)
        for key, value in obs.items()
        if isinstance(value, (np.ndarray, list, tuple, np.number))
    }


def _image_obs(obs: Mapping[str, Any], ev: Any) -> dict[str, np.ndarray]:
    images = ev.get_libero_image(obs)
    return {str(key): np.asarray(value, dtype=np.uint8).copy() for key, value in images.items()}


def _obs_max_abs_error(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> float:
    if set(expected) != set(actual):
        raise RunnerError(f"Observation keys changed across restore: {set(expected)} vs {set(actual)}")
    maximum = 0.0
    for key in expected:
        lhs = np.asarray(expected[key])
        rhs = np.asarray(actual[key])
        if lhs.shape != rhs.shape:
            raise RunnerError(f"Observation shape changed for {key}: {lhs.shape} vs {rhs.shape}")
        if np.issubdtype(lhs.dtype, np.number):
            maximum = max(maximum, float(np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64)))))
        elif not np.array_equal(lhs, rhs):
            raise RunnerError(f"Non-numeric observation changed for {key}.")
    return maximum


def _native_state_handles(sim: Any) -> tuple[Any, Any]:
    model = getattr(getattr(sim, "model", None), "_model", None)
    data = getattr(getattr(sim, "data", None), "_data", None)
    if model is None or data is None:
        raise RunnerError("MuJoCo wrapper does not expose sim.model._model and sim.data._data.")
    return model, data


def _capture_native_state(sim: Any) -> dict[str, Any]:
    """Capture MuJoCo integration state, including solver warmstart where exposed."""
    try:
        import mujoco

        model, data = _native_state_handles(sim)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        size = int(mujoco.mj_stateSize(model, spec))
        state = np.empty(size, dtype=np.float64)
        mujoco.mj_getState(model, data, state, spec)
        return {"available": True, "spec": "mjSTATE_INTEGRATION", "size": size, "state": state}
    except Exception as exc:
        raise RunnerError(f"Native mjSTATE_INTEGRATION capture unavailable: {exc}") from exc


def _restore_native_state(sim: Any, saved: Mapping[str, Any]) -> None:
    if not bool(saved.get("available", False)):
        raise RunnerError("Snapshot has no native mjSTATE_INTEGRATION state.")
    try:
        import mujoco

        model, data = _native_state_handles(sim)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.asarray(saved["state"], dtype=np.float64)
        expected_size = int(mujoco.mj_stateSize(model, spec))
        if state.size != expected_size or int(saved.get("size", state.size)) != expected_size:
            raise RunnerError(f"Native state size changed: {state.size} vs {expected_size}")
        mujoco.mj_setState(model, data, state, spec)
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError(f"Native mjSTATE_INTEGRATION restore failed: {exc}") from exc


def _capture_array_fields(obj: Any, names: Sequence[str], label: str, required: Sequence[str] = ()) -> dict[str, Any]:
    if obj is None:
        raise RunnerError(f"Required state object is missing: {label}")
    required_set = set(required)
    result: dict[str, Any] = {}
    missing: list[str] = []
    for name in names:
        if not hasattr(obj, name):
            if name in required_set:
                missing.append(name)
            continue
        value = getattr(obj, name)
        if callable(value):
            raise RunnerError(f"Required state field is callable: {label}.{name}")
        result[name] = np.asarray(value).copy()
    if missing:
        raise RunnerError(f"Required state fields are missing: {label}.{', '.join(missing)}")
    return result


def _restore_array_fields(obj: Any, fields: Mapping[str, Any], label: str) -> None:
    if obj is None:
        raise RunnerError(f"Required state object is missing: {label}")
    failures: list[str] = []
    for name, saved in fields.items():
        if not hasattr(obj, name):
            failures.append(f"{label}.{name}: attribute missing")
            continue
        try:
            target = np.asarray(getattr(obj, name))
            value = np.asarray(saved, dtype=target.dtype)
            if target.shape != value.shape:
                raise RunnerError(f"shape {target.shape} vs {value.shape}")
            np.copyto(target, value)
        except Exception as exc:
            failures.append(f"{label}.{name}: {exc}")
    if failures:
        raise RunnerError("Array state restore failed: " + "; ".join(failures))


def _capture_model_state(sim: Any) -> dict[str, Any]:
    model = getattr(sim, "model", None)
    # Fixture placement is written into model.body_pos/body_quat during BDDL
    # reset and is not part of mjSTATE_INTEGRATION.  Visual state is included
    # because LIBERO post-processing may update rgba fields.
    return _capture_array_fields(
        model,
        _MODEL_STATE_ATTRS,
        "sim.model",
        required=("body_pos", "body_quat"),
    )


def _restore_model_state(sim: Any, fields: Mapping[str, Any]) -> None:
    _restore_array_fields(getattr(sim, "model", None), fields, "sim.model")


def _capture_sim_fields(sim: Any) -> dict[str, Any]:
    return _capture_array_fields(
        getattr(sim, "data", None),
        _PHYSICS_STATE_ATTRS,
        "sim.data",
        required=("qpos", "qvel", "ctrl"),
    )


def _restore_sim_fields(sim: Any, fields: Mapping[str, Any]) -> None:
    _restore_array_fields(getattr(sim, "data", None), fields, "sim.data")


def _numeric_max_abs(expected: Any, actual: Any) -> float:
    lhs = np.asarray(expected)
    rhs = np.asarray(actual)
    if lhs.shape != rhs.shape:
        raise RunnerError(f"State shape changed: {lhs.shape} vs {rhs.shape}")
    if lhs.size == 0:
        return 0.0
    if np.issubdtype(lhs.dtype, np.number) and np.issubdtype(rhs.dtype, np.number):
        return float(np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))))
    if not np.array_equal(lhs, rhs):
        raise RunnerError("Non-numeric state changed during restore")
    return 0.0


def _verify_array_fields(obj: Any, fields: Mapping[str, Any], label: str, *, atol: float = 0.0) -> float:
    maximum = 0.0
    for name, expected in fields.items():
        if not hasattr(obj, name):
            raise RunnerError(f"Restored state field is missing: {label}.{name}")
        actual = np.asarray(getattr(obj, name))
        error = _numeric_max_abs(expected, actual)
        maximum = max(maximum, error)
        if error > atol:
            raise RunnerError(f"{label}.{name} mismatch after restore: max_abs_error={error}")
    return maximum


def _state_tree_max_abs(expected: Any, actual: Any, label: str = "state") -> float:
    """Compare explicit state leaves without comparing object identities."""
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            raise RunnerError(f"State keys changed at {label}")
        return max((_state_tree_max_abs(expected[k], actual[k], f"{label}.{k}") for k in expected), default=0.0)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            raise RunnerError(f"State length changed at {label}: {len(expected)} vs {len(actual)}")
        return max((_state_tree_max_abs(a, b, f"{label}[{i}]") for i, (a, b) in enumerate(zip(expected, actual))), default=0.0)
    try:
        return _numeric_max_abs(expected, actual)
    except (TypeError, ValueError):
        if expected != actual:
            raise RunnerError(f"State value changed at {label}: {expected!r} vs {actual!r}")
        return 0.0


def _verify_contacts(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    if bool(expected.get("available")) != bool(actual.get("available")):
        raise RunnerError(f"Contact availability changed: {expected} vs {actual}")
    if not bool(expected.get("available")):
        raise RunnerError("Contact snapshot is unavailable; refusing to score restore")
    if int(expected.get("count", -1)) != int(actual.get("count", -1)):
        raise RunnerError(f"Contact count changed: {expected.get('count')} vs {actual.get('count')}")
    if expected.get("pairs") != actual.get("pairs"):
        raise RunnerError(f"Contact geom pairs changed: {expected.get('pairs')} vs {actual.get('pairs')}")


def _capture_snapshot(
    env: Any,
    obs: Mapping[str, Any],
    *,
    task_id: int,
    episode: int,
    t: int,
    replan_idx: int,
    action_prefix: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    sim = getattr(env, "sim", None)
    if sim is None or not hasattr(sim, "get_state"):
        raise RunnerError("LIBERO environment does not expose sim.get_state().")
    sim_state = sim.get_state()
    if hasattr(sim_state, "flatten"):
        sim_flat = np.asarray(sim_state.flatten(), dtype=np.float64).copy()
    else:
        sim_flat = np.asarray(sim_state, dtype=np.float64).copy()
    return {
        "schema": 3,
        "task_id": int(task_id),
        "episode": int(episode),
        "t": int(t),
        "replan_idx": int(replan_idx),
        "sim_flat": sim_flat,
        "native_state": _capture_native_state(sim),
        "model_arrays": _capture_model_state(sim),
        "sim_fields": _capture_sim_fields(sim),
        "env_mutable": _capture_env_mutable(env),
        "rng": _capture_rng(),
        "obs": _copy_obs(obs),
        "images": _image_obs(obs, _RUNTIME.ev),
        "physics": _capture_physics(env),
        "contacts": _capture_contacts(env),
        "action_prefix": [] if action_prefix is None else _copy_state_value(action_prefix),
        "pending_actions": [],
        "controller_mode": "reference_action_chunk",
    }


def _restore_snapshot(env: Any, snapshot: Mapping[str, Any]) -> dict[str, np.ndarray]:
    sim = getattr(env, "sim", None)
    try:
        if "native_state" not in snapshot or "model_arrays" not in snapshot or "sim_fields" not in snapshot:
            raise RunnerError("Snapshot lacks native/model/physics state; recollect with schema 3.")
        _restore_model_state(sim, snapshot["model_arrays"])
        _restore_native_state(sim, snapshot["native_state"])
        # Native integration state does not include ctrl and external forces.
        _restore_sim_fields(sim, snapshot["sim_fields"])
        sim.forward()
        # Forward may update derived acceleration / warmstart values; put all
        # explicitly captured control and solver inputs back afterward.
        _restore_sim_fields(sim, snapshot["sim_fields"])
        # robosuite observations are sampled/cached inside physics stepping.
        # Regenerating them here advances sensor bookkeeping and changes the
        # policy input (confirmed by CPU preflight: up to 23 pixel levels).
        # Restore the actual sampled observations and their clocks unchanged;
        # original-vs-replayed NEXT-step functional equality is checked by the
        # CPU preflight and by scoring against the untouched collected trace.
        core = _env_core(env)
        _restore_env_mutable(env, snapshot["env_mutable"], restore_observables=True, restore_obs_cache=True)
        if hasattr(core, "_get_observations"):
            restored_obs = core._get_observations()
        else:
            restored_obs = env._get_observations()
        max_error = _obs_max_abs_error(snapshot["obs"], restored_obs)
        if max_error > 1.0e-4:
            raise RunnerError(f"Observation mismatch after bookkeeping restore: max_abs_error={max_error}")
        _verify_contacts(snapshot["contacts"], _capture_contacts(env))
        _verify_array_fields(getattr(sim, "model", None), snapshot["model_arrays"], "sim.model", atol=0.0)
        _verify_array_fields(getattr(sim, "data", None), snapshot["sim_fields"], "sim.data", atol=0.0)
        native_now = _capture_native_state(sim)
        native_expected = np.asarray(snapshot["native_state"]["state"], dtype=np.float64)
        if not np.array_equal(native_expected, np.asarray(native_now["state"], dtype=np.float64)):
            raise RunnerError("Native mjSTATE_INTEGRATION state is not exact after restore")
        flat_now = np.asarray(sim.get_state().flatten(), dtype=np.float64)
        if not np.array_equal(np.asarray(snapshot["sim_flat"], dtype=np.float64), flat_now):
            raise RunnerError("Flattened MuJoCo state is not exact after native restore")
        mutable_now = _capture_env_mutable(env)
        mutable_error = _state_tree_max_abs(snapshot["env_mutable"], mutable_now, "env_mutable")
        if mutable_error > 1.0e-7:
            raise RunnerError(f"Controller/observable mutable state mismatch: max_abs_error={mutable_error}")
        if isinstance(snapshot, MutableMapping):
            snapshot["restore_diagnostics"] = {
                "native_state_exact": True,
                "sim_flat_exact": True,
                "sim_fields_exact": True,
                "model_arrays_exact": True,
                "contacts_exact": True,
                "mutable_max_abs_error": float(mutable_error),
                "observation_max_abs_error": float(max_error),
                "comparison": "exact for copied simulator fields; <=1e-7 for mutable numeric leaves; <=1e-4 for observations",
            }
        _restore_rng(snapshot["rng"])
        return _copy_obs(restored_obs)
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError(f"Simulator/controller restore failed: {exc}") from exc


def _invalidate_model_caches(model: Any) -> None:
    # compile_action_infer is forbidden for this phase.  Clear only transient
    # inference artifacts so ref/repeat/identity branches cannot share state.
    for obj in (model, getattr(model, "mot", None), getattr(model, "video_expert", None), getattr(model, "action_expert", None)):
        if obj is None or not hasattr(obj, "__dict__"):
            continue
        for name in list(vars(obj)):
            if name.endswith("_compiled"):
                delattr(obj, name)


def _resolve_attr(root: Any, path: str) -> Any:
    current = root
    for component in path.split("."):
        if component.isdigit() and isinstance(current, (list, tuple)):
            current = current[int(component)]
        elif component.isdigit() and hasattr(current, "__getitem__"):
            current = current[int(component)]
        else:
            current = getattr(current, component)
    return current


@dataclass(frozen=True)
class Site:
    site_id: str
    family: str
    depth: int
    module_path: str
    role: str
    shape: tuple[int, ...]
    dtype: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "family": self.family,
            "depth": self.depth,
            "module_path": self.module_path,
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "bits": 4,
            "signed_range": [-7, 7],
            "granularity": "per_output_channel",
            "scale": "maxabs/7",
            "rounding": "torch.round ties-to-even",
            "zero_point": 0,
            "mode": "weight_only_fake_quant",
        }


def _build_registry(model: Any) -> tuple[list[Site], dict[str, Any]]:
    import torch.nn as nn

    sites: list[Site] = []
    for family in ("video", "action"):
        expert = getattr(model, f"{family}_expert", None)
        if expert is None or len(getattr(expert, "blocks", [])) < max(DEFAULT_DEPTHS) + 1:
            raise RunnerError(f"Missing {family}_expert.blocks required by registry8.")
        for depth in DEFAULT_DEPTHS:
            module_path = f"{family}_expert.blocks.{depth}.self_attn.q"
            module = _resolve_attr(model, module_path)
            if not isinstance(module, nn.Linear):
                raise RunnerError(f"Registry path is not nn.Linear: {module_path}")
            sites.append(
                Site(
                    site_id=f"{family}_d{depth:02d}_self_attn_q",
                    family=family,
                    depth=depth,
                    module_path=module_path,
                    role="weight",
                    shape=tuple(int(x) for x in module.weight.shape),
                    dtype=str(module.weight.dtype),
                )
            )
    if len(sites) != 8:
        raise RunnerError(f"Expected registry8, got {len(sites)}")
    registry = {
        "revision": 1,
        "sites": [site.as_dict() for site in sites],
        "selection": "self_attn.q at representative block depths 0,10,20,29; 4 video + 4 action",
    }
    return sites, registry


def _fake_quant_w4(weight: Any) -> tuple[Any, Any]:
    import torch

    if weight.ndim != 2:
        raise RunnerError(f"W4 registry currently requires 2D Linear weights, got {tuple(weight.shape)}")
    # Compute in FP32 but write back in the model's actual dtype.  torch.round
    # uses ties-to-even, and clamp makes the declared signed range explicit.
    work = weight.detach().float()
    scale = work.abs().amax(dim=1, keepdim=True) / 7.0
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    q = torch.round(work / scale).clamp(-7.0, 7.0)
    dequant = (q * scale).to(dtype=weight.dtype, device=weight.device)
    return dequant, scale.detach().cpu()


@contextmanager
def _single_site_quantized(model: Any, site: Site):
    module = _resolve_attr(model, site.module_path)
    original = module.weight.detach().clone()
    quantized, scale = _fake_quant_w4(module.weight)
    try:
        with __import__("torch").no_grad():
            module.weight.copy_(quantized)
        yield {
            "scale_shape": list(scale.shape),
            "scale_min": float(scale.min().item()),
            "scale_max": float(scale.max().item()),
        }
    finally:
        with __import__("torch").no_grad():
            module.weight.copy_(original)
        if not __import__("torch").equal(module.weight, original):
            raise RunnerError(f"Original weight was not restored for {site.site_id}")


def _tensor_from_output(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for child in value.values():
            try:
                return _tensor_from_output(child)
            except RunnerError:
                continue
    if isinstance(value, (tuple, list)):
        for child in value:
            try:
                return _tensor_from_output(child)
            except RunnerError:
                continue
    raise RunnerError(f"Forward hook output contains no tensor: {type(value)!r}")


class HookRecorder:
    """Collect reference outputs or compare one branch online, never persist tensors."""

    def __init__(self, model: Any, sites: Sequence[Site], reference: Mapping[str, Sequence[np.ndarray]] | None = None):
        self.model = model
        self.sites = list(sites)
        self.reference = reference
        self.outputs: dict[str, list[np.ndarray]] = {site.site_id: [] for site in self.sites}
        self.counts: dict[str, int] = {site.site_id: 0 for site in self.sites}
        self.mse_sum: dict[str, float] = {site.site_id: 0.0 for site in self.sites}
        self.mse_numel: dict[str, int] = {site.site_id: 0 for site in self.sites}
        self.handles: list[Any] = []

    def __enter__(self) -> "HookRecorder":
        for site in self.sites:
            module = _resolve_attr(self.model, site.module_path)

            def hook(_module: Any, _inputs: Any, output: Any, *, _site: Site = site) -> None:
                tensor = _tensor_from_output(output).detach().float()
                index = self.counts[_site.site_id]
                self.counts[_site.site_id] += 1
                if self.reference is None:
                    self.outputs[_site.site_id].append(tensor.cpu().numpy().copy())
                    return
                expected = self.reference[_site.site_id]
                if index >= len(expected):
                    raise RunnerError(f"Extra hook invocation for {_site.site_id}")
                ref = np.asarray(expected[index], dtype=np.float32)
                actual = tensor.cpu().numpy()
                if ref.shape != actual.shape:
                    raise RunnerError(
                        f"Hook shape mismatch at {_site.site_id} invocation {index}: {ref.shape} vs {actual.shape}"
                    )
                diff = actual - ref
                self.mse_sum[_site.site_id] += float(np.sum(diff * diff, dtype=np.float64))
                self.mse_numel[_site.site_id] += int(diff.size)
        
            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if exc is None and self.reference is not None:
            for site in self.sites:
                expected_count = len(self.reference[site.site_id])
                actual_count = self.counts[site.site_id]
                if actual_count != expected_count:
                    raise RunnerError(
                        f"Hook invocation count mismatch for {site.site_id}: {actual_count} vs {expected_count}"
                    )

    def reference_outputs(self) -> dict[str, list[np.ndarray]]:
        if self.reference is not None:
            raise RunnerError("Reference outputs requested from a comparison recorder.")
        return self.outputs

    def local_mse(self, site_id: str) -> float | None:
        count = self.mse_numel[site_id]
        return None if count == 0 else self.mse_sum[site_id] / float(count)


def _reference_hook_mean_square(reference_outputs: Mapping[str, Sequence[np.ndarray]], site_id: str) -> float:
    values = reference_outputs.get(site_id, ())
    if not values:
        return EPS * EPS
    total = 0.0
    numel = 0
    for value in values:
        array = np.asarray(value, dtype=np.float32)
        total += float(np.sum(array * array, dtype=np.float64))
        numel += int(array.size)
    return max(total / float(max(numel, 1)), EPS * EPS)


def _invalidate_before_infer(model: Any) -> None:
    _invalidate_model_caches(model)
    if bool(getattr(model, "compile_training_denoise", False)):
        raise RunnerError("compile_training_denoise unexpectedly enabled.")


def _timed_infer(runtime: "Runtime", env: Any, obs: Mapping[str, Any], recorder: HookRecorder | None) -> dict[str, Any]:
    import torch

    _invalidate_before_infer(runtime.model)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        context = recorder if recorder is not None else _NullContext()
        with context:
            action, _imgs, predicted = runtime.ev._predict_action_chunk(
                obs=obs,
                task_description=runtime.task_description,
                model=runtime.model,
                processor=runtime.processor,
                cfg=runtime.cfg,
                action_horizon=runtime.action_horizon,
                input_w=runtime.input_w,
                input_h=runtime.input_h,
                model_device=runtime.model_device,
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "action": np.asarray(action, dtype=np.float32).copy(),
        "predicted_future_frames": predicted,
        "elapsed_s": float(elapsed),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _new_branch(
    runtime: "Runtime",
    snapshot: Mapping[str, Any],
    *,
    site: Site | None = None,
    reference_outputs: Mapping[str, Sequence[np.ndarray]] | None = None,
    identity_hook: bool = False,
    capture_hooks: bool = True,
) -> dict[str, Any]:
    import torch

    obs = _restore_snapshot(runtime.env, snapshot)
    _restore_rng(snapshot["rng"])
    torch.cuda.reset_peak_memory_stats()
    if identity_hook and reference_outputs is None:
        raise RunnerError("identity_hook requires reference hook outputs for online comparison.")
    recorder = HookRecorder(runtime.model, runtime.sites, reference_outputs) if capture_hooks and (site is not None or identity_hook or reference_outputs is None) else None
    quant_meta: dict[str, Any] | None = None
    try:
        if site is None:
            result = _timed_infer(runtime, runtime.env, obs, recorder)
        else:
            with _single_site_quantized(runtime.model, site) as quant_meta:
                result = _timed_infer(runtime, runtime.env, obs, recorder)
        action = result["action"]
        if action.ndim != 2 or action.shape[1] != 7:
            raise RunnerError(f"Unexpected action shape: {action.shape}")
        next_obs, reward, done, info = runtime.env.step(action[0].tolist())
        next_images = _image_obs(next_obs, runtime.ev)
        result.update(
            {
                "next_obs": _copy_obs(next_obs),
                "next_images": next_images,
                "next_physics": _capture_physics(runtime.env),
                "next_contacts": _filter_task_contacts(_capture_contacts(runtime.env)),
                "reward": float(reward),
                "done": bool(done),
                "info": _copy_state_value(info) if isinstance(info, Mapping) else str(info),
                "quant_meta": quant_meta,
                "hook_counts": dict(recorder.counts) if recorder is not None else {},
                "hook_local_mse": (
                    {site_id: recorder.local_mse(site_id) for site_id in recorder.counts}
                    if recorder is not None and reference_outputs is not None
                    else {}
                ),
                "reference_hook_outputs": (
                    recorder.reference_outputs() if recorder is not None and reference_outputs is None else None
                ),
            }
        )
        return result
    finally:
        # The context manager restores a quantized weight even on inference
        # errors.  Clear transient compile/cache attributes before the next branch.
        _invalidate_model_caches(runtime.model)


@dataclass
class Runtime:
    base: Path
    fastwam_root: Path
    cfg: Any
    model: Any
    processor: Any
    ev: Any
    env: Any
    task_description: str
    task_id: int
    action_horizon: int
    input_w: int
    input_h: int
    model_device: str
    sites: list[Site]
    registry: dict[str, Any]
    runtime_manifest: dict[str, Any]


_RUNTIME: Runtime | None = None


def _make_runtime(args: argparse.Namespace) -> Runtime:
    global _RUNTIME
    allocation = _require_pbs_allocation()
    gpu = _verify_gpu()
    base = _expand_path(args.base)
    fastwam_root = base / "FastWAM"
    if not fastwam_root.is_dir():
        raise RunnerError(f"Fast-WAM root is missing: {fastwam_root}")
    identity_gate = _expand_path(args.out) / "identity_check.json"
    revision = _git_revision(fastwam_root, identity_gate=identity_gate)
    checkpoint = _expand_path(args.checkpoint or (base / "checkpoints" / DEFAULT_CHECKPOINT_NAME))
    stats_path = _expand_path(args.stats or (base / "checkpoints" / DEFAULT_STATS_NAME))
    checkpoint_meta = _checkpoint_identity(checkpoint)
    if not stats_path.is_file():
        raise RunnerError(f"Dataset stats are missing: {stats_path}")
    if args.mixed_precision != "bf16":
        raise RunnerError("Phase 1 is pinned to the official BF16 reference config; no FP16 fallback.")
    if args.compile_action_infer:
        raise RunnerError("compile_action_infer must remain false for replayable Phase 1.")

    for path in (fastwam_root, fastwam_root / "src", fastwam_root / "experiments" / "libero"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.chdir(fastwam_root)
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from libero.libero import benchmark
    from fastwam.utils.pytorch_utils import set_global_seed
    from experiments.libero import eval_libero_single as ev

    with initialize_config_dir(version_base="1.3", config_dir=str(fastwam_root / "configs")):
        overrides = [
            f"task={DEFAULT_TASK_CONFIG}",
            f"ckpt={checkpoint}",
            "mixed_precision=bf16",
            "EVALUATION.task_suite_name=libero_goal",
            f"EVALUATION.task_id={int(args.task_id)}",
            "EVALUATION.num_trials=4",
            "EVALUATION.replan_steps=10",
            "EVALUATION.num_steps_wait=30",
            "EVALUATION.sigma_shift=1.0",
            "EVALUATION.compile_action_infer=false",
            "+EVALUATION.action_infer_mode=idm",
            "EVALUATION.use_action_ensembler=false",
            "EVALUATION.visualize_future_video=false",
            "EVALUATION.device=cuda",
            f"EVALUATION.dataset_stats_path={stats_path}",
        ]
        cfg = compose(config_name="sim_libero", overrides=overrides)
    if cfg.EVALUATION.action_infer_mode != "idm" or float(cfg.EVALUATION.sigma_shift) != 1.0:
        raise RunnerError("Hydra configuration did not resolve to IDM sigma_shift=1.0.")
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(checkpoint))
    model = model.to("cuda").eval()
    if getattr(model, "torch_dtype", None) != torch.bfloat16:
        raise RunnerError(f"Loaded model dtype is {getattr(model, 'torch_dtype', None)}, expected torch.bfloat16")
    dataset_stats = load_dataset_stats_from_json(str(stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    task_suite = benchmark.get_benchmark_dict()[DEFAULT_SUITE]()
    task = task_suite.get_task(int(args.task_id))
    env, task_description = ev.get_libero_env(task, ev.LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    action_horizon = int(cfg.EVALUATION.action_horizon or (int(cfg.data.train.num_frames) - 1))
    video_size = cfg.data.train.video_size
    input_h, input_w = int(video_size[0]), int(video_size[1])
    _RUNTIME = Runtime(
        base=base,
        fastwam_root=fastwam_root,
        cfg=cfg,
        model=model,
        processor=processor,
        ev=ev,
        env=env,
        task_description=str(task_description),
        task_id=int(args.task_id),
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device="cuda",
        sites=[],
        registry={},
        runtime_manifest={
            "schema": 2,
            "allocation": allocation,
            "gpu": gpu,
            "fastwam_revision": revision,
            "identity_gate": str(identity_gate) if identity_gate.is_file() else None,
            "checkpoint": checkpoint_meta,
            "checkpoint_config": DEFAULT_TASK_CONFIG,
            "mode": "idm",
            "sigma_shift": 1.0,
            "compile_action_infer": False,
            "reference_dtype": str(model.torch_dtype),
            "reference_precision_note": "actual BF16 runtime; no FP16 relabeling",
            "renderer": "OSMesa CPU",
            "suite": DEFAULT_SUITE,
            "task_id": int(args.task_id),
            "task_description": str(task_description),
            "action_horizon": action_horizon,
            "replan_steps": DEFAULT_REPLAN_STEPS,
            "num_steps_wait": DEFAULT_NUM_STEPS_WAIT,
            "max_steps": DEFAULT_MAX_STEPS,
            "dataset_stats_path": str(stats_path),
        },
    )
    sites, registry = _build_registry(model)
    _RUNTIME.sites = sites
    _RUNTIME.registry = registry
    _RUNTIME.runtime_manifest["site_registry"] = registry
    return _RUNTIME


def _load_initial_state(runtime: Runtime, episode: int) -> Any:
    import torch

    task_suite = runtime.ev.benchmark.get_benchmark_dict()[DEFAULT_SUITE]() if hasattr(runtime.ev, "benchmark") else None
    if task_suite is None:
        from libero.libero import benchmark

        task_suite = benchmark.get_benchmark_dict()[DEFAULT_SUITE]()
    task = task_suite.get_task(runtime.task_id)
    path = Path(runtime.ev.get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    states = torch.load(path, weights_only=False)
    if episode < 0 or episode >= len(states):
        raise RunnerError(f"Episode {episode} is outside initial-state file ({len(states)} states).")
    return states[episode]


def _start_episode(runtime: Runtime, episode: int, *, wait_steps: int = 0) -> tuple[dict[str, np.ndarray], Any]:
    initial_state = _load_initial_state(runtime, episode)
    runtime.env.reset()
    obs = runtime.env.set_init_state(initial_state)
    for _ in range(int(wait_steps)):
        obs, _reward, done, _info = runtime.env.step(runtime.ev.get_libero_dummy_action())
        if done:
            raise RunnerError(f"Episode terminated during {wait_steps}-step reference warmup.")
    return _copy_obs(obs), initial_state


def _probe(args: argparse.Namespace) -> dict[str, Any]:
    runtime = _make_runtime(args)
    obs, initial_state = _start_episode(runtime, int(args.episode), wait_steps=DEFAULT_NUM_STEPS_WAIT)
    snapshot = _capture_snapshot(runtime.env, obs, task_id=args.task_id, episode=args.episode, t=DEFAULT_NUM_STEPS_WAIT, replan_idx=0)
    snapshot["task_contacts"] = _filter_task_contacts(snapshot["contacts"])
    import torch

    torch.cuda.reset_peak_memory_stats()
    reference = _new_branch(runtime, snapshot)
    reference_outputs = reference.pop("reference_hook_outputs")
    repeat = _new_branch(runtime, snapshot, reference_outputs=reference_outputs)
    identity = _new_branch(runtime, snapshot, reference_outputs=reference_outputs, identity_hook=True)
    quant_site = runtime.sites[0]
    quant = _new_branch(runtime, snapshot, site=quant_site, reference_outputs=reference_outputs)

    def action_max(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> float:
        return float(np.max(np.abs(np.asarray(lhs["action"]) - np.asarray(rhs["action"]))))

    def image_l1(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> float:
        vals = []
        for key in ("image", "wrist_image"):
            vals.append(float(np.mean(np.abs(lhs["next_images"][key].astype(np.float32) - rhs["next_images"][key].astype(np.float32))) / 255.0))
        return float(np.mean(vals))

    result = {
        "schema": 2,
        "mode": "probe",
        "runtime": runtime.runtime_manifest,
        "registry": runtime.registry,
        "state": {k: v for k, v in snapshot.items() if k not in {"env_mutable", "rng"}},
        "reference": {
            "action_chunk": reference["action"],
            "next_images": reference["next_images"],
            "elapsed_s": reference["elapsed_s"],
            "peak_memory_bytes": reference["peak_memory_bytes"],
            "done": reference["done"],
            "hook_counts": reference["hook_counts"],
        },
        "repeat": {
            "action_max_abs": action_max(reference, repeat),
            "next_obs_l1": image_l1(reference, repeat),
            "elapsed_s": repeat["elapsed_s"],
            "peak_memory_bytes": repeat["peak_memory_bytes"],
            "done": repeat["done"],
        },
        "identity_hook": {
            "action_max_abs": action_max(reference, identity),
            "next_obs_l1": image_l1(reference, identity),
            "hook_counts": identity["hook_counts"],
        },
        "single_site": {
            "site_id": quant_site.site_id,
            "action_chunk_mse": float(np.mean((quant["action"] - reference["action"]) ** 2)),
            "first_action_mse": float(np.mean((quant["action"][0] - reference["action"][0]) ** 2)),
            "next_obs_l1": image_l1(reference, quant),
            "local_hook_mse": quant["hook_local_mse"].get(quant_site.site_id),
            "hook_counts": quant["hook_counts"],
            "elapsed_s": quant["elapsed_s"],
            "peak_memory_bytes": quant["peak_memory_bytes"],
            "quant_meta": quant["quant_meta"],
            "done": quant["done"],
        },
        "raw": {
            "reference_action": reference["action"],
            "repeat_action": repeat["action"],
            "identity_action": identity["action"],
            "quant_action": quant["action"],
            "reference_next_images": reference["next_images"],
            "quant_next_images": quant["next_images"],
            "reference_next_physics": reference["next_physics"],
            "quant_next_physics": quant["next_physics"],
            "reference_contacts": reference["next_contacts"],
            "quant_contacts": quant["next_contacts"],
        },
        "status": "ok",
    }
    out = _expand_path(args.out)
    _json_dump(out / f"probe_task{args.task_id}_episode{args.episode}.json", result)
    _torch_save(out / f"probe_task{args.task_id}_episode{args.episode}.pt", result)
    _close_runtime(runtime)
    print(json.dumps({"mode": "probe", "status": "ok", "site": quant_site.site_id}, cls=NumpyEncoder))
    return result


def _filter_task_contacts(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Keep task contacts while separating robot-self and persistent support contact."""

    if not bool(raw.get("available")):
        return {
            "available": False,
            "contact_present": None,
            "count": None,
            "pairs": [],
            "raw_count": raw.get("raw_count", raw.get("count")),
            "ignored_persistent_or_self": None,
        }

    robot_words = ("robot", "panda", "gripper", "finger", "hand", "wrist", "forearm")
    support_words = ("table", "floor", "wall", "support", "mount", "stand", "world", "base")

    def text(value: Any) -> str:
        return str(value).lower()

    kept: list[list[Any]] = []
    ignored = 0
    for pair in raw.get("pairs", []):
        if len(pair) != 2:
            ignored += 1
            continue
        lhs, rhs = pair
        ltxt, rtxt = text(lhs), text(rhs)
        lhs_robot = any(word in ltxt for word in robot_words)
        rhs_robot = any(word in rtxt for word in robot_words)
        persistent = any(word in ltxt for word in support_words) or any(word in rtxt for word in support_words)
        if (lhs_robot and rhs_robot) or persistent:
            ignored += 1
            continue
        kept.append([lhs, rhs])
    kept = sorted(kept, key=str)
    return {
        "available": True,
        "contact_present": bool(kept),
        "count": len(kept),
        "pairs": kept,
        "raw_count": raw.get("raw_count", raw.get("count")),
        "ignored_persistent_or_self": ignored,
    }


def _contact_transition(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    if not bool(before.get("available")) or not bool(after.get("available")):
        return False
    return before.get("pairs") != after.get("pairs") or before.get("count") != after.get("count")


def _collect(args: argparse.Namespace) -> dict[str, Any]:
    runtime = _make_runtime(args)
    obs, initial_state = _start_episode(runtime, int(args.episode))
    pending: list[list[float]] = []
    replan_records: list[dict[str, Any]] = []
    t = 0
    replan_idx = 0
    done = False
    limit = DEFAULT_NUM_STEPS_WAIT + DEFAULT_MAX_STEPS
    while t < limit:
        if t < DEFAULT_NUM_STEPS_WAIT:
            obs, _reward, done, _info = runtime.env.step(runtime.ev.get_libero_dummy_action())
            obs = _copy_obs(obs)
            t += 1
            if done:
                break
            continue
        if not pending:
            snapshot = _capture_snapshot(runtime.env, obs, task_id=args.task_id, episode=args.episode, t=t, replan_idx=replan_idx)
            snapshot["task_contacts"] = _filter_task_contacts(snapshot["contacts"])
            before_contacts = snapshot["task_contacts"]
            # Reference trace is measured before any restore, so later scoring
            # must reproduce an untouched trajectory rather than two identically
            # corrupted restored branches.
            inference = _timed_infer(runtime, runtime.env, obs, None)
            next_obs, reward, first_done, info = runtime.env.step(inference["action"][0].tolist())
            inference.update({"next_images": _image_obs(next_obs, runtime.ev),
                              "next_physics": _capture_physics(runtime.env),
                              "next_contacts": _filter_task_contacts(_capture_contacts(runtime.env)),
                              "done": bool(first_done)})
            # The native rollout must continue from its post-inference state.
            # _new_branch executes the first action, so restore and replay the
            # complete reference action chunk below to preserve exact order.
            obs = _restore_snapshot(runtime.env, snapshot)
            _restore_rng(snapshot["rng"])
            action_chunk = inference["action"]
            pending = action_chunk[:DEFAULT_REPLAN_STEPS].tolist()
            snapshot["reference_action_chunk"] = action_chunk
            snapshot["inference_elapsed_s"] = inference["elapsed_s"]
            snapshot["inference_peak_memory_bytes"] = inference["peak_memory_bytes"]
            snapshot["reference_next_images"] = inference["next_images"]
            snapshot["reference_next_physics"] = inference["next_physics"]
            snapshot["reference_next_contacts"] = inference["next_contacts"]
            snapshot["contact_present_before_first_action"] = before_contacts.get("contact_present")
            snapshot["contact_present_after_first_action"] = inference["next_contacts"].get("contact_present")
            snapshot["contact_transition_first_action"] = _contact_transition(before_contacts, inference["next_contacts"])
            snapshot["contact_transition"] = snapshot["contact_transition_first_action"]
            snapshot["terminal_after_first_action"] = bool(inference["done"])
            replan_records.append(snapshot)
            replan_idx += 1
        action = pending.pop(0)
        obs, _reward, done, _info = runtime.env.step(action)
        obs = _copy_obs(obs)
        t += 1
        if done:
            break
    # Prefer actual contact transitions, then active contacts, then free
    # states; stable t/replan ordering resolves equal priorities.
    def priority(record: Mapping[str, Any]) -> tuple[int, int, int]:
        transition = bool(record.get("contact_transition", False))
        active = bool(record.get("contact_present_before_first_action", False))
        return (0 if transition else 1 if active else 2, int(record["t"]), int(record["replan_idx"]))

    ordered = sorted(replan_records, key=priority)
    contact_candidates = [r for r in ordered if r.get("contact_present_before_first_action") or r.get("contact_transition_first_action")]
    free_candidates = [r for r in ordered if r.get("contact_present_before_first_action") is False and not r.get("contact_transition_first_action")]
    selected = contact_candidates[:2] + free_candidates[:2]
    selected_ids = {id(r) for r in selected}
    if len(selected) < DEFAULT_NUM_STATES:
        selected.extend(r for r in ordered if id(r) not in selected_ids and len(selected) < DEFAULT_NUM_STATES)
        selected_ids = {id(r) for r in selected}
    selected = selected[:DEFAULT_NUM_STATES]
    for rank, record in enumerate(selected):
        record["selected_rank"] = rank
    if len(selected) < DEFAULT_NUM_STATES:
        raise RunnerError(f"Native rollout yielded only {len(selected)} replan states; need {DEFAULT_NUM_STATES}.")
    metadata = {
        "schema": 2,
        "mode": "collect",
        "runtime": runtime.runtime_manifest,
        "registry": runtime.registry,
        "task_id": int(args.task_id),
        "episode": int(args.episode),
        "initial_state": _copy_state_value(initial_state),
        "replan_count": len(replan_records),
        "rollout_terminal": bool(done),
        "selected_state_ids": [f"task{args.task_id}/episode{args.episode}/replan{r['replan_idx']}" for r in selected],
        "selection": "up to 2 reference contact/first-action transition states plus up to 2 reference free states; stable native order; fill from remaining strata only when unavailable",
        "official_rollout": {
            "replan_steps": DEFAULT_REPLAN_STEPS,
            "num_steps_wait": DEFAULT_NUM_STEPS_WAIT,
            "max_steps": DEFAULT_MAX_STEPS,
            "action_ensembler": False,
            "pending_queue_at_snapshot": [],
        },
        "raw_replan_metadata": [
            {
                "replan_idx": int(r["replan_idx"]),
                "t": int(r["t"]),
                "contact_present_before_first_action": r.get("contact_present_before_first_action"),
                "contact_present_after_first_action": r.get("contact_present_after_first_action"),
                "contact_transition_first_action": bool(r.get("contact_transition_first_action", False)),
                "contact_transition": bool(r.get("contact_transition", False)),
                "contacts": r.get("contacts", {}),
                "terminal_after_first_action": bool(r.get("terminal_after_first_action", False)),
            }
            for r in replan_records
        ],
    }
    out = _expand_path(args.out)
    stem = f"states_task{args.task_id}_episode{args.episode}"
    _torch_save(out / f"{stem}.pt", {"metadata": metadata, "states": selected})
    _json_dump(out / f"{stem}.json", metadata)
    _close_runtime(runtime)
    print(json.dumps({"mode": "collect", "status": "ok", "selected": len(selected), "replans": len(replan_records)}))
    return metadata


def _find_state_file(data_dir: Path, task_id: int, episode: int) -> Path:
    candidates = sorted(data_dir.rglob(f"states_task{task_id}_episode{episode}.pt"))
    if len(candidates) != 1:
        raise RunnerError(f"Expected exactly one collected state file, found {len(candidates)}: {candidates}")
    return candidates[0]


def _calibration_normalizers(data_dir: Path, task_ids: Sequence[int]) -> tuple[float, float, list[str]]:
    action_chunks: list[np.ndarray] = []
    next_images: list[np.ndarray] = []
    files: list[str] = []
    for task_id in task_ids:
        for episode in (0, 1):
            path = _find_state_file(data_dir, task_id, episode)
            payload = __import__("torch").load(path, weights_only=False)
            files.append(str(path))
            for state in payload["states"]:
                action_chunks.append(np.asarray(state["reference_action_chunk"], dtype=np.float32).reshape(-1))
                ims = state["reference_next_images"]
                next_images.append(np.concatenate([np.asarray(ims["image"], dtype=np.float32).reshape(-1), np.asarray(ims["wrist_image"], dtype=np.float32).reshape(-1)]))
    if not action_chunks or not next_images:
        raise RunnerError("CAL normalization requires collected task0/task1 episodes0/1.")
    action_values = np.concatenate(action_chunks)
    sigma_a2 = max(float(np.var(action_values)), EPS * EPS)
    image_values = np.stack(next_images, axis=0) / 255.0
    sigma_o = max(float(np.std(image_values, axis=0).mean()), EPS)
    return sigma_a2, sigma_o, files


def _score(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = _expand_path(args.data_dir)
    collected_path = _find_state_file(data_dir, args.task_id, args.episode)
    payload = __import__("torch").load(collected_path, weights_only=False)
    states = payload["states"]
    if len(states) != DEFAULT_NUM_STATES:
        raise RunnerError(f"Score requires exactly {DEFAULT_NUM_STATES} selected states, got {len(states)}")
    runtime = _make_runtime(args)
    # Initialise the newly-created env before restoring a mid-episode snapshot;
    # this constructs controller/gripper buffers and observable instances.
    _start_episode(runtime, int(args.episode), wait_steps=0)
    sigma_a2, sigma_o, cal_files = _calibration_normalizers(data_dir, (0, 1))
    split = "CAL" if int(args.episode) in (0, 1) else "CHECK"
    rows: list[dict[str, Any]] = []
    site_summaries: dict[str, list[dict[str, Any]]] = {site.site_id: [] for site in runtime.sites}
    for state_index, snapshot in enumerate(states):
        reference = _new_branch(runtime, snapshot)
        reference_outputs = reference.pop("reference_hook_outputs")
        repeat = _new_branch(runtime, snapshot, reference_outputs=reference_outputs)
        identity = _new_branch(runtime, snapshot, reference_outputs=reference_outputs, identity_hook=True)
        collected_action_error = float(np.max(np.abs(reference["action"] - snapshot["reference_action_chunk"])))
        collected_pixel_error = float(np.mean([np.mean(np.abs(reference["next_images"][k].astype(np.float32) - snapshot["reference_next_images"][k].astype(np.float32))) / 255.0 for k in ("image", "wrist_image")]))
        if collected_action_error > 1e-5 or collected_pixel_error > 1e-6:
            raise RunnerError(f"Collected trace not reproduced: action={collected_action_error}, pixels={collected_pixel_error}")
        validation = {
            "collected_reference_action_max_abs": collected_action_error,
            "collected_reference_next_obs_l1": collected_pixel_error,
            "reference_repeat_action_max_abs": float(np.max(np.abs(reference["action"] - repeat["action"]))),
            "reference_identity_action_max_abs": float(np.max(np.abs(reference["action"] - identity["action"]))),
            "reference_repeat_next_obs_l1": float(np.mean([np.mean(np.abs(reference["next_images"][k].astype(np.float32) - repeat["next_images"][k].astype(np.float32))) / 255.0 for k in ("image", "wrist_image")])),
            "reference_identity_next_obs_l1": float(np.mean([np.mean(np.abs(reference["next_images"][k].astype(np.float32) - identity["next_images"][k].astype(np.float32))) / 255.0 for k in ("image", "wrist_image")])),
        }
        if validation["reference_repeat_action_max_abs"] > 1e-5 or validation["reference_identity_action_max_abs"] > 1e-5 or validation["reference_repeat_next_obs_l1"] > 1e-6 or validation["reference_identity_next_obs_l1"] > 1e-6:
            raise RunnerError(f"Reference/identity replay noise exceeds preregistered gate: {validation}")
        for site in runtime.sites:
            try:
                quant = _new_branch(runtime, snapshot, site=site, reference_outputs=reference_outputs)
                action_mse = float(np.mean((quant["action"] - reference["action"]) ** 2))
                first_mse = float(np.mean((quant["action"][0] - reference["action"][0]) ** 2))
                obs_l1 = float(np.mean([np.mean(np.abs(quant["next_images"][k].astype(np.float32) - reference["next_images"][k].astype(np.float32))) / 255.0 for k in ("image", "wrist_image")]))
                local_mse = quant["hook_local_mse"].get(site.site_id)
                local_reference_ms = _reference_hook_mean_square(reference_outputs, site.site_id)
                row = {
                    "status": "ok",
                    "terminal": bool(quant["done"]),
                    "state_index": state_index,
                    "state_id": f"task{args.task_id}/episode{args.episode}/replan{snapshot['replan_idx']}",
                    "task_id": int(args.task_id),
                    "episode": int(args.episode),
                    "split": split,
                    "site_id": site.site_id,
                    "contact_present_before_first_action": snapshot.get("contact_present_before_first_action"),
                    "contact_present_after_first_action": snapshot.get("contact_present_after_first_action"),
                    "contact_transition_first_action": bool(snapshot.get("contact_transition_first_action", False)),
                    "contact_transition": bool(snapshot.get("contact_transition", False)),
                    "action_chunk_mse": action_mse,
                    "first_action_mse": first_mse,
                    "executed_action_l2": float(np.linalg.norm(quant["action"][0] - reference["action"][0])),
                    "next_obs_l1_01": obs_l1,
                    "local_hook_output_mse": local_mse,
                    "local_hook_reference_mean_square": local_reference_ms,
                    "local_hook_output_mse_normalized": None if local_mse is None else float(local_mse / local_reference_ms),
                    "d_a": float(action_mse / max(sigma_a2, EPS * EPS)),
                    "d_o": float(obs_l1 / max(sigma_o, EPS)),
                    "C_r": float(action_mse / max(sigma_a2, EPS * EPS) + obs_l1 / max(sigma_o, EPS)),
                    "hook_counts": quant["hook_counts"],
                    "quant_meta": quant["quant_meta"],
                    "elapsed_s": quant["elapsed_s"],
                    "peak_memory_bytes": quant["peak_memory_bytes"],
                    "ref_next_qpos": reference["next_physics"].get("qpos"),
                    "quant_next_qpos": quant["next_physics"].get("qpos"),
                    "ref_next_qvel": reference["next_physics"].get("qvel"),
                    "quant_next_qvel": quant["next_physics"].get("qvel"),
                    "ref_next_contacts": reference["next_contacts"],
                    "quant_next_contacts": quant["next_contacts"],
                    "contact_transition_quant": _contact_transition(reference["next_contacts"], quant["next_contacts"]),
                    "raw": {
                        "reference_action_chunk": reference["action"],
                        "quant_action_chunk": quant["action"],
                        "reference_first_action": reference["action"][0],
                        "quant_first_action": quant["action"][0],
                        "reference_next_images": reference["next_images"],
                        "quant_next_images": quant["next_images"],
                    },
                    "validation": validation,
                }
            except RunnerError:
                raise
            except Exception as exc:
                # Retain terminal/error rows for audit.  Restoration mismatches
                # are RunnerError and abort above; ordinary branch failures are
                # visible and are never silently dropped.
                row = {
                    "status": "error",
                    "terminal": bool(snapshot.get("terminal_after_first_action", False)),
                    "state_index": state_index,
                    "state_id": f"task{args.task_id}/episode{args.episode}/replan{snapshot['replan_idx']}",
                    "task_id": int(args.task_id),
                    "episode": int(args.episode),
                    "split": split,
                    "site_id": site.site_id,
                    "error": repr(exc),
                    "contact_transition": bool(snapshot.get("contact_transition", False)),
                    "validation": validation,
                }
            rows.append(row)
            site_summaries[site.site_id].append(row)
    summary = {}
    for site_id, site_rows in site_summaries.items():
        valid = [r for r in site_rows if r.get("status") == "ok"]
        summary[site_id] = {
            "n_rows": len(site_rows),
            "n_ok": len(valid),
            "n_terminal": sum(bool(r.get("terminal")) for r in site_rows),
            "mean_action_chunk_mse": None if not valid else float(np.mean([r["action_chunk_mse"] for r in valid])),
            "mean_first_action_mse": None if not valid else float(np.mean([r["first_action_mse"] for r in valid])),
            "mean_local_hook_output_mse": None if not valid else float(np.mean([r["local_hook_output_mse"] for r in valid if r.get("local_hook_output_mse") is not None])),
            "mean_d_a": None if not valid else float(np.mean([r["d_a"] for r in valid])),
            "mean_d_o": None if not valid else float(np.mean([r["d_o"] for r in valid])),
            "mean_C_r": None if not valid else float(np.mean([r["C_r"] for r in valid])),
        }
    result = {
        "schema": 2,
        "mode": "score",
        "runtime": runtime.runtime_manifest,
        "registry": runtime.registry,
        "task_id": int(args.task_id),
        "episode": int(args.episode),
        "split": split,
        "source_states": str(collected_path),
        "normalization": {
            "population": "CAL task0/task1 episodes0/1 selected states only",
            "calibration_files": cal_files,
            "sigma_a2": sigma_a2,
            "sigma_o": sigma_o,
            "epsilon_a": EPS,
            "epsilon_o": EPS,
            "lambda_a": 1.0,
            "lambda_o": 1.0,
        },
        "site_summary": summary,
        "rows": rows,
        "retention_policy": "all selected states and all terminal rows retained; branch errors are explicit rows; replay mismatches abort",
        "status": "ok",
    }
    out = _expand_path(args.out)
    stem = f"score_task{args.task_id}_episode{args.episode}"
    errors = [row for row in rows if row.get("status") != "ok"]
    result["status"] = "partial_failure" if errors else "ok"
    _torch_save(out / f"{stem}.pt", result)
    compact = dict(result)
    compact["rows"] = []
    for row in rows:
        row_copy = dict(row)
        row_copy.pop("raw", None)
        compact["rows"].append(row_copy)
    compact["raw_arrays_artifact"] = f"{stem}.pt"
    _json_dump(out / f"{stem}.json", compact)
    _close_runtime(runtime)
    if errors:
        raise RunnerError(f"{len(errors)} score branches failed; partial artifacts retained")
    print(json.dumps({"mode": "score", "status": "ok", "split": split, "rows": len(rows)}))
    return result


def _close_runtime(runtime: Runtime) -> None:
    try:
        close = getattr(runtime.env, "close", None)
        if close is not None:
            close()
    finally:
        global _RUNTIME
        _RUNTIME = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "collect", "score"), required=True)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16", "no"), default="bf16")
    parser.add_argument("--compile-action-infer", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    _require_pbs_allocation()  # Before score can read any serialized state files.
    if args.mode == "score" and not args.data_dir:
        raise SystemExit("--data-dir is required for score mode")
    try:
        if args.mode == "probe":
            _probe(args)
        elif args.mode == "collect":
            _collect(args)
        else:
            _score(args)
    except Exception as exc:
        LOGGER.error("OTC-PTQ %s failed: %s", args.mode, exc)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
