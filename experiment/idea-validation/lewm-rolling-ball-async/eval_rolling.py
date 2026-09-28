"""Guarded native Rolling Ball evaluator adapter; runtime requires an RTX allocation."""

from __future__ import annotations

import argparse
import builtins
import concurrent.futures
import copy
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import socket
import subprocess
import sys
import time
import traceback
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest


REFLEXBENCH_COMMIT = "8bb931485093c6d98f8729774ad01bf824964e16"
TASK_ID = "RollingBallInterception-Franka-DataCollection-v0"
TASK_PROFILE = "rolling_ball_interception"
CONTROL_DT_S = 0.04
PHYSICS_DT_S = 0.01
DECIMATION = 4
MAX_CONTROL_TICKS = 75
RESET_RENDER_FRAMES = 256
ACTION_SCALE = 0.1
TASK_BUFFER_NAMES = (
    "task_phase",
    "rolling_detected",
    "gate_timer",
    "gate_delay",
    "ball_in_catcher_counter",
    "ramp_exit_pos",
    "predicted_intercept_pos",
    "predicted_intercept_mouth_pos",
    "predicted_intercept_time",
    "ball_lane_y",
)
EVENT_BUFFER_NAMES = (
    "_interval_term_time_left",
    "_reset_term_last_triggered_step_id",
    "_reset_term_last_triggered_once",
)
DELAY_TICKS_DEFAULT = (0, 1, 2, 4, 8, 16)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="Stdlib-only scheduler/hold check")
    parser.add_argument("--reflexbench-root", type=Path)
    parser.add_argument("--asset-mirror", type=Path,
                        help="Local mirror containing the original S3 Assets/Isaac/... paths")
    parser.add_argument("--policy-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-bank", type=Path)
    parser.add_argument("--mode", choices=("pair-smoke", "sync", "fixed-delay", "true-async"))
    parser.add_argument("--pairing-protocol", choices=("strict-rgb-v1", "physical-v2"), default="strict-rgb-v1")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--limit-episodes", type=int, default=None,
                        help="Use a prefix of the frozen fixture bank; e.g. 3 for interface/capability smoke")
    parser.add_argument("--seed-base", type=int, default=2026092700)
    parser.add_argument("--delay-ticks", default=",".join(map(str, DELAY_TICKS_DEFAULT)))
    parser.add_argument("--http-timeout-s", type=float, default=180.0)
    parser.add_argument("--rented-host", action="store_true",
                        help="Explicitly opt into a user-rented AutoDL instance instead of PBS")
    parser.add_argument("--expected-host", default=None,
                        help="Exact Linux hostname supplied for this rented instance")
    parser.add_argument("--instance-id", default=None,
                        help="User-supplied rented instance identity, recorded in results")
    args = parser.parse_args(argv)
    if args.self_check:
        return args
    if args.reflexbench_root is None or args.output is None or args.mode is None:
        parser.error("--reflexbench-root, --output, and --mode are required")
    if args.num_episodes <= 0 or (args.limit_episodes is not None and args.limit_episodes <= 0):
        parser.error("episode counts must be positive")
    if args.http_timeout_s <= 0:
        parser.error("--http-timeout-s must be positive")
    if args.rented_host and not args.expected_host:
        parser.error("--rented-host requires the user-supplied exact --expected-host")
    if not args.rented_host and (args.expected_host or args.instance_id):
        parser.error("rented-host identity flags require --rented-host")
    try:
        delays = tuple(int(value.strip()) for value in args.delay_ticks.split(",") if value.strip())
    except ValueError:
        parser.error("--delay-ticks must be comma-separated nonnegative integers")
    if not delays or any(tick < 0 or tick > MAX_CONTROL_TICKS for tick in delays):
        parser.error(f"delay ticks must be in [0,{MAX_CONTROL_TICKS}]")
    if args.mode == "fixed-delay" and delays != DELAY_TICKS_DEFAULT:
        parser.error(f"fixed-delay stage is frozen to {DELAY_TICKS_DEFAULT}")
    args.delay_ticks_parsed = delays
    host = urlparse.urlparse(args.policy_url).hostname
    if host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("policy server must be on loopback; remote HTTP policy endpoints are not accepted")
    if args.mode == "pair-smoke" and args.limit_episodes is not None:
        parser.error("pair-smoke uses all --num-episodes fixtures; do not truncate the fixture bank")
    if args.mode != "pair-smoke" and args.fixture_bank is None:
        parser.error("evaluation stages require --fixture-bank from a passing pair-smoke")
    return args


def fixed_delay_decisions(delay_ticks: int) -> list[str]:
    """Yield the exact action schedule: K holds followed by one new target."""
    if delay_ticks < 0:
        raise ValueError("delay_ticks must be nonnegative")
    return ["hold"] * delay_ticks + ["new_target"]


def absolute_joint_to_relative(target: tuple[float, ...], current: tuple[float, ...],
                               scale: float = ACTION_SCALE) -> tuple[float, ...]:
    if len(target) != 8 or len(current) != 7 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("expected an 8-D absolute target, 7 current joints, and positive scale")
    return tuple((target[i] - current[i]) / scale for i in range(7)) + (1.0 if target[7] > 0.5 else -1.0,)


def self_check() -> dict[str, Any]:
    # K=0 is the same action schedule as synchronous inference; K>0 injects
    # exactly K env.step() holds, independent of HTTP/planner wall latency.
    assert fixed_delay_decisions(0) == ["new_target"]
    assert fixed_delay_decisions(3) == ["hold", "hold", "hold", "new_target"]
    current_a = (0.0,) * 7
    current_b = (0.01,) + (0.0,) * 6
    target = (0.02,) + (0.0,) * 6 + (1.0,)
    assert abs(absolute_joint_to_relative(target, current_a)[0] - 0.2) < 1e-7
    assert abs(absolute_joint_to_relative(target, current_b)[0] - 0.1) < 1e-7
    assert absolute_joint_to_relative(target, current_a)[-1] == 1.0
    mirror = Path("asset-mirror").resolve()
    original_uri = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Mugs/SM_Mug_A2.usd"
    assert mirrored_asset_path(original_uri, mirror) == mirror / "Assets/Isaac/5.1/Isaac/Props/Mugs/SM_Mug_A2.usd"
    for bad_uri in (original_uri.replace("5.1", ".."), original_uri.replace("omniverse-content-production", "other-bucket")):
        try:
            mirrored_asset_path(bad_uri, mirror)
        except ValueError:
            pass
        else:
            raise AssertionError("Untrusted asset path was accepted")
    run_episode_checks = _self_check_run_episode()
    rows = {"scene_state": {"pass": True}, "fixed_cam_rgb": {"pass": False}}
    assert not pairing_gate(rows, "strict-rgb-v1")
    assert pairing_gate(rows, "physical-v2")
    rows["scene_state"]["pass"] = False
    assert not pairing_gate(rows, "physical-v2")
    return {"status": "PASS", "checks": [
        "fixed_delay_K_exact", "K0_sync_equivalence", "hold_recomputes_relative_target",
        "native_asset_uri_mapping_and_path_guard",
        "physical_pairing_never_accepts_failed_state_and_preserves_strict_rgb_result",
        *run_episode_checks,
    ]}


def _self_check_run_episode() -> list[str]:
    """Exercise scheduler branches with stdlib fakes; no simulator/framework imports."""
    from types import SimpleNamespace

    saved = {name: globals()[name] for name in
             ("restore_fixture", "reset_server", "hold_target", "environment_step", "_request_for_tick")}

    class FakeNative:
        def __init__(self, *, terminate_after_any_step: bool = False):
            self.results = []
            self.trace = []
            self.step_count = 0
            self.applied_count = 0
            self.terminate_after_any_step = terminate_after_any_step

        def _start_rtf_timer(self):
            pass

    def request_factory(native, args, obs, tick, records, np, executor=None):
        index = len(records)
        target_value = float(index + 1)
        target = [target_value] * 7 + [1.0]
        record = {
            "request_index": index,
            "capture_start_monotonic_s": time.monotonic(),
            "capture_control_tick": tick,
            "response_arrival_wall_ns": 333_000 + index,
            "deadline_missed": False,
            "status": "pending",
            "first_action_apply_wall_ns": None,
            "first_action_apply_sim_s": None,
            "first_action_apply_control_tick": None,
            "observation_age_wall_s": None,
            "observation_age_sim_s": None,
        }
        response = {
            "actions": [[target]],
            "latency_s": 0.123,
            "cem_latency_s": 0.111,
            "request_id": f"fake-{index}",
            "server_received_unix_ns": 777_000 + index,
            "server_completed_unix_ns": 999_000 + index,
        }
        records.append(record)
        if executor is None:
            return {"record": record, "result": response, "future": None}

        class FakeFuture:
            def done(self):
                return not native.terminate_after_any_step and native.step_count >= 1

            def result(self):
                return response, {
                    "response_arrival_wall_ns": 333_000 + index,
                    "response_arrival_monotonic_s": time.monotonic(),
                    "client_http_latency_s": 0.123,
                    "request_start_wall_ns": 222_000 + index,
                    "request_start_monotonic_s": time.monotonic() - 0.123,
                }

        return {"record": record, "result": None, "future": FakeFuture()}

    def fake_environment_step(native, target, np, *, tick, apply_record=None, before_process=None):
        native.trace.append((float(target[0]), apply_record is not None))
        if apply_record is not None:
            mark_applied(apply_record, target, tick)
            native.applied_count += 1
        native.step_count += 1
        done = native.terminate_after_any_step or native.applied_count >= 1
        if done and before_process is not None:
            before_process()
        if done:
            native.results.append({
                "episode": 0, "success": False, "length": native.step_count,
                "max_phase": 0, "terminated_phase": 0, "end_reason": "terminated",
            })
        now = time.monotonic()
        return None, done, {
            "terminal_step_monotonic_s": now if done else None,
            "terminal_post_step_cleanup_wall_s": 0.001 if done else 0.0,
        }

    globals()["restore_fixture"] = lambda native, fixture, torch, np: (None, {"pass": True})
    globals()["reset_server"] = lambda *args, **kwargs: None
    globals()["hold_target"] = lambda native, np: [-1.0] * 7 + [0.0]
    globals()["environment_step"] = fake_environment_step
    globals()["_request_for_tick"] = request_factory
    try:
        args = SimpleNamespace(policy_url="http://127.0.0.1:8000", http_timeout_s=1.0)
        sync_native = FakeNative()
        fixed0_native = FakeNative()
        sync_result = run_episode(sync_native, args, {"seed": 0}, "sync", None, None, None)
        fixed0_result = run_episode(fixed0_native, args, {"seed": 0}, "fixed-delay", 0, None, None)
        assert sync_native.trace == fixed0_native.trace == [(1.0, True)]
        assert sync_result["requests"][0]["response_arrival_wall_ns"] == 333_000
        assert sync_result["requests"][0]["server_completed_unix_ns"] == 999_000

        fixed2_native = FakeNative()
        run_episode(fixed2_native, args, {"seed": 0}, "fixed-delay", 2, None, None)
        assert fixed2_native.trace == [(-1.0, False), (-1.0, False), (1.0, True)]

        async_native = FakeNative()
        async_result = run_episode(async_native, args, {"seed": 0}, "true-async", None, None, None,
                                   executor=object())
        async_record = async_result["requests"][0]
        assert async_native.trace == [(-1.0, False), (1.0, True)]
        assert async_native.step_count == 2
        assert async_record["status"] == "applied"
        assert async_record["first_action_apply_wall_ns"] is not None
        assert async_record["observation_age_sim_s"] == CONTROL_DT_S
        assert async_record["response_arrival_wall_ns"] == 333_000
        assert async_record["server_completed_unix_ns"] == 999_000

        terminal_native = FakeNative(terminate_after_any_step=True)
        terminal_result = run_episode(terminal_native, args, {"seed": 0}, "true-async", None, None, None,
                                      executor=object())
        terminal_record = terminal_result["requests"][0]
        assert terminal_native.step_count == 1
        assert terminal_record["status"] == "dropped"
        assert terminal_record["first_action_apply_wall_ns"] is None
        assert terminal_record["response_arrival_wall_ns"] == 333_000
        return ["run_episode_K0_trace_equivalence", "run_episode_K2_exact_holds",
                "true_async_advances_while_pending_and_applies_at_boundary",
                "terminal_pending_future_drained_and_marked_dropped",
                "client_arrival_and_server_completion_timestamps_separate"]
    finally:
        globals().update(saved)


def require_runtime_host(args: argparse.Namespace) -> dict[str, str]:
    actual_host = socket.gethostname().strip()
    host = actual_host.lower()
    host_short = host.split(".")[0]
    if any(token in host for token in ("login", "submit", "head", "localhost")):
        raise RuntimeError(f"Refusing control/login host: {actual_host}")
    if args.rented_host:
        if sys.platform != "linux" or platform.system() != "Linux":
            raise RuntimeError("AutoDL mode requires the actual rented Linux host")
        if os.environ.get("PBS_JOBID") or os.environ.get("PBS_NODEFILE"):
            raise RuntimeError("Rented-host mode refuses inherited PBS allocation variables")
        expected = args.expected_host.strip()
        if actual_host != expected:
            raise RuntimeError(f"Actual host {actual_host!r} does not exactly match supplied host {expected!r}")
        mode = "autodl"
        job_id = ""
    else:
        job_id = os.environ.get("PBS_JOBID", "").strip()
        nodefile_path = os.environ.get("PBS_NODEFILE", "").strip()
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not job_id or not nodefile_path or not visible_devices or visible_devices.lower() in ("-1", "none"):
            raise RuntimeError("PBS_JOBID, PBS_NODEFILE, and allocated CUDA_VISIBLE_DEVICES are required")
        nodefile = Path(nodefile_path)
        if not nodefile.is_file():
            raise RuntimeError("PBS_NODEFILE is not a readable allocation file")
        allocated = {line.strip().split(".")[0].lower()
                     for line in nodefile.read_text(encoding="utf-8").splitlines() if line.strip()}
        if host_short not in allocated:
            raise RuntimeError(f"Actual host {actual_host!r} is absent from PBS_NODEFILE")
        mode = "pbs"
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    gpus = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if not gpus or not any("RTX" in name.upper() for name in gpus):
        raise RuntimeError(f"Isaac RGB evaluation requires an RTX GPU with RT cores; visible names: {gpus}")
    if args.rented_host and not any("4090" in name for name in gpus):
        raise RuntimeError(f"Rented-host route requires RTX 4090; visible names: {gpus}")
    return {"execution_environment": mode, "host": actual_host, "job_id": job_id,
            "gpu_names": "; ".join(gpus), "instance_id": args.instance_id or ""}


def verify_reflexbench_source(root: Path) -> dict[str, str]:
    """Require source identity evidence for the exact evaluator/task revision."""
    git_marker = root / ".git"
    if git_marker.exists():
        probe = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        commit = probe.stdout.strip()
        if commit != REFLEXBENCH_COMMIT:
            raise RuntimeError(f"ReflexBench git HEAD {commit!r} differs from frozen pin {REFLEXBENCH_COMMIT}")
        return {"identity_source": "git HEAD", "commit": commit}

    pin_path = root / "PINNED.json"
    if pin_path.is_file():
        pin = json.loads(pin_path.read_text(encoding="utf-8"))
        if pin.get("repository") != "LxRoboticsLab/ReflexBench" or pin.get("commit") != REFLEXBENCH_COMMIT:
            raise RuntimeError(f"ReflexBench PINNED.json differs from frozen identity: {pin_path}")
        return {"identity_source": str(pin_path), "commit": pin["commit"]}

    # The bounded PBS source prep stores its completed manifest beside source/.
    prep_path = root.parent / "preparation.json"
    if prep_path.is_file():
        prep = json.loads(prep_path.read_text(encoding="utf-8"))
        if (prep.get("status") == "PASS"
                and prep.get("reflexbench_commit") == REFLEXBENCH_COMMIT
                and "scripts/evaluation/eval.py" in prep.get("source_files", [])):
            return {"identity_source": str(prep_path), "commit": prep["reflexbench_commit"]}
    raise RuntimeError("ReflexBench identity is unverified; require pinned git HEAD, PINNED.json, or passing source-prep manifest")


def _native_argv(args: argparse.Namespace, num_episodes: int) -> list[str]:
    eval_path = args.reflexbench_root.resolve() / "scripts" / "evaluation" / "eval.py"
    return [
        str(eval_path), "--task", TASK_ID, "--task_profile", TASK_PROFILE,
        "--num_envs", "1", "--num_episodes", str(num_episodes),
        "--backend", "server", "--server_url", args.policy_url,
        "--inference_mode", "sync", "--control", "joint_pos", "--state_format", "joint",
        "--obs_mode", "vla", "--cam_names", "fixed_cam", "--image_history", "1",
        "--ctrl_freq", "25", "--action_scale", "0.1", "--seed", str(args.seed_base),
        "--device", "cuda:0", "--headless",
    ]


def mirrored_asset_path(uri: str, mirror: Path) -> Path:
    """Map the pinned bucket's object path without changing the USD contents."""
    parsed = urlparse.urlparse(uri)
    allowed_hosts = {"omniverse-content-production.s3-us-west-2.amazonaws.com",
                     "omniverse-content-production.s3.us-west-2.amazonaws.com"}
    relative = PurePosixPath(parsed.path.lstrip("/"))
    if (parsed.scheme not in ("http", "https") or parsed.hostname not in allowed_hosts
            or parsed.query or parsed.fragment or ".." in relative.parts
            or relative.parts[:2] != ("Assets", "Isaac")):
        raise ValueError(f"Unsupported native asset URI: {uri}")
    return mirror.resolve().joinpath(*relative.parts)


def load_native_evaluator(args: argparse.Namespace, num_episodes: int):
    eval_path = args.reflexbench_root.resolve() / "scripts" / "evaluation" / "eval.py"
    if not eval_path.is_file():
        raise FileNotFoundError(f"Pinned native evaluator not found: {eval_path}")
    app_dir = str(eval_path.parent)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    old_argv = sys.argv
    original_import = builtins.__import__
    import_compatibility = []

    def import_with_legacy_clone(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name == "isaaclab.sim.utils.prims":
            if tuple(fromlist) != ("clone",):
                raise ImportError("Only the native clone decorator has a verified legacy import mapping")
            module = original_import("isaaclab.sim.utils", globals, locals, fromlist, 0)
            if not callable(getattr(module, "clone", None)):
                raise ImportError("Isaac Lab's native clone decorator is unavailable")
            import_compatibility.append({"requested": name + ".clone", "resolved": "isaaclab.sim.utils.clone"})
            return module
        return original_import(name, globals, locals, fromlist, level)

    sys.argv = _native_argv(args, num_episodes)
    builtins.__import__ = import_with_legacy_clone
    try:
        spec = importlib.util.spec_from_file_location("reflexbench_pinned_eval_rolling", eval_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load pinned ReflexBench evaluation module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = original_import
        sys.argv = old_argv
    evaluator = module.Evaluator(module.args_cli)
    asset_mapping = []
    render_mapping = []
    original_make = module.gym.make
    def make_with_local_assets(task, *positional, **kwargs):
        if task == TASK_ID:
            cfg = kwargs["cfg"]
            if not hasattr(cfg, "num_rerenders_on_reset"):
                cfg.num_rerenders_on_reset = int(cfg.rerender_on_reset)
                import_compatibility.append({"requested": "cfg.num_rerenders_on_reset",
                                             "resolved": "int(cfg.rerender_on_reset)",
                                             "value": cfg.num_rerenders_on_reset})
            render_mapping.append({"setting": "antialiasing_mode", "native_cfg": cfg.sim.render.antialiasing_mode,
                                   "effective": "FXAA" if args.pairing_protocol == "strict-rgb-v1" else "native_default",
                                   "reason": "strict RGB replay diagnostic" if args.pairing_protocol == "strict-rgb-v1" else "preserve native rendering; physical pairing only"})
            if args.pairing_protocol == "strict-rgb-v1":
                cfg.sim.render.antialiasing_mode = "FXAA"
            if args.asset_mirror is not None:
                for name in ("robot", "catcher", "plane"):
                    spawn = getattr(cfg.scene, name).spawn
                    original_uri = spawn.usd_path
                    local = mirrored_asset_path(original_uri, args.asset_mirror)
                    if not local.is_file():
                        raise FileNotFoundError(f"Native {name} asset is not cached: {local}")
                    spawn.usd_path = str(local)
                    asset_mapping.append({"scene_asset": name, "original_uri": original_uri,
                                          "local_path": str(local)})
        return original_make(task, *positional, **kwargs)
    # Native setup overrides the robot before gym.make, so map at that boundary.
    module.gym.make = make_with_local_assets
    try:
        evaluator.setup()
    finally:
        module.gym.make = original_make
    evaluator._rolling_asset_mapping = asset_mapping
    evaluator._rolling_import_compatibility = import_compatibility
    evaluator._rolling_render_mapping = render_mapping
    evaluator._rolling_pairing_protocol = args.pairing_protocol
    evaluator._rolling_reset_render_frames = RESET_RENDER_FRAMES if args.pairing_protocol == "strict-rgb-v1" else 1
    _assert_native_contract(evaluator, args, module)
    return module, evaluator


def _assert_native_contract(native, args: argparse.Namespace, module) -> None:
    checks = {
        "task": native.task == TASK_ID,
        "profile": native.task_profile is not None and native.task_profile.get("success_type") == "task_phase_4",
        "one_env": native.num_envs == 1,
        "camera_vla": native.obs_mode == "vla" and native.cam_names == ["fixed_cam"],
        "joint_proprio": native.state_fmt == "joint" and native.control_mode == "joint_pos",
        "action_scale": math.isclose(native.action_scale, ACTION_SCALE, abs_tol=1e-12),
        "physics_dt": math.isclose(native.sim_dt, PHYSICS_DT_S, abs_tol=1e-9),
        "control_dt": math.isclose(native.env_dt, CONTROL_DT_S, abs_tol=1e-9),
        "decimation": native.decimation == DECIMATION,
        "native_success_snapshot_hook": bool(getattr(native, "_has_term_snapshot_hook", False)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Native environment differs from frozen contract: {checks}")
    native.inference_mode = "sync"
    native.use_real_latency = False
    native._rtf_wall_t0 = None
    if not hasattr(module, "simulation_app"):
        raise RuntimeError("Pinned evaluator did not expose its SimulationApp handle")


def _copy_tree(value: Any, torch):
    if torch.is_tensor(value):
        return value.detach().clone().cpu()
    if isinstance(value, dict):
        return {key: _copy_tree(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tree(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_tree(item, torch) for item in value)
    return copy.deepcopy(value)


def _tree_to_device(value: Any, device, torch):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _tree_to_device(item, device, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_device(item, device, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device, torch) for item in value)
    return copy.deepcopy(value)


def _capture_rng(torch, np) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": _copy_tree(np.random.get_state(), torch),
        "torch_cpu": torch.get_rng_state().clone().cpu(),
        "torch_cuda": [item.clone().cpu() for item in torch.cuda.get_rng_state_all()],
    }


def _restore_rng(state: dict[str, Any], torch, np) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def capture_fixture(native, torch, np) -> dict[str, Any]:
    env = native.menv
    missing = [name for name in TASK_BUFFER_NAMES if not hasattr(env, name)]
    if missing:
        raise RuntimeError(f"Pinned Rolling Ball task buffers missing at reset: {missing}")
    event = env.event_manager
    missing = [name for name in EVENT_BUFFER_NAMES if not hasattr(event, name)]
    if missing:
        raise RuntimeError(f"Isaac EventManager state is incomplete; cannot call fixture paired: {missing}")
    if not hasattr(env, "_sim_step_counter") or not hasattr(env, "episode_length_buf"):
        raise RuntimeError("Environment control counters unavailable; refusing fixture claim")
    counters = {"_sim_step_counter": int(env._sim_step_counter),
                "episode_length_buf": _copy_tree(env.episode_length_buf, torch)}
    absent_counters = []
    for name in ("reset_buf", "reset_terminated", "reset_time_outs", "common_step_counter"):
        if hasattr(env, name):
            counters[name] = _copy_tree(getattr(env, name), torch)
        else:
            absent_counters.append(name)
    event_state = {name: _copy_tree(getattr(event, name), torch) for name in EVENT_BUFFER_NAMES}
    task_state = {name: _copy_tree(getattr(env, name), torch) for name in TASK_BUFFER_NAMES}
    image = native._capture_camera_images()["fixed_cam"][0].copy()
    if image.shape != (224, 224, 3) or image.dtype.name != "uint8":
        raise RuntimeError(f"fixed_cam must be uint8[224,224,3], got {image.shape} {image.dtype}")
    return {
        "scene_state": _copy_tree(env.scene.get_state(is_relative=False), torch),
        "task_state": task_state,
        "event_state": event_state,
        "counters": counters,
        "absent_counters": absent_counters,
        "rng_state": _capture_rng(torch, np),
        "fixed_cam_rgb": image,
        "snapshot_scope": "IsaacLab scene.get_state; explicit task/event/control/RNG buffers; not a raw PhysX solver cache",
    }


def restore_fixture(native, fixture: dict[str, Any], torch, np) -> tuple[Any, dict[str, Any]]:
    env = native.menv
    state = _tree_to_device(fixture["scene_state"], env.device, torch)
    env.reset_to(state, env_ids=torch.tensor([0], dtype=torch.long, device=env.device), seed=None, is_relative=False)
    for name, value in fixture["task_state"].items():
        current = getattr(env, name, None)
        if not torch.is_tensor(current):
            raise RuntimeError(f"Task buffer {name} is not a tensor during restore")
        current.copy_(value.to(device=current.device))
    event = env.event_manager
    for name, saved_list in fixture["event_state"].items():
        current_list = getattr(event, name, None)
        if not isinstance(current_list, list) or len(current_list) != len(saved_list):
            raise RuntimeError(f"EventManager {name} list shape differs from fixture")
        for current, saved in zip(current_list, saved_list):
            current.copy_(saved.to(device=current.device))
    for name, value in fixture["counters"].items():
        current = getattr(env, name, None)
        if torch.is_tensor(current):
            current.copy_(value.to(device=current.device))
        else:
            setattr(env, name, int(value))
    for name in fixture.get("absent_counters", []):
        if hasattr(env, name):
            if name not in vars(env):
                raise RuntimeError(f"Cannot restore initially absent counter property: {name}")
            delattr(env, name)
    native._ep_return = [0.0]
    native._ep_length = [0]
    native._max_phase = [0]
    native._ep_actions = [[]]
    native._success_at_term = [False]
    native._phase_at_term = [-1]
    native._clear_image_history([0])
    obs = native._refresh_obs_after_reset()
    for _ in range(native._rolling_reset_render_frames):
        env.sim.render()
        env.scene["fixed_cam"].update(0.0, force_recompute=True)
    _restore_rng(fixture["rng_state"], torch, np)
    actual = capture_fixture(native, torch, np)
    comparison = compare_fixture(fixture, actual, torch, np, pairing_protocol=native._rolling_pairing_protocol)
    if not comparison["pass"]:
        from PIL import Image
        diagnostic_dir = native._rolling_diagnostics_dir
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fixture["fixed_cam_rgb"]).save(diagnostic_dir / "fixture_rgb.png")
        Image.fromarray(actual["fixed_cam_rgb"]).save(diagnostic_dir / "restored_rgb.png")
        import carb
        settings = carb.settings.get_settings()
        previous = actual["fixed_cam_rgb"]
        rows = []
        for index in range(16):
            env.sim.render()
            env.scene["fixed_cam"].update(0.0, force_recompute=True)
            frame = native._capture_camera_images()["fixed_cam"][0].copy()
            reference_diff = np.abs(frame.astype(np.int16) - fixture["fixed_cam_rgb"].astype(np.int16))
            previous_diff = np.abs(frame.astype(np.int16) - previous.astype(np.int16))
            rows.append({"extra_render": index + 1, "reference_mean": float(reference_diff.mean()),
                         "reference_p99": float(np.percentile(reference_diff, 99)),
                         "previous_mean": float(previous_diff.mean()),
                         "previous_p99": float(np.percentile(previous_diff, 99))})
            previous = frame
        Image.fromarray(previous).save(diagnostic_dir / "static_render_last_rgb.png")
        write_json(diagnostic_dir / "rgb_replay_diagnostic.json", {
            "physics_steps_advanced": 0, "render_settings": {key: settings.get(key) for key in
                ("/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx/directLighting/enabled",
                 "/rtx/indirectDiffuse/enabled", "/rtx/ambientOcclusion/enabled")}, "frames": rows})
        raise RuntimeError(f"Fixture restore failed closed: {comparison}")
    return obs, comparison


def _tree_compare(expected: Any, actual: Any, torch, np, atol: float = 1e-6) -> tuple[bool, float]:
    if torch.is_tensor(expected) and torch.is_tensor(actual):
        if tuple(expected.shape) != tuple(actual.shape):
            return False, float("inf")
        diff = (expected.detach().cpu().float() - actual.detach().cpu().float()).abs()
        maximum = float(diff.max().item()) if diff.numel() else 0.0
        return bool(torch.allclose(expected.detach().cpu().float(), actual.detach().cpu().float(), rtol=0.0, atol=atol)), maximum
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape:
            return False, float("inf")
        if expected.dtype.kind in "biu" and actual.dtype.kind in "biu":
            same = bool(np.array_equal(expected, actual))
            return same, 0.0 if same else float(np.max(np.abs(expected.astype(np.int64) - actual.astype(np.int64))))
        diff = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        maximum = float(np.max(diff)) if diff.size else 0.0
        return bool(np.allclose(expected, actual, rtol=0.0, atol=atol)), maximum
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return False, float("inf")
        results = [_tree_compare(expected[key], actual[key], torch, np, atol) for key in expected]
        return all(ok for ok, _ in results), max((value for _, value in results), default=0.0)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False, float("inf")
        results = [_tree_compare(a, b, torch, np, atol) for a, b in zip(expected, actual)]
        return all(ok for ok, _ in results), max((value for _, value in results), default=0.0)
    same = expected == actual
    return bool(same), 0.0 if same else float("inf")


def pairing_gate(rows: dict[str, Any], protocol: str) -> bool:
    if protocol not in ("strict-rgb-v1", "physical-v2"):
        raise ValueError(f"Unknown pairing protocol: {protocol}")
    return all(item["pass"] for key, item in rows.items()
               if protocol == "strict-rgb-v1" or key != "fixed_cam_rgb")


def compare_fixture(expected: dict[str, Any], actual: dict[str, Any], torch, np,
                    *, state_atol: float = 1e-6, pairing_protocol: str = "strict-rgb-v1") -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key in ("scene_state", "task_state", "event_state", "counters", "rng_state"):
        ok, max_diff = _tree_compare(expected[key], actual[key], torch, np, state_atol)
        rows[key] = {"pass": ok, "max_abs_diff": max_diff}
    first = expected["fixed_cam_rgb"].astype(np.int16)
    second = actual["fixed_cam_rgb"].astype(np.int16)
    image_diff = np.abs(first - second)
    image_mean = float(image_diff.mean())
    image_p99 = float(np.percentile(image_diff, 99))
    rows["fixed_cam_rgb"] = {"pass": image_mean <= 0.25 and image_p99 <= 2.0,
                             "mean_abs_diff": image_mean, "p99_abs_diff": image_p99,
                             "thresholds": {"mean": 0.25, "p99": 2.0}}
    return {"pass": pairing_gate(rows, pairing_protocol), "pairing_protocol": pairing_protocol,
            "strict_rgb_full_gate_pass": pairing_gate(rows, "strict-rgb-v1"), "components": rows}


def save_fixture_bank(path: Path, payload: dict[str, Any], torch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(path)


def load_fixture_bank(path: Path, torch) -> dict[str, Any]:
    try:
        bank = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bank = torch.load(path, map_location="cpu")
    if not isinstance(bank, dict) or bank.get("format") != "rolling_ball_fixture_bank_v1":
        raise RuntimeError("Unrecognized fixture bank")
    if bank.get("pair_check", {}).get("status") != "PASS":
        raise RuntimeError("Fixture bank has no passing same-first-action pair check")
    if bank.get("source_commit") != REFLEXBENCH_COMMIT or bank.get("task_id") != TASK_ID:
        raise RuntimeError("Fixture bank source/task identity differs from this frozen evaluator")
    return bank


def capture_reset_fixture_bank(native, args: argparse.Namespace, module, torch, np) -> dict[str, Any]:
    fixtures = []
    for offset in range(args.num_episodes):
        seed = args.seed_base + offset
        module.seed_everything(seed)
        native._reset_env_for_episode()
        native._seed_image_history()
        for _ in range(native._rolling_reset_render_frames):
            native.menv.sim.render()
            native.menv.scene["fixed_cam"].update(0.0, force_recompute=True)
        fixture = capture_fixture(native, torch, np)
        fixture["seed"] = seed
        fixtures.append(fixture)
        if len(fixtures) % 10 == 0:
            print(json.dumps({"event": "fixture_captured", "count": len(fixtures),
                              "expected": args.num_episodes, "reset_render_frames": native._rolling_reset_render_frames}), flush=True)
    return {
        "format": "rolling_ball_fixture_bank_v1",
        "source_commit": REFLEXBENCH_COMMIT,
        "task_id": TASK_ID,
        "fixture_count": len(fixtures),
        "seed_ids": [item["seed"] for item in fixtures],
        "snapshot_scope": fixtures[0]["snapshot_scope"],
        "reset_render_frames": native._rolling_reset_render_frames,
        "pairing_protocol": args.pairing_protocol,
        "fixtures": fixtures,
        "pair_check": {"status": "NOT_RUN"},
    }


def _set_native_trackers_for_pair(native) -> None:
    native._ep_return = [0.0]
    native._ep_length = [0]
    native._max_phase = [0]
    native._ep_actions = [[]]
    native._success_at_term = [False]
    native._phase_at_term = [-1]


def _step_pair_action(native, target: Any, torch, np) -> tuple[dict[str, Any], bool]:
    target_array = np.asarray([[target]], dtype=np.float32)
    action = native._convert_abs_action_from(target_array, 0)
    native._track_phase()
    native._ep_actions[0].append(action[0].detach().cpu().numpy())
    _, _, terminated, truncated, _ = native.env.step(action)
    if bool(terminated[0].item()) or bool(truncated[0].item()):
        return {}, False
    native._record_control_step()
    native._capture_frame()
    return capture_fixture(native, torch, np), True


def run_pair_smoke(native, bank: dict[str, Any], torch, np, args: argparse.Namespace) -> dict[str, Any]:
    if not bank["fixtures"]:
        raise RuntimeError("Fixture bank is empty")
    fixture = bank["fixtures"][0]
    obs, initial_check = restore_fixture(native, fixture, torch, np)
    robot = native.menv.scene["robot"]
    q = robot.data.joint_pos[0, native.arm_ids].detach().cpu().numpy().astype(np.float32)
    finger = float(robot.data.joint_pos[0, native.finger_ids].mean().item())
    target = np.concatenate([q.copy(), np.array([1.0 if finger > 0.035 else 0.0], dtype=np.float32)])
    target[0] += 0.005
    reset_server(args.policy_url, args.http_timeout_s)
    first, first_step_ok = _step_pair_action(native, target, torch, np)
    if not first_step_ok:
        raise RuntimeError("Pair-smoke first absolute-target step terminated; fixture cannot be gated")
    obs, restored_check = restore_fixture(native, fixture, torch, np)
    reset_server(args.policy_url, args.http_timeout_s)
    second, second_step_ok = _step_pair_action(native, target, torch, np)
    if not second_step_ok:
        raise RuntimeError("Pair-smoke replay first absolute-target step terminated")
    replay_check = compare_fixture(first, second, torch, np, state_atol=1e-4,
                                   pairing_protocol=native._rolling_pairing_protocol)
    if not replay_check["pass"]:
        raise RuntimeError(f"Same-first-action replay failed; no paired sweep allowed: {replay_check}")
    return {
        "status": "PASS",
        "fixture_seed": fixture["seed"],
        "initial_restore": initial_check,
        "restore_before_replay": restored_check,
        "same_absolute_target": target.tolist(),
        "replayed_next_state_and_events": replay_check,
        "snapshot_scope": bank["snapshot_scope"],
        "raw_physx_solver_cache_saved": False,
        "paired_claim_limit": "Canonical scene/task/event/counter/RNG restore plus same-first-action replay only; not a full PhysX solver snapshot.",
    }


def validate_request_payload(payload: dict[str, Any]) -> None:
    allowed = {"type", "num_envs", "step_ids", "proprioception", "control_mode", "action_format",
               "state_format", "orientation_rep", "images", "task_description"}
    if set(payload) != allowed:
        raise RuntimeError(f"Native VLA request fields differ from allowed RGB+proprio schema: {set(payload)}")
    if payload["type"] != "vla" or payload["num_envs"] != 1:
        raise RuntimeError("Planner receives only a single-environment VLA request")
    if payload["control_mode"] != "joint_pos" or payload["action_format"] != "abs_joint" or payload["state_format"] != "joint":
        raise RuntimeError("Planner request control/state format differs from frozen contract")
    if set(payload["images"]) != {"fixed_cam"}:
        raise RuntimeError("Only fixed_cam RGB may enter the planner")
    if set(payload["proprioception"]) != {"joint_positions", "gripper_state"}:
        raise RuntimeError("Only current arm joints and gripper intent may enter the planner")
    forbidden = ("ball", "phase", "intercept", "rolling", "catcher", "future", "privileged")
    def scan(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if any(word in str(key).lower() for word in forbidden):
                    raise RuntimeError(f"Forbidden privileged planner field: {key}")
                scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)
    scan(payload)


def http_json(url: str, method: str, payload: dict[str, Any] | None, timeout_s: float) -> tuple[dict[str, Any], dict[str, Any]]:
    body = None if payload is None else json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    request = urlrequest.Request(url, data=body, headers=headers, method=method)
    start_mono = time.monotonic()
    start_wall_ns = time.time_ns()
    try:
        with urlrequest.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
            status = response.status
    except urlerror.URLError as exc:
        raise RuntimeError(f"Policy endpoint {url} failed: {exc}") from exc
    arrived_mono = time.monotonic()
    arrived_wall_ns = time.time_ns()
    result = json.loads(raw.decode("utf-8"))
    if status != 200 or not isinstance(result, dict):
        raise RuntimeError(f"Policy endpoint returned HTTP {status} or non-object JSON")
    return result, {"request_start_wall_ns": start_wall_ns,
                    "response_arrival_wall_ns": arrived_wall_ns,
                    "request_start_monotonic_s": start_mono,
                    "response_arrival_monotonic_s": arrived_mono,
                    "client_http_latency_s": arrived_mono - start_mono}


def reset_server(base_url: str, timeout_s: float) -> None:
    response, _ = http_json(base_url.rstrip("/") + "/reset", "POST", {"env_ids": [0]}, timeout_s)
    if response.get("status") != "ok":
        raise RuntimeError(f"Policy server reset was not acknowledged: {response}")


def check_server_info(base_url: str, timeout_s: float) -> dict[str, Any]:
    info, _ = http_json(base_url.rstrip("/") + "/info", "GET", None, timeout_s)
    if info.get("action_dim") != 8 or info.get("action_horizon") != 1 or info.get("control_mode") != "joint_pos":
        raise RuntimeError(f"Policy server /info does not match horizon-1 8-D joint_pos: {info}")
    return info


def submit_request(native, base_url: str, timeout_s: float, obs, tick: int, np,
                   executor: concurrent.futures.ThreadPoolExecutor | None = None) -> dict[str, Any]:
    capture_start_wall_ns = time.time_ns()
    capture_start_mono = time.monotonic()
    capture_sim_s = tick * native.env_dt
    frames = native._capture_camera_images()
    capture_end_mono = time.monotonic()
    capture_end_wall_ns = time.time_ns()
    frame = frames.get("fixed_cam", [None])[0]
    if frame is None or frame.shape != (224, 224, 3):
        raise RuntimeError("Native fixed-camera observation is absent or has a wrong shape")
    payload = native._build_obs(obs, current_frames=frames)
    validate_request_payload(payload)
    record = {
        "request_index": -1,
        "capture_start_wall_ns": capture_start_wall_ns,
        "capture_end_wall_ns": capture_end_wall_ns,
        "capture_start_monotonic_s": capture_start_mono,
        "capture_end_monotonic_s": capture_end_mono,
        "capture_sim_s": capture_sim_s,
        "capture_control_tick": tick,
        "http_started": False,
        "deadline_missed": False,
        "status": "pending",
        "first_action_apply_wall_ns": None,
        "first_action_apply_sim_s": None,
        "first_action_apply_control_tick": None,
        "observation_age_wall_s": None,
        "observation_age_sim_s": None,
    }
    if executor is None:
        result, times = http_json(base_url.rstrip("/") + "/predict", "POST", payload, timeout_s)
        record.update(times)
        record["http_started"] = True
        attach_response(record, result)
        return {"record": record, "result": result, "future": None}
    future = executor.submit(http_json, base_url.rstrip("/") + "/predict", "POST", payload, timeout_s)
    record["http_started"] = True
    record["request_start_wall_ns"] = time.time_ns()
    record["request_start_monotonic_s"] = time.monotonic()
    return {"record": record, "result": None, "future": future}


def attach_response(record: dict[str, Any], response: dict[str, Any]) -> list[float]:
    actions = response.get("actions")
    if not isinstance(actions, list) or len(actions) != 1 or not isinstance(actions[0], list) or len(actions[0]) != 1:
        raise RuntimeError(f"Policy response must have [1,1,8] actions, got {actions!r}")
    target = actions[0][0]
    if not isinstance(target, list) or len(target) != 8 or not all(math.isfinite(float(x)) for x in target):
        raise RuntimeError("Policy action must be eight finite absolute joint/gripper values")
    for key in ("latency_s", "cem_latency_s", "request_id", "server_received_unix_ns", "server_completed_unix_ns"):
        if key not in response:
            raise RuntimeError(f"Policy response is missing timing/identity field {key}")
    record["server_latency_s"] = float(response["latency_s"])
    record["cem_latency_s"] = float(response["cem_latency_s"])
    record["request_id"] = str(response["request_id"])
    record["server_received_unix_ns"] = int(response["server_received_unix_ns"])
    record["server_completed_unix_ns"] = int(response["server_completed_unix_ns"])
    record["absolute_target"] = [float(x) for x in target]
    record["deadline_missed"] = record.get("deadline_missed", False) or float(record.get("client_http_latency_s", 0.0)) > CONTROL_DT_S
    return record["absolute_target"]


def receive_future(pending: dict[str, Any]) -> list[float]:
    result, times = pending["future"].result()
    record = pending["record"]
    record.update(times)
    attach_response(record, result)
    pending["result"] = result
    return record["absolute_target"]


def mark_applied(record: dict[str, Any], target: list[float], tick: int) -> None:
    apply_mono = time.monotonic()
    apply_wall_ns = time.time_ns()
    record["first_action_apply_wall_ns"] = apply_wall_ns
    record["first_action_apply_sim_s"] = tick * CONTROL_DT_S
    record["first_action_apply_control_tick"] = tick
    record["observation_age_wall_s"] = apply_mono - record["capture_start_monotonic_s"]
    record["observation_age_sim_s"] = max(0, tick - record["capture_control_tick"]) * CONTROL_DT_S
    record["status"] = "applied"
    record["applied_absolute_target"] = [float(x) for x in target]


def hold_target(native, np) -> list[float]:
    robot = native.menv.scene["robot"]
    q = robot.data.joint_pos[0, native.arm_ids].detach().cpu().numpy().astype(np.float32)
    finger = float(robot.data.joint_pos[0, native.finger_ids].mean().item())
    return np.concatenate([q, np.array([1.0 if finger > 0.035 else 0.0], dtype=np.float32)]).tolist()


def environment_step(native, target: list[float], np, *, tick: int,
                     apply_record: dict[str, Any] | None = None,
                     before_process=None) -> tuple[Any, bool, dict[str, Any]]:
    abs_array = np.asarray([[target]], dtype=np.float32)
    action = native._convert_abs_action_from(abs_array, 0)
    native._track_phase()
    native._ep_actions[0].append(action[0].detach().cpu().numpy())
    if apply_record is not None:
        mark_applied(apply_record, target, tick)
    obs, rew, terminated, truncated, info = native.env.step(action)
    step_return_monotonic_s = time.monotonic()
    native._record_control_step()
    native._capture_frame()
    done = bool(terminated[0].item()) or bool(truncated[0].item())
    profile_success = (native.task_profile is not None
                       and native.task_profile.get("success_type") == "task_phase_4"
                       and hasattr(native.menv, "task_phase")
                       and int(native.menv.task_phase[0].item()) == 4)
    cleanup_start = time.monotonic()
    if (done or profile_success) and before_process is not None:
        before_process()
    reset_ids = native._process_dones(rew, terminated, truncated)
    is_terminal = done or profile_success or 0 in reset_ids
    return obs, done or (0 in reset_ids), {
        "terminated": done and bool(terminated[0].item()),
        "truncated": done and bool(truncated[0].item()),
        "reset_ids": reset_ids,
        "terminal_step_monotonic_s": step_return_monotonic_s if is_terminal else None,
        "terminal_post_step_cleanup_wall_s": time.monotonic() - cleanup_start if is_terminal else 0.0,
    }


def _request_for_tick(native, args, obs, tick: int, request_records: list[dict[str, Any]], np,
                      executor=None):
    submitted = submit_request(native, args.policy_url, args.http_timeout_s, obs, tick, np, executor)
    submitted["record"]["request_index"] = len(request_records)
    request_records.append(submitted["record"])
    return submitted


def _finalize_dropped(record: dict[str, Any], reason: str) -> None:
    record["status"] = "dropped"
    record["drop_reason"] = reason


def _count_deadline_miss(record: dict[str, Any]) -> bool:
    record["deadline_missed"] = True
    if record.get("deadline_miss_counted", False):
        return False
    record["deadline_miss_counted"] = True
    return True


def run_episode(native, args: argparse.Namespace, fixture: dict[str, Any], mode: str,
                delay_ticks: int | None, torch, np,
                executor: concurrent.futures.ThreadPoolExecutor | None = None) -> dict[str, Any]:
    obs, restore_report = restore_fixture(native, fixture, torch, np)
    reset_server(args.policy_url, args.http_timeout_s)
    native._start_rtf_timer()
    episode_start = time.monotonic()
    initial_target = hold_target(native, np)
    held_target = initial_target
    request_records: list[dict[str, Any]] = []
    tick = 0
    done = False
    deadline_misses = 0
    pacing_overruns: list[float] = []
    pending = None
    terminal_step_monotonic_s = None
    terminal_post_step_cleanup_wall_s = 0.0

    if mode in ("sync", "fixed-delay"):
        while not done:
            request = _request_for_tick(native, args, obs, tick, request_records, np)
            rec = request["record"]
            target = attach_response(rec, request["result"])
            if rec["deadline_missed"]:
                deadline_misses += int(_count_deadline_miss(rec))
            if mode == "fixed-delay":
                for decision in fixed_delay_decisions(delay_ticks or 0)[:-1]:
                    if tick >= MAX_CONTROL_TICKS:
                        break
                    obs, done, step_info = environment_step(native, held_target, np, tick=tick)
                    tick += 1
                    if done:
                        terminal_step_monotonic_s = step_info["terminal_step_monotonic_s"]
                        terminal_post_step_cleanup_wall_s = step_info["terminal_post_step_cleanup_wall_s"]
                        _finalize_dropped(rec, "native_terminal_during_fixed_sim_delay")
                        break
            if done:
                break
            if tick >= MAX_CONTROL_TICKS:
                raise RuntimeError("75 control ticks elapsed without a native terminal; refusing fabricated success")
            held_target = target
            obs, done, step_info = environment_step(native, target, np, tick=tick, apply_record=rec)
            tick += 1
            if done:
                terminal_step_monotonic_s = step_info["terminal_step_monotonic_s"]
                terminal_post_step_cleanup_wall_s = step_info["terminal_post_step_cleanup_wall_s"]
            if not done and tick >= MAX_CONTROL_TICKS:
                raise RuntimeError("75 control ticks elapsed without a native terminal; refusing fabricated success")
    elif mode == "true-async":
        if executor is None:
            raise RuntimeError("true-async mode requires the single-worker HTTP executor")
        next_deadline = time.monotonic()
        pacing_started = next_deadline
        first_boundary_seen: set[int] = set()
        while not done:
            if pending is None:
                pending = _request_for_tick(native, args, obs, tick, request_records, np, executor)
            sleep_s = next_deadline - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            lateness = max(0.0, time.monotonic() - next_deadline)
            if lateness > 0.001:
                pacing_overruns.append(lateness)

            rec = pending["record"]
            if pending["future"].done() and tick > rec["capture_control_tick"]:
                target = receive_future(pending)
                if rec["deadline_missed"] or rec.get("client_http_latency_s", 0.0) > CONTROL_DT_S:
                    deadline_misses += int(_count_deadline_miss(rec))
                held_target = target
                pending = None
                action_target = target
                apply_record = rec
            else:
                if tick > rec["capture_control_tick"] and rec["request_index"] not in first_boundary_seen:
                    first_boundary_seen.add(rec["request_index"])
                    if not rec["deadline_missed"]:
                        deadline_misses += int(_count_deadline_miss(rec))
                action_target = held_target
                apply_record = None

            if tick >= MAX_CONTROL_TICKS:
                raise RuntimeError("75 control ticks elapsed without a native terminal; refusing fabricated success")
            def drain_if_terminal() -> None:
                nonlocal pending, deadline_misses
                if pending is None:
                    return
                rec0 = pending["record"]
                try:
                    receive_future(pending)
                except Exception as exc:
                    rec0["request_error"] = f"{type(exc).__name__}: {exc}"
                if rec0.get("deadline_missed", False):
                    deadline_misses += int(_count_deadline_miss(rec0))
                _finalize_dropped(rec0, "native_terminal_before_response_apply")
                pending = None

            obs, done, step_info = environment_step(native, action_target, np, tick=tick,
                                                    apply_record=apply_record, before_process=drain_if_terminal)
            tick += 1
            if done:
                terminal_step_monotonic_s = step_info["terminal_step_monotonic_s"]
                terminal_post_step_cleanup_wall_s = step_info["terminal_post_step_cleanup_wall_s"]
            next_deadline = pacing_started + tick * CONTROL_DT_S
            if not done and tick >= MAX_CONTROL_TICKS:
                raise RuntimeError("75 control ticks elapsed without a native terminal; refusing fabricated success")
    else:
        raise ValueError(f"Unsupported runtime mode: {mode}")

    matches = [item for item in native.results if item.get("episode") is not None]
    if not matches:
        raise RuntimeError("Native evaluator did not record a completed episode")
    result = copy.deepcopy(matches[-1])
    if terminal_step_monotonic_s is None:
        raise RuntimeError("Native terminal step did not expose a wall timestamp")
    wall_elapsed = terminal_step_monotonic_s - episode_start
    sim_elapsed = tick * CONTROL_DT_S
    result.update({
        "fixture_seed": int(fixture["seed"]),
        "mode": mode,
        "fixed_delay_ticks": delay_ticks,
        "control_ticks": tick,
        "sim_elapsed_s": sim_elapsed,
        "wall_elapsed_s": wall_elapsed,
        "terminal_post_step_cleanup_wall_s": terminal_post_step_cleanup_wall_s,
        "measured_rtf": sim_elapsed / wall_elapsed if wall_elapsed > 0 else 0.0,
        "request_count": len(request_records),
        "planner_deadline_misses": deadline_misses,
        "wall_pacing_overruns": len(pacing_overruns),
        "wall_pacing_overrun_s": pacing_overruns,
        "fixture_restore": restore_report,
        "requests": request_records,
    })
    return result


def run_pair_mode(native, args: argparse.Namespace, torch, np, runtime_identity: dict[str, str]) -> dict[str, Any]:
    module = getattr(native, "_rolling_eval_module", None)
    if module is None or not callable(getattr(module, "seed_everything", None)):
        raise RuntimeError("Pinned evaluator seed_everything helper is unavailable")
    bank = capture_reset_fixture_bank(native, args, module, torch, np)
    pair_check = run_pair_smoke(native, bank, torch, np, args)
    bank["pair_check"] = pair_check
    bank["runtime_identity"] = runtime_identity
    bank["seed_base"] = args.seed_base
    save_fixture_bank(args.fixture_bank or args.output.with_suffix(".fixtures.pt"), bank, torch)
    return {
        "status": "PASS_fixture_gate",
        "closed_loop_evaluated": False,
        "task_success": None,
        "fixture_bank": str((args.fixture_bank or args.output.with_suffix(".fixtures.pt")).resolve()),
        "fixture_count": bank["fixture_count"],
        "seed_ids": bank["seed_ids"],
        "pair_check": pair_check,
        "runtime_identity": runtime_identity,
        "evidence_limit": "Pair fixture check only; no policy action or task success claim.",
    }


def run_stage(native, args: argparse.Namespace, torch, np,
              runtime_identity: dict[str, str]) -> dict[str, Any]:
    # reset_to acts on menv; the outer Gym wrapper also needs its native first reset.
    native._reset_env_for_episode()
    bank = load_fixture_bank(args.fixture_bank, torch)
    if bank.get("pairing_protocol", "strict-rgb-v1") != args.pairing_protocol:
        raise RuntimeError("Fixture bank pairing protocol differs from the requested protocol")
    expected_seeds = [args.seed_base + i for i in range(args.num_episodes)]
    if bank["seed_ids"] != expected_seeds:
        raise RuntimeError("Fixture bank seed IDs do not exactly match EVAL_FREEZE episode IDs")
    count = args.limit_episodes or args.num_episodes
    if count > len(bank["fixtures"]):
        raise RuntimeError("--limit-episodes exceeds fixture bank length")
    fixtures = bank["fixtures"][:count]
    conditions: list[dict[str, Any]] = []
    modes = [("sync", None)] if args.mode == "sync" else (
        [("fixed-delay", delay) for delay in args.delay_ticks_parsed] if args.mode == "fixed-delay" else [("true-async", None)]
    )
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="rolling-policy-http") if args.mode == "true-async" else None
    try:
        for mode, delay in modes:
            native.results = []
            episodes = []
            for fixture in fixtures:
                episodes.append(run_episode(native, args, fixture, mode, delay, torch, np, executor))
                last = episodes[-1]
                progress = {"status": "RUNNING", "mode": mode, "fixed_delay_ticks": delay,
                            "completed_episodes": len(episodes), "expected_episodes": len(fixtures),
                            "native_successes": sum(bool(ep["success"]) for ep in episodes),
                            "last_episode": {key: last[key] for key in
                                             ("fixture_seed", "success", "control_ticks", "wall_elapsed_s", "request_count")},
                            "completed_conditions": [item["summary"] for item in conditions]}
                write_json(args.output.with_suffix(".progress.json"), progress)
                print(json.dumps({"event": "episode_completed", **progress}), flush=True)
            successes = sum(bool(episode["success"]) for episode in episodes)
            sim_total = sum(float(episode["sim_elapsed_s"]) for episode in episodes)
            wall_total = sum(float(episode["wall_elapsed_s"]) for episode in episodes)
            conditions.append({
                "condition": {"mode": mode, "fixed_delay_ticks": delay,
                              "fixed_delay_ms": None if delay is None else delay * 40},
                "episodes": episodes,
                "summary": {
                    "episodes": len(episodes), "native_successes": successes,
                    "native_success_rate": successes / len(episodes),
                    "total_sim_elapsed_s": sim_total,
                    "total_wall_elapsed_s": wall_total,
                    "aggregate_rtf": sim_total / wall_total if wall_total > 0 else 0.0,
                    "planner_deadline_misses": sum(ep["planner_deadline_misses"] for ep in episodes),
                    "wall_pacing_overruns": sum(ep["wall_pacing_overruns"] for ep in episodes),
                },
            })
            write_json(args.output.parent / f"condition-{mode}-{delay}.json", conditions[-1])
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    return {
        "status": "COMPLETED",
        "closed_loop_evaluated": True,
        "source_commit": REFLEXBENCH_COMMIT,
        "task_id": TASK_ID,
        "mode_requested": args.mode,
        "smoke_only": count < args.num_episodes,
        "paired_fixture_gate": bank["pair_check"],
        "fixture_bank": str(args.fixture_bank.resolve()),
        "runtime_identity": runtime_identity,
        "conditions": conditions,
        "evidence_limit": "Native successes are reported only from ReflexBench evaluator termination hooks. Pairing covers saved scene/task/event/counter/RNG fixture and first-action replay gate, not raw PhysX solver cache equality.",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        print(json.dumps(self_check(), sort_keys=True))
        return 0
    runtime_identity = require_runtime_host(args)
    # All simulator/framework, numpy, and torch imports happen after the real-host guard.
    import numpy as np
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable after the RTX host guard")
    root = args.reflexbench_root.resolve()
    source_identity = verify_reflexbench_source(root)
    module = None
    native = None
    result: dict[str, Any] = {"status": "FAIL", "closed_loop_evaluated": False,
                              "native_task_success": None, "runtime_identity": runtime_identity}
    try:
        module, native = load_native_evaluator(args, args.num_episodes)
        native._rolling_eval_module = module
        native._rolling_diagnostics_dir = args.output.parent
        info = check_server_info(args.policy_url, args.http_timeout_s)
        result["server_info"] = info
        if args.mode == "pair-smoke":
            bank_path = args.fixture_bank or args.output.with_suffix(".fixtures.pt")
            args.fixture_bank = bank_path
            result = run_pair_mode(native, args, torch, np, runtime_identity)
        else:
            result = run_stage(native, args, torch, np, runtime_identity)
            result["server_info"] = info
        result["checkpoint_status"] = (
            "The formal epoch-100 last checkpoint is the frozen policy target; /info checks only protocol shape. "
            "Record policy-server startup evidence identifying the loaded checkpoint before interpreting this run."
        )
        result["reflexbench_source_identity"] = source_identity
        result["native_asset_mapping"] = native._rolling_asset_mapping
        result["native_import_compatibility"] = native._rolling_import_compatibility
        result["native_render_mapping"] = native._rolling_render_mapping
        result["pairing_protocol"] = native._rolling_pairing_protocol
        result["reset_render_frames"] = native._rolling_reset_render_frames
        result["paired"] = bool(args.mode == "pair-smoke" and result.get("pair_check", {}).get("status") == "PASS") or bool(
            args.mode != "pair-smoke" and result.get("paired_fixture_gate", {}).get("status") == "PASS")
        write_json(args.output, result)
        print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                          "paired": result.get("paired"), "closed_loop_evaluated": result.get("closed_loop_evaluated")}), flush=True)
        return 0
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        if native is not None:
            result["native_render_mapping"] = native._rolling_render_mapping
            result["reset_render_frames"] = native._rolling_reset_render_frames
            result["pairing_protocol"] = native._rolling_pairing_protocol
        result["error_traceback"] = traceback.format_exc()
        traceback.print_exc()
        sys.stderr.flush()
        result["closed_loop_evaluated"] = False
        result["paired"] = False
        write_json(args.output, result)
        raise
    finally:
        if native is not None and native.env is not None:
            native.env.close()
        if module is not None and getattr(module, "simulation_app", None) is not None:
            module.simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
