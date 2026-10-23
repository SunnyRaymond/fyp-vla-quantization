"""One bounded, end-to-end Cosmos Policy LIBERO smoke episode."""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

from cosmos_runtime import CosmosRuntime, guard


def write_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    guard(args.root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    start_epoch = float(os.environ.get("COSMOS_JOB_START_EPOCH", time.time()))
    soft_deadline_epoch = start_epoch + 6600
    runtime = None
    records = []
    error = None
    try:
        runtime = CosmosRuntime(protocol, args.root, args.out, arm="bf16")
        runtime.soft_deadline_epoch = soft_deadline_epoch
        runtime.verify_action_only_parity = True
        smoke = protocol["smoke"]
        task_id = int(smoke["original_task_id"])
        original_task = "open_the_middle_drawer_of_the_cabinet"
        for attempt, (state_id, seed) in enumerate(zip(smoke["state_ids"], smoke["seeds"])):
            row = {
                "suite": str(smoke["suite"]),
                "task_id": task_id,
                "original_task": original_task,
                "language": "open the middle drawer of the cabinet",
                "dimension": "smoke_original",
                "variant_id": f"smoke-libero_goal-task{task_id}-state{state_id}-seed{seed}",
                "state_id": int(state_id),
                "env_seed": int(seed),
                "sampler_seed": int(seed),
                "smoke_original": True,
            }
            video_path = args.out / "rollouts" / f"cosmos-libero-goal-task{task_id}-state{state_id}-seed{seed}.mp4"
            record = runtime.episode(row, index=attempt, video_path=video_path)
            records.append(record)
            if (record.get("success") and record.get("complete")
                    and record.get("closed_loop_complete") and smoke.get("stop_after_first_success", True)):
                break
            parity = getattr(runtime, "action_only_parity", None)
            if parity and not parity.get("parity", False):
                break
            if record.get("termination") == "soft_deadline":
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:
                error = error or f"runtime close: {type(exc).__name__}: {exc}"

    video_paths = sorted(
        str(path) for path in (args.out / "rollouts").glob("cosmos-libero-goal-task0-state*-seed*.mp4")
        if path.is_file() and path.stat().st_size > 0
    )
    successful_complete = any(
        item.get("success") and item.get("complete") and item.get("closed_loop_complete")
        for item in records
    )
    successful_video_complete = any(
        item.get("success") and item.get("complete") and item.get("closed_loop_complete")
        and item.get("rollout_path") in video_paths
        for item in records
    )
    first_acted = next((item for item in records if item.get("first_action_chunk") is not None), None)
    summary = {
        "protocol": protocol.get("protocol"),
        "status": "complete" if successful_video_complete else "failed",
        "summary_saved": False,
        "closed_loop_complete": successful_complete,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "host": socket.gethostname(),
        "modelqueries": sum(int(item.get("modelqueries", 0)) for item in records),
        "envsteps": sum(int(item.get("steps", 0)) for item in records),
        "episodes": len(records),
        "episode_success": any(item.get("success") for item in records),
        "termination": records[-1].get("termination") if records else "runtime_initialization_error",
        "first_action_chunk": first_acted.get("first_action_chunk") if first_acted else None,
        "action_min": first_acted.get("action_min") if first_acted else None,
        "action_max": first_acted.get("action_max") if first_acted else None,
        "gripper_action_min": first_acted.get("gripper_action_min") if first_acted else None,
        "gripper_action_max": first_acted.get("gripper_action_max") if first_acted else None,
        "video_paths": video_paths,
        "episodes_detail": records,
        "episode_errors": [
            {"index": item.get("index"), "error": item["error"]}
            for item in records if item.get("error")
        ],
        "action_only_parity": getattr(runtime, "action_only_parity", None) if runtime else None,
        "error": error,
    }
    summary_path = args.out / "summary.json"
    write_json_atomic(summary_path, summary)
    summary["summary_saved"] = summary_path.is_file() and summary_path.stat().st_size > 0
    write_json_atomic(summary_path, summary)
    if not summary["summary_saved"]:
        raise RuntimeError("Smoke summary did not persist")
    print(json.dumps({key: summary[key] for key in (
        "status", "summary_saved", "closed_loop_complete", "modelqueries", "envsteps",
        "episodes", "video_paths", "error",
    )}, allow_nan=False), flush=True)
    if not successful_video_complete:
        raise RuntimeError(f"Cosmos closed-loop smoke incomplete; see {summary_path}")


if __name__ == "__main__":
    main()
