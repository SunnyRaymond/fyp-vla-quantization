"""Aggregate all ten complete fixed-protocol LIBERO-Goal task shards; PBS only."""

import argparse
import json
import os
from pathlib import Path

from suite import (
    ARMS, EXPECTED_SITE_COUNT, PROTOCOL_ID, STATE_IDS, screen_runner,
    validate_execution_state_ids,
)


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_result(path):
    path = Path(path).resolve()
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result.get("protocol", {})
    task_id = protocol.get("task_id")
    execution_state_ids = validate_execution_state_ids(
        result.get("execution_state_ids", list(STATE_IDS))
    )
    expected = {(arm, state_id) for arm in ARMS for state_id in execution_state_ids}
    expected_count = len(expected)
    if (protocol.get("protocol_id") != PROTOCOL_ID or task_id not in range(10)
            or result.get("task_id") != task_id or result.get("complete") is not True
            or result.get("slots_attempted") != expected_count
            or result.get("slots_complete") != expected_count
            or result.get("coverage_validated") is not True
            or result.get("reference_gate_passed") is not True
            or protocol.get("expected_site_count") != EXPECTED_SITE_COUNT
            or protocol.get("initial_state_ids") != list(STATE_IDS)
            or protocol.get("arms") != list(ARMS)):
        raise ValueError(f"Incomplete or invalid suite task result: {path}")
    if (protocol.get("action_mode") != "first_frame"
            or protocol.get("num_inference_steps_cfg_and_actual") != 20
            or protocol.get("sigma_shift") != 1.0
            or protocol.get("compile_action_infer") is not False
            or protocol.get("action_horizon") != 32
            or protocol.get("warmup_steps") != 30
            or protocol.get("replan_steps") != 10
            or protocol.get("max_control_steps") != 400
            or protocol.get("locked_selection_source_job_id") != "25666715.pbs101"):
        raise ValueError(f"Task protocol settings do not match the frozen suite: {path}")
    expected_scope = (
        "full 10-task LIBERO-Goal suite shard; 50 initial states per arm"
        if execution_state_ids == STATE_IDS else
        f"LIBERO-Goal state shard for task {task_id}; {len(execution_state_ids)} initial states per arm"
    )
    if result.get("scope") != expected_scope:
        raise ValueError(f"Task result scope does not match its declared state subset: {path}")

    records_path = path.parent / result.get("records_file", "episodes.json")
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} episode rows in {records_path}")
    slots = {}
    for row in records:
        arm, state_id = row.get("arm"), row.get("state_id")
        key = (arm, state_id)
        if (arm not in ARMS or state_id not in execution_state_ids or key in slots
                or row.get("task_id") != task_id or row.get("initial_state_id") != state_id
                or row.get("complete") is not True or not isinstance(row.get("success"), bool)):
            raise ValueError(f"Invalid, duplicated, or incomplete episode slot {key} in {records_path}")
        slots[key] = row
    if set(slots) != expected:
        raise ValueError(f"Task {task_id} does not have all {expected_count} declared arm/state slots")
    summaries = result.get("arm_summaries", {})
    for arm in ARMS:
        arm_summary = summaries.get(arm, {})
        successes = sum(slots[(arm, state_id)]["success"] for state_id in execution_state_ids)
        if (arm_summary.get("attempted") != len(execution_state_ids)
                or arm_summary.get("complete") != len(execution_state_ids)
                or arm_summary.get("successes") != successes
                or arm_summary.get("denominator") != len(execution_state_ids)):
            raise ValueError(f"Task {task_id} {arm} summary does not match its state shard")
    coverage = json.loads((path.parent / "coverage.json").read_text(encoding="utf-8"))
    gate = json.loads((path.parent / "reference_gate.json").read_text(encoding="utf-8"))
    sites = coverage.get("executed_sites", [])
    if (coverage.get("task_id") != task_id or coverage.get("full_path") is not True
            or len(sites) != EXPECTED_SITE_COUNT or len(set(sites)) != EXPECTED_SITE_COUNT
            or coverage.get("action_horizon") != 32 or gate.get("passed") is not True
            or gate.get("task_id") != task_id):
        raise ValueError(f"Task {task_id} coverage/reference artifact failed validation")
    trace_index = json.loads((path.parent / result["trace_index_file"]).read_text(encoding="utf-8"))
    indexed = trace_index.get("episodes", {})
    expected_trace_keys = {f"{arm}:{state_id:02d}" for arm, state_id in expected}
    if (trace_index.get("protocol_id") != PROTOCOL_ID or trace_index.get("task_id") != task_id
            or set(indexed) != expected_trace_keys):
        raise ValueError(f"Task {task_id} does not have a complete declared-slot trace index")
    for (arm, state_id), row in slots.items():
        key = f"{arm}:{state_id:02d}"
        item = indexed.get(key, {})
        rel = Path(row.get("trace_file", ""))
        trace_path = (path.parent / rel).resolve()
        if (item.get("trace_file") != row.get("trace_file")
                or item.get("trace_entries") != row.get("replans")
                or item.get("complete") is not True
                or rel.is_absolute() or not trace_path.is_relative_to(path.parent)
                or not trace_path.is_file() or trace_path.stat().st_size == 0):
            raise ValueError(f"Task {task_id} episode trace is missing or inconsistent: {key}")
    return path, result, protocol, slots, execution_state_ids


def common_identity(protocol):
    ignored = {"task_id", "task_description", "quant_arm_order_seed", "quant_arm_order_by_state"}
    return {key: value for key, value in protocol.items() if key not in ignored}


def merge_loaded_results(loaded):
    if not loaded:
        raise ValueError("No task result files were supplied")
    slots, task_protocols, result_paths = {}, {}, {task_id: [] for task_id in range(10)}
    identity = None
    for path, _result, protocol, task_slots, execution_state_ids in loaded:
        task_id = protocol["task_id"]
        expected = {(arm, state_id) for arm in ARMS for state_id in execution_state_ids}
        if set(task_slots) != expected:
            raise ValueError(f"Task {task_id} result slots do not match their declared state subset")
        if task_id in task_protocols and task_protocols[task_id] != protocol:
            raise ValueError(f"Task {task_id} result files have inconsistent protocols")
        task_protocols[task_id] = protocol
        current_identity = common_identity(protocol)
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError("Task shards have mismatched checkpoint/protocol identity")
        result_paths[task_id].append(str(path))
        for (arm, state_id), row in task_slots.items():
            key = (task_id, arm, state_id)
            if key in slots:
                raise ValueError(f"Duplicate task/arm/state slot across result files: {key}")
            slots[key] = row
    expected_all = {
        (task_id, arm, state_id)
        for task_id in range(10) for arm in ARMS for state_id in STATE_IDS
    }
    if set(slots) != expected_all:
        raise ValueError(
            f"Refusing to score: expected exactly {len(expected_all)} unique complete slots, got {len(slots)}"
        )
    return identity, slots, task_protocols, result_paths


def paired_counts(slots, left, right):
    differences = [
        int(slots[(task_id, left, state_id)]["success"])
        - int(slots[(task_id, right, state_id)]["success"])
        for task_id in range(10)
        for state_id in STATE_IDS
    ]
    return {
        "left_arm": left,
        "right_arm": right,
        "paired_slots": len(differences),
        "left_gain": sum(value == 1 for value in differences),
        "right_gain": sum(value == -1 for value in differences),
        "ties": sum(value == 0 for value in differences),
    }


def self_check():
    slots = {
        (task_id, arm, state_id): {"success": (task_id + state_id + arm_idx) % 2 == 0}
        for task_id in range(10)
        for arm_idx, arm in enumerate(ARMS)
        for state_id in STATE_IDS
    }
    counts = paired_counts(slots, "learned", "rtn")
    if (counts["paired_slots"] != 500
            or counts["left_gain"] + counts["right_gain"] + counts["ties"] != 500):
        raise RuntimeError("Aggregate self-check failed: 500 paired task/state keys")

    ranges = ((0, 13), (13, 26), (26, 38), (38, 50))
    if (len(ranges) * 7 != 28
            or tuple(state_id for start, stop in ranges for state_id in range(start, stop)) != STATE_IDS):
        raise RuntimeError("Aggregate self-check failed: 28 state shards do not partition 7 tasks")

    loaded = []
    for task_id in range(10):
        state_groups = (STATE_IDS,) if task_id < 3 else tuple(
            tuple(range(start, stop)) for start, stop in ranges
        )
        for state_ids in state_groups:
            protocol = {
                "protocol_id": PROTOCOL_ID,
                "task_id": task_id,
                "task_description": f"task {task_id}",
                "quant_arm_order_seed": task_id,
                "quant_arm_order_by_state": {},
                "fixed_identity": "self-check",
            }
            task_slots = {
                (arm, state_id): {"success": (task_id + state_id + arm_idx) % 2 == 0}
                for arm_idx, arm in enumerate(ARMS) for state_id in state_ids
            }
            loaded.append((Path(f"task-{task_id}-{state_ids[0]}.json"),
                           {"execution_state_ids": list(state_ids)}, protocol, task_slots, state_ids))

    identity, merged, _, _ = merge_loaded_results(loaded)
    if identity != common_identity(loaded[0][2]) or len(merged) != 2_000:
        raise RuntimeError("Aggregate self-check failed: full 2000-slot merge")

    def must_reject(items, label):
        try:
            merge_loaded_results(items)
        except ValueError:
            return
        raise RuntimeError(f"Aggregate self-check failed: accepted {label} result slots")

    must_reject(loaded + [loaded[0]], "duplicate")
    must_reject(loaded[:-1], "missing")
    path, result, protocol, task_slots, state_ids = loaded[3]
    out_of_subset = dict(task_slots)
    out_of_subset[(ARMS[0], 13)] = {"success": False}
    must_reject(loaded[:3] + [(path, result, protocol, out_of_subset, state_ids)] + loaded[4:], "out-of-subset")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True,
                        help="complete task result.json files; full-task and state-shard results may be mixed")
    parser.add_argument("--out", type=Path, default=Path("aggregate_result.json"))
    args = parser.parse_args()

    screen_runner.guard()
    self_check()
    loaded = [load_result(path) for path in args.results]
    identity, slots, task_protocols, result_paths = merge_loaded_results(loaded)
    task_rows = []
    for task_id in range(10):
        protocol = task_protocols[task_id]
        task_slots = {
            (arm, state_id): slots[(task_id, arm, state_id)]
            for arm in ARMS for state_id in STATE_IDS
        }
        task_rows.append({
            "task_id": task_id,
            "task_description": protocol["task_description"],
            "episodes_per_arm": 50,
            "arm_summaries": {
                arm: {
                    "successes": sum(task_slots[(arm, state_id)]["success"] for state_id in STATE_IDS),
                    "denominator": 50,
                    "success_rate": sum(task_slots[(arm, state_id)]["success"] for state_id in STATE_IDS) / 50,
                }
                for arm in ARMS
            },
            "result_files": result_paths[task_id],
        })

    overall = {}
    for arm in ARMS:
        successes = sum(slots[(task_id, arm, state_id)]["success"]
                        for task_id in range(10) for state_id in STATE_IDS)
        overall[arm] = {"successes": successes, "denominator": 500, "success_rate": successes / 500}
    paired = {
        name: paired_counts(slots, left, right)
        for left, right, name in (
            ("bf16", "rtn", "bf16_vs_rtn"),
            ("bf16", "independent", "bf16_vs_independent"),
            ("bf16", "learned", "bf16_vs_learned"),
            ("learned", "rtn", "learned_vs_rtn"),
            ("learned", "independent", "learned_vs_independent"),
        )
    }
    aggregate = {
        "complete": True,
        "scope": "LIBERO-Goal fixed-protocol full suite: 10 tasks x 50 initial states x 4 arms",
        "score_scope": "fixed-protocol score; not a reproduction of the paper's environment/seed results",
        "episodes_per_arm": 500,
        "episodes_total": 2000,
        "protocol_identity": identity,
        "per_task": task_rows,
        "overall": overall,
        "paired_success_gain_loss": paired,
    }
    save_json(args.out, aggregate)
    print("LIBERO-Goal fixed-protocol suite SR")
    print("task | BF16 | RTN W4A8 | independent | learned phase")
    for row in task_rows:
        cells = [f"{row['arm_summaries'][arm]['successes']}/50 ({row['arm_summaries'][arm]['success_rate']:.3f})"
                 for arm in ARMS]
        print(f"{row['task_id']:>4} | " + " | ".join(cells))
    print("all  | " + " | ".join(
        f"{overall[arm]['successes']}/500 ({overall[arm]['success_rate']:.3f})" for arm in ARMS
    ))
    for name, counts in paired.items():
        print(f"{name}: left_gain={counts['left_gain']} right_gain={counts['right_gain']} ties={counts['ties']}")
    print(f"WROTE {args.out.resolve()}")


if __name__ == "__main__":
    main()
