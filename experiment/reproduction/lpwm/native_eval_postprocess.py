#!/usr/bin/env python3
"""Validate official native planner outputs and save paired-task identifiers."""
import ast
import json
import math
import pathlib
import pickle
import re
import sys

import numpy as np


def parse_success_vector(text):
    arrays = list(re.finditer(r"['\"]success['\"]\s*:\s*array\(\[(.*?)\]\)", text, re.S))
    final_rate_pos = text.rfind("Success rate:")
    if final_rate_pos < 0:
        raise ValueError("native final Success rate line is absent")
    after_final_rate = [match for match in arrays if match.start() > final_rate_pos]
    if not after_final_rate or after_final_rate[-1] is not arrays[-1]:
        raise ValueError("last success array does not follow the final native Success rate line")
    body = after_final_rate[-1].group(1)
    if not re.fullmatch(r"\s*(?:True|False)(?:\s*,\s*(?:True|False))*\s*", body):
        raise ValueError("native success array is truncated or contains unexpected text")
    values = re.findall(r"True|False", body)
    if len(values) != 50:
        raise ValueError(f"expected 50 native success outcomes, got {len(values)}")
    return [value == "True" for value in values]


def select_final_eval_record(rows):
    finals = [row for row in rows if any(str(key).startswith("final_eval/") for key in row)]
    if len(finals) != 1:
        raise ValueError(f"expected exactly one native final_eval record, got {len(finals)}")
    if not rows or finals[0] is not rows[-1]:
        raise ValueError("native final_eval record is not the last logs.json record")
    return finals[0]


def self_check():
    earlier = ", ".join(["True"] * 50)
    final = ", ".join(["False"] * 49 + ["True"])
    sample = (
        "Success rate: 1.0\n{'success': array([" + earlier + "])}\n"
        "Success rate: 0.02\n{'success': array([" + final + "])}\n"
    )
    assert parse_success_vector(sample) == ([False] * 49 + [True])
    truncated = "Success rate: 0.5\n{'success': array([ True, False, ...])}\n"
    try:
        parse_success_vector(truncated)
    except ValueError:
        pass
    else:
        raise AssertionError("truncated success vector was accepted")

    planner_rows = [
        {"mpc/iteration": 0, "planner/cost": 8.0},
        {"mpc/iteration": 9, "planner/cost": 2.0},
        {"final_eval/success_rate": 0.02, "final_eval/mean_visual_dist": 1.0},
    ]
    assert select_final_eval_record(planner_rows) is planner_rows[-1]
    duplicate_final = planner_rows + [{"final_eval/success_rate": 0.02}]
    try:
        select_final_eval_record(duplicate_final)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate final_eval record was accepted")
    print("NATIVE_EVAL_POSTPROCESS_SELFCHECK=PASS planner_records=2 final_eval_records=1")


def write_atomic(path, payload):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def main():
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        return
    self_check()
    if len(sys.argv) != 6:
        raise SystemExit("usage: native_eval_postprocess.py PLAN_LOG REPO RUN_NAME MANIFEST PLAN_START")

    plan_log, repo, manifest_path = map(pathlib.Path, (sys.argv[1], sys.argv[2], sys.argv[4]))
    run_name = sys.argv[3]
    plan_start = int(sys.argv[5])
    text = plan_log.read_text(encoding="utf-8", errors="replace")
    needle = "Planning result saved dir:"
    matches = [line.split(needle, 1)[1].strip() for line in text.splitlines() if needle in line]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one native plan output path, found {len(matches)}")
    actual = pathlib.Path(matches[0])
    if not actual.is_absolute():
        raise SystemExit(f"native plan output path is not absolute: {actual}")
    actual = actual.resolve()
    plan_root = (repo / "plan_outputs").resolve()
    if plan_root not in actual.parents or run_name not in actual.name:
        raise SystemExit(f"native plan output is outside the frozen run location: {actual}")

    logs_path = actual / "logs.json"
    if not logs_path.is_file() or logs_path.stat().st_size == 0:
        raise SystemExit(f"native plan logs.json is absent or empty: {logs_path}")
    if int(logs_path.stat().st_mtime) < plan_start:
        raise SystemExit(f"native plan logs.json predates this invocation: {logs_path}")
    rows = [json.loads(line) for line in logs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        row = select_final_eval_record(rows)
    except ValueError as exc:
        raise SystemExit(str(exc))
    required = (
        "final_eval/success_rate",
        "final_eval/mean_visual_dist",
        "final_eval/mean_proprio_dist",
        "final_eval/mean_div_visual_emb",
    )
    if not all(key in row for key in required):
        raise SystemExit("last native final_eval record lacks the expected fields")
    success_rate = row["final_eval/success_rate"]
    if (
        isinstance(success_rate, bool)
        or not isinstance(success_rate, (int, float))
        or not math.isfinite(success_rate)
        or not 0 <= success_rate <= 1
    ):
        raise SystemExit(f"invalid native final_eval success rate: {success_rate!r}")

    seed_matches = list(re.finditer(r"eval_seed:\s*(\[[^\]\r\n]*\])", text))
    if len(seed_matches) != 1:
        raise SystemExit(f"expected exactly one printed eval_seed list, found {len(seed_matches)}")
    eval_seeds = ast.literal_eval(seed_matches[0].group(1))
    expected_seeds = [99 * index + 1 for index in range(50)]
    if (
        type(eval_seeds) is not list
        or any(type(seed) is not int for seed in eval_seeds)
        or eval_seeds != expected_seeds
    ):
        raise SystemExit("native eval_seed order differs from the frozen sequence")
    successes = parse_success_vector(text)
    observed_rate = sum(successes) / 50
    if abs(observed_rate - success_rate) > 1e-12:
        raise SystemExit(f"native success vector/rate mismatch: {observed_rate} vs {success_rate}")

    targets_path = actual / "plan_targets.pkl"
    if not targets_path.is_file() or targets_path.stat().st_size == 0:
        raise SystemExit(f"native plan_targets.pkl is absent or empty: {targets_path}")
    if int(targets_path.stat().st_mtime) < plan_start:
        raise SystemExit(f"native plan_targets.pkl predates this invocation: {targets_path}")
    # This unpickles the complete native target payload, including observations;
    # only state_0/state_g are exported, with no extra dataset reads or image hashes.
    with targets_path.open("rb") as handle:
        targets = pickle.load(handle)
    if not isinstance(targets, dict) or not {"state_0", "state_g"}.issubset(targets):
        raise SystemExit("native plan_targets.pkl lacks state_0/state_g arrays")
    state_0 = np.asarray(targets["state_0"])
    state_g = np.asarray(targets["state_g"])
    for key, states in (("state_0", state_0), ("state_g", state_g)):
        if states.ndim < 2 or states.shape[0] != 50:
            raise SystemExit(f"native {key} must contain 50 task states; got shape={states.shape}")
        if not np.issubdtype(states.dtype, np.number) or not np.isfinite(states).all():
            raise SystemExit(f"native {key} must contain finite numeric values; got dtype={states.dtype}")
    if state_0.shape != state_g.shape:
        raise SystemExit(f"native state_0/state_g shapes differ: {state_0.shape} vs {state_g.shape}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outcomes_path = manifest_path.parent / "final_eval_outcomes.json"
    target_identity_path = manifest_path.parent / "final_eval_target_states.json"
    task_order = [{"index": i, "eval_seed": eval_seeds[i]} for i in range(50)]
    outcomes = {
        "schema": "lpwm-native-final-eval-outcomes-v1",
        "successes": successes,
        "task_order": task_order,
        "eval_seed_base": 99,
        "source_commit": manifest["source_commit"],
        "pbs_job_id": manifest["pbs_job_id"],
        "checkpoint_path": manifest["outputs"]["checkpoint_latest"],
    }
    target_identity = {
        "schema": "lpwm-native-final-eval-target-states-v1",
        "pbs_job_id": manifest["pbs_job_id"],
        "source_commit": manifest["source_commit"],
        "arm": manifest["arm"],
        "plan_targets_path": str(targets_path),
        "eval_seed_base": 99,
        "task_order": task_order,
        "state_0_dtype": str(state_0.dtype),
        "state_0_shape": list(state_0.shape),
        "state_0": state_0.tolist(),
        "state_g_dtype": str(state_g.dtype),
        "state_g_shape": list(state_g.shape),
        "state_g": state_g.tolist(),
    }
    write_atomic(outcomes_path, outcomes)
    write_atomic(target_identity_path, target_identity)
    manifest["outputs"]["native_plan_output_dir"] = str(actual)
    manifest["outputs"]["native_plan_logs"] = str(logs_path)
    manifest["outputs"]["native_final_eval_outcomes"] = str(outcomes_path)
    manifest["outputs"]["native_plan_targets"] = str(targets_path)
    manifest["outputs"]["native_final_eval_target_states"] = str(target_identity_path)
    manifest["evaluation"]["native_final_eval_record_count"] = len(rows)
    manifest["evaluation"]["native_final_eval"] = row
    manifest["evaluation"]["native_final_eval_outcome_count"] = len(successes)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"NATIVE_FINAL_EVAL_OUTCOMES=PASS count={len(successes)} successes={sum(successes)} path={outcomes_path}")
    print(f"NATIVE_FINAL_EVAL_LOGS=PASS records={len(rows)} final_eval_records=1 path={logs_path}")
    print(f"NATIVE_PLAN_TARGET_STATES=PASS count=50 path={target_identity_path}")
    print(f"NATIVE_PLAN_OUTPUT_DIR={actual}")


if __name__ == "__main__":
    main()
