"""Aggregate a completed Fast-WAM VQ action-aware run on a PBS CPU allocation."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
from pathlib import Path
from typing import Any


PROTOCOL_NAME = "fastwam-vq-action-aware-v1"
EXPECTED_TARGET_LINEARS = 614
EXPECTED_TEST_CASES = 22
EXPECTED_SELECTION_CASES = 4
EXPECTED_SEEDS = (0, 1)
ACTION_ARM = "row_action_vq_a4"
PRIMARY_METRIC = "motor_rmse_first10_vs_bf16"
METRIC_ALIASES = {
    PRIMARY_METRIC: (PRIMARY_METRIC, "motor_rmse_first10"),
    "gripper_rmse_first10_vs_bf16": ("gripper_rmse_first10_vs_bf16", "gripper_rmse_first10"),
    "motor_rmse_32": ("motor_rmse_32", "motor_rmse_32_vs_bf16", "motor_rmse_32_vs_original_bf16"),
}
BYTE_ALIASES = (
    "encoded_bytes_with_scales_books_indices_tails_transforms",
    "encoded_bytes_with_transforms",
    "encoded_weight_payload_bytes_with_transforms",
    "encoded_bytes",
    "encoding_bytes",
    "encoded_weight_payload_bytes",
    "total_encoded_bytes",
    "storage_bytes",
    "payload_bytes",
)
BPW_ALIASES = ("effective_bpw_with_transform", "effective_bpw", "bits_per_weight", "effective_bits_per_weight", "bpw")
SERIALIZED_BYTE_ALIASES = ("serialized_bytes", "serialized_size_bytes", "bank_file_bytes")
TRANSFORM_ALIASES = ("transforms", "transform", "transform_name", "applied_transforms", "transform_metadata")


def require_pbs_allocation() -> None:
    """Require a real nodefile allocation and reject login-node execution."""
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    host = socket.gethostname().lower()
    if not job_id or not nodefile:
        raise RuntimeError("analysis requires PBS_JOBID and PBS_NODEFILE")
    if "login" in host:
        raise RuntimeError(f"refusing analysis on login host {host}")
    path = Path(nodefile)
    if not path.is_file():
        raise RuntimeError("PBS_NODEFILE must name an existing allocation node file")
    allocated = {name.split(".", 1)[0].lower() for name in path.read_text(encoding="utf-8").split()}
    if host.split(".", 1)[0] not in allocated:
        raise RuntimeError(f"current host {host} is absent from PBS_NODEFILE")


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"required analysis input is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"expected an object at {path}:{line_number}")
                row["_line_order"] = len(rows)
                rows.append(row)
    except FileNotFoundError:
        raise FileNotFoundError(f"required analysis input is missing: {path}") from None
    if not rows:
        raise ValueError(f"no records found in {path}")
    return rows


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _find_values(root: Any, names: tuple[str, ...] | list[str]) -> list[Any]:
    wanted = {_norm_key(name) for name in names}
    found = []
    for mapping in _walk(root):
        for key, value in mapping.items():
            if _norm_key(str(key)) in wanted and value is not None:
                found.append(value)
    return found


def _first_value(root: Any, names: tuple[str, ...] | list[str]) -> Any:
    values = _find_values(root, names)
    return values[0] if values else None


def _number(value: Any, label: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if nonnegative and result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _int_count(value: Any, label: str) -> int:
    if isinstance(value, list):
        return len(value)
    number = _number(value, label)
    if not number.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(number)


def _load_plan(source: Path) -> tuple[dict[int, dict[str, Any]], Path]:
    candidates = (source / "plan.json", source.parent / "plan.json", source.parent.parent / "plan.json")
    plan_path = next((path for path in candidates if path.is_file()), None)
    if plan_path is None:
        raise FileNotFoundError("fresh-input plan.json was not found beside results/full")
    plan = _json(plan_path)
    inputs = plan.get("inputs") if isinstance(plan, dict) else None
    if not isinstance(inputs, list):
        raise ValueError("plan.json must contain an inputs list")
    rows: dict[int, dict[str, Any]] = {}
    for row in inputs:
        if not isinstance(row, dict) or "case_id" not in row:
            raise ValueError("plan.json contains a malformed case row")
        case_id = int(row["case_id"])
        if case_id in rows:
            raise ValueError(f"duplicate case_id in plan.json: {case_id}")
        rows[case_id] = row
    if len(rows) != 26:
        raise ValueError(f"plan.json must contain 26 evaluation cases, found {len(rows)}")
    excluded_ids = plan.get("excluded_variant_ids")
    fresh_ids = plan.get("fresh_plus_variant_ids")
    plan_plus_ids = {row.get("variant_id") for row in rows.values() if row.get("domain") == "plus"}
    if (not isinstance(excluded_ids, list) or not excluded_ids or
            not isinstance(fresh_ids, list) or len(fresh_ids) != 12 or
            len(set(fresh_ids)) != 12 or set(fresh_ids) != plan_plus_ids):
        raise ValueError("plan.json must identify twelve fresh Plus variants and the excluded old variants")
    if set(fresh_ids) & set(excluded_ids):
        raise ValueError("fresh Plus variants overlap the excluded old variants")
    prepare_summary_path = plan_path.with_name("prepare_summary.json")
    if not prepare_summary_path.is_file():
        raise FileNotFoundError("prepare_summary.json is required to verify fresh Plus case selection")
    prepared = _json(prepare_summary_path)
    if (prepared.get("status") != "complete" or prepared.get("cases") != 26 or
            prepared.get("selection_cases") != 4 or prepared.get("test_cases") != 22 or
            prepared.get("episodes_executed") != 0):
        raise ValueError("prepare_summary.json does not certify the complete zero-episode 26-case plan")
    if prepared.get("old_variant_exclusion_passed") is not True:
        raise ValueError("fresh Plus plan lacks a passing old-variant exclusion record")
    prepared_excluded = prepared.get("excluded_old_variant_ids")
    if not isinstance(prepared_excluded, list) or set(prepared_excluded) != set(excluded_ids):
        raise ValueError("prepare_summary.json excluded variant IDs disagree with plan.json")
    return rows, plan_path


def _validate_protocol(protocol: dict[str, Any]) -> tuple[list[str], list[int], list[int], list[int], list[float], list[str]]:
    if protocol.get("protocol") != PROTOCOL_NAME:
        raise ValueError(f"unsupported protocol: {protocol.get('protocol')!r}")
    arms = protocol.get("arms")
    expected_arms = ["bf16", "bf16w_a4", "scalar_w4a4", "original_vq_a4", "row_scale_vq_a4", "row_output_vq_a4", ACTION_ARM]
    if arms != expected_arms:
        raise ValueError(f"protocol arms must be exactly {expected_arms}")
    selection_cases = protocol.get("selection_cases")
    test_cases = protocol.get("test_cases")
    seeds = protocol.get("evaluation_seed_indices")
    lambdas = protocol.get("action_sensitivity", {}).get("lambda_candidates")
    dimensions = protocol.get("selection", {}).get("plus_dimensions")
    if not isinstance(selection_cases, list) or len(selection_cases) != EXPECTED_SELECTION_CASES or len(set(selection_cases)) != len(selection_cases):
        raise ValueError("protocol must declare four distinct selection cases")
    if not isinstance(test_cases, list) or len(test_cases) != EXPECTED_TEST_CASES or len(set(test_cases)) != len(test_cases):
        raise ValueError("protocol must declare 22 distinct test cases")
    if set(selection_cases) & set(test_cases):
        raise ValueError("selection and test cases must be disjoint")
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError("evaluation_seed_indices must be [0, 1]")
    if not isinstance(lambdas, list) or len(lambdas) < 2:
        raise ValueError("protocol must declare at least two selection-only action lambda candidates")
    lambda_values = [_number(value, "action lambda", nonnegative=False) for value in lambdas]
    if len(set(lambda_values)) != len(lambda_values):
        raise ValueError("action lambda candidates must be unique")
    if not isinstance(dimensions, list) or len(dimensions) != 6 or len(set(dimensions)) != 6:
        raise ValueError("protocol must declare six Plus dimensions")
    if protocol.get("evaluation_episodes") != 0 or protocol.get("predicted_actions_executed") != 0:
        raise ValueError("protocol must declare zero executed episodes and zero executed predicted actions")
    if protocol.get("restoration_experiments") is not False or protocol.get("native_latency_claim") is not False:
        raise ValueError("protocol must keep restoration and native-latency claims disabled")
    return expected_arms, [int(x) for x in selection_cases], [int(x) for x in test_cases], list(EXPECTED_SEEDS), lambda_values, dimensions


def _validate_plan(
    plan_rows: dict[int, dict[str, Any]], protocol: dict[str, Any],
    selection_cases: list[int], test_cases: list[int], dimensions: list[str],
) -> dict[int, tuple[str, str]]:
    if set(plan_rows) != set(selection_cases + test_cases):
        raise ValueError("plan.json case IDs do not match the declared selection/test cases")
    metadata = {}
    for case_id, row in plan_rows.items():
        split = "selection" if case_id in selection_cases else "test"
        if row.get("group") != split:
            raise ValueError(f"plan case {case_id} has group={row.get('group')!r}, expected {split}")
        domain, dimension = row.get("domain"), row.get("dimension")
        if domain not in ("original", "plus") or not isinstance(dimension, str):
            raise ValueError(f"plan case {case_id} lacks domain/dimension metadata")
        if split == "selection" and (domain != "original" or dimension != "unperturbed"):
            raise ValueError("all selection cases must be familiar original, unperturbed cases")
        if split == "test" and domain == "plus" and dimension not in dimensions:
            raise ValueError(f"test case {case_id} has undeclared Plus dimension {dimension!r}")
        if split == "test" and domain == "original" and dimension != "unperturbed":
            raise ValueError(f"original test case {case_id} must be unperturbed")
        if len(row.get("sampler_seeds", [])) != 2:
            raise ValueError(f"plan case {case_id} must have two sampler seeds")
        metadata[case_id] = (domain, dimension)

    test_rows = [plan_rows[case_id] for case_id in test_cases]
    original = [row for row in test_rows if row["domain"] == "original"]
    plus = [row for row in test_rows if row["domain"] == "plus"]
    if len(original) != 10 or {row.get("task_id") for row in original} != set(range(10)) or len(plus) != 12:
        raise ValueError("test plan must contain ten familiar original tasks and twelve Plus cases")
    if any(sum(row["dimension"] == dimension for row in plus) != 2 for dimension in dimensions):
        raise ValueError("test plan must contain exactly two Plus cases per declared dimension")
    plus_ids = [row.get("variant_id") for row in plus]
    if any(not isinstance(value, str) for value in plus_ids) or len(set(plus_ids)) != 12:
        raise ValueError("Plus test cases must have twelve distinct variant IDs")
    if protocol.get("selection", {}).get("exclude_all_old_plus_variants") is not True:
        raise ValueError("protocol must require excluding prior Plus variants")
    return metadata


def _extract_metric(row: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    value = _first_value(row, aliases)
    if value is None:
        return None
    return _number(value, aliases[0])


def _extract_action(row: dict[str, Any]) -> list[list[float]] | None:
    value = _first_value(row, ("raw_normalized_action_32x7", "action_32x7", "action"))
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 32:
        raise ValueError("raw action must be a 32x7 JSON array")
    action = []
    for timestep, values in enumerate(value):
        if not isinstance(values, list) or len(values) != 7:
            raise ValueError(f"raw action row {timestep} must have seven coordinates")
        action.append([_number(item, "raw action coordinate", nonnegative=False) for item in values])
    return action


def _action_metrics(action: list[list[float]], reference: list[list[float]]) -> dict[str, float]:
    difference = [[action[t][d] - reference[t][d] for d in range(7)] for t in range(32)]
    motor_first10 = [difference[t][d] for t in range(10) for d in range(6)]
    gripper_first10 = [difference[t][6] for t in range(10)]
    motor_all = [difference[t][d] for t in range(32) for d in range(6)]
    return {
        PRIMARY_METRIC: math.sqrt(sum(value * value for value in motor_first10) / len(motor_first10)),
        "gripper_rmse_first10_vs_bf16": math.sqrt(sum(value * value for value in gripper_first10) / len(gripper_first10)),
        "motor_rmse_32": math.sqrt(sum(value * value for value in motor_all) / len(motor_all)),
    }


def _parse_lambda(row: dict[str, Any], raw_arm: str) -> tuple[str, float | None, bool]:
    arms = {"bf16", "bf16w_a4", "scalar_w4a4", "original_vq_a4", "row_scale_vq_a4", "row_output_vq_a4", ACTION_ARM}
    value = _first_value(row, ("action_lambda", "lambda_candidate", "lambda_value", "lambda"))
    parsed = None if value is None else _number(float(value), "action_lambda", nonnegative=False)
    if raw_arm in arms:
        return raw_arm, parsed, False
    if raw_arm.startswith(ACTION_ARM):
        match = re.search(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$", raw_arm)
        if match is None:
            raise ValueError(f"unrecognized action-lambda arm label: {raw_arm}")
        suffix = float(match.group(1))
        if parsed is not None and not math.isclose(parsed, suffix, rel_tol=0, abs_tol=1e-12):
            raise ValueError("action lambda in arm label disagrees with action_lambda field")
        return ACTION_ARM, suffix if parsed is None else parsed, True
    raise ValueError(f"unknown arm label: {raw_arm}")


def _normalise_rows(
    rows: list[dict[str, Any]], source_name: str, plan_metadata: dict[int, tuple[str, str]],
) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        split = row.get("split", row.get("group"))
        if split not in ("selection", "test"):
            continue
        try:
            case_id = int(row["case_id"])
            seed = int(row.get("seed_index", row.get("evaluation_seed_index")))
            raw_arm = row.get("arm", row.get("label"))
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{source_name} contains an evaluation row missing case_id/seed_index/arm") from None
        if case_id not in plan_metadata or seed not in EXPECTED_SEEDS:
            raise ValueError(f"{source_name} has undeclared case/seed {(case_id, seed)}")
        if not isinstance(raw_arm, str):
            raise ValueError(f"{source_name} has a non-string arm label")
        arm, action_lambda, candidate_label = _parse_lambda(row, raw_arm)
        if split == "selection" and arm == ACTION_ARM and action_lambda is None:
            raise ValueError("selection action-arm rows must identify their lambda candidate")
        if split == "test" and candidate_label:
            raise ValueError("action-lambda candidate labels are permitted only in selection rows")
        domain, dimension = plan_metadata[case_id]
        if row.get("domain", domain) != domain or row.get("dimension", dimension) != dimension:
            raise ValueError(f"{source_name} metadata disagrees with plan for case {case_id}")

        metric_values = {name: _extract_metric(row, aliases) for name, aliases in METRIC_ALIASES.items()}
        episode_value = _first_value(row, ("episodes0", "episodes", "episode_count", "evaluation_episodes"))
        if episode_value is None:
            raise ValueError(f"{source_name} is missing the zero-episode counter")
        episodes = 0 if episode_value is True else _int_count(episode_value, "episodes")
        if episodes != 0:
            raise ValueError(f"{source_name} reports executed episodes")
        post_steps = _first_value(row, ("post_query_env_steps", "environment_steps_after_query"))
        if post_steps is not None and _int_count(post_steps, "post_query_env_steps") != 0:
            raise ValueError(f"{source_name} reports environment steps after a query")

        normalized.append({
            "case_id": case_id,
            "seed_index": seed,
            "split": split,
            "arm": arm,
            "action_lambda": action_lambda,
            "candidate_label": candidate_label,
            "domain": domain,
            "dimension": dimension,
            "metrics": metric_values,
            "action": _extract_action(row),
            "lambda_frozen_before_test": row.get("lambda_frozen_before_test"),
            "line_order": int(row.get("_line_order", len(normalized))),
            "source": source_name,
        })
    return normalized


def _token(row: dict[str, Any]) -> tuple[Any, ...]:
    lambda_key = row["action_lambda"] if row["split"] == "selection" and row["arm"] == ACTION_ARM else None
    return row["split"], row["case_id"], row["seed_index"], row["arm"], lambda_key


def _unique_rows(rows: list[dict[str, Any]], label: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        key = _token(row)
        if key in result:
            raise ValueError(f"duplicate {label} evaluation context: {key}")
        result[key] = row
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-7)


def _merge_rows(
    query_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]],
    arms: list[str], selection_cases: list[int], test_cases: list[int], seeds: list[int], lambdas: list[float],
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[str, int]]:
    queries = _unique_rows(query_rows, "queries.jsonl")
    actions = _unique_rows(action_rows, "actions.jsonl")
    if set(queries) != set(actions):
        missing_actions = sorted(map(str, set(queries) - set(actions)))[:5]
        missing_queries = sorted(map(str, set(actions) - set(queries)))[:5]
        raise ValueError(f"queries/actions coverage differs; missing actions={missing_actions}, missing queries={missing_queries}")

    expected_selection = set()
    for case_id in selection_cases:
        for seed in seeds:
            for arm in arms:
                if arm == ACTION_ARM:
                    expected_selection.update(("selection", case_id, seed, arm, value) for value in lambdas)
                else:
                    expected_selection.add(("selection", case_id, seed, arm, None))
    expected_test = {("test", case_id, seed, arm, None) for case_id in test_cases for seed in seeds for arm in arms}
    expected = expected_selection | expected_test
    if set(actions) != expected:
        missing = sorted(map(str, expected - set(actions)))[:8]
        extra = sorted(map(str, set(actions) - expected))[:8]
        raise ValueError(f"action coverage mismatch; missing={missing}, extra={extra}")

    merged = {}
    for key in expected:
        query, action = queries[key], actions[key]
        candidate = dict(action)
        for metric in METRIC_ALIASES:
            left, right = action["metrics"][metric], query["metrics"][metric]
            if left is not None and right is not None and not _close(left, right):
                raise ValueError(f"query/action metric mismatch for {key}, {metric}")
            candidate["metrics"][metric] = left if left is not None else right
        if candidate["action"] is None:
            candidate["action"] = query["action"]
        if candidate["action_lambda"] is None:
            candidate["action_lambda"] = query["action_lambda"]
        if candidate["split"] == "test":
            if action["action_lambda"] is not None and query["action_lambda"] is not None and not math.isclose(
                action["action_lambda"], query["action_lambda"], rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(f"query/action lambda mismatch for {key}")
        merged[key] = candidate

    by_context = {}
    for record in merged.values():
        context = (record["split"], record["case_id"], record["seed_index"])
        if record["arm"] == "bf16":
            by_context.setdefault(context, {})["bf16"] = record
    for context, records in by_context.items():
        reference = records["bf16"]["action"]
        for record in merged.values():
            if (record["split"], record["case_id"], record["seed_index"]) != context:
                continue
            action = record["action"]
            if action is not None and reference is not None:
                measured = _action_metrics(action, reference)
                for metric, value in measured.items():
                    saved = record["metrics"][metric]
                    if saved is not None and not _close(saved, value):
                        raise ValueError(f"saved {metric} disagrees with raw action tensors for {context}/{record['arm']}")
                    record["metrics"][metric] = value
            if record["arm"] == "bf16":
                for metric in METRIC_ALIASES:
                    value = record["metrics"][metric]
                    if value is not None and not _close(value, 0.0):
                        raise ValueError(f"BF16 self-comparison must be zero for {context}")
                    record["metrics"][metric] = 0.0
            for metric in METRIC_ALIASES:
                value = record["metrics"][metric]
                if value is None:
                    raise ValueError(f"missing {metric} for {context}/{record['arm']}")
                record["metrics"][metric] = _number(value, metric)

    counts = {
        "selection_query_action_rows": len(expected_selection),
        "test_query_action_rows": len(expected_test),
        "total_evaluation_context_rows": len(expected),
    }
    return merged, counts


def _same_lambda(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    return sum(values) / len(values)


def _selection_lambda(records: dict[tuple[Any, ...], dict[str, Any]], cases: list[int], seeds: list[int], lambdas: list[float]) -> dict[str, Any]:
    scores = {}
    per_case = {}
    for candidate in lambdas:
        case_means = {}
        for case_id in cases:
            values = [records[("selection", case_id, seed, ACTION_ARM, candidate)]["metrics"][PRIMARY_METRIC] for seed in seeds]
            case_means[str(case_id)] = _mean(values)
        per_case[str(candidate)] = case_means
        scores[str(candidate)] = _mean(list(case_means.values()))
    winner = min(lambdas, key=lambda value: scores[str(value)])
    return {"winner": winner, "scores_by_lambda": scores, "per_case_two_seed_means": per_case,
            "rule": "lowest mean of four selection-case means after averaging two seeds within each case; protocol order breaks exact ties"}


def _selected_lambda_metadata(roots: list[Any]) -> list[float]:
    aliases = ("selected_action_lambda", "frozen_action_lambda", "action_lambda_winner", "lambda_winner")
    values = []
    for root in roots:
        for value in _find_values(root, aliases):
            if isinstance(value, dict):
                nested = value.get("value", value.get("lambda"))
                value = nested
            if value is not None:
                values.append(_number(float(value), "frozen action lambda", nonnegative=False))
    return values


def _verify_freeze(
    summary: dict[str, Any], frozen: Any, queries: list[dict[str, Any]], actions: list[dict[str, Any]],
    records: dict[tuple[Any, ...], dict[str, Any]], winner: float,
) -> dict[str, Any]:
    roots = [summary, frozen]
    declared = _selected_lambda_metadata(roots)
    test_counters = _find_values(roots, ("test_queries_before_freeze", "test_before_freeze_count"))
    if any(_int_count(value, "test_queries_before_freeze") != 0 for value in test_counters):
        raise ValueError("full run reports test queries before action-lambda freeze")

    for rows, name in ((actions, "actions.jsonl"), (queries, "queries.jsonl")):
        selection_orders = [row["_line_order"] for row in rows if row.get("split", row.get("group")) == "selection"]
        test_orders = [row["_line_order"] for row in rows if row.get("split", row.get("group")) == "test"]
        if not selection_orders or not test_orders or max(selection_orders) >= min(test_orders):
            raise ValueError(f"{name} must record all selection rows before any test row")

    test_action_records = [record for record in records.values() if record["split"] == "test" and record["arm"] == ACTION_ARM]
    explicit_test_lambdas = [record["action_lambda"] for record in test_action_records if record["action_lambda"] is not None]
    if any(not _same_lambda(value, winner) for value in explicit_test_lambdas):
        raise ValueError("test action rows do not use the selection-frozen lambda")
    row_flags = [record["lambda_frozen_before_test"] for record in test_action_records]
    if any(value is False for value in row_flags):
        raise ValueError("a test action row says action lambda was not frozen")
    if declared and any(not _same_lambda(value, winner) for value in declared):
        raise ValueError("saved lambda winner disagrees with selection-only candidate scores")
    if not declared and not explicit_test_lambdas:
        raise ValueError("no saved frozen action lambda or test lambda record is available")
    if not test_counters and not declared and not all(value is True for value in row_flags):
        raise ValueError("no authoritative evidence that action lambda was frozen before test")
    return {"selected_action_lambda": winner, "test_queries_before_freeze": 0,
            "selection_rows_precede_test_rows": True, "saved_winner_matches_selection_scores": True}


def _arm_records(root: Any, arm: str) -> list[dict[str, Any]]:
    found = []
    seen = set()
    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if arm in node and isinstance(node[arm], dict) and id(node[arm]) not in seen:
                found.append(node[arm]); seen.add(id(node[arm]))
            labels = (node.get("arm"), node.get("label"), node.get("name"), node.get("method"))
            if arm in labels and id(node) not in seen:
                found.append(node); seen.add(id(node))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(root)
    return [record for record in found if _first_value(record, BYTE_ALIASES) is not None]


def _storage_for_arm(arm: str, roots: list[Any], protocol: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for root in roots:
        candidates.extend(_arm_records(root, arm))
    if not candidates:
        raise ValueError(f"no per-arm storage receipt for {arm}")
    signatures = set()
    parsed = []
    for record in candidates:
        encoded = _int_count(_first_value(record, BYTE_ALIASES), f"{arm} encoded bytes")
        bpw_value = _first_value(record, BPW_ALIASES)
        bpw = None if bpw_value is None else _number(bpw_value, f"{arm} bpw")
        serialized_value = _first_value(record, SERIALIZED_BYTE_ALIASES)
        serialized = None if serialized_value is None else _int_count(serialized_value, f"{arm} serialized bytes")
        transforms = _first_value(record, TRANSFORM_ALIASES)
        module_value = _first_value(record, ("module_count", "target_module_count", "encoded_module_count"))
        module_count = None if module_value is None else _int_count(module_value, f"{arm} module_count")
        signatures.add((encoded, bpw, serialized, json.dumps(transforms, sort_keys=True, ensure_ascii=False)))
        parsed.append((encoded, bpw, serialized, transforms, module_count))
    if len(signatures) > 1:
        raise ValueError(f"conflicting per-arm storage receipts for {arm}")
    encoded, bpw, serialized, transforms, module_count = parsed[0]
    if transforms is None:
        transforms = protocol.get("transform")
    if module_count is not None and module_count != EXPECTED_TARGET_LINEARS:
        raise ValueError(f"{arm} storage receipt covers {module_count} modules, expected 614")
    return {"encoded_bytes": encoded, "bpw": bpw, "serialized_bytes": serialized,
            "transforms": transforms, "module_count": module_count,
            "accounting_source": "per-arm full-run receipt; bytes are reported by the encoder"}


def _module_coverage(roots: list[Any], storage: dict[str, Any]) -> dict[str, Any]:
    target_names = ("target_module_count", "target_linear_count", "target_linears", "modules_targeted")
    calibrated_names = ("calibrated_module_count", "calibrated_linear_count", "calibrated_linears", "modules_calibrated")
    target_values = [_int_count(value, "target module coverage") for root in roots for value in _find_values(root, target_names)]
    calibrated_values = [_int_count(value, "calibrated module coverage") for root in roots for value in _find_values(root, calibrated_names)]
    if target_values and any(value != EXPECTED_TARGET_LINEARS for value in target_values):
        raise ValueError(f"full summary reports target-module counts other than 614: {target_values}")
    if calibrated_values and any(value != EXPECTED_TARGET_LINEARS for value in calibrated_values):
        raise ValueError(f"full summary reports calibrated-module counts other than 614: {calibrated_values}")
    receipt_counts = [value["module_count"] for value in storage.values() if value["module_count"] is not None]
    if receipt_counts and any(value != EXPECTED_TARGET_LINEARS for value in receipt_counts):
        raise ValueError("one or more arm receipts do not cover all 614 target linears")
    if not target_values and not receipt_counts:
        raise ValueError("no authoritative target-linear count or per-arm module coverage was recorded")
    if not calibrated_values and len(receipt_counts) < len(storage):
        raise ValueError("no authoritative evidence that all 614 target linears were calibrated/encoded")
    return {"target_linears": EXPECTED_TARGET_LINEARS,
            "target_count_source": "full summary" if target_values else "per-arm receipts",
            "calibrated_linears": EXPECTED_TARGET_LINEARS,
            "calibrated_count_source": "full summary" if calibrated_values else "all per-arm receipts"}


def _per_case_metrics(
    records: dict[tuple[Any, ...], dict[str, Any]], split: str, arms: list[str], cases: list[int], seeds: list[int],
) -> dict[str, dict[str, dict[str, Any]]]:
    result = {}
    for arm in arms:
        case_means = {}
        seed_values = {}
        for case_id in cases:
            values = {metric: [records[(split, case_id, seed, arm, None)]["metrics"][metric] for seed in seeds]
                      for metric in METRIC_ALIASES}
            seed_values[str(case_id)] = values
            case_means[str(case_id)] = {metric: _mean(items) for metric, items in values.items()}
        summary = {}
        for metric in METRIC_ALIASES:
            per_case = [case_means[str(case_id)][metric] for case_id in cases]
            seed_level = [seed_values[str(case_id)][metric][index] for case_id in cases for index in range(len(seeds))]
            summary[metric] = {"mean_of_case_means": _mean(per_case), "case_count": len(per_case),
                               "seed_context_mean_descriptive_only": _mean(seed_level)}
        result[arm] = {"per_case_two_seed_values": seed_values, "per_case_two_seed_means": case_means,
                       "summary": summary, "case_count": len(cases), "context_count": len(cases) * len(seeds)}
    return result


def _domain_dimension_report(
    records: dict[tuple[Any, ...], dict[str, Any]], arms: list[str], test_cases: list[int], seeds: list[int],
    case_metadata: dict[int, tuple[str, str]], dimensions: list[str],
) -> list[dict[str, Any]]:
    groups = [("original", "unperturbed")] + [("plus", "all_dimensions")] + [("plus", dimension) for dimension in dimensions]
    report = []
    for domain, dimension in groups:
        cases = [case for case in test_cases if case_metadata[case][0] == domain and
                 (dimension == "all_dimensions" or case_metadata[case][1] == dimension)]
        by_arm = {}
        for arm in arms:
            case_means = {str(case): _mean([records[("test", case, seed, arm, None)]["metrics"][PRIMARY_METRIC] for seed in seeds])
                          for case in cases}
            by_arm[arm] = {"case_count": len(cases), "context_count": len(cases) * len(seeds),
                           "mean_of_case_means": None if not cases else _mean(list(case_means.values())),
                           "per_case_two_seed_mean": case_means}
        report.append({"domain": domain, "dimension": dimension, "case_count": len(cases),
                       "context_count": len(cases) * len(seeds), "by_arm": by_arm})
    return report


def _paired_differences(test_by_arm: dict[str, dict[str, Any]], arms: list[str], test_cases: list[int]) -> dict[str, Any]:
    reference = "scalar_w4a4"
    reference_cases = test_by_arm[reference]["per_case_two_seed_means"]
    result = {}
    for arm in arms:
        differences = {str(case): test_by_arm[arm]["per_case_two_seed_means"][str(case)][PRIMARY_METRIC] -
                       reference_cases[str(case)][PRIMARY_METRIC] for case in test_cases}
        values = list(differences.values())
        result[arm] = {"reference_arm": reference, "unit": "paired case mean RMSE difference; negative favors this arm",
                       "case_count": len(values), "mean_difference": _mean(values),
                       "lower_rmse_cases": sum(value < 0 for value in values),
                       "higher_rmse_cases": sum(value > 0 for value in values),
                       "equal_cases": sum(value == 0 for value in values),
                       "per_case_difference": differences}
    return result


def _markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    test = report["test_metrics_by_arm"]
    lines = [
        "# Fast-WAM Action-Aware VQ 汇总", "",
        f"Full run 状态为 `{report['source']['full_status']}`，作业 `{report['source']['full_job_id']}`；protocol `{report['protocol']}`。",
        f"本分析覆盖 {coverage['target_linears']} 个 target Linear，{coverage['test_cases']} 个 test cases、",
        f"{coverage['test_contexts']} 个 case/seed contexts；执行 episodes 与 query 后环境步数均为 0。", "",
        "## 选择冻结", "",
        f"Action lambda 只依据 {coverage['selection_cases']} 个 selection cases 选择：每个 case 先平均两个 sampler seeds，再对 cases 等权平均。",
        f"冻结值为 `{report['action_lambda_selection']['winner']}`；测试记录均晚于 selection 记录，且使用该值。", "",
        "| Lambda | Selection mean of four case means |",
        "|---:|---:|",
    ]
    for value, score in report["action_lambda_selection"]["scores_by_lambda"].items():
        lines.append(f"| {value} | {score:.8g} |")
    lines.extend(["", "## Held-out fixed-input action RMSE", "",
                  "Primary 为相同 case、seed 下相对 BF16 reference 的前 10 步 × 6 motor 维 RMSE。先在每个 case 内平均两个 seeds，再对 case 等权平均。",
                  "Plus cases 与原任务可能共享底层任务，因此 22 个 case 按描述性观察汇总；不报告独立样本置信区间或 p 值。", "",
                  "| Arm | Test cases | Contexts | Primary motor RMSE | First10 gripper RMSE | 32-step motor RMSE |",
                  "|---|---:|---:|---:|---:|---:|"])
    for arm in report["arms"]:
        entry = test[arm]
        metric = entry["metrics"]
        lines.append(f"| {arm} | {entry['case_count']} | {entry['context_count']} | "
                     f"{metric[PRIMARY_METRIC]['mean_of_case_means']:.8g} | "
                     f"{metric['gripper_rmse_first10_vs_bf16']['mean_of_case_means']:.8g} | "
                     f"{metric['motor_rmse_32']['mean_of_case_means']:.8g} |")
    lines.extend(["", "## 按 domain 和 dimension", "",
                  "每格仍是 case 内 seed 平均后，对该组 cases 等权平均。", "",
                  "| Arm | Domain | Dimension | Cases | Contexts | Mean primary RMSE |",
                  "|---|---|---|---:|---:|---:|"])
    for group in report["test_by_domain_dimension"]:
        for arm in report["arms"]:
            value = group["by_arm"][arm]["mean_of_case_means"]
            formatted = "n/a" if value is None else f"{value:.8g}"
            lines.append(f"| {arm} | {group['domain']} | {group['dimension']} | {group['case_count']} | "
                         f"{group['context_count']} | {formatted} |")
    lines.extend(["", "## 与 scalar W4A4 的配对 case 差", "",
                  "这是两种 arm 的 primary RMSE 之差，不是 raw action 向量差；每个 case 的两 seed 均先平均。负值表示该 arm 的 RMSE 较低。", "",
                  "| Arm | Mean paired difference | Lower-RMSE cases | Higher-RMSE cases |",
                  "|---|---:|---:|---:|"])
    for arm in report["arms"]:
        value = report["paired_case_differences_vs_scalar_w4a4"][arm]
        lines.append(f"| {arm} | {value['mean_difference']:.8g} | {value['lower_rmse_cases']} | {value['higher_rmse_cases']} |")
    lines.extend(["", "## 编码存储", "",
                  "字节数和 bpw 来自 full-run 每 arm receipt；transform 字段单独列出，receipt 声明的 bytes accounting 原样保留。bf16 reference 的缺失 bpw 显示为 n/a。", "",
                  "| Arm | Encoded bytes | bpw | Serialized bytes | Transform |",
                  "|---|---:|---:|---:|---|"])
    for arm in report["arms"]:
        item = report["storage"][arm]
        bpw = "n/a" if item["bpw"] is None else f"{item['bpw']:.8g}"
        serialized = "n/a" if item["serialized_bytes"] is None else str(item["serialized_bytes"])
        transform = json.dumps(item["transforms"], ensure_ascii=False, separators=(",", ":"))
        lines.append(f"| {arm} | {item['encoded_bytes']} | {bpw} | {serialized} | `{transform}` |")
    lines.extend(["", "## 范围与限制", "",
                  f"Test 集包含 10 个熟悉的 original tasks 与 12 个 Plus input variants（6 dimensions，每维 2 个）；这不是 unseen-task generalization。",
                  "Action-aware arm 使用采样的一阶 action projection proxy；它不等于完整 final-action sensitivity，也不单独证明 action 改善机制。",
                  "本结果来自固定 observation 的数值查询，episodes=0；不提供任务成功率、native VQ kernel 延迟或部署速度结论。", "",
                  f"Primary metric：`{report['primary_metric']}`。源数据：`{report['source']['source_dir']}`。", ""])
    return "\n".join(lines)


def analyze(protocol_path: Path, out_root: Path, source_root: Path) -> dict[str, Any]:
    """Validate and aggregate a completed full run; refuses any existing output path."""
    require_pbs_allocation()
    protocol_path, out_root, source_root = protocol_path.resolve(), out_root.resolve(), source_root.resolve()
    if out_root.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis path: {out_root}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"full source directory is missing: {source_root}")

    protocol = _json(protocol_path)
    summary_path = source_root / "summary.json"
    summary = _json(summary_path)
    if summary.get("status") != "complete" or summary.get("phase", "full") != "full":
        raise ValueError("source summary must describe a complete full run")
    full_job_id = _first_value(summary, ("pbs_jobid", "full_job_id", "job_id"))
    if not isinstance(full_job_id, (str, int)) or isinstance(full_job_id, bool) or not str(full_job_id).strip():
        raise ValueError("complete full summary must identify its PBS job")
    if summary.get("preflight_only") is True or summary.get("inform_final_selection") is False:
        raise ValueError("source summary is preflight-only or not authorized for final selection")
    if summary.get("episodes") not in (None, 0) or summary.get("evaluation_episodes") not in (None, 0):
        raise ValueError("full summary reports executed evaluation episodes")

    arms, selection_cases, test_cases, seeds, lambdas, dimensions = _validate_protocol(protocol)
    plan_metadata, plan_path = _load_plan(source_root)
    case_metadata = _validate_plan(plan_metadata, protocol, selection_cases, test_cases, dimensions)
    queries_path, actions_path = source_root / "queries.jsonl", source_root / "actions.jsonl"
    raw_queries, raw_actions = _jsonl(queries_path), _jsonl(actions_path)
    query_rows = _normalise_rows(raw_queries, "queries.jsonl", case_metadata)
    action_rows = _normalise_rows(raw_actions, "actions.jsonl", case_metadata)
    frozen_path = source_root / "frozen_winners.json"
    frozen = _json(frozen_path) if frozen_path.is_file() else {}

    records, counts = _merge_rows(query_rows, action_rows, arms, selection_cases, test_cases, seeds, lambdas)
    lambda_result = _selection_lambda(records, selection_cases, seeds, lambdas)
    freeze_result = _verify_freeze(summary, frozen, raw_queries, raw_actions, records, lambda_result["winner"])

    receipt_path = source_root / "receipts.json"
    receipts = _json(receipt_path) if receipt_path.is_file() else {}
    storage_roots = [receipts, summary]
    storage = {arm: _storage_for_arm(arm, storage_roots, protocol) for arm in arms}
    coverage = _module_coverage([summary, receipts], storage)

    # Compare protocol/sample plan counts with the realized per-arm rows.
    test_by_arm = _per_case_metrics(records, "test", arms, test_cases, seeds)
    selection_static_arms = [arm for arm in arms if arm != ACTION_ARM]
    selection_by_arm = _per_case_metrics(records, "selection", selection_static_arms, selection_cases, seeds)
    selection_lambda_metrics = {}
    for value in lambdas:
        case_means = {str(case): _mean([records[("selection", case, seed, ACTION_ARM, value)]["metrics"][PRIMARY_METRIC] for seed in seeds])
                      for case in selection_cases}
        selection_lambda_metrics[str(value)] = {"case_count": len(selection_cases), "context_count": len(selection_cases) * len(seeds),
                                                "per_case_two_seed_mean": case_means,
                                                "mean_of_case_means": _mean(list(case_means.values()))}
    for case_id, (_, dimension) in case_metadata.items():
        if case_id in test_cases and dimension not in ("unperturbed", *dimensions, "all_dimensions"):
            raise ValueError(f"unexpected dimension in test case {case_id}: {dimension}")
    test_domain_dimension = _domain_dimension_report(records, arms, test_cases, seeds, case_metadata, dimensions)
    paired = _paired_differences(test_by_arm, arms, test_cases)

    selected_rows = [row for row in action_rows if row["split"] == "selection"]
    test_rows = [row for row in action_rows if row["split"] == "test"]
    coverage.update({
        "selection_cases": len(selection_cases), "selection_contexts": len(selection_cases) * len(seeds),
        "test_cases": len(test_cases), "test_contexts": len(test_cases) * len(seeds),
        "distinct_original_test_cases": 10, "distinct_plus_test_cases": 12,
        "plus_cases_per_dimension": {dimension: 2 for dimension in dimensions},
        "expected_arms": arms, "selection_rows": len(selected_rows), "test_rows": len(test_rows),
        "query_action_coverage_matched": True,
    })
    result = {
        "status": "complete",
        "protocol": protocol["protocol"],
        "source": {
            "source_dir": str(source_root), "protocol_path": str(protocol_path),
            "full_summary": str(summary_path), "queries": str(queries_path), "actions": str(actions_path),
            "plan": str(plan_path), "receipts": str(receipt_path) if receipt_path.is_file() else "summary.json",
            "full_job_id": str(full_job_id),
            "full_status": summary["status"], "phase": summary.get("phase", "full"),
            "protocol_source_full_job_reference": protocol.get("source_full_job"),
        },
        "arms": arms,
        "coverage": coverage,
        "raw_record_counts": {"queries_jsonl": len(raw_queries), "actions_jsonl": len(raw_actions), **counts},
        "action_lambda_selection": {**lambda_result, **freeze_result,
                                     "candidate_metrics": selection_lambda_metrics},
        "test_metrics_by_arm": test_by_arm,
        "selection_metrics_by_static_arm": selection_by_arm,
        "test_by_domain_dimension": test_domain_dimension,
        "paired_case_differences_vs_scalar_w4a4": paired,
        "storage": storage,
        "primary_metric": protocol.get("primary_metric", PRIMARY_METRIC),
        "unit": protocol.get("unit"),
        "claim_scope": protocol.get("claim_scope"),
        "limitations": {
            "seed_unit": "two sampler seeds are averaged within each case; cases are descriptive paired observations",
            "plus_dependence": "Plus variants may share underlying tasks; no independent-sample confidence interval or p-value is computed",
            "action_sensitivity": "sampled first-order action projection proxy only; not full final-action sensitivity",
            "episodes": 0, "success_rate_claim": False, "native_kernel_latency_claim": False,
        },
    }
    markdown = _markdown(result)
    out_root.parent.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(exist_ok=False)
    (out_root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out_root / "RESULTS.zh.md").write_text(markdown, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.protocol, args.out, args.source)
    print(json.dumps({"status": result["status"], "full_job_id": result["source"]["full_job_id"],
                      "test_cases": result["coverage"]["test_cases"], "test_contexts": result["coverage"]["test_contexts"],
                      "selected_action_lambda": result["action_lambda_selection"]["winner"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
