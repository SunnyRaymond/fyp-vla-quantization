"""Standard-library metrics for the frozen visited-state action diagnostic."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence


HORIZON = 32
ACTION_WIDTH = 7
MOTOR_DIMS = 6
MOTOR_STEPS = 10
GRIPPER_LABELS = (-1, 0, 1)
EXPECTED_CASES = 40
EXPECTED_CASES_PER_SUITE = 10
EXPECTED_VARIANTS = ("base", "image_plus", "image_minus", "proprio_plus", "proprio_minus")
SENSITIVITY_FAMILIES = ("image", "proprio")
SENSITIVITY_ARMS = ("clip_only", "joint")


def _matrix(sample: Mapping[str, Any], key: str) -> list[list[float]]:
    if not isinstance(sample, Mapping) or key not in sample:
        raise ValueError(f"sample must contain {key!r}")
    value = sample[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != HORIZON:
        raise ValueError(f"{key} must have shape [{HORIZON}, {ACTION_WIDTH}]")
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != ACTION_WIDTH:
            raise ValueError(f"{key} must have shape [{HORIZON}, {ACTION_WIDTH}]")
        converted = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(float(item)):
                raise ValueError(f"{key} must contain only finite numbers")
            converted.append(float(item))
        rows.append(converted)
    return rows


def _sample(sample: Mapping[str, Any]) -> tuple[list[list[float]], list[list[float]]]:
    return _matrix(sample, "action"), _matrix(sample, "commands")


def _stats(errors: Sequence[float]) -> dict[str, float]:
    if not errors:
        raise ValueError("cannot summarize an empty metric")
    return {
        "rmse": math.sqrt(sum(x * x for x in errors) / len(errors)),
        "mae": sum(abs(x) for x in errors) / len(errors),
        "signed_bias": sum(errors) / len(errors),
        "max_abs": max(abs(x) for x in errors),
    }


def _motor_error(student: list[list[float]], teacher: list[list[float]]) -> dict[str, Any]:
    per_dimension = []
    all_errors = []
    for dim in range(MOTOR_DIMS):
        errors = [student[t][dim] - teacher[t][dim] for t in range(MOTOR_STEPS)]
        all_errors.extend(errors)
        per_dimension.append({"dimension": dim, **_stats(errors)})
    return {
        "steps": MOTOR_STEPS,
        "per_dimension": per_dimension,
        "overall_rmse": math.sqrt(sum(x * x for x in all_errors) / len(all_errors)),
    }


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _signs(commands: list[list[float]], start: int = 0, stop: int = MOTOR_STEPS) -> list[int]:
    return [_sign(commands[t][6]) for t in range(start, stop)]


def _switch_indices(signs: Sequence[int]) -> list[int]:
    # Indices name the first step in the new sign state; step zero has no prior state.
    return [i for i in range(1, len(signs)) if signs[i] != signs[i - 1]]


def endpoint_metrics(student: Mapping[str, Any], teacher: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two action/actual-command samples; gripper signs use actual commands."""
    student_action, student_commands = _sample(student)
    teacher_action, teacher_commands = _sample(teacher)

    # The deployed action is a 10-step chunk; score gripper execution on that chunk.
    student_gripper = _signs(student_commands, 0, MOTOR_STEPS)
    teacher_gripper = _signs(teacher_commands, 0, MOTOR_STEPS)
    disagreements = [i for i, (a, b) in enumerate(zip(student_gripper, teacher_gripper)) if a != b]
    confusion = [[0 for _ in GRIPPER_LABELS] for _ in GRIPPER_LABELS]
    for actual, predicted in zip(teacher_gripper, student_gripper):
        confusion[GRIPPER_LABELS.index(actual)][GRIPPER_LABELS.index(predicted)] += 1

    student_first10 = student_gripper
    teacher_first10 = teacher_gripper
    student_switches = _switch_indices(student_first10)
    teacher_switches = _switch_indices(teacher_first10)
    student_transition = [i in student_switches for i in range(1, MOTOR_STEPS)]
    teacher_transition = [i in teacher_switches for i in range(1, MOTOR_STEPS)]
    timing_mismatch = [i for i, (a, b) in enumerate(zip(student_transition, teacher_transition), start=1) if a != b]

    gripper_errors = [student_action[t][6] - teacher_action[t][6] for t in range(MOTOR_STEPS)]
    return {
        "motor": {
            "normalized_action": _motor_error(student_action, teacher_action),
            "executed_command": _motor_error(student_commands, teacher_commands),
        },
        "gripper": {
            "normalized_rmse": math.sqrt(sum(x * x for x in gripper_errors) / MOTOR_STEPS),
            "command_sign_disagreement": {
                "count": len(disagreements),
                "rate": len(disagreements) / MOTOR_STEPS,
                "steps": disagreements,
            },
            "command_sign_confusion": {
                "rows": "teacher",
                "columns": "student",
                "labels": list(GRIPPER_LABELS),
                "matrix": confusion,
            },
            "first10_switches": {
                "student_signs": student_first10,
                "teacher_signs": teacher_first10,
                "student_indices": student_switches,
                "teacher_indices": teacher_switches,
                "timing_mismatch_steps": timing_mismatch,
                "timing_mismatch_count": len(timing_mismatch),
            },
        },
    }


def _series_metrics(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "rms": math.sqrt(sum(x * x for x in values) / len(values)),
    }


def _relative_error(error_rms: float, reference_rms: float) -> float | None:
    # A zero reference has no defined relative error; its absolute error is still returned.
    return error_rms / reference_rms if reference_rms else None


def _motor_derivative(plus: list[list[float]], minus: list[list[float]], epsilon: float) -> list[list[float]]:
    return [[(plus[t][d] - minus[t][d]) / (2 * epsilon) for d in range(MOTOR_DIMS)] for t in range(MOTOR_STEPS)]


def _response(variant: list[list[float]], base: list[list[float]], epsilon: float | None = None) -> list[list[float]]:
    return [[(variant[t][d] - base[t][d]) / epsilon if epsilon is not None else variant[t][d] - base[t][d]
             for d in range(MOTOR_DIMS)] for t in range(MOTOR_STEPS)]


def _compare_responses(student: list[list[float]], teacher: list[list[float]]) -> list[dict[str, Any]]:
    result = []
    for dim in range(MOTOR_DIMS):
        errors = [student[t][dim] - teacher[t][dim] for t in range(MOTOR_STEPS)]
        student_stats = _series_metrics([student[t][dim] for t in range(MOTOR_STEPS)])
        teacher_stats = _series_metrics([teacher[t][dim] for t in range(MOTOR_STEPS)])
        error_rms = _series_metrics(errors)["rms"]
        result.append({
            "dimension": dim,
            "mean": student_stats["mean"],
            "rms": student_stats["rms"],
            "teacher_mean": teacher_stats["mean"],
            "teacher_rms": teacher_stats["rms"],
            "teacher_error_rmse": error_rms,
            "teacher_relative_error": _relative_error(error_rms, teacher_stats["rms"]),
        })
    return result


def _flip_pattern(sample: list[int], base: list[int]) -> list[int]:
    return [i for i, (value, original) in enumerate(zip(sample, base)) if value != original]


def sensitivity_metrics(
    base: Mapping[str, Any],
    plus: Mapping[str, Any],
    minus: Mapping[str, Any],
    teacher_base: Mapping[str, Any],
    teacher_plus: Mapping[str, Any],
    teacher_minus: Mapping[str, Any],
    epsilon: float,
) -> dict[str, Any]:
    """Compare central and one-sided motor responses plus actual gripper sign flips."""
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real) or not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive number")
    epsilon = float(epsilon)
    base_a, base_c = _sample(base)
    plus_a, plus_c = _sample(plus)
    minus_a, minus_c = _sample(minus)
    teacher_base_a, teacher_base_c = _sample(teacher_base)
    teacher_plus_a, teacher_plus_c = _sample(teacher_plus)
    teacher_minus_a, teacher_minus_c = _sample(teacher_minus)

    student_central = _motor_derivative(plus_a, minus_a, epsilon)
    teacher_central = _motor_derivative(teacher_plus_a, teacher_minus_a, epsilon)
    central = _compare_responses(student_central, teacher_central)

    side_responses: dict[str, Any] = {}
    for name, action, commands, teacher_action, teacher_commands in (
        ("plus", plus_a, plus_c, teacher_plus_a, teacher_plus_c),
        ("minus", minus_a, minus_c, teacher_minus_a, teacher_minus_c),
    ):
        response = _response(action, base_a)
        teacher_response = _response(teacher_action, teacher_base_a)
        flips = _flip_pattern(_signs(commands, 0, MOTOR_STEPS), _signs(base_c, 0, MOTOR_STEPS))
        teacher_flips = _flip_pattern(_signs(teacher_commands, 0, MOTOR_STEPS), _signs(teacher_base_c, 0, MOTOR_STEPS))
        side_responses[name] = {
            "motor_per_dimension": _compare_responses(response, teacher_response),
            "gripper_flip_count": len(flips),
            "gripper_flip_steps": flips,
            "teacher_gripper_flip_count": len(teacher_flips),
            "teacher_gripper_flip_steps": teacher_flips,
            "gripper_flip_pattern_mismatch_steps": sorted(set(flips).symmetric_difference(teacher_flips)),
        }

    return {
        "epsilon": epsilon,
        "central_directional_derivative": {
            "student_per_dimension": [_series_metrics([student_central[t][d] for t in range(MOTOR_STEPS)]) | {"dimension": d}
                                       for d in range(MOTOR_DIMS)],
            "teacher_per_dimension": [_series_metrics([teacher_central[t][d] for t in range(MOTOR_STEPS)]) | {"dimension": d}
                                       for d in range(MOTOR_DIMS)],
            "teacher_comparison_per_dimension": central,
        },
        "one_sided_response": side_responses,
    }


def _finite_metric(value: Any, name: str, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number" + (" or None" if allow_none else ""))
    return float(value)


def _sensitivity_central(row: Mapping[str, Any], key: str) -> list[dict[str, float]]:
    try:
        dimensions = row["central_directional_derivative"]["teacher_comparison_per_dimension"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{key} must contain central teacher-comparison metrics") from exc
    if not isinstance(dimensions, Sequence) or len(dimensions) != MOTOR_DIMS:
        raise ValueError(f"{key} central metrics must contain {MOTOR_DIMS} dimensions")
    result = []
    for dim, values in enumerate(dimensions):
        if not isinstance(values, Mapping):
            raise ValueError(f"{key} central dimension {dim} must be a mapping")
        result.append({
            "student_rms": _finite_metric(values.get("rms"), f"{key}.central[{dim}].rms"),
            "teacher_rms": _finite_metric(values.get("teacher_rms"), f"{key}.central[{dim}].teacher_rms"),
            "teacher_error_rmse": _finite_metric(values.get("teacher_error_rmse"), f"{key}.central[{dim}].teacher_error_rmse"),
        })
    return result


def _sensitivity_side(row: Mapping[str, Any], key: str, side: str) -> list[float]:
    try:
        dimensions = row["one_sided_response"][side]["motor_per_dimension"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{key} must contain {side} one-sided response metrics") from exc
    if not isinstance(dimensions, Sequence) or len(dimensions) != MOTOR_DIMS:
        raise ValueError(f"{key} {side} response must contain {MOTOR_DIMS} dimensions")
    result = []
    for dim, values in enumerate(dimensions):
        if not isinstance(values, Mapping):
            raise ValueError(f"{key} {side} dimension {dim} must be a mapping")
        result.append(_finite_metric(values.get("teacher_error_rmse"), f"{key}.{side}[{dim}].teacher_error_rmse"))
    return result


def _sensitivity_gripper(row: Mapping[str, Any], side: str) -> dict[str, int]:
    try:
        values = row["normalized"]["one_sided_response"][side]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"normalized must contain {side} gripper flip metrics") from exc
    if not isinstance(values, Mapping):
        raise ValueError(f"normalized {side} response must be a mapping")
    keys = ("gripper_flip_count", "teacher_gripper_flip_count")
    counts = {}
    for key in keys:
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"normalized {side}.{key} must be a non-negative integer")
        counts[key] = value
    mismatch = values.get("gripper_flip_pattern_mismatch_steps")
    if not isinstance(mismatch, Sequence) or isinstance(mismatch, (str, bytes)):
        raise ValueError(f"normalized {side}.gripper_flip_pattern_mismatch_steps must be a sequence")
    counts["gripper_flip_pattern_mismatch_count"] = len(mismatch)
    return counts


def _sensitivity_geometry(row: Mapping[str, Any]) -> dict[str, float | None]:
    geometry = row.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("geometry must be a mapping")
    fields = ("student_rms", "teacher_rms", "error_rms", "gain", "cosine")
    return {field: _finite_metric(geometry.get(field), f"geometry.{field}", allow_none=True) for field in fields}


def _validate_sensitivity_row(row: Mapping[str, Any]) -> dict[str, Any]:
    required = ("case_id", "suite", "arm", "family", "epsilon", "normalized", "executed_command", "geometry")
    if not isinstance(row, Mapping) or any(key not in row for key in required):
        raise ValueError(f"each sensitivity row must contain {', '.join(required)}")
    arm = str(row["arm"]).strip().lower()
    family = str(row["family"]).strip().lower()
    if arm not in SENSITIVITY_ARMS:
        raise ValueError(f"unsupported sensitivity arm {row['arm']!r}")
    if family not in SENSITIVITY_FAMILIES:
        raise ValueError(f"unsupported sensitivity family {row['family']!r}")
    epsilon = _finite_metric(row["epsilon"], "epsilon")
    if epsilon is None or epsilon <= 0:
        raise ValueError("epsilon must be positive")
    normalized = row["normalized"]
    executed = row["executed_command"]
    if not isinstance(normalized, Mapping) or not isinstance(executed, Mapping):
        raise ValueError("normalized and executed_command must be mappings")
    central_normalized = _sensitivity_central(normalized, "normalized")
    central_executed = _sensitivity_central(executed, "executed_command")
    sides = {
        side: {
            "normalized_error": _sensitivity_side(normalized, "normalized", side),
            "executed_error": _sensitivity_side(executed, "executed_command", side),
            "gripper": _sensitivity_gripper(row, side),
        }
        for side in ("plus", "minus")
    }
    return {
        "case_id": str(row["case_id"]),
        "suite": str(row["suite"]),
        "arm": arm,
        "family": family,
        "epsilon": epsilon,
        "central_normalized": central_normalized,
        "central_executed": central_executed,
        "sides": sides,
        "geometry": _sensitivity_geometry(row),
    }


def _mean_defined(values: Sequence[float | None]) -> dict[str, Any]:
    defined = [value for value in values if value is not None]
    return {
        "mean": sum(defined) / len(defined) if defined else None,
        "defined_case_count": len(defined),
    }


def _summarize_sensitivity_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    central = {}
    for label, key in (("normalized", "central_normalized"), ("executed_command", "central_executed")):
        central[label] = [
            {
                "dimension": dim,
                **{
                    field: sum(row[key][dim][field] for row in rows) / len(rows) if rows else None
                    for field in ("student_rms", "teacher_rms", "teacher_error_rmse")
                },
            }
            for dim in range(MOTOR_DIMS)
        ]

    one_sided = {}
    for side in ("plus", "minus"):
        one_sided[side] = {
            label: [
                {
                    "dimension": dim,
                    "teacher_error_rmse": (
                        sum(row["sides"][side][metric_key][dim] for row in rows) / len(rows) if rows else None
                    ),
                }
                for dim in range(MOTOR_DIMS)
            ]
            for label, metric_key in (("normalized", "normalized_error"), ("executed_command", "executed_error"))
        }

    gripper = {}
    for side in ("plus", "minus"):
        gripper[side] = {}
        for metric, key in (
            ("flip_count", "gripper_flip_count"),
            ("teacher_flip_count", "teacher_gripper_flip_count"),
            ("flip_pattern_mismatch_steps", "gripper_flip_pattern_mismatch_count"),
        ):
            total = sum(row["sides"][side]["gripper"][key] for row in rows)
            gripper[side][metric] = {
                "total": total,
                "case_mean": total / len(rows) if rows else None,
            }

    geometry = {
        field: _mean_defined([row["geometry"][field] for row in rows])
        for field in ("student_rms", "teacher_rms", "error_rms", "gain", "cosine")
    }
    return {
        "case_count": len(rows),
        "central_directional_derivative": central,
        "one_sided_response": one_sided,
        "gripper": gripper,
        "geometry": geometry,
    }


def _sensitivity_joint_counts(clip_rows: Sequence[Mapping[str, Any]], joint_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clip_by_case = {row["case_id"]: row for row in clip_rows}
    joint_by_case = {row["case_id"]: row for row in joint_rows}
    shared = sorted(clip_by_case.keys() & joint_by_case.keys())
    counts = {"improved": [0] * MOTOR_DIMS, "worsened": [0] * MOTOR_DIMS, "tied": [0] * MOTOR_DIMS}
    for case_id in shared:
        clip = clip_by_case[case_id]["central_normalized"]
        joint = joint_by_case[case_id]["central_normalized"]
        for dim, (clip_value, joint_value) in enumerate(zip(clip, joint)):
            a, b = clip_value["teacher_error_rmse"], joint_value["teacher_error_rmse"]
            if math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15):
                counts["tied"][dim] += 1
            elif b < a:
                counts["improved"][dim] += 1
            else:
                counts["worsened"][dim] += 1
    return {"paired_case_count": len(shared), **counts}


def aggregate_sensitivity(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case sensitivity rows without retaining their raw metrics."""
    parsed_rows = []
    seen = set()
    case_suite: dict[str, str] = {}
    epsilons: dict[str, float] = {}
    for row in rows:
        parsed = _validate_sensitivity_row(row)
        case_id, suite = parsed["case_id"], parsed["suite"]
        key = (case_id, parsed["family"], parsed["arm"])
        if key in seen:
            raise ValueError(f"duplicate sensitivity row for case={case_id!r}, family={parsed['family']!r}, arm={parsed['arm']!r}")
        seen.add(key)
        if case_id in case_suite and case_suite[case_id] != suite:
            raise ValueError(f"case {case_id!r} has inconsistent suite")
        case_suite[case_id] = suite
        family, epsilon = parsed["family"], parsed["epsilon"]
        if family in epsilons and not math.isclose(epsilons[family], epsilon, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"family {family!r} has inconsistent epsilon")
        epsilons[family] = epsilon
        parsed_rows.append(parsed)

    suite_case_counts: dict[str, int] = {}
    for suite in case_suite.values():
        suite_case_counts[suite] = suite_case_counts.get(suite, 0) + 1
    cohort_complete = len(case_suite) == EXPECTED_CASES and len(suite_case_counts) == 4 and all(
        count == EXPECTED_CASES_PER_SUITE for count in suite_case_counts.values()
    )

    groups = {
        family: {
            arm: [row for row in parsed_rows if row["family"] == family and row["arm"] == arm]
            for arm in SENSITIVITY_ARMS
        }
        for family in SENSITIVITY_FAMILIES
    }

    def group_coverage(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        case_ids = {row["case_id"] for row in group}
        suites: dict[str, int] = {}
        for row in group:
            suites[row["suite"]] = suites.get(row["suite"], 0) + 1
        complete = len(group) == EXPECTED_CASES and len(case_ids) == EXPECTED_CASES and len(suites) == 4 and all(
            count == EXPECTED_CASES_PER_SUITE for count in suites.values()
        ) and case_ids == set(case_suite)
        return {
            "row_count": len(group),
            "case_count": len(case_ids),
            "suite_case_counts": dict(sorted(suites.items())),
            "complete": complete,
        }

    family_summaries = {}
    group_coverage_summary = {}
    for family in SENSITIVITY_FAMILIES:
        family_groups = groups[family]
        coverage_by_arm = {arm: group_coverage(family_groups[arm]) for arm in SENSITIVITY_ARMS}
        group_coverage_summary[family] = coverage_by_arm
        suite_names = sorted({row["suite"] for arm_rows in family_groups.values() for row in arm_rows})

        def scope_summary(selected: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
            unique_cases = {row["case_id"] for arm_rows in selected.values() for row in arm_rows}
            return {
                "case_count": len(unique_cases),
                "arms": {arm: _summarize_sensitivity_arm(selected[arm]) for arm in SENSITIVITY_ARMS},
                "joint_vs_clip_normalized_derivative_error_by_dimension": _sensitivity_joint_counts(
                    selected["clip_only"], selected["joint"]
                ),
            }

        overall = scope_summary(family_groups)
        by_suite = {}
        for suite in suite_names:
            selected = {
                arm: [row for row in family_groups[arm] if row["suite"] == suite]
                for arm in SENSITIVITY_ARMS
            }
            by_suite[suite] = scope_summary(selected)
        family_summaries[family] = {
            "epsilon": epsilons.get(family),
            "overall": overall,
            "by_suite": by_suite,
            "coverage": {
                "by_arm": coverage_by_arm,
                "complete_family": cohort_complete and all(item["complete"] for item in coverage_by_arm.values()),
            },
        }

    return {
        "coverage": {
            "row_count": len(parsed_rows),
            "case_count": len(case_suite),
            "suite_case_counts": dict(sorted(suite_case_counts.items())),
            "expected_cases": EXPECTED_CASES,
            "expected_cases_per_suite": EXPECTED_CASES_PER_SUITE,
            "expected_families": list(SENSITIVITY_FAMILIES),
            "expected_arms_per_family": list(SENSITIVITY_ARMS),
            "complete_40_case_4_suite_cohort": cohort_complete,
            "complete_160_row_panel": len(parsed_rows) == 160 and cohort_complete and all(
                item["complete"] for family in group_coverage_summary.values() for item in family.values()
            ),
        },
        "families": family_summaries,
    }


def _canonical_arm(value: Any) -> str:
    arm = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "clip": "clip_only",
        "clip_only": "clip_only",
        "joint": "joint",
        "bf16": "bf16",
        "teacher": "bf16",
    }
    if arm not in aliases:
        raise ValueError(f"unsupported arm {value!r}; expected clip_only, joint, or BF16")
    return aliases[arm]


def _mean_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"case_count": 0, "clip_only_rmse_by_dimension": None, "joint_rmse_by_dimension": None}
    clip = [sum(row["clip_only_rmse_by_dimension"][d] for row in rows) / len(rows) for d in range(MOTOR_DIMS)]
    joint = [sum(row["joint_rmse_by_dimension"][d] for row in rows) / len(rows) for d in range(MOTOR_DIMS)]
    return {
        "case_count": len(rows),
        "clip_only_rmse_by_dimension": clip,
        "joint_rmse_by_dimension": joint,
        "clip_only_mean_dimension_rmse": sum(clip) / MOTOR_DIMS,
        "joint_mean_dimension_rmse": sum(joint) / MOTOR_DIMS,
        "clip_only_endpoint_case_weighted": _mean_endpoint(rows, "clip_only_endpoint"),
        "joint_endpoint_case_weighted": _mean_endpoint(rows, "joint_endpoint"),
    }


def _mean_endpoint(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    endpoints = [row[key] for row in rows]
    motor = {}
    for field in ("normalized_action", "executed_command"):
        motor[field] = [
            {
                "dimension": dim,
                **{
                    metric: (
                        max(endpoint["motor"][field]["per_dimension"][dim][metric] for endpoint in endpoints)
                        if metric == "max_abs"
                        else sum(endpoint["motor"][field]["per_dimension"][dim][metric] for endpoint in endpoints)
                        / len(endpoints)
                    )
                    for metric in ("rmse", "mae", "signed_bias", "max_abs")
                },
            }
            for dim in range(MOTOR_DIMS)
        ]
        motor[field + "_overall_rmse_case_weighted"] = sum(
            endpoint["motor"][field]["overall_rmse"] for endpoint in endpoints
        ) / len(endpoints)
    confusion = [
        [sum(endpoint["gripper"]["command_sign_confusion"]["matrix"][r][c] for endpoint in endpoints)
         for c in range(len(GRIPPER_LABELS))]
        for r in range(len(GRIPPER_LABELS))
    ]
    grip = [endpoint["gripper"] for endpoint in endpoints]
    return {
        "motor": motor,
        "gripper": {
            "normalized_rmse": sum(item["normalized_rmse"] for item in grip) / len(grip),
            "command_sign_disagreement_count": sum(item["command_sign_disagreement"]["count"] for item in grip),
            "command_sign_disagreement_rate_case_weighted": sum(
                item["command_sign_disagreement"]["rate"] for item in grip
            ) / len(grip),
            "command_sign_confusion_counts": confusion,
            "switch_timing_mismatch_count": sum(item["first10_switches"]["timing_mismatch_count"] for item in grip),
            "switch_timing_mismatch_rate_case_weighted": sum(
                item["first10_switches"]["timing_mismatch_count"] / (MOTOR_STEPS - 1) for item in grip
            ) / len(grip),
        },
    }


def _joint_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    counts = {"improved": [0] * MOTOR_DIMS, "worsened": [0] * MOTOR_DIMS, "tied": [0] * MOTOR_DIMS}
    for row in rows:
        for d, (joint, clip) in enumerate(zip(row["joint_rmse_by_dimension"], row["clip_only_rmse_by_dimension"])):
            if math.isclose(joint, clip, rel_tol=1e-12, abs_tol=1e-15):
                counts["tied"][d] += 1
            elif joint < clip:
                counts["improved"][d] += 1
            else:
                counts["worsened"][d] += 1
    return counts


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize paired clip-only/joint errors against same-case BF16 records.

    Required record keys: case_id, suite, arm, variant, action, commands,
    sampler_seed. Optional input_perturbation metadata is carried into each case row.
    A case must use one sampler seed across arms and variants. Missing arm/variant
    triplets are reported and excluded from paired arm comparisons.
    """
    indexed: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    case_info: dict[str, tuple[str, Any]] = {}
    suite_cases: dict[str, set[str]] = {}
    record_count = 0
    for record in records:
        record_count += 1
        if not isinstance(record, Mapping):
            raise ValueError("each record must be a mapping")
        required = ("case_id", "suite", "arm", "variant", "sampler_seed")
        if any(key not in record for key in required):
            raise ValueError(f"each record must contain {', '.join(required)}")
        case_id, suite = str(record["case_id"]), str(record["suite"])
        variant, seed = str(record["variant"]), record["sampler_seed"]
        arm = _canonical_arm(record["arm"])
        _sample(record)
        if case_id in case_info and case_info[case_id] != (suite, seed):
            raise ValueError(f"case {case_id!r} has inconsistent suite or sampler_seed")
        case_info[case_id] = (suite, seed)
        suite_cases.setdefault(suite, set()).add(case_id)
        group = indexed.setdefault((case_id, variant, str(seed)), {})
        if arm in group:
            raise ValueError(f"duplicate record for case={case_id!r}, variant={variant!r}, arm={arm!r}")
        group[arm] = record

    by_variant: dict[str, list[dict[str, Any]]] = {}
    incomplete: dict[str, list[dict[str, Any]]] = {}
    for (case_id, variant, _seed), arms in sorted(indexed.items()):
        missing = sorted({"clip_only", "joint", "bf16"} - set(arms))
        if missing:
            incomplete.setdefault(variant, []).append({"case_id": case_id, "missing_arms": missing})
            continue
        teacher = arms["bf16"]
        suite, seed = case_info[case_id]
        clip_endpoint = endpoint_metrics(arms["clip_only"], teacher)
        joint_endpoint = endpoint_metrics(arms["joint"], teacher)
        row: dict[str, Any] = {
            "case_id": case_id,
            "suite": suite,
            "sampler_seed": seed,
            "variant": variant,
            "clip_only_rmse_by_dimension": [item["rmse"] for item in clip_endpoint["motor"]["normalized_action"]["per_dimension"]],
            "joint_rmse_by_dimension": [item["rmse"] for item in joint_endpoint["motor"]["normalized_action"]["per_dimension"]],
            "clip_only_endpoint": clip_endpoint,
            "joint_endpoint": joint_endpoint,
        }
        perturbation = next((arms[arm].get("input_perturbation", arms[arm].get("perturbation"))
                             for arm in ("clip_only", "joint", "bf16")
                             if "input_perturbation" in arms[arm] or "perturbation" in arms[arm]), None)
        if perturbation is not None:
            row["input_perturbation"] = perturbation
        by_variant.setdefault(variant, []).append(row)

    variant_summaries: dict[str, Any] = {}
    for variant, rows in sorted(by_variant.items()):
        rows.sort(key=lambda row: row["case_id"])
        by_suite = {
            suite: _mean_rows([row for row in rows if row["suite"] == suite])
            for suite in sorted({row["suite"] for row in rows})
        }
        variant_summaries[variant] = {
            "paired_case_count": len(rows),
            "overall_case_weighted": _mean_rows(rows),
            "by_suite_case_weighted": by_suite,
            "joint_vs_clip_by_motor_dimension": _joint_counts(rows),
        }

    suites = {suite: len(cases) for suite, cases in sorted(suite_cases.items())}
    complete_cohort = len(case_info) == EXPECTED_CASES and len(suites) == 4 and all(
        count == EXPECTED_CASES_PER_SUITE for count in suites.values()
    )
    complete_panel = (
        complete_cohort
        and record_count == EXPECTED_CASES * len(EXPECTED_VARIANTS) * 3
        and {variant for _, variant, _ in indexed} == set(EXPECTED_VARIANTS)
        and all(len(arms) == 3 for arms in indexed.values())
        and all(len(rows) == EXPECTED_CASES for rows in by_variant.values())
    )
    endpoint = {
        "base": variant_summaries.get("base"),
        "perturbed": {name: value for name, value in variant_summaries.items() if name != "base"},
    }
    case_metrics = [
        {
            "case_id": row["case_id"],
            "suite": row["suite"],
            "sampler_seed": row["sampler_seed"],
            "variant": row["variant"],
            **({"input_perturbation": row["input_perturbation"]} if "input_perturbation" in row else {}),
            "clip_only_endpoint": row["clip_only_endpoint"],
            "joint_endpoint": row["joint_endpoint"],
        }
        for rows in by_variant.values()
        for row in rows
    ]
    case_metrics.sort(key=lambda row: (row["variant"], row["case_id"]))
    return {
        "coverage": {
            "record_count": record_count,
            "case_count": len(case_info),
            "suite_case_counts": suites,
            "sampler_seed_count": len({str(seed) for _, seed in case_info.values()}),
            "variant_count": len({variant for _, variant, _ in indexed}),
            "expected_case_count": EXPECTED_CASES,
            "expected_cases_per_suite": EXPECTED_CASES_PER_SUITE,
            "expected_variants": list(EXPECTED_VARIANTS),
            "complete_40_case_4_suite_cohort": complete_cohort,
            "complete_600_record_panel": complete_panel,
        },
        "endpoint": endpoint,
        "case_metrics": case_metrics,
        "incomplete_case_variants": incomplete,
    }
