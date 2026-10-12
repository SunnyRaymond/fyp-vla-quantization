"""Read-only CPU diagnostic for the frozen Fast-WAM calibration actions.

Run inside an approved PBS compute allocation. The source campaign is opened
read-only; all generated files go beneath --out.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import statistics

import numpy as np


def guard() -> None:
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE")
    if not os.environ.get("PBS_JOBID") or not nodefile or "login" in host.lower():
        raise RuntimeError("Approved PBS compute allocation required")
    nodes = {name.split(".")[0] for name in Path(nodefile).read_text(encoding="utf-8").split()}
    if host not in nodes:
        raise RuntimeError("Host is absent from PBS_NODEFILE")


def motor_delta(action, baseline) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if action.shape != (32, 7) or baseline.shape != (32, 7):
        raise ValueError(f"Expected action and baseline shapes [32,7], got {action.shape} and {baseline.shape}")
    if not np.isfinite(action).all() or not np.isfinite(baseline).all():
        raise ValueError("Action and baseline must be finite")
    return (action[:10, :6] - baseline[:10, :6]).reshape(60)


def motor_rmse(delta) -> float:
    values = np.asarray(delta, dtype=np.float64).reshape(-1)
    if values.size != 60 or not np.isfinite(values).all():
        raise ValueError("Motor delta must contain 60 finite values")
    return float(np.sqrt(np.mean(values * values)))


def rms(values) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("RMS requires finite values")
    return float(np.sqrt(np.mean(values * values)))


def projected_values(delta, probes) -> np.ndarray:
    values = np.asarray(delta, dtype=np.float64).reshape(-1)
    rows = np.asarray(probes, dtype=np.float64).reshape(-1, values.size)
    if values.size != 60 or not np.isfinite(values).all() or not np.isfinite(rows).all():
        raise ValueError("Expected finite 60-dimensional motor delta and probes")
    return rows @ values


def span_energy_fraction(delta, probes) -> tuple[float, int]:
    """Return energy captured by the probe row-span, orthonormalized by SVD."""
    values = np.asarray(delta, dtype=np.float64).reshape(-1)
    rows = np.asarray(probes, dtype=np.float64).reshape(-1, values.size)
    if values.size != 60 or not np.isfinite(values).all() or not np.isfinite(rows).all():
        raise ValueError("Expected finite 60-dimensional motor delta and probes")
    energy = float(values @ values)
    _, singular, right = np.linalg.svd(rows, full_matrices=False)
    if not singular.size or singular[0] == 0:
        return 0.0, 0
    tolerance = np.finfo(np.float64).eps * max(rows.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if energy == 0.0:
        return 0.0, rank
    if rank == 0:
        return 0.0, 0
    coefficients = right[:rank] @ values
    fraction = float((coefficients @ coefficients) / energy)
    return min(1.0, max(0.0, fraction)), rank


def make_probes(torch, seed: int, count: int) -> np.ndarray:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    probes = torch.randint(0, 2, (count, 60), generator=generator, dtype=torch.int64)
    probes = probes.mul_(2).sub_(1).float().div_(60**0.5).reshape(count, 10, 6)
    return probes.numpy().copy()


def arm_label(row) -> str:
    arm = str(row.get("arm", ""))
    if arm == "row_action_vq_a4":
        if "action_lambda" not in row:
            raise ValueError("row_action_vq_a4 record is missing action_lambda")
        return f"row_action_vq_a4_lambda_{float(row['action_lambda']):g}"
    return arm


def read_action_records(path: Path, cases: list[int], arms: list[str]) -> dict:
    wanted_cases, wanted_arms = set(cases), set(arms)
    found = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "calibration" or int(row.get("case_id", -1)) not in wanted_cases:
                continue
            label = arm_label(row)
            if label not in wanted_arms:
                continue
            if int(row.get("seed_index", -1)) != 0:
                raise ValueError(f"Expected calibration seed_index 0 at {path}:{line_number}")
            key = (int(row["case_id"]), label)
            if key in found:
                raise ValueError(f"Duplicate calibration action record for {key}")
            if "action" not in row or "sampler_seed" not in row:
                raise ValueError(f"Missing action or sampler_seed at {path}:{line_number}")
            found[key] = row
    validate_coverage(found, cases, arms)
    return found


def validate_coverage(found: dict, cases: list[int], arms: list[str]) -> None:
    expected = {(case, arm) for case in cases for arm in arms}
    missing = sorted(expected - set(found))
    extra = sorted(set(found) - expected)
    if missing or extra or len(found) != len(expected):
        raise ValueError(f"Calibration action coverage mismatch: expected {len(expected)}, found {len(found)}, missing={missing}, extra={extra}")


def load_capture(torch, path: Path, seed: int, probes_per_batch: int):
    # mmap keeps unrelated captured layer tensors lazy; only these small fields are read.
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    baseline = payload["baseline_action"].detach().cpu().numpy().copy()
    saved_probes = payload["probe_vectors"].detach().cpu()
    regenerated = torch.from_numpy(make_probes(torch, seed, probes_per_batch))
    if tuple(saved_probes.shape) != (probes_per_batch, 10, 6):
        raise ValueError(f"Unexpected saved probe shape in {path}: {tuple(saved_probes.shape)}")
    if not torch.equal(saved_probes, regenerated):
        raise ValueError(f"Saved old probes do not match seed {seed} in {path}")
    summary = payload.get("summary", {})
    if summary.get("action_shape") not in (None, [32, 7]) or summary.get("action_slice") not in (None, ["first10", "motor6"]):
        raise ValueError(f"Capture summary action scope differs in {path}")
    return baseline, saved_probes.numpy().copy(), summary


def optional_summary_sampler_seed(summary):
    for key in ("sampler_seed", "query_sampler_seed"):
        if key in summary:
            return int(summary[key])
    return None


def save_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def mean(values) -> float:
    return float(statistics.mean(values))


def run(protocol_path: Path, out: Path) -> None:
    guard()
    if out.exists():
        raise FileExistsError(f"Refusing existing output path: {out}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_root = Path(protocol["source_root"])
    cases = [int(case) for case in protocol["calibration_cases"]]
    arms = [str(arm) for arm in protocol["frozen_arms"]]
    old_seed = int(protocol.get("old_probe_seed", protocol.get("old_probe_seed20261009", 20261009)))
    new_seeds = [int(seed) for seed in protocol["new_probe_seeds"]]
    probes_per_batch = int(protocol["probes_per_batch"])
    if len(cases) != 8 or len(set(cases)) != 8:
        raise ValueError("Protocol must name exactly eight distinct calibration cases")
    expected_arms = {
        "row_scale_vq_a4", "row_output_vq_a4",
        "row_action_vq_a4_lambda_0.1", "row_action_vq_a4_lambda_1",
    }
    if set(arms) != expected_arms or len(arms) != 4:
        raise ValueError(f"Frozen arms must be exactly {sorted(expected_arms)}")
    if old_seed != 20261009 or set(new_seeds) != {20261010, 20261011, 20261012, 20261013}:
        raise ValueError("Protocol probe seeds differ from the frozen diagnostic seeds")
    if probes_per_batch != 4:
        raise ValueError("This diagnostic expects four probes per batch")

    full = source_root / "results" / "full"
    actions_path = full / "actions.jsonl"
    action_rows = read_action_records(actions_path, cases, arms)
    input_plan_path = Path(protocol['roots']['old_inputs']) / 'plan.json'
    input_plan = json.loads(input_plan_path.read_text(encoding='utf-8'))
    planned_seeds = {int(row['case_id']): int(row['sampler_seeds'][0]) for row in input_plan['inputs']}
    torch = __import__("torch")
    new_probes = {seed: make_probes(torch, seed, probes_per_batch) for seed in new_seeds}
    per_case = {}
    for case in cases:
        capture_path = full / "sensitivity" / f"case_{case}.pt"
        baseline, old_probes, capture_summary = load_capture(torch, capture_path, old_seed, probes_per_batch)
        sampler_seeds = {int(action_rows[(case, arm)]["sampler_seed"]) for arm in arms}
        if len(sampler_seeds) != 1:
            raise ValueError(f"Calibration arms have mismatched sampler seeds for case {case}: {sorted(sampler_seeds)}")
        sampler_seed = next(iter(sampler_seeds))
        if sampler_seed != planned_seeds[case]:
            raise ValueError(f'Action sampler seed differs from the original input plan for case {case}')
        captured_seed = optional_summary_sampler_seed(capture_summary)
        if captured_seed is not None and captured_seed != sampler_seed:
            raise ValueError(f"Action sampler seed differs from capture summary for case {case}")
        old_matrix = old_probes.reshape(probes_per_batch, 60)
        case_rows = {"sampler_seed": sampler_seed, "capture_sampler_seed_check": "matched" if captured_seed is not None else "not_in_capture_summary",
                     "old_probe_span_rank": None, "arms": {}}
        for arm in arms:
            row = action_rows[(case, arm)]
            delta = motor_delta(row["action"], baseline)
            actual_rmse = motor_rmse(delta)
            old_values = projected_values(delta, old_matrix)
            span_fraction, span_rank = span_energy_fraction(delta, old_matrix)
            if case_rows["old_probe_span_rank"] is None:
                case_rows["old_probe_span_rank"] = span_rank
            elif case_rows["old_probe_span_rank"] != span_rank:
                raise AssertionError("Old probe span rank changed within a case")
            new_values = {str(seed): projected_values(delta, probes.reshape(probes_per_batch, 60)).tolist()
                          for seed, probes in new_probes.items()}
            all_new = np.asarray([value for values in new_values.values() for value in values], dtype=np.float64)
            logged_proxy = row.get("projected_action_rmse_sampled_proxy")
            if logged_proxy is None:
                raise ValueError(f"Missing logged sampled-proxy RMS for case {case}, arm {arm}")
            stored_rmse = row.get("motor_rmse_first10_vs_bf16")
            stored_check = None if stored_rmse is None else abs(float(stored_rmse) - actual_rmse)
            if stored_check is not None and stored_check > 2e-6:
                raise ValueError(f"Recomputed action RMSE differs from saved metric for case {case}, arm {arm}: {stored_check}")
            case_rows["arms"][arm] = {
                "actual_motor_rmse_first10": actual_rmse,
                "logged_sampled_proxy_rmse": None if logged_proxy is None else float(logged_proxy),
            "logged_proxy_minus_actual_old_four_projection_rms": None if logged_proxy is None else float(logged_proxy) - rms(old_values),
                "actual_old_four_signed_projections": old_values.tolist(),
            "actual_old_four_projection_rms": rms(old_values),
                "old_probe_span_energy_fraction": span_fraction,
                "new_projection_values_by_seed": new_values,
                "new_16_projection_rms": rms(all_new),
                "saved_metric_abs_difference": stored_check,
            }
        per_case[str(case)] = case_rows

    summaries = {}
    for arm in arms:
        rows = [per_case[str(case)]["arms"][arm] for case in cases]
        summaries[arm] = {
            "mean_actual_motor_rmse_first10": mean([row["actual_motor_rmse_first10"] for row in rows]),
            "mean_logged_sampled_proxy_rmse": mean([row["logged_sampled_proxy_rmse"] for row in rows if row["logged_sampled_proxy_rmse"] is not None]),
            "mean_actual_old_four_projection_rms": mean([row["actual_old_four_projection_rms"] for row in rows]),
            "mean_logged_proxy_minus_actual_old_four_projection_rms": mean([row["logged_proxy_minus_actual_old_four_projection_rms"] for row in rows if row["logged_proxy_minus_actual_old_four_projection_rms"] is not None]),
            "mean_old_probe_span_energy_fraction": mean([row["old_probe_span_energy_fraction"] for row in rows]),
            "mean_new_16_projection_rms": mean([row["new_16_projection_rms"] for row in rows]),
            "mean_new_projection_rms_by_seed": {
                str(seed): mean([
                    rms(per_case[str(case)]["arms"][arm]["new_projection_values_by_seed"][str(seed)])
                    for case in cases
                ]) for seed in new_seeds
            },
        }
    contrast_pairs = [("row_output_vq_a4", "row_scale_vq_a4"),
                     ("row_action_vq_a4_lambda_0.1", "row_output_vq_a4"),
                     ("row_action_vq_a4_lambda_1", "row_output_vq_a4"),
                     ("row_action_vq_a4_lambda_0.1", "row_scale_vq_a4"),
                     ("row_action_vq_a4_lambda_1", "row_scale_vq_a4")]
    contrasts = {}
    for treatment, reference in contrast_pairs:
        differences = {
            str(case): per_case[str(case)]["arms"][treatment]["actual_motor_rmse_first10"] -
                      per_case[str(case)]["arms"][reference]["actual_motor_rmse_first10"]
            for case in cases
        }
        contrasts[f"{treatment}_minus_{reference}"] = {
            "mean_paired_difference": mean(list(differences.values())),
            "improved_cases": sum(value < 0 for value in differences.values()),
            "worsened_cases": sum(value > 0 for value in differences.values()),
            "per_case_difference": differences,
        }
    summary = {
        "status": "complete",
        "phase": "original",
        "job_id": os.environ["PBS_JOBID"],
        "diagnostic": "original action projection coverage",
        "source": {
            "source_root": str(source_root),
            "actions_jsonl": str(actions_path),
            "sensitivity_artifacts": [str(full / "sensitivity" / f"case_{case}.pt") for case in cases],
            "source_full_job": protocol.get("source_full_job", "25743693.pbs101"),
        },
        "scope": {
            "calibration_cases": cases,
            "frozen_arms": arms,
            "record_count": len(action_rows),
            "original_target_linear_scope": 614,
            "original_campaign_modified": False,
            "sampled_metric_only": True,
        },
        "probe_design": {
            "old_seed": old_seed,
            "old_probe_validation": "exact match against saved sensitivity probe_vectors",
            "new_seeds": new_seeds,
            "probes_per_batch": probes_per_batch,
            "action_coordinates": 60,
            "projection_normalization": "each Rademacher vector has unit L2 norm; RMS across projections estimates the 60-coordinate motor RMSE",
        },
        "sampler_seed_check": {
            "arm_seeds_match_within_case": True,
            "original_input_plan_seed_index0_matched": True,
            "original_input_plan": str(input_plan_path),
            "capture_summary_seed": "checked when explicit sampler_seed or query_sampler_seed is present",
            "cases_without_capture_seed": [case for case in cases if per_case[str(case)]["capture_sampler_seed_check"] == "not_in_capture_summary"],
        },
        "per_arm_means": summaries,
        "paired_actual_motor_rmse_contrasts": contrasts,
        "limits": [
            "Four original probes span at most four directions in a 60-dimensional action slice; low old-span energy indicates a possible blind spot, not a model-selection result.",
            "Logged sampled-proxy vectors were not saved. Only their scalar RMS is compared with recomputed actual projection and motor RMSE scalars; signed proxy values are not inferred.",
            "The new independent probes are a descriptive coverage diagnostic on eight frozen calibration cases; they do not change the full 614-Linear scope or establish closed-loop success.",
        ],
    }
    out.mkdir(parents=True)
    save_json(out / "summary.json", summary)
    save_json(out / "per_case.json", per_case)
    lines = [
        "# 原始 action projection 覆盖诊断",
        "",
        f"来源 campaign：`{summary['source']['source_root']}`（full job `{summary['source']['source_full_job']}`）。",
        f"固定范围：{len(cases)} 个 calibration cases × {len(arms)} 个 arms，共 {len(action_rows)} 条 action 记录；原 campaign 的 614 个 target Linears 范围不变。",
        "",
        "| Arm | 实际 motor RMSE | 日志 proxy RMSE | 原 4 个 probe 的实际投影 RMS | 新 16 个投影 RMS | 原 probe span 能量占比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        item = summaries[arm]
        lines.append(f"| `{arm}` | {item['mean_actual_motor_rmse_first10']:.6g} | {item['mean_logged_sampled_proxy_rmse']:.6g} | {item['mean_actual_old_four_projection_rms']:.6g} | {item['mean_new_16_projection_rms']:.6g} | {item['mean_old_probe_span_energy_fraction']:.4f} |")
    lines += ["", "配对差值使用实际前 10 步、6 个 motor 的 RMSE；负值表示前一个 arm 更低。", ""]
    for name, item in contrasts.items():
        lines.append(f"- `{name}`: mean paired difference {item['mean_paired_difference']:.6g}; improved in {item['improved_cases']}/8 cases, worsened in {item['worsened_cases']}/8.")
    lines += ["", "没有保存 sampled-proxy 向量，因此这里只比较 scalar RMS，不重构 signed proxy projection。该诊断仅描述固定输入 calibration observations，不用于 test-set 选择，也不支持 closed-loop 结论。", ""]
    (out / "ORIGINAL_PROJECTIONS.zh.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.out)


if __name__ == "__main__":
    main()
