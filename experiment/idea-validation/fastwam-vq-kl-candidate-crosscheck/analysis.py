"""Cross-score cached MSE- and KL-gradient candidates on the frozen C80 cases."""
import argparse
import json
import math
import os
from pathlib import Path
import socket


PROTOCOL = "fastwam-vq-kl-candidate-crosscheck-v1"
CASES = list(range(2000, 2080))
DIRECTIONS = ("saved_adam", "fresh_adam")
FRACTIONS = (1.0, 0.5, 0.25)
EXPECTED_SCALE = {
    "action_kl_coefficient": 0.012302479276508236,
    "latent_kl_coefficient": 0.31297895224548067,
    "original_latent_weight": 0.04245452683169643,
}
ZERO_COUNTS = ("new_model_forwards", "new_teacher_forwards", "new_backwards", "training_updates")
MAX_JSON_BYTES = 200 * 1024


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def close(a, b, rtol=1e-6, atol=1e-9):
    return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)


def guard():
    job, nodefile = os.environ.get("PBS_JOBID", "").strip(), os.environ.get("PBS_NODEFILE", "").strip()
    host = socket.getfqdn().lower().rstrip(".")
    require(job and nodefile and Path(nodefile).is_file(), "A real PBS allocation is required")
    require("login" not in host, f"Refusing analysis on login host {host}")
    members = {row.strip().lower().rstrip(".") for row in Path(nodefile).read_text().splitlines() if row.strip()}
    require(host in members or host.split(".", 1)[0] in {name.split(".", 1)[0] for name in members},
            "Current host is absent from PBS_NODEFILE")
    return job


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def finite_nonnegative(value, label):
    number = float(value)
    require(math.isfinite(number) and number >= 0.0, f"Invalid nonnegative metric: {label}")
    return number


def job_ids(root, unit):
    handles = read_json(Path(root) / "control" / "handles.json")
    for container in (handles, handles.get("arms", {}), handles.get("units", {})):
        entry = container.get(unit) if isinstance(container, dict) else None
        if isinstance(entry, dict) and isinstance(entry.get("job_ids"), list):
            return {str(value) for value in entry["job_ids"]}
    raise RuntimeError(f"No {unit}.job_ids list in {root}/control/handles.json")


def label(direction, fraction):
    return f"{direction}_f{str(float(fraction)).replace('.', 'p')}"


def score(row, formula, scale):
    if formula == "mse_nmse":
        return row["action_mse"] + scale["original_latent_weight"] * row["latent_nmse"]
    return (scale["action_kl_coefficient"] * row["action_kl"]
            + scale["original_latent_weight"] * scale["latent_kl_coefficient"] * row["latent_kl"])


def normalize_metrics(row, context):
    keys = ("action_mse", "latent_nmse", "action_kl", "latent_kl")
    values = {key: finite_nonnegative(row[key], f"{context}/{key}") for key in keys}
    values["action_rmse"] = math.sqrt(values["action_mse"])
    return values


def mean(values):
    require(bool(values), "Cannot average an empty list")
    return sum(values) / len(values)


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def validate_protocol(p):
    require(p.get("protocol") == PROTOCOL and p.get("calibration_case_ids") == CASES
            and p.get("seed_index") == 0 and p.get("directions") == list(DIRECTIONS)
            and p.get("fractions") == list(FRACTIONS) and p.get("heldout_used") is False,
            "Frozen C80 cross-check protocol differs")
    require(all(p.get(key) == 0 for key in ZERO_COUNTS), "This task must remain zero-forward/backward/update")
    require(p.get("source_mse_job") == "25862598.pbs101"
            and p.get("source_geometry_job") == "25882486.pbs101",
            "Frozen source job bindings differ")


def validate_sources(p):
    source_root = Path(p["source_root"])
    old = read_json(source_root / "protocol.json")
    scale = read_json(source_root / "results" / "prepare" / "loss_scaling.json")
    require(old.get("remote_root") == str(source_root)
            and old.get("calibration", {}).get("case_ids") == CASES
            and old.get("calibration", {}).get("seed_index") == 0
            and old.get("source_per_channel_root") == p["source_mse_root"]
            and old.get("source_per_channel_job") == p["source_mse_job"],
            "Original dual-KL source protocol does not bind the MSE80 source/cohort")
    require(scale.get("protocol") == old["protocol"] and scale.get("heldout_used") is False
            and scale.get("case_ids") == CASES and scale.get("seed_indices") == [0],
            "Frozen C80 loss scaling is incomplete or includes held-out data")
    for key, expected in EXPECTED_SCALE.items():
        require(close(scale.get(key, float("nan")), expected), f"Frozen loss coefficient differs: {key}")
    return old, scale


def validate_geometry(p):
    root = Path(p["source_geometry_root"])
    data = read_json(root / "results" / "summary.json")
    require(data.get("protocol") == "fastwam-vq-kl-gradient-geometry-v1"
            and data.get("complete") is True and data.get("pbs_jobid") == p["source_geometry_job"]
            and data.get("source_mse_job") == p["source_mse_job"]
            and data.get("source_parity_passed") is True
            and data.get("baseline_metrics_match") is True
            and data.get("candidate_checkpoints_passed") is True
            and data.get("heldout_used") is False
            and all(data.get(key) == 0 for key in ZERO_COUNTS),
            "Cached gradient-geometry summary failed its provenance/scope gates")
    rows = data.get("projected_step_alignment", [])
    require(len(rows) == 6, "Gradient-geometry summary must contain six family/fraction step comparisons")
    indexed = {}
    for row in rows:
        key = (row.get("optimizer"), float(row.get("fraction", -1)))
        require(key not in indexed and key[0] in DIRECTIONS and key[1] in FRACTIONS,
                "Gradient-geometry step alignment keys are invalid")
        groups = row.get("groups", {})
        for group in ("ratio", "channel", "overall"):
            item = groups.get(group, {})
            cosine = item.get("cosine")
            nm, nk = float(item.get("norm_mse_step", float("nan"))), float(item.get("norm_kl_step", float("nan")))
            require(math.isfinite(nm) and math.isfinite(nk) and nm >= 0 and nk >= 0
                    and (cosine is None or math.isfinite(float(cosine))),
                    "Gradient-geometry cosine/norm is invalid")
        indexed[key] = groups
    require(set(indexed) == {(direction, fraction) for direction in DIRECTIONS for fraction in FRACTIONS},
            "Gradient-geometry family/fraction coverage differs")
    return data, indexed


def load_baseline(p, old, scale):
    root = Path(p["source_root"])
    rows = {}
    for case in CASES:
        raw = read_json(root / "results" / "calibration" / "mse_nmse" / "queries" / f"{case}_0.json")
        require(raw.get("protocol") == old["protocol"] and raw.get("arm") == "mse_nmse"
                and raw.get("split") == "calibration" and raw.get("updates") == 80
                and raw.get("case_id") == case and raw.get("seed_index") == 0,
                f"Original MSE80 baseline binding differs for case {case}")
        # The original MSE80 file uses these legacy names for the same fixed metrics.
        metrics = normalize_metrics({**raw, "action_mse": raw["first10motor_mse"],
                                    "latent_nmse": raw["future_latent_nmse"]}, f"baseline/{case}")
        objective = finite_nonnegative(raw.get("objective_loss"), f"baseline/{case}/objective_loss")
        require(close(objective, score(metrics, "mse_nmse", scale)), f"Baseline own objective differs for {case}")
        rows[case] = {**metrics, "suite": raw["suite"], "plan_row": raw["plan_row"],
                      "sampler_seed": raw["sampler_seed"], "objective_mse": objective}
    suites = sorted({row["suite"] for row in rows.values()})
    require(len(suites) == 4 and all(sum(row["suite"] == suite for row in rows.values()) == 20 for suite in suites),
            "Baseline cohort must have four equally sized suites")
    return rows, suites


def candidate_summary_index(summary, arm):
    if arm == "mse":
        require(summary.get("complete") is True and summary.get("heldout_used") is False,
                "MSE candidate summary is incomplete or heldout-selected")
        candidates = summary.get("selection", {}).get("candidates", [])
        result = {}
        for row in candidates:
            key = (row.get("direction"), float(row.get("fraction", -1)))
            require(key not in result and row.get("queries") == 80,
                    "MSE summary candidate keys/query count are invalid")
            result[key] = finite_nonnegative(row.get("objective_mean"), "mse summary objective_mean")
        require(set(result) == {(d, f) for d in DIRECTIONS for f in FRACTIONS},
                "MSE summary does not contain exactly six candidates")
        return result
    require(summary.get("complete") is True and summary.get("checkpoint_arm") == "mse_nmse"
            and summary.get("loss") == "dual_kl" and summary.get("heldout_used") is False,
            "KL candidate summary is incomplete or has the wrong checkpoint/loss")
    return None


def candidate_rows(p, arm, baseline, suites, scale, allowed_jobs, own_summary):
    root = Path(p["mse_candidate_root"] if arm == "mse" else p["kl_candidate_root"])
    unit = "mse_nmse" if arm == "mse" else "mse_dual"
    protocol = "fastwam-vq-kl-fullbatch-refine-v1" if arm == "mse" else "fastwam-vq-kl-direction-v1"
    output, by_condition = [], {}
    if arm == "kl":
        direction_summary = read_json(root / "results" / unit / "direction_summary.json")
        require(isinstance(direction_summary, list) and len(direction_summary) == 6,
                "KL direction_summary must contain six complete conditions")
        own_summary = {}
        for row in direction_summary:
            key = (row.get("direction"), float(row.get("fraction", -1)))
            require(key not in own_summary and row.get("queries") == 80,
                    "KL direction_summary key/query count is invalid")
            own_summary[key] = finite_nonnegative(row.get("objective"), "KL direction_summary objective")
        require(set(own_summary) == {(d, f) for d in DIRECTIONS for f in FRACTIONS},
                "KL direction_summary grid differs")
    for direction in DIRECTIONS:
        for fraction in FRACTIONS:
            key = (direction, fraction)
            label_text = label(direction, fraction)
            case_rows = {}
            for case in CASES:
                path = (root / "results" / unit / "candidates" / label_text / f"{case}_0.json"
                        if arm == "mse" else
                        root / "results" / unit / "directions" / label_text / f"{case}_0.json")
                raw = read_json(path)
                require(raw.get("protocol") == protocol and raw.get("unit") == unit
                        and raw.get("case_id") == case and raw.get("seed_index") == 0
                        and str(raw.get("pbs_jobid")) in allowed_jobs,
                        f"{arm} candidate protocol/case/seed/PBS binding differs: {label_text}/{case}")
                if arm == "mse":
                    require(raw.get("condition") == label_text and raw.get("candidate_direction") == direction
                            and float(raw.get("candidate_fraction", -1)) == fraction
                            and raw.get("split") == "calibration" and raw.get("source_updates") == 80
                            and raw.get("refinement_updates") == 1 and raw.get("source_record_reused") is False
                            and raw.get("plan_row") == baseline[case]["plan_row"]
                            and raw.get("sampler_seed") == baseline[case]["sampler_seed"],
                            f"MSE candidate frozen source/condition binding differs: {label_text}/{case}")
                else:
                    require(raw.get("direction") == direction and float(raw.get("fraction", -1)) == fraction
                            and raw.get("suite") == baseline[case]["suite"]
                            and raw.get("sampler_seed") == baseline[case]["sampler_seed"],
                            f"KL candidate frozen source/condition binding differs: {label_text}/{case}")
                metrics = normalize_metrics(raw, f"{arm}/{label_text}/{case}")
                own = "mse_nmse" if arm == "mse" else "dual_kl"
                calculated = score(metrics, own, scale)
                field = "objective_loss" if arm == "mse" else "objective"
                recorded = finite_nonnegative(raw.get(field), f"{arm}/{label_text}/{case}/{field}")
                require(close(recorded, calculated), f"{arm} per-case own objective differs: {label_text}/{case}")
                if arm == "mse":
                    require(raw.get("first10motor_mse") is not None and raw.get("future_latent_nmse") is not None
                            and close(raw["first10motor_mse"], metrics["action_mse"])
                            and close(raw["future_latent_nmse"], metrics["latent_nmse"]),
                            f"MSE candidate metric aliases differ: {label_text}/{case}")
                require(raw.get("suite") == baseline[case]["suite"],
                        f"{arm} candidate suite differs from baseline: {label_text}/{case}")
                case_rows[case] = {**metrics, "suite": raw["suite"],
                                   "objective_mse": score(metrics, "mse_nmse", scale),
                                   "objective_kl": score(metrics, "dual_kl", scale)}
            require(len(case_rows) == 80 and all(sum(row["suite"] == s for row in case_rows.values()) == 20
                                                  for s in suites),
                    f"{arm}/{label_text} does not cover balanced C80")
            suite_objective = {formula: {s: mean([score(row, formula, scale) for row in case_rows.values()
                                                   if row["suite"] == s]) for s in suites}
                               for formula in ("mse_nmse", "dual_kl")}
            means = {formula: mean(list(suite_objective[formula].values()))
                     for formula in ("mse_nmse", "dual_kl")}
            expected = own_summary[key]
            require(close(means["mse_nmse" if arm == "mse" else "dual_kl"], expected),
                    f"{arm}/{label_text} C80 own-objective mean differs from summary")
            by_condition[key] = case_rows
            output.append({"arm": arm, "direction": direction, "fraction": fraction,
                           "label": label_text, "records": len(case_rows),
                           "own_objective_mean": expected, "formula_means": means,
                           "formula_suite_means": suite_objective,
                           "mean_case_motor_rmse": mean([row["action_rmse"] for row in case_rows.values()]),
                           "mean_latent_nmse": mean([row["latent_nmse"] for row in case_rows.values()]),
                           "mean_action_kl": mean([row["action_kl"] for row in case_rows.values()]),
                           "mean_latent_kl": mean([row["latent_kl"] for row in case_rows.values()])})
    return output, by_condition


def against_baseline(condition, baseline, suites, formula, scale):
    deltas = {case: score(row, formula, scale) - score(baseline[case], formula, scale)
              for case, row in condition.items()}
    base_mean = mean([score(row, formula, scale) for row in baseline.values()])
    delta_mean = mean(list(deltas.values()))
    suite_delta = {suite: mean([deltas[c] for c in deltas if baseline[c]["suite"] == suite]) for suite in suites}
    return {"mean_delta": mean(list(suite_delta.values())),
            "delta_pct_of_baseline_mean": 100.0 * delta_mean / base_mean if base_mean else None,
            "median_case_delta": median(list(deltas.values())),
            "improved_cases": sum(value < 0 for value in deltas.values()),
            "worsened_cases": sum(value > 0 for value in deltas.values()),
            "tied_cases": sum(value == 0 for value in deltas.values()),
            "suite_mean_deltas": suite_delta}


def matched_pair(mse_rows, kl_rows, baseline, suites, scale):
    result = {}
    for formula in ("mse_nmse", "dual_kl"):
        deltas = {case: score(kl_rows[case], formula, scale) - score(mse_rows[case], formula, scale)
                  for case in CASES}
        suite_deltas = {suite: mean([deltas[case] for case in CASES if baseline[case]["suite"] == suite])
                        for suite in suites}
        result[formula] = {"kl_candidate_minus_mse_candidate_mean_delta": mean(list(suite_deltas.values())),
                           "median_case_delta": median(list(deltas.values())),
                           "improved_cases_for_kl_candidate": sum(value < 0 for value in deltas.values()),
                           "worsened_cases_for_kl_candidate": sum(value > 0 for value in deltas.values()),
                           "tied_cases": sum(value == 0 for value in deltas.values()),
                           "suite_mean_deltas": suite_deltas}
    action_bounds = [abs(kl_rows[c]["action_rmse"] - mse_rows[c]["action_rmse"]) for c in CASES]
    latent_bounds = [abs(math.sqrt(kl_rows[c]["latent_nmse"]) - math.sqrt(mse_rows[c]["latent_nmse"]))
                     for c in CASES]
    return {"direction": None, "fraction": None, "formula_deltas": result,
            "mean_case_motor_rmse_delta_kl_minus_mse": mean([kl_rows[c]["action_rmse"] - mse_rows[c]["action_rmse"]
                                                               for c in CASES]),
            "mean_latent_nmse_delta_kl_minus_mse": mean([kl_rows[c]["latent_nmse"] - mse_rows[c]["latent_nmse"]
                                                          for c in CASES]),
            "mean_action_kl_delta_kl_minus_mse": mean([kl_rows[c]["action_kl"] - mse_rows[c]["action_kl"]
                                                        for c in CASES]),
            "mean_latent_kl_delta_kl_minus_mse": mean([kl_rows[c]["latent_kl"] - mse_rows[c]["latent_kl"]
                                                        for c in CASES]),
            "student_output_distance_lower_bound_only": {
                "action_rms_mean": mean(action_bounds), "action_rms_max": max(action_bounds),
                "sqrt_latent_nmse_mean": mean(latent_bounds), "sqrt_latent_nmse_max": max(latent_bounds)}}


def step_distances(indexed):
    output = []
    for direction in DIRECTIONS:
        for fraction in FRACTIONS:
            groups = {}
            for group, values in indexed[(direction, fraction)].items():
                cosine = values["cosine"]
                nm, nk = float(values["norm_mse_step"]), float(values["norm_kl_step"])
                distance = None if cosine is None else math.sqrt(max(0.0, nm * nm + nk * nk - 2 * nm * nk * float(cosine)))
                groups[group] = {"cosine": cosine, "mse_step_norm": nm, "kl_step_norm": nk,
                                 "distance": distance,
                                 "distance_relative_to_mse_step_norm": distance / nm if distance is not None and nm else None}
            output.append({"optimizer": direction, "fraction": fraction, "groups": groups})
    return output


def make_report(summary):
    lines = [
        "# Cached KL candidate cross-check", "",
        "这是同一冻结 C80、同一 MSE80 起点上的两个已有候选集交叉评分。只读取 JSON；没有 held-out、forward、backward、teacher 查询或训练更新。各套每个 condition 80 cases，四个 suite 各 20。", "",
        f"公式：`Jm = action_mse + {summary['loss_scaling']['original_latent_weight']:.9g} × latent_nmse`；"
        f"`Jk = {summary['loss_scaling']['action_kl_coefficient']:.9g} × action_kl + "
        f"{summary['loss_scaling']['original_latent_weight']:.9g} × {summary['loss_scaling']['latent_kl_coefficient']:.9g} × latent_kl`。"
        "两公式分别交叉评分；各自系数固定来自原有 C80 标定。", "",
        "## 候选集得分与原始指标均值", "",
        "| 候选集 | family | fraction | Jm mean | Jk mean | mean case motor RMSE | latent NMSE | action KL | latent KL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base = summary["baseline_metrics"]
    lines.append(f"| original MSE80 | source/no-step | 0 | "
                 f"{summary['baseline_formula_means']['mse_nmse']:.8g} | "
                 f"{summary['baseline_formula_means']['dual_kl']:.8g} | "
                 f"{base['mean_case_motor_rmse']:.8g} | {base['mean_latent_nmse']:.8g} | "
                 f"{base['mean_action_kl']:.8g} | {base['mean_latent_kl']:.8g} |")
    for row in summary["conditions"]:
        lines.append(f"| {row['arm']} | {row['direction']} | {row['fraction']} | "
                     f"{row['formula_means']['mse_nmse']:.8g} | {row['formula_means']['dual_kl']:.8g} | "
                     f"{row['mean_case_motor_rmse']:.8g} | {row['mean_latent_nmse']:.8g} | "
                     f"{row['mean_action_kl']:.8g} | {row['mean_latent_kl']:.8g} |")
    lines += ["", "## 相对共同 no-step 的配对变化", "",
              "Δ 为每 case 候选减 baseline；负值表示公式分数下降。suite 列按四个 suite 等权列出 mean Δ。", "",
              "| 候选 | 公式 | mean Δ | mean Δ% | median case Δ | improved cases | suite mean Δ |",
              "|---|---|---:|---:|---:|---:|---|"]
    for row in summary["conditions"]:
        for formula in ("mse_nmse", "dual_kl"):
            item = row["vs_common_baseline"][formula]
            suites = ", ".join(f"{name}:{value:.6g}" for name, value in item["suite_mean_deltas"].items())
            lines.append(f"| {row['arm']}/{row['direction']}/{row['fraction']} | {formula} | "
                         f"{item['mean_delta']:.8g} | {item['delta_pct_of_baseline_mean']} | "
                         f"{item['median_case_delta']:.8g} | {item['improved_cases']} | {suites} |")
    lines += ["", "## 同 family/fraction 下 KL 候选减 MSE 候选", "",
              "目标差值定义为 KL 候选减 MSE 候选，负值代表对应公式分数较低。配对 80 个 C80 case；这些是固定队列的描述统计，不是显著性或跨训练轨迹推断。", "",
              "| family | fraction | Jm mean/median Δ | Jm improved | Jm suite Δ | Jk mean/median Δ | Jk improved | Jk suite Δ | motor RMSE Δ | latent NMSE Δ | action lower bound mean/max | latent lower bound mean/max |",
              "|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|"]
    for pair in summary["matched_pairs"]:
        jm, jk, bound = pair["formula_deltas"]["mse_nmse"], pair["formula_deltas"]["dual_kl"], pair["student_output_distance_lower_bound_only"]
        jm_suite = ", ".join(f"{name}:{value:.6g}" for name, value in jm["suite_mean_deltas"].items())
        jk_suite = ", ".join(f"{name}:{value:.6g}" for name, value in jk["suite_mean_deltas"].items())
        lines.append(f"| {pair['direction']} | {pair['fraction']} | "
                     f"{jm['kl_candidate_minus_mse_candidate_mean_delta']:.8g}/{jm['median_case_delta']:.8g} | "
                     f"{jm['improved_cases_for_kl_candidate']} | {jm_suite} | "
                     f"{jk['kl_candidate_minus_mse_candidate_mean_delta']:.8g}/{jk['median_case_delta']:.8g} | "
                     f"{jk['improved_cases_for_kl_candidate']} | {jk_suite} | "
                     f"{pair['mean_case_motor_rmse_delta_kl_minus_mse']:.8g} | "
                     f"{pair['mean_latent_nmse_delta_kl_minus_mse']:.8g} | "
                     f"action {bound['action_rms_mean']:.8g}/{bound['action_rms_max']:.8g} | "
                     f"{bound['sqrt_latent_nmse_mean']:.8g}/{bound['sqrt_latent_nmse_max']:.8g} |")
    lines += ["", "下界来自同 target 上 RMSE 差的三角不等式；KL 记录没有 raw action/latent 输出，因此不能由此得到 student 间实际距离。参数步距离只在 ratio/channel 参数子空间计算，不是全模型参数比例；cosine 接近 1 也不证明两个目标具有相同损失变化。没有据此将差异归因为 STE、D 或任何单一机制。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    job = guard()
    p = read_json(args.protocol)
    validate_protocol(p)
    old, scale = validate_sources(p)
    geometry, geometry_index = validate_geometry(p)
    baseline, suites = load_baseline(p, old, scale)
    mse_root, kl_root = Path(p["mse_candidate_root"]), Path(p["kl_candidate_root"])
    mse_jobs, kl_jobs = job_ids(mse_root, "mse_nmse"), job_ids(kl_root, "mse_dual")
    mse_summary = read_json(mse_root / "results" / "mse_nmse" / "summary.json")
    kl_summary = read_json(kl_root / "results" / "mse_dual" / "summary.json")
    mse_own = candidate_summary_index(mse_summary, "mse")
    candidate_summary_index(kl_summary, "kl")
    mse_candidates, mse_rows = candidate_rows(p, "mse", baseline, suites, scale, mse_jobs, mse_own)
    kl_candidates, kl_rows = candidate_rows(p, "kl", baseline, suites, scale, kl_jobs, None)
    conditions, pairs = [], []
    for arm, candidate_list, indexed in (("mse", mse_candidates, mse_rows), ("kl", kl_candidates, kl_rows)):
        for row in candidate_list:
            key = (row["direction"], row["fraction"])
            case_rows = indexed[key]
            row["vs_common_baseline"] = {
                formula: against_baseline(case_rows, baseline, suites, formula, scale)
                for formula in ("mse_nmse", "dual_kl")}
            conditions.append(row)
    for direction in DIRECTIONS:
        for fraction in FRACTIONS:
            pair = matched_pair(mse_rows[(direction, fraction)], kl_rows[(direction, fraction)],
                                baseline, suites, scale)
            pair["direction"], pair["fraction"] = direction, fraction
            pairs.append(pair)
    summary = {
        "protocol": PROTOCOL, "complete": True, "pbs_jobid": job, "heldout_used": False,
        "new_model_forwards": 0, "new_teacher_forwards": 0, "new_backwards": 0, "training_updates": 0,
        "source_mse_job": p["source_mse_job"], "baseline_records": 80,
        "baseline_formula_means": {
            formula: mean([score(row, formula, scale) for row in baseline.values()])
            for formula in ("mse_nmse", "dual_kl")},
        "baseline_metrics": {
            "mean_case_motor_rmse": mean([row["action_rmse"] for row in baseline.values()]),
            "mean_latent_nmse": mean([row["latent_nmse"] for row in baseline.values()]),
            "mean_action_kl": mean([row["action_kl"] for row in baseline.values()]),
            "mean_latent_kl": mean([row["latent_kl"] for row in baseline.values()])},
        "candidate_records": {"mse": 480, "kl": 480}, "baseline_source_same": True,
        "source_geometry_validated": True, "source_geometry_job": p["source_geometry_job"],
        "source_geometry_step_distances": step_distances(geometry_index),
        "loss_scaling": {**{key: scale[key] for key in EXPECTED_SCALE}, "heldout_used": False},
        "suite_case_counts": {suite: 20 for suite in suites}, "conditions": conditions,
        "matched_pairs": pairs,
        "evidence_scope": "fixed-input cached full-C80 records; no significance, training-replication, causal, or closed-loop claim",
    }
    require(len(summary["conditions"]) == 12 and len(summary["matched_pairs"]) == 6,
            "Final C80 cross-check coverage differs")
    output = Path(p["remote_root"]) / "results"
    output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    require(len(payload) < MAX_JSON_BYTES, "summary.json exceeds 200 KiB")
    temp = output / "summary.json.tmp"
    temp.write_bytes(payload)
    temp.replace(output / "summary.json")
    (output / "ANALYSIS.zh.md").write_text(make_report(summary), encoding="utf-8")
    print("KL_CANDIDATE_CROSSCHECK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
