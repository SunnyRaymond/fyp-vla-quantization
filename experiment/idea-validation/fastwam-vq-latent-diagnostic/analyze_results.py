"""Analyze a frozen Fast-WAM latent diagnostic scope on a PBS compute node."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket
import statistics


EXPECTED_ARMS = (
    "bf16", "parent_self", "parent_teacher_future",
    "distilled_self", "distilled_teacher_future",
)
ACTION_METRICS = ("first10motor_rmse", "full32motor_rmse", "late22motor_rmse")
LATENT_METRICS = (
    "future_latent_rmse", "future_latent_nmse",
    "temporal_delta_global_rmse", "temporal_delta_global_nmse",
)
LATENT_PROFILE_METRICS = (
    "future_rmse", "future_nmse", "temporal_delta_rmse", "temporal_delta_nmse",
)
MAX_ARTIFACT_BYTES = 256 * 1024


def guard():
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE")
    if not os.environ.get("PBS_JOBID") or not nodefile or "login" in host.lower():
        raise RuntimeError("Analysis requires a genuine PBS compute allocation")
    nodes = {line.split(".")[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    if host not in nodes:
        raise RuntimeError("This host is outside the PBS allocation")


def read_json(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise RuntimeError(f"Missing or oversized bounded JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_number(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)), f"{name} must be a finite number")
    return float(value)


def validate_shard_summary(summary, shard, contexts, protocol):
    require(summary.get("status") == "complete", f"shard {shard} summary is not complete")
    require(summary.get("protocol") == protocol["protocol"], f"shard {shard} protocol mismatch")
    require(summary.get("shard") == shard, f"shard summary identity mismatch: {shard}")
    require(bool(summary.get("job_id")), f"shard {shard} summary has no PBS job id")
    require(summary.get("context_count") == 30 and summary.get("expected_context_count") == 30,
            f"shard {shard} must complete exactly 30 contexts")
    require(summary.get("native_context_queries") == 90,
            f"shard {shard} must contain 90 native queries")
    require(summary.get("teacher_future_queries") == 60,
            f"shard {shard} must contain 60 teacher-future queries")
    require(summary.get("selected_epoch") == 2, f"shard {shard} did not use frozen epoch 2")
    require(summary.get("target_linears") == protocol["target_linears"],
            f"shard {shard} target Linear count mismatch")
    require(summary.get("episodes") == 0, f"shard {shard} unexpectedly reports episodes")

    parity = summary.get("self_future_parity")
    require(isinstance(parity, dict) and set(parity) == {"bf16", "parent", "distilled"},
            f"shard {shard} self-future parity must cover bf16, parent, and distilled")
    valid_contexts = {(row["case_id"], row["seed_index"]) for row in contexts}
    for arm, value in parity.items():
        require(isinstance(value, dict), f"shard {shard} parity result for {arm} is malformed")
        require((value.get("case_id"), value.get("seed_index")) in valid_contexts,
                f"shard {shard} parity result for {arm} is outside its frozen shard")
        max_abs = finite_number(value.get("max_abs"), f"{arm} parity max_abs")
        rmse = finite_number(value.get("rmse"), f"{arm} parity rmse")
        require(max_abs <= protocol["parity_max_abs_tolerance"]
                and rmse <= protocol["parity_rmse_tolerance"],
                f"shard {shard} self-future parity failed for {arm}")
    return {
        "job_id": summary["job_id"],
        "context_count": summary["context_count"],
        "native_context_queries": summary["native_context_queries"],
        "teacher_future_queries": summary["teacher_future_queries"],
        "selected_epoch": summary["selected_epoch"],
        "self_future_parity": parity,
    }


def load_receipts(protocol, results_root, contexts_by_shard, selected_shards):
    rows = []
    case_metadata = {}
    shard_gates = {}
    expected_contexts = set()

    for shard in selected_shards:
        shard_contexts = contexts_by_shard[shard]
        require(len(shard_contexts) == protocol["contexts_per_shard"],
                f"selected shard {shard} must contain all 30 frozen contexts")
        directory = Path(results_root) / f"shard-{shard}"
        summary = read_json(directory / "summary.json")
        shard_gates[str(shard)] = validate_shard_summary(summary, shard, shard_contexts, protocol)
        native_count = teacher_count = 0

        for context in shard_contexts:
            pair = (context["split"], context["case_id"], context["seed_index"])
            require(pair not in expected_contexts, f"duplicate frozen context: {pair!r}")
            expected_contexts.add(pair)
            stem = f"case_{context['case_id']}_seed_{context['seed_index']}"
            receipt = read_json(directory / f"{stem}.json")
            require(receipt.get("protocol") == protocol["protocol"] and receipt.get("context") == context,
                    f"receipt context/protocol mismatch: shard {shard} {stem}")
            require(receipt.get("native_queries") == 3 and receipt.get("teacher_future_queries") == 2,
                    f"receipt query counts mismatch: shard {shard} {stem}")
            native_count += receipt["native_queries"]
            teacher_count += receipt["teacher_future_queries"]

            arms = receipt.get("arms")
            require(isinstance(arms, list) and len(arms) == len(EXPECTED_ARMS)
                    and set(arms) == set(EXPECTED_ARMS),
                    f"receipt must contain all five arms: shard {shard} {stem}")
            receipt_rows = receipt.get("rows")
            require(isinstance(receipt_rows, list) and len(receipt_rows) == len(EXPECTED_ARMS),
                    f"receipt must contain five metric rows: shard {shard} {stem}")
            row_arms = set()
            for row in receipt_rows:
                require(row.get("case_id") == context["case_id"]
                        and row.get("seed_index") == context["seed_index"]
                        and row.get("split") == context["split"],
                        f"row context mismatch: shard {shard} {stem}")
                arm = row.get("arm")
                require(arm in EXPECTED_ARMS and arm not in row_arms,
                        f"duplicate or unknown arm in receipt: shard {shard} {stem}")
                row_arms.add(arm)
                domain = str(row.get("domain", "")).strip().lower()
                dimension = str(row.get("dimension", "")).strip()
                require(domain in ("original", "plus") and dimension,
                        f"missing domain/dimension metadata: shard {shard} {stem}")
                metadata = (domain, dimension)
                previous = case_metadata.setdefault(context["case_id"], metadata)
                require(previous == metadata, f"case metadata changed across rows: {context['case_id']}")
                rows.append(row)
            require(row_arms == set(EXPECTED_ARMS), f"incomplete arm rows: shard {shard} {stem}")

        require(native_count == 3 * len(shard_contexts)
                and teacher_count == 2 * len(shard_contexts),
                f"receipt totals mismatch for shard {shard}")

    expected_context_count = protocol["contexts_per_shard"] * len(selected_shards)
    require(len(expected_contexts) == expected_context_count
            and len(rows) == len(expected_contexts) * len(EXPECTED_ARMS),
            f"selected scope requires exactly {expected_context_count} contexts and "
            f"{expected_context_count * len(EXPECTED_ARMS)} arm rows")
    return rows, case_metadata, shard_gates


def build_scope(protocol, contexts_by_shard, selected_shards, case_metadata=None):
    contexts = [context for shard in selected_shards for context in contexts_by_shard[shard]]
    split_case_counts = {
        split: len({row["case_id"] for row in contexts if row["split"] == split})
        for split in ("calibration", "selection", "test")
    }
    split_context_counts = {
        split: sum(row["split"] == split for row in contexts)
        for split in ("calibration", "selection", "test")
    }
    test_cases = [case_id for case_id in protocol["test_cases"]
                  if any(row["split"] == "test" and row["case_id"] == case_id for row in contexts)]
    test_domain_case_counts = None
    if case_metadata is not None:
        test_domain_case_counts = {
            domain: sum(case_metadata[case_id][0] == domain for case_id in test_cases)
            for domain in ("original", "plus")
        }
    full = tuple(selected_shards) == (0, 1)
    return {
        "kind": "full_cohort" if full else "shard_interim",
        "selected_shards": list(selected_shards),
        "context_count": len(contexts),
        "expected_context_count": protocol["total_contexts"],
        "test_case_count": len(test_cases),
        "expected_test_case_count": len(protocol["test_cases"]),
        "split_case_counts": split_case_counts,
        "split_context_counts": split_context_counts,
        "test_domain_case_counts": test_domain_case_counts,
        "test_cases": test_cases,
        "full_cohort_complete": full,
    }


def scope_label(scope):
    if scope["full_cohort_complete"]:
        return (f"full frozen cohort，{scope['context_count']}/{scope['expected_context_count']} contexts，"
                f"{scope['test_case_count']}/{scope['expected_test_case_count']} testcases，"
                "full_cohort_complete=true")
    shard = scope["selected_shards"][0]
    return (f"shard{shard} interim，{scope['context_count']}/{scope['expected_context_count']} contexts，"
            f"{scope['test_case_count']}/{scope['expected_test_case_count']} testcases，"
            "full_cohort_complete=false")


def rank(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for index, _ in ordered[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def spearman(x_values, y_values):
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_rank, y_rank = rank(x_values), rank(y_values)
    x_mean = sum(x_rank) / len(x_rank)
    y_mean = sum(y_rank) / len(y_rank)
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_rank, y_rank))
    x_energy = sum((x - x_mean) ** 2 for x in x_rank)
    y_energy = sum((y - y_mean) ** 2 for y in y_rank)
    if x_energy == 0 or y_energy == 0:
        return None
    return covariance / math.sqrt(x_energy * y_energy)


def summarize_case_deltas(case_records):
    metrics = (*LATENT_METRICS, *ACTION_METRICS)
    return {
        "unit": "test case; seeds averaged within case",
        "metrics": {
            metric: {
                "improved_count": sum(row[f"distilled_minus_parent_{metric}"] < 0
                                       for row in case_records),
                "worsened_count": sum(row[f"distilled_minus_parent_{metric}"] > 0
                                       for row in case_records),
                "unchanged_count": sum(row[f"distilled_minus_parent_{metric}"] == 0
                                        for row in case_records),
                "median_delta": statistics.median(
                    row[f"distilled_minus_parent_{metric}"] for row in case_records),
            }
            for metric in metrics
        },
    }


def aggregate_latent_time_profiles(rows, protocol):
    grouped = {}
    expected_seeds = {
        "calibration": tuple(protocol["calibration_seed_indices"]),
        "selection": tuple(protocol["evaluation_seed_indices"]),
        "test": tuple(protocol["evaluation_seed_indices"]),
    }
    for row in rows:
        if row["arm"] not in ("parent_self", "distilled_self"):
            continue
        shape = row.get("latent_shape")
        require(isinstance(shape, list) and len(shape) == 5
                and all(isinstance(size, int) and not isinstance(size, bool) and size > 0
                        for size in shape),
                f"{row['arm']} latent_shape must be five positive integers")
        vector_length = shape[2] - 1
        require(vector_length > 0, f"{row['arm']} latent sequence must include a future frame")
        vectors = {}
        for metric in LATENT_PROFILE_METRICS:
            values = row.get(metric)
            require(isinstance(values, list) and len(values) == vector_length,
                    f"{row['arm']} {metric} length must equal latent_shape[2]-1")
            vectors[metric] = [finite_number(value, f"{row['arm']} {metric}")
                               for value in values]
        key = (row["split"], row["arm"])
        case = grouped.setdefault(key, {}).setdefault(row["case_id"], {})
        require(row["seed_index"] not in case,
                f"duplicate latent profile context: {key!r}/{row['case_id']}/{row['seed_index']}")
        case[row["seed_index"]] = (shape, vectors)

    result = {}
    for (split, arm), cases in grouped.items():
        expected = set(expected_seeds[split])
        case_vectors = {metric: [] for metric in LATENT_PROFILE_METRICS}
        shape = None
        context_count = 0
        for case_id, contexts in cases.items():
            require(set(contexts) == expected,
                    f"latent profile seeds incomplete for {split}/{arm}/{case_id}")
            context_count += len(contexts)
            for context_shape, _ in contexts.values():
                require(shape is None or context_shape == shape,
                        f"latent_shape changed within {split}/{arm}")
                shape = context_shape
            for metric in LATENT_PROFILE_METRICS:
                case_vectors[metric].append([
                    sum(contexts[seed][1][metric][index] for seed in expected) / len(expected)
                    for index in range(shape[2] - 1)
                ])
        profile = {
            "latent_shape": shape,
            "case_count": len(cases),
            "context_count": context_count,
            "future_latent_frame_indices": list(range(1, shape[2])),
            "temporal_delta_transitions": [
                {"from_latent_frame": index, "to_latent_frame": index + 1}
                for index in range(shape[2] - 1)
            ],
        }
        for metric, per_case in case_vectors.items():
            profile[metric] = [sum(values[index] for values in per_case) / len(per_case)
                               for index in range(shape[2] - 1)]
        result.setdefault(split, {})[arm] = profile
    return result


def summarize_case_teacher_future_rescue(case_records):
    return {
        "unit": "test case; seeds averaged within case",
        "positive_is_improvement": True,
        "policies": {
            policy: {
                metric: {
                    "improved_count": sum(row[f"{policy}_teacher_future_{metric}_rescue"] > 0
                                           for row in case_records),
                    "worsened_count": sum(row[f"{policy}_teacher_future_{metric}_rescue"] < 0
                                           for row in case_records),
                    "unchanged_count": sum(row[f"{policy}_teacher_future_{metric}_rescue"] == 0
                                            for row in case_records),
                    "median_rescue": statistics.median(
                        row[f"{policy}_teacher_future_{metric}_rescue"] for row in case_records),
                }
                for metric in ACTION_METRICS
            }
            for policy in ("parent", "distilled")
        },
    }


def index_groups(groups):
    return {(group["split"], group["arm"]): group for group in groups}


def compact_groups(groups):
    return [{key: group[key] for key in ("split", "arm", "case_count", "context_count", "metrics")}
            for group in groups]


def relative_mean_changes(groups):
    indexed = index_groups(groups)

    def percentage(numerator, denominator):
        return 100.0 * numerator / denominator if denominator else None

    result = {}
    for split in ("calibration", "selection", "test"):
        parent = indexed[(split, "parent_self")]["metrics"]
        distilled = indexed[(split, "distilled_self")]["metrics"]
        result[split] = {
            "native_distilled_minus_parent_pct": {
                metric: percentage(distilled[metric] - parent[metric], parent[metric])
                for metric in (*LATENT_METRICS, *ACTION_METRICS)
            },
            "teacher_future_rmse_reduction_pct": {
                policy: {
                    metric: percentage(
                        indexed[(split, f"{policy}_self")]["metrics"][metric]
                        - indexed[(split, f"{policy}_teacher_future")]["metrics"][metric],
                        indexed[(split, f"{policy}_self")]["metrics"][metric])
                    for metric in ACTION_METRICS
                }
                for policy in ("parent", "distilled")
            },
        }
    return {
        "definition": "relative difference of case means, not mean of per-case percentages or causal attribution",
        "splits": result,
    }


def per_case_index(groups):
    return {
        (group["split"], group["arm"], row["case_id"]): row["metrics"]
        for group in groups for row in group["per_case"]
    }


def paired_rows(rows):
    by_context = {}
    for row in rows:
        by_context.setdefault((row["split"], row["case_id"], row["seed_index"]), {})[row["arm"]] = row

    rescue_rows = []
    delta_rows = []
    for (split, case_id, seed_index), arms in by_context.items():
        require(set(arms) == set(EXPECTED_ARMS), f"unpaired arms at {(split, case_id, seed_index)!r}")
        domain = arms["bf16"]["domain"]
        dimension = arms["bf16"]["dimension"]
        for policy in ("parent", "distilled"):
            self_arm = arms[f"{policy}_self"]
            teacher_arm = arms[f"{policy}_teacher_future"]
            rescue = {f"{name}_rescue": finite_number(self_arm[name], name)
                      - finite_number(teacher_arm[name], name) for name in ACTION_METRICS}
            rescue_rows.append({"split": split, "case_id": case_id, "seed_index": seed_index,
                                "arm": f"{policy}_teacher_future_rescue", "domain": domain,
                                "dimension": dimension, **rescue})

        parent = arms["parent_self"]
        distilled = arms["distilled_self"]
        names = (*LATENT_METRICS, *ACTION_METRICS)
        changes = {f"delta_{name}": finite_number(distilled[name], name)
                   - finite_number(parent[name], name) for name in names}
        delta_rows.append({"split": split, "case_id": case_id, "seed_index": seed_index,
                           "arm": "distilled_minus_parent", "domain": domain,
                           "dimension": dimension, **changes})
    return rescue_rows, delta_rows


def build_case_records(test_cases, case_metadata, aggregate_groups, rescue_groups, delta_groups):
    main_cases = per_case_index(aggregate_groups)
    rescue_cases = per_case_index(rescue_groups)
    delta_cases = per_case_index(delta_groups)
    records = []
    for case_id in test_cases:
        domain, dimension = case_metadata[case_id]
        row = {"case_id": case_id, "domain": domain, "dimension": dimension}
        for arm in EXPECTED_ARMS:
            values = main_cases[("test", arm, case_id)]
            for metric in ACTION_METRICS:
                row[f"{arm}_{metric}"] = values[metric]
            if arm in ("parent_self", "distilled_self"):
                for metric in LATENT_METRICS:
                    row[f"{arm}_{metric}"] = values[metric]
        for policy in ("parent", "distilled"):
            values = rescue_cases[("test", f"{policy}_teacher_future_rescue", case_id)]
            for metric in ACTION_METRICS:
                row[f"{policy}_teacher_future_{metric}_rescue"] = values[f"{metric}_rescue"]
        changes = delta_cases[("test", "distilled_minus_parent", case_id)]
        for metric in (*LATENT_METRICS, *ACTION_METRICS):
            row[f"distilled_minus_parent_{metric}"] = changes[f"delta_{metric}"]
        records.append(row)
    return records


def make_plot(path, case_records, spearman_value, scope):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [row["case_id"] for row in case_records]
    x = list(range(len(cases)))
    domain_boundary = next((i for i, row in enumerate(case_records) if row["domain"] == "plus"), len(cases))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)

    for arm, label, color in (
        ("parent_self", "Parent self", "#2864a5"),
        ("distilled_self", "Distilled self", "#d17a00"),
    ):
        axes[0].plot(x, [row[f"{arm}_future_latent_rmse"] for row in case_records],
                     marker="o", markersize=3, linewidth=1.2, label=label, color=color)
    axes[0].set_title("Future latent RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].legend(fontsize=8)

    for arm, label, color, style in (
        ("parent_self", "Parent self", "#2864a5", "-"),
        ("parent_teacher_future", "Parent teacher future", "#2864a5", "--"),
        ("distilled_self", "Distilled self", "#d17a00", "-"),
        ("distilled_teacher_future", "Distilled teacher future", "#d17a00", "--"),
    ):
        axes[1].plot(x, [row[f"{arm}_first10motor_rmse"] for row in case_records],
                     marker=".", markersize=4, linewidth=1, linestyle=style, label=label, color=color)
    axes[1].set_title("First 10 motor action RMSE")
    axes[1].legend(fontsize=7)

    for policy, label, color in (
        ("parent", "Parent", "#2864a5"),
        ("distilled", "Distilled", "#d17a00"),
    ):
        axes[2].plot(x, [row[f"{policy}_teacher_future_first10motor_rmse_rescue"]
                         for row in case_records], marker="o", markersize=3,
                     linewidth=1.1, label=label, color=color)
    axes[2].axhline(0, color="#555555", linewidth=0.8)
    rho_text = "NA" if spearman_value is None else f"{spearman_value:.2f}"
    axes[2].set_title("Teacher-future rescue")
    axes[2].set_ylabel("Self error − teacher-future error")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(cases, rotation=60, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.22)
        if 0 < domain_boundary < len(cases):
            axis.axvline(domain_boundary - 0.5, color="#888888", linestyle=":", linewidth=0.8)
    axes[0].set_xlabel("Original cases  |  Plus cases")
    axes[1].set_xlabel("Original cases  |  Plus cases")
    axes[2].set_xlabel("Original cases  |  Plus cases")
    fig.suptitle(f"{scope}; seeds averaged within case; exploratory Δlatent/Δaction ρ={rho_text}",
                 fontsize=10)
    fig.savefig(path, dpi=86, metadata={"Software": "Fast-WAM latent diagnostic"})
    if Path(path).stat().st_size > MAX_ARTIFACT_BYTES:
        fig.savefig(path, dpi=62, metadata={"Software": "Fast-WAM latent diagnostic"})
    plt.close(fig)
    require(Path(path).stat().st_size <= MAX_ARTIFACT_BYTES, "paired diagnostic PNG exceeds 256 KiB")


def markdown_report(analysis):
    aggregate = index_groups(analysis["aggregates"]["arms"])
    rescue = index_groups(analysis["aggregates"]["teacher_future_rescue"])
    delta = index_groups(analysis["aggregates"]["distilled_minus_parent"])
    domain_groups = {name: index_groups(groups)
                     for name, groups in analysis["aggregates"]["test_by_domain"].items()}

    def values(group, keys):
        return " / ".join(f"{group['metrics'].get(key, float('nan')):.6g}" for key in keys)

    lines = [
        "# Fast-WAM latent / teacher-future paired diagnostic",
        "",
        "本报告汇总 frozen epoch 2 的固定输入比较。Calibration、selection、test 分开汇总；每个 case 先平均 seeds，再对 cases 等权平均。",
        "",
        f"分析范围：{scope_label(analysis['scope'])}。已选 shard 均通过完成门槛：{analysis['gates']['context_count']} contexts；native queries {analysis['gates']['native_context_queries']}；teacher-future queries {analysis['gates']['teacher_future_queries']}；每个已选 shard 的 self-future parity 覆盖 3 个 arms。Test 覆盖 {analysis['gates']['test_case_count']} cases（Original {analysis['gates']['test_domain_case_counts']['original']}，Plus {analysis['gates']['test_domain_case_counts']['plus']}）。",
        "",
        "## 各 split 的 latent 与 action fidelity",
        "",
        "Action RMSE 均相对同一 context 的 BF16 teacher action；较低表示更接近该 reference。Latent 指标仅对 self arms 计算。",
        "",
        "| Split | Arm | Cases / contexts | Future latent RMSE / NMSE | Transition RMSE / NMSE | First10 / full32 / late22 motor RMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("calibration", "selection", "test"):
        for arm in ("parent_self", "distilled_self"):
            group = aggregate[(split, arm)]
            m = group["metrics"]
            lines.append(
                f"| {split} | {arm} | {group['case_count']} / {group['context_count']} | "
                f"{values(group, LATENT_METRICS[:2])} | {values(group, LATENT_METRICS[2:])} | "
                f"{values(group, ACTION_METRICS)} |"
            )
    lines += ["", "## VAE latent time-position profiles", "",
              "向量按 seed 先在 case 内平均，再对 cases 等权平均；第 0 帧 clamp 排除。Future latent 的位置是 VAE latent frame 1..T−1；transition 的第一项表示观测帧 0→第 1 个 future latent。它们是 VAE latent 时间位置，不是 32 个 action steps。",
              "", "| Split | Arm | Latent shape | Future frame: RMSE / NMSE | Transition: RMSE / NMSE |",
              "|---|---|---|---|---|"]
    for split in ("calibration", "selection", "test"):
        for arm in ("parent_self", "distilled_self"):
            profile = analysis["latent_time_profiles"][split][arm]
            shape = "×".join(map(str, profile["latent_shape"]))
            future = "; ".join(
                f"{frame}:{rmse:.5g}/{nmse:.5g}"
                for frame, rmse, nmse in zip(profile["future_latent_frame_indices"],
                                               profile["future_rmse"], profile["future_nmse"]))
            transitions = "; ".join(
                f"{transition['from_latent_frame']}→{transition['to_latent_frame']}:"
                f"{rmse:.5g}/{nmse:.5g}"
                for transition, rmse, nmse in zip(profile["temporal_delta_transitions"],
                                                   profile["temporal_delta_rmse"],
                                                   profile["temporal_delta_nmse"]))
            lines.append(f"| {split} | {arm} | {shape} | {future} | {transitions} |")
    lines += ["", "## Test domain breakdown", "", "| Domain | Arm | Cases / contexts | Future latent RMSE / NMSE | First10 / full32 / late22 action RMSE |", "|---|---|---:|---:|---:|"]
    for domain in (name for name in ("original", "plus")
                   if analysis["gates"]["test_domain_case_counts"][name]):
        for arm in ("parent_self", "distilled_self"):
            group = domain_groups[domain][("test", arm)]
            lines.append(f"| {domain} | {arm} | {group['case_count']} / {group['context_count']} | "
                         f"{values(group, LATENT_METRICS[:2])} | {values(group, ACTION_METRICS)} |")

    lines += ["", "## Teacher-future paired rescue", "",
              "Rescue 定义为同一 case/seed 上 `self action RMSE − teacher-future action RMSE`，再先平均 seeds、后等权平均 cases。正值表示替换未来 latent 后与 BF16 action 的误差下降。",
              "", "| Split | Policy | Cases / contexts | First10 / full32 / late22 rescue |", "|---|---|---:|---:|"]
    for split in ("calibration", "selection", "test"):
        for policy in ("parent", "distilled"):
            group = rescue[(split, f"{policy}_teacher_future_rescue")]
            lines.append(f"| {split} | {policy} | {group['case_count']} / {group['context_count']} | "
                         f"{values(group, [f'{m}_rescue' for m in ACTION_METRICS])} |")

    test_parent_rescue = rescue[("test", "parent_teacher_future_rescue")]["metrics"]
    test_distilled_rescue = rescue[("test", "distilled_teacher_future_rescue")]["metrics"]
    parent_sign = test_parent_rescue["first10motor_rmse_rescue"]
    distilled_sign = test_distilled_rescue["first10motor_rmse_rescue"]
    if parent_sign > 0 and distilled_sign > 0:
        interpretation = "当前覆盖的 test cases 中，parent 与 distilled 的 first10 平均 rescue 均为正，方向上与局部未来 latent 对 action fidelity 有上游影响相符。"
    elif parent_sign < 0 and distilled_sign < 0:
        interpretation = "当前覆盖的 test cases 中，两种 policy 的 first10 平均 rescue 均为负；teacher-future 替换在此测量中没有带来 action-fidelity rescue，且平均误差更高。"
    elif parent_sign == 0 and distilled_sign == 0:
        interpretation = "当前覆盖的 test cases 中，两种 policy 的 first10 平均 rescue 均为零；此测量没有显示平均 action-fidelity 改善。"
    else:
        interpretation = ("当前覆盖的 test cases 中，parent 与 distilled 的 first10 rescue 符号不同："
                          f"parent={parent_sign:.6g}，distilled={distilled_sign:.6g}；两者方向不一致。")
    lines += ["", interpretation,
              "这个 teacher-future 操作只检验固定输入下的局部替换。观察到正向 rescue 时，它至多支持未来 latent 对后续 action 有上游贡献；不能换算成可加的因果百分比，也不代表 closed-loop success 或 native-kernel latency。",
              "", "## Distilled minus parent paired change", "",
              "差值定义为 `distilled_self − parent_self`；负值表示 distilled 的对应 error metric 较低。",
              "", "| Split | Cases / contexts | Future latent RMSE / NMSE Δ | Transition RMSE / NMSE Δ | First10 / full32 / late22 action Δ |", "|---|---:|---:|---:|---:|"]
    for split in ("calibration", "selection", "test"):
        group = delta[(split, "distilled_minus_parent")]
        lines.append(f"| {split} | {group['case_count']} / {group['context_count']} | "
                     f"{values(group, [f'delta_{m}' for m in LATENT_METRICS[:2]])} | "
                     f"{values(group, [f'delta_{m}' for m in LATENT_METRICS[2:]])} | "
                     f"{values(group, [f'delta_{m}' for m in ACTION_METRICS])} |")

    case_deltas = analysis["exploratory_case_deltas"]["metrics"]
    lines += ["", "## Exploratory per-case delta direction", "",
              "每个 test case 先平均 seeds，再计算 distilled−parent；负值计为 error 改善，正值计为恶化。下表描述本范围内逐 case 的方向和中位数，不报告显著性或 p 值。",
              "", "| Metric | Improved cases | Worsened cases | Unchanged | Median delta |",
              "|---|---:|---:|---:|---:|"]
    for metric, summary in case_deltas.items():
        lines.append(f"| {metric} | {summary['improved_count']} | {summary['worsened_count']} | "
                     f"{summary['unchanged_count']} | {summary['median_delta']:.6g} |")

    case_rescue = analysis["exploratory_case_teacher_future_rescue"]["policies"]
    lines += ["", "## Exploratory per-case teacher-future rescue direction", "",
              "每个 test case 先平均 seeds；rescue = self RMSE−teacher-future RMSE，正值表示 teacher-future 使 error 降低。计数和中位数描述本范围内全部 test cases，不报告显著性或 p 值。",
              "", "| Policy | Horizon | Improved cases | Worsened cases | Unchanged | Median rescue |",
              "|---|---|---:|---:|---:|---:|"]
    for policy in ("parent", "distilled"):
        for metric, summary in case_rescue[policy].items():
            lines.append(f"| {policy} | {metric} | {summary['improved_count']} | "
                         f"{summary['worsened_count']} | {summary['unchanged_count']} | "
                         f"{summary['median_rescue']:.6g} |")

    association = analysis["exploratory_association"]
    rho = association["spearman_rho"]
    rho_text = "无法计算（至少一个变量没有变化）" if rho is None else f"Spearman ρ={rho:.4f}"
    highlights = analysis["case_highlights"]
    covered_highlights = [case_id for case_id in (1016, 1017) if str(case_id) in highlights]
    missing_highlights = [case_id for case_id in (1016, 1017) if str(case_id) not in highlights]
    lines += ["", "## Test case association and highlighted cases", "",
              f"以每个已覆盖 test case 的 seeds 均值为一个单位（n={association['n_cases']}，不是 seed-context 数），比较 `Δ future latent RMSE` 与 `Δ first10 action RMSE`（均为 distilled−parent）。{rho_text}。这是 exploratory 描述，不报告显著性或 p 值。",
              "", "逐 case 指标保留在 paired_cases.csv 与图中。", ""]
    if missing_highlights:
        lines.append("本范围未覆盖的 highlight cases：" + ", ".join(map(str, missing_highlights)) + "。")
    if covered_highlights:
        lines.append("本范围实际覆盖的 highlight cases：" + ", ".join(map(str, covered_highlights)) + "。")
    for case_id in (1016, 1017):
        row = highlights.get(str(case_id))
        if row is None:
            continue
        lines.append(
            f"- Case {case_id} ({row['domain']}/{row['dimension']}): "
            f"parent latent RMSE/NMSE={row['parent_self_future_latent_rmse']:.6g}/"
            f"{row['parent_self_future_latent_nmse']:.6g}, transition RMSE/NMSE="
            f"{row['parent_self_temporal_delta_global_rmse']:.6g}/"
            f"{row['parent_self_temporal_delta_global_nmse']:.6g}; "
            f"distilled latent RMSE/NMSE={row['distilled_self_future_latent_rmse']:.6g}/"
            f"{row['distilled_self_future_latent_nmse']:.6g}, transition RMSE/NMSE="
            f"{row['distilled_self_temporal_delta_global_rmse']:.6g}/"
            f"{row['distilled_self_temporal_delta_global_nmse']:.6g}; "
            f"parent first10 self→teacher-future={row['parent_self_first10motor_rmse']:.6g}→"
            f"{row['parent_teacher_future_first10motor_rmse']:.6g} "
            f"(rescue {row['parent_teacher_future_first10motor_rmse_rescue']:.6g}); "
            f"distilled={row['distilled_self_first10motor_rmse']:.6g}→"
            f"{row['distilled_teacher_future_first10motor_rmse']:.6g} "
            f"(rescue {row['distilled_teacher_future_first10motor_rmse_rescue']:.6g})."
        )
    lines += ["", "## Evidence boundary", "",
              "这是 decoded-BF16、fixed-input fidelity 与 teacher-future 内部 intervention 的诊断。没有训练、loss 改动、控制策略改动或 episode evaluation；不能据此声称闭环成功率提升、因果归因比例或 native-kernel 延迟变化。", ""]
    return "\n".join(lines)


def write_bounded(path, content):
    path = Path(path)
    path.write_text(content, encoding="utf-8", newline="\n")
    require(path.stat().st_size <= MAX_ARTIFACT_BYTES, f"artifact exceeds 256 KiB: {path.name}")


def run(args):
    guard()
    import metrics

    protocol = read_json(args.protocol)
    require(protocol.get("protocol") == "fastwam-vq-latent-diagnostic-v1", "unexpected latent diagnostic protocol")
    require(protocol.get("arms") == list(EXPECTED_ARMS), "protocol arms differ from the frozen five-arm cohort")
    require(protocol.get("calibration_seed_indices") == [0]
            and protocol.get("evaluation_seed_indices") == [0, 1],
            "frozen seed lists must be calibration [0] and selection/test [0,1]")
    require(protocol.get("shards") == 2 and protocol.get("contexts_per_shard") == 30
            and protocol.get("total_contexts") == 60, "frozen shard coverage changed")
    require(protocol.get("selected_epoch") == 2, "selected epoch must be frozen at 2")

    require(args.shard in (None, 0, 1), "--shard must be 0 or 1")
    all_contexts = metrics.frozen_contexts(protocol)
    contexts_by_shard = {shard: [row for row in all_contexts if row["shard"] == shard]
                         for shard in (0, 1)}
    require([len(contexts_by_shard[shard]) for shard in (0, 1)] == [30, 30],
            "the frozen cohort must have exactly 30 contexts per shard")
    selected_shards = (args.shard,) if args.shard is not None else (0, 1)
    scope = build_scope(protocol, contexts_by_shard, selected_shards)
    rows, case_metadata, shard_gates = load_receipts(
        protocol, args.results_root, contexts_by_shard, selected_shards)
    scope = build_scope(protocol, contexts_by_shard, selected_shards, case_metadata)

    split_case_counts = {split: len(protocol[f"{split}_cases"])
                         for split in ("calibration", "selection", "test")}
    require(split_case_counts == {"calibration": 8, "selection": 4, "test": 22},
            "frozen calibration/selection/test case counts changed")
    domain_counts = scope["test_domain_case_counts"]
    if scope["full_cohort_complete"]:
        require(domain_counts == {"original": 10, "plus": 12},
                f"full test domain coverage must be Original 10 and Plus 12, got {domain_counts}")
        require({1016, 1017}.issubset(set(scope["test_cases"])),
                "highlight cases 1016 and 1017 must remain in the full test cohort")

    aggregate_groups = metrics.aggregate(rows)
    expected_counts = {split: (scope["split_case_counts"][split],
                               scope["split_context_counts"][split])
                       for split in ("calibration", "selection", "test")}
    require(len(aggregate_groups) == 15, "expected one aggregate group per split and arm")
    aggregate_map = index_groups(aggregate_groups)
    for split, (case_count, context_count) in expected_counts.items():
        for arm in EXPECTED_ARMS:
            group = aggregate_map.get((split, arm))
            require(group is not None and group["case_count"] == case_count
                    and group["context_count"] == context_count,
                    f"aggregate coverage mismatch for {split}/{arm}")

    rescue_rows, delta_rows = paired_rows(rows)
    rescue_groups = metrics.aggregate(rescue_rows)
    delta_groups = metrics.aggregate(delta_rows)
    test_by_domain = {
        domain: metrics.aggregate([row for row in rows if row["split"] == "test"
                                   and row["domain"].strip().lower() == domain])
        for domain in ("original", "plus")
    }
    for domain, count in domain_counts.items():
        if not count:
            continue
        groups = index_groups(test_by_domain[domain])
        for arm in EXPECTED_ARMS:
            group = groups.get(("test", arm))
            require(group is not None and group["case_count"] == count
                    and group["context_count"] == 2 * count,
                    f"domain aggregate coverage mismatch for {domain}/{arm}")

    case_records = build_case_records(scope["test_cases"], case_metadata,
                                      aggregate_groups, rescue_groups, delta_groups)
    require(len(case_records) == scope["test_case_count"],
            "all selected test cases must appear in the paired case output")
    x_values = [row["distilled_minus_parent_future_latent_rmse"] for row in case_records]
    y_values = [row["distilled_minus_parent_first10motor_rmse"] for row in case_records]
    rho = spearman(x_values, y_values)
    latent_profiles = aggregate_latent_time_profiles(rows, protocol)
    analysis = {
        "status": "complete",
        "status_scope": "selected_shards",
        "scope_complete": True,
        "job_id": os.environ["PBS_JOBID"],
        "protocol": protocol["protocol"],
        "selected_epoch": protocol["selected_epoch"],
        "selected_shards": list(selected_shards),
        "scope": scope,
        "full_cohort_complete": scope["full_cohort_complete"],
        "gates": {
            "shards": shard_gates,
            "context_count": scope["context_count"],
            "native_context_queries": sum(gate["native_context_queries"]
                                           for gate in shard_gates.values()),
            "teacher_future_queries": sum(gate["teacher_future_queries"]
                                           for gate in shard_gates.values()),
            "test_case_count": len(case_records),
            "test_domain_case_counts": domain_counts,
            "all_five_arms_per_context": True,
            "self_future_parity_arms_per_shard": 3,
        },
        "aggregates": {
            "arms": compact_groups(aggregate_groups),
            "teacher_future_rescue": compact_groups(rescue_groups),
            "distilled_minus_parent": compact_groups(delta_groups),
            "test_by_domain": {name: compact_groups(groups) for name, groups in test_by_domain.items()},
        },
        "exploratory_association": {
            "n_cases": len(case_records),
            "unit": "test case; seeds averaged within case",
            "x": "distilled_minus_parent_future_latent_rmse",
            "y": "distilled_minus_parent_first10motor_rmse",
            "spearman_rho": rho,
            "p_value": None,
        },
        "exploratory_case_deltas": summarize_case_deltas(case_records),
        "latent_time_profiles": latent_profiles,
        "relative_mean_changes": relative_mean_changes(aggregate_groups),
        "exploratory_case_teacher_future_rescue": summarize_case_teacher_future_rescue(case_records),
        "case_highlights": {str(case_id): row for case_id in (1016, 1017)
                            for row in case_records if row["case_id"] == case_id},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    write_bounded(args.out / "analysis.json", json.dumps(analysis, indent=2, ensure_ascii=False,
                                                            allow_nan=False) + "\n")
    fieldnames = list(case_records[0])
    with (args.out / "paired_cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(case_records)
    require((args.out / "paired_cases.csv").stat().st_size <= MAX_ARTIFACT_BYTES,
            "paired_cases.csv exceeds 256 KiB")
    write_bounded(args.out / "REPORT.zh.md", markdown_report(analysis))
    plot_scope = ("Full frozen cohort" if scope["full_cohort_complete"]
                  else f"Shard {scope['selected_shards'][0]} interim")
    plot_scope += f"; {scope['context_count']}/60 contexts; {len(case_records)}/22 test cases"
    make_plot(args.out / "paired_diagnostic.png", case_records, rho, plot_scope)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True,
                        help="directory containing shard-0/ and shard-1/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), default=None,
                        help="analyze one complete shard as an interim scope; default analyzes both")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
