"""Post-hoc leave-case-out summaries for the completed action-latent report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import socket

HERE = Path(__file__).resolve().parent
ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-action-latent-20261014"
JOB_NAME = "fwvzl1014ts"
SOURCE_ANALYSIS_JOB = "25799647.pbs101"
SOURCE_EXPERIMENT_JOB = "25798439.pbs101"
PROTOCOL = "fastwam-vq-action-latent-v1"
ANALYSIS_PATH = f"{ROOT}/results/analysis-{SOURCE_ANALYSIS_JOB}/analysis.json"
LIMIT = 256 * 1024
CASES = list(range(1004, 1026))
EXCLUSIONS = {
    "all_22": set(),
    "omit_1015": {1015},
    "omit_1016": {1016},
    "omit_1025": {1025},
    "omit_1016_1025": {1016, 1025},
}
METRICS = {
    "first10motor_rmse": "First-10 motor RMSE",
    "full32motor_rmse": "Full-32 motor RMSE",
    "late22motor_rmse": "Late-22 motor RMSE",
    "future_latent_nmse": "Future latent NMSE",
    "temporal_delta_global_nmse": "Temporal-delta global NMSE",
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def worker_guard():
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    require(os.environ.get("PBS_JOBID", "").strip() and nodefile and "login" not in host.lower(),
            "Post-hoc analysis requires a genuine PBS compute allocation")
    nodes = {line.split(".")[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    require(host in nodes, "Current host is outside the PBS allocation")
    require(not os.environ.get("CUDA_VISIBLE_DEVICES", ""), "This post-hoc job must remain CPU-only")


def finite(row, field, case):
    value = row.get(field)
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
            f"case {case} has invalid {field}")
    return float(value)


def make_delta_summary(rows, baseline_field, target_field, delta_field, excluded):
    selected = [row for row in rows if row["case_id"] not in excluded]
    require(selected, "Sensitivity subset is empty")
    baseline = [finite(row, baseline_field, row["case_id"]) for row in selected]
    target = [finite(row, target_field, row["case_id"]) for row in selected]
    deltas = [finite(row, delta_field, row["case_id"]) for row in selected]
    for row, base, end, delta in zip(selected, baseline, target, deltas):
        require(abs((end - base) - delta) <= 1e-6,
                f"Stored paired delta differs from case rows at case {row['case_id']} ({delta_field})")
    baseline_mean = sum(baseline) / len(baseline)
    target_mean = sum(target) / len(target)
    delta_mean = sum(deltas) / len(deltas)
    require(abs((target_mean - baseline_mean) - delta_mean) <= 1e-6,
            f"Paired means do not reconcile for {delta_field}")
    require(baseline_mean != 0.0, f"Cannot compute relative mean change from zero baseline: {baseline_field}")
    return {
        "case_count": len(selected),
        "excluded_case_ids": sorted(excluded),
        "baseline_mean": baseline_mean,
        "action_latent_chosen_mean": target_mean,
        "mean_paired_delta": delta_mean,
        "relative_mean_change_pct": 100.0 * delta_mean / baseline_mean,
    }


def paired_overlap(rows, action_delta_field, future_delta_field):
    result = {"double_improvement": 0, "action_only_improvement": 0,
              "future_only_improvement": 0, "double_deterioration": 0,
              "boundary_or_unchanged": 0}
    for row in rows:
        action = finite(row, action_delta_field, row["case_id"])
        future = finite(row, future_delta_field, row["case_id"])
        if action < 0 and future < 0:
            result["double_improvement"] += 1
        elif action < 0 and future >= 0:
            result["action_only_improvement"] += 1
        elif action >= 0 and future < 0:
            result["future_only_improvement"] += 1
        elif action > 0 and future > 0:
            result["double_deterioration"] += 1
        else:
            result["boundary_or_unchanged"] += 1
    require(sum(result.values()) == len(rows), "Improvement-overlap counts do not cover the cohort")
    return result


def comparison_spec(comparison, metric):
    if metric in ("future_latent_nmse", "temporal_delta_global_nmse"):
        baseline = "parent_self" if comparison == "parent" else "old_action_only"
    else:
        baseline = "source_row_output_vq_a4" if comparison == "parent" else "old_action_only"
    delta_prefix = "new_minus_parent" if comparison == "parent" else "new_minus_old_action_only"
    return f"{baseline}_{metric}", f"action_latent_chosen_{metric}", f"{delta_prefix}_{metric}"


def markdown_report(result):
    lines = [
        "# Tail-case sensitivity（post hoc）", "",
        f"- Source analysis job: `{SOURCE_ANALYSIS_JOB}`；experiment job: `{SOURCE_EXPERIMENT_JOB}`。",
        "- Frozen primary cohort remains all 22 test cases / 44 contexts. Every omission below is a post hoc sensitivity view; it does not replace the 22-case result.",
        "- Relative mean change = mean paired delta / mean baseline × 100%；negative values indicate lower error. No p-values or new model forwards.",
        "- All action and future values come from the stored `case_rows` and their existing paired-delta fields in the one source `analysis.json`.", "",
        "## Full cohort and omission sensitivities", "",
        "| Comparison | Metric | View | Cases | Baseline mean | Chosen mean | Mean paired delta | Relative change (%) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for comparison, values in result["comparisons"].items():
        for metric, views in values.items():
            for view, summary in views.items():
                lines.append(f"| {comparison} | {METRICS[metric]} | {view} | {summary['case_count']} | "
                             f"{summary['baseline_mean']:.8g} | {summary['action_latent_chosen_mean']:.8g} | "
                             f"{summary['mean_paired_delta']:.8g} | {summary['relative_mean_change_pct']:.4f} |")
    lines += ["", "## Focused checks", "",
              "| Question | Full 22 mean delta | After omission mean delta | Full relative change (%) | After omission relative change (%) | Direction after omission |",
              "|---|---:|---:|---:|---:|---|"]
    for name, focused in result["focused_sensitivity"].items():
        full, omitted = focused["all_22"], focused[focused["omission_view"]]
        direction = "improvement" if omitted["mean_paired_delta"] < 0 else (
            "deterioration" if omitted["mean_paired_delta"] > 0 else "unchanged")
        lines.append(f"| {focused['question']} | {full['mean_paired_delta']:.8g} | "
                     f"{omitted['mean_paired_delta']:.8g} | {full['relative_mean_change_pct']:.4f} | "
                     f"{omitted['relative_mean_change_pct']:.4f} | {direction} |")
    lines += ["", "## Action / future overlap counts", "",
              "Negative paired delta means improvement. Categories use strict signs; zero-boundary cases are reported separately.", "",
              "| Comparison | Both improve | Action only improves | Future only improves | Both deteriorate | Boundary / unchanged | Total |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for comparison, counts in result["action_future_overlap_counts"].items():
        lines.append(f"| {comparison} | {counts['double_improvement']} | {counts['action_only_improvement']} | "
                     f"{counts['future_only_improvement']} | {counts['double_deterioration']} | "
                     f"{counts['boundary_or_unchanged']} | 22 |")
    lines += ["", "## Figure", "",
              "`paired_recovery_case_ids.png` redraws the existing paired-recovery values from the same case rows. The upper x-axis labels are actual case IDs, not row indices; the original analysis and image remain untouched.", ""]
    text = "\n".join(lines)
    require(len(text.encode("utf-8")) <= LIMIT, "Tail sensitivity Markdown exceeds 256 KiB")
    return text


def make_case_plot(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [row["case_id"] for row in rows]
    xs = list(range(len(cases)))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.2), constrained_layout=True)
    panels = (
        (axes[0], "First-10 motor RMSE paired delta", "new_minus_parent_first10motor_rmse",
         "new_minus_old_action_only_first10motor_rmse"),
        (axes[1], "Future latent NMSE paired delta", "new_minus_parent_future_latent_nmse",
         "new_minus_old_action_only_future_latent_nmse"),
    )
    for ax, ylabel, parent_key, old_key in panels:
        ax.bar([x - width / 2 for x in xs], [finite(row, parent_key, row["case_id"]) for row in rows],
               width, color="#276fbf", label="chosen − parent")
        ax.bar([x + width / 2 for x in xs], [finite(row, old_key, row["case_id"]) for row in rows],
               width, color="#d95f59", label="chosen − old action-only")
        ax.axhline(0, color="#777777", linewidth=0.75)
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs, [str(case) for case in cases], fontsize=7, rotation=90)
        ax.tick_params(axis="x", which="both", top=True, labeltop=(ax is axes[0]),
                       bottom=True, labelbottom=(ax is axes[1]), labelsize=7)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, ncol=2, fontsize=8)
    axes[0].set_title("Paired fixed-input case deltas; negative values indicate lower error")
    axes[1].set_xlabel("Test case ID")
    fig.savefig(path, dpi=105)
    plt.close(fig)
    require(Path(path).stat().st_size <= LIMIT, "Corrected paired-recovery PNG exceeds 256 KiB")


def run_worker():
    worker_guard()
    input_path = Path(ANALYSIS_PATH)
    require(input_path.is_file() and input_path.stat().st_size <= LIMIT,
            "The frozen source analysis.json is missing or exceeds 256 KiB")
    report = json.loads(input_path.read_text(encoding="utf-8"))
    require(report.get("status") == "complete" and report.get("analysis_job") == SOURCE_ANALYSIS_JOB
            and report.get("experiment_job") == SOURCE_EXPERIMENT_JOB and report.get("protocol") == PROTOCOL,
            "Source analysis identity or completion status mismatch")
    require(report.get("new_model_forwards") == 0, "Source analysis reports nonzero model forwards")
    cohort = report.get("cohort", {})
    require(cohort.get("test_cases") == 22 and cohort.get("test_contexts") == 44,
            "Source analysis main cohort is not the frozen 22 cases / 44 contexts")
    rows = report.get("case_rows")
    require(isinstance(rows, list) and len(rows) == 22, "Source analysis case_rows must contain 22 cases")
    require([row.get("case_id") for row in rows] == CASES, "Source analysis case_rows differ from the frozen case IDs")

    comparisons = {}
    for comparison in ("parent", "old_action_only"):
        comparisons[comparison] = {}
        for metric in METRICS:
            baseline_field, target_field, delta_field = comparison_spec(comparison, metric)
            comparisons[comparison][metric] = {
                view: make_delta_summary(rows, baseline_field, target_field, delta_field, excluded)
                for view, excluded in EXCLUSIONS.items()
            }

    focused = {
        "full32_vs_old_action_only": {
            "question": "Full-32 motor vs old action-only: omit case 1016",
            "omission_view": "omit_1016",
            **comparisons["old_action_only"]["full32motor_rmse"],
        },
        "late22_vs_old_action_only": {
            "question": "Late-22 motor vs old action-only: omit case 1016",
            "omission_view": "omit_1016",
            **comparisons["old_action_only"]["late22motor_rmse"],
        },
        "late22_vs_parent": {
            "question": "Late-22 motor vs parent: omit case 1015",
            "omission_view": "omit_1015",
            **comparisons["parent"]["late22motor_rmse"],
        },
    }
    overlap = {
        comparison: paired_overlap(
            rows,
            "new_minus_parent_first10motor_rmse" if comparison == "parent"
            else "new_minus_old_action_only_first10motor_rmse",
            "new_minus_parent_future_latent_nmse" if comparison == "parent"
            else "new_minus_old_action_only_future_latent_nmse",
        ) for comparison in ("parent", "old_action_only")
    }
    result = {
        "status": "complete", "protocol": PROTOCOL, "source_analysis_job": SOURCE_ANALYSIS_JOB,
        "source_experiment_job": SOURCE_EXPERIMENT_JOB, "analysis_input": ANALYSIS_PATH,
        "test_case_count": 22, "test_context_count": 44, "case_ids": CASES,
        "sensitivity_only": True, "primary_22_case_result_replaced": False,
        "new_model_forwards": 0, "comparisons": comparisons,
        "focused_sensitivity": focused, "action_future_overlap_counts": overlap,
    }
    output = Path(ROOT) / "results" / f"tail-{os.environ['PBS_JOBID']}"
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "tail_sensitivity.json"
    md_path = output / "tail_sensitivity.zh.md"
    png_path = output / "paired_recovery_case_ids.png"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    md_path.write_text(markdown_report(result), encoding="utf-8")
    make_case_plot(png_path, rows)
    for path in (json_path, md_path, png_path):
        require(path.stat().st_size <= LIMIT, f"Output exceeds 256 KiB: {path.name}")
    summary = {
        "status": "complete", "job_id": os.environ["PBS_JOBID"], "protocol": PROTOCOL,
        "source_analysis_job": SOURCE_ANALYSIS_JOB, "source_experiment_job": SOURCE_EXPERIMENT_JOB,
        "test_case_count": 22, "test_context_count": 44, "new_model_forwards": 0,
        "output_directory": str(output),
        "output_files": {path.name: path.stat().st_size for path in (json_path, md_path, png_path)},
        "input_analysis_bytes": input_path.stat().st_size,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                            encoding="utf-8")
    require(summary_path.stat().st_size <= LIMIT, "Post-hoc summary exceeds 256 KiB")
    print("TAIL_SENSITIVITY_READY", json.dumps(summary))


def load_control():
    spec = importlib.util.spec_from_file_location("tail_sensitivity_existing_control", HERE / "control.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def control_files():
    files = (HERE / "tail_sensitivity.py", HERE / "tail_sensitivity.pbs")
    for path in files:
        require(path.is_file() and path.stat().st_size <= LIMIT,
                f"Missing or oversized post-hoc control file: {path.name}")
    return files


def submit():
    control = load_control()
    remote = control.remote
    intent_path = HERE / "tail_sensitivity_intent.json"
    handle_path = HERE / "tail_sensitivity_handle.json"
    if handle_path.is_file():
        print(handle_path.read_text(encoding="utf-8"))
        return
    intent = json.loads(intent_path.read_text(encoding="utf-8")) if intent_path.is_file() else None
    jump = transport = None
    try:
        jump, transport = remote.direct_connect()
        rc, output = remote.command(transport,
                                    f"qselect -x -u yguo017 -N {shlex.quote(JOB_NAME)} 2>&1", timeout=35)
        if rc not in (0, 1) or (rc == 1 and output.strip()):
            raise RuntimeError("Tail-sensitivity submission history is unavailable")
        matches = output.split()
        if matches:
            if len(matches) != 1 or not control.trusted.HANDLE.fullmatch(matches[0]):
                raise RuntimeError("Ambiguous tail-sensitivity submission history")
            receipt = {**(intent or {}), "job_id": matches[0], "job_name": JOB_NAME, "status": "recovered"}
            control.save(handle_path, receipt)
            control.save(intent_path, receipt)
            print(json.dumps(receipt))
            return
        if intent:
            raise RuntimeError("Unconfirmed intent exists and no job was found; refusing duplicate qsub")
        rc, output = remote.command(transport, "mkdir -p " + shlex.quote(ROOT), timeout=25)
        if rc:
            raise RuntimeError(f"Unable to access remote experiment directory: {output}")
        with remote.sftp_client(transport) as sftp:
            for path in control_files():
                data = path.read_bytes().replace(b"\r\n", b"\n")
                with sftp.file(f"{ROOT}/{path.name}", "wb") as stream:
                    stream.write(data)
        receipt = {"status": "intent_before_qsub", "job_name": JOB_NAME, "remote_root": ROOT,
                   "created_utc": datetime.now(timezone.utc).isoformat(), "queue": "normal",
                   "project": "personal-yguo017", "select": "select=1:ncpus=1:mem=4gb",
                   "walltime": "00:05:00", "source_analysis_job": SOURCE_ANALYSIS_JOB}
        control.save(intent_path, receipt)
        rc, output = remote.command(transport, "qsub " + shlex.quote(ROOT + "/tail_sensitivity.pbs"),
                                    timeout=40)
        if rc or not control.trusted.HANDLE.fullmatch(output.strip()):
            receipt.update(status="unknown_or_rejected", message=output[:1000])
            control.save(intent_path, receipt)
            raise RuntimeError("qsub outcome is unknown or rejected; recover with qselect, do not resubmit")
        receipt.update(job_id=output.strip(), status="confirmed")
        control.save(intent_path, receipt)
        control.save(handle_path, receipt)
        print(json.dumps(receipt))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


def terminal():
    control = load_control()
    remote = control.remote
    handle_path = HERE / "tail_sensitivity_handle.json"
    require(handle_path.is_file() and handle_path.stat().st_size <= LIMIT, "Post-hoc handle is missing or oversized")
    handle = json.loads(handle_path.read_text(encoding="utf-8"))
    job = handle.get("job_id")
    require(isinstance(job, str) and control.trusted.HANDLE.fullmatch(job), "Post-hoc PBS handle is invalid")
    jump = transport = None
    try:
        jump, transport = remote.direct_connect()
        observed = control.scheduler(transport, job)
        print("SCHEDULER", json.dumps(observed))
        if observed["query_rc"] or observed["job_state"] not in ("F", "X"):
            return
        artifact = f"{ROOT}/artifacts/tail-{job}"
        out = f"{ROOT}/results/tail-{job}"
        tail_rc, log_tail = remote.command(transport,
                                           f"test ! -f {shlex.quote(artifact + '/job.log')} || "
                                           f"tail -c 6000 {shlex.quote(artifact + '/job.log')}", timeout=25)
        if tail_rc:
            log_tail = ""
        if observed["Exit_status"] != "0":
            control.save(HERE / "tail_sensitivity_terminal.json", {
                "status": "failed", "job_id": job, "scheduler": observed, "job_log_tail": log_tail[:6000]})
            raise RuntimeError(f"Tail-sensitivity PBS job failed: {observed}; log tail: {log_tail}")
        checks = (f"test -f {shlex.quote(artifact + '/TAIL_SENSITIVITY_COMPLETE')} && "
                  f"test \"$(cat {shlex.quote(artifact + '/exit_code.txt')})\" = 0 && "
                  f"test -s {shlex.quote(artifact + '/summary.json')} && "
                  f"test \"$(cat {shlex.quote(artifact + '/summary.json')} | wc -c)\" -le {LIMIT}")
        rc, _ = remote.command(transport, checks, timeout=25)
        if rc:
            raise RuntimeError("Artifact completion marker, exit receipt, or bounded summary is missing")
        summary = control.read_remote_json(transport, artifact + "/summary.json")
        require(summary.get("status") == "complete" and summary.get("job_id") == job
                and summary.get("protocol") == PROTOCOL
                and summary.get("source_analysis_job") == SOURCE_ANALYSIS_JOB
                and summary.get("source_experiment_job") == SOURCE_EXPERIMENT_JOB
                and summary.get("test_case_count") == 22 and summary.get("test_context_count") == 44
                and summary.get("new_model_forwards") == 0,
                "Artifact summary identity or frozen 22-case coverage mismatch")
        names = ("tail_sensitivity.json", "tail_sensitivity.zh.md",
                 "paired_recovery_case_ids.png", "summary.json")
        downloaded = {}
        local_dir = HERE / "retrieved" / "results" / f"tail-{job}"
        local_dir.mkdir(parents=True, exist_ok=True)
        with remote.sftp_client(transport) as sftp:
            for name in names:
                source = f"{out}/{name}"
                size = sftp.stat(source).st_size
                require(size <= LIMIT, f"Refusing oversized post-hoc result file: {name}")
                with sftp.file(source, "rb") as stream:
                    data = stream.read(LIMIT + 1)
                require(len(data) == size and len(data) <= LIMIT, f"Post-hoc result grew while reading: {name}")
                (local_dir / name).write_bytes(data)
                downloaded[name] = {"path": str(local_dir / name), "bytes": len(data)}
        result_summary = json.loads((local_dir / "summary.json").read_text(encoding="utf-8"))
        require(result_summary.get("job_id") == job and result_summary.get("source_analysis_job") == SOURCE_ANALYSIS_JOB
                and result_summary.get("new_model_forwards") == 0
                and result_summary.get("test_case_count") == 22 and result_summary.get("test_context_count") == 44,
                "Pulled output summary identity or coverage mismatch")
        evidence = {"status": "complete", "job_id": job, "scheduler": observed,
                    "marker": True, "artifact_exit_code": 0, "source_analysis_job": SOURCE_ANALYSIS_JOB,
                    "source_experiment_job": SOURCE_EXPERIMENT_JOB, "summary": summary,
                    "downloaded": downloaded, "job_log_tail": log_tail[:6000]}
        control.save(HERE / "tail_sensitivity_terminal.json", evidence)
        print(json.dumps(evidence))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", choices=("submit", "status", "verify"))
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        run_worker()
    elif args.action == "submit":
        submit()
    elif args.action in ("status", "verify"):
        terminal()
    else:
        parser.error("choose submit/status/verify or --worker")


if __name__ == "__main__":
    main()
