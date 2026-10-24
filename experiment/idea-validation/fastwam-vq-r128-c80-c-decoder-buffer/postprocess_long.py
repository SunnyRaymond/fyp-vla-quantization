"""Recompute the final C decoder/buffer report from cached CPU tensors only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import tempfile


PROTOCOL_ID = "fastwam-vq-r128-c80-c-decoder-buffer-post-v1"
SOURCE_PROTOCOL_ID = "fastwam-vq-r128-c80-c-decoder-buffer-v1"
SOURCE_JOB = "25960283.pbs101"
SOURCE_JOB_NAME = "vqcdl003"
SOURCE_EXIT_STATUS = 1
SOURCE_RESULT_ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-c-decoder-buffer-20261024-r3-long/results"
SOURCE_ARTIFACT_ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-c-decoder-buffer-20261024-r3-long/artifacts/25960283.pbs101"
POST_ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-c-decoder-buffer-20261024-post-r1-long"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(data) >= 256 * 1024:
        raise ValueError(f"Refusing oversized control receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    temporary.replace(path)


def compute_guard() -> str:
    job_id = os.environ.get("PBS_JOBID", "")
    nodefile = Path(os.environ.get("PBS_NODEFILE", ""))
    if not re.fullmatch(r"\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+", job_id) or not nodefile.is_file():
        raise RuntimeError("A real PBS compute allocation is required")
    host = socket.gethostname().split(".")[0]
    if "login" in host.lower():
        raise RuntimeError("Refusing to run postprocessing on a login node")
    nodes = {line.strip().split(".")[0] for line in nodefile.read_text().splitlines() if line.strip()}
    if host not in nodes:
        raise RuntimeError("Current host is not listed in PBS_NODEFILE")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("CPU-only postprocessing requires CUDA_VISIBLE_DEVICES to be empty")
    return job_id


def validate_report(report: dict, source_protocol: dict, post_protocol: dict) -> dict:
    source = post_protocol["source"]
    modes = report.get("modes", {})
    checks = {
        "recomputed_report_complete": report.get("complete") is True,
        "producer_job_preserved": report.get("pbs_jobid") == source["job_id"],
        "candidate_modes_exact": report.get("candidate_modes") == ["c_decoder", "c_buffer"],
        "frozen_full80_declared": report.get("full_queries_per_mode") == 80,
        "inventory_passed": report.get("storage_inventory", {}).get("passed") is True,
        "training_updates_zero": report.get("new_training_updates") == 0,
        "closed_loop_zero": report.get("closed_loop_episodes") == 0,
        "source_protocol_matches": source_protocol.get("protocol") == source["protocol"],
        "source_root_matches": source_protocol.get("remote_root") == source["artifact_root"].rsplit("/artifacts/", 1)[0],
        "decoder_reuse_source_matches": source_protocol.get("completed_candidate_reuse") == post_protocol.get("completed_candidate_reuse"),
        "source_raw_summary_is_incomplete": source.get("raw_complete") is False,
        "source_failure_metadata_fixed": source.get("job_state") == "F" and source.get("exit_status") == SOURCE_EXIT_STATUS,
    }
    expected_checks = (
        "expected_runtime_kind", "logical_coverage_5258_per_query",
        "validation_full80_exact_80", "decoder_bitwise_all_modules",
        "anchor_action_max_abs_eq_0", "anchor_future_max_abs_eq_0",
        "only_expected_graph_regions", "video_step_replays_exact",
        "video_prefill_replays_exact", "action_step_replays_exact",
        "parameter_execution_probe_scope_is_limited",
    )
    for mode in ("c_decoder", "c_buffer"):
        row = modes.get(mode, {})
        parity = row.get("parity_vs_primary", {})
        validation = row.get("validation_checks", {})
        validation_flags = validation.get("checks", {})
        numerics = row.get("numerical_flags", {})
        stage = row.get("stage_gate", {})
        prefix = mode + "_"
        checks[prefix + "full80_exact_parity"] = (
            parity.get("passed") is True
            and parity.get("exact") is True
            and parity.get("queries") == 80
            and parity.get("max_abs") == 0.0
            and parity.get("rmse") == 0.0
        )
        checks[prefix + "validation_passed"] = validation.get("passed") is True
        checks[prefix + "candidate_stage_passed"] = stage.get("passed") is True
        checks[prefix + "614_bitwise_and_anchor_gates"] = all(
            validation_flags.get(key) is True for key in expected_checks
        )
        checks[prefix + "numeric_flags"] = (
            numerics.get("deterministic_algorithms") is True
            and numerics.get("fill_uninitialized_memory") is True
            and numerics.get("allow_tf32") is False
            and numerics.get("allow_bf16_reduced_precision_reduction") is True
            and numerics.get("compiled") is False
            and numerics.get("cuda_graph") is True
            and numerics.get("new_training_updates") == 0
            and numerics.get("closed_loop_episodes") == 0
        )
        checks[prefix + "measured_80_queries"] = row.get("queries") == 80

    decoder = modes.get("c_decoder", {})
    decoder_source = post_protocol.get("completed_candidate_reuse", {})
    provenance = decoder.get("provenance", {})
    checks["c_decoder_completed_reuse_identity"] = (
        decoder.get("status") == "reused_candidate"
        and provenance.get("reused") is True
        and provenance.get("pbs_jobid") == decoder_source.get("source_job")
        and provenance.get("protocol") == decoder_source.get("source_protocol")
        and provenance.get("source_job") == decoder_source.get("source_job")
    )
    buffer = modes.get("c_buffer", {})
    checks["c_buffer_is_producer_measurement"] = (
        buffer.get("status") == "measured"
        and buffer.get("provenance", {}).get("reused") is False
        and buffer.get("provenance", {}).get("pbs_jobid") == source["job_id"]
    )
    checks["two_mode_report_present"] = all(mode in modes for mode in ("c_decoder", "c_buffer"))
    return {"passed": all(checks.values()), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--post-protocol", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--patched-finalize", type=Path, required=True)
    parser.add_argument("--post-results", type=Path, required=True)
    args = parser.parse_args()

    post_job = compute_guard()
    post_protocol = read_json(args.post_protocol)
    source_protocol = read_json(args.source_protocol)
    if (post_protocol.get("protocol") != PROTOCOL_ID
            or post_protocol.get("remote_root") != POST_ROOT
            or post_protocol.get("source", {}).get("job_id") != SOURCE_JOB
            or post_protocol.get("source", {}).get("result_root") != SOURCE_RESULT_ROOT
            or post_protocol.get("source", {}).get("artifact_root") != SOURCE_ARTIFACT_ROOT):
        raise RuntimeError("Postprocess protocol does not match the frozen source")
    if (source_protocol.get("protocol") != SOURCE_PROTOCOL_ID
            or source_protocol.get("remote_root") != SOURCE_ARTIFACT_ROOT.rsplit("/artifacts/", 1)[0]
            or str(args.source_results) != SOURCE_RESULT_ROOT
            or source_protocol.get("completed_candidate_reuse") != post_protocol.get("completed_candidate_reuse")):
        raise RuntimeError("Source protocol or result root differs from the frozen producer")
    raw_summary = read_json(args.source_results / "summary.json")
    if (Path(SOURCE_ARTIFACT_ROOT) / 'exit_code.txt').read_text().strip() != '1':
        raise RuntimeError('Source artifact exit differs from the declared failed wrapper')
    if (raw_summary.get("protocol") != SOURCE_PROTOCOL_ID
            or raw_summary.get("pbs_jobid") != SOURCE_JOB
            or raw_summary.get("complete") is not False):
        raise RuntimeError("The expected incomplete raw producer receipt was not found")

    spec = importlib.util.spec_from_file_location("patched_finalize", args.patched_finalize)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the patched final report module")
    finalize = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finalize)
    import torch

    # final_report only reads cached arrays via map_location='cpu'; it never loads a model.
    report = finalize.final_report(source_protocol, args.source_results, SOURCE_JOB, torch)
    validation = validate_report(report, source_protocol, post_protocol)
    source = post_protocol["source"]
    report.update({
        "postprocess_pbs_jobid": post_job,
        "source_raw_complete": False,
        "source_main_state": source["job_state"],
        "source_main_exit_status": source["exit_status"],
        "source_main_F1": True,
        "postprocess_kind": "cpu_recompute",
        "new_model_forwards": 0,
        "postprocess_validation": validation,
    })
    args.post_results.mkdir(parents=True, exist_ok=True)
    atomic_json(args.post_results / "summary.json", report)
    atomic_json(args.post_results / "stage_gates.json", {
        "protocol": report["protocol"],
        "pbs_jobid": report["pbs_jobid"],
        "modes": report.get("stage_gates", {}),
    })
    receipt = {
        "protocol": PROTOCOL_ID,
        "source_protocol": SOURCE_PROTOCOL_ID,
        "source_main_pbs_jobid": SOURCE_JOB,
        "source_main_state": source["job_state"],
        "source_main_exit_status": source["exit_status"],
        "source_raw_complete": False,
        "postprocess_pbs_jobid": post_job,
        "postprocess_kind": "cpu_recompute",
        "new_model_forwards": 0,
        "report_complete": report.get("complete") is True,
        "validation": validation,
    }
    atomic_json(args.post_results / "postprocess_receipt.json", receipt)
    print("CPU_POSTPROCESS " + json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
