"""Small, durable PBS control for the frozen C decoder/buffer profile."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex
import tempfile


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "runtime_abc_control_base",
    HERE.parent / "fastwam-vq-r128-c80-fused-tuning" / "control.py",
)
control = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(control)
_save_state_base = control._save_state

REFERENCE_MODES = ("bf16", "eager", "input_fused", "layer_decode_fp32")
REUSED_MODES = ("runtime_c",)
CANDIDATE_MODES = ("c_decoder", "c_buffer")
SOURCES = (
    "protocol.json", "run.pbs", "environment.sh", "reference_profile.py",
    "export_runtime.py", "fused_input.py", "fused_vq.py", "kernel_runtime.py",
    "layer_runtime.py", "condition_cache.py", "abc_runtime.py", "graph_runtime.py",
    "decoder_runtime.py", "inventory.py", "profile_worker.py", "finalize.py",
)


def configure(lane: str) -> dict:
    path = HERE / lane / "protocol.json"
    if not path.is_file() or path.stat().st_size >= control.LIMIT:
        raise RuntimeError("Missing or oversized lane protocol: " + str(path))
    p = json.loads(path.read_text(encoding="utf-8"))
    allocation = p["allocation"]
    control.HERE = HERE / lane
    control.PROTOCOL_ID = p["protocol"]
    control.ROOT = p["remote_root"]
    control.JOB_NAME = p["job_name"]
    control.JOB_NAME_RE = re.compile("^" + re.escape(control.JOB_NAME) + "$")
    control.QUEUE = allocation["queue"]
    control.EXECUTION_QUEUE = allocation["execution_queue"]
    control.SELECT = "1:ncpus=16:mem=110gb:ngpus=1"
    control.WALLTIME = allocation["walltime"]
    control.ALLOCATION = {
        k: allocation[k]
        for k in ("queue", "execution_queue", "gpus", "ncpus", "memory_gb", "walltime")
    }
    control.FILES = SOURCES
    control.ARTIFACTS = control.ROOT + "/artifacts"
    control.RESULTS = control.ROOT + "/results"
    control.STATE_PATH = control.ROOT + "/control/state.json"
    return p


def local_bytes(name: str) -> bytes:
    path = control.HERE / name if name == "protocol.json" else HERE / name
    if not path.is_file() or path.stat().st_size >= control.LIMIT:
        raise RuntimeError("Missing or oversized source: " + str(path))
    data = path.read_bytes().replace(b"\r\n", b"\n")
    data.decode("utf-8")
    return data


def save_state(transport, value: dict) -> None:
    handle = value.get("handle")
    if handle:
        handle.update(ncpus=16, memory_gb=110)
    _save_state_base(transport, value)


def complete_gate(value: dict) -> bool:
    try:
        handle, scheduler = value["handle"], value["scheduler"]
        summary, artifact = value["summary"], value["artifact"]
        p = json.loads((control.HERE / "protocol.json").read_text(encoding="utf-8"))
        modes = summary["modes"]
        reuse = p["candidate_reuse"]
        reused_modes = tuple(reuse["modes"])

        def reused_ok(name: str) -> bool:
            row = modes[name]
            provenance = row.get("provenance", {})
            return (
                row.get("status") == "reused_candidate"
                and row.get("queries") == 80
                and provenance.get("reused") is True
                and provenance.get("protocol") == reuse["source_protocol"]
                and provenance.get("pbs_jobid") == reuse["source_job"]
                and provenance.get("source_job") == reuse["source_job"]
                and row.get("parity_vs_primary", {}).get("passed") is True
                and row.get("parity_vs_primary", {}).get("max_abs") == 0.0
                and row.get("validation_checks", {}).get("passed") is True
                and row.get("source_stage_gate", {}).get("passed") is True
            )

        def candidate_ok(name: str) -> bool:
            row = modes[name]
            provenance = row.get("provenance", {})
            checks = row.get("validation_checks", {}).get("checks", {})
            parity = row.get("parity_vs_primary", {})
            return (
                row.get("status") == "measured"
                and row.get("queries") == 80
                and provenance.get("reused") is False
                and provenance.get("protocol") == control.PROTOCOL_ID
                and provenance.get("pbs_jobid") == handle.get("job_id")
                and parity.get("passed") is True
                and parity.get("max_abs") == 0.0
                and row.get("validation_checks", {}).get("passed") is True
                and checks.get("expected_runtime_kind") is True
                and checks.get("logical_coverage_5258_per_query") is True
                and checks.get("validation_full80_exact_80") is True
                and checks.get("decoder_receipt_bound_to_job") is True
                and checks.get("decoder_reuse_source_bound") is True
                and checks.get("decoder_frequency_weights_full80") is True
                and checks.get("decoder_bitwise_all_modules") is True
                and checks.get("anchor_action_max_abs_eq_0") is True
                and checks.get("anchor_future_max_abs_eq_0") is True
                and checks.get("parameter_execution_probe_scope_is_limited") is True
                and checks.get("activation_workspace_layout_exact") is True
                and (name != "c_buffer" or (
                    checks.get("decoder_tuning_matches_c_decoder") is True
                    and checks.get("decoder_frequency_panel_matches_c_decoder") is True
                    and checks.get("decoder_checkpoint_matches_c_decoder") is True
                ))
                and row.get("stage_gate", {}).get("passed") is True
                and row.get("stage_gate", {}).get("parity_vs_primary", {}).get("max_abs") == 0.0
            )

        return (
            handle.get("remote_root") == control.ROOT
            and handle.get("protocol") == control.PROTOCOL_ID
            and handle.get("job_name") == control.JOB_NAME
            and scheduler.get("query_rc") == 0
            and scheduler.get("job_state") == "F"
            and scheduler.get("Exit_status") == "0"
            and scheduler.get("Job_Name") == control.JOB_NAME
            and scheduler.get("queue") == control.EXECUTION_QUEUE
            and scheduler.get("Resource_List.walltime") == control.WALLTIME
            and artifact.get("exit_code") == "0"
            and artifact.get("pipeline_complete") is True
            and summary.get("protocol") == control.PROTOCOL_ID
            and summary.get("pbs_jobid") == handle.get("job_id")
            and summary.get("complete") is True
            and summary.get("new_training_updates") == 0
            and summary.get("closed_loop_episodes") == 0
            and summary.get("full_queries_per_mode") == 80
            and summary.get("candidate_modes") == list(CANDIDATE_MODES)
            and summary.get("stage_dependencies") == p["stages"]["dependencies"]
            and summary.get("performance_regressions_fatal") is False
            and set(modes) == set(REFERENCE_MODES + REUSED_MODES + CANDIDATE_MODES)
            and set(reused_modes) == set(REUSED_MODES)
            and all(reused_ok(name) for name in REUSED_MODES)
            and all(candidate_ok(name) for name in CANDIDATE_MODES)
            and summary.get("candidate_reuse", {}).get("source_job") == reuse["source_job"]
            and summary.get("candidate_reuse", {}).get("source_job_name") == reuse["source_job_name"]
            and summary.get("candidate_reuse", {}).get("modes") == list(reused_modes)
            and summary.get("candidate_reuse", {}).get("source_protocol") == reuse["source_protocol"]
            and summary.get("candidate_reuse", {}).get("source_result_root") == reuse["result_root"]
            and summary.get("candidate_reuse", {}).get("whole_pipeline_completion_required") is False
            and summary.get("storage_inventory", {}).get("passed") is True
            and all(
                modes[name].get("status") == "cached"
                and modes[name].get("queries") == 80
                for name in REFERENCE_MODES
            )
            and all(
                summary.get("stage_gates", {}).get(name, {}).get("passed") is True
                for name in CANDIDATE_MODES
            )
            and p.get("protocol") == control.PROTOCOL_ID
        )
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError, OSError):
        return False


def pull(transport) -> dict:
    value = control.status(transport)
    handle = value.get("handle")
    if not handle:
        raise RuntimeError("No submitted job")
    target = HERE / "pulled" / handle["job_id"]
    names = [
        "summary.json", "stage_gates.json", "progress.json", "storage.json", "storage_inventory.json",
        "storage_inventory_modules.json", "inventory.md", "parameter_execution.json",
        "orchestration.json", "RESULTS.zh.md",
        "runtime_c/source_stage_gates.json",
    ]
    for mode in REFERENCE_MODES + REUSED_MODES + CANDIDATE_MODES:
        names.extend(
            mode + "/" + name
            for name in ("summary.json", "samples.json", "validation.json", "failure.json",
                         "parity.json", "operators.json", "profiler_failure.json", "tuning.json",
                         "parameter_execution.json")
        )
    files, skipped = [], {}
    with control.remote.sftp_client(transport) as sftp:
        try:
            names.extend(
                name for name in sftp.listdir(control.RESULTS)
                if name.startswith("storage_inventory_modules_") and name.endswith(".json")
                and name != "storage_inventory_modules.json"
            )
        except OSError:
            skipped["storage_inventory_modules_parts"] = "directory listing failed"
        for mode in REFERENCE_MODES + REUSED_MODES + CANDIDATE_MODES:
            try:
                names.extend(
                    mode + "/" + name for name in sftp.listdir(control.RESULTS + "/" + mode)
                    if name.startswith("parameter_execution_") and name.endswith(".json")
                )
            except OSError:
                pass
        for name in names:
            data, reason = control._pull_bytes(sftp, control.RESULTS + "/" + name)
            if reason:
                skipped[name] = reason
            elif data is not None:
                dest = "RESULTS.generated.md" if name == "RESULTS.zh.md" else name
                control._save_bytes(target / dest, data)
                files.append(dest)
    control._save_bytes(
        target / "status.json",
        (json.dumps(control._compact(value), indent=2) + "\n").encode("utf-8"),
    )
    return {**control._compact(value), "pulled_files": files, "skipped": skipped}


def race(transport) -> dict:
    states = {}
    for lane in ("long", "short"):
        configure(lane)
        states[lane] = control.status(transport)

    p = configure("long")
    with control.remote.sftp_client(transport) as sftp:
        winner = control._json(sftp, p["race_root"] + "/control/winner.json")

    winner_lane = None
    if winner is not None:
        if winner.get("protocol") != p["protocol"] or winner.get("lane") not in states:
            raise RuntimeError("Shared winner belongs to a different protocol or lane")
        winner_lane = winner["lane"]
        winner_handle = states[winner_lane].get("handle") or {}
        if (winner.get("job_id") != winner_handle.get("job_id")
                or winner.get("job_name") != winner_handle.get("job_name")):
            raise RuntimeError("Shared winner does not match its durable PBS handle")
    else:
        running = [
            lane for lane, value in states.items()
            if (value.get("scheduler") or {}).get("query_rc") == 0
            and (value.get("scheduler") or {}).get("job_state") == "R"
        ]
        if len(running) == 1:
            winner_lane = running[0]

    cancellations = []
    if winner_lane:
        other = "short" if winner_lane == "long" else "long"
        value = states[other]
        scheduler = value.get("scheduler") or {}
        handle = value.get("handle") or {}
        if scheduler.get("query_rc") == 0 and scheduler.get("job_state") in ("Q", "R", "H"):
            configure(other)
            if (
                scheduler.get("Job_Name") != control.JOB_NAME
                or not control.HANDLE.fullmatch(str(handle.get("job_id", "")))
                or handle.get("protocol") != control.PROTOCOL_ID
                or handle.get("remote_root") != control.ROOT
            ):
                raise RuntimeError("Refusing cancellation without the exact durable loser handle")
            rc, output = control.remote.command(
                transport, "qdel " + shlex.quote(handle["job_id"]), timeout=35
            )
            after = control._scheduler(transport, handle["job_id"])
            verified = (
                rc == 0
                and after.get("query_rc") == 0
                and after.get("Job_Name") == control.JOB_NAME
                and after.get("job_state") == "F"
            )
            cancellations.append(dict(
                lane=other, job_id=handle["job_id"], qdel_rc=rc,
                initial_job_state=scheduler.get("job_state"),
                response=output[-500:], scheduler_after=after,
                exact_loser_cancellation_verified=verified,
            ))

    result = dict(
        checked_utc=datetime.now(timezone.utc).isoformat(),
        winner=winner, winner_lane=winner_lane,
        lanes={lane: control._compact(value) for lane, value in states.items()},
        cancellations=cancellations,
    )
    control._save_bytes(
        HERE / "race_status.json",
        (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    return result


control._local_bytes = local_bytes
control._save_state = save_state
control.complete_gate = complete_gate
control.pull = pull


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "submit", "status", "verify", "logs", "pull", "race"))
    parser.add_argument("--lane", choices=("long", "short"), default="long")
    args = parser.parse_args()
    configure(args.lane)
    jump, transport = control.direct_connect()
    try:
        if args.action == "race":
            result = race(transport)
        elif args.action == "upload":
            result = {"uploaded": control.upload(transport)}
        elif args.action == "submit":
            result = control.submit(transport)
        elif args.action == "status":
            result = control._compact(control.status(transport))
        elif args.action == "logs":
            result = control.logs(transport)
        elif args.action == "pull":
            result = pull(transport)
        else:
            value = control.status(transport)
            result = {
                **control._compact(value),
                "verified_complete": complete_gate(value),
            }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if args.action == "verify" and not result["verified_complete"] else 0
    finally:
        transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
