"""Small-file upload and durable PBS submission receipts for the experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
HANDLE = re.compile(r"^\d+\.[A-Za-z0-9_.-]+$")
DEFAULT_STAGES = {
    "language_cache": ("cpvq24lang", "language_cache.pbs"),
    "bootstrap_bundle": ("cpvq24b2", "bootstrap_bundle.pbs"),
    "baseline_followup_bundle": ("cpvq24c2", "baseline_followup_bundle.pbs"),
    "reader_math": ("cpvq24math", "reader_math.pbs"),
    "rotation_diagnostic": ("cpvq24orth", "rotation_diagnostic.pbs"),
    "result_extract": ("cpvq24extract", "result_extract.pbs"),
    "gripper_diagnostic": ("cpvq24grip", "gripper_diagnostic.pbs"),
    "prepare": ("cpvq24prep", "prepare.pbs"),
    "teacher": ("cpvq24teach", "teacher.pbs"), "quarot": ("cpvq24qrot", "quarot.pbs"),
    "svdquant": ("cpvq24svd", "svdquant.pbs"), "viditq": ("cpvq24vid", "viditq.pbs"),
    "ours_short": ("cpvq24osht", "ours_short.pbs"), "ours_long": ("cpvq24olon", "ours_long.pbs"),
    "perf_bf16": ("cpvq24pbf16", "perf_bf16.pbs"),
    "perf_spinquant": ("cpvq24pspin", "perf_spinquant.pbs"),
    "perf_quarot": ("cpvq24pqrot", "perf_quarot.pbs"),
    "perf_svdquant": ("cpvq24psvd", "perf_svdquant.pbs"),
    "perf_viditq": ("cpvq24pvid", "perf_viditq.pbs"),
    "perf_ours": ("cpvq24pours", "perf_ours.pbs"),
    "evaluation": ("cpvq24eval", "evaluation.pbs"),
}
DEFAULT_RESOURCES = {
    "language_cache": {"gpus": 0, "ncpus": 4, "memory_gb": 16, "walltime": "00:20:00", "queue": "normal"},
    "bootstrap_bundle": {"gpus": 3, "ncpus": 48, "memory_gb": 330, "walltime": "04:00:00", "queue": "normal"},
    "baseline_followup_bundle": {"gpus": 2, "ncpus": 32, "memory_gb": 220,
                                 "walltime": "04:00:00", "queue": "normal",
                                 "required_execution_queue": "g2"},
    "reader_math": {"gpus": 0, "ncpus": 4, "memory_gb": 16, "walltime": "00:20:00", "queue": "normal"},
    "rotation_diagnostic": {"gpus": 0, "ncpus": 4, "memory_gb": 16, "walltime": "00:20:00", "queue": "normal"},
    "result_extract": {"gpus": 0, "ncpus": 4, "memory_gb": 16, "walltime": "00:20:00", "queue": "normal"},
    "gripper_diagnostic": {"gpus": 0, "ncpus": 1, "memory_gb": 16, "walltime": "00:20:00", "queue": "normal"},
    "prepare": {"gpus": 0, "ncpus": 4, "memory_gb": 32, "walltime": "04:00:00", "queue": "normal"},
    "teacher": {"gpus": 1, "ncpus": 16, "memory_gb": 110, "walltime": "01:59:00", "queue": "normal"},
    "quarot": {"gpus": 1, "ncpus": 16, "memory_gb": 110, "walltime": "01:59:00", "queue": "normal"},
    "svdquant": {"gpus": 1, "ncpus": 16, "memory_gb": 110, "walltime": "01:59:00", "queue": "normal"},
    "viditq": {"gpus": 1, "ncpus": 16, "memory_gb": 110, "walltime": "01:59:00", "queue": "normal"},
    "ours_short": {"gpus": 1, "ncpus": 16, "memory_gb": 110, "walltime": "01:59:00", "queue": "normal"},
    "ours_long": {"gpus": 1, "ncpus": 16, "memory_gb": 110, "walltime": "12:00:00", "queue": "normal"},
    **{stage: {"gpus": 1, "ncpus": 16, "memory_gb": 110, "walltime": "01:59:00", "queue": "normal"}
       for stage in ("perf_bf16", "perf_spinquant", "perf_quarot", "perf_svdquant", "perf_viditq", "perf_ours")},
    "evaluation": {"gpus": 4, "ncpus": 64, "memory_gb": 440, "walltime": "24:00:00", "queue": "normal"},
}


def load_protocol(path: Path | None = None) -> dict:
    return json.loads((path or HERE / "protocol.json").read_text(encoding="utf-8"))


def stages(protocol: dict) -> dict[str, tuple[str, str]]:
    result = dict(DEFAULT_STAGES)
    for stage, fields in protocol.get("stages", {}).items():
        result[stage] = (fields.get("job_name", result.get(stage, (stage, ""))[0]),
                         fields.get("script", result.get(stage, ("", stage + ".pbs"))[1]))
    return result


def resources(protocol: dict, stage: str) -> dict:
    value = dict(DEFAULT_RESOURCES[stage])
    value.update(protocol.get("resources", {}).get(stage, {}))
    # Execution queues are from_route_only; normal performs the resource-based routing.
    if value.get("required_execution_queue") and value['queue'] != 'normal':
        raise ValueError('Submit through normal and verify the required execution queue after routing')
    if stage == "evaluation":
        stage_name, script = stages(protocol)["evaluation"]
        workers = protocol.get("load_balancing", {}).get("persistent_workers")
        profile = (value.get("gpus"), value.get("required_execution_queue"), workers, script)
        single_node = (profile == (4, "g3", 4, "evaluation.pbs")
                       and value.get("ncpus") == 64 and value.get("memory_gb") == 440
                       and value.get("nodes", 1) == 1 and value.get("gpus_per_node", 4) == 4
                       and value.get("ncpus_per_node", 64) == 64
                       and value.get("memory_gb_per_node", 440) == 440)
        two_nodes = (profile == (8, "g4", 8, "evaluation_g4.pbs")
                     and value.get("ncpus") == 128 and value.get("memory_gb") == 880
                     and value.get("nodes") == 2 and value.get("gpus_per_node") == 4
                     and value.get("ncpus_per_node") == 64
                     and value.get("memory_gb_per_node") == 440)
        if (not (single_node or two_nodes) or value.get("walltime") != "24:00:00"
                or value.get("queue") != "normal" or stage_name != "cpvq24eval"):
            raise ValueError("evaluation must match the exact 4-GPU/g3 or 8-GPU/two-node g4 profile")
        value.setdefault("nodes", 1)
        value.setdefault("gpus_per_node", 4)
        value.setdefault("ncpus_per_node", 64)
        value.setdefault("memory_gb_per_node", 440)
    return value


def summary_relative_path(protocol: dict, stage: str) -> str:
    if stage == 'language_cache':
        return 'results/language_cache/summary.json'
    if stage == 'bootstrap_bundle':
        return 'results/bootstrap_bundle/summary.json'
    if stage == 'baseline_followup_bundle':
        return 'results/baseline_followup_bundle/summary.json'
    if stage == 'reader_math':
        return 'results/reader_math/reader_math.json'
    if stage == 'rotation_diagnostic':
        return 'results/rotation_diagnostic/summary.json'
    if stage == 'result_extract':
        return 'results/reporting/summary.json'
    if stage == 'gripper_diagnostic':
        return 'results/reporting/summary.json'
    if stage == "prepare":
        return "results/preparation/summary.json"
    if stage == "teacher":
        return "results/teacher/summary.json"
    if stage in ("quarot", "svdquant", "viditq"):
        return f"results/{stage}/summary.json"
    if stage in ("ours_short", "ours_long"):
        return "results/ours/summary.json"
    if stage.startswith("perf_"):
        arm = protocol.get("stage_arms", {}).get(stage)
        if arm not in protocol.get("arms", []):
            raise ValueError(f"Performance stage has no protocol-bound arm: {stage}")
        return f"results/performance/{arm}/summary.json"
    if stage == "evaluation":
        return "results/evaluation/summary.json"
    raise ValueError(f"No summary path is defined for stage: {stage}")


def validate_checkpointed_summary(protocol: dict, stage: str, summary: dict, previous: dict) -> None:
    if summary.get("protocol") != protocol.get("protocol") or summary.get("status") != "checkpointed":
        raise RuntimeError(f"{stage} resume requires a matching checkpointed summary")
    if summary.get("pbs_job_id") != previous.get("job_id"):
        raise RuntimeError(f"{stage} checkpoint summary belongs to a different PBS job")
    expected_arm = protocol.get("stage_arms", {}).get(stage)
    if expected_arm and summary.get("arm") != expected_arm:
        raise RuntimeError(f"{stage} checkpoint summary names a different arm")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def remote_module():
    helper = HERE.parent / "fastwam-libero-plus-pilot" / "remote.py"
    spec = importlib.util.spec_from_file_location("cosmos_vqwam_remote", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Trusted remote helper not found: {helper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _qstat_attribute(output: str, key: str) -> str | None:
    """Read an attribute and its indented continuation lines from qstat -xf."""
    pattern = re.compile(r"^(?: {4}|(?=\S))" + re.escape(key) + r"\s*=\s*(.*)$")
    lines = output.splitlines()
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        parts = [match.group(1).strip()]
        for continuation in lines[index + 1:]:
            if not continuation.strip():
                break
            expanded = continuation.expandtabs(8)
            indent = len(expanded) - len(expanded.lstrip())
            if indent <= 4:
                break
            parts.append(continuation.strip())
        return "".join(part for part in parts if part)
    return None


def parse_status(remote, transport, job_id: str) -> dict:
    if not HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS job handle")
    rc, output = remote.command(transport, "qstat -xf " + shlex.quote(job_id), timeout=35)
    result = {"job_id": job_id, "query_rc": rc, "observed_utc": datetime.now(timezone.utc).isoformat()}
    keys = ("Job_Name", "job_state", "queue", "exec_host", "Exit_status", "comment",
            "resources_used.walltime", "stime", "depend")
    for key in keys:
        result[key] = _qstat_attribute(output, key)
    return result


def _explicit_exit_status(status: dict) -> int | None:
    value = status.get("Exit_status")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pbs_field_missing(value) -> bool:
    return value is None or str(value).strip().lower() in ("", "none")


def _repair_kind(current: dict) -> str:
    if current.get("job_state") not in ("F", "X"):
        raise RuntimeError("Repair requires an identity-verified terminal PBS job")
    exit_status = _explicit_exit_status(current)
    if exit_status is not None and exit_status != 0:
        return "nonzero_exit"
    raise RuntimeError("Repair requires an explicit nonzero Exit_status; use repair_never_started for a dependency terminal")


def _known_dependency_stage(parent_id: str, protocol: dict) -> str | None:
    match = _known_dependency_handle(parent_id, protocol)
    return match[0] if match else None


def _known_dependency_handle(parent_id: str, protocol: dict) -> tuple[str, dict, str] | None:
    """Find one unambiguous, identity-matched current or archived local handle."""
    stage_map = stages(protocol)
    candidates = []
    for stage, (expected_name, _) in stage_map.items():
        paths = [HERE / f"{stage}_handle.json", *HERE.glob(f"{stage}_r*_handle.json")]
        for path in paths:
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (not isinstance(receipt, dict) or receipt.get("job_id") != parent_id
                    or receipt.get("job_name") != expected_name
                    or receipt.get("stage") not in (None, stage)
                    or (receipt.get("protocol") is not None
                        and receipt.get("protocol") != protocol.get("protocol"))):
                continue
            candidates.append((stage, receipt, path))
    if not candidates:
        return None
    identities = {(stage, receipt.get("job_name"), receipt.get("dependency"))
                  for stage, receipt, _ in candidates}
    if len(identities) != 1:
        raise RuntimeError("Local handles for the recorded dependency conflict in identity or dependency")
    # Prefer the current alias, but retain a deterministic archived path in the proof.
    candidates.sort(key=lambda row: (row[2].name != f"{row[0]}_handle.json", row[2].name))
    return candidates[0][0], candidates[0][1], str(candidates[0][2])


def _afterok_matches_exactly(depend: str | None, parent_id: str) -> bool:
    compact = re.sub(r"\s+", "", depend or "").rstrip(",")
    clauses = [part for part in compact.split(",") if part]
    afterok_ids = []
    for clause in clauses:
        kind, separator, value = clause.partition(":")
        if separator and kind == "afterok":
            afterok_ids.extend(item.split("@", 1)[0] for item in value.split(":") if item)
    return len(afterok_ids) == 1 and afterok_ids[0] == parent_id


def _never_started_status(status: dict) -> bool:
    return (status.get("job_state") in ("F", "X")
            and _pbs_field_missing(status.get("Exit_status"))
            and _pbs_field_missing(status.get("stime"))
            and _pbs_field_missing(status.get("exec_host"))
            and _pbs_field_missing(status.get("resources_used.walltime")))


def _verify_never_started_dependency(remote, transport, protocol: dict,
                                     current: dict, previous: dict) -> dict:
    stage_map = stages(protocol)
    child_id = previous.get("job_id")
    child_name = previous.get("job_name")
    parent_id = previous.get("dependency")
    if (not HANDLE.fullmatch(str(child_id or "")) or child_name not in {entry[0] for entry in stage_map.values()}
            or current.get("query_rc") != 0 or current.get("job_id") != child_id
            or current.get("Job_Name") != child_name or not _never_started_status(current)
            or not parent_id or not HANDLE.fullmatch(str(parent_id))):
        raise RuntimeError("Never-started repair requires an identity-verified F/X child with missing Exit_status "
                           "and no stime, exec_host, or walltime")
    if not _afterok_matches_exactly(current.get("depend"), str(parent_id)):
        raise RuntimeError("Child qstat depend must contain one exact afterok reference to its recorded parent")

    chain = []
    visited = {child_id}
    next_id = str(parent_id)
    for depth in range(1, 9):
        if next_id in visited:
            raise RuntimeError("Dependency ancestry contains a cycle")
        visited.add(next_id)
        known = _known_dependency_handle(next_id, protocol)
        if known is None:
            raise RuntimeError("Recorded dependency has no identity-matched current or archived local handle")
        parent_stage, parent_handle, handle_path = known
        parent_name = stage_map[parent_stage][0]
        parent = parse_status(remote, transport, next_id)
        if (parent.get("query_rc") != 0 or parent.get("job_id") != next_id
                or parent.get("Job_Name") != parent_name):
            raise RuntimeError("Dependency ancestor cannot be identity-verified from qstat")
        exit_status = _explicit_exit_status(parent)
        chain_row = {"job_id": next_id, "stage": parent_stage, "job_name": parent_name,
                     "handle_path": handle_path, "job_state": parent.get("job_state"),
                     "recorded_dependency": parent_handle.get("dependency"),
                     "qstat_depend": parent.get("depend"), "exit_status": exit_status}
        chain.append(chain_row)

        # The first identity-matched F ancestor with a real nonzero exit is the root cause.
        if parent.get("job_state") == "F" and exit_status not in (None, 0):
            cascade = len(chain) > 1
            return {
                "repair_kind": ("never_started_dependency_cascade_terminal_exit_unavailable" if cascade
                                else "never_started_dependency_terminal_exit_unavailable"),
                "proof": {
                    "child": {"job_id": child_id, "job_name": child_name,
                              "job_state": current.get("job_state"),
                              "depend": current.get("depend"), "recorded_dependency": parent_id},
                    "ancestry": chain,
                    "root_failed_ancestor": {"job_id": next_id, "stage": parent_stage,
                                             "job_name": parent_name, "job_state": "F",
                                             "exit_status": exit_status},
                    "summary": (f"Verified {len(chain) - 1} never-started intermediate dependency job(s) "
                                f"and explicit nonzero terminal ancestor {next_id} exit={exit_status}.")
                    if cascade else
                    f"Verified direct dependency {next_id} is terminal F with explicit nonzero exit={exit_status}.",
                },
            }

        if not _never_started_status(parent):
            raise RuntimeError("Dependency intermediary is active, started, has an Exit_status, or records walltime")
        recorded_parent = parent_handle.get("dependency")
        if not recorded_parent or not HANDLE.fullmatch(str(recorded_parent)):
            raise RuntimeError("Never-started dependency intermediary lacks a valid recorded parent")
        if not _afterok_matches_exactly(parent.get("depend"), str(recorded_parent)):
            raise RuntimeError("Intermediate qstat depend must contain one exact afterok reference to its recorded parent")
        next_id = str(recorded_parent)

    raise RuntimeError("Dependency ancestry exceeds the eight-level safety limit")


def _require_checkpoint_marker(remote, transport, root: str, previous: dict) -> None:
    marker = root + "/artifacts/" + previous["job_id"] + "/PIPELINE_CHECKPOINTED"
    rc, output = remote.command(transport, "test -f " + shlex.quote(marker), timeout=20)
    if rc:
        raise RuntimeError("Checkpointed resume requires the matching per-job PIPELINE_CHECKPOINTED marker")


def upload(remote, transport, protocol: dict) -> dict:
    root = protocol["remote_root"]
    rc, output = remote.command(transport, "mkdir -p " + shlex.quote(root + "/control") + " " + shlex.quote(root + "/artifacts"))
    if rc:
        raise RuntimeError(output)
    names = []
    with remote.sftp_client(transport) as sftp:
        for path in sorted(HERE.iterdir()):
            if path.suffix not in {".py", ".pbs", ".sh"} and path.name != "protocol.json":
                continue
            data = path.read_bytes().replace(b"\r\n", b"\n")
            if len(data) > remote.LIMIT:
                raise RuntimeError(f"Refusing non-small control file: {path.name}")
            destination = root + "/" + path.name
            with sftp.file(destination + ".tmp", "wb") as stream:
                stream.write(data)
            sftp.posix_rename(destination + ".tmp", destination)
            names.append(path.name)
    return {"uploaded": names, "remote_root": root}


def read_remote_json(remote, transport, path: str) -> dict:
    with remote.sftp_client(transport) as sftp:
        with sftp.file(path, "rb") as stream:
            data = stream.read(remote.LIMIT + 1)
    if len(data) > remote.LIMIT:
        raise RuntimeError(f"Remote control receipt exceeds the small-file limit: {path}")
    return json.loads(data)


def write_remote_json(remote, transport, path: str, value: dict) -> None:
    with remote.sftp_client(transport) as sftp:
        with sftp.file(path + ".tmp", "wb") as stream:
            stream.write((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())
        sftp.posix_rename(path + ".tmp", path)


def _remote_exists(remote, transport, path: str) -> bool:
    rc, output = remote.command(transport, "test -e " + shlex.quote(path), timeout=20)
    if rc == 0:
        return True
    if rc == 1:
        return False
    raise RuntimeError(f"Could not verify remote path existence for {path}: {output}")


def _summary_update_counts(value: object, path: str = "") -> list[dict]:
    updates = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in ("completed_updates", "rotation_completed"):
                try:
                    count = float(item)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"Unverifiable update counter at {child_path}") from exc
                if not count.is_integer():
                    raise RuntimeError(f"Non-integer update counter at {child_path}")
                if count != 0:
                    updates.append({"path": child_path, "value": item})
            else:
                updates.extend(_summary_update_counts(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            updates.extend(_summary_update_counts(item, f"{path}[{index}]"))
    return updates


def reconcile_bootstrap_dependencies(remote, transport, protocol: dict) -> dict:
    """Mirror verified allocation-time dependency migrations into durable control handles."""
    root = protocol['remote_root']
    ready = read_remote_json(remote, transport, root + '/control/bootstrap_race/submissions_ready.json')
    winner = read_remote_json(remote, transport, root + '/control/bootstrap_race/ready_for_work.json')
    if winner.get('registration') != ready.get('registration') or winner.get('candidate') != 'bundle':
        return {'updated': [], 'reason': 'single_path_or_unmatched_registration'}
    new_parent = ready['bundle']['job_id']
    proofs = {row['job_id']: row for row in winner['dependency_migrations']}
    updated = []
    for row in ready['dependents']:
        if row['job_id'] not in proofs:
            raise RuntimeError('Winner receipt lacks this dependent migration')
        stage = row['stage']
        current = parse_status(remote, transport, row['job_id'])
        if current.get('query_rc') != 0 or current.get('Job_Name') != row['job_name']:
            raise RuntimeError('Cannot verify migrated child identity: ' + row['job_id'])
        for suffix in ('handle', 'intent'):
            path = HERE / f'{stage}_{suffix}.json'
            value = json.loads(path.read_text(encoding='utf-8'))
            if value.get('job_id') != row['job_id'] or value.get('job_name') != row['job_name']:
                raise RuntimeError('Local control receipt differs from migrated child')
            if value.get('dependency') != new_parent:
                value['previous_dependency'] = value.get('dependency')
                value['dependency'] = new_parent
                value['dependency_migration'] = {'registration': ready['registration'],
                                                'coordinator_job': winner['coordinator_job']}
                atomic_json(path, value)
                write_remote_json(remote, transport, root + '/control/' + path.name, value)
        updated.append({'stage': stage, 'job_id': row['job_id'], 'dependency': new_parent})
    atomic_json(HERE / 'bootstrap_race/dependency_reconciliation.json', {'updated': updated})
    return {'updated': updated}


def submit(remote, transport, protocol: dict, stage: str, dependency: str | None = None,
           resume: bool = False, repair_note: str | None = None,
           repair_never_started: bool = False, initial_hold: bool = False,
           *, _ours_pair_dependency_repair: bool = False,
           _ours_startup_repair: bool = False,
           _startup_repair_proof: dict | None = None,
           _ours_preclaim_resume: bool = False,
           _ours_preclaim_resume_proof: dict | None = None) -> dict:
    root = protocol["remote_root"]
    stage_map = stages(protocol)
    if stage not in stage_map:
        raise ValueError(f"Unknown stage: {stage}")
    name, script = stage_map[stage]
    if resume and (repair_note is not None or repair_never_started):
        raise ValueError("Resume and repair are separate operations")
    if repair_never_started and repair_note is not None:
        raise ValueError("Never-started repair and explicit failure repair are separate operations")
    if _ours_pair_dependency_repair and (stage not in ("ours_short", "ours_long")
                                         or not repair_never_started or resume):
        raise ValueError("Internal ours-pair repair is valid only for a never-started pair replacement")
    if _ours_startup_repair and (stage not in ("ours_short", "ours_long")
                                 or repair_note is None or not repair_note.strip()
                                 or resume or repair_never_started or not initial_hold
                                 or _startup_repair_proof is None):
        raise ValueError("Internal Ours startup repair requires one held startup proof")
    if _ours_preclaim_resume and (stage not in ("ours_short", "ours_long") or not resume
                                  or repair_never_started or _ours_startup_repair
                                  or _ours_preclaim_resume_proof is None):
        raise ValueError("Internal Ours pre-claim resume requires its exact failed-candidate proof")
    if stage in ("ours_short", "ours_long") and repair_note is not None and not _ours_startup_repair:
        raise RuntimeError("Ours recovery must use the existing race-resume path")
    if repair_never_started and stage in ("ours_short", "ours_long") and not _ours_pair_dependency_repair:
        raise RuntimeError("Ours recovery must use the existing race-resume path")
    handle_path, intent_path = HERE / f"{stage}_handle.json", HERE / f"{stage}_intent.json"
    sequence, previous, repair_kind, repair_proof = 0, None, None, None
    if handle_path.exists():
        previous = json.loads(handle_path.read_text(encoding="utf-8"))
        if (not HANDLE.fullmatch(str(previous.get("job_id", "")))
                or previous.get("job_name") != name
                or previous.get("stage") not in (None, stage)):
            raise RuntimeError("Existing stage handle does not match this stage identity")
        current = parse_status(remote, transport, previous["job_id"])
        if current["query_rc"] != 0 or current["Job_Name"] != name:
            raise RuntimeError("Existing stage handle cannot be identity-verified; refusing a new submission")
        if current["job_state"] in ("Q", "H", "R", "E"):
            if resume or repair_note is not None or repair_never_started:
                raise RuntimeError("The exact previous stage job is still active; refusing another submission")
            return current
        if not resume and repair_note is None and not repair_never_started:
            return current
        if stage in ("ours_short", "ours_long"):
            if _ours_pair_dependency_repair:
                repair_proof = _verify_never_started_dependency(remote, transport, protocol, current, previous)
                repair_kind = repair_proof["repair_kind"]
                if not dependency or dependency == previous.get("dependency"):
                    raise RuntimeError("Ours pair repair requires a new replacement dependency handle")
            elif _ours_startup_repair:
                if (current.get("job_state") != "F" or _explicit_exit_status(current) in (None, 0)
                        or _startup_repair_proof.get("old_winner_job_id") != previous.get("job_id")
                        or _startup_repair_proof.get("stage") != stage):
                    raise RuntimeError("Ours startup proof does not bind the exact failed winner")
                repair_kind = "ours_startup_failure_retry"
                repair_proof = {"repair_kind": repair_kind, "proof": _startup_repair_proof}
            else:
                if not resume:
                    raise RuntimeError("Ours recovery must use the existing race-resume path")
                if current["job_state"] not in ("F", "X"):
                    raise RuntimeError("Only a terminal stage can resume")
                if _ours_preclaim_resume:
                    if (current.get("job_state") != "F" or _explicit_exit_status(current) != 64
                            or _ours_preclaim_resume_proof.get("candidate_job_id") != previous.get("job_id")
                            or _ours_preclaim_resume_proof.get("source_winner_job_id")
                            != previous.get("previous_job_id")
                            or _ours_preclaim_resume_proof.get("stage") != stage):
                        raise RuntimeError("Pre-claim resume proof does not bind this exact failed candidate")
                    repair_kind = "ours_preclaim_startup_failure_resume"
                    repair_proof = {"repair_kind": repair_kind, "proof": _ours_preclaim_resume_proof}
                else:
                    winner = read_remote_json(remote, transport, root + "/control/ours_race/winner.json")
                    summary = read_remote_json(remote, transport, root + "/results/ours/summary.json")
                    if (winner.get("stage") != stage or winner.get("job_id") != previous["job_id"]
                            or winner.get("job_name") != name or summary.get("status") != "checkpointed"
                            or summary.get("pbs_job_id") != previous["job_id"]):
                        raise RuntimeError("Resume requires the same terminal race winner and a validated checkpointed summary")
        else:
            if resume:
                allowed_exits = (0, 2) if stage == 'teacher' else (0,)
                if current["job_state"] != "F" or _explicit_exit_status(current) not in allowed_exits:
                    raise RuntimeError("Checkpointed resume requires PBS state F with explicit Exit_status 0")
                summary = read_remote_json(remote, transport,
                                           root + "/" + summary_relative_path(protocol, stage))
                validate_checkpointed_summary(protocol, stage, summary, previous)
                _require_checkpoint_marker(remote, transport, root, previous)
            else:
                if repair_never_started:
                    repair_proof = _verify_never_started_dependency(remote, transport, protocol, current, previous)
                    repair_kind = repair_proof["repair_kind"]
                    if not dependency or dependency == previous.get("dependency"):
                        raise RuntimeError("Never-started repair requires a new replacement dependency handle")
                else:
                    if not isinstance(repair_note, str) or not repair_note.strip():
                        raise RuntimeError("Repair requires a non-empty repair_note")
                    repair_kind = _repair_kind(current)
        if dependency is None and previous.get("dependency") and not _ours_startup_repair:
            dependency = previous["dependency"]
        sequence = int(previous.get("sequence", 0)) + 1
        atomic_json(HERE / f"{stage}_r{sequence-1}_handle.json", previous)
        handle_path = HERE / f"{stage}_r{sequence}_handle.json"
        intent_path = HERE / f"{stage}_r{sequence}_intent.json"
    elif resume:
        raise RuntimeError("No previous stage handle exists to resume")
    elif repair_note is not None or repair_never_started:
        raise RuntimeError("Repair requires an existing identity-verified stage handle")
    if intent_path.exists():
        raise RuntimeError("Durable submission intent exists without a confirmed receipt; reconcile before retrying")
    rc, existing = remote.command(transport, "qselect -u yguo017 -N " + shlex.quote(name), timeout=35)
    if existing.strip():
        raise RuntimeError("A PBS job with this stage identity already exists: " + existing.strip())
    if rc not in (0, 153):
        raise RuntimeError("Could not establish that no matching PBS job exists")
    if dependency and not HANDLE.fullmatch(dependency):
        raise ValueError("Invalid PBS dependency handle")
    resource = resources(protocol, stage)
    if stage == "evaluation" and resource.get("nodes") == 2:
        select = (f"2:ncpus={resource['ncpus_per_node']}:mem={resource['memory_gb_per_node']}gb:"
                  f"ngpus={resource['gpus_per_node']}:mpiprocs=1")
    else:
        select = f"1:ncpus={resource['ncpus']}:mem={resource['memory_gb']}gb"
        if resource["gpus"]:
            select += f":ngpus={resource['gpus']}"
    args = ["qsub", "-P", "personal-yguo017", "-q", resource["queue"], "-N", name,
            "-l", "select=" + select, "-l", "walltime=" + resource["walltime"], "-j", "oe",
            "-o", root + "/artifacts/" + f"{stage}_r{sequence}.pbs.out"]
    if dependency:
        args += ["-W", "depend=afterok:" + dependency]
    if stage == "evaluation" and resource.get("nodes") == 2:
        args += ["-l", "place=scatter"]
    if initial_hold:
        args.append('-h')
    args.append(root + "/" + script)
    intent = {"stage": stage, "job_name": name, "remote_root": root, "resources": resource,
              "dependency": dependency, "created_utc": datetime.now(timezone.utc).isoformat(),
              "sequence": sequence, "previous_job_id": previous["job_id"] if previous else None,
              "state": "unresolved_until_qsub_receipt", "initial_hold": bool(initial_hold)}
    if repair_kind:
        intent["repair_kind"] = repair_kind
        if repair_proof is not None:
            intent["repair_proof"] = repair_proof["proof"]
        default_note = ("Startup-marker retry failed before race ownership/model load; resume the validated checkpoint"
                        if repair_kind == "ours_preclaim_startup_failure_resume"
                        else "PBS dependency terminal before start; Exit_status unavailable")
        intent["repair_note"] = (repair_note.strip() if isinstance(repair_note, str) and repair_note.strip()
                                  else default_note)
        if repair_never_started:
            intent["repair_source_dependency"] = previous.get("dependency")
    atomic_json(intent_path, intent)
    with remote.sftp_client(transport) as sftp:
        with sftp.file(root + "/control/" + intent_path.name, "wb") as stream:
            stream.write((json.dumps(intent, ensure_ascii=False, indent=2) + "\n").encode())
    rc, output = remote.command(transport, " ".join(map(shlex.quote, args)), timeout=45)
    lines = output.strip().splitlines()
    if rc or not lines or not HANDLE.fullmatch(lines[-1].strip()):
        raise RuntimeError("qsub outcome is uncertain; retain the intent and reconcile, do not repeat: " + output[-1000:])
    result = {**intent, "job_id": lines[-1].strip(), "state": "submitted"}
    atomic_json(handle_path, result)
    atomic_json(HERE / f"{stage}_handle.json", result)
    atomic_json(intent_path, result)
    atomic_json(HERE / f"{stage}_intent.json", result)
    with remote.sftp_client(transport) as sftp:
        with sftp.file(root + "/control/" + handle_path.name, "wb") as stream:
            stream.write((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode())
        with sftp.file(root + "/control/" + f"{stage}_handle.json", "wb") as stream:
            stream.write((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode())
        with sftp.file(root + "/control/" + intent_path.name, "wb") as stream:
            stream.write((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode())
        with sftp.file(root + "/control/" + f"{stage}_intent.json", "wb") as stream:
            stream.write((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode())
    return result


def submit_ours_pair(remote, transport, protocol: dict, dependency: str | None = None) -> dict:
    """Submit both authorized competitors and publish their identities before either loads a model."""
    pair = {}
    for label, stage in (("short", "ours_short"), ("long", "ours_long")):
        result = submit(remote, transport, protocol, stage, dependency=dependency)
        if (not result.get("job_id") or result.get("Job_Name", result.get("job_name"))
                != stages(protocol)[stage][0] or result.get("job_state") not in (None, "Q", "H", "R")):
            raise RuntimeError(f"Cannot establish a current live {stage} handle; race receipt not published")
        pair[label] = {"job_id": result["job_id"], "job_name": stages(protocol)[stage][0]}
    ready = {"protocol": protocol["protocol"], "jobs": pair, **pair,
             "created_utc": datetime.now(timezone.utc).isoformat()}
    path = HERE / "ours_race" / "submissions_ready.json"
    atomic_json(path, ready)
    remote_path = protocol["remote_root"] + "/control/ours_race/submissions_ready.json"
    rc, output = remote.command(transport, "mkdir -p " + shlex.quote(protocol["remote_root"] + "/control/ours_race"))
    if rc:
        raise RuntimeError("Could not create remote ours race receipt directory: " + output)
    with remote.sftp_client(transport) as sftp:
        with sftp.file(remote_path + ".tmp", "wb") as stream:
            stream.write((json.dumps(ready, ensure_ascii=False, indent=2) + "\n").encode())
        sftp.posix_rename(remote_path + ".tmp", remote_path)
    return ready


def repair_ours_pair_dependency(remote, transport, protocol: dict,
                                new_dependency: str) -> dict:
    """Replace only an Ours pair proven to have failed before PBS ever started it."""
    if not HANDLE.fullmatch(str(new_dependency)):
        raise ValueError("Invalid replacement PBS dependency handle")
    root = protocol["remote_root"]
    stage_map = stages(protocol)
    old_handles = {}
    old_dependency = None
    statuses = {}

    for label, stage in (("short", "ours_short"), ("long", "ours_long")):
        name = stage_map[stage][0]
        handle_path = HERE / f"{stage}_handle.json"
        if not handle_path.is_file():
            raise RuntimeError(f"Missing prior {stage} handle")
        previous = json.loads(handle_path.read_text(encoding="utf-8"))
        if (not HANDLE.fullmatch(str(previous.get("job_id", "")))
                or previous.get("job_name") != name
                or previous.get("stage") not in (None, stage)):
            raise RuntimeError(f"Prior {stage} handle has a mismatched identity")
        current = parse_status(remote, transport, previous["job_id"])
        if (current.get("query_rc") != 0 or current.get("Job_Name") != name
                or current.get("job_state") not in ("F", "X")):
            raise RuntimeError(f"Prior {stage} is not identity-verified terminal F/X")
        if not _pbs_field_missing(current.get("resources_used.walltime")):
            raise RuntimeError(f"Prior {stage} has recorded walltime and may have started")
        old_parent_proof = _verify_never_started_dependency(remote, transport, protocol, current, previous)
        if old_dependency is None:
            old_dependency = previous.get("dependency")
        elif previous.get("dependency") != old_dependency:
            raise RuntimeError("The two Ours candidates do not share the same recorded parent")
        old_handles[label] = previous
        statuses[label] = {"job_id": previous["job_id"], "job_name": name,
                           "job_state": current["job_state"], "proof": old_parent_proof}

    if new_dependency == old_dependency:
        raise RuntimeError("Replacement dependency must differ from the failed original parent")

    # The replacement must itself be the exact, currently live bootstrap bundle.
    new_parent_stage = _known_dependency_stage(new_dependency, protocol)
    if new_parent_stage != "bootstrap_bundle":
        raise RuntimeError("Replacement dependency lacks an identity-matched bootstrap bundle handle")
    new_parent = parse_status(remote, transport, new_dependency)
    if (new_parent.get("query_rc") != 0
            or new_parent.get("Job_Name") != stage_map["bootstrap_bundle"][0]
            or new_parent.get("job_state") not in ("Q", "H", "R")):
        raise RuntimeError("Replacement bootstrap bundle is not identity-verified and live")

    race_root = root + "/control/ours_race"
    for relative in ("winner.json", "claim", "resume_claim"):
        path = race_root + "/" + relative
        rc, output = remote.command(transport, "test -e " + shlex.quote(path), timeout=20)
        if rc == 0:
            raise RuntimeError(f"Ours race already has {relative}; pair replacement is unsafe")
        if rc != 1:
            raise RuntimeError(f"Could not prove Ours race {relative} is absent: {output}")

    ready_path = HERE / "ours_race" / "submissions_ready.json"
    if not ready_path.is_file():
        raise RuntimeError("Missing local Ours race readiness receipt")
    old_ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if old_ready.get("protocol") != protocol.get("protocol"):
        raise RuntimeError("Local Ours race readiness receipt has a different protocol")
    for label in ("short", "long"):
        entry = old_ready.get(label)
        if (not isinstance(entry, dict)
                or entry.get("job_id") != old_handles[label]["job_id"]
                or entry.get("job_name") != old_handles[label]["job_name"]):
            raise RuntimeError("Local Ours readiness receipt differs from the exact prior pair")
    remote_ready = read_remote_json(remote, transport, race_root + "/submissions_ready.json")
    if remote_ready != old_ready:
        raise RuntimeError("Remote Ours readiness receipt differs from the local prior pair")

    # Do not archive or submit until both old candidates and the replacement
    # dependency have passed all checks above.
    sequence = max(int(value.get("sequence", 0)) for value in old_handles.values()) + 1
    archive_dir = HERE / "ours_race" / "archive"
    archive_path = archive_dir / f"submissions_ready_r{sequence}.json"
    if archive_path.exists():
        raise RuntimeError("Ours pair archive path already exists; refusing to overwrite it")
    for label, stage in (("short", "ours_short"), ("long", "ours_long")):
        if (HERE / f"{stage}_r{sequence}_intent.json").exists():
            raise RuntimeError(f"Unresolved {stage} intent exists for the next sequence")
    atomic_json(archive_path, old_ready)

    replacement = {}
    for label, stage in (("short", "ours_short"), ("long", "ours_long")):
        result = submit(remote, transport, protocol, stage, dependency=new_dependency,
                        repair_never_started=True, _ours_pair_dependency_repair=True)
        name = stage_map[stage][0]
        if (not result.get("job_id") or result.get("job_name") != name
                or result.get("state") != "submitted"):
            raise RuntimeError(f"New {stage} did not return a durable submitted handle")
        replacement[label] = {"job_id": result["job_id"], "job_name": name}

    ready = {"protocol": protocol["protocol"], "jobs": replacement, **replacement,
             "dependency": new_dependency, "previous_dependency": old_dependency,
             "repaired_from": {label: old_handles[label]["job_id"] for label in ("short", "long")},
             "created_utc": datetime.now(timezone.utc).isoformat()}
    # Publish only after both replacement IDs have durable submit receipts.
    write_remote_json(remote, transport, race_root + "/submissions_ready.json", ready)
    atomic_json(ready_path, ready)
    return {"repaired_dependency": new_dependency, "previous_dependency": old_dependency,
            "jobs": replacement, "local_archive": str(archive_path), "race_receipt": ready,
            "source_dependency_proofs": {label: statuses[label]["proof"]["proof"]
                                          for label in ("short", "long")}}


def _verify_ours_preclaim_resume(remote, transport, protocol, stage, source_winner,
                                 source_status, summary, candidate, candidate_status,
                                 previous_pair, sibling_label, winner_label):
    root = protocol["remote_root"]
    expected_name = stages(protocol)[stage][0]
    candidate_id = candidate.get("job_id")
    source_id = source_winner.get("job_id")
    if (candidate.get("stage") not in (None, stage) or candidate.get("job_name") != expected_name
            or not HANDLE.fullmatch(str(candidate_id or ""))
            or candidate.get("previous_job_id") != source_id
            or type(candidate.get("sequence")) is not int):
        raise RuntimeError("Failed Ours retry handle does not have the exact winner lineage")
    if (candidate_status.get("query_rc") != 0 or candidate_status.get("job_id") != candidate_id
            or candidate_status.get("Job_Name") != expected_name
            or candidate_status.get("job_state") != "F"
            or _explicit_exit_status(candidate_status) != 64):
        raise RuntimeError("Pre-claim Ours retry must be exact F/Exit_status 64")
    if (source_status.get("query_rc") != 0 or source_status.get("job_id") != source_id
            or source_status.get("Job_Name") != expected_name
            or source_status.get("job_state") not in ("F", "X")
            or _explicit_exit_status(source_status) is None):
        raise RuntimeError("Original Ours winner is not identity-verified terminal")
    if (source_winner.get("stage") != stage or source_winner.get("job_name") != expected_name
            or (source_winner.get("protocol") is not None
                and source_winner.get("protocol") != protocol.get("protocol"))):
        raise RuntimeError("Race winner receipt does not match the Ours checkpoint source")
    if (summary.get("protocol") != protocol.get("protocol") or summary.get("arm") != "ours_vqwam"
            or summary.get("status") != "checkpointed" or summary.get("pbs_job_id") != source_id
            or summary.get("rotation_completed") != 8 or summary.get("completed_updates") != 0):
        raise RuntimeError("Checkpoint summary is not the exact committed R8/VQ0 source winner")

    race_root = root + "/control/ours_race"
    pair_path = HERE / "ours_race" / "submissions_ready.json"
    if not pair_path.is_file():
        raise RuntimeError("Missing local Ours pair receipt for pre-claim resume")
    local_pair = json.loads(pair_path.read_text(encoding="utf-8"))
    remote_pair = previous_pair
    if local_pair != remote_pair or remote_pair.get("protocol") != protocol.get("protocol"):
        raise RuntimeError("Local and remote Ours pair receipts differ or bind another protocol")
    winner_entry = remote_pair.get(winner_label)
    sibling = remote_pair.get(sibling_label)
    sibling_stage = "ours_long" if stage == "ours_short" else "ours_short"
    sibling_name = stages(protocol)[sibling_stage][0]
    if (not isinstance(winner_entry, dict) or winner_entry.get("job_id") != candidate_id
            or winner_entry.get("job_name") != expected_name
            or not isinstance(sibling, dict) or not HANDLE.fullmatch(str(sibling.get("job_id", "")))
            or sibling.get("job_name") != sibling_name):
        raise RuntimeError("Ready receipt does not bind the failed candidate and exact sibling")
    sibling_status = parse_status(remote, transport, sibling["job_id"])
    if (sibling_status.get("query_rc") != 0 or sibling_status.get("job_id") != sibling["job_id"]
            or sibling_status.get("Job_Name") != sibling_name
            or sibling_status.get("job_state") not in ("F", "X")):
        raise RuntimeError("Pre-claim resume sibling is not identity-verified terminal")
    remote_handle = read_remote_json(remote, transport, root + "/control/" + f"{stage}_handle.json")
    if remote_handle != candidate:
        raise RuntimeError("Remote current handle differs from the failed local retry handle")

    marker = read_remote_json(remote, transport, race_root + "/startup_repair.json")
    startup_owner = read_remote_json(remote, transport, race_root + "/startup_repair_claim/owner.json")
    if (marker.get("protocol") != protocol.get("protocol") or marker.get("status") != "claimed"
            or marker.get("claimed_job_id") != source_id
            or marker.get("new_winner", {}).get("stage") != stage
            or marker.get("new_winner", {}).get("job_id") != source_id
            or startup_owner.get("stage") != stage or startup_owner.get("job_id") != source_id):
        raise RuntimeError("Existing startup marker/claim belongs to another owner, not the checkpoint source")
    current_winner = read_remote_json(remote, transport, race_root + "/winner.json")
    if (current_winner.get("stage") != stage or current_winner.get("job_id") != source_id
            or current_winner.get("job_name") != expected_name):
        raise RuntimeError("Failed retry acquired or replaced race ownership")
    if _remote_exists(remote, transport, race_root + "/resume_claim"):
        raise RuntimeError("An Ours resume claim is already active")

    checkpoint_path = root + "/results/ours/checkpoint.pt"
    if not _remote_exists(remote, transport, checkpoint_path):
        raise RuntimeError("The committed Ours checkpoint is missing")
    artifacts = root + "/artifacts/" + candidate_id
    if _remote_exists(remote, transport, artifacts + "/control_sources"):
        raise RuntimeError("Failed retry reached source snapshot/model setup; refusing checkpoint reuse")
    log_path = artifacts + "/job.log"
    rc, log_tail = remote.command(transport, "tail -c 131072 -- " + shlex.quote(log_path), timeout=20)
    if rc != 0 or "Startup-repair claim is already held; refusing duplicate retry" not in log_tail:
        raise RuntimeError("Failed retry log lacks the exact stale-startup-claim error")
    if "RACE_RESUME_WINNER" in log_tail:
        raise RuntimeError("Failed retry log shows it acquired Ours resume ownership")

    return {
        "kind": "failed_stale_startup_claim_before_model_load",
        "stage": stage, "source_winner_job_id": source_id,
        "candidate_job_id": candidate_id, "candidate_exit_status": 64,
        "candidate_sequence": candidate["sequence"],
        "sibling_job_id": sibling["job_id"], "sibling_job_name": sibling_name,
        "summary_protocol": summary["protocol"], "summary_status": summary["status"],
        "summary_pbs_job_id": summary["pbs_job_id"],
        "rotation_completed": summary["rotation_completed"],
        "completed_updates": summary["completed_updates"],
        "checkpoint_path": checkpoint_path, "checkpoint_exists": True,
        "race_winner_still_source": True, "startup_claim_owner_job_id": source_id,
        "resume_claim_absent": True, "control_sources_absent": True,
        "stale_startup_error_seen": True, "resume_winner_event_absent": True,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }


def _ours_release_receipt_paths(stage: str, job_id: str):
    filename = "resume_release_" + job_id.replace(".", "_") + ".json"
    return (HERE / "ours_race" / filename, "control/ours_race/" + filename)


def _write_ours_release_receipt(remote, transport, protocol, stage, job_id, receipt):
    local_path, remote_relative = _ours_release_receipt_paths(stage, job_id)
    remote_path = protocol["remote_root"] + "/" + remote_relative
    atomic_json(local_path, receipt)
    write_remote_json(remote, transport, remote_path, receipt)
    remote_readback = read_remote_json(remote, transport, remote_path)
    local_readback = json.loads(local_path.read_text(encoding="utf-8"))
    if remote_readback != receipt or local_readback != receipt:
        raise RuntimeError("Ours resume release receipt did not round-trip; reconcile before retrying")
    return local_path, remote_path


def resume_ours(remote, transport, protocol: dict) -> dict:
    root = protocol["remote_root"]
    race_root = root + "/control/ours_race"
    winner = read_remote_json(remote, transport, race_root + "/winner.json")
    stage = winner.get("stage")
    if stage not in ("ours_short", "ours_long"):
        raise RuntimeError("Race winner receipt does not identify an ours training stage")
    expected_name = stages(protocol)[stage][0]
    winner_id = winner.get("job_id")
    if not HANDLE.fullmatch(str(winner_id or "")):
        raise RuntimeError("Race winner receipt has no valid PBS handle")
    source_status = parse_status(remote, transport, winner_id)
    if (source_status["query_rc"] != 0 or source_status["Job_Name"] != expected_name
            or source_status["job_state"] not in ("F", "X")):
        raise RuntimeError("The exact winner job is not verified terminal")
    summary = read_remote_json(remote, transport, root + "/results/ours/summary.json")
    if summary.get("status") != "checkpointed":
        raise RuntimeError("The ours result is not checkpointed; refusing to resume")
    previous_pair = read_remote_json(remote, transport, race_root + "/submissions_ready.json")
    sibling_label = "long" if stage == "ours_short" else "short"
    winner_label = "short" if stage == "ours_short" else "long"
    sibling = previous_pair.get(sibling_label)
    if not isinstance(sibling, dict) or not HANDLE.fullmatch(str(sibling.get("job_id", ""))):
        raise RuntimeError("Previous race receipt lacks the exact sibling handle")
    sibling_name = stages(protocol)["ours_long" if stage == "ours_short" else "ours_short"][0]
    sibling_status = parse_status(remote, transport, sibling["job_id"])
    if (sibling_status["query_rc"] != 0 or sibling_status["Job_Name"] != sibling_name
            or sibling.get("job_name") != sibling_name or sibling_status["job_state"] not in ("F", "X")):
        raise RuntimeError("Previous sibling is not identity-verified terminal")

    handle_path = HERE / f"{stage}_handle.json"
    if not handle_path.is_file():
        raise RuntimeError("Current Ours stage handle is missing; refusing to resume")
    candidate = json.loads(handle_path.read_text(encoding="utf-8"))
    if (candidate.get("stage") not in (None, stage) or candidate.get("job_name") != expected_name
            or not HANDLE.fullmatch(str(candidate.get("job_id", "")))):
        raise RuntimeError("Current Ours stage handle has a mismatched identity")
    release_local_path, release_remote_relative = _ours_release_receipt_paths(stage, candidate["job_id"])
    release_remote_path = root + "/" + release_remote_relative
    local_release_exists = release_local_path.exists()
    remote_release_exists = _remote_exists(remote, transport, release_remote_path)
    if local_release_exists or remote_release_exists:
        if not (local_release_exists and remote_release_exists):
            raise RuntimeError("Ours resume release receipt is only partially present; reconcile before retrying")
        local_release = json.loads(release_local_path.read_text(encoding="utf-8"))
        remote_release = read_remote_json(remote, transport, release_remote_path)
        if local_release != remote_release or local_release.get("job_id") != candidate["job_id"]:
            raise RuntimeError("Ours resume release receipts disagree; reconcile before retrying")
        if local_release.get("status") != "released":
            raise RuntimeError("Ours resume release outcome is unresolved; reconcile this receipt before retrying")
    preclaim_resume = candidate["job_id"] != winner_id
    proof = None
    if preclaim_resume:
        candidate_status = parse_status(remote, transport, candidate["job_id"])
        proof = _verify_ours_preclaim_resume(remote, transport, protocol, stage, winner,
            source_status, summary, candidate, candidate_status, previous_pair,
            sibling_label, winner_label)
        archive_dir = HERE / "ours_race" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_name = "preclaim_resume_" + candidate["job_id"].replace(".", "_") + "_submissions_ready.json"
        local_archive = archive_dir / archive_name
        remote_archive_dir = race_root + "/archive"
        remote_archive = remote_archive_dir + "/" + archive_name
        if local_archive.exists() or _remote_exists(remote, transport, remote_archive):
            raise RuntimeError("Pre-claim resume pair archive already exists; reconcile before retrying")
        rc, output = remote.command(transport, "mkdir -p " + shlex.quote(remote_archive_dir), timeout=20)
        if rc:
            raise RuntimeError("Could not create remote Ours pair archive directory: " + output)
        atomic_json(local_archive, previous_pair)
        write_remote_json(remote, transport, remote_archive, previous_pair)

    result = submit(remote, transport, protocol, stage, resume=True, initial_hold=True,
                    _ours_preclaim_resume=preclaim_resume,
                    _ours_preclaim_resume_proof=proof)
    if result.get("job_id") is None:
        raise RuntimeError("Resumed stage submission did not return a durable handle")
    new_id = result["job_id"]
    held = parse_status(remote, transport, new_id)
    if (held.get("query_rc") != 0 or held.get("job_id") != new_id
            or held.get("Job_Name") != expected_name or held.get("job_state") != "H"):
        raise RuntimeError(f"Resumed Ours job {new_id} is not identity-verified held; reconcile before release")
    pair = dict(previous_pair)
    pair["created_utc"] = datetime.now(timezone.utc).isoformat()
    pair[winner_label] = {"job_id": new_id, "job_name": expected_name}
    pair["jobs"] = {"short": dict(pair["short"]), "long": dict(pair["long"])}
    pair["jobs"][winner_label] = dict(pair[winner_label])
    if preclaim_resume:
        pair["preclaim_resume"] = {"failed_candidate_job_id": candidate["job_id"],
                                   "source_winner_job_id": winner_id,
                                   "kind": proof["kind"]}
    atomic_json(HERE / "ours_race" / "submissions_ready.json", pair)
    write_remote_json(remote, transport, race_root + "/submissions_ready.json", pair)
    local_pair_readback = json.loads((HERE / "ours_race" / "submissions_ready.json").read_text(encoding="utf-8"))
    remote_pair_readback = read_remote_json(remote, transport, race_root + "/submissions_ready.json")
    if (local_pair_readback != pair or remote_pair_readback != pair
            or pair.get(winner_label, {}).get("job_id") != new_id
            or pair.get("jobs", {}).get(winner_label, {}).get("job_id") != new_id):
        raise RuntimeError("Updated Ours pair did not round-trip with the held resumed job; refusing release")

    release = {
        "protocol": protocol.get("protocol"), "stage": stage, "job_id": new_id,
        "job_name": expected_name, "sequence": result.get("sequence"),
        "status": "pending", "pair_receipt_path": "control/ours_race/submissions_ready.json",
        "pair_readback_confirmed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    release_local_path, release_remote_path = _ours_release_receipt_paths(stage, new_id)
    release_remote_path = root + "/" + release_remote_path
    if release_local_path.exists() or _remote_exists(remote, transport, release_remote_path):
        raise RuntimeError("Ours resume release receipt already exists; reconcile before release")
    release_local_path, release_remote_path = _write_ours_release_receipt(
        remote, transport, protocol, stage, new_id, release)
    try:
        qrls_rc, qrls_output = remote.command(transport, "qrls " + shlex.quote(new_id), timeout=35)
    except Exception as exc:
        qrls_rc, qrls_output = None, f"{type(exc).__name__}: {exc}"
    if qrls_rc != 0:
        release.update({"status": "uncertain", "qrls_rc": qrls_rc,
                        "qrls_output": str(qrls_output)[-1000:],
                        "uncertain_utc": datetime.now(timezone.utc).isoformat()})
        try:
            _write_ours_release_receipt(remote, transport, protocol, stage, new_id, release)
        except Exception:
            pass
        raise RuntimeError(f"qrls outcome for {new_id} is uncertain; reconcile {release_remote_path} before retrying")

    released_status = parse_status(remote, transport, new_id)
    if (released_status.get("query_rc") != 0 or released_status.get("job_id") != new_id
            or released_status.get("Job_Name") != expected_name
            or released_status.get("job_state") not in ("Q", "R", "E", "F", "X")):
        release.update({"status": "uncertain", "qrls_rc": qrls_rc,
                        "qrls_output": str(qrls_output)[-1000:], "pbs_readback": released_status,
                        "uncertain_utc": datetime.now(timezone.utc).isoformat()})
        try:
            _write_ours_release_receipt(remote, transport, protocol, stage, new_id, release)
        except Exception:
            pass
        raise RuntimeError(f"qrls succeeded but {new_id} release is not identity-verified; reconcile {release_remote_path}")

    release.update({"status": "released", "qrls_rc": qrls_rc,
                    "qrls_output": str(qrls_output)[-1000:], "pbs_readback": released_status,
                    "released_utc": datetime.now(timezone.utc).isoformat()})
    try:
        _write_ours_release_receipt(remote, transport, protocol, stage, new_id, release)
    except Exception as exc:
        raise RuntimeError(f"{new_id} was released but its final receipt could not be confirmed; reconcile {release_remote_path}") from exc
    return {"resumed_stage": stage, "job": result, "race_receipt": pair,
            "release_receipt": release, "preclaim_resume_proof": proof}


def repair_ours_startup(remote, transport, protocol: dict, repair_note: str) -> dict:
    """Retry only the failed race winner after proving no optimizer update/checkpoint exists.

    The replacement is submitted held. The caller must verify the returned held ID,
    then explicitly release it after the startup_repair and pair receipts are visible.
    """
    if not isinstance(repair_note, str) or not repair_note.strip():
        raise ValueError("Ours startup repair requires a non-empty repair_note")
    root = protocol["remote_root"]
    race_root = root + "/control/ours_race"
    winner_path = race_root + "/winner.json"
    ready_path = race_root + "/submissions_ready.json"
    marker_path = race_root + "/startup_repair.json"
    stage_map = stages(protocol)

    if _remote_exists(remote, transport, marker_path) or (HERE / "ours_race" / "startup_repair.json").exists():
        raise RuntimeError("An Ours startup-repair marker already exists; reconcile it before retrying")
    if not _remote_exists(remote, transport, race_root + "/claim"):
        raise RuntimeError("Ours startup repair requires the original atomic race claim")
    if _remote_exists(remote, transport, race_root + "/resume_claim"):
        raise RuntimeError("An Ours resume claim exists; startup retry is not allowed")

    winner = read_remote_json(remote, transport, winner_path)
    stage = winner.get("stage")
    if stage not in ("ours_short", "ours_long"):
        raise RuntimeError("Ours race winner receipt does not identify a training stage")
    stage_name = stage_map[stage][0]
    old_id = winner.get("job_id")
    if (not HANDLE.fullmatch(str(old_id or "")) or winner.get("job_name") != stage_name
            or (winner.get("protocol") is not None and winner.get("protocol") != protocol.get("protocol"))):
        raise RuntimeError("Ours race winner receipt has a mismatched stage/job identity")

    sibling_label = "long" if stage == "ours_short" else "short"
    winner_label = "short" if stage == "ours_short" else "long"
    sibling_stage = "ours_long" if stage == "ours_short" else "ours_short"
    sibling_name = stage_map[sibling_stage][0]
    local_ready_path = HERE / "ours_race" / "submissions_ready.json"
    if not local_ready_path.is_file():
        raise RuntimeError("Missing local Ours race readiness receipt")
    old_ready = json.loads(local_ready_path.read_text(encoding="utf-8"))
    remote_ready = read_remote_json(remote, transport, ready_path)
    if (old_ready != remote_ready or old_ready.get("protocol") != protocol.get("protocol")):
        raise RuntimeError("Local and remote Ours pair receipts differ or bind another protocol")
    for label, expected_stage in (("short", "ours_short"), ("long", "ours_long")):
        expected_name = stage_map[expected_stage][0]
        entry = old_ready.get(label)
        jobs_entry = old_ready.get("jobs", {}).get(label)
        if (not isinstance(entry, dict) or not isinstance(jobs_entry, dict)
                or entry.get("job_id") != jobs_entry.get("job_id")
                or entry.get("job_name") != jobs_entry.get("job_name")
                or entry.get("job_name") != expected_name):
            raise RuntimeError("Ours pair receipt has conflicting or mismatched candidate identities")
    winner_entry = old_ready[winner_label]
    sibling_entry = old_ready[sibling_label]
    if winner_entry.get("job_id") != old_id or winner_entry.get("job_name") != stage_name:
        raise RuntimeError("Winner receipt differs from the exact current pair entry")
    sibling_id = sibling_entry.get("job_id")
    if not HANDLE.fullmatch(str(sibling_id or "")) or sibling_entry.get("job_name") != sibling_name:
        raise RuntimeError("Ours sibling receipt has an invalid identity")

    local_handles = {}
    for candidate_stage, candidate_id, candidate_name in (
            (stage, old_id, stage_name), (sibling_stage, sibling_id, sibling_name)):
        path = HERE / f"{candidate_stage}_handle.json"
        if not path.is_file():
            raise RuntimeError(f"Missing local {candidate_stage} PBS handle")
        handle = json.loads(path.read_text(encoding="utf-8"))
        if (handle.get("job_id") != candidate_id or handle.get("job_name") != candidate_name
                or handle.get("stage") not in (None, candidate_stage)):
            raise RuntimeError(f"Local {candidate_stage} handle differs from the exact race receipt")
        local_handles[candidate_stage] = handle

    old_status = parse_status(remote, transport, old_id)
    if (old_status.get("query_rc") != 0 or old_status.get("job_id") != old_id
            or old_status.get("Job_Name") != stage_name or old_status.get("job_state") != "F"
            or _explicit_exit_status(old_status) in (None, 0)):
        raise RuntimeError("Current Ours winner must be identity-verified F with explicit nonzero Exit_status")
    sibling_status = parse_status(remote, transport, sibling_id)
    if (sibling_status.get("query_rc") != 0 or sibling_status.get("job_id") != sibling_id
            or sibling_status.get("Job_Name") != sibling_name
            or not _never_started_status(sibling_status)):
        raise RuntimeError("The other Ours candidate must be terminal and verified never started")

    for candidate_name in (stage_name, sibling_name):
        rc, matches = remote.command(
            transport, "qselect -u yguo017 -N " + shlex.quote(candidate_name), timeout=35)
        if matches.strip() or rc not in (0, 153):
            raise RuntimeError(f"Cannot prove there is no live PBS job named {candidate_name}: {matches.strip()}")

    checkpoint_path = root + "/results/ours/checkpoint.pt"
    if _remote_exists(remote, transport, checkpoint_path):
        raise RuntimeError("Ours checkpoint.pt exists; startup retry would discard trained state")
    summary_path = root + "/results/ours/summary.json"
    summary_exists = _remote_exists(remote, transport, summary_path)
    summary = read_remote_json(remote, transport, summary_path) if summary_exists else None
    update_counts = _summary_update_counts(summary) if summary is not None else []
    if update_counts:
        raise RuntimeError(f"Ours summary records optimizer/rotation updates: {update_counts}")
    if summary is not None:
        if (summary.get("protocol") not in (None, protocol.get("protocol"))
                or summary.get("pbs_job_id") not in (None, old_id, sibling_id)
                or summary.get("status") in ("complete", "checkpointed")):
            raise RuntimeError("Ours summary is unrelated or indicates a resumable/completed run")

    proof = {
        "protocol": protocol.get("protocol"),
        "old_winner_job_id": old_id,
        "old_winner_job_name": stage_name,
        "stage": stage,
        "old_winner_exit_status": _explicit_exit_status(old_status),
        "sibling_job_id": sibling_id,
        "sibling_job_name": sibling_name,
        "sibling_state": sibling_status.get("job_state"),
        "sibling_never_started": True,
        "old_winner_handle_archive": "control/" + f"{stage}_r{int(local_handles[stage].get('sequence', 0))}_handle.json",
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": False,
        "summary_path": summary_path,
        "summary_exists": summary_exists,
        "summary_update_counts": update_counts,
        "no_completed_updates": True,
        "no_checkpoint": True,
        "same_name_qselect_empty": True,
        "old_winner_receipt": winner,
        "old_pair_receipt": old_ready,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }

    sequence = int(local_handles[stage].get("sequence", 0)) + 1
    old_handle_archive = f"{stage}_r{sequence - 1}_handle.json"
    if ((HERE / old_handle_archive).exists()
            and load_protocol(HERE / old_handle_archive) != local_handles[stage]):
        raise RuntimeError("Ours startup-repair handle archive differs; refusing overwrite")
    archive_dir = HERE / "ours_race" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    winner_archive = archive_dir / f"startup_repair_r{sequence}_winner.json"
    pair_archive = archive_dir / f"startup_repair_r{sequence}_submissions_ready.json"
    if winner_archive.exists() or pair_archive.exists():
        raise RuntimeError("Ours startup-repair archive already exists; refusing overwrite")
    remote_archive = race_root + "/archive"
    rc, output = remote.command(transport, "mkdir -p " + shlex.quote(remote_archive), timeout=20)
    if rc:
        raise RuntimeError("Could not create remote Ours repair archive directory: " + output)
    remote_winner_archive = remote_archive + "/" + winner_archive.name
    remote_pair_archive = remote_archive + "/" + pair_archive.name
    if (_remote_exists(remote, transport, remote_winner_archive)
            or _remote_exists(remote, transport, remote_pair_archive)):
        raise RuntimeError("Remote Ours startup-repair archive already exists; refusing overwrite")
    if (_remote_exists(remote, transport, root + "/control/" + old_handle_archive)
            and read_remote_json(remote, transport, root + "/control/" + old_handle_archive)
            != local_handles[stage]):
        raise RuntimeError("Remote Ours startup-repair handle archive differs; refusing overwrite")

    atomic_json(winner_archive, winner)
    atomic_json(pair_archive, old_ready)
    write_remote_json(remote, transport, remote_winner_archive, winner)
    write_remote_json(remote, transport, remote_pair_archive, old_ready)
    write_remote_json(remote, transport, root + "/control/" + old_handle_archive,
                      local_handles[stage])

    result = submit(remote, transport, protocol, stage, dependency=None,
                    repair_note=repair_note, initial_hold=True,
                    _ours_startup_repair=True, _startup_repair_proof=proof)
    new_id = result.get("job_id")
    if not HANDLE.fullmatch(str(new_id or "")):
        raise RuntimeError("Ours startup retry did not return a valid PBS handle")
    held = parse_status(remote, transport, new_id)
    if (held.get("query_rc") != 0 or held.get("job_id") != new_id
            or held.get("Job_Name") != stage_name or held.get("job_state") != "H"):
        raise RuntimeError(f"Ours startup retry {new_id} is not identity-verified held; do not release")

    new_winner_entry = {"job_id": new_id, "job_name": stage_name}
    new_ready = dict(old_ready)
    new_ready[winner_label] = new_winner_entry
    new_ready["jobs"] = {"short": dict(new_ready["short"]), "long": dict(new_ready["long"])}
    new_ready["jobs"][winner_label] = dict(new_winner_entry)
    new_ready["startup_repair"] = {"old_winner_job_id": old_id, "new_winner_job_id": new_id,
                                    "stage": stage, "sibling_job_id": sibling_id}
    new_ready["created_utc"] = datetime.now(timezone.utc).isoformat()
    marker = {
        "protocol": protocol.get("protocol"),
        "status": "ready_for_release",
        "repair_note": repair_note.strip(),
        "old_winner": {"stage": stage, "job_id": old_id, "job_name": stage_name},
        "new_winner": {"stage": stage, "job_id": new_id, "job_name": stage_name},
        "sibling": {"stage": sibling_stage, "job_id": sibling_id, "job_name": sibling_name},
        "zero_updates": True,
        "no_checkpoint": True,
        "proof": proof,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Both remote records are atomic; keep the new stage held until the caller releases it.
    write_remote_json(remote, transport, marker_path, marker)
    write_remote_json(remote, transport, ready_path, new_ready)
    atomic_json(HERE / "ours_race" / "startup_repair.json", marker)
    atomic_json(local_ready_path, new_ready)
    result["held_for_registration"] = True
    return {"stage": stage, "old_winner_job_id": old_id, "new_job_id_held": new_id,
            "sibling_job_id": sibling_id, "requires_qrls": True,
            "release_command": "qrls " + new_id,
            "startup_repair": marker, "race_receipt": new_ready,
            "winner_archive": str(winner_archive), "pair_archive": str(pair_archive),
            "job": result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "submit", "submit-ours-pair", "resume-ours",
                                            "repair-ours-startup", "status", "pull"))
    parser.add_argument("--stage", choices=tuple(DEFAULT_STAGES), default="prepare")
    parser.add_argument("--dependency")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repair-note")
    parser.add_argument("--repair-never-started", action="store_true")
    parser.add_argument("--files", nargs="*", default=[])
    args = parser.parse_args()
    protocol = load_protocol()
    remote = remote_module()
    jump, transport = remote.direct_connect()
    try:
        if args.action == "upload":
            result = upload(remote, transport, protocol)
        elif args.action == "submit":
            if args.stage in ("ours_short", "ours_long"):
                raise RuntimeError("Use submit-ours-pair for the initial race and resume-ours for checkpointed continuation")
            result = submit(remote, transport, protocol, args.stage, args.dependency, args.resume,
                            args.repair_note, args.repair_never_started)
        elif args.action == "submit-ours-pair":
            result = submit_ours_pair(remote, transport, protocol, args.dependency)
        elif args.action == "resume-ours":
            result = resume_ours(remote, transport, protocol)
        elif args.action == "repair-ours-startup":
            result = repair_ours_startup(remote, transport, protocol, args.repair_note or "")
        elif args.action == "status":
            handle = json.loads((HERE / f"{args.stage}_handle.json").read_text(encoding="utf-8"))
            result = parse_status(remote, transport, handle["job_id"])
            atomic_json(HERE / f"{args.stage}_status.json", result)
        else:
            root = protocol["remote_root"]
            result = {}
            with remote.sftp_client(transport) as sftp:
                for name in args.files:
                    if Path(name).is_absolute() or ".." in Path(name).parts:
                        raise ValueError("Only task-relative small files may be retrieved")
                    with sftp.file(root + "/" + name.replace("\\", "/"), "rb") as stream:
                        data = stream.read(remote.LIMIT + 1)
                    if len(data) > remote.LIMIT:
                        raise RuntimeError("Retrieved file exceeds the small-file limit: " + name)
                    destination = HERE / "retrieved" / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    result[name] = {"bytes": len(data), "path": str(destination)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
