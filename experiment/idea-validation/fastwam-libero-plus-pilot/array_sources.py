#!/usr/bin/env python3
"""Write aggregate.py shard sources from one PBS array's compact qstat state."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

import remote
from submit_array import RANGE_PLAN_REMOTE, validate_range_plan

TOTAL_VARIANTS = 1400
ARMS = 4
# PBS uses X for completed/deleted array subjobs; Exit_status remains required.
ARRAY_TERMINAL_STATES = frozenset(("F", "X"))
PARENT_RE = re.compile(r"^(\d+)\[\]\.([A-Za-z0-9_.-]+)$")
CHILD_RE = re.compile(r"^(\d+)\[(\d+)\]\.([A-Za-z0-9_.-]+)$")
SCALAR_RE = re.compile(r"^(\d+)\.([A-Za-z0-9_.-]+)$")
AWK = r'''
function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t\r]+$/, "", s); return s }
function emit() { if (jid != "") printf "%s\t%s\t%s\n", jid, state, estatus }
/^Job Id:/ { emit(); jid=$0; sub(/^Job Id:[ \t]*/, "", jid); state=""; estatus=""; next }
/^qstat:/ { print "QUERY_ERROR\t" $0; next }
{
  value=$0
  if (value ~ /^[ \t]*job_state[ \t]*=/) {
    sub(/^[ \t]*job_state[ \t]*=[ \t]*/, "", value); state=trim(value)
  } else if (value ~ /^[ \t]*Exit_status[ \t]*=/) {
    sub(/^[ \t]*Exit_status[ \t]*=[ \t]*/, "", value); estatus=trim(value)
  }
}
END { emit() }
'''


def require_project_python():
    expected = remote.PROJECT / "nscc-access" / ".venv" / "Scripts" / "python.exe"
    if os.name == "nt" and os.path.normcase(str(Path(sys.executable).resolve())) != os.path.normcase(str(expected.resolve())):
        raise SystemExit(f"Run with the project control interpreter: {expected}")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temp = Path(stream.name)
    os.replace(temp, path)


def child_id_parts(job_id):
    match = CHILD_RE.fullmatch(job_id)
    return (match.group(1), int(match.group(2)), match.group(3)) if match else None


def range_child_source(index, ranges):
    item = ranges[index]
    return {
        "pbs_array_index": index,
        "start": item["start"],
        "stop": item["stop"],
        "expected_episode_count": ARMS * (item["stop"] - item["start"]),
    }


def read_compact_qstat(transport, array_jobid):
    parent = PARENT_RE.fullmatch(array_jobid) or SCALAR_RE.fullmatch(array_jobid)
    expected_base, expected_server = parent.group(1), parent.group(2)
    inner = (
        f"set -o pipefail; qstat -x -t -f {shlex.quote(array_jobid)} 2>&1 "
        f"| awk {shlex.quote(AWK)}"
    )
    rc, output = remote.command(transport, f"bash -c {shlex.quote(inner)}", timeout=60)
    children, parent_state, errors, unexpected, duplicate_indices = {}, None, [], [], []
    for line in output.splitlines():
        if line.startswith("QUERY_ERROR\t"):
            errors.append(line.partition("\t")[2])
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3:
            if line.strip():
                errors.append(f"unparsed qstat output: {line[:160]}")
            continue
        job_id, state, status = (part.strip() for part in parts)
        child = child_id_parts(job_id)
        if child:
            base, index, server = child
            if (base, server) != (expected_base, expected_server):
                unexpected.append(job_id)
                continue
            children.setdefault(index, []).append({
                "pbs_jobid": job_id,
                "state": state or None,
                "exit_status": int(status) if re.fullmatch(r"-?\d+", status) else None,
            })
        else:
            parent = PARENT_RE.fullmatch(job_id)
            scalar = SCALAR_RE.fullmatch(job_id)
            if parent or scalar:
                parent_state = state or None
            else:
                unexpected.append(job_id)
    for index, rows in children.items():
        if len(rows) > 1:
            duplicate_indices.append(index)
    return rc, children, parent_state, errors, unexpected, duplicate_indices


def main():
    require_project_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, help="confirmed attempt token used by submit_array.py")
    parser.add_argument("--handles", type=Path, help="local confirmed handle JSON (defaults to array_handles/<attempt>.json)")
    parser.add_argument("--output", type=Path, help="defaults to shard_sources_<attempt>.json beside this script")
    performance = parser.add_mutually_exclusive_group()
    performance.add_argument("--performance-jobid", help="PBS job id whose artifacts contain performance/storage.json")
    performance.add_argument("--performance-artifact-dir", help="explicit aggregate-visible performance artifact directory")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9]{1,13}", args.attempt):
        parser.error("--attempt must be 1-13 alphanumeric characters")

    handles_path = args.handles or (remote.HERE / "array_handles" / f"{args.attempt}.json")
    if not handles_path.is_file():
        raise SystemExit(f"No confirmed handle file: {handles_path}")
    handle = json.loads(handles_path.read_text(encoding="utf-8"))
    if handle.get("state") != "confirmed" or not handle.get("pbs_jobid"):
        raise SystemExit("Array handle is not confirmed; refusing to query or construct sources")
    range_plan = validate_range_plan(handle["range_plan"]) if "range_plan" in handle else None
    if range_plan is not None:
        if handle.get("shard_size") is not None or handle.get("range_plan_path") != RANGE_PLAN_REMOTE:
            raise SystemExit("Saved range handle has an unexpected shard size or remote plan path")
        array_first, array_last = handle.get("array_first"), handle.get("array_last")
        if (type(array_first) is not int or type(array_last) is not int
                or not 0 <= array_first <= array_last < len(range_plan["ranges"])):
            raise SystemExit("Saved range window is outside the complete range plan")
        shard_size = None
        expected_children = array_last - array_first + 1
    else:
        shard_size = int(handle["shard_size"])
        expected_children = math.ceil(TOTAL_VARIANTS / shard_size)
        array_first, array_last = 0, expected_children - 1
        if handle.get("array_first") != array_first or handle.get("array_last") != array_last:
            raise SystemExit("Saved array range does not match shard_size and the 1400-row manifest")
    expected_indices = range(array_first, array_last + 1)
    array_jobid = handle["pbs_jobid"]
    if not (PARENT_RE.fullmatch(array_jobid) or SCALAR_RE.fullmatch(array_jobid)):
        raise SystemExit(f"Invalid saved array handle: {array_jobid!r}")

    jump = transport = None
    try:
        jump, transport = remote.connect()
        rc, children, parent_state, errors, unexpected, duplicate_indices = read_compact_qstat(transport, array_jobid)
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()

    parent_match = PARENT_RE.fullmatch(array_jobid) or SCALAR_RE.fullmatch(array_jobid)
    base, server = parent_match.group(1), parent_match.group(2)
    unexpected.extend(f"out-of-window array index {index}" for index in children if index not in expected_indices)
    incomplete = []
    if rc != 0:
        incomplete.append(f"qstat exited {rc}")
    incomplete.extend(errors)
    if unexpected:
        incomplete.append(f"unexpected qstat job ids: {unexpected}")
    if duplicate_indices:
        incomplete.append(f"duplicate child records: {duplicate_indices}")

    sources = []
    terminal = 0
    missing_rows = active = missing_exit = 0
    for index in expected_indices:
        rows = children.get(index, [])
        row = rows[0] if len(rows) == 1 else None
        pbs_jobid = row["pbs_jobid"] if row else f"{base}[{index}].{server}"
        state = row["state"] if row else None
        exit_status = row["exit_status"] if row else None
        if row is None:
            missing_rows += 1
        elif state not in ARRAY_TERMINAL_STATES:
            active += 1
        else:
            terminal += 1
        if row is not None and state in ARRAY_TERMINAL_STATES and exit_status is None:
            missing_exit += 1
        if row is None:
            incomplete.append(f"array child {index}: qstat record missing")
        elif state not in ARRAY_TERMINAL_STATES:
            incomplete.append(f"array child {index}: state={state or 'unknown'}")
        elif exit_status is None:
            incomplete.append(f"array child {index}: terminal Exit_status missing")
        source = {
            "pbs_jobid": pbs_jobid,
            "exit_status": exit_status,
            "state": state,
            "artifact_dir": f"{remote.REMOTE}/artifacts/{pbs_jobid}",
        }
        if range_plan is None:
            source.update({"array_index": index, "shard_size": shard_size})
        else:
            source.update(range_child_source(index, range_plan["ranges"]))
        sources.append(source)

    performance_dir = args.performance_artifact_dir
    if args.performance_jobid:
        if not re.fullmatch(r"\d+\.pbs101", args.performance_jobid):
            parser.error("--performance-jobid must be a scalar PBS id such as 12345678.pbs101")
        performance_dir = f"{remote.REMOTE}/artifacts/{args.performance_jobid}"
    if not performance_dir:
        incomplete.append("performance_artifact_dir not supplied")

    scheduler_complete = (
        rc == 0 and len(sources) == expected_children and terminal == expected_children
        and missing_exit == 0 and missing_rows == 0 and not unexpected and not duplicate_indices and not errors
    )
    window_variants = sum(source["stop"] - source["start"] for source in sources) if range_plan is not None else TOTAL_VARIANTS
    document = {
        "attempt": args.attempt,
        "pbs_array_jobid": array_jobid,
        "array_job_state": parent_state,
        "expected_array_indices": [array_first, array_last],
        "expected_variants": TOTAL_VARIANTS,
        "expected_window_variants": window_variants,
        "expected_episode_slots": window_variants * ARMS,
        "scheduler_complete": scheduler_complete,
        "incomplete": incomplete,
        "shards": sources,
        "generated_at_utc": utc_now(),
    }
    if range_plan is None:
        document["shard_size"] = shard_size
    else:
        document["range_plan"] = range_plan
        document["range_plan_path"] = handle["range_plan_path"]
    if performance_dir:
        document["performance_artifact_dir"] = performance_dir
    output = args.output or (remote.HERE / f"shard_sources_{args.attempt}.json")
    atomic_json(output, document)

    partition_label = f"shard_size={shard_size}" if range_plan is None else f"ranges={array_first}-{array_last}"
    print(f"array={array_jobid} state={parent_state or 'unknown'} {partition_label} children={expected_children}")
    print(f"terminal={terminal}/{expected_children} missing_rows={missing_rows} active={active} terminal_missing_exit={missing_exit}")
    print(f"scheduler_complete={scheduler_complete}; sources={output}")
    for item in incomplete:
        print(f"INCOMPLETE: {item}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ARRAY_SOURCES_FAILED: {exc}", file=sys.stderr)
        raise
