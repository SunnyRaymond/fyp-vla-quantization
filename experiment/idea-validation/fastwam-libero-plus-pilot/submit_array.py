#!/usr/bin/env python3
"""Submit one bounded Fast-WAM PBS array, or recover its confirmed handle.

Run locally with nscc-access/.venv/Scripts/python.exe and a unique --attempt.
The script stores a durable intent before qsub and never resubmits an unknown
outcome under that attempt name.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

import remote

TOTAL_VARIANTS = 1400
MAX_CONCURRENCY = 10
PROTOCOL = "fastwam-optional-idm-plus-pilot-v1"
CELL_SIZE = 50
RANGE_PLAN_REMOTE = f"{remote.REMOTE}/shard_ranges.json"
PBS_ARRAY_PARENT = re.compile(r"^(\d+)\[\]\.([A-Za-z0-9_.-]+)$")
PBS_ARRAY_CHILD = re.compile(r"^(\d+)\[\d+\]\.([A-Za-z0-9_.-]+)$")


def require_project_python():
    expected = remote.PROJECT / "nscc-access" / ".venv" / "Scripts" / "python.exe"
    if os.name == "nt" and os.path.normcase(str(Path(sys.executable).resolve())) != os.path.normcase(str(expected.resolve())):
        raise SystemExit(f"Run with the project control interpreter: {expected}")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def durable_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temp_path = Path(stream.name)
    os.replace(temp_path, path)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def qselect_ids(transport, job_name):
    cmd = f"qselect -x -u yguo017 -N {shlex.quote(job_name)} 2>&1"
    rc, output = remote.command(transport, cmd, timeout=35)
    if rc not in (0, 1) or (rc == 1 and output.strip()):
        raise RuntimeError(f"qselect failed (exit {rc}); refusing to submit")
    ids = [line.strip() for line in output.splitlines() if line.strip()]
    for job_id in ids:
        if not (PBS_ARRAY_PARENT.fullmatch(job_id) or PBS_ARRAY_CHILD.fullmatch(job_id)):
            raise RuntimeError(f"Unexpected qselect output {job_id!r}; refusing to submit")
    return ids


def recover_parent(ids):
    if not ids:
        return None
    parsed = []
    for job_id in ids:
        match = PBS_ARRAY_PARENT.fullmatch(job_id) or PBS_ARRAY_CHILD.fullmatch(job_id)
        parsed.append((match.group(1), match.group(2)))
    bases = set(parsed)
    if len(bases) != 1:
        raise RuntimeError(f"Ambiguous matching PBS jobs: {ids}")
    base, server = next(iter(bases))
    return f"{base}[].{server}"


def validate_range_plan(plan):
    if not isinstance(plan, dict) or not {"protocol", "variant_count", "ranges"}.issubset(plan):
        raise ValueError("range plan must contain protocol, variant_count, and ranges")
    if plan["protocol"] != PROTOCOL or type(plan["variant_count"]) is not int or plan["variant_count"] != TOTAL_VARIANTS:
        raise ValueError("range plan protocol/variant_count does not match the frozen cohort")
    ranges = plan["ranges"]
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("range plan ranges must be a nonempty list")
    expected_start = 0
    cell_dimensions = {}
    validated = []
    for item in ranges:
        if not isinstance(item, dict) or set(item) != {"start", "stop", "dimension"}:
            raise ValueError("each range must contain exactly start, stop, and dimension")
        start, stop, dimension = item["start"], item["stop"], item["dimension"]
        if type(start) is not int or type(stop) is not int or start != expected_start or not start < stop <= TOTAL_VARIANTS:
            raise ValueError(f"range [{start!r},{stop!r}) is not a contiguous nonempty cohort interval")
        if not isinstance(dimension, str) or not dimension:
            raise ValueError("range dimension must be a nonempty string")
        if (stop - 1) // CELL_SIZE != start // CELL_SIZE:
            raise ValueError(f"range [{start},{stop}) crosses a 50-variant cell boundary")
        cell = start // CELL_SIZE
        if cell in cell_dimensions and cell_dimensions[cell] != dimension:
            raise ValueError(f"dimension changes within cell {cell}")
        cell_dimensions[cell] = dimension
        limit = 1 if dimension == "sensor_noise" else 4
        if stop - start > limit:
            raise ValueError(f"range [{start},{stop}) exceeds the {dimension!r} child size limit {limit}")
        validated.append({"start": start, "stop": stop, "dimension": dimension})
        expected_start = stop
    if expected_start != TOTAL_VARIANTS:
        raise ValueError(f"range plan ends at {expected_start}; expected {TOTAL_VARIANTS}")
    return {"protocol": PROTOCOL, "variant_count": TOTAL_VARIANTS, "ranges": validated}


def load_range_plan(path):
    return validate_range_plan(json.loads(Path(path).read_text(encoding="utf-8")))


def range_window(range_plan, first=0, count=None):
    total = len(range_plan["ranges"])
    if type(first) is not int or not 0 <= first < total:
        raise ValueError(f"--first must be in [0, {total - 1}]")
    if count is not None and (type(count) is not int or count <= 0):
        raise ValueError("--count must be a positive integer")
    last = total - 1 if count is None else first + count - 1
    if last >= total:
        raise ValueError(f"range window ends at {last}; last plan index is {total - 1}")
    return first, last


def expected_config(attempt, shard_size, concurrency, range_plan=None, first=0, count=None):
    if type(first) is not int:
        raise ValueError("--first must be an integer")
    config = {
        "attempt": attempt,
        "job_name": f"fw{attempt}",
        "concurrency": concurrency,
    }
    if range_plan is None:
        if first != 0 or count is not None:
            raise ValueError("--first/--count windows require --ranges")
        config.update({"shard_size": shard_size,
                       "array_first": 0,
                       "array_last": (TOTAL_VARIANTS + shard_size - 1) // shard_size - 1})
    else:
        array_first, array_last = range_window(range_plan, first, count)
        config.update({"shard_size": None, "range_plan": range_plan,
                       "range_plan_path": RANGE_PLAN_REMOTE,
                       "array_first": array_first, "array_last": array_last})
    return config


def main():
    require_project_python()
    parser = argparse.ArgumentParser(description=__doc__)
    partition = parser.add_mutually_exclusive_group(required=True)
    partition.add_argument("--shard-size", type=int,
                           help="manifest variants assigned to each array child (1..1400)")
    partition.add_argument("--ranges", type=Path,
                           help="validated cell-aligned shard_ranges.json plan")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="PBS array concurrency cap (default 10; maximum 10)")
    parser.add_argument("--first", type=int, default=0,
                        help="first global range-plan child index (default 0)")
    parser.add_argument("--count", type=int,
                        help="number of consecutive range-plan children to submit")
    parser.add_argument("--attempt", required=True,
                        help="unique alphanumeric attempt token, 1-13 characters")
    args = parser.parse_args()
    if args.shard_size is not None and not 1 <= args.shard_size <= TOTAL_VARIANTS:
        parser.error("--shard-size must be in [1, 1400]")
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        parser.error("--concurrency must be in [1, 10]")
    if not re.fullmatch(r"[A-Za-z0-9]{1,13}", args.attempt):
        parser.error("--attempt must be 1-13 alphanumeric characters")

    try:
        range_plan = load_range_plan(args.ranges) if args.ranges else None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"invalid --ranges plan: {exc}")
    try:
        config = expected_config(args.attempt, args.shard_size, args.concurrency,
                                 range_plan, args.first, args.count)
    except ValueError as exc:
        parser.error(str(exc))
    state_path = remote.HERE / "array_handles" / f"{args.attempt}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
    if state and any(state.get(key) != value for key, value in config.items()):
        raise SystemExit(f"Attempt {args.attempt} already exists with different settings; choose a new attempt token")
    if state and state.get("state") == "confirmed":
        print(f"CONFIRMED_ARRAY_HANDLE {state['pbs_jobid']}")
        print(f"Saved: {state_path}")
        return

    jump = transport = None
    try:
        jump, transport = remote.connect()
        if state:
            # A prior controller run may have stopped after qsub. Resolve that exact
            # unique PBS name once; an empty lookup remains unknown and is never retried.
            matches = qselect_ids(transport, config["job_name"])
            recovered = recover_parent(matches)
            if recovered is None:
                state["state"] = "unknown"
                state["last_checked_utc"] = utc_now()
                durable_json(state_path, state)
                raise RuntimeError("Submission outcome remains unknown; no matching job is visible. Do not resubmit this attempt.")
            state.update({"state": "confirmed", "pbs_jobid": recovered,
                          "confirmed_utc": utc_now(), "recovered": True})
            durable_json(state_path, state)
            print(f"RECOVERED_CONFIRMED_ARRAY_HANDLE {recovered}")
            print(f"Saved: {state_path}")
            return

        # Guard against a reused attempt name before recording intent.
        matches = qselect_ids(transport, config["job_name"])
        if matches:
            raise RuntimeError(f"Matching PBS job exists without a local intent record: {matches}; inspect it and choose a fresh attempt token")

        run_state_path = remote.HERE / "RUN_STATE.json"
        if (run_state_path.is_file() and json.loads(run_state_path.read_text(encoding="utf-8"))
                .get("new_shard_submission_allowed") is False):
            raise RuntimeError("New evaluation shards are paused by the user; do not submit")

        # Persist intent before qsub. If the process dies during qsub, a rerun can
        # only query the same name and recover; it can never blindly submit again.
        state = {**config, "state": "submitting", "intent_utc": utc_now(),
                 "root": remote.REMOTE, "script": f"{remote.REMOTE}/run.pbs"}
        durable_json(state_path, state)
        array_last = config["array_last"]
        array_first = config["array_first"]
        array_spec = f"{array_first}-{array_last}%{args.concurrency}"
        variables = (f"FW_MODE=shard,FW_RANGE_PLAN={config['range_plan_path']}"
                     if range_plan is not None else f"FW_MODE=shard,FW_SHARD_SIZE={args.shard_size}")
        cmd = (
            f"qsub -q normal -J {array_spec} -N {config['job_name']} "
            f"-v {variables} "
            f"{shlex.quote(remote.REMOTE + '/run.pbs')}"
        )
        partition_label = f"ranges={len(range_plan['ranges'])}" if range_plan is not None else f"shard_size={args.shard_size}"
        print(f"SUBMITTING_ARRAY range={array_first}-{array_last} cap={args.concurrency} {partition_label}", flush=True)
        try:
            rc, output = remote.command(transport, cmd, timeout=45)
        except Exception as exc:
            rc, output = None, ""
            print(f"QSUB_OUTCOME_UNKNOWN {type(exc).__name__}; checking the same attempt name once", flush=True)
        if rc == 0:
            returned = [line.strip() for line in output.splitlines() if line.strip()]
            if len(returned) == 1 and PBS_ARRAY_PARENT.fullmatch(returned[0]):
                parent_id = returned[0]
            else:
                parent_id = None
        else:
            parent_id = None

        if parent_id is None:
            try:
                matches = qselect_ids(transport, config["job_name"])
                parent_id = recover_parent(matches)
            except Exception as exc:
                state.update({"state": "unknown", "last_error": f"{type(exc).__name__}: {exc}"})
                durable_json(state_path, state)
                raise RuntimeError("qsub was not confirmed and PBS history could not resolve it; do not resubmit this attempt") from exc
            if parent_id is None:
                state.update({"state": "unknown", "last_error": f"qsub exit={rc}, output={output.strip()!r}"})
                durable_json(state_path, state)
                raise RuntimeError("No confirmed handle is visible; outcome is unknown and this attempt must not be resubmitted")
            recovered = True
        else:
            recovered = False

        state.update({"state": "confirmed", "pbs_jobid": parent_id,
                      "confirmed_utc": utc_now(), "recovered": recovered})
        durable_json(state_path, state)
        print(f"CONFIRMED_ARRAY_HANDLE {parent_id}")
        print(f"array_indices={array_first}-{array_last}%{args.concurrency}; shards={array_last - array_first + 1}; expected_variants={TOTAL_VARIANTS}")
        print(f"Saved: {state_path}")
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ARRAY_CONTROL_FAILED: {exc}", file=sys.stderr)
        raise
