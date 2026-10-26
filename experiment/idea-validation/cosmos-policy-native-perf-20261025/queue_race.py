"""Coordinate the short-shard versus wide-job PBS race without loading Torch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import tempfile
import time
import uuid

from prepare_sources import guard


SHORT_STAGES = ("gate_spin", "gate_qua", "gate_svd", "gate_vidit",
                "ours_gate", "perf_bf16")
WIDE_STAGES = ("gate_wide", "gate_wide_g1", "gate_wide_g2", "gate_wide_g3")
WIDE_STAGE = WIDE_STAGES[0]
OWNER_WAIT_SECONDS = 10.0
OWNER_MAX_BYTES = 4096


def _decision(status: str, exit_code: int, **fields) -> dict:
    return {"status": status, "exit_code": exit_code, **fields}


def _race_id(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("queue_race.json race_id must be a safe basename")
    return value


def _paths(root: Path, race_id: str) -> tuple[Path, Path]:
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    control_real = control.resolve()
    if control_real.parent != root:
        raise RuntimeError("Race control directory must remain inside the requested root")
    races = control_real / "races"
    races.mkdir(exist_ok=True)
    races_real = races.resolve()
    if races_real.parent != control_real:
        raise RuntimeError("Race directory must remain directly inside root/control/races")
    race = races_real / race_id
    race.mkdir(exist_ok=True)
    race_real = race.resolve()
    if race_real.parent != races_real:
        raise RuntimeError("Race ID resolved outside root/control/races")
    claim = race_real / "claim"
    return race_real, claim


def _validated_owner(path: Path, race_id: str) -> dict:
    if path.stat().st_size > OWNER_MAX_BYTES:
        raise RuntimeError("Race owner.json is unexpectedly large")
    try:
        owner = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Race owner.json is damaged: {exc}") from exc
    if not isinstance(owner, dict):
        raise RuntimeError("Race owner.json must contain an object")
    if owner.get("race_id") != race_id:
        raise RuntimeError("Race owner.json names a different race")
    branch = owner.get("branch")
    if branch not in ("short", "wide"):
        raise RuntimeError(f"Race owner.json has an unknown branch: {branch!r}")
    stage = owner.get("stage")
    expected_branch = ("short" if stage in SHORT_STAGES
                       else "wide" if stage in WIDE_STAGES else None)
    if expected_branch != branch:
        raise RuntimeError("Race owner.json stage and branch do not match")
    for field in ("pbs_job_id", "hostname", "created_utc"):
        if not isinstance(owner.get(field), str) or not owner[field].strip():
            raise RuntimeError(f"Race owner.json lacks a valid {field}")
    return owner


def _await_owner(path: Path, race_id: str) -> dict:
    deadline = time.monotonic() + OWNER_WAIT_SECONDS
    while True:
        if path.is_symlink():
            raise RuntimeError("Race owner.json cannot be a symlink")
        if path.exists():
            if not path.is_file():
                raise RuntimeError("Race owner.json exists but is not a file")
            return _validated_owner(path, race_id)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Race claim exists but owner.json did not appear; lock was retained")
        time.sleep(min(0.1, remaining))


def _write_owner(path: Path, owner: dict) -> None:
    temporary = path.with_name(f".owner.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(owner, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(root: Path, stage: str) -> dict:
    root = Path(root).expanduser().resolve()
    config_path = root / "queue_race.json"
    if not config_path.is_file():
        return _decision("skipped", 0, reason="race configuration is absent")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read queue race configuration: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError("queue_race.json must contain an object")
    if config.get("enabled") is not True:
        return _decision("skipped", 0, reason="race is not enabled")
    scope_stages = config.get("scope_stages")
    wide_stage = config.get("wide_stage")
    configured_wide_stages = config.get("wide_stages")
    if (not isinstance(scope_stages, list)
            or any(not isinstance(item, str) or item not in SHORT_STAGES for item in scope_stages)
            or len(scope_stages) != len(set(scope_stages))
            or set(scope_stages) != set(SHORT_STAGES)):
        raise RuntimeError("queue_race.json scope_stages must list the six short stages")
    if configured_wide_stages is None:
        # Legacy config chooses a single wide worker; newer config can race all four variants.
        effective_wide_stages = (WIDE_STAGE if wide_stage is None else wide_stage,)
    else:
        if (not isinstance(configured_wide_stages, list)
                or not configured_wide_stages
                or any(not isinstance(item, str) or item not in WIDE_STAGES
                       for item in configured_wide_stages)
                or len(configured_wide_stages) != len(set(configured_wide_stages))):
            raise RuntimeError("queue_race.json wide_stages must be a non-empty list of fixed wide stages")
        if wide_stage is not None and (wide_stage not in WIDE_STAGES
                                       or wide_stage not in configured_wide_stages):
            raise RuntimeError("queue_race.json wide_stage must be present in wide_stages")
        effective_wide_stages = tuple(configured_wide_stages)
    if any(item not in WIDE_STAGES for item in effective_wide_stages):
        raise RuntimeError("queue_race.json wide_stage is unsupported")
    if set(scope_stages).intersection(effective_wide_stages):
        raise RuntimeError("queue_race.json short and wide stage sets must not overlap")
    if stage.startswith("gate_wide") and stage not in WIDE_STAGES:
        raise ValueError(f"Unknown wide stage: {stage}")
    if stage not in scope_stages and stage not in effective_wide_stages:
        return _decision("skipped", 0, reason="stage is outside the configured race")

    race_id = _race_id(config.get("race_id"))
    branch = "wide" if stage in WIDE_STAGES else "short"
    # Guard immediately before touching the shared lock directory.
    guard()
    job_id = os.environ.get("PBS_JOBID")
    if not isinstance(job_id, str) or not job_id.strip():
        raise RuntimeError("PBS_JOBID is required after the allocation guard")

    race_dir, claim_dir = _paths(root, race_id)
    try:
        claim_dir.mkdir(exist_ok=False)
    except FileExistsError:
        if claim_dir.is_symlink() or not claim_dir.is_dir() or claim_dir.resolve() != claim_dir:
            raise RuntimeError("Existing race claim is not a directory at the expected path")
        owner = _await_owner(claim_dir / "owner.json", race_id)
        if owner["branch"] != branch:
            return _decision("branch_conflict", 75, race_id=race_id,
                             branch=branch, owner=owner)
        if branch == "wide" and owner["pbs_job_id"] != job_id:
            return _decision("wide_job_conflict", 75, race_id=race_id,
                             branch=branch, owner=owner)
        return _decision("same_branch", 0, race_id=race_id,
                         branch=branch, owner=owner)

    if claim_dir.resolve() != claim_dir or claim_dir.parent != race_dir:
        raise RuntimeError("New race claim escaped its race directory")
    owner = {"race_id": race_id, "branch": branch, "stage": stage,
             "pbs_job_id": job_id, "hostname": socket.gethostname(),
             "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_owner(claim_dir / "owner.json", owner)
    return _decision("claimed", 0, race_id=race_id, branch=branch, owner=owner)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.root, args.stage)
    except Exception as exc:
        result = _decision("error", 2, error_type=type(exc).__name__,
                           error=str(exc)[:1000])
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
