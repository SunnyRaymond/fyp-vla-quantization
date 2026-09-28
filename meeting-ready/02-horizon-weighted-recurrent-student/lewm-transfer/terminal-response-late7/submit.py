#!/usr/bin/env python3
"""Upload the small frozen controls and submit the guarded late7 PBS job."""

from __future__ import annotations

import importlib.util
import json
import re
import stat
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
ACCESS = HERE.parents[3] / "nscc-access" / "aspire2a_shell.py"
REMOTE = "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/terminal-response-late7"
TRANSFER = "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer"
BASELINE = f"{TRANSFER}/official-lewm-dataset-teacher-baseline/artifacts/25534994.pbs101"
PREVIOUS = f"{TRANSFER}/terminal-response-loss/artifacts/25549480.pbs101"
PREVIOUS_FREEZE = f"{TRANSFER}/terminal-response-loss/FREEZE.json"
CANDIDATE_CHECKPOINT = f"{PREVIOUS}/terminal_response_step3000.pt"
STAGE_ROOT = "/scratch/users/ntu/yguo017/lewm-pusht-iteration"
MAX_CONTROL_FILE_BYTES = 200_000
MAX_SMALL_SOURCE_BYTES = 2_000_000
FILES = (
    "FREEZE.json",
    "PROTOCOL.zh.md",
    "runner.py",
    "run_terminal_response_late7.pbs",
)
LOCAL_JOB_ID = HERE / "results" / "submitted_job_id.txt"


def load_connector():
    spec = importlib.util.spec_from_file_location("aspire2a_shell", ACCESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("verified ASPIRE2A connector is unavailable")
    connector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connector)
    return connector


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def validate_local_controls() -> dict[str, int]:
    sizes = {}
    for name in FILES:
        source = HERE / name
        if not source.is_file():
            raise FileNotFoundError(f"required control file is missing: {name}")
        size = source.stat().st_size
        if size <= 0 or size > MAX_CONTROL_FILE_BYTES:
            raise RuntimeError(f"refusing invalid or oversized control file: {name} ({size} bytes)")
        sizes[name] = size

    freeze = read_json(HERE / "FREEZE.json")
    if freeze.get("candidate_checkpoint_path") != CANDIDATE_CHECKPOINT:
        raise RuntimeError("frozen candidate checkpoint path differs from the terminal-response job")
    provenance = freeze.get("candidate_checkpoint_provenance")
    if not isinstance(provenance, dict) or (
        provenance.get("job_id") != "25549480.pbs101"
        or provenance.get("filename") != "terminal_response_step3000.pt"
        or provenance.get("source_arm") != "terminal_response"
        or provenance.get("updates") != 3000
    ):
        raise RuntimeError("candidate checkpoint provenance differs from job 25549480.pbs101")
    task_set = freeze.get("task_set")
    tasks = task_set.get("tasks") if isinstance(task_set, dict) else None
    if not isinstance(tasks, list) or len(tasks) != 50:
        raise RuntimeError("freeze must contain the exact 50 baseline task rows")
    normalized = [tuple(int(row[key]) for key in ("row_index", "episode_idx", "start_step")) for row in tasks]
    if (
        len(set(normalized)) != 50
        or len({row[0] for row in normalized}) != 50
        or len({row[1] for row in normalized}) != 50
    ):
        raise RuntimeError("frozen task identities and source episodes must be unique")
    if (
        task_set.get("source_job_id") != "25534994.pbs101"
        or task_set.get("source_path") != f"{BASELINE}/selected_tasks.json"
        or task_set.get("selection_rng_seed") != 42
        or task_set.get("task_count") != 50
    ):
        raise RuntimeError("frozen task source differs from the validated 25534994.pbs101 task set")
    if set(freeze.get("arms", [])) != {"student_only", "late_teacher7", "teacher_only"}:
        raise RuntimeError("frozen arm set differs from the three-arm protocol")
    run_order = freeze.get("run_order", {})
    if not isinstance(run_order, dict) or (
        run_order.get("seed") != 20260925
        or run_order.get("frozen_order") != ["late_teacher7", "student_only", "teacher_only"]
    ):
        raise RuntimeError("frozen execution order differs from the reviewed protocol")
    schedule = freeze.get("schedule_rounds", {})
    if not isinstance(schedule, dict) or schedule.get("late_teacher7") != [24, 25, 26, 27, 28, 29, 30]:
        raise RuntimeError("frozen late_teacher7 schedule differs from rounds 24-30")
    validity = freeze.get("validity")
    successes = validity.get("teacher_only_expected_successes") if isinstance(validity, dict) else None
    if not isinstance(successes, list) or len(successes) != 50 or any(not isinstance(value, bool) for value in successes):
        raise RuntimeError("freeze must bind the expected teacher-only 50-task vector")
    if sum(bool(value) for value in successes) != 49 or validity.get("teacher_only_expected_success_count") != 49:
        raise RuntimeError("frozen teacher-only reference vector differs from the validated 49/50 baseline")
    return sizes


def check_remote_sources(sftp) -> None:
    paths = (
        f"{BASELINE}/summary.json",
        f"{BASELINE}/selected_tasks.json",
        f"{BASELINE}/job_status",
        f"{BASELINE}/final_exit_status.txt",
        f"{PREVIOUS}/terminal_response_loss_summary.json",
        f"{PREVIOUS}/selection.json",
        CANDIDATE_CHECKPOINT,
        PREVIOUS_FREEZE,
        f"{TRANSFER}/interface-probe/artifacts/24554356.pbs101/interface_probe.json",
        f"{STAGE_ROOT}/stablewm_home/pusht_expert_train.h5",
        f"{STAGE_ROOT}/stablewm_home/pusht/lewm_object.ckpt",
    )
    for path in paths:
        attrs = sftp.stat(path)
        if not stat.S_ISREG(attrs.st_mode) or attrs.st_size <= 0:
            raise RuntimeError(f"required remote source is missing or empty: {PurePosixPath(path).name}")
        if path.startswith((BASELINE, PREVIOUS)) and attrs.st_size > MAX_SMALL_SOURCE_BYTES and path != CANDIDATE_CHECKPOINT:
            raise RuntimeError(f"unexpectedly large control source: {PurePosixPath(path).name}")
        print(f"source present ({attrs.st_size} bytes): {PurePosixPath(path).name}")


def main() -> None:
    sizes = validate_local_controls()
    connector = load_connector()
    jump, nscc = connector.connect()
    try:
        with nscc.open_sftp_client() as sftp:
            sftp.stat(REMOTE.rsplit("/", 1)[0])
            check_remote_sources(sftp)
            try:
                sftp.stat(REMOTE)
            except OSError as exc:
                if getattr(exc, "errno", None) != 2:
                    raise
                sftp.mkdir(REMOTE)
            else:
                raise RuntimeError(f"refusing to overwrite existing remote experiment directory: {REMOTE}")

            for name in FILES:
                source = HERE / name
                remote_file = f"{REMOTE}/{name}"
                sftp.put(str(source), remote_file)
                if sftp.stat(remote_file).st_size != sizes[name]:
                    raise RuntimeError(f"uploaded size mismatch: {name}")
                print(f"uploaded {name} ({sizes[name]} bytes)")

        channel = nscc.open_session(timeout=20)
        try:
            channel.exec_command(f"qsub {REMOTE}/run_terminal_response_late7.pbs")
            stdout = channel.makefile("rb").read().decode("utf-8", errors="replace").strip()
            stderr = channel.makefile_stderr("rb").read().decode("utf-8", errors="replace").strip()
            status = channel.recv_exit_status()
            if status != 0:
                raise RuntimeError(f"qsub failed (exit {status}): {stderr or stdout}")
            match = re.search(r"(?m)^(\d+(?:\.[A-Za-z0-9._-]+)?)\s*$", stdout)
            if match is None:
                raise RuntimeError(f"qsub succeeded but returned an unrecognized job id: {stdout}")
            job_id = match.group(1)
            LOCAL_JOB_ID.parent.mkdir(parents=True, exist_ok=True)
            LOCAL_JOB_ID.write_text(job_id + "\n", encoding="utf-8")
            print(f"submitted {job_id}")
        finally:
            channel.close()
    finally:
        nscc.close()
        jump.close()


if __name__ == "__main__":
    main()
