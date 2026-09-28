"""Small, host-key-verified control-file staging and PBS control for this pilot."""
import argparse
import importlib.util
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = "/scratch/users/ntu/yguo017/lewm-block-local-pilot"
HELPER = ROOT.parent / "block-local-action-dynamics" / "cluster_control.py"
spec = importlib.util.spec_from_file_location("existing_aspire_control", HELPER)
access = importlib.util.module_from_spec(spec)
spec.loader.exec_module(access)

MANIFEST = ROOT / "STAGE.json"
TRACKER = ROOT / "JOB.json"
REJECTED_TRACKER = ROOT / "JOB_g1_rejected.json"
PBS_FILE = "run_pilot.pbs"
MAX_CONTROL_BYTES = 200_000


def names_from_manifest(key):
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = value.get(key, [])
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError(f"STAGE.json {key!r} must be a list of basenames")
    if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        raise ValueError(f"STAGE.json {key!r} contains duplicate or unsafe basenames")
    return names


def local_sources():
    names = names_from_manifest("files")
    paths = []
    for name in names:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or path.suffix not in {".py", ".json", ".md", ".pbs"}:
            raise ValueError(f"Refusing non-source control file: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    if sum(path.stat().st_size for path in paths) > MAX_CONTROL_BYTES:
        raise ValueError("Control-file staging exceeds 200 KB")
    if PBS_FILE not in {path.name for path in paths}:
        raise ValueError(f"{PBS_FILE} must be listed in STAGE.json")
    return paths


def write_tracker(value):
    tmp = TRACKER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(TRACKER)


def tracked_job():
    if not TRACKER.exists():
        return None
    return json.loads(TRACKER.read_text(encoding="utf-8"))


def valid_job(job):
    return bool(job) and all(c.isalnum() or c in "._-" for c in job)


def stage(transport):
    paths = local_sources()
    remote_source = REMOTE + "/source"
    access.command(transport, f"mkdir -p {shlex.quote(remote_source)} {shlex.quote(REMOTE + '/runs')}")
    sftp = access.paramiko.SFTPClient.from_transport(transport)
    try:
        targets = [(path, remote_source + "/" + path.name) for path in paths]
        for _, target in targets:
            for candidate in (target, target + ".uploading"):
                try:
                    sftp.stat(candidate)
                except FileNotFoundError:
                    continue
                raise FileExistsError(f"Remote stage target already exists: {target}")
        for path, target in targets:
            with sftp.open(target + ".uploading", "wb") as stream:
                stream.write(path.read_bytes().replace(b"\r\n", b"\n"))
            sftp.posix_rename(target + ".uploading", target)
    finally:
        sftp.close()
    print(f"Staged {len(paths)} small source/control files")


def submit(transport):
    previous = tracked_job()
    if previous and (previous.get("submission_state") != "rearmed" or previous.get("job_id")):
        raise RuntimeError("JOB.json exists without an explicit no-job recovery; inspect status")
    if not previous and TRACKER.exists():
        raise RuntimeError("Unreadable JOB.json; inspect it before any submission")
    # Record intent before the one allowed qsub attempt; an interrupted response stays locked.
    record = previous or {
        "job_id": None,
        "submission_state": "uncertain",
        "remote_root": REMOTE,
        "pbs_file": PBS_FILE,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    record.update(job_id=None, submission_state="uncertain", queue="normal",
                  started_utc=datetime.now(timezone.utc).isoformat())
    access.command(
        transport,
        "test -f " + shlex.quote(REMOTE + "/source/" + PBS_FILE)
        + " && test -f " + shlex.quote(REMOTE + "/source/runner.py")
        + " && test -f " + shlex.quote(REMOTE + "/source/FREEZE.json"),
    )
    write_tracker(record)
    try:
        # Atomic remote lock prevents a second checkout from issuing another qsub.
        access.command(transport, "mkdir " + shlex.quote(REMOTE + "/submit.once"))
        output = access.command(
            transport,
            "cd " + shlex.quote(REMOTE + "/source") + " && qsub " + shlex.quote(PBS_FILE),
        )
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        job = lines[-1] if lines else ""
        if not valid_job(job):
            raise RuntimeError("Uncertain qsub response; inspect PBS status before any further action")
        record.update(job_id=job, submission_state="submitted")
        write_tracker(record)
        print(f"Recorded PBS job {job}")
    except Exception:
        write_tracker(record)
        raise


def recover_g1_denied(transport):
    record = tracked_job()
    if not record or record.get("submission_state") != "uncertain" or record.get("job_id"):
        raise RuntimeError("Recovery requires the unresolved g1 attempt with no job id")
    if REJECTED_TRACKER.exists():
        raise RuntimeError("JOB_g1_rejected.json already exists; recovery is one-time only")

    # Confirm the rejected job was not accepted before preserving/rearming its lock.
    jobs = access.command(transport, "qstat -u yguo017")
    if any("lewm_block_pilot" in line.split() for line in jobs.splitlines()):
        raise RuntimeError("A lewm_block_pilot job is visible; do not rearm or resubmit")

    source = REMOTE + "/source"
    old_source = REMOTE + "/source_g1_rejected"
    lock = REMOTE + "/submit.once"
    old_lock = REMOTE + "/submit_g1_rejected.once"
    access.command(
        transport,
        "test -d " + shlex.quote(source)
        + " && test -d " + shlex.quote(lock)
        + " && test ! -e " + shlex.quote(old_source)
        + " && test ! -e " + shlex.quote(old_lock),
    )

    record.update(
        submission_state="rejected",
        queue="g1",
        rejection_reason="qsub returned Access to queue is denied; qstat showed no lewm_block_pilot job",
    )
    write_tracker(record)
    TRACKER.replace(REJECTED_TRACKER)
    access.command(
        transport,
        "mv " + shlex.quote(source) + " " + shlex.quote(old_source)
        + " && mv " + shlex.quote(lock) + " " + shlex.quote(old_lock),
    )
    write_tracker({
        "job_id": None,
        "submission_state": "rearmed",
        "queue": "normal",
        "attempt": 2,
        "remote_root": REMOTE,
        "pbs_file": PBS_FILE,
        "previous_attempt": REJECTED_TRACKER.name,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    })
    print("Confirmed no accepted pilot job; preserved the g1 attempt and rearmed one normal-route submission")


def job_for(args):
    job = args.job
    if not job:
        record = tracked_job()
        job = record.get("job_id") if record else None
    if job and not valid_job(job):
        raise ValueError("Invalid PBS job id")
    return job


def status(transport, job):
    if job:
        access.command(transport, "qstat -xf " + shlex.quote(job), compact=True)
        return
    record = tracked_job()
    if record:
        print("Submission has no confirmed job id; checking the user's current PBS jobs and remote lock.")
    access.command(transport, "qstat -u yguo017")
    access.command(
        transport,
        "if test -d " + shlex.quote(REMOTE + "/submit.once")
        + "; then echo PILOT_SUBMIT_LOCK_PRESENT; else echo PILOT_SUBMIT_LOCK_ABSENT; fi",
    )


def progress(transport, job):
    if not job:
        status(transport, None)
        return
    output = REMOTE + "/runs/" + job
    sftp = access.paramiko.SFTPClient.from_transport(transport)
    try:
        for name in ("progress.json", "summary_progress.json", "FAILURE.json", "DONE.json", "job_status.txt"):
            path = output + "/" + name
            try:
                size = sftp.stat(path).st_size
            except FileNotFoundError:
                continue
            if size > MAX_CONTROL_BYTES:
                print(f"{name}: exceeds the 200 KB summary limit")
                continue
            with sftp.open(path, "rb") as stream:
                print(f"--- {name}\n" + stream.read().decode("utf-8", errors="replace"))
        selection = output + "/stage_b/selection.json"
        try:
            size = sftp.stat(selection).st_size
        except FileNotFoundError:
            pass
        else:
            if size <= MAX_CONTROL_BYTES:
                with sftp.open(selection, "rb") as stream:
                    value = json.loads(stream.read().decode("utf-8"))
                print(f"STAGE_B_TRIGGERED selected_episodes={len(value.get('episodes', []))}")
            else:
                print("STAGE_B_TRIGGERED selection.json exceeds the summary limit")
        cases = output + "/stage_b/cases.jsonl"
        try:
            case_bytes = sftp.stat(cases).st_size
        except FileNotFoundError:
            pass
        else:
            print(f"STAGE_B_CASE_LOG_BYTES={case_bytes}")
    finally:
        sftp.close()
    log = shlex.quote(output + "/job.log")
    access.command(transport, f"if test -f {log}; then tail -n 16 {log}; fi")


def fetch(transport, job):
    if not job:
        raise RuntimeError("No confirmed job id; inspect status before fetching")
    names = names_from_manifest("results")
    if not names:
        raise ValueError("STAGE.json results list is empty")
    destination = ROOT / "results" / job
    destination.mkdir(parents=True, exist_ok=True)
    fetched = []
    skipped = []
    missing = []
    sftp = access.paramiko.SFTPClient.from_transport(transport)
    try:
        for name in names:
            path = REMOTE + "/runs/" + job + "/" + name
            try:
                size = sftp.stat(path).st_size
            except FileNotFoundError:
                missing.append(name)
                continue
            if size > MAX_CONTROL_BYTES:
                skipped.append(name)
                continue
            sftp.get(path, str(destination / name))
            fetched.append(name)
    finally:
        sftp.close()
    print(f"Fetched {len(fetched)} bounded summary files to {destination}")
    if skipped:
        print("Skipped files larger than 200 KB: " + ", ".join(skipped))
    if missing:
        print("Not present at fetch time: " + ", ".join(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "recover-g1-denied", "submit", "status", "progress", "fetch"))
    parser.add_argument("--job")
    args = parser.parse_args()
    jump, transport = access.connect()
    try:
        if args.action == "stage":
            stage(transport)
        elif args.action == "recover-g1-denied":
            recover_g1_denied(transport)
        elif args.action == "submit":
            submit(transport)
        elif args.action == "status":
            status(transport, job_for(args))
        elif args.action == "progress":
            progress(transport, job_for(args))
        else:
            fetch(transport, job_for(args))
    finally:
        transport.close()
        if jump is not None:
            jump.close()


if __name__ == "__main__":
    main()
