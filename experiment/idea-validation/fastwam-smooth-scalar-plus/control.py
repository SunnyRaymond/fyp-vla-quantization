"""Bounded pinned-host control and durable submission for the extra Scalar arm."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rotation_control", HERE.parent / "fastwam-rotation-baselines" / "control.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
ROOT = "/scratch/users/ntu/yguo017/fastwam-smooth-scalar-plus-20261009"
JOB_NAME = "fwscalar1009"
FILES = ("control.py", "run.pbs", "protocol.json", "scalar_native.py", "evaluate.py", "aggregate.py")


def submit():
    path = HERE / "submit_handle.json"
    if path.exists():
        intent = json.loads(path.read_text())
        if intent.get("job_id"):
            return intent
        raise RuntimeError("Unresolved durable submission intent; inspect PBS history before retry")
    rc, out = previous.direct_native_command(None, "qselect -x -u yguo017 -N " + JOB_NAME, timeout=35)
    if rc not in (0, 1) or (rc and out.strip()):
        raise RuntimeError("PBS history query failed; no submission attempted")
    if out.strip():
        raise RuntimeError("Matching PBS history already exists; reconcile it without submitting again: " + out)
    rc, out = previous.direct_native_command(None, "mkdir -p " + shlex.quote(ROOT + "/results") + " " + shlex.quote(ROOT + "/artifacts"), timeout=30)
    if rc:
        raise RuntimeError(out)
    _, transport = previous.remote.direct_connect()
    try:
        with previous.remote.sftp_client(transport) as sftp:
            for name in FILES:
                data = (HERE / name).read_bytes()
                if len(data) > previous.remote.LIMIT:
                    raise RuntimeError("Not a small control file: " + name)
                destination = ROOT + "/" + name
                with sftp.file(destination + ".tmp", "wb") as stream:
                    stream.write(data)
                sftp.posix_rename(destination + ".tmp", destination)
    finally:
        transport.close()
    rc, out = previous.direct_native_command(None, "bash -n " + shlex.quote(ROOT + "/run.pbs"), timeout=30)
    if rc:
        raise RuntimeError("PBS script syntax failed: " + out)
    intent = {"job_name": JOB_NAME, "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "intent_before_qsub", "remote_root": ROOT, "gpus": 2, "walltime": "12:00:00",
              "episodes": 280, "workers": 2, "variants_per_worker": 140}
    previous.save(path, intent)
    rc, out = previous.direct_native_command(None, "qsub -N " + JOB_NAME + " -o " + shlex.quote(ROOT + "/artifacts/pbs.out") + " " + shlex.quote(ROOT + "/run.pbs"), timeout=35)
    if rc or not previous.HANDLE.fullmatch(out.strip()):
        intent.update(status="unresolved", query_rc=rc, output=out)
        previous.save(path, intent)
        raise RuntimeError("Submission outcome unresolved; no blind retry: " + out)
    intent.update(status="submitted", job_id=out.strip())
    previous.save(path, intent)
    return intent


def status(job_id):
    if not previous.HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS handle")
    rc, out = previous.direct_native_command(None, "qstat -xf " + shlex.quote(job_id), timeout=35)
    fields = {"job_id": job_id, "query_rc": rc}
    for key in ("job_state", "Exit_status", "queue", "exec_host", "comment", "resources_used.walltime", "Resource_List.ngpus", "Resource_List.ncpus", "Resource_List.mem"):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", out, re.M)
        fields[key] = match.group(1).strip() if match else None
    previous.save(HERE / "scheduler_snapshot.json", fields)
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("submit", "status"))
    args = parser.parse_args()
    if args.action == "submit":
        result = submit()
    else:
        handle = json.loads((HERE / "submit_handle.json").read_text())
        result = status(handle["job_id"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
