"""Controller for the trajectory-coherent proxy test (traj.json): uploads only its own files, runs its CPU test, submits
the gdev job and reads status / logs / results. Built on control.py's configure and base, with its own state file
(control/state_traj<attempt>.json), so the other phases' sources and submission states are left alone."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import control as c  # noqa: E402

base = c.base
TRAJ = json.loads((HERE / "traj.json").read_text(encoding="utf-8"))
FILES = ("traj.json", "traj_worker.py", "run_traj.pbs", "test_traj.py", "cputest_traj.pbs")
SMALL = (["traj/summary.json", "traj/RESULTS.zh.md", "traj/h_all_check.json", "traj/layers.json"]
         + [f"progress/{TRAJ['allocation']['job_name']}{a}_0.json" for a in ("", "r1", "r2", "r3")])


def configure(attempt):
    c.CONFIG["traj"] = (TRAJ["allocation"], "run_traj.pbs", SMALL)
    a = c.configure(attempt, "traj")
    base.FILES = FILES
    return a


def cputest_status(transport, job_id):
    scheduler = base._scheduler(transport, job_id)
    found = {}
    with base.remote.sftp_client(transport) as sftp:
        for ext in ("json", "log"):
            data = base._small_bytes(sftp, f"{base.RESULTS}/cputest/{job_id}/test_traj.{ext}")
            if data is not None:
                text = data.decode("utf-8", errors="replace")
                found[f"test_traj.{ext}"] = json.loads(text) if ext == "json" else text[-3000:]
        found["passed"] = base._small_bytes(sftp, f"{base.RESULTS}/cputest/{job_id}/PASSED") is not None
    return dict(job_state=scheduler.get("job_state"), exit_status=scheduler.get("Exit_status"), **found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="")
    parser.add_argument("--id")
    parser.add_argument("action", choices=("upload", "cputest", "cputest-status", "submit", "status", "logs", "pull"))
    args = parser.parse_args()
    gpus = configure(args.attempt)["gpus"]
    jump, transport = base.direct_connect()
    try:
        if args.action == "upload":
            result = {"uploaded": base.upload(transport)}
        elif args.action == "cputest":
            result = {"uploaded": base.upload(transport), **c.cpu_submit(transport, "cputest_traj.pbs", "fwtrct")}
        elif args.action == "cputest-status":
            result = cputest_status(transport, args.id)
        elif args.action == "submit":
            result = base.submit(transport)
        elif args.action == "logs":
            result = base.logs(transport)
        elif args.action == "pull":
            result = c.pull(transport, gpus, SMALL)
        else:
            result = c.status(transport, gpus)
            result.pop("log_tail", None) if result.get("status") == "queued" else None
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
