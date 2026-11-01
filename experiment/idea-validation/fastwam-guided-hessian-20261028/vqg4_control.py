"""Controller for G4 (vqg4.json): uploads only its own files (plus the unchanged worker chain it imports), runs its CPU
test, submits the gdev job and reads status / logs / results. Built on control.py's configure and base, with its own
state file (control/state_vqg4<attempt>.json), so the other phases' sources and submission states are left alone."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import control as c  # noqa: E402

base = c.base
G4 = json.loads((HERE / "vqg4.json").read_text(encoding="utf-8"))
FILES = ("vqg4.json", "vqg4_worker.py", "run_vqg4.pbs", "test_vqg4.py", "cputest_vqg4.pbs")
SMALL = (["vqg4/GPTVQ/summary.json", "vqg4/GPTVQ/RESULTS.zh.md", "vqg4/GPTVQ/codes.json"]
         + [f"progress/{G4['allocation']['job_name']}{a}_0.json" for a in ("", "r1", "r2", "r3")])


def configure(attempt, route="gdev"):
    alloc = G4["allocation"] if route == "gdev" else G4["race_allocation"]
    small = list(SMALL) + [f"progress/{alloc['job_name']}{a}_0.json" for a in ("", "r1", "r2", "r3")]
    c.CONFIG["vqg4" + ("" if route == "gdev" else "g")] = (alloc, "run_vqg4.pbs", small)
    a = c.configure(attempt, "vqg4" + ("" if route == "gdev" else "g"))
    base.FILES = FILES
    return a


def cputest_status(transport, job_id):
    scheduler = base._scheduler(transport, job_id)
    found = {}
    with base.remote.sftp_client(transport) as sftp:
        for ext in ("json", "log"):
            data = base._small_bytes(sftp, f"{base.RESULTS}/cputest/{job_id}/test_vqg4.{ext}")
            if data is not None:
                text = data.decode("utf-8", errors="replace")
                found[f"test_vqg4.{ext}"] = json.loads(text) if ext == "json" else text[-3000:]
        found["passed"] = base._small_bytes(sftp, f"{base.RESULTS}/cputest/{job_id}/PASSED") is not None
    return dict(job_state=scheduler.get("job_state"), exit_status=scheduler.get("Exit_status"), **found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="")
    parser.add_argument("--id")
    parser.add_argument("--route", choices=("gdev", "g1"), default="gdev")
    parser.add_argument("action", choices=("upload", "cputest", "cputest-status", "submit", "status", "logs", "pull"))
    args = parser.parse_args()
    gpus = configure(args.attempt, args.route)["gpus"]
    jump, transport = base.direct_connect()
    try:
        if args.action == "upload":
            result = {"uploaded": base.upload(transport)}
        elif args.action == "cputest":
            result = {"uploaded": base.upload(transport), **c.cpu_submit(transport, "cputest_vqg4.pbs", "fwtrct")}
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
