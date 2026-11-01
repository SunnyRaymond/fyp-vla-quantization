"""Controller for RAVQ uncalibrated (ravq.json): uploads only the files its jobs read, runs its CPU test, submits the gdev
job (--phase ravq: fwrv1, 4 GPUs), the g3 closed loop (--phase ravql: fwrvl, the arms results/ravq/DECISION.json
lists) and reads status / results. Built on control.py's configure and base, with its own
state files (control/state_ravq*.json), so the other phases' sources and submission states are left alone.

  diff   which of the uploaded files differ from the remote copies (look before upload overwrites)
  cancel qdel the phase's submitted job (race.qdel)
  watch  quiet until the job ends or a worker fails; pulls; prints one JSON line
"""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import control as c  # noqa: E402
import race  # noqa: E402

base = c.base
RV = json.loads((HERE / "ravq.json").read_text(encoding="utf-8"))
FILES = ("protocol.json", "host.json", "gate.json", "ravq.json", "run_ravq.pbs", "run_ravq_loop.pbs", "environment.sh",
         "worker.py", "gate_worker.py", "ravq_worker.py", "ravq_qtip.py", "probe.py", "test_ravq.py", "cputest_ravq.pbs",
         "run_ravq_trace.pbs", "trace_worker.py")
ATTEMPTS = ("", "r1", "r2", "r3")
SHARDS = RV["allocation_job"]["gpus"]
JOBS = dict(ravq=(RV["allocation_job"], "run_ravq.pbs"), ravql=(RV["loop_job"], "run_ravq_loop.pbs"),
            ravqt=(RV["trace_job"], "run_ravq_trace.pbs"))
PLANNED = [a for a, v in RV["arms"].items() if not v.get("reference")]
SMALL = ([f"ravq/{n}" for n in ("summary.json", "RESULTS.zh.md", "DECISION.json", "plan.json")]
         + [f"quantized/{a}/{n}" for a in PLANNED for n in ("COMPLETE.json", "sources.json")]
         + [f"heldout/{a}.json" for a in list(RV["arms"]) + ["QTIP_U_W2"]]
         + [f"progress/{RV['allocation_job']['job_name']}{a}_{k}.json" for a in ATTEMPTS for k in range(SHARDS)]
         + [f"failures/{RV['allocation_job']['job_name']}{a}_{k}.json" for a in ATTEMPTS for k in range(SHARDS)]
         + ["ravq/loop/summary.json", "ravq/loop/RESULTS.zh.md", "ravq/trace/summary.json", "ravq/trace/RESULTS.zh.md"]
         + [f"loops/{a}/{n}" for a in PLANNED for n in ("summary.json", "RESULTS.zh.md", "DONE")])


def configure(attempt, phase="ravq"):
    a, script = JOBS[phase]
    c.CONFIG[phase] = (dict(a), script, SMALL)
    a = c.configure(attempt, phase)
    base.FILES = FILES
    return a


def diff(transport):
    out = {}
    with base.remote.sftp_client(transport) as sftp:
        for name in FILES:
            remote = base._small_bytes(sftp, base.ROOT + "/" + name)
            out[name] = "missing" if remote is None else "same" if remote == base._local_bytes(name) else "differs"
    return out


def cputest_status(transport, job_id):
    scheduler = base._scheduler(transport, job_id)
    found = {}
    with base.remote.sftp_client(transport) as sftp:
        for ext in ("json", "log"):
            data = base._small_bytes(sftp, f"{base.RESULTS}/cputest/{job_id}/test_ravq.{ext}")
            if data is not None:
                text = data.decode("utf-8", errors="replace")
                found[f"test_ravq.{ext}"] = json.loads(text) if ext == "json" else text[-3000:]
        found["passed"] = base._small_bytes(sftp, f"{base.RESULTS}/cputest/{job_id}/PASSED") is not None
    return dict(job_state=scheduler.get("job_state"), exit_status=scheduler.get("Exit_status"), **found)


def watch(attempt, interval, phase):
    """Quiet until the job is finished or a worker writes a failure; tolerates query failures for 10 h (VPN drops)."""
    configure(attempt, phase)
    failed = 0
    while True:
        try:
            jump, transport = base.direct_connect()
            try:
                value = c.status(transport, SHARDS)
                failures = {k: v["failure"] for k, v in value.get("workers", {}).items() if v.get("failure")}
                state = (value.get("scheduler") or {}).get("job_state")
                if failures or state in ("F", "X") or value.get("status") in ("complete", "failed"):
                    pulled = c.pull(transport, SHARDS, SMALL)
                    print(json.dumps(dict(event="terminal" if not failures else "failure", state=state,
                                          status=value.get("status"), failures=failures, pulled=pulled), default=str),
                          flush=True)
                    return 0
            finally:
                if transport:
                    transport.close()
                if jump:
                    jump.close()
            failed = 0
        except Exception as error:
            failed += 1
            if failed * interval >= 36000:
                print(json.dumps(dict(event="query_failed", error=f"{type(error).__name__}: {error}")), flush=True)
                return 1
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="")
    parser.add_argument("--phase", choices=tuple(JOBS), default="ravq")
    parser.add_argument("--id")
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("action", choices=("diff", "upload", "cputest", "cputest-status", "submit", "status", "logs",
                                           "pull", "watch", "cancel"))
    args = parser.parse_args()
    if args.action == "watch":
        return watch(args.attempt, args.interval, args.phase)
    configure(args.attempt, args.phase)
    jump, transport = base.direct_connect()
    try:
        if args.action == "diff":
            result = diff(transport)
        elif args.action == "upload":
            result = {"uploaded": base.upload(transport)}
        elif args.action == "cputest":
            result = {"uploaded": base.upload(transport), **c.cpu_submit(transport, "cputest_ravq.pbs", "fwrvct")}
        elif args.action == "cputest-status":
            result = cputest_status(transport, args.id)
        elif args.action == "submit":
            result = base.submit(transport)
        elif args.action == "logs":
            result = base.logs(transport)
        elif args.action == "pull":
            result = c.pull(transport, SHARDS, SMALL)
        elif args.action == "cancel":
            result = race.qdel(transport, c.status(transport, SHARDS)["handle"]["job_id"])
        else:
            result = c.status(transport, SHARDS)
            result.pop("log_tail", None) if result.get("status") == "queued" else None
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
