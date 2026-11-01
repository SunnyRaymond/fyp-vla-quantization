"""Controller for RAVQ calibrated (ravqc.json): uploads only the files its jobs read, runs its CPU test, submits
--phase ravqc (fwrc1, gdev 4 GPUs: codes + reader metric), --phase ravqc2 (fwrc2, g2 2 GPUs: calibration + Z_M + final),
--phase ravqcg2 / ravqcg3 (fwrcg2 / fwrcg3, gdev 4 GPUs: per-context table gradients and their transfer) and --phase
ravqcl (fwrcl, g3 closed loop), reads status / results; resummary (CPU, afterok on --after) reruns the gradtx summary
from the stored per-context files (gradtx_resummary.py). Built on control.py's configure and base with its own state
files (control/state_ravqc*.json), so the other phases' sources and submission states are left alone.

  diff   which of the uploaded files differ from the remote copies (look before upload overwrites)
  cancel qdel the phase's submitted job (race.qdel)
  watch  quiet until the job ends or a worker fails; pulls; prints one JSON line
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import control as c  # noqa: E402
import race  # noqa: E402

base = c.base
RC = json.loads((HERE / "ravqc.json").read_text(encoding="utf-8"))
FILES = ("protocol.json", "host.json", "gate.json", "ravq.json", "ravqc.json", "run_ravqc.pbs", "run_ravqc_loop.pbs",
         "environment.sh", "worker.py", "gate_worker.py", "ravq_worker.py", "ravq_qtip.py", "ravqc_worker.py",
         "ravqc_loop.py", "probe.py", "test_ravqc.py", "cputest_ravqc.pbs", "gradtx_resummary.py", "gradtx_resummary.pbs")
ATTEMPTS = ("", "r1", "r2", "r3", "r4")
JOBS = dict(ravqc=(RC["allocation_job"], "run_ravqc.pbs"), ravqc2=(RC["calib_job"], "run_ravqc.pbs"),
            ravqcl=(RC["loop_job"], "run_ravqc_loop.pbs"),
            **{"ravqcg" + a[-1]: (j, "run_ravqc.pbs") for a, j in RC["gradtx_jobs"].items()})
ARMS = list(RC["base_arms"])
SMALL = (["ravqc/summary.json", "ravqc/RESULTS.zh.md", "ravqc/DECISION.json", "ravqc/codes/summary.json",
          "ravqc/metric/summary.json", "ravqc/metric/RESULTS.zh.md", "ravqc/loop/summary.json", "ravqc/loop/RESULTS.zh.md"]
         + [f"ravqc/codes/reports/{j['job_name']}{a}_{k}.json" for j in (RC["allocation_job"],) for a in ATTEMPTS
            for k in range(j["gpus"])]
         + [f"ravqc/{k}/{n}" for k in ARMS for n in ("arm.json", "step.json", "training.json", "fit_before.json",
                                                    "fit_after.json")]
         + [f"ravqc/heldout/{k}.json" for k in ARMS] + [f"ravqc/zm/{a}.json" for a in RC["zm_arms"]]
         + [f"ravqc/gradtx/{k}/{n}" for k in ARMS for n in ("summary.json", "RESULTS.zh.md")]
         + [f"quantized/{k}/{n}" for k in ARMS for n in ("COMPLETE.json", "sources.json")]
         + [f"loops/{k}/{n}" for k in ARMS for n in ("summary.json", "RESULTS.zh.md", "DONE")]
         + [f"loops/{k}/rc_check_{w}.json" for k in ARMS for w in range(8)]
         + [f"progress/{j['job_name']}{a}_{k}.json" for j, _ in JOBS.values() for a in ATTEMPTS for k in range(j["gpus"])]
         + [f"failures/{j['job_name']}{a}_{k}.json" for j, _ in JOBS.values() for a in ATTEMPTS for k in range(j["gpus"])])


def configure(attempt, phase):
    a, script = JOBS[phase]
    c.CONFIG[phase] = (dict(a), script, SMALL)
    a = c.configure(attempt, phase)
    base.FILES = FILES
    return a


def depend_on(job_id):
    """qsub with -W depend=afterok:<id>[:<id> ...]: the job is queued now and starts only after every listed job exits
    0 (a failed producer drops it)."""
    if not re.fullmatch(r"\d+\.pbs101(:\d+\.pbs101)*", job_id or ""):
        raise ValueError(f"Bad PBS job id for --after: {job_id!r}")
    plain = base._qsub_command
    base._qsub_command = lambda: plain().replace("qsub ", f"qsub -W depend=afterok:{job_id} ", 1)


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
            data = base._small_bytes(sftp, f"{base.RESULTS}/cputest/{job_id}/test_ravqc.{ext}")
            if data is not None:
                text = data.decode("utf-8", errors="replace")
                found[f"test_ravqc.{ext}"] = json.loads(text) if ext == "json" else text[-4000:]
        found["passed"] = base._small_bytes(sftp, f"{base.RESULTS}/cputest/{job_id}/PASSED") is not None
    return dict(job_state=scheduler.get("job_state"), exit_status=scheduler.get("Exit_status"), **found)


def watch(attempt, interval, phase):
    """Quiet until the job is finished or a worker writes a failure; tolerates query failures for 10 h (VPN drops)."""
    shards = configure(attempt, phase)["gpus"]
    failed = 0
    while True:
        try:
            jump, transport = base.direct_connect()
            try:
                value = c.status(transport, shards)
                failures = {k: v["failure"] for k, v in value.get("workers", {}).items() if v.get("failure")}
                state = (value.get("scheduler") or {}).get("job_state")
                if failures or state in ("F", "X") or value.get("status") in ("complete", "failed"):
                    pulled = c.pull(transport, shards, SMALL)
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


def cputest_watch(job_id, interval):
    configure("", "ravqc")
    while True:
        try:
            jump, transport = base.direct_connect()
            try:
                value = cputest_status(transport, job_id)
            finally:
                if transport:
                    transport.close()
                if jump:
                    jump.close()
            if value.get("job_state") in ("F", "X") or value.get("passed"):
                print(json.dumps(value, ensure_ascii=False, default=str), flush=True)
                return 0
        except Exception:
            pass
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="")
    parser.add_argument("--phase", choices=tuple(JOBS), default="ravqc")
    parser.add_argument("--id")
    parser.add_argument("--after", help="submit with -W depend=afterok:<job id>")
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("action", choices=("diff", "upload", "cputest", "cputest-status", "cputest-watch", "submit",
                                           "resummary", "status", "logs", "pull", "watch", "cancel"))
    args = parser.parse_args()
    if args.action == "watch":
        return watch(args.attempt, args.interval, args.phase)
    if args.action == "cputest-watch":
        return cputest_watch(args.id, args.interval)
    shards = configure(args.attempt, args.phase)["gpus"]
    jump, transport = base.direct_connect()
    try:
        if args.action == "diff":
            result = diff(transport)
        elif args.action == "upload":
            result = {"uploaded": base.upload(transport)}
        elif args.action == "cputest":
            result = {"uploaded": base.upload(transport), **c.cpu_submit(transport, "cputest_ravqc.pbs", "fwrcct")}
        elif args.action == "resummary":
            result = {"uploaded": base.upload(transport),
                      **c.cpu_submit(transport, "gradtx_resummary.pbs", "fwrcgs", after=args.after)}
        elif args.action == "cputest-status":
            result = cputest_status(transport, args.id)
        elif args.action == "submit":
            if args.after:
                depend_on(args.after)
            result = base.submit(transport)
        elif args.action == "logs":
            result = base.logs(transport)
        elif args.action == "pull":
            result = c.pull(transport, shards, SMALL)
        elif args.action == "cancel":
            result = race.qdel(transport, c.status(transport, shards)["handle"]["job_id"])
        else:
            result = c.status(transport, shards)
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
