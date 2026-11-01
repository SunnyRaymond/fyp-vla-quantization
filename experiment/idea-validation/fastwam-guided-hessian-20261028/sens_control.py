"""Controller for FLAQ module 1 (sens.json): uploads only the files its jobs read, runs its CPU test, submits the gdev
job (--phase sens: fwse1, 4 GPUs, all 4 shards) and the g1 jobs (--phase sens0..sens3: fwse1g0..3, one shard each),
races them and reads status / results. Built on control.py's configure and base, with its own state files
(control/state_sens*.json), so the other phases' sources and submission states are left alone.

  diff            which of the uploaded files differ from the remote copies (look before upload overwrites)
  race            after all five submits: the first job to start wins (gdev -> qdel the g1 jobs; a g1 job -> qdel
                  gdev); then quiet until every surviving job ends or a worker fails; pulls; prints one JSON line
  watch --phases  quiet until the listed phases' jobs all end or a worker fails; pulls; prints one JSON line
--phase sensx is the g3 four-GPU job fwse1x that took over shards 1-3 after their g1 jobs were cancelled in the queue.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import control as c  # noqa: E402

base = c.base
SENS = json.loads((HERE / "sens.json").read_text(encoding="utf-8"))
HOST = SENS["host"]
JOBS = SENS["allocation_jobs"]
FILES = ("protocol.json", "host.json", "gate.json", "sens.json", "run_sens.pbs", "environment.sh", "worker.py",
         "gate_worker.py", "sens_worker.py", "probe.py", "test_sens.py", "cputest_sens.pbs")
CONFIGS = (["none", "all"] + [f"{e}.b{b:02d}" for e in ("action", "video") for b in range(30)]
           + ["action.cond", "action.io", "video.cond", "video.head"])
ATTEMPTS = ("", "r1", "r2", "r3")


def allocation(phase):
    if phase == "sens":
        return dict(JOBS["gdev"])
    if phase == "sensx":
        return dict(JOBS["g3"])
    k = int(phase[len("sens"):])
    a = dict(JOBS["g1"])
    a["job_name"] = a["job_name"].format(k=k)
    return a


def shard_ids(phase):
    if phase in ("sens", "sensx"):
        return list(JOBS["gdev" if phase == "sens" else "g3"]["shards"])
    return [int(phase[len("sens"):])]


PHASES = ["sens"] + [f"sens{k}" for k in range(JOBS["shards"])] + ["sensx"]
NAMES = [allocation(p)["job_name"] for p in PHASES]
SMALL = ([f"sens/{HOST}/{n}" for n in ("summary.json", "selection.json", "RESULTS.zh.md")]
         + [f"sens/{HOST}/configs/{x}.json" for x in CONFIGS]
         + [f"sens/{HOST}/{d}/{n}{a}_{k}.json" for d in ("shards", "waiting") for n in NAMES for a in ATTEMPTS
            for k in range(JOBS["shards"])]
         + [f"progress/{n}{a}_{k}.json" for n in NAMES for a in ATTEMPTS for k in range(JOBS["shards"])])


def configure(attempt, phase):
    c.CONFIG[phase] = (allocation(phase), "run_sens.pbs", SMALL)
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


def status(transport, phase):
    value = base.status(transport)
    if value.get("handle"):
        with base.remote.sftp_client(transport) as sftp:
            value["workers"] = {
                str(k): dict(progress=base._json(sftp, base.RESULTS + f"/progress/{base.JOB_NAME}_{k}.json"),
                             failure=base._json(sftp, base.RESULTS + f"/failures/{base.JOB_NAME}_{k}.json"))
                for k in shard_ids(phase)}
    return value


def cputest_status(transport, job_id):
    scheduler = base._scheduler(transport, job_id)
    found = {}
    with base.remote.sftp_client(transport) as sftp:
        for ext in ("json", "log"):
            data = base._small_bytes(sftp, f"{base.RESULTS}/cputest/{job_id}/test_sens.{ext}")
            if data is not None:
                text = data.decode("utf-8", errors="replace")
                found[f"test_sens.{ext}"] = json.loads(text) if ext == "json" else text[-3000:]
        found["passed"] = base._small_bytes(sftp, f"{base.RESULTS}/cputest/{job_id}/PASSED") is not None
    return dict(job_state=scheduler.get("job_state"), exit_status=scheduler.get("Exit_status"), **found)


def point(attempt, phase):
    """Only the job name and state file of a phase (configure wraps base._save_state again on every call)."""
    base.JOB_NAME = allocation(phase)["job_name"] + attempt
    base.STATE_PATH = base.ROOT + f"/control/state_{phase}{attempt}.json"


def snapshot(attempt):
    """phase -> (job id, job state, failures) for every submitted phase of this attempt."""
    jump, transport = base.direct_connect()
    try:
        out = {}
        for phase in PHASES:
            point(attempt, phase)
            handle, _ = base._resolve(transport)
            if not handle:
                continue
            state = base._scheduler(transport, handle["job_id"]).get("job_state")
            if state is None:
                raise RuntimeError("qstat failed")
            with base.remote.sftp_client(transport) as sftp:
                failures = [k for k in shard_ids(phase)
                            if base._json(sftp, base.RESULTS + f"/failures/{base.JOB_NAME}_{k}.json")]
            out[phase] = dict(job_id=handle["job_id"], state=state, failures=failures)
        return out
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


def qdel(job_id):
    if not base.HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS job handle")
    jump, transport = base.direct_connect()
    try:
        code, out = base.remote.command(transport, "qdel " + shlex.quote(job_id), timeout=35)
        return dict(job_id=job_id, rc=code, output=out[-300:])
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


def race(attempt, interval, winner=None, watched=()):
    """First start wins, then wait for the survivors; quiet until the end. Query failures are tolerated for 10 h
    (the jump host is only reachable over the NTU VPN, which drops at night)."""
    configure(attempt, "sens")
    failed, decided, cancelled = 0, winner or ("watch" if watched else None), []  # decided earlier: only wait
    while True:
        try:
            snap = snapshot(attempt)
            failed = 0
        except Exception as error:
            failed += 1
            if failed * interval >= 36000:
                print(json.dumps(dict(event="query_failed", error=f"{type(error).__name__}: {error}")), flush=True)
                return 1
            time.sleep(interval)
            continue
        if decided is None:
            started = [p for p, v in snap.items() if v["state"] in ("R", "F")]
            if started:
                decided = "sens" if "sens" in started else "g1"
                losers = [p for p in snap if (p != "sens") == (decided == "sens")]
                cancelled = [qdel(snap[p]["job_id"]) for p in losers if snap[p]["state"] in ("Q", "H", "R")]
                print(json.dumps(dict(event="race_decided", winner=decided, started=started, cancelled=cancelled)),
                      flush=True)
                continue
        live = {p: v for p, v in snap.items()
                if ((p in watched) if watched
                    else (decided and p != "sensx" and ((p == "sens") == (decided == "sens"))))}
        failures = {p: v["failures"] for p, v in live.items() if v["failures"]}
        if failures:
            print(json.dumps(dict(event="failure", winner=decided, jobs=snap, failures=failures)), flush=True)
            return 0
        if live and all(v["state"] in ("F", "X") for v in live.values()):
            pulled = {}
            jump, transport = base.direct_connect()
            try:
                for p in live:
                    gpus = configure(attempt, p)["gpus"]
                    pulled[p] = c.pull(transport, gpus, SMALL)
            finally:
                if transport:
                    transport.close()
                if jump:
                    jump.close()
            print(json.dumps(dict(event="terminal", winner=decided, jobs=snap, pulled=pulled), default=str), flush=True)
            return 0
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="")
    parser.add_argument("--phase", choices=PHASES, default="sens")
    parser.add_argument("--id")
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--winner", choices=("sens", "g1"), help="race already decided: only wait for these jobs")
    parser.add_argument("--phases", nargs="+", choices=PHASES, help="watch: the phases to wait for")
    parser.add_argument("action", choices=("diff", "upload", "cputest", "cputest-status", "submit", "status", "logs",
                                           "pull", "race", "watch"))
    args = parser.parse_args()
    if args.action == "race":
        return race(args.attempt, args.interval, args.winner)
    if args.action == "watch":
        if not args.phases:
            parser.error("watch needs --phases")
        return race(args.attempt, args.interval, watched=tuple(args.phases))
    gpus = configure(args.attempt, args.phase)["gpus"]
    jump, transport = base.direct_connect()
    try:
        if args.action == "diff":
            result = diff(transport)
        elif args.action == "upload":
            result = {"uploaded": base.upload(transport)}
        elif args.action == "cputest":
            result = {"uploaded": base.upload(transport), **c.cpu_submit(transport, "cputest_sens.pbs", "fwsect")}
        elif args.action == "cputest-status":
            result = cputest_status(transport, args.id)
        elif args.action == "submit":
            result = base.submit(transport)
        elif args.action == "logs":
            result = base.logs(transport)
        elif args.action == "pull":
            result = c.pull(transport, gpus, SMALL)
        else:
            result = status(transport, args.phase)
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
