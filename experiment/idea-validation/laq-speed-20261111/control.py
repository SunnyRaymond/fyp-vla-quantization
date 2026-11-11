"""Controller of the LAQ speed experiment (CUDA graphs): upload, cputest, submit, status, pull, logs, cancel.
Its own remote root and control/jobs.json; it reads the LAQ experiment and never submits, uploads or writes there.

  control.py upload
  control.py cputest
  control.py submit --jobs calib_short calib_long loop_short loop_long [--after cputest]
  control.py status | pull | logs --job calib_short [--name job.log]
The calibration is submitted twice (a job under 2 h and a longer one); the two share the work through claims and each has
its own closed-loop job behind it (afterok). watch.py cancels the pair that loses.
Job ids live in control/jobs.json on the cluster; a job that is queued, held or running is never submitted twice, and
uploads are refused while a recorded job is live.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("w4kv_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
remote = base.remote
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
ROOT = P["remote_root"]
FILES = ("protocol.json", "env_fastwam.sh", "slots.sh", "run.pbs", "cputest.pbs", "qops.py", "methods.py", "laq.py",
         "calib.py", "fastwam_loop.py", "loopcommon.py", "fw_common.py", "sched.py", "analysis.py", "kvsites.py", "graphs.py",
         "graph_bench.py", "test_laq.py", "test_graph.py", "int4.py", "int4_kernels.py", "int4_bench.py", "test_int4.py")
JOBS_PATH = ROOT + "/control/jobs.json"
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
KEYS = {"graph": "graph", "int4": "int4"}
AFTER = {}
CPU_SELECT, CPU_WALLTIME = "1:ncpus=8:mem=32gb", "02:30:00"


def run(t, command, timeout=60):
    return remote.command(t, command, timeout=timeout)


def load_jobs(sftp):
    data = base._small_bytes(sftp, JOBS_PATH)
    return json.loads(data) if data else {}


def save_jobs(sftp, jobs):
    with sftp.file(JOBS_PATH + ".tmp", "wb") as stream:
        stream.write((json.dumps(jobs, indent=2) + "\n").encode())
    sftp.posix_rename(JOBS_PATH + ".tmp", JOBS_PATH)
    (HERE / "pulled").mkdir(exist_ok=True)
    (HERE / "pulled" / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")


def qstat(t, job_id):
    _, out = run(t, "qstat -xf " + shlex.quote(job_id), timeout=40)
    fields = {}
    for key in ("job_state", "Exit_status", "queue", "resources_used.walltime", "Resource_List.walltime", "comment", "stime"):
        m = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", out, re.M)
        fields[key] = m.group(1).strip() if m else None
    return fields


def live(state):
    return state["job_state"] in ("Q", "H", "R", "B", "W", "E")


def upload(t, force=False):
    run(t, "mkdir -p " + " ".join(shlex.quote(ROOT + d) for d in ("/control", "/artifacts", "/results")))
    changed = []
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        busy = {k for k, v in jobs.items() if live(qstat(t, v["job_id"]))}
        for name in FILES:
            data = (HERE / name).read_bytes().replace(b"\r\n", b"\n")
            target = ROOT + "/" + name
            try:
                with sftp.file(target, "rb") as stream:
                    previous = stream.read()
            except OSError:
                previous = None
            if previous == data:
                continue
            if busy and not force:
                raise RuntimeError(f"{name} differs but jobs {sorted(busy)} are live; refusing to upload")
            with sftp.file(target + ".tmp", "wb") as stream:
                stream.write(data)
            sftp.posix_rename(target + ".tmp", target)
            changed.append(name)
    run(t, "chmod +x " + " ".join(shlex.quote(ROOT + "/" + n) for n in FILES if n.endswith((".pbs", ".sh"))))
    return changed


def qsub(t, name, select, walltime, script, variables=None, after=()):
    parts = ["qsub", "-N", name, "-q", "normal", "-P", base.PROJECT, "-l", "select=" + select,
             "-l", "walltime=" + walltime, "-o", ROOT + "/artifacts/" + name + ".pbs.out", "-j", "oe"]
    if variables:
        parts += ["-v", ",".join(f"{k}={v}" for k, v in variables.items())]
    if after:
        parts += ["-W", "depend=afterok:" + ":".join(after)]
    code, out = run(t, " ".join(shlex.quote(x) for x in parts) + " " + shlex.quote(ROOT + "/" + script), timeout=60)
    job_id = out.strip()
    if code or not HANDLE.fullmatch(job_id):
        raise RuntimeError(f"qsub {name} failed: {out[-800:]}")
    return job_id


def producers(t, jobs, keys):
    after = []
    for key in keys:
        if key not in jobs:
            continue
        state = qstat(t, jobs[key]["job_id"])
        if state["job_state"] != "F":
            after.append(jobs[key]["job_id"])
        elif state["Exit_status"] != "0":
            raise RuntimeError(f"Producer {key} {jobs[key]['job_id']} finished with exit {state['Exit_status']}")
    return after


def submit(t, keys, extra_after=()):
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        for name in FILES:
            data = (HERE / name).read_bytes().replace(b"\r\n", b"\n")
            with sftp.file(ROOT + "/" + name, "rb") as stream:
                if stream.read() != data:
                    raise RuntimeError(f"Remote {name} differs from local; upload first")
    submitted = {}
    for key in keys:
        if key in jobs and live(qstat(t, jobs[key]["job_id"])):
            submitted[key] = dict(existing=jobs[key]["job_id"])
            continue
        j = P["models"]["fastwam"]["jobs"][KEYS[key]]
        select = f"1:ncpus={j['ncpus']}:mem={j['memory_gb']}gb:ngpus={j['gpus']}"
        after = producers(t, jobs, [*AFTER.get(key, ()), *extra_after])
        job_id = qsub(t, j["name"], select, j["walltime"], "run.pbs", dict(JOB=KEYS[key]), after)
        jobs[key] = dict(job_id=job_id, after=after)
        submitted[key] = dict(job_id=job_id, after=after)
        with remote.sftp_client(t) as sftp:
            save_jobs(sftp, jobs)
    return submitted


def _count(sftp, path):
    try:
        return len(sftp.listdir(path))
    except OSError:
        return 0


def status(t):
    d = f"{ROOT}/results/fastwam"
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        out = dict(jobs={k: dict(v, **qstat(t, v["job_id"])) for k, v in jobs.items()}, failed={})
        try:
            arms = sftp.listdir(f"{d}/calib/units")
        except OSError:
            arms = []
        out["calib"] = dict(units=sum(_count(sftp, f"{d}/calib/units/{a}") for a in arms), arms=len(arms),
                            allocated=base._small_bytes(sftp, f"{d}/calib/allocation.json") is not None,
                            complete=base._small_bytes(sftp, f"{d}/calib/COMPLETE.json") is not None)
        out["episodes"] = {arm: _count(sftp, f"{d}/loops/{arm}/episodes") for arm in P["arms"]}
        for key, v in jobs.items():
            text = base._text(sftp, f"{ROOT}/artifacts/{v['job_id']}/FAILED", 4000)
            if text:
                out["failed"][key] = text
    return out


def pull(t):
    target = HERE / "pulled"
    names = ["results/fastwam/summary.json", "results/fastwam/RESULTS.zh.md"]
    names += [f"results/fastwam/calib/{f}" for f in ("summary.json", "RESULTS.zh.md", "allocation.json", "COMPLETE.json")]
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        try:
            names += [f"results/fastwam/calib/{f}" for f in sftp.listdir(f"{ROOT}/results/fastwam/calib") if f.startswith("startup_")]
        except OSError:
            pass
        try:
            names += [f"results/fastwam/bench/{b}/{f}" for b in sftp.listdir(f"{ROOT}/results/fastwam/bench")
                      for f in ("summary.json", "RESULTS.zh.md")]
        except OSError:
            pass
        for arm in P["arms"]:
            try:
                workers = [w for w in sftp.listdir(f"{ROOT}/results/fastwam/loops/{arm}") if w.startswith(("worker_", "smoke"))]
            except OSError:
                workers = []
            names += [f"results/fastwam/loops/{arm}/{w}/startup.json" for w in workers]
        for v in jobs.values():
            names += [f"artifacts/{v['job_id']}/{f}" for f in ("exit_code.txt", "FAILED", "PIPELINE_COMPLETE")]
        found = 0
        for name in names:
            data = base._small_bytes(sftp, ROOT + "/" + name, 4 << 20)
            if data is not None:
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                found += 1
        for v in jobs.values():
            _, log = run(t, "tail -c 200000 " + shlex.quote(f"{ROOT}/artifacts/{v['job_id']}/job.log") + " 2>/dev/null")
            if log:
                path = target / "artifacts" / v["job_id"] / "job.log.tail"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(log, encoding="utf-8")
    return dict(directory=str(target), files=found)


def cputest(t):
    job_id = qsub(t, "lsgct", CPU_SELECT, CPU_WALLTIME, "cputest.pbs")
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        jobs["cputest"] = dict(job_id=job_id, after=[])
        save_jobs(sftp, jobs)
    return dict(job_id=job_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "cputest", "submit", "status", "pull", "logs", "cancel"))
    parser.add_argument("--jobs", nargs="*")
    parser.add_argument("--after", nargs="*", default=[])
    parser.add_argument("--job")
    parser.add_argument("--name", default="job.log")
    parser.add_argument("--bytes", type=int, default=6000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    jump, t = base.direct_connect()
    try:
        if args.action == "upload":
            result = dict(uploaded=upload(t, args.force))
        elif args.action == "cputest":
            result = cputest(t)
        elif args.action == "submit":
            result = submit(t, args.jobs or list(KEYS), args.after)
        elif args.action == "status":
            result = status(t)
        elif args.action == "pull":
            result = pull(t)
        elif args.action == "cancel":
            with remote.sftp_client(t) as sftp:
                jobs = load_jobs(sftp)
            result = {k: run(t, "qdel " + shlex.quote(jobs[k]["job_id"]))[0] for k in args.jobs if k in jobs}
        else:
            with remote.sftp_client(t) as sftp:
                job_id = load_jobs(sftp)[args.job]["job_id"]
            print(run(t, f"tail -c {int(args.bytes)} " + shlex.quote(f"{ROOT}/artifacts/{job_id}/{args.name}"))[1])
            return
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        t.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
