"""Controller for the Cosmos A16 baselines: upload, cputest, submit (gdev single-GPU stages + the 4-GPU loop job with
afterok dependencies), status, pull, logs. Uses the host-key-verified SSH helper of the earlier controllers.

  control.py upload
  control.py cputest
  control.py submit --jobs t5 prep gptq qtip3 ...   (default: every single-GPU job, then the loop job)
  control.py submit --jobs loop --after cputest     (extra afterok producers)
  control.py status | pull | logs --job prep [--name job.log]
Job ids live in control/jobs.json on the cluster (and pulled/jobs.json here); a job already recorded and not
finished is never submitted twice, and uploads are refused while a recorded job is queued, held or running.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import posixpath
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cosmos_a16_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
remote = base.remote
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
ROOT = P["remote_root"]
FILES = ("protocol.json", "environment.sh", "slots.sh", "run_gpu.pbs", "run_loop.pbs", "cputest.pbs", "common.py",
         "prep_t5.py", "prep_hess.py", "quant_worker.py", "loop_worker.py", "sched.py", "analysis.py",
         "test_cosmos_a16.py")
JOBS_PATH = ROOT + "/control/jobs.json"
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
SINGLE = P["single_gpu"]
CPU_SELECT, CPU_WALLTIME = "1:ncpus=8:mem=32gb", "02:30:00"  # > 2 h routes to a q queue, not the often-full qdev


def sources():
    names = list(FILES)
    names += sorted(str(f.relative_to(HERE)).replace("\\", "/") for f in (HERE / "methods").rglob("*")
                    if f.is_file() and "__pycache__" not in f.parts)
    return names


def run(t, command, timeout=60):
    code, out = remote.command(t, command, timeout=timeout)
    return code, out


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
    code, out = run(t, "qstat -xf " + shlex.quote(job_id), timeout=40)
    fields = {}
    for key in ("job_state", "Exit_status", "queue", "resources_used.walltime", "Resource_List.walltime",
                "comment", "stime"):
        m = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", out, re.M)
        fields[key] = m.group(1).strip() if m else None
    return fields


def active(t, jobs):
    return {k: v for k, v in jobs.items() if qstat(t, v["job_id"])["job_state"] in ("Q", "H", "R", "B", "W")}


def upload(t, force=False):
    run(t, "mkdir -p " + " ".join(shlex.quote(ROOT + d) for d in ("/control", "/artifacts", "/results")))
    changed = []
    dirs = sorted({posixpath.dirname(ROOT + "/" + n) for n in sources()})
    run(t, "mkdir -p " + " ".join(shlex.quote(d) for d in dirs))
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        busy = active(t, jobs) if jobs else {}
        for name in sources():
            data = (HERE / name).read_bytes()
            if not name.endswith(".npy"):
                data = data.replace(b"\r\n", b"\n")
            target = ROOT + "/" + name
            try:
                with sftp.file(target, "rb") as stream:
                    previous = stream.read()
            except OSError:
                previous = None
            if previous == data:
                continue
            if busy and not force:
                raise RuntimeError(f"{name} differs but jobs {sorted(busy)} are queued/running; refusing to upload")
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
    """afterok ids of the recorded producers still queued or running; a finished producer must have exited 0."""
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
        for name in sources():
            data = (HERE / name).read_bytes()
            data = data if name.endswith(".npy") else data.replace(b"\r\n", b"\n")
            with sftp.file(ROOT + "/" + name, "rb") as stream:
                if stream.read() != data:
                    raise RuntimeError(f"Remote {name} differs from local; upload first")
    submitted = {}
    for key in keys:
        if key in jobs and qstat(t, jobs[key]["job_id"])["job_state"] in ("Q", "H", "R", "B", "W", None):
            submitted[key] = dict(existing=jobs[key]["job_id"])
            continue
        if key == "loop":
            cfg = P["loop"]
            after = producers(t, jobs, [*cfg["after"], *extra_after])
            job_id = qsub(t, cfg["name"], cfg["select"], cfg["walltime"], "run_loop.pbs", after=after)
        else:
            cfg = SINGLE["jobs"][key]
            variables = dict(STAGE=cfg["stage"])
            if "plan" in cfg:
                variables["PLAN"] = cfg["plan"].replace(" ", "+")
            after = producers(t, jobs, [*cfg.get("after", ()), *extra_after])
            job_id = qsub(t, cfg["name"], SINGLE["select"], cfg["walltime"], "run_gpu.pbs", variables, after)
        jobs[key] = dict(job_id=job_id, after=after)
        submitted[key] = dict(job_id=job_id, after=after)
        with remote.sftp_client(t) as sftp:
            save_jobs(sftp, jobs)
    return submitted


def status(t):
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        quant = {}
        for arm, cfg in P["arms"].items():
            if cfg["method"]:
                d = f"{ROOT}/results/quantized/{arm}"
                try:
                    layers = len(sftp.listdir(d + "/stats"))
                except OSError:
                    layers = 0
                quant[arm] = dict(layers=layers, complete=base._small_bytes(sftp, d + "/COMPLETE.json") is not None)
        loops = {}
        for arm in P["arm_order"]:
            d = f"{ROOT}/results/loops/{arm}"
            try:
                loops[arm] = len(sftp.listdir(d + "/episodes"))
            except OSError:
                loops[arm] = 0
        prep = {n: base._small_bytes(sftp, f"{ROOT}/results/prep/{n}") is not None
                for n in ("COMPLETE.json", "T5_COMPLETE.json", "reuse_check.json")}
        failed = {}
        for key, v in jobs.items():
            text = base._text(sftp, f"{ROOT}/artifacts/{v['job_id']}/FAILED", 4000)
            if text:
                failed[key] = text
    return dict(jobs={k: dict(v, **qstat(t, v["job_id"])) for k, v in jobs.items()}, prep=prep, quantized=quant,
                loops=loops, failed=failed)


def pull(t):
    target = HERE / "pulled"
    names = ["results/summary.json", "results/RESULTS.zh.md", "results/prep/modules.json",
             "results/prep/COMPLETE.json", "results/prep/T5_COMPLETE.json", "results/prep/reuse_check.json"]
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        for arm in P["arms"]:
            names += [f"results/quantized/{arm}/COMPLETE.json"]
            try:
                workers = [w for w in sftp.listdir(f"{ROOT}/results/loops/{arm}") if w.startswith("worker_")]
            except OSError:
                workers = []
            names += [f"results/loops/{arm}/{w}/{f}" for w in workers for f in ("startup.json", "summary.json")]
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
            code, out = run(t, "tail -c 200000 " + shlex.quote(f"{ROOT}/artifacts/{v['job_id']}/job.log") + " 2>/dev/null")
            if out:
                path = target / "artifacts" / v["job_id"] / "job.log.tail"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(out, encoding="utf-8")
    return dict(directory=str(target), files=found)


def cputest(t):
    job_id = qsub(t, "cabct", CPU_SELECT, CPU_WALLTIME, "cputest.pbs")
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        jobs["cputest"] = dict(job_id=job_id, after=[])
        save_jobs(sftp, jobs)
    return dict(job_id=job_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "cputest", "submit", "status", "pull", "logs"))
    parser.add_argument("--jobs", nargs="*")
    parser.add_argument("--after", nargs="*", default=[], help="extra recorded producer keys, e.g. cputest")
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
            result = submit(t, args.jobs or [*SINGLE["jobs"], "loop"], args.after)
        elif args.action == "status":
            result = status(t)
        elif args.action == "pull":
            result = pull(t)
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
