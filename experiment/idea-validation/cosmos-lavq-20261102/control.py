"""Controller for LAVQ on Cosmos-Policy: upload, cputest, submit (the g2 pipeline job, then the g3 loop job afterok),
status, pull, logs. Uses the host-key-verified SSH helper of the earlier controllers.

  control.py upload
  control.py cputest
  control.py submit --jobs pipeline loop smoke [--after cputest]
  control.py status | pull | logs --job pipeline [--name job.log]
Job ids live in control/jobs.json on the cluster (and pulled/jobs.json here); a job already recorded and not finished
is never submitted twice, and uploads are refused while a recorded job is queued, held or running.
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
spec = importlib.util.spec_from_file_location("cosmos_lavq_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
remote = base.remote
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
ROOT = P["remote_root"]
FILES = ("protocol.json", "environment.sh", "slots.sh", "run.pbs", "run_loop.pbs", "cputest.pbs", "common.py",
         "cosmos_lavq.py", "qmath.py", "ravq_qtip.py", "softbin_loss.py", "lavq_worker.py", "loop_worker.py",
         "sched.py", "analysis.py", "test_lavq.py")
JOBS_PATH = ROOT + "/control/jobs.json"
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
SCRIPTS = {"cputest": "cputest.pbs", "pipeline": "run.pbs", "smoke": "run.pbs", "loop": "run_loop.pbs"}
AFTER = {"loop": ["pipeline"]}


def sources():
    names = list(FILES)
    names += sorted(str(f.relative_to(HERE)).replace("\\", "/") for f in (HERE / "methods").rglob("*")
                    if f.is_file() and "__pycache__" not in f.parts)
    return names


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
    code, out = run(t, "qstat -xf " + shlex.quote(job_id), timeout=40)
    fields = {}
    for k in ("job_state", "Exit_status", "queue", "resources_used.walltime", "Resource_List.walltime", "comment",
              "stime"):
        m = re.search(r"^\s*" + re.escape(k) + r"\s*=\s*(.*)$", out, re.M)
        fields[k] = m.group(1).strip() if m else None
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


def qsub(t, name, select, walltime, script, after=()):
    parts = ["qsub", "-N", name, "-q", "normal", "-P", base.PROJECT, "-l", "select=" + select,
             "-l", "walltime=" + walltime, "-o", ROOT + "/artifacts/" + name + ".pbs.out", "-j", "oe"]
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
    for k in keys:
        if k not in jobs:
            continue
        state = qstat(t, jobs[k]["job_id"])
        if state["job_state"] != "F":
            after.append(jobs[k]["job_id"])
        elif state["Exit_status"] != "0":
            raise RuntimeError(f"Producer {k} {jobs[k]['job_id']} finished with exit {state['Exit_status']}")
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
    for k in keys:
        if k in jobs and qstat(t, jobs[k]["job_id"])["job_state"] in ("Q", "H", "R", "B", "W", None):
            submitted[k] = dict(existing=jobs[k]["job_id"])
            continue
        cfg = P["jobs"][k]
        after = producers(t, jobs, [*AFTER.get(k, ()), *extra_after])
        job_id = qsub(t, cfg["name"], cfg["select"], cfg["walltime"], SCRIPTS[k], after)
        jobs[k] = dict(job_id=job_id, after=after)
        submitted[k] = dict(job_id=job_id, after=after)
        with remote.sftp_client(t) as sftp:
            save_jobs(sftp, jobs)
    return submitted


STAGE_FILES = ("quantized/QTIP_W1/COMPLETE.json", "quantized/QTIP_W4/COMPLETE.json", "teacher/summary.json",
               "metric/summary.json", "alloc/plan.json", "alloc/chosen.json", "heldout/summary.json",
               "calib/LAVQ_W2/arm.json", "calib/LAVQ_W3/arm.json", "DECISION.json", "loop_summary.json")


def status(t):
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        stages = {n: base._small_bytes(sftp, f"{ROOT}/results/{n}") is not None for n in STAGE_FILES}
        quant = {}
        for arm in ("QTIP_W1", "QTIP_W4"):
            try:
                quant[arm] = len(sftp.listdir(f"{ROOT}/results/quantized/{arm}/stats"))
            except OSError:
                quant[arm] = 0
        loops = {}
        decision = base._small_bytes(sftp, f"{ROOT}/results/DECISION.json")
        for arm in (json.loads(decision)["loop_arms"] if decision else []):
            try:
                loops[arm] = len(sftp.listdir(f"{ROOT}/results/loops/{arm}/episodes"))
            except OSError:
                loops[arm] = 0
        failed = {}
        for k, v in jobs.items():
            text = base._text(sftp, f"{ROOT}/artifacts/{v['job_id']}/FAILED", 4000)
            if text:
                failed[k] = text
    return dict(jobs={k: dict(v, **qstat(t, v["job_id"])) for k, v in jobs.items()}, prep=stages, quantized=quant,
                loops=loops, failed=failed)


def pull(t):
    target = HERE / "pulled"
    names = [f"results/{n}" for n in STAGE_FILES] + [
        "results/summary.json", "results/RESULTS.zh.md", "results/LOOP_RESULTS.zh.md", "results/teacher/anchor_parity.json",
        "results/quantized/LAVQU_W2/COMPLETE.json", "results/quantized/LAVQU_W3/COMPLETE.json",
        "results/quantized/LAVQ_W2/COMPLETE.json", "results/quantized/LAVQ_W3/COMPLETE.json",
        "smoke/RESULTS.zh.md", "smoke/summary.json", "smoke/DECISION.json"]
    for b in ("2", "3"):
        names += [f"results/calib/LAVQ_W{b}/{f}" for f in ("step.json", "training.json")]
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        for v in jobs.values():
            names += [f"artifacts/{v['job_id']}/{f}" for f in ("exit_code.txt", "FAILED", "PIPELINE_COMPLETE")]
        found = 0
        for name in names:
            data = base._small_bytes(sftp, ROOT + "/" + name, 8 << 20)
            if data is not None:
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                found += 1
        for v in jobs.values():
            d = f"{ROOT}/artifacts/{v['job_id']}"
            try:
                logs = [f for f in sftp.listdir(d) if f.endswith(".log")]
            except OSError:
                logs = []
            for f in logs:
                code, out = run(t, "tail -c 60000 " + shlex.quote(f"{d}/{f}") + " 2>/dev/null")
                if out:
                    path = target / "artifacts" / v["job_id"] / (f + ".tail")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(out, encoding="utf-8")
    return dict(directory=str(target), files=found)


def cputest(t):
    cfg = P["jobs"]["cputest"]
    job_id = qsub(t, cfg["name"], cfg["select"], cfg["walltime"], "cputest.pbs")
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
            result = submit(t, args.jobs or ["pipeline", "loop"], args.after)
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
