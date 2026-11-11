"""Controller of the Fast-WAM future swap: upload, submit, status, pull, logs (host-key-verified SSH).

  control.py upload
  control.py submit
  control.py status | pull | logs [--name job.log]
The job id lives in control/jobs.json on the cluster and in pulled/jobs.json; a live job is never submitted twice.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("swap_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
remote = base.remote
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
ROOT = P["remote_root"]
FILES = ("protocol.json", "run.pbs", "swap.py")
JOBS_PATH = ROOT + "/control/jobs.json"
HANDLE = re.compile(r"^\d+\.[A-Za-z0-9_.-]+$")


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


def upload(t):
    run(t, "mkdir -p " + " ".join(shlex.quote(ROOT + d) for d in ("/control", "/artifacts", "/results")))
    changed = []
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        if "swap" in jobs and live(qstat(t, jobs["swap"]["job_id"])):
            raise RuntimeError(f"Job {jobs['swap']['job_id']} is live; refusing to upload")
        for name in FILES:
            data = (HERE / name).read_bytes().replace(b"\r\n", b"\n")
            target = ROOT + "/" + name
            with sftp.file(target + ".tmp", "wb") as stream:
                stream.write(data)
            sftp.posix_rename(target + ".tmp", target)
            changed.append(name)
    run(t, "chmod +x " + shlex.quote(ROOT + "/run.pbs"))
    return changed


def submit(t):
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        for name in FILES:
            with sftp.file(ROOT + "/" + name, "rb") as stream:
                if stream.read() != (HERE / name).read_bytes().replace(b"\r\n", b"\n"):
                    raise RuntimeError(f"Remote {name} differs from local; upload first")
    if "swap" in jobs and live(qstat(t, jobs["swap"]["job_id"])):
        return dict(existing=jobs["swap"]["job_id"])
    j = P["job"]
    parts = ["qsub", "-N", j["name"], "-q", "normal", "-P", base.PROJECT, "-l", "select=" + j["select"],
             "-l", "walltime=" + j["walltime"], "-o", ROOT + "/artifacts/" + j["name"] + ".pbs.out", "-j", "oe"]
    code, out = run(t, " ".join(shlex.quote(x) for x in parts) + " " + shlex.quote(ROOT + "/run.pbs"))
    job_id = out.strip()
    if code or not HANDLE.fullmatch(job_id):
        raise RuntimeError(f"qsub failed: {out[-800:]}")
    jobs["swap"] = dict(job_id=job_id)
    with remote.sftp_client(t) as sftp:
        save_jobs(sftp, jobs)
    return dict(job_id=job_id)


def status(t):
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        out = dict(jobs={k: dict(v, **qstat(t, v["job_id"])) for k, v in jobs.items()}, failed={})
        units = {}
        for arm in P["arms"]:
            try:
                units[arm] = len(sftp.listdir(f"{ROOT}/results/units/{arm}"))
            except OSError:
                units[arm] = 0
        out["units"] = units
        out["complete"] = base._small_bytes(sftp, f"{ROOT}/results/COMPLETE.json") is not None
        for key, v in jobs.items():
            text = base._text(sftp, f"{ROOT}/artifacts/{v['job_id']}/FAILED", 4000)
            if text:
                out["failed"][key] = text
    return out


def pull(t):
    target = HERE / "pulled"
    names = [f"results/{f}" for f in ("summary.json", "RESULTS.zh.md", "COMPLETE.json")]
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        try:
            names += [f"results/{f}" for f in sftp.listdir(f"{ROOT}/results") if f.startswith("startup_")]
        except OSError:
            pass
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
            for log in ("job.log", "swap_worker.log"):
                _, text = run(t, "tail -c 200000 " + shlex.quote(f"{ROOT}/artifacts/{v['job_id']}/{log}") + " 2>/dev/null")
                if text:
                    path = target / "artifacts" / v["job_id"] / (log + ".tail")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text, encoding="utf-8")
    return dict(directory=str(target), files=found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "submit", "status", "pull", "logs"))
    parser.add_argument("--name", default="job.log")
    parser.add_argument("--bytes", type=int, default=6000)
    args = parser.parse_args()
    jump, t = base.direct_connect()
    try:
        if args.action == "upload":
            result = dict(uploaded=upload(t))
        elif args.action == "submit":
            result = submit(t)
        elif args.action == "status":
            result = status(t)
        elif args.action == "pull":
            result = pull(t)
        else:
            with remote.sftp_client(t) as sftp:
                job_id = load_jobs(sftp)["swap"]["job_id"]
            print(run(t, f"tail -c {int(args.bytes)} " + shlex.quote(f"{ROOT}/artifacts/{job_id}/{args.name}"))[1])
            return
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        t.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
