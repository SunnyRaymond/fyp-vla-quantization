"""Controller of the Fig. 3 export: upload, submit, status, pull (host-key-verified SSH).

  control.py upload
  control.py submit [--jobs maps frames]
  control.py status | pull
maps    CPU job: reader-metric maps (fig3_maps.py)          -> results/maps.json
frames  one-GPU job under 2 h: decoded BF16 futures         -> results/frames/<case>/*.png, results/frames/index.json
The two jobs are independent. Job ids live in control/jobs.json on the cluster and in pulled/jobs.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fig3_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
remote = base.remote
ROOT = "/scratch/users/ntu/yguo017/paper-fig3-export-20261112"
FILES = ("fig3_maps.py", "fig3_frames.py", "maps.pbs", "frames.pbs")
JOBS = {
    "maps": dict(name="fig3mp", script="maps.pbs", select="1:ncpus=2:mem=16gb", walltime="02:30:00"),
    "frames": dict(name="fig3fr", script="frames.pbs", select="1:ncpus=16:mem=110gb:ngpus=1", walltime="01:00:00"),
}
JOBS_PATH = ROOT + "/control/jobs.json"
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")


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
    for key in ("job_state", "Exit_status", "queue", "resources_used.walltime", "comment"):
        m = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", out, re.M)
        fields[key] = m.group(1).strip() if m else None
    return fields


def upload(t):
    run(t, "mkdir -p " + " ".join(shlex.quote(ROOT + d) for d in ("/control", "/artifacts", "/results")))
    with remote.sftp_client(t) as sftp:
        for name in FILES:
            data = (HERE / name).read_bytes().replace(b"\r\n", b"\n")
            with sftp.file(ROOT + "/" + name + ".tmp", "wb") as stream:
                stream.write(data)
            sftp.posix_rename(ROOT + "/" + name + ".tmp", ROOT + "/" + name)
    run(t, "chmod +x " + " ".join(shlex.quote(ROOT + "/" + n) for n in FILES if n.endswith(".pbs")))
    return list(FILES)


def submit(t, keys):
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
    submitted = {}
    for key in keys:
        if key in jobs and qstat(t, jobs[key]["job_id"])["job_state"] in ("Q", "H", "R", "B", "W", "E"):
            submitted[key] = dict(existing=jobs[key]["job_id"])
            continue
        j = JOBS[key]
        parts = ["qsub", "-N", j["name"], "-q", "normal", "-P", base.PROJECT, "-l", "select=" + j["select"],
                 "-l", "walltime=" + j["walltime"], "-o", ROOT + "/artifacts/" + j["name"] + ".pbs.out", "-j", "oe",
                 ROOT + "/" + j["script"]]
        code, out = run(t, " ".join(shlex.quote(x) for x in parts), timeout=60)
        job_id = out.strip()
        if code or not HANDLE.fullmatch(job_id):
            raise RuntimeError(f"qsub {key} failed: {out[-800:]}")
        jobs[key] = dict(job_id=job_id)
        submitted[key] = dict(job_id=job_id)
        with remote.sftp_client(t) as sftp:
            save_jobs(sftp, jobs)
    return submitted


def status(t):
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
    result = {}
    for key, job in jobs.items():
        state = qstat(t, job["job_id"])
        _, done = run(t, "test -e " + shlex.quote(f"{ROOT}/artifacts/{job['job_id']}/PIPELINE_COMPLETE") + " && echo yes || echo no")
        result[key] = dict(job_id=job["job_id"], complete=done.strip() == "yes", **state)
    return result


def pull(t):
    local = HERE / "pulled"
    _, listing = run(t, "cd " + shlex.quote(ROOT) + " && find results -type f \\( -name '*.png' -o -name '*.json' \\) -printf '%P\\n';"
                        " find artifacts -maxdepth 2 -type f \\( -name job.log -o -name exit_code.txt \\) -printf 'artifacts/%P\\n'",
                     timeout=60)
    names = []
    with remote.sftp_client(t) as sftp:
        for line in listing.splitlines():
            line = line.strip()
            if not line:
                continue
            source = ROOT + "/" + (line if line.startswith("artifacts/") else "results/" + line)
            if "/runtime/" in source:
                continue
            target = local / (line if line.startswith("artifacts/") else "results/" + line)
            target.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(source, str(target))
            names.append(line)
    return dict(files=len(names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "submit", "status", "pull"))
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    jump, t = base.direct_connect()
    try:
        if args.action == "upload":
            result = dict(uploaded=upload(t))
        elif args.action == "submit":
            result = submit(t, args.jobs or list(JOBS))
        elif args.action == "status":
            result = status(t)
        else:
            result = pull(t)
    finally:
        t.close()
        if jump:
            jump.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
