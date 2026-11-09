"""Controller for the AgileX BF16 Fast-WAM fine-tune on ASPIRE2A (host-key-verified SSH helper of the earlier runs).

  control.py upload                    sources + configs (refused while a recorded job is queued/running)
  control.py upload-data               the 78 non-empty episode HDF5 files + run meta + raw/manifest.json (sha256)
  control.py submit --jobs prep train  train is submitted afterok on prep
  control.py status | pull
  control.py logs --job train [--name train_bs8_ga1.log] [--bytes 6000]
  control.py race-openloop             open-loop check submitted twice (gdev <= 2 h, g1 > 2 h); the first to start
                                       cancels the other (in-job lock + poll_openloop.py)
Job ids live in control/jobs.json on the cluster (mirrored to pulled/jobs.json).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import posixpath
import re
import shlex
import time

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1].parent
spec = importlib.util.spec_from_file_location(
    "agilex_base", HERE.parents[1] / "idea-validation/fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
remote = base.remote
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
ROOT = P["remote_root"]
TASK = P["train"]["task"]
FILES = ("protocol.json", "environment.sh", "prep.pbs", "train.pbs", "memprobe.pbs", "openloop.pbs", "manifest.pbs", "fw_agilex.py",
         "train_agilex.py", "convert_agilex.py", "check_data.py", "check_model.py", "ds_zero2.json", "accelerate_zero2.yaml",
         "configs/data/agilex_2cam.yaml", "configs/task/agilex_optional_idm_2cam224.yaml", "release/agilex_policy.py",
         "release/openloop_check.py")
JOBS_PATH = ROOT + "/control/jobs.json"
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
SCRIPTS = {"prep": "prep.pbs", "train": "train.pbs", "memprobe": "memprobe.pbs", "openloop_gdev": "openloop.pbs",
           "openloop_g1": "openloop.pbs", "manifest": "manifest.pbs"}
RACE = ("openloop_gdev", "openloop_g1")
OPENLOOP_FILES = ("report.json", "selftest.json", "policy_config.json", "prompts.json", "selftest_reference.json",
                  "versions_cluster.json", "bundle_listing.txt", "OPENLOOP_COMPLETE", "SHA256SUMS", "step_delta.json")
AFTER = {"train": ["prep"]}
PREP_FILES = ("convert.json", "video.json", "backend.txt", "check_data.json", "check_model.json", "init_load.json",
              "composed_config.yaml", "PREP_COMPLETE")


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
    dirs = sorted({posixpath.dirname(ROOT + "/" + n) for n in FILES} | {ROOT + d for d in ("/control", "/artifacts")})
    run(t, "mkdir -p " + " ".join(shlex.quote(d) for d in dirs))
    changed = []
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        busy = active(t, jobs) if jobs else {}
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
                raise RuntimeError(f"{name} differs but jobs {sorted(busy)} are queued/running; refusing to upload")
            with sftp.file(target + ".tmp", "wb") as stream:
                stream.write(data)
            sftp.posix_rename(target + ".tmp", target)
            changed.append(name)
    run(t, "chmod +x " + " ".join(shlex.quote(ROOT + "/" + n) for n in FILES if n.endswith((".pbs", ".sh"))))
    return changed


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def manifest():
    src = PROJECT_ROOT / P["data"]["local_root"]
    files, skipped, extras = [], [], []
    for run_name in P["data"]["runs"]:
        for f in sorted((src / run_name / "data/chunk-000").glob("episode_??????.hdf5")):
            rel = f"{run_name}/data/chunk-000/{f.name}"
            if f.stat().st_size == 0:
                skipped.append(dict(path=rel, reason="0 bytes"))
                continue
            files.append(dict(path=rel, size=f.stat().st_size, sha256=sha256(f)))
        extras += [f"{run_name}/meta/{n}" for n in ("info.json", "episodes.jsonl")]
    return src, dict(files=files, skipped=skipped, meta=extras)


def upload_data(t):
    src, man = manifest()
    raw = ROOT + "/raw"
    dirs = {posixpath.dirname(f"{raw}/{m['path']}") for m in man["files"]} | {
        posixpath.dirname(f"{raw}/{p}") for p in man["meta"]}
    run(t, "mkdir -p " + " ".join(shlex.quote(d) for d in sorted(dirs)))
    sent, kept, total = 0, 0, 0
    t0 = time.time()
    with remote.sftp_client(t) as sftp:
        for i, m in enumerate(man["files"] + [dict(path=p, size=(src / p).stat().st_size) for p in man["meta"]]):
            target = f"{raw}/{m['path']}"
            try:
                if sftp.stat(target).st_size == m["size"]:
                    kept += 1
                    continue
            except OSError:
                pass
            sftp.put(str(src / m["path"]), target + ".tmp", confirm=True)
            sftp.posix_rename(target + ".tmp", target)
            sent += 1
            total += m["size"]
            rate = total / max(time.time() - t0, 1e-6) / 2 ** 20
            print(f"[{i + 1}/{len(man['files']) + len(man['meta'])}] {m['path']} {m['size'] / 2 ** 20:.1f} MiB "
                  f"({rate:.1f} MiB/s)", flush=True)
        with sftp.file(raw + "/manifest.json.tmp", "wb") as stream:
            stream.write((json.dumps(man, indent=2) + "\n").encode())
        sftp.posix_rename(raw + "/manifest.json.tmp", raw + "/manifest.json")
    (HERE / "pulled").mkdir(exist_ok=True)
    (HERE / "pulled" / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    return dict(files=len(man["files"]), sent=sent, already_there=kept, bytes_sent=total, skipped=man["skipped"],
                seconds=round(time.time() - t0, 1))


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


def submit(t, keys):
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        for name in FILES:
            data = (HERE / name).read_bytes().replace(b"\r\n", b"\n")
            with sftp.file(ROOT + "/" + name, "rb") as stream:
                if stream.read() != data:
                    raise RuntimeError(f"Remote {name} differs from local; upload first")
        if base._small_bytes(sftp, ROOT + "/raw/manifest.json") is None:
            raise RuntimeError("raw/manifest.json missing; run upload-data first")
    submitted = {}
    for k in keys:
        if k in jobs and qstat(t, jobs[k]["job_id"])["job_state"] in ("Q", "H", "R", "B", "W", None):
            submitted[k] = dict(existing=jobs[k]["job_id"])
            continue
        cfg = P["jobs"][k]
        after = producers(t, jobs, AFTER.get(k, ()))
        job_id = qsub(t, cfg["name"], cfg["select"], cfg["walltime"], SCRIPTS[k], after)
        jobs[k] = dict(job_id=job_id, after=after)
        submitted[k] = dict(job_id=job_id, after=after)
        with remote.sftp_client(t) as sftp:
            save_jobs(sftp, jobs)
    return submitted


def race_openloop(t):
    """Submit both open-loop copies, then record them in control/openloop_race.txt for the in-job lock."""
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        for name in FILES:
            data = (HERE / name).read_bytes().replace(b"\r\n", b"\n")
            with sftp.file(ROOT + "/" + name, "rb") as stream:
                if stream.read() != data:
                    raise RuntimeError(f"Remote {name} differs from local; upload first")
    for k in RACE:
        if k in jobs and qstat(t, jobs[k]["job_id"])["job_state"] in ("Q", "H", "R", "B", "W"):
            raise RuntimeError(f"{k} {jobs[k]['job_id']} is still queued/running")
    ids = []
    for k in RACE:
        cfg = P["jobs"][k]
        ids.append(qsub(t, cfg["name"], cfg["select"], cfg["walltime"], SCRIPTS[k]))
        jobs[k] = dict(job_id=ids[-1], after=[])
    with remote.sftp_client(t) as sftp:
        with sftp.file(ROOT + "/control/openloop_race.txt.tmp", "wb") as stream:
            stream.write((" ".join([ids[0]] + ids) + "\n").encode())
        sftp.posix_rename(ROOT + "/control/openloop_race.txt.tmp", ROOT + "/control/openloop_race.txt")
        save_jobs(sftp, jobs)
    return dict(zip(RACE, ids))


def train_progress(t, job_id):
    art = f"{ROOT}/artifacts/{job_id}"
    # rich renders the trainer's "[train]" prefix as markup, so match the step field instead
    _, last = run(t, f"grep -ah -A1 'epoch=[0-9]* step=' {shlex.quote(art)}/train_bs*_ga*.log 2>/dev/null | tail -n 2"
                     " | tr -s ' \\n' ' '")
    _, gpu = run(t, f"tail -n 4 {shlex.quote(art)}/gpu_samples.log 2>/dev/null | grep GPU_SAMPLE")
    _, ckpt = run(t, f"ls {shlex.quote(ROOT)}/runs/{TASK}/checkpoints/weights 2>/dev/null")
    _, spec_ = run(t, f"cat {shlex.quote(ROOT)}/runs/{TASK}/train_spec.txt 2>/dev/null")
    mem = [int(m) for m in re.findall(r",\s*(\d+) MiB,\s*40960 MiB", gpu)]
    return dict(last=last.strip() or None, gpu_mem_mib=mem, weights=ckpt.split(), spec=spec_.strip() or None)


def status(t):
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        prep = {n: base._small_bytes(sftp, f"{ROOT}/results/prep/{n}", 8 << 20) is not None for n in PREP_FILES}
        failed = {}
        for k, v in jobs.items():
            text = base._text(sftp, f"{ROOT}/artifacts/{v['job_id']}/FAILED", 4000)
            if text:
                failed[k] = text
    states = {k: dict(v, **qstat(t, v["job_id"])) for k, v in jobs.items()}
    train = train_progress(t, jobs["train"]["job_id"]) if "train" in jobs and states["train"]["job_state"] in (
        "R", "F") else None
    with remote.sftp_client(t) as sftp:
        openloop = {n: base._small_bytes(sftp, f"{ROOT}/results/openloop/{n}", 8 << 20) is not None
                    for n in OPENLOOP_FILES}
        lost = {k: base._small_bytes(sftp, f"{ROOT}/artifacts/{jobs[k]['job_id']}/RACE_LOST") is not None
                for k in RACE if k in jobs}
    return dict(jobs=states, prep=prep, train=train, failed=failed, openloop=openloop, race_lost=lost)


def pull(t):
    target = HERE / "pulled"
    names = [f"results/prep/{n}" for n in PREP_FILES] + ["data/dataset_stats.json", f"runs/{TASK}/train_spec.txt",
                                                       f"runs/{TASK}/init_load.json", f"runs/{TASK}/config.yaml"]
    names += [f"results/openloop/{n}" for n in OPENLOOP_FILES]
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        for v in jobs.values():
            names += [f"artifacts/{v['job_id']}/{f}" for f in ("exit_code.txt", "FAILED", "TRAIN_COMPLETE", "RACE_LOST")]
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
                entries = sftp.listdir(d)
            except OSError:
                entries = []
            logs = [f for f in entries if f.endswith(".log")]
            for probe in (f for f in entries if f.startswith("probe_") and not f.endswith(".log")):
                for f in sftp.listdir(f"{d}/{probe}"):
                    data = base._small_bytes(sftp, f"{d}/{probe}/{f}", 8 << 20)
                    if data is not None:
                        path = target / "artifacts" / v["job_id"] / probe / f
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                        found += 1
            for f in logs:
                _, out = run(t, "tail -c 60000 " + shlex.quote(f"{d}/{f}") + " 2>/dev/null")
                if out:
                    path = target / "artifacts" / v["job_id"] / (f + ".tail")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(out, encoding="utf-8")
    return dict(directory=str(target), files=found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "upload-data", "submit", "status", "pull", "logs", "race-openloop"))
    parser.add_argument("--jobs", nargs="*")
    parser.add_argument("--job")
    parser.add_argument("--name", default="job.log")
    parser.add_argument("--bytes", type=int, default=6000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    jump, t = base.direct_connect()
    try:
        if args.action == "upload":
            result = dict(uploaded=upload(t, args.force))
        elif args.action == "upload-data":
            result = upload_data(t)
        elif args.action == "submit":
            result = submit(t, args.jobs or ["prep", "train"])
        elif args.action == "status":
            result = status(t)
        elif args.action == "pull":
            result = pull(t)
        elif args.action == "race-openloop":
            result = race_openloop(t)
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
