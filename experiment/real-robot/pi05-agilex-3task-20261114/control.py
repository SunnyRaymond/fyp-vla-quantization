"""Controller for the senior's three-task AgileX pi0.5 (openpi, full fine-tuning) run on ASPIRE2A.

  control.py upload                  sources (refused while a recorded job is queued/running, unless --force)
  control.py upload-data             cabinet_lemon + circle HDF5 + meta + raw/manifest.json (sha256), resumable
  control.py submit --jobs prep train   CPU prep, then g3 training afterok on prep while prep is queued/running
  control.py submit --jobs deliver      one-GPU job: robot bundle release/agilex3_pi05 + self-test + SHA256SUMS
  control.py status | pull
  control.py logs --job train [--name train.log] [--bytes 6000]
  control.py readme                  upload release/README.md (robot-side instructions) into the bundle
  control.py fetch                   download the bundle into deliver/agilex3_pi05, every file checked against SHA256SUMS
                                     (login-node SFTP of ~12 GB: only with the user's go-ahead; resumes partial files)
cluster_grasp is read from the Fast-WAM upload (fastwam-agilex-3task-20261110/raw, sha256 manifest there).
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
CKPT = f"{ROOT}/checkpoints/{P['train']['config']}/{P['train']['exp_name']}"
FILES = ("protocol.json", "container.sh", "prep.pbs", "train.pbs", "agilex_openpi.py", "convert_openpi.py",
         "deliver.pbs", "release_tools.py", "release/agilex3_pi05.py")
JOBS_PATH = ROOT + "/control/jobs.json"
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
SCRIPTS = {"prep": "prep.pbs", "train": "train.pbs", "deliver": "deliver.pbs"}
AFTER = {"train": ["prep"], "deliver": ["train"]}
PREP_FILES = ("convert.json", "norm_stats.json", "PREP_COMPLETE")
DELIVER_FILES = ("robot_stats.json", "selftest.json", "selftest_check.json")
BUNDLE = P["release"]["bundle"]
LOCAL_BUNDLE = HERE / P["release"]["local"]
BLOCK = 8 << 20


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
    for run_name in P["data"]["upload"]:
        for f in sorted((src / run_name / "data/chunk-000").glob("episode_??????.hdf5")):
            rel = f"{run_name}/data/chunk-000/{f.name}"
            if f.stat().st_size == 0:
                skipped.append(dict(path=rel, reason="0 bytes"))
                continue
            files.append(dict(path=rel, size=f.stat().st_size, sha256=sha256(f)))
        extras += [f"{run_name}/meta/{n}" for n in ("info.json", "episodes.jsonl")]
    return src, dict(files=files, skipped=skipped, meta=extras)


def upload_data(t):
    t0 = time.time()
    src, man = manifest()
    print(f"manifest: {len(man['files'])} files, {sum(m['size'] for m in man['files']) / 2 ** 30:.2f} GiB, "
          f"skipped {len(man['skipped'])}, sha256 in {time.time() - t0:.0f} s", flush=True)
    raw = ROOT + "/raw"
    dirs = {posixpath.dirname(f"{raw}/{m['path']}") for m in man["files"]} | {
        posixpath.dirname(f"{raw}/{p}") for p in man["meta"]}
    run(t, "mkdir -p " + " ".join(shlex.quote(d) for d in sorted(dirs)))
    sent, kept, total = 0, 0, 0
    t0 = time.time()
    items = man["files"] + [dict(path=p, size=(src / p).stat().st_size) for p in man["meta"]]
    with remote.sftp_client(t) as sftp:
        for i, m in enumerate(items):
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
            print(f"[{i + 1}/{len(items)}] {m['path']} {m['size'] / 2 ** 20:.1f} MiB ({rate:.1f} MiB/s)", flush=True)
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
        if state["job_state"] is None:  # purged from PBS history; train.pbs itself requires PREP_COMPLETE
            continue
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
        for raw in P["data"]["remote_raw_roots"]:
            if base._small_bytes(sftp, raw + "/manifest.json") is None:
                raise RuntimeError(f"{raw}/manifest.json missing; run upload-data first")
    submitted = {}
    for k in keys:
        if k in jobs and qstat(t, jobs[k]["job_id"])["job_state"] in ("Q", "H", "R", "B", "W"):
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


def train_progress(t, job_id):
    art = f"{ROOT}/artifacts/{job_id}"
    # openpi's "Step N:" lines start with tqdm's carriage returns, so no ^ anchor.
    _, last = run(t, f"grep -ao 'Step [0-9]*:.*' {shlex.quote(art)}/train.log 2>/dev/null | tail -n 1")
    _, rate = run(t, f"grep -ao '[0-9.]*s/it\\|[0-9.]*it/s' {shlex.quote(art)}/train.log 2>/dev/null | tail -n 1")
    _, gpu = run(t, f"grep -a GPU_SAMPLE {shlex.quote(art)}/job.log 2>/dev/null | tail -n 4")
    _, ckpt = run(t, f"ls {shlex.quote(CKPT)} 2>/dev/null")
    util = [int(m) for m in re.findall(r",\s*(\d+) %,", gpu)]
    return dict(last=last.strip() or None, rate=rate.strip() or None, gpu_util=util,
                checkpoints=[c for c in ckpt.split() if c.isdigit()])


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
    stage = None
    if "prep" in jobs and states["prep"]["job_state"] == "R":
        _, stage = run(t, f"grep -a '^STAGE\\|^sha256\\|^cluster_grasp\\|^cabinet_lemon\\|^circle' "
                          f"{shlex.quote(ROOT)}/artifacts/{jobs['prep']['job_id']}/job.log 2>/dev/null | tail -n 2")
    train = train_progress(t, jobs["train"]["job_id"]) if "train" in jobs and states["train"]["job_state"] in (
        "R", "F") else None
    deliver = None
    if "deliver" in jobs and states["deliver"]["job_state"] in ("R", "F"):
        _, deliver = run(t, f"grep -a '^STAGE\\|^SELFTEST\\|^pooled\\|^reference\\|^DELIVER\\|^ERROR' "
                            f"{shlex.quote(ROOT)}/artifacts/{jobs['deliver']['job_id']}/job.log 2>/dev/null | tail -n 4")
    return dict(jobs=states, prep=prep, prep_stage=(stage or "").strip() or None, train=train,
                deliver=(deliver or "").strip() or None, failed=failed)


def pull(t):
    target = HERE / "pulled"
    names = [f"results/prep/{n}" for n in PREP_FILES] + [f"results/deliver/{n}" for n in DELIVER_FILES] + [
        "release/agilex3_pi05/selftest_reference.json", "release/agilex3_pi05/SHA256SUMS",
        "release/agilex3_pi05/env/versions_cluster.json"]
    with remote.sftp_client(t) as sftp:
        jobs = load_jobs(sftp)
        for v in jobs.values():
            names += [f"artifacts/{v['job_id']}/{f}" for f in ("exit_code.txt", "FAILED", "TRAIN_COMPLETE",
                                                               "DELIVER_COMPLETE")]
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
            for f in (e for e in entries if e.endswith(".log")):
                _, out = run(t, "tail -c 60000 " + shlex.quote(f"{d}/{f}") + " 2>/dev/null")
                if out:
                    path = target / "artifacts" / v["job_id"] / (f + ".tail")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(out, encoding="utf-8")
            if "train.log" in entries:
                _, steps = run(t, "grep -ao 'Step [0-9]*:.*' " + shlex.quote(f"{d}/train.log"))
                if steps:
                    (target / "loss_steps.txt").write_text(steps, encoding="utf-8")
    return dict(directory=str(target), files=found)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def fetch_one(sftp, rel, dst):
    size = sftp.stat(BUNDLE + "/" + rel).st_size
    have = dst.stat().st_size if dst.exists() else 0
    if have > size:
        dst.unlink()
        have = 0
    if have < size:
        s0 = time.time()
        with sftp.open(BUNDLE + "/" + rel, "rb") as src, dst.open("ab") as out:
            src.seek(have)
            src.prefetch(size)
            done = have
            while done < size:
                block = src.read(min(BLOCK, size - done))
                if not block:
                    raise IOError(f"short read on {rel} at {done}/{size}")
                out.write(block)
                done += len(block)
        mb = (size - have) / 2 ** 20
        if mb > 50:
            print(f"got {rel} {mb:.1f} MiB in {time.time() - s0:.0f}s ({mb / max(time.time() - s0, 1e-3):.1f} MiB/s)",
                  flush=True)
    return size


def readme(t):
    data = (HERE / "release" / "README.md").read_bytes().replace(b"\r\n", b"\n")
    with remote.sftp_client(t) as sftp:
        with sftp.file(BUNDLE + "/README.md.tmp", "wb") as f:
            f.write(data)
        sftp.posix_rename(BUNDLE + "/README.md.tmp", BUNDLE + "/README.md")
    return dict(uploaded=BUNDLE + "/README.md", bytes=len(data))


def fetch(t):
    t0, report = time.time(), []
    with remote.sftp_client(t) as sftp:
        with sftp.open(BUNDLE + "/SHA256SUMS", "rb") as f:
            sums = [line.split("  ", 1) for line in f.read().decode().splitlines() if line.strip()]
        LOCAL_BUNDLE.mkdir(parents=True, exist_ok=True)
        (LOCAL_BUNDLE / "SHA256SUMS").write_bytes("".join(f"{h}  {p}\n" for h, p in sums).encode())
        for digest, rel in sums:
            dst = LOCAL_BUNDLE / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            size = fetch_one(sftp, rel, dst)
            ok = file_sha256(dst) == digest
            report.append(dict(file=rel, bytes=size, sha256_ok=ok))
            if not ok:
                dst.unlink()
                raise RuntimeError(f"sha256 mismatch on {rel}; removed, rerun to fetch it again")
        readme_path = LOCAL_BUNDLE / "README.md"
        if readme_path.exists():
            readme_path.unlink()
        fetch_one(sftp, "README.md", readme_path)
    total = sum(r["bytes"] for r in report)
    return dict(local=str(LOCAL_BUNDLE), files=len(report), bytes=total, gib=round(total / 2 ** 30, 2),
                all_sha256_ok=all(r["sha256_ok"] for r in report), seconds=round(time.time() - t0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "upload-data", "submit", "status", "pull", "logs", "readme",
                                           "fetch"))
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
            if not args.jobs:
                parser.error("submit needs --jobs")
            result = submit(t, args.jobs)
        elif args.action == "status":
            result = status(t)
        elif args.action == "pull":
            result = pull(t)
        elif args.action == "readme":
            result = readme(t)
        elif args.action == "fetch":
            result = fetch(t)
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
