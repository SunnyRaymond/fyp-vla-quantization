"""Small-file control only; all tensors and rendering stay in PBS allocations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shlex

HERE = Path(__file__).resolve().parent
ROOT = "/scratch/users/ntu/yguo017/fastwam-prism-visualization-20261012"
LIMIT = 256 * 1024
FILES = ("protocol.json", "collect.py", "test_collect.py", "plot.py", "delivery.py", "README.zh.md", "control.py", "environment.sh", "gpu.pbs", "plot.pbs", "weight_protocol.json", "weight_analysis.py", "weight_analysis_plot.py", "weight_analysis_gallery.py", "WEIGHT_ANALYSIS.zh.md", "weight.pbs")
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
spec = importlib.util.spec_from_file_location("prism_trusted_control", HERE.parent / "fastwam-rotation-baselines" / "control.py")
trusted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trusted)
remote = trusted.remote


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def command(value, timeout=35):
    return trusted.direct_native_command(None, value, timeout=timeout)


def upload(names):
    code, output = command("mkdir -p " + shlex.quote(ROOT))
    if code:
        raise RuntimeError(output)
    jump, transport = remote.direct_connect()
    try:
        with remote.sftp_client(transport) as sftp:
            for name in names:
                if name not in FILES:
                    raise ValueError("Not an approved task control file")
                data = (HERE / name).read_bytes().replace(b"\r\n", b"\n")
                if len(data) > LIMIT:
                    raise ValueError("Control file exceeds small-file limit")
                temporary = ROOT + "/" + name + ".uploading"
                with sftp.file(temporary, "wb") as stream:
                    stream.write(data)
                sftp.posix_rename(temporary, ROOT + "/" + name)
                print("UPLOADED", name, len(data))
    finally:
        transport.close()
        if jump:
            jump.close()


def current_job(phase):
    value = json.loads((HERE / (phase + "_handle.json")).read_text(encoding="utf-8"))
    job = value["job_id"]
    if not HANDLE.fullmatch(job):
        raise ValueError("Invalid recorded PBS handle")
    return job


def status(phase, job=None):
    job = job or current_job(phase)
    if not HANDLE.fullmatch(job):
        raise ValueError("Invalid PBS handle")
    code, output = command("qstat -xf " + shlex.quote(job))
    fields = {}
    for key in ("job_state", "Exit_status", "Job_Name", "queue", "comment", "exec_host", "resources_used.walltime"):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        if match:
            fields[key] = match.group(1).strip()
    # PBS wraps long comments in the middle of words.
    lines = output.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^\s*comment\s*=\s*(.*)$", line)
        if match:
            comment = match.group(1)
            for continuation in lines[index + 1:]:
                if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_.]*\s*=", continuation) or not continuation[:1].isspace():
                    break
                comment += continuation.strip()
            fields["comment"] = comment.strip()
            break
    result = {"job_id": job, "phase": phase, "query_rc": code, "observed_utc": datetime.now(timezone.utc).isoformat(), **fields}
    save(HERE / (phase + "_scheduler_snapshot.json"), result)
    print(json.dumps(result, ensure_ascii=False))
    if fields.get("job_state") in ("R", "F", "X"):
        _, tail = command("test ! -f " + shlex.quote(ROOT + "/artifacts/" + job + "/job.log") + " || tail -c 3500 " + shlex.quote(ROOT + "/artifacts/" + job + "/job.log"))
        print(tail)
    return result


def submit(phase, attempt):
    if not re.fullmatch(r"[a-z][a-z0-9]{0,5}", attempt):
        raise ValueError("Invalid attempt label")
    name = "fwpr1012" + {"capture": "g", "plot": "c", "weight": "w"}[phase] + attempt
    intent = HERE / (phase + "_intent_" + attempt + ".json")
    handle = HERE / (phase + "_handle.json")
    if intent.exists():
        receipt = json.loads(intent.read_text(encoding="utf-8"))
        if receipt.get("job_id"):
            save(handle, receipt)
            print("CONFIRMED_HANDLE", receipt["job_id"])
            return
        raise RuntimeError("Unresolved submission intent; inspect named PBS history before any retry")
    if handle.exists():
        old = status(phase)
        if old.get("query_rc") != 0 or old.get("job_state") not in ("F", "X"):
            raise RuntimeError("Previous handle is not confirmed terminal")
    dependency = current_job("capture") if phase == "plot" else None
    code, output = command("qselect -x -u yguo017 -N " + shlex.quote(name) + " 2>&1")
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError("Submission history unavailable")
    matches = output.split()
    if matches:
        raise RuntimeError("Named submission already exists; recover its handle instead of duplicating")
    receipt = {"status": "intent_before_qsub", "phase": phase, "attempt": attempt, "job_name": name,
               "remote_root": ROOT, "walltime": "02:00:00", "created_utc": datetime.now(timezone.utc).isoformat(),
               "dependency": dependency}
    save(intent, receipt)
    dep = "-W " + shlex.quote("depend=afterok:" + dependency) + " " if dependency else ""
    script = {"capture": "gpu.pbs", "plot": "plot.pbs", "weight": "weight.pbs"}[phase]
    code, output = command("qsub " + dep + "-N " + shlex.quote(name) + " " + shlex.quote(ROOT + "/" + script), timeout=40)
    if code or not HANDLE.fullmatch(output.strip()):
        receipt["status"] = "unknown_or_rejected"
        receipt["message"] = output[:1000]
        save(intent, receipt)
        raise RuntimeError("Unknown/rejected qsub outcome; do not automatically resubmit")
    receipt.update(job_id=output.strip(), status="confirmed")
    save(intent, receipt)
    save(handle, receipt)
    print("CONFIRMED_HANDLE", receipt["job_id"])


def pull(names):
    jump, transport = remote.direct_connect()
    try:
        with remote.sftp_client(transport) as sftp:
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Invalid artifact path")
                source = ROOT + "/" + name
                size = sftp.stat(source).st_size
                if size > LIMIT:
                    raise RuntimeError("Heavy transfer prohibited through login; leave full artifact on compute storage")
                destination = HERE / "retrieved" / Path(name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(source, str(destination))
                print("RETRIEVED", name, size)
    finally:
        transport.close()
        if jump:
            jump.close()


def verify(phase):
    job = current_job(phase)
    result = status(phase, job)
    prefix = ROOT + "/artifacts/" + job
    code, output = command("test -f " + shlex.quote(prefix + "/PIPELINE_COMPLETE") + " && test \"$(cat " + shlex.quote(prefix + "/exit_code.txt") + ")\" = 0")
    result["completion_marker_and_exit"] = code == 0
    save(HERE / (phase + "_terminal_evidence.json"), result)
    if result.get("job_state") not in ("F", "X") or result.get("Exit_status") != "0" or code:
        raise RuntimeError("Not confirmed complete")
    print("TERMINAL_SUCCESS", job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "submit", "status", "verify", "pull"))
    parser.add_argument("--phase", choices=("capture", "plot", "weight"), default="capture")
    parser.add_argument("--attempt", default="a")
    parser.add_argument("--files", nargs="+")
    args = parser.parse_args()
    if args.action == "upload": upload(args.files or FILES)
    elif args.action == "submit": submit(args.phase, args.attempt)
    elif args.action == "status": status(args.phase)
    elif args.action == "verify": verify(args.phase)
    else:
        if not args.files: parser.error("pull requires --files")
        pull(args.files)


if __name__ == "__main__":
    main()
