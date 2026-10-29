"""Controller for the A16 baselines: upload (incl. methods/), per-job submit from jobs.json, status, pull.

Reuses the host-key-verified base controller (fused-tuning -> loss-pair -> nscc-access helper).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "a16_control_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
JOBS = json.loads((HERE / "jobs.json").read_text(encoding="utf-8"))
FLAT = ("protocol.json", "jobs.json", "run.pbs", "environment.sh", "common.py", "prep_worker.py", "quant_worker.py",
        "loop_worker.py", "analysis.py", "cputest.pbs")
CPU_SELECT, CPU_WALLTIME = "1:ncpus=32:mem=128gb", "02:30:00"  # > 2 h routes to q2 instead of qdev
ROOT = P["remote_root"]


def configure(key, attempt=""):
    j = JOBS[key]
    base.HERE, base.PROTOCOL_ID, base.ROOT = HERE, P["protocol"], ROOT
    base.JOB_NAME = key + attempt
    base.JOB_NAME_RE = re.compile("^" + re.escape(base.JOB_NAME) + "$")
    base.QUEUE, base.EXECUTION_QUEUE = j["queue"], j["execution_queue"]
    base.SELECT = f"1:ncpus={j['ncpus']}:mem={j['memory_gb']}gb:ngpus={j['gpus']}"
    base.WALLTIME = j["walltime"]
    base.ARTIFACTS, base.RESULTS = ROOT + "/artifacts", ROOT + "/results"
    base.STATE_PATH = ROOT + f"/control/state_{base.JOB_NAME}.json"
    base.FILES = FLAT

    def load_protocol():
        if json.loads((HERE / "protocol.json").read_text(encoding="utf-8")) != P:
            raise RuntimeError("protocol.json changed while the controller was running")
        return P

    def save_state(transport, state, _original=base._save_state):
        if state.get("handle"):
            state["handle"].update(gpus=j["gpus"], ncpus=j["ncpus"], memory_gb=j["memory_gb"], plan=j["plan"])
        _original(transport, state)

    base.load_protocol, base._save_state = load_protocol, save_state


def method_files():
    return sorted(f for f in (HERE / "methods").rglob("*")
                  if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc")


def upload(transport):
    """Flat sources through the bounded base uploader (refuses protocol changes); methods/ tree via SFTP."""
    configure(next(iter(JOBS)), "_upload")
    done = base.upload(transport)
    files = method_files()
    dirs = sorted({ROOT + "/" + f.parent.relative_to(HERE).as_posix() for f in files})
    code, out = base.remote.command(transport, "mkdir -p " + " ".join(shlex.quote(d) for d in dirs), timeout=30)
    if code:
        raise RuntimeError("Cannot create method directories: " + out[:500])
    with base.remote.sftp_client(transport) as sftp:
        for f in files:
            target = ROOT + "/" + f.relative_to(HERE).as_posix()
            data = f.read_bytes()
            if f.suffix in (".py", ".md", ".json", ".txt"):
                data = data.replace(b"\r\n", b"\n")
            try:
                with sftp.file(target, "rb") as stream:
                    if stream.read() == data:
                        continue
            except OSError:
                pass
            with sftp.file(target + ".tmp", "wb") as stream:
                stream.write(data)
            sftp.posix_rename(target + ".tmp", target)
            done.append(f.relative_to(HERE).as_posix())
    return done


SMALL = ["prep/COMPLETE.json", "prep/t5_cache.json", "prep/shard_0.json", "prep/shard_1.json", "prep/shard_2.json",
         "prep/shard_3.json"]


def small_results(sftp):
    names = list(SMALL)
    for arm in P["arms"]:
        names += [f"quantized/{arm}/COMPLETE.json", f"loops/{arm}/summary.json", f"loops/{arm}/RESULTS.zh.md",
                  f"loops/{arm}/DONE", f"loops/{arm}/worker_0/startup.json"]
    found = {}
    for name in names:
        data = base._small_bytes(sftp, ROOT + "/results/" + name)
        if data is not None:
            found[name] = data
    return found


def status(transport, key):
    handle, _ = base._resolve(transport)
    value = dict(job=key, handle=handle, scheduler=None, log_tail=None)
    if handle:
        value["scheduler"] = base._scheduler(transport, handle["job_id"])
        code, tail = base.remote.command(
            transport, "tail -c 3000 " + shlex.quote(f"{ROOT}/artifacts/{handle['job_id']}/job.log") + " 2>/dev/null",
            timeout=20)
        value["log_tail"] = tail if code == 0 else None
        with base.remote.sftp_client(transport) as sftp:
            value["exit_code"] = base._text(sftp, f"{ROOT}/artifacts/{handle['job_id']}/exit_code.txt", 64)
    with base.remote.sftp_client(transport) as sftp:
        found = small_results(sftp)
    value["prep_complete"] = "prep/COMPLETE.json" in found
    value["quantized"] = {k.split("/")[1]: json.loads(v).get("bpw") for k, v in found.items()
                          if k.startswith("quantized/")}
    value["loops"] = {k.split("/")[1]: dict(successes=json.loads(v).get("successes"),
                                            episodes=json.loads(v).get("episodes"))
                      for k, v in found.items() if k.startswith("loops/") and k.endswith("summary.json")}
    value["loops_done"] = sorted(k.split("/")[1] for k in found if k.endswith("/DONE"))
    return value


def pull(transport):
    target = HERE / "pulled"
    with base.remote.sftp_client(transport) as sftp:
        found = small_results(sftp)
    for name, data in found.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return sorted(found)


def cputest_submit(transport, methods):
    """Plain qsub of cputest.pbs on a CPU node (tests only; no durable intent state)."""
    name = "a16ct"
    command = ("qsub -N " + name + " -q normal -P " + base.PROJECT + " -l select=" + CPU_SELECT
               + " -l walltime=" + CPU_WALLTIME + " -v " + shlex.quote("FW_METHODS=" + "+".join(methods))
               + " -o " + shlex.quote(ROOT + "/artifacts/" + name + ".pbs.out") + " -j oe "
               + shlex.quote(ROOT + "/cputest.pbs"))
    code, out = base.remote.command(transport, command, timeout=45)
    if code or not base.HANDLE.fullmatch(out.strip()):
        raise RuntimeError("qsub failed: " + out[-500:])
    return dict(job_id=out.strip(), methods=methods)


def cputest_status(transport, job_id):
    scheduler = base._scheduler(transport, job_id)
    found = {}
    with base.remote.sftp_client(transport) as sftp:
        for m in ("gptq", "gptvq", "quipsharp", "qtip"):
            for ext in ("json", "log"):
                data = base._small_bytes(sftp, f"{ROOT}/results/cputest/{job_id}/{m}.{ext}")
                if data is not None:
                    text = data.decode("utf-8", errors="replace")
                    found[f"{m}.{ext}"] = json.loads(text) if ext == "json" else text[-4000:]
    return dict(job_state=scheduler.get("job_state"), exit_status=scheduler.get("Exit_status"), **found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "submit", "status", "pull", "cputest", "cputest-status"))
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--id")
    parser.add_argument("--job", choices=sorted(JOBS))
    parser.add_argument("--attempt", default="")
    args = parser.parse_args()
    if args.action in ("submit", "status") and not args.job:
        parser.error("--job is required")
    jump, transport = base.direct_connect()
    try:
        if args.action == "upload":
            result = {"uploaded": upload(transport)}
        elif args.action == "submit":
            configure(args.job, args.attempt)
            result = base.submit(transport)
        elif args.action == "cputest":
            configure(next(iter(JOBS)), "_upload")
            result = {"uploaded": upload(transport), **cputest_submit(transport, args.methods)}
        elif args.action == "cputest-status":
            configure(next(iter(JOBS)), "_upload")
            result = cputest_status(transport, args.id)
        elif args.action == "status":
            configure(args.job, args.attempt)
            result = status(transport, args.job)
        else:
            result = {"pulled": pull(transport)}
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
