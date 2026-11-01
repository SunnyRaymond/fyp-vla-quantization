"""Controller for the 840-episode closed loop of the calibrated RAVQ arms: upload, submit (one g2 job), status, pull,
CPU tests.

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
    "r840_control_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
AL = P["allocation"]
FILES = ("protocol.json", "run.pbs", "slots.sh", "panel.py", "sched.py", "t5_merge.py", "gate.py", "panel_worker.py",
         "analysis_840.py", "cputest.pbs", "test_840.py")
ROOT = P["remote_root"]
GPUS, NCPUS, MEMORY_GB = AL["gpus"], AL["ncpus"], AL["memory_gb"]
CPU_SELECT, CPU_WALLTIME = "1:ncpus=4:mem=16gb", "02:30:00"  # > 2 h routes to a q queue instead of the often-full qdev


def configure(attempt=""):
    base.HERE, base.PROTOCOL_ID, base.ROOT = HERE, P["protocol"], ROOT
    base.JOB_NAME = AL["job_name"] + attempt
    base.JOB_NAME_RE = re.compile("^" + re.escape(base.JOB_NAME) + "$")
    base.QUEUE, base.EXECUTION_QUEUE = AL["queue"], AL["execution_queue"]
    base.SELECT, base.WALLTIME = AL["select"], AL["walltime"]
    base.ARTIFACTS, base.RESULTS = ROOT + "/artifacts", ROOT + "/results"
    base.STATE_PATH = ROOT + f"/control/state_{base.JOB_NAME}.json"
    base.FILES = FILES

    def load_protocol():
        if json.loads((HERE / "protocol.json").read_text(encoding="utf-8")) != P:
            raise RuntimeError("protocol.json changed while the controller was running")
        return P

    def save_state(transport, state, _original=base._save_state):
        if state.get("handle"):
            state["handle"].update(gpus=GPUS, ncpus=NCPUS, memory_gb=MEMORY_GB, select=AL["select"])
        _original(transport, state)

    base.load_protocol, base._save_state = load_protocol, save_state


def listdir(sftp, path):
    try:
        return sftp.listdir(path)
    except OSError:
        return []


def small_results(sftp, job_id=None):
    names = ["summary.json", "RESULTS.zh.md", "arms.json", "prep/t5_cache.json", "prep/T5_COMPLETE.json",
             "prep/T5_FAILED.json"]
    for arm in P["arm_order"]:
        names += [f"loops/{arm}/summary.json", f"loops/{arm}/DONE"]
        for w in listdir(sftp, f"{ROOT}/results/loops/{arm}"):
            if w.startswith("worker_"):
                names += [f"loops/{arm}/{w}/startup.json", f"loops/{arm}/{w}/summary.json",
                          f"loops/{arm}/{w}/install_check.json"]
    found = {}
    for name in names:
        data = base._small_bytes(sftp, ROOT + "/results/" + name, 1 << 20)
        if data is not None:
            found["results/" + name] = data
    if job_id:
        for name in ("FAILED", "exit_code.txt", "PIPELINE_COMPLETE"):
            data = base._small_bytes(sftp, f"{ROOT}/artifacts/{job_id}/{name}", 1 << 20)
            if data is not None:
                found["artifacts/" + name] = data
    return found


def status(transport):
    handle, _ = base._resolve(transport)
    value = dict(handle=handle, scheduler=None, log_tail=None)
    job_id = handle["job_id"] if handle else None
    if handle:
        value["scheduler"] = base._scheduler(transport, job_id)
        code, tail = base.remote.command(
            transport, "tail -c 3000 " + shlex.quote(f"{ROOT}/artifacts/{job_id}/job.log") + " 2>/dev/null", timeout=20)
        value["log_tail"] = tail if code == 0 else None
    with base.remote.sftp_client(transport) as sftp:
        value["arms"] = {arm: dict(total=280 * len(P["arms"][arm]["panels"]),
                                   claimed=len(listdir(sftp, f"{ROOT}/results/loops/{arm}/claims")),
                                   finished=len(listdir(sftp, f"{ROOT}/results/loops/{arm}/episodes")),
                                   done=base._small_bytes(sftp, f"{ROOT}/results/loops/{arm}/DONE") is not None)
                         for arm in P["arm_order"]}
        value["t5_complete"] = base._small_bytes(sftp, f"{ROOT}/results/prep/T5_COMPLETE.json") is not None
        if job_id:
            value["failed"] = base._text(sftp, f"{ROOT}/artifacts/{job_id}/FAILED", 4000)
            value["exit_code"] = base._text(sftp, f"{ROOT}/artifacts/{job_id}/exit_code.txt", 64)
            value["pipeline_complete"] = base._small_bytes(sftp, f"{ROOT}/artifacts/{job_id}/PIPELINE_COMPLETE") is not None
    return value


def pull(transport):
    handle, _ = base._resolve(transport)
    job_id = handle["job_id"] if handle else None
    target = HERE / "pulled" / (job_id or "latest")
    with base.remote.sftp_client(transport) as sftp:
        found = small_results(sftp, job_id)
    for name, data in found.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return dict(directory=str(target), files=len(found))


def logs(transport, job_id, name, nbytes):
    code, out = base.remote.command(
        transport, f"tail -c {int(nbytes)} " + shlex.quote(f"{ROOT}/artifacts/{job_id}/{name}") + " 2>&1", timeout=30)
    return out


def cputest_submit(transport):
    name = "fwr840ct"
    command = ("qsub -N " + name + " -q normal -P " + base.PROJECT + " -l select=" + CPU_SELECT
               + " -l walltime=" + CPU_WALLTIME + " -o " + shlex.quote(ROOT + "/artifacts/" + name + ".pbs.out")
               + " -j oe " + shlex.quote(ROOT + "/cputest.pbs"))
    code, out = base.remote.command(transport, command, timeout=45)
    if code or not base.HANDLE.fullmatch(out.strip()):
        raise RuntimeError("qsub failed: " + out[-500:])
    return dict(job_id=out.strip())


def cputest_status(transport, job_id):
    scheduler = base._scheduler(transport, job_id)
    with base.remote.sftp_client(transport) as sftp:
        result = base._json(sftp, f"{ROOT}/results/cputest/{job_id}/result.json")
        log = base._text(sftp, f"{ROOT}/results/cputest/{job_id}/test.log", 1 << 16)
    return dict(job_state=scheduler.get("job_state"), exit_status=scheduler.get("Exit_status"), result=result,
                log=(log or "")[-6000:])


def depend_on(job_id):
    """qsub with -W depend=afterok:<id>[:<id> ...]: the job is queued now and starts only after every listed job exits
    0 (a failed producer drops it)."""
    if not re.fullmatch(r"\d+\.pbs101(:\d+\.pbs101)*", job_id or ""):
        raise ValueError(f"Bad PBS job id for --after: {job_id!r}")
    plain = base._qsub_command
    base._qsub_command = lambda: plain().replace("qsub ", f"qsub -W depend=afterok:{job_id} ", 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "submit", "status", "pull", "logs", "cputest", "cputest-status"))
    parser.add_argument("--attempt", default="")
    parser.add_argument("--id")
    parser.add_argument("--after", help="submit with -W depend=afterok:<job id>")
    parser.add_argument("--name", default="job.log")
    parser.add_argument("--bytes", type=int, default=6000)
    args = parser.parse_args()
    jump, transport = base.direct_connect()
    try:
        configure(args.attempt)
        if args.action == "upload":
            result = {"uploaded": base.upload(transport)}
        elif args.action == "submit":
            if args.after:
                depend_on(args.after)
            result = base.submit(transport)
        elif args.action == "status":
            result = status(transport)
        elif args.action == "pull":
            result = pull(transport)
        elif args.action == "logs":
            print(logs(transport, args.id, args.name, args.bytes))
            return
        elif args.action == "cputest":
            configure("_cputest")
            result = {"uploaded": base.upload(transport), **cputest_submit(transport)}
        else:
            result = cputest_status(transport, args.id)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
