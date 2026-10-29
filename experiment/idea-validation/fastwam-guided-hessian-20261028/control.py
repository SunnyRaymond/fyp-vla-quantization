"""Controller over the bounded, host-key-verified PBS base (upload / submit / status / logs / pull): --phase hosts is
the g2 two-GPU job of host.json (run.pbs), --phase rows the g1 one-GPU job of rows.json (run_rows.pbs); attempt "" or
rN (resubmission, resumes from the files on disk)."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "fwgh_control_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
H = json.loads((HERE / "host.json").read_text(encoding="utf-8"))
ARMS = list(H["hosts"]["arms"])  # phase 3; phases 1 and 2 are in pulled/26043905.pbs101 and pulled/26046240.pbs101
SMALL = (["hosts/summary.json", "hosts/RESULTS.zh.md", "hosts/check/summary.json",
          "hosts/check/loop/results/episode.json", "hosts/check/loop/results/startup/startup.json"]
         + [p for arm in ARMS for p in (f"quantized/{arm}/COMPLETE.json", f"loops/{arm}/summary.json",
                                         f"loops/{arm}/RESULTS.zh.md")]
         + [f"heldout/{arm}.json" for arm in H["hosts"]["eval_arms"]]
         + [f"progress/fwgh{a}_{k}.json" for a in ("", "r3", "r4", "r5") for k in range(H["allocation"]["gpus"])])
ROWS = json.loads((HERE / "rows.json").read_text(encoding="utf-8"))
ROWS_SMALL = (["rows/summary.json", "rows/RESULTS.zh.md"]
              + [p.format(arm=ROWS["arm"]) for p in ("quantized/{arm}/COMPLETE.json", "heldout/{arm}.json",
                                                     "loops/{arm}/summary.json", "loops/{arm}/RESULTS.zh.md")]
              + [f"progress/fwgr{a}_0.json" for a in ("", "r1", "r2", "r3")])


def configure(attempt="", phase="hosts"):
    if not re.fullmatch(r"(r\d)?", attempt):
        raise ValueError(f"Unknown attempt: {attempt!r}")
    a = H["allocation"] if phase == "hosts" else ROWS["allocation"]
    base.HERE, base.PROTOCOL_ID, base.ROOT = HERE, P["protocol"], P["remote_root"]
    base.JOB_NAME = a["job_name"] + attempt
    base.JOB_NAME_RE = re.compile("^" + re.escape(base.JOB_NAME) + "$")
    base.ALLOCATION = {k: a[k] for k in ("queue", "execution_queue", "gpus", "ncpus", "memory_gb", "walltime")}
    base.QUEUE, base.EXECUTION_QUEUE, base.WALLTIME = a["queue"], a["execution_queue"], a["walltime"]
    base.SELECT = f"1:ncpus={a['ncpus']}:mem={a['memory_gb']}gb:ngpus={a['gpus']}"
    base.ARTIFACTS, base.RESULTS = base.ROOT + "/artifacts", base.ROOT + "/results"
    base.STATE_PATH = base.ROOT + f"/control/state{'' if phase == 'hosts' else '_' + phase}{attempt}.json"
    base.FILES = ("protocol.json", "run.pbs", "environment.sh", "worker.py", "probe.py", "host.json", "loop_check.py",
                  "rowcheck.py", "rowcheck.pbs", "test_probe.py", "test_hosts.py", "cputest.pbs",
                  "rows.json", "rows_worker.py", "run_rows.pbs", "test_rows.py")
    script = "run.pbs" if phase == "hosts" else "run_rows.pbs"
    base._qsub_command = lambda: (
        "qsub -N " + base.JOB_NAME + " -q " + base.QUEUE + " -P " + base.PROJECT + " -l select=" + shlex.quote(base.SELECT)
        + " -l walltime=" + base.WALLTIME + " -o " + shlex.quote(base.ARTIFACTS + "/" + base.JOB_NAME + ".pbs.out")
        + " -j oe " + shlex.quote(base.ROOT + "/" + script))

    def save_state(transport, state, _original=base._save_state):
        if state.get("handle"):
            state["handle"].update(gpus=a["gpus"], ncpus=a["ncpus"], memory_gb=a["memory_gb"])
        _original(transport, state)
    base._save_state = save_state
    return a


def load_protocol():
    value = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    if value.get("protocol") != base.PROTOCOL_ID or value.get("remote_root") != base.ROOT:
        raise RuntimeError("protocol.json identity differs")
    return value


def complete_gate(value):
    h, s, a = (value.get(k) or {} for k in ("handle", "scheduler", "artifact"))
    return (h.get("protocol") == P["protocol"] and h.get("job_name") == base.JOB_NAME
            and s.get("query_rc") == 0 and s.get("job_state") == "F" and s.get("Exit_status") == "0"
            and s.get("Job_Name") == base.JOB_NAME and s.get("queue") == base.EXECUTION_QUEUE
            and a.get("exit_code") == "0" and a.get("pipeline_complete") is True)


base.load_protocol, base.complete_gate = load_protocol, complete_gate


def status(transport, gpus):
    value = base.status(transport)
    if value.get("handle"):
        with base.remote.sftp_client(transport) as sftp:
            value["workers"] = {
                str(k): dict(progress=base._json(sftp, base.RESULTS + f"/progress/{base.JOB_NAME}_{k}.json"),
                             failure=base._json(sftp, base.RESULTS + f"/failures/{base.JOB_NAME}_{k}.json"))
                for k in range(gpus)}
    return value


def pull(transport, gpus, small=SMALL):
    value = status(transport, gpus)
    if not value.get("handle"):
        raise RuntimeError("No submitted job")
    target = HERE / "pulled" / value["handle"]["job_id"]
    files = []
    with base.remote.sftp_client(transport) as sftp:
        for name in small:
            data = base._small_bytes(sftp, base.RESULTS + "/" + name, limit=16 * 1024 * 1024)
            if data is not None:
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                files.append(name)
    (HERE / "monitor_status.json").write_text(
        json.dumps({k: v for k, v in value.items() if k != "log_tail"}, indent=2, default=str) + "\n",
        encoding="utf-8")
    return dict(directory=str(target), files=len(files), status=value.get("status"))


def cpu_submit(transport, script, name):
    """Plain qsub of a CPU-node script (2 cores, 16 GB; walltime > 2 h routes to q2 instead of the slow qdev)."""
    command = ("qsub -N " + name + " -q normal -P " + base.PROJECT + " -l select=1:ncpus=2:mem=16gb -l walltime=02:30:00"
               + " -o " + base.ARTIFACTS + f"/{name}.pbs.out -j oe " + P["remote_root"] + "/" + script)
    code, out = base.remote.command(transport, command, timeout=45)
    if code or not base.HANDLE.fullmatch(out.strip()):
        raise RuntimeError("qsub failed: " + out[-500:])
    return dict(job_id=out.strip())


def rowcheck_status(transport, job_id):
    scheduler = base._scheduler(transport, job_id)
    with base.remote.sftp_client(transport) as sftp:
        data = base._small_bytes(sftp, base.RESULTS + "/rowcheck/summary.json", limit=16 * 1024 * 1024)
    if data is not None:
        target = HERE / "pulled" / "rowcheck" / "summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        data = json.loads(data)["summary"]
    return dict(job_state=scheduler.get("job_state"), exit_status=scheduler.get("Exit_status"), summary=data)


def cputest_status(transport, job_id):
    scheduler = base._scheduler(transport, job_id)
    found = {}
    with base.remote.sftp_client(transport) as sftp:
        for name in ("test_probe", "test_hosts", "test_rows"):
            for ext in ("json", "log"):
                data = base._small_bytes(sftp, f"{base.RESULTS}/cputest/{job_id}/{name}.{ext}")
                if data is not None:
                    text = data.decode("utf-8", errors="replace")
                    found[f"{name}.{ext}"] = json.loads(text) if ext == "json" else text[-3000:]
        found["passed"] = base._small_bytes(sftp, f"{base.RESULTS}/cputest/{job_id}/PASSED") is not None
    return dict(job_state=scheduler.get("job_state"), exit_status=scheduler.get("Exit_status"), **found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="")
    parser.add_argument("--phase", choices=("hosts", "rows"), default="hosts")
    parser.add_argument("--id")
    parser.add_argument("action", choices=("upload", "submit", "status", "logs", "pull", "rowcheck", "rowcheck-status",
                                           "cputest", "cputest-status"))
    args = parser.parse_args()
    gpus = configure(args.attempt, args.phase)["gpus"]
    jump, transport = base.direct_connect()
    try:
        if args.action == "rowcheck":
            result = {"uploaded": base.upload(transport), **cpu_submit(transport, "rowcheck.pbs", "fwghrc")}
        elif args.action == "rowcheck-status":
            result = rowcheck_status(transport, args.id)
        elif args.action == "cputest":
            result = {"uploaded": base.upload(transport), **cpu_submit(transport, "cputest.pbs", "fwghct")}
        elif args.action == "cputest-status":
            result = cputest_status(transport, args.id)
        elif args.action == "upload":
            result = {"uploaded": base.upload(transport)}
        elif args.action == "submit":
            result = base.submit(transport)
        elif args.action == "logs":
            result = base.logs(transport)
        elif args.action == "pull":
            result = pull(transport, gpus, SMALL if args.phase == "hosts" else ROWS_SMALL)
        else:
            result = status(transport, gpus)
            result.pop("log_tail", None) if result.get("status") == "queued" else None
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
