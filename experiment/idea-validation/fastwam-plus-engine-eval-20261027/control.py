"""Per-arm PBS control for the engine LIBERO-Plus evaluator, reusing the host-key-verified base controller."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "plus_engine_control_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
ARMS = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))


def configure(arm):
    cfg = ARMS[arm]
    a = {**P["allocation"], **cfg.get("allocation", {})}
    base.HERE, base.PROTOCOL_ID, base.ROOT = HERE, P["protocol"], P["remote_root"]
    base.JOB_NAME = cfg["job_name"]
    base.JOB_NAME_RE = re.compile("^" + re.escape(base.JOB_NAME) + "$")
    base.QUEUE, base.EXECUTION_QUEUE = a["queue"], a["execution_queue"]
    base.SELECT = f"1:ncpus={a['ncpus']}:mem={a['memory_gb']}gb:ngpus={a['gpus']}"
    base.WALLTIME = a["walltime"]
    base.ALLOCATION = {k: a[k] for k in ("queue", "execution_queue", "gpus", "ncpus", "memory_gb", "walltime")}
    base.ARTIFACTS = base.ROOT + "/artifacts"
    base.RESULTS = base.ROOT + "/results/" + arm
    base.STATE_PATH = base.ROOT + f"/control/state_{arm}.json"
    base.FILES = ("protocol.json", "arms.json", "run.pbs", "environment.sh", "worker.py",
                  "plus_engine.py", "analysis.py")
    save_state_original = base._save_state

    def load_protocol():
        if json.loads((HERE / "protocol.json").read_text(encoding="utf-8")) != P:
            raise RuntimeError("protocol.json changed while the controller was running")
        return P

    def qsub_command():
        return ("qsub -N " + base.JOB_NAME + " -q " + base.QUEUE + " -P " + base.PROJECT
                + " -l select=" + shlex.quote(base.SELECT) + " -l walltime=" + base.WALLTIME
                + " -v " + shlex.quote("FW_ARM=" + arm)
                + " -o " + shlex.quote(base.ARTIFACTS + "/" + base.JOB_NAME + ".pbs.out")
                + " -j oe " + shlex.quote(base.ROOT + "/run.pbs"))

    def save_state(transport, state):
        if state.get("handle"):
            state["handle"].update(arm=arm, gpus=a["gpus"], ncpus=a["ncpus"], memory_gb=a["memory_gb"])
        save_state_original(transport, state)

    def complete_gate(value):
        h, s, art, d = (value.get(k) or {} for k in ("handle", "scheduler", "artifact", "summary"))
        return (h.get("job_name") == base.JOB_NAME and s.get("query_rc") == 0 and s.get("job_state") == "F"
                and s.get("Exit_status") == "0" and art.get("exit_code") == "0"
                and art.get("pipeline_complete") is True and d.get("protocol") == P["protocol"]
                and d.get("arm") == arm and d.get("pbs_jobid") == h.get("job_id") and d.get("complete") is True
                and (not cfg.get("compare_history") or d.get("repro_passed") is True))

    base.load_protocol, base._qsub_command = load_protocol, qsub_command
    base._save_state, base.complete_gate = save_state, complete_gate
    return cfg


def status(transport, arm):
    value = base.status(transport)
    if value.get("handle"):
        a = {**P["allocation"], **ARMS[arm].get("allocation", {})}
        with base.remote.sftp_client(transport) as sftp:
            value["workers"] = [base._json(sftp, base.RESULTS + f"/worker_{i}/progress.json")
                                for i in range(a["gpus"] * a["workers_per_gpu"])]
            value["first_use_repro"] = base._json(
                sftp, base.ROOT + f"/results/{P['repro_arm']}/summary.json")
    return value


def pull(transport, arm):
    value = status(transport, arm)
    if not value.get("handle"):
        raise RuntimeError("No submitted job for " + arm)
    target = HERE / "pulled" / arm / value["handle"]["job_id"]
    names = [f"{arm}/summary.json", f"{arm}/RESULTS.zh.md",
             f"{P['repro_arm']}/summary.json", f"{P['repro_arm']}/RESULTS.zh.md"]
    group = ARMS[arm].get("group")
    if group:
        names += [f"{group}/summary.json", f"{group}/RESULTS.zh.md"]
    with base.remote.sftp_client(transport) as sftp:
        for name in names:
            data = base._small_bytes(sftp, base.ROOT + "/results/" + name)
            if data is not None:
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    (HERE / f"monitor_status_{arm}.json").write_text(json.dumps(value, indent=2, default=str) + "\n",
                                                     encoding="utf-8")
    return dict(directory=str(target), status=value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "submit", "status", "logs", "pull"))
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    args = parser.parse_args()
    configure(args.arm)
    jump, transport = base.direct_connect()
    try:
        if args.action == "upload":
            result = {"uploaded": base.upload(transport)}
        elif args.action == "submit":
            result = base.submit(transport)
        elif args.action == "logs":
            result = base.logs(transport)
        elif args.action == "pull":
            result = pull(transport, args.arm)
        else:
            result = status(transport, args.arm)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
