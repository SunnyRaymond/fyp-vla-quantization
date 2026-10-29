"""Reuse the established bounded, host-key-verified PBS controller for the single-GPU gdev 1a check."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "a8_accuracy_control_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
A = P["allocation"]
base.HERE, base.PROTOCOL_ID, base.ROOT = HERE, P["protocol"], P["remote_root"]
# Attempt r1 (26029093.pbs101) failed at startup on a worker bug; its handle stays in control/state.json.
ATTEMPT = "r2"
base.JOB_NAME = P["job_name"] + ATTEMPT
base.JOB_NAME_RE = re.compile("^" + re.escape(base.JOB_NAME) + "$")
base.ALLOCATION = {k: A[k] for k in ("queue", "execution_queue", "gpus", "ncpus", "memory_gb", "walltime")}
base.QUEUE, base.EXECUTION_QUEUE = A["queue"], A["execution_queue"]
base.SELECT = f"1:ncpus={A['ncpus']}:mem={A['memory_gb']}gb:ngpus={A['gpus']}"
base.WALLTIME = A["walltime"]
base.ARTIFACTS, base.RESULTS = base.ROOT + "/artifacts", base.ROOT + "/results"
base.STATE_PATH = base.ROOT + f"/control/state_{ATTEMPT}.json"
base.FILES = ("protocol.json", "run.pbs", "environment.sh", "worker.py", "analysis.py")
save_state_original = base._save_state


def save_state(transport, state):
    if state.get("handle"):
        state["handle"].update(gpus=A["gpus"], ncpus=A["ncpus"], memory_gb=A["memory_gb"])
    save_state_original(transport, state)


def complete_gate(value):
    h, s, a, d = (value.get(k) or {} for k in ("handle", "scheduler", "artifact", "summary"))
    return (h.get("protocol") == P["protocol"] and h.get("job_name") == base.JOB_NAME
            and s.get("query_rc") == 0 and s.get("job_state") == "F"
            and s.get("Exit_status") == "0" and s.get("Job_Name") == base.JOB_NAME
            and a.get("exit_code") == "0" and a.get("pipeline_complete") is True
            and d.get("protocol") == P["protocol"] and d.get("pbs_jobid") == h.get("job_id")
            and d.get("complete") is True and d.get("arms_complete") == 3
            and d.get("heldout_queries_per_arm") == 80 and d.get("new_teacher_queries") == 0)


base._save_state, base.complete_gate = save_state, complete_gate


def status(transport):
    value = base.status(transport)
    if value.get("handle"):
        with base.remote.sftp_client(transport) as sftp:
            value["workers"] = {0: dict(progress=base._json(sftp, base.RESULTS + "/worker_0/progress.json"),
                                        failure=base._json(sftp, base.RESULTS + "/worker_0/failure.json"))}
    return value


def pull(transport):
    value = status(transport)
    if not value.get("handle"):
        raise RuntimeError("No submitted job")
    target = HERE / "pulled" / value["handle"]["job_id"]
    names = ["summary.json", "RESULTS.zh.md", "weight_diagnostic.json"]
    names += [f"arms/{arm}.json" for arm in P["arms"]]
    names += [f"worker_0/{name}" for name in ("progress.json", "startup.json", "summary.json", "failure.json")]
    with base.remote.sftp_client(transport) as sftp:
        for name in names:
            data = base._small_bytes(sftp, base.RESULTS + "/" + name)
            if data is not None:
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    (HERE / "monitor_status.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return dict(directory=str(target), status=value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "submit", "status", "logs", "pull"))
    action = parser.parse_args().action
    jump, transport = base.direct_connect()
    try:
        if action == "upload":
            result = {"uploaded": base.upload(transport)}
        elif action == "submit":
            result = base.submit(transport)
        elif action == "logs":
            result = base.logs(transport)
        elif action == "pull":
            result = pull(transport)
        else:
            result = status(transport)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
