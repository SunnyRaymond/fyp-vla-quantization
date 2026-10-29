"""Two-GPU g2 controller over the bounded, host-key-verified PBS base (upload / submit / status / logs / pull)."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "fwtqa4_control_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
A = P["allocation"]
SMALL = (["summary.json", "RESULTS.zh.md", "d1/raw.json", "gates/h_w16a4_first_endpoint.json"]
         + [f"arms/{arm}.json" for arm in P["arms"]]
         + [f"worker_{w}/{n}" for w in P["workers"] for n in ("progress.json", "startup.json", "summary.json",
                                                             "failure.json")])


def configure(attempt=""):
    base.HERE, base.PROTOCOL_ID, base.ROOT = HERE, P["protocol"], P["remote_root"]
    base.JOB_NAME = A["job_name"] + attempt
    base.JOB_NAME_RE = re.compile("^" + re.escape(base.JOB_NAME) + "$")
    base.ALLOCATION = {k: A[k] for k in ("queue", "execution_queue", "gpus", "ncpus", "memory_gb", "walltime")}
    base.QUEUE, base.EXECUTION_QUEUE = A["queue"], A["execution_queue"]
    base.SELECT = f"1:ncpus={A['ncpus']}:mem={A['memory_gb']}gb:ngpus={A['gpus']}"
    base.WALLTIME = A["walltime"]
    base.ARTIFACTS, base.RESULTS = base.ROOT + "/artifacts", base.ROOT + "/results"
    base.STATE_PATH = base.ROOT + f"/control/state{attempt}.json"
    base.FILES = ("protocol.json", "run.pbs", "environment.sh", "worker.py", "temporal_quant.py", "analysis.py")


def load_protocol():
    value = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    if value.get("protocol") != base.PROTOCOL_ID or value.get("remote_root") != base.ROOT:
        raise RuntimeError("protocol.json identity differs")
    return value


save_state_original = base._save_state


def save_state(transport, state):
    if state.get("handle"):
        state["handle"].update(gpus=A["gpus"], ncpus=A["ncpus"], memory_gb=A["memory_gb"])
    save_state_original(transport, state)


def complete_gate(value):
    h, s, a, d = (value.get(k) or {} for k in ("handle", "scheduler", "artifact", "summary"))
    return (h.get("protocol") == P["protocol"] and h.get("job_name") == base.JOB_NAME
            and s.get("query_rc") == 0 and s.get("job_state") == "F" and s.get("Exit_status") == "0"
            and s.get("Job_Name") == base.JOB_NAME and s.get("queue") == base.EXECUTION_QUEUE
            and a.get("exit_code") == "0" and a.get("pipeline_complete") is True
            and d.get("protocol") == P["protocol"] and d.get("complete") is True
            and d.get("arms_complete") == len(P["arms"]))


base.load_protocol, base.complete_gate, base._save_state = load_protocol, complete_gate, save_state


def status(transport):
    value = base.status(transport)
    if value.get("handle"):
        with base.remote.sftp_client(transport) as sftp:
            value["workers"] = {w: dict(progress=base._json(sftp, base.RESULTS + f"/worker_{w}/progress.json"),
                                        failure=base._json(sftp, base.RESULTS + f"/worker_{w}/failure.json"))
                                for w in P["workers"]}
    return value


def pull(transport):
    value = status(transport)
    if not value.get("handle"):
        raise RuntimeError("No submitted job")
    target = HERE / "pulled" / value["handle"]["job_id"]
    files = []
    with base.remote.sftp_client(transport) as sftp:
        for name in SMALL:
            data = base._small_bytes(sftp, base.RESULTS + "/" + name, limit=16 * 1024 * 1024)
            if data is not None:
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                files.append(name)
    (HERE / "monitor_status.json").write_text(
        json.dumps({k: v for k, v in value.items() if k != "log_tail"}, indent=2, default=str) + "\n",
        encoding="utf-8")
    return dict(directory=str(target), files=files, status=value.get("status"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="")
    parser.add_argument("action", choices=("upload", "submit", "status", "logs", "pull"))
    args = parser.parse_args()
    configure(args.attempt)
    jump, transport = base.direct_connect()
    try:
        if args.action == "upload":
            result = {"uploaded": base.upload(transport)}
        elif args.action == "submit":
            result = base.submit(transport)
        elif args.action == "logs":
            result = base.logs(transport)
        elif args.action == "pull":
            result = pull(transport)
        else:
            result = status(transport)
            result.pop("log_tail", None) if result.get("status") == "queued" else None
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
