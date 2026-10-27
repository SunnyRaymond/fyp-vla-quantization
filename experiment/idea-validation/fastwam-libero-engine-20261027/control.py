"""Reuse the established bounded, host-key-verified PBS controller."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "libero_engine_control_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
base.HERE, base.PROTOCOL_ID, base.ROOT = HERE, P["protocol"], P["remote_root"]
base.JOB_NAME = P["job_name"]
base.JOB_NAME_RE = re.compile("^" + re.escape(base.JOB_NAME) + "$")
base.ALLOCATION = P["allocation"]
base.QUEUE, base.EXECUTION_QUEUE = "normal", "g3"
base.SELECT, base.WALLTIME = "1:ncpus=32:mem=220gb:ngpus=4", "06:00:00"
base.ARTIFACTS, base.RESULTS = base.ROOT + "/artifacts", base.ROOT + "/results"
base.STATE_PATH = base.ROOT + "/control/state.json"
base.FILES = ("protocol.json", "run.pbs", "environment.sh", "worker.py", "analysis.py")
save_state_original = base._save_state


def save_state(transport, state):
    if state.get("handle"):
        state["handle"].update(gpus=4, ncpus=32, memory_gb=220)
    save_state_original(transport, state)


def complete_gate(value):
    h, s, a, d = (value.get(k) or {} for k in ("handle", "scheduler", "artifact", "summary"))
    return (h.get("protocol") == P["protocol"] and h.get("job_name") == P["job_name"]
            and s.get("query_rc") == 0 and s.get("job_state") == "F"
            and s.get("Exit_status") == "0" and s.get("Job_Name") == P["job_name"]
            and s.get("Resource_List.walltime") == "06:00:00"
            and a.get("exit_code") == "0" and a.get("pipeline_complete") is True
            and d.get("protocol") == P["protocol"] and d.get("pbs_jobid") == h.get("job_id")
            and d.get("complete") is True and d.get("episodes") == 400
            and d.get("tasks") == 40 and d.get("engine_workers_passed") == 4
            and d.get("new_calibration_updates") == 0)


base._save_state, base.complete_gate = save_state, complete_gate


def status(transport):
    value = base.status(transport)
    if value.get("handle"):
        with base.remote.sftp_client(transport) as sftp:
            value["workers"] = [base._json(sftp, base.RESULTS + f"/worker_{i}/progress.json")
                                for i in range(4)]
    return value


def pull(transport):
    value = status(transport)
    if not value.get("handle"):
        raise RuntimeError("No submitted job")
    target = HERE / "pulled" / value["handle"]["job_id"]
    names = ["summary.json", "RESULTS.zh.md", "tasks.json", "task_queue.json"]
    names += [f"worker_{i}/{name}" for i in range(4)
              for name in ("progress.json", "startup.json", "summary.json")]
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
    parser.add_argument("action", choices=("upload", "submit", "status", "logs", "pull", "inspect"))
    action = parser.parse_args().action
    jump, transport = base.direct_connect()
    try:
        if action == "upload":
            result = {"uploaded": base.upload(transport)}
        elif action == "submit":
            result = base.submit(transport)
        elif action == "pull":
            result = pull(transport)
        elif action == "inspect":
            paths = [P["engine"]["bundle"], P["engine"]["anchor_tensor"],
                     P["engine"]["source_results"] + "/summary.json",
                     "/scratch/users/ntu/yguo017/fastwam-smoke/libero-config/config.yaml",
                     "/scratch/users/ntu/yguo017/fastwam-smoke/libero-compat/libero/__init__.py"]
            result = []
            with base.remote.sftp_client(transport) as sftp:
                for path in paths:
                    row = dict(path=path, bytes=sftp.stat(path).st_size)
                    if row["bytes"] < 18000:
                        row["text"] = sftp.file(path).read().decode()
                    result.append(row)
        else:
            result = status(transport)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
