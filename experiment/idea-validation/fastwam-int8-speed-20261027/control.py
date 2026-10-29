"""Reuse the established bounded, host-key-verified PBS controller (single g1 GPU)."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "int8_speed_control_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
base.HERE, base.PROTOCOL_ID, base.ROOT = HERE, P["protocol"], P["remote_root"]
base.JOB_NAME = P["job_name"]
base.JOB_NAME_RE = re.compile("^" + re.escape(base.JOB_NAME) + "$")
base.ALLOCATION = P["allocation"]
base.QUEUE, base.EXECUTION_QUEUE = "normal", "g1"
base.SELECT, base.WALLTIME = "1:ncpus=16:mem=110gb:ngpus=1", "03:00:00"
base.ARTIFACTS, base.RESULTS = base.ROOT + "/artifacts", base.ROOT + "/results"
base.STATE_PATH = base.ROOT + "/control/state.json"
base.FILES = ("protocol.json", "run.pbs", "environment.sh", "bench_worker.py", "analysis.py")
save_state_original = base._save_state


def save_state(transport, state):
    if state.get("handle"):
        state["handle"].update(gpus=1, ncpus=16, memory_gb=110)
    save_state_original(transport, state)


def complete_gate(value):
    h, s, a, d = (value.get(k) or {} for k in ("handle", "scheduler", "artifact", "summary"))
    return (h.get("protocol") == P["protocol"] and h.get("job_name") == P["job_name"]
            and s.get("query_rc") == 0 and s.get("job_state") == "F"
            and s.get("Exit_status") == "0" and s.get("Job_Name") == P["job_name"]
            and s.get("Resource_List.walltime") == "03:00:00"
            and a.get("exit_code") == "0" and a.get("pipeline_complete") is True
            and d.get("protocol") == P["protocol"] and d.get("pbs_jobid") == h.get("job_id")
            and d.get("complete") is True and d.get("modes_complete") == len(P["modes"]))


base._save_state, base.complete_gate = save_state, complete_gate


def pull(transport):
    value = base.status(transport)
    if not value.get("handle"):
        raise RuntimeError("No submitted job")
    job = value["handle"]["job_id"]
    target = HERE / "pulled" / job
    # shapes.json can exceed the 256 KiB control limit; gemm/summary.json carries the unique shapes.
    names = ["results/summary.json", "results/RESULTS.zh.md"]
    names += [f"results/{mode}/summary.json" for mode in P["modes"]]
    names += [f"results/{mode}/samples.json" for mode in ("bf16_eager", "bf16_cache_graph", "q0_c_buffer")]
    names += [f"artifacts/{job}/{mode}.exit" for mode in P["modes"]]
    names += [f"artifacts/{job}/exit_code.txt"]
    skipped = []
    with base.remote.sftp_client(transport) as sftp:
        for name in names:
            try:
                data = base._small_bytes(sftp, base.ROOT + "/" + name)
            except RuntimeError:
                skipped.append(name)
                continue
            if data is not None:
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    (HERE / "monitor_status.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return dict(directory=str(target), skipped_oversize=skipped, status=base._compact(value))


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
            result = base.status(transport)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
