"""Two-stage controller (prep on gdev, train on g1) over the bounded, host-key-verified PBS base."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "fw1b_control_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
# protocol.json is frozen on the remote; later stages are declared here (control-side only).
P["stages"]["train_lh"] = {
    "job_name": "fw1blh",
    "allocation": {"queue": "normal", "execution_queue": "g1", "gpus": 1, "ncpus": 16, "memory_gb": 110,
                   "walltime": "05:00:00"},
    "work": "1b train plus per-layer learnable butterfly H (7x64 Givens angles, init identity after H128)",
}
for _arm, _work in (("h", "H coordinates"), ("g", "two-stage per-layer butterfly G (learn G on VQ error, refit)")):
    P["stages"]["c8_" + _arm] = {
        "job_name": "fw1bc8" + _arm,
        "allocation": {"queue": "normal", "execution_queue": "gdev", "gpus": 1, "ncpus": 16, "memory_gb": 110,
                       "walltime": "02:00:00"},
        "work": f"C8 decision run, {_work}: 8 A+Z+K updates (one C8 context each) -> held-out 40x2",
    }
P["stages"]["c8_hg"] = {
    "job_name": "fw1bc8hg",
    "allocation": {"queue": "normal", "execution_queue": "g2", "gpus": 2, "ncpus": 32, "memory_gb": 220,
                   "walltime": "04:00:00"},
    "work": "g2 racer for the gdev C8 jobs: c8_h on GPU0 and c8_g on GPU1 in one allocation",
}
STAGE = None
SMALL = {
    "prep": ["stages/prep/summary.json", "stages/prep/progress.json", "stages/prep/failure.json",
             "stages/prep/init_anchor.json", "fit_stats.json", "heldout/VQ0_prime.json"],
    "train": ["stages/train/summary.json", "stages/train/progress.json", "stages/train/failure.json",
              "stages/train/RESULTS.zh.md", "heldout/H_INT8VQ_A8_KL.json", "heldout/VQ0_prime.json",
              "reader_preparation/summary.json", "arms/reader_kl/training.json", "arms/reader_kl/summary.json",
              "arms/reader_kl/setup_checks.json", "arms/reader_kl/anchor.json", "export/manifest.json"],
    "train_lh": ["stages/train_lh/summary.json", "stages/train_lh/progress.json", "stages/train_lh/failure.json",
                 "stages/train_lh/RESULTS.zh.md", "heldout/LH_INT8VQ_A8_KL.json", "arms/learned_h/training.json",
                 "arms/learned_h/summary.json", "arms/learned_h/setup_checks.json", "arms/learned_h/anchor.json",
                 "arms/learned_h/angles.json", "export_lh/manifest.json"],
    "c8_h": ["stages/c8_h/summary.json", "stages/c8_h/progress.json", "stages/c8_h/failure.json",
             "stages/c8_h/RESULTS.zh.md", "heldout/C8_H.json", "arms/c8_h/training.json"],
    "c8_g": ["stages/c8_g/summary.json", "stages/c8_g/progress.json", "stages/c8_g/failure.json",
             "stages/c8_g/RESULTS.zh.md", "heldout/C8_G.json", "heldout/VQ0_G.json", "arms/c8_g/training.json",
             "fit_g_stats.json"],
}
LARGE = {"fit_g_stats.json": 16 << 20}  # per-layer proxy histories for 614 Linears
SMALL["c8_hg"] = ["stages/c8_hg/summary.json"] + SMALL["c8_h"] + SMALL["c8_g"]


def configure(stage, attempt=""):
    global STAGE
    STAGE = stage
    s, a = P["stages"][stage], P["stages"][stage]["allocation"]
    base.HERE, base.PROTOCOL_ID, base.ROOT = HERE, P["protocol"], P["remote_root"]
    base.JOB_NAME = s["job_name"] + attempt  # e.g. "r2" for a retry after a failed attempt
    base.JOB_NAME_RE = re.compile("^" + re.escape(base.JOB_NAME) + "$")
    base.ALLOCATION = dict(a)
    base.QUEUE, base.EXECUTION_QUEUE = a["queue"], a["execution_queue"]
    base.SELECT = f"1:ncpus={a['ncpus']}:mem={a['memory_gb']}gb:ngpus={a['gpus']}"
    base.WALLTIME = a["walltime"]
    base.ARTIFACTS = base.ROOT + "/artifacts"
    base.RESULTS = base.ROOT + "/results/stages/" + stage
    base.STATE_PATH = base.ROOT + f"/control/state_{stage}{attempt}.json"
    base.FILES = ("protocol.json", "run.pbs", "environment.sh", "worker.py")


def load_protocol():
    value = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    if value.get("protocol") != base.PROTOCOL_ID or value.get("remote_root") != base.ROOT:
        raise RuntimeError("protocol.json identity differs")
    return value


save_state_original = base._save_state


def save_state(transport, state):
    if state.get("handle"):
        a = P["stages"][STAGE]["allocation"]
        state["handle"].update(stage=STAGE, gpus=a["gpus"], ncpus=a["ncpus"], memory_gb=a["memory_gb"])
    save_state_original(transport, state)


def complete_gate(value):
    h, s, a, d = (value.get(k) or {} for k in ("handle", "scheduler", "artifact", "summary"))
    return (h.get("protocol") == P["protocol"] and h.get("job_name") == base.JOB_NAME
            and s.get("query_rc") == 0 and s.get("job_state") == "F" and s.get("Exit_status") == "0"
            and s.get("Job_Name") == base.JOB_NAME and s.get("queue") == base.EXECUTION_QUEUE
            and a.get("exit_code") == "0" and a.get("pipeline_complete") is True
            and d.get("protocol") == P["protocol"] and d.get("pbs_jobid") == h.get("job_id")
            and d.get("stage") == STAGE and d.get("complete") is True)


base.load_protocol, base._save_state, base.complete_gate = load_protocol, save_state, complete_gate


def pull(transport):
    value = base.status(transport)
    if not value.get("handle"):
        raise RuntimeError("No submitted job for stage " + STAGE)
    target = HERE / "pulled" / value["handle"]["job_id"]
    root = base.ROOT + "/results/"
    files = []
    with base.remote.sftp_client(transport) as sftp:
        for name in SMALL[STAGE]:
            data = base._small_bytes(sftp, root + name, LARGE.get(name, base.LIMIT))
            if data is not None:
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                files.append(name)
    (HERE / f"monitor_status_{STAGE}.json").write_text(
        json.dumps({k: v for k, v in value.items() if k != "log_tail"}, indent=2, default=str) + "\n",
        encoding="utf-8")
    return dict(directory=str(target), files=files, status=value.get("status"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(P["stages"]), required=True)
    parser.add_argument("--attempt", default="")
    parser.add_argument("action", choices=("upload", "submit", "status", "logs", "pull"))
    args = parser.parse_args()
    configure(args.stage, args.attempt)
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
            result = base.status(transport)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
