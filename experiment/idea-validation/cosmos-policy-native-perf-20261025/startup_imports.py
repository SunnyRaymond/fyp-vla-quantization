"""CPU-allocation import diagnosis; does not load checkpoints or run forwards."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import socket
import time

from prepare_sources import guard
from startup_trace import startup_trace

MODULES = ("torch", "cosmos_policy.experiments.robot.libero.run_libero_eval",
           "cosmos_policy.experiments.robot.cosmos_utils", "libero.libero.benchmark")


def run(out):
    guard()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("Import diagnosis must use a CPU-only allocation")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "startup_imports.json"
    result = {"format": "cosmos_startup_imports_v1", "status": "running",
              "pbs_job_id": os.environ.get("PBS_JOBID"), "host": socket.gethostname(),
              "created_utc": datetime.now(timezone.utc).isoformat(), "imports": [],
              "model_forwards": 0, "training_updates": 0, "closed_loop_episodes": 0}

    def save():
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(path)

    save()
    try:
        with startup_trace(out):
            for name in MODULES:
                result["currently_importing"] = name
                save()
                started = time.monotonic()
                importlib.import_module(name)
                result["imports"].append({"module": name,
                                           "seconds": time.monotonic() - started})
                save()
        result.update(status="complete", currently_importing=None)
        save()
        return result
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().out)), flush=True)
