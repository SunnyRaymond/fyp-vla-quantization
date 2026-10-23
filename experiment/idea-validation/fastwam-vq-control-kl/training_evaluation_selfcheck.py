"""Allocation-only small checks; no model forward or training."""
import os
from pathlib import Path
import subprocess
import sys

from control_common import guard, read, save


def main():
    guard()
    here = Path(__file__).resolve().parent
    parent = read(here / "protocol.json")
    checks = ["test_trained_deployment.py", "test_training_evaluation.py"]
    for name in checks:
        subprocess.run([sys.executable, "-B", str(here / name)], check=True)
    save(Path(parent["remote_root"]) / "results/training_evaluation_selfcheck/summary.json", {
        "protocol": parent["protocol"], "unit": "training_evaluation_selfcheck",
        "pbs_jobid": os.environ["PBS_JOBID"], "complete": True,
        "passed_checks": checks, "new_model_queries": 0, "training_updates": 0,
        "scope": "CPU toy checkpoint installation, coverage, aggregation and input binding only"})


if __name__ == "__main__":
    main()
