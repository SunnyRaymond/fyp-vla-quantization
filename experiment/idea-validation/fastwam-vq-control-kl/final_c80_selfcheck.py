"""Small allocation-only checks for the final-C80 evaluation entry."""
import os
from pathlib import Path
import subprocess
import sys

from control_common import guard, read, save


def main():
    guard()
    here = Path(__file__).resolve().parent
    parent = read(here / "protocol.json")
    checks = ["test_final_c80_eval.py", "test_final_c80_aggregate.py"]
    for name in checks:
        subprocess.run([sys.executable, "-B", str(here / name)], check=True)
    save(Path(parent["remote_root"]) / "results/final_c80_selfcheck/summary.json", {
        "protocol": parent["protocol"], "unit": "final_c80_selfcheck",
        "pbs_jobid": os.environ["PBS_JOBID"], "complete": True,
        "passed_checks": checks, "new_model_queries": 0, "training_updates": 0,
        "scope": "CPU toy binding, full-C80 coverage and common loss composition; no actual model evaluation"})


if __name__ == "__main__":
    main()
