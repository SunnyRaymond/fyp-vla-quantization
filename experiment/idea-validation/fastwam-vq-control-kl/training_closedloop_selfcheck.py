"""CPU checks of the new episode loop and original paired280 references."""
import os
from pathlib import Path
import subprocess
import sys

from control_common import guard, read, save


def main():
    guard()
    here = Path(__file__).resolve().parent
    parent = read(here / "protocol.json")
    checks = ["test_training_closedloop.py", "test_training_closedloop_aggregate.py"]
    for name in checks:
        subprocess.run([sys.executable, "-B", str(here / name)], check=True)
    from training_closedloop_aggregate import original_references
    rows, references = original_references(parent)
    from training_closedloop import load_manifest, validate_closedloop_protocol
    validate_closedloop_protocol(parent)
    _document, worker_rows = load_manifest(parent)
    if worker_rows != rows:
        raise ValueError("New worker and original loader do not read the same paired280 manifest")
    save(Path(parent["remote_root"]) / "results/training_closedloop_selfcheck/summary.json", {
        "protocol": parent["protocol"], "unit": "training_closedloop_selfcheck",
        "pbs_jobid": os.environ["PBS_JOBID"], "complete": True,
        "passed_checks": checks, "original_manifest_variants": len(rows),
        "reference_methods": len(references), "reference_episode_slots": 4200,
        "reference_successes": {arm: sum(row["success"] for row in records.values())
                                for arm, records in references.items()},
        "new_model_queries": 0, "training_updates": 0, "environment_steps": 0,
        "scope": "CPU fake-episode checks and read-only original paired280 reference loading; no new model or environment"})


if __name__ == "__main__":
    main()
