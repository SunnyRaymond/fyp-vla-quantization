"""Check training configuration and resume semantics on a CPU allocation."""
import os
from pathlib import Path
import subprocess
import sys

from control_common import guard, read, save


def main():
    guard()
    here = Path(__file__).resolve().parent
    protocol = read(here / 'protocol.json')
    subprocess.run([sys.executable, '-B', str(here / 'test_control_training.py')], check=True)
    save(Path(protocol['remote_root']) / 'results/training_runtime_selfcheck/summary.json', {
        'protocol': protocol['protocol'], 'unit': 'training_runtime_selfcheck',
        'pbs_jobid': os.environ['PBS_JOBID'], 'complete': True,
        'passed_checks': ['test_control_training.py'],
        'model_queries': 0, 'training_updates': 0,
        'scope': 'CPU toy config/gate binding and checkpoint-prefix checks; no actual model or optimizer training'})


if __name__ == '__main__':
    main()
