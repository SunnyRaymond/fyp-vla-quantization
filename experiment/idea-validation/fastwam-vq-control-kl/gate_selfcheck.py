"""Run the changed gate's small checks on an approved CPU allocation."""
import os
from pathlib import Path
import subprocess
import sys

from control_common import guard, read, save


def main():
    guard()
    here = Path(__file__).resolve().parent
    p = read(here / 'protocol.json')
    tests = ('test_parameter_gate.py', 'test_c80_gate.py', 'test_c80_gate_eval.py')
    for name in tests:
        subprocess.run([sys.executable, '-B', str(here / name)], check=True)
    save(Path(p['remote_root']) / 'results/gate_selfcheck/summary.json', {
        'protocol': p['protocol'], 'unit': 'gate_selfcheck',
        'pbs_jobid': os.environ['PBS_JOBID'], 'complete': True,
        'passed_checks': list(tests), 'model_queries': 0, 'training_updates': 0,
        'scope': 'small CPU checks only; GPU residency and actual-model parity remain unproven'})


if __name__ == '__main__':
    main()
