"""Check the composed training loss on a CPU allocation before model use."""
import os
from pathlib import Path
import subprocess
import sys

from control_common import guard, read, save


def main():
    guard()
    here = Path(__file__).resolve().parent
    protocol = read(here / 'protocol.json')
    subprocess.run([sys.executable, '-B', str(here / 'test_training_objective.py')], check=True)
    save(Path(protocol['remote_root']) / 'results/training_loss_selfcheck/summary.json', {
        'protocol': protocol['protocol'], 'unit': 'training_loss_selfcheck',
        'pbs_jobid': os.environ['PBS_JOBID'], 'complete': True,
        'passed_checks': ['test_training_objective.py'],
        'model_queries': 0, 'training_updates': 0,
        'scope': 'CPU toy chain-rule and loss-reduction checks; no experiment coefficients frozen; actual D/r and C80 checks still required'})


if __name__ == '__main__':
    main()
