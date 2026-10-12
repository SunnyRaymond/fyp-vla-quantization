"""Run the experiment's existing numerical self-checks in a CPU PBS allocation."""
import json
import os
from pathlib import Path

import run_experiment

run_experiment.guard()
import micro_codebook

micro_codebook.self_check(device='cpu')
artifact = Path(os.environ['ARTIFACTS'])
summary = {'status': 'complete', 'phase': 'cpu_preflight', 'job_id': os.environ['PBS_JOBID'],
           'protocol': 'fastwam-vq-microscale-v1', 'device': 'cpu',
           'checks': ['e0 decoder parity', 'tail shape', 'zero row fit', 'block scale squared metric'],
           'scope': 'existing toy assertions only; no fitted model bank or action evaluation',
           'walltime_limit': '00:10:00'}
run_experiment.save(artifact / 'summary.json', summary)
print('CPU_PREFLIGHT_COMPLETE', flush=True)
