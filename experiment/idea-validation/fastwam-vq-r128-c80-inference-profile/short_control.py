"""Use the existing bounded controller for the authorized <2h race submission."""
import importlib.util
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('inference_short_base', HERE/'control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
control.HERE = HERE/'short'
control.ROOT = '/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-inference-profile-20261023-short'
control.JOB_NAME = 'vpinf0004'
control.JOB_NAME_RE = re.compile(r'^vpinf0004$')
control.EXECUTION_QUEUE = 'gdev'
control.WALLTIME = '01:59:00'
control.ALLOCATION = {**control.ALLOCATION, 'execution_queue': 'gdev', 'walltime': control.WALLTIME}
control.ARTIFACTS = control.ROOT+'/artifacts'
control.RESULTS = control.ROOT+'/results'
control.STATE_PATH = control.ROOT+'/control/state.json'

if __name__ == '__main__':
    raise SystemExit(control.main())
