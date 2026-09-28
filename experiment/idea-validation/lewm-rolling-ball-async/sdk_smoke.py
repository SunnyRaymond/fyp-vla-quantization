"""Bounded version of Isaac Lab v2.3.1 create_empty.py; run via run_rented.sh."""
import argparse
import json
import os
import platform
import socket
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--expected-host', required=True)
parser.add_argument('--output', type=Path, required=True)
known, _ = parser.parse_known_args()
if platform.system() != 'Linux' or socket.gethostname() != known.expected_host or os.environ.get('PBS_JOBID'):
    raise SystemExit('Rented-host guard failed')

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
application = AppLauncher(args).app
try:
    from isaaclab.sim import SimulationCfg, SimulationContext
    sim = SimulationContext(SimulationCfg(dt=0.01))
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    sim.reset()
    for _ in range(10):
        sim.step()
    report = {'status': 'PASS_SDK_EMPTY_SCENE_10_STEPS', 'host': socket.gethostname(),
              'physics_steps': 10, 'physics_dt_s': 0.01, 'task_rgb_verified': False,
              'closed_loop_evaluated': False}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
finally:
    # The SDK's full stage teardown hangs on this container; all smoke work is saved.
    application.close(skip_cleanup=True)
