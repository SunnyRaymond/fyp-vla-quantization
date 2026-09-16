"""Execute an archived, explicit argument list inside one allocated GPU job."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    runner = spec['runner']
    if runner not in ('screen_runner.py', 'probe_runner.py', 'joint_fidelity.py'):
        raise ValueError(f'Unsupported runner: {runner}')
    substitutions = {'root': str(args.root), 'output': str(args.output)}
    command = [sys.executable, str(args.output / runner)]
    command += [part.format(**substitutions) for part in spec['arguments']]
    record = {'stage': spec['stage'], 'command': command,
              'started_epoch': time.time(), 'status': 'running'}
    record_path = args.output / 'stage_status.json'
    record_path.write_text(json.dumps(record, indent=2))
    result = subprocess.run(command, check=False)
    record.update(ended_epoch=time.time(), exit_code=result.returncode,
                  status='complete' if result.returncode == 0 else 'failed')
    record_path.write_text(json.dumps(record, indent=2))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
