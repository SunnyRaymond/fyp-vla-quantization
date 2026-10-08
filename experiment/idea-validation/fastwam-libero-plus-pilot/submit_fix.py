"""Submit the repaired environment and dependent GPU checks exactly once."""
import argparse
import faulthandler
import json
import os
import re
import sys
import threading
from pathlib import Path

print('CONTROL_START', flush=True)
faulthandler.dump_traceback_later(45, repeat=False)


def deadline():
    print('CONTROL_DEADLINE: inspect returned handles and PBS history before retry', flush=True)
    os._exit(124)


watchdog = threading.Timer(120, deadline)
watchdog.daemon = True
watchdog.start()
import remote

parser = argparse.ArgumentParser()
parser.add_argument('--jump', action='store_true')
parser.add_argument('--upload', action='store_true')
parser.add_argument('--attempt', type=int, default=1)
parser.add_argument('--native-first', action='store_true')
parser.add_argument('--cpu-only', action='store_true')
parser.add_argument('--manifest-only', action='store_true')
args = parser.parse_args()
if args.attempt < 1:
    raise ValueError('--attempt must be positive')
handles_name = 'repair_handles.json' if args.attempt == 1 else f'repair_handles_attempt{args.attempt}.json'
handles_path = Path(__file__).with_name(handles_name)
handles = json.loads(handles_path.read_text()) if handles_path.exists() else {}
jump = transport = None
try:
    print('CONNECTING_TRUSTED_ROUTE', flush=True)
    jump, transport = remote.connect() if args.jump else remote.direct_connect()
    print('CONNECTED', flush=True)
    if args.upload:
        with remote.sftp_client(transport) as sftp:
            for name in ('prepare_manifest.py', 'remote.py'):
                data = (remote.HERE / name).read_bytes()
                if len(data) > remote.LIMIT:
                    raise RuntimeError('Oversized control file')
                with sftp.file(remote.REMOTE + '/' + name, 'wb') as output:
                    output.write(data)
                print('UPLOADED', name, len(data), flush=True)
    phases = (
        ('prepare', 'fw_plus_prep_fix', 'native_setup' if args.native_first else None,
         '' if args.manifest_only else '-l walltime=01:00:00',
         'prepare_manifest_only.pbs' if args.manifest_only else 'prepare_env.pbs'),
        ('environment_check', 'fw_plus_env_fix', 'prepare', '', 'preflight_env.pbs'),
        ('preflight', 'fw_plus_gpu_fix', 'environment_check', '-v FW_MODE=preflight', 'run.pbs'),
        ('performance', 'fw_plus_perf_fix', 'preflight', '-v FW_MODE=performance', 'run.pbs'),
    )
    if args.cpu_only:
        phases = phases[:2]
    if args.native_first:
        phases = (('native_setup', 'fw_plus_wand_fix', None, '', 'setup_native_wand.pbs'),) + phases
    for label, name, parent, options, script in phases:
        if args.attempt > 1:
            name += str(args.attempt)
        if label in handles:
            print('REUSING_CONFIRMED_HANDLE', label, handles[label], flush=True)
            continue
        print('CHECKING_SUBMISSION_HISTORY', name, flush=True)
        code, output = remote.command(transport, 'qselect -x -u yguo017 -N ' + name, timeout=25)
        if code not in (0, 1):
            raise RuntimeError('PBS history lookup failed; refusing to submit')
        prior = output.split()
        if prior:
            if len(prior) != 1 or re.fullmatch(r'\d+\.pbs101', prior[0]) is None:
                raise RuntimeError('Ambiguous PBS submission history')
            handle = prior[0]
            print('RECOVERED_EXISTING_HANDLE', label, handle, flush=True)
        else:
            dependency = '-W depend=afterok:' + handles[parent] if parent else ''
            command = f'qsub -N {name} {dependency} {options} {remote.REMOTE}/{script}'
            print('SUBMITTING', label, flush=True)
            code, output = remote.command(transport, command, timeout=35)
            if code or re.fullmatch(r'\d+\.pbs101', output.strip()) is None:
                raise RuntimeError('Unknown/rejected qsub outcome; inspect PBS history')
            handle = output.strip()
        handles[label] = handle
        handles_path.write_text(json.dumps(handles, indent=2) + '\n')
        print('CONFIRMED_HANDLE', label, handle, flush=True)
    print('CHAIN_SUBMITTED', json.dumps(handles), flush=True)
finally:
    faulthandler.cancel_dump_traceback_later()
    watchdog.cancel()
    if transport:
        transport.close()
    if jump:
        jump.close()
