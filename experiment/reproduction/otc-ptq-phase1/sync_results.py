"""Retrieve only small control/results files; never transfer raw state tensors."""
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[2] / 'nscc-access'))
import aspire2a_shell as access
import paramiko

ROOT = '/scratch/users/ntu/yguo017/otc-ptq-phase1'
jobs = json.loads((BASE / 'jobs.json').read_text())
selected = set(sys.argv[1:])
jump, transport = access.connect()
try:
    with paramiko.SFTPClient.from_transport(transport) as sftp:
        for job in jobs:
            jid = job['id']
            if selected and jid not in selected:
                continue
            assert re.fullmatch(r'\d+\.pbs101', jid)
            channel = transport.open_session()
            channel.exec_command('qstat -xf ' + jid)
            output = channel.makefile().read().decode()
            errors = channel.makefile_stderr().read().decode()
            if channel.recv_exit_status():
                print(jid, 'query unavailable', errors.strip())
                continue
            dest = BASE / 'artifacts' / jid
            dest.mkdir(parents=True, exist_ok=True)
            (dest / 'qstat.txt').write_text(output)
            job['state'] = re.search(r'job_state = (\w)', output).group(1)
            match = re.search(r'resources_used.walltime = (\d+):(\d+):(\d+)', output)
            if match:
                h, m, sec = map(int, match.groups())
                job['allocated_walltime_seconds'] = h * 3600 + m * 60 + sec
            match = re.search(r'Exit_status = (-?\d+)', output)
            if match:
                job['exit_status'] = int(match.group(1))
            match = re.search(r'Resource_List.ngpus = (\d+)', output)
            job['ngpus'] = int(match.group(1)) if match else job.get('ngpus', 0)
            print(jid, job['stage'], job['state'], job.get('exit_status'), job.get('allocated_walltime_seconds'))
            if job['state'] != 'F':
                continue
            names = ['exit_code.txt', 'identity_check.json', 'job.log', 'manifest.json', 'runner.py']
            if job['stage'] == 'cpu_collected_replay':
                names += ['collected_replay_preflight.json', 'collected_replay_preflight.py']
            if job['stage'] in ('collect', 'score'):
                stem = 'states' if job['stage'] == 'collect' else 'score'
                names += [f"{stem}_task{job['task']}_episode{ep}.json" for ep in job['episodes']]
            for name in names:
                src = f'{ROOT}/artifacts/{jid}/{name}'
                try:
                    size = sftp.stat(src).st_size
                except FileNotFoundError:
                    continue
                limit = 524288 if name.endswith('.json') else 262144
                if size > limit:
                    print('not transferred: exceeds small-file cap', jid, name, size)
                    continue
                sftp.get(src, str(dest / name))
finally:
    (BASE / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    transport.close()
    jump.close()
