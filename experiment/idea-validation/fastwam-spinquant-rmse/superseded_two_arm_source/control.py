"""Small-file control; model and action work only runs inside PBS."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
ROOT = '/scratch/users/ntu/yguo017/fastwam-spinquant-rmse-20261011'
FILES = ('protocol.json', 'control.py', 'environment.sh', 'gpu.pbs', 'run_rmse.py')
spec = importlib.util.spec_from_file_location('spin_rmse_trusted', HERE.parent / 'fastwam-vq-proxy-diagnostic/control.py')
trusted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trusted)
trusted.HERE, trusted.ROOT, trusted.FILES = HERE, ROOT, FILES
remote, save, scheduler, read_remote_json = trusted.remote, trusted.save, trusted.scheduler, trusted.read_remote_json


def submit(transport, attempt):
    if not re.fullmatch(r'[a-z][a-z0-9]{0,5}', attempt):
        raise ValueError('Invalid attempt label')
    trusted.protocol()
    name = f'fwspin1011{attempt}'
    intent = HERE / f'intent_{attempt}.json'
    handle = HERE / 'full_handle.json'
    if intent.exists():
        receipt = json.loads(intent.read_text())
        if receipt.get('job_id'):
            if receipt.get('job_name') != name or not trusted.HANDLE.fullmatch(receipt['job_id']):
                raise RuntimeError('Attempt receipt mismatch')
            save(handle, receipt)
            print('CONFIRMED_HANDLE', receipt['job_id'])
            return
    else:
        for path in HERE.glob('intent_*.json'):
            if not json.loads(path.read_text()).get('job_id'):
                raise RuntimeError('Earlier unresolved submission intent')
        if handle.exists():
            old = scheduler(transport, json.loads(handle.read_text())['job_id'])
            if old['query_rc'] != 0 or old['job_state'] not in ('F', 'X'):
                raise RuntimeError('Earlier attempt is not confirmed terminal')
        receipt = {'status': 'intent_before_qsub', 'job_name': name, 'attempt': attempt,
                   'remote_root': ROOT, 'walltime': '02:00:00', 'gpus': 1,
                   'created_utc': datetime.now(timezone.utc).isoformat()}
    rc, output = remote.command(transport, f'qselect -x -u yguo017 -N {shlex.quote(name)} 2>&1', timeout=35)
    matches = output.split()
    if rc not in (0, 1) or any(not trusted.HANDLE.fullmatch(x) for x in matches) or len(matches) > 1:
        raise RuntimeError('Ambiguous named submission history')
    if matches:
        receipt.update(status='recovered', job_id=matches[0])
    elif intent.exists():
        raise RuntimeError('Existing submission outcome is unknown; refusing blind retry')
    else:
        save(intent, receipt)
        rc, output = remote.command(transport, f'qsub -N {shlex.quote(name)} {shlex.quote(ROOT + "/gpu.pbs")}', timeout=40)
        if rc or not trusted.HANDLE.fullmatch(output.strip()):
            receipt.update(status='unknown_or_rejected')
            save(intent, receipt)
            raise RuntimeError('qsub outcome unknown/rejected; inspect before any retry')
        receipt.update(status='confirmed', job_id=output.strip())
    save(intent, receipt)
    save(handle, receipt)
    print('CONFIRMED_HANDLE', receipt['job_id'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('upload', 'submit', 'status', 'verify', 'pull'))
    parser.add_argument('--attempt', default='a')
    parser.add_argument('--job')
    parser.add_argument('--files', nargs='+')
    args = parser.parse_args()
    jump = transport = None
    try:
        jump, transport = remote.direct_connect()
        if args.action == 'upload':
            trusted.protocol()
            trusted.upload(transport, args.files)
        elif args.action == 'submit':
            submit(transport, args.attempt)
        elif args.action == 'pull':
            if not args.files:
                raise ValueError('--files required for pull')
            trusted.pull(transport, args.files)
        else:
            job = args.job or json.loads((HERE / 'full_handle.json').read_text())['job_id']
            observed = scheduler(transport, job)
            save(HERE / 'last_scheduler_snapshot.json', observed)
            print(json.dumps(observed))
            if args.action == 'verify':
                value = read_remote_json(transport, f'{ROOT}/artifacts/{job}/summary.json')
                marker = 'PIPELINE_COMPLETE' if value['status'] == 'complete' else 'PIPELINE_CHECKPOINTED'
                rc, _ = remote.command(transport,
                    f'test -f {shlex.quote(ROOT + "/artifacts/" + job + "/" + marker)} && '
                    f'test "$(cat {shlex.quote(ROOT + "/artifacts/" + job + "/exit_code.txt")})" = 0', timeout=25)
                if (rc or observed['job_state'] not in ('F', 'X') or observed['Exit_status'] != '0'
                        or value['job_id'] != job or value['protocol'] != 'fastwam-spinquant-rmse-v1'
                        or value['status'] not in ('complete', 'checkpointed')):
                    raise RuntimeError('Unverified terminal attempt')
                if value['status'] == 'complete':
                    if value['new_queries'] != 104 or value['target_linears'] != 614:
                        raise RuntimeError('Missing complete query/module coverage')
                    for arm in ('spinquant_decoded_bf16_w4a4', 'spinquant_native_w4a4'):
                        item = value['arms'][arm]
                        if (item['selection']['case_count'], item['selection']['context_count'],
                            item['test']['case_count'], item['test']['context_count']) != (4, 8, 22, 44):
                            raise RuntimeError('Missing paired cohort coverage')
                save(HERE / 'terminal_evidence.json', {'scheduler': observed, 'summary': value})
                print('VERIFIED', value['status'])
            elif observed['job_state'] in ('R', 'F', 'X'):
                rc, output = remote.command(transport, f'tail -c 5000 {shlex.quote(ROOT + "/artifacts/" + job + "/job.log")}', timeout=25)
                print(output if rc == 0 else 'Job log not yet available')
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == '__main__':
    main()
