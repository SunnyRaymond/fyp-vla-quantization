"""Bounded submission and retrieval for one cached-record CPU analysis."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('crosscheck_source', HERE.parent/'fastwam-vq-dual-kl/control.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
remote = c.remote
P = json.loads((HERE/'protocol.json').read_text(encoding='utf-8-sig'))
ROOT = P['remote_root']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def observe(t, f, handle):
    if not handle:
        return {'handle': None, 'passed': False}
    artifact = ROOT+'/artifacts/'+handle['job_id']
    row = {'handle': handle, 'scheduler': c.source.previous.scheduler(t, handle['job_id']),
           'summary': c.source.read_json(f, ROOT+'/results/summary.json'), 'artifact_exit': None, 'marker': False}
    try:
        with f.file(artifact+'/exit_code.txt') as stream:
            row['artifact_exit'] = int(stream.read(32))
        f.stat(artifact+'/PHASE_COMPLETE')
        row['marker'] = True
    except OSError as exc:
        if exc.errno != 2:
            raise
    s, q = row['summary'] or {}, row['scheduler']
    coverage = (s.get('complete') is True and s.get('protocol') == P['protocol']
                and s.get('pbs_jobid') == handle['job_id'] and s.get('source_mse_job') == P['source_mse_job']
                and s.get('heldout_used') is False and s.get('baseline_records') == 80
                and s.get('candidate_records') == {'mse': 480, 'kl': 480}
                and s.get('baseline_source_same') is True and s.get('source_geometry_validated') is True
                and len(s.get('conditions', [])) == 12 and len(s.get('matched_pairs', [])) == 6
                and all(s.get(key) == 0 for key in ('new_model_forwards', 'new_teacher_forwards', 'new_backwards', 'training_updates')))
    row['passed'] = (q['query_rc'] == 0 and q['job_state'] in ('F', 'X') and q['Exit_status'] == '0'
                     and row['artifact_exit'] == 0 and row['marker'] and coverage)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('upload', 'submit', 'status', 'logs', 'pull'))
    args = parser.parse_args()
    jump, t = remote.direct_connect()
    try:
        with remote.sftp_client(t) as f:
            handle = c.source.read_json(f, ROOT+'/control/handle.json')
            if args.action == 'upload':
                if handle:
                    raise RuntimeError('Existing submission must be reconciled before any edit')
                rc, out = remote.command(t, 'mkdir -p '+shlex.quote(ROOT+'/control')+' '+shlex.quote(ROOT+'/artifacts'), timeout=20)
                if rc:
                    raise RuntimeError(out)
                old = c.source.read_json(f, ROOT+'/protocol.json')
                if old and old != P:
                    raise RuntimeError('Frozen protocol differs')
                for name in ('protocol.json', 'analysis.py', 'cpu.pbs'):
                    data = (HERE/name).read_bytes().replace(b'\r\n', b'\n')
                    if len(data) > remote.LIMIT:
                        raise RuntimeError('Control file exceeds bounded upload')
                    with f.file(ROOT+'/'+name+'.tmp', 'wb') as stream:
                        stream.write(data)
                    f.posix_rename(ROOT+'/'+name+'.tmp', ROOT+'/'+name)
                rc, out = remote.command(t, 'bash -n '+shlex.quote(ROOT+'/cpu.pbs'), timeout=20)
                if rc:
                    raise RuntimeError(out)
                result = {'uploaded': True, 'gpus': 0, 'training_updates': 0}
            elif args.action == 'submit':
                if handle:
                    result = handle
                else:
                    intent = ROOT+'/control/intent.json'
                    if c.source.read_json(f, intent):
                        raise RuntimeError('Reconcile existing intent; never duplicate qsub')
                    if c.source.read_json(f, ROOT+'/protocol.json') != P:
                        raise RuntimeError('Protocol differs')
                    handle = {'status': 'intent_before_qsub', 'gpus': 0, 'walltime': '00:10:00',
                              'created_utc': datetime.now(timezone.utc).isoformat()}
                    c.source.write_json(f, intent, handle)
                    save(HERE/'intent.json', handle)
                    rc, out = remote.command(t, 'qsub -N fwklcross -o '+shlex.quote(ROOT+'/artifacts/cpu.pbs.out')+' '+shlex.quote(ROOT+'/cpu.pbs'), timeout=40)
                    if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+', out.strip()):
                        raise RuntimeError('Uncertain submission; reconcile same intent: '+out[:400])
                    handle.update(status='confirmed', job_id=out.strip())
                    c.source.write_json(f, intent, handle)
                    c.source.write_json(f, ROOT+'/control/handle.json', handle)
                    save(HERE/'handle.json', handle)
                    result = handle
            else:
                row = observe(t, f, handle)
                save(HERE/'last_status.json', {'checked_utc': datetime.now(timezone.utc).isoformat(), **row})
                if args.action == 'logs':
                    if not handle:
                        raise RuntimeError('No submitted job')
                    rc, out = remote.command(t, 'tail -c 12000 '+shlex.quote(ROOT+'/artifacts/'+handle['job_id']+'/job.log'), timeout=20)
                    result = {'query_rc': rc, 'tail': out}
                else:
                    if args.action == 'pull':
                        if not row['passed']:
                            raise RuntimeError('Terminal success and full record coverage not yet verified')
                        save(HERE/'results/summary.json', row['summary'])
                        save(HERE/'results/completion_evidence.json', row)
                        with f.file(ROOT+'/results/ANALYSIS.zh.md', 'rb') as stream:
                            data = stream.read(remote.LIMIT+1)
                        if len(data) > remote.LIMIT:
                            raise RuntimeError('Report exceeds bounded retrieval')
                        (HERE/'results/ANALYSIS.zh.md').write_bytes(data)
                    result = {key: row.get(key) for key in ('handle', 'scheduler', 'passed', 'artifact_exit', 'marker')}
            print(json.dumps(result, indent=2))
    finally:
        t.close()
        if jump:
            jump.close()


if __name__ == '__main__':
    main()
