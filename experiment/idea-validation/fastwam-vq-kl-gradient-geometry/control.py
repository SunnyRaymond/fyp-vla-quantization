"""Bounded controls for one CPU-only cached-gradient analysis; no training entry point."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('geometry_control_source', HERE.parent/'fastwam-vq-dual-kl/control.py')
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
           'summary': c.source.read_json(f, ROOT+'/results/summary.json'),
           'selfcheck': c.source.read_json(f, artifact+'/direction_selfcheck.json'),
           'artifact_exit': None, 'marker': False}
    try:
        with f.file(artifact+'/exit_code.txt') as stream:
            row['artifact_exit'] = int(stream.read(32))
        f.stat(artifact+'/PHASE_COMPLETE')
        row['marker'] = True
    except OSError as exc:
        if exc.errno != 2:
            raise
    s, q, check = row['summary'] or {}, row['scheduler'], row['selfcheck'] or {}
    coverage = (s.get('protocol') == P['protocol'] and s.get('pbs_jobid') == handle['job_id']
                and s.get('complete') is True and s.get('heldout_used') is False
                and s.get('source_parity_passed') is True and s.get('baseline_metrics_match') is True
                and s.get('source_mse_job') == P['source_mse_job'] and s.get('source_updates') == 80
                and s.get('candidate_checkpoints_passed') is True
                and s.get('ratio_only_adam_group_assertions_passed') is True
                and s.get('gradient_counts') == {'mse': 80, 'kl': 80}
                and s.get('parameter_counts') == {'ratio': 614, 'channel': 1806863}
                and all(s.get(key) == 0 for key in ('new_model_forwards', 'new_teacher_forwards', 'new_backwards', 'training_updates'))
                and len(s.get('step_simulations', [])) == 12
                and check.get('status') == 'passed' and check.get('pbs_jobid') == handle['job_id'])
    row['passed'] = (q['query_rc'] == 0 and q['job_state'] in ('F', 'X') and q['Exit_status'] == '0'
                     and row['artifact_exit'] == 0 and row['marker'] and coverage)
    return row


def repairable(t, f, handle):
    if not handle:
        return False
    row = observe(t, f, handle)
    q = row['scheduler']
    rc, tail = remote.command(t, 'tail -c 4000 '+shlex.quote(ROOT+'/artifacts/'+handle['job_id']+'/job.log'), timeout=20)
    return (q['query_rc'] == 0 and q['job_state'] in ('F', 'X') and q['Exit_status'] == '1'
            and row['artifact_exit'] == 1 and rc == 0
            and 'environment.sh: line 51: FW_JOB_START_EPOCH: unbound variable' in tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('upload', 'submit', 'status', 'logs', 'pull'))
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    if args.repair and args.action not in ('upload', 'submit'):
        parser.error('--repair is only for upload or submit')
    jump, t = remote.direct_connect()
    try:
        with remote.sftp_client(t) as f:
            handle = c.source.read_json(f, ROOT+'/control/handle.json')
            if args.action == 'upload':
                if handle and not (args.repair and repairable(t, f, handle)):
                    raise RuntimeError('CPU analysis already submitted; inspect its result before editing')
                rc, out = remote.command(t, 'mkdir -p '+shlex.quote(ROOT+'/control')+' '+shlex.quote(ROOT+'/artifacts'), timeout=20)
                if rc:
                    raise RuntimeError(out)
                old = c.source.read_json(f, ROOT+'/protocol.json')
                if old and old != P:
                    raise RuntimeError('Frozen analysis protocol differs')
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
                if handle and not args.repair:
                    result = handle
                else:
                    previous = handle
                    if args.repair and not repairable(t, f, previous):
                        raise RuntimeError('Repair requires the exact verified terminal environment-variable failure')
                    intent = ROOT+'/control/'+('repair_'+previous['job_id'] if previous else 'intent')+'.json'
                    if c.source.read_json(f, intent):
                        raise RuntimeError('Reconcile existing qsub intent; never duplicate')
                    if c.source.read_json(f, ROOT+'/protocol.json') != P:
                        raise RuntimeError('Protocol differs')
                    handle = {'status': 'intent_before_qsub', 'gpus': 0, 'walltime': '00:10:00',
                              'created_utc': datetime.now(timezone.utc).isoformat()}
                    if previous:
                        handle.update(repair_of=previous['job_id'], repair='Export shared runtime start time before sourcing environment.sh')
                    c.source.write_json(f, intent, handle)
                    save(HERE/'intent.json', handle)
                    rc, out = remote.command(t, 'qsub -N fwklgeom -o '+shlex.quote(ROOT+'/artifacts/cpu.pbs.out')+' '+shlex.quote(ROOT+'/cpu.pbs'), timeout=40)
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
                        raise RuntimeError('No job submitted')
                    rc, out = remote.command(t, 'tail -c 12000 '+shlex.quote(ROOT+'/artifacts/'+handle['job_id']+'/job.log'), timeout=20)
                    result = {'query_rc': rc, 'tail': out}
                else:
                    if args.action == 'pull':
                        if not row['passed']:
                            raise RuntimeError('CPU completion and zero-forward coverage not yet verified')
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
