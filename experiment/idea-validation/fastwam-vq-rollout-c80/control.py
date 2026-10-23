"""Small control files and PBS status only; model work runs in allocations."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rollout_parent', HERE.parent / 'fastwam-vq-c80-joint-rmse/control.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
remote, source = parent.remote, parent.source
P = json.loads((HERE / 'protocol.json').read_text())
ROOT = P['remote_root']
UNITS = {'sources': ('fwrasrc', 0, 'sources.pbs'), 'prepare': ('fwraprep', 1, 'run.pbs'),
         'train_eval': ('fwrajnt', 1, 'run.pbs')}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def observe(t, f, unit, handle):
    q = source.previous.scheduler(t, handle['job_id'])
    prefix = ROOT + '/artifacts/' + handle['job_id']
    result = {'handle': handle, 'scheduler': q, 'artifact_exit_code': None, 'phase_complete_marker': False,
              'progress': source.read_json(f, ROOT + '/results/' + unit + '/progress.json'),
              'summary': source.read_json(f, ROOT + '/results/' + unit + '/summary.json')}
    try:
        with f.file(prefix + '/exit_code.txt') as stream:
            result['artifact_exit_code'] = int(stream.read(32))
    except OSError as exc:
        if exc.errno != 2: raise
    try:
        f.stat(prefix + '/PHASE_COMPLETE')
        result['phase_complete_marker'] = True
    except OSError as exc:
        if exc.errno != 2: raise
    return result


def passed(row):
    q, s = row['scheduler'], row.get('summary') or {}
    unit = row['handle']['unit']
    coverage = {'sources': s.get('source_contexts') == 40,
                'prepare': s.get('contexts') == 80 and s.get('visited_contexts') == 40 and s.get('new_teacher_queries') == 40,
                'train_eval': s.get('updates') == 80 and s.get('evaluation_queries') == 80}.get(unit, False)
    return (q['query_rc'] == 0 and q['job_state'] in ('F', 'X') and q['Exit_status'] == '0'
            and row['artifact_exit_code'] == 0 and row['phase_complete_marker'] and s.get('complete') is True and coverage
            and s.get('pbs_jobid') == row['handle']['job_id'])


def submit(t, unit, resume=False):
    with remote.sftp_client(t) as f:
        if source.read_json(f, ROOT + '/protocol.json') != P:
            raise RuntimeError('Remote protocol changed')
        handles = source.read_json(f, ROOT + '/control/handles.json') or {}
        old = handles.get(unit)
        if old and not resume:
            return old
        if resume:
            if not old: raise RuntimeError('No prior handle to resume')
            observed = observe(t, f, unit, old)
            q = observed['scheduler']
            if q['query_rc'] != 0 or q['job_state'] not in ('F', 'X') or q['Exit_status'] != '75' or observed['artifact_exit_code'] != 75:
                raise RuntimeError('Only verified terminal checkpointed job may resume')
        prerequisite = {'prepare': 'sources', 'train_eval': 'prepare'}.get(unit)
        if prerequisite and (prerequisite not in handles or not passed(observe(t, f, prerequisite, handles[prerequisite]))):
            raise RuntimeError('Complete prerequisite required: ' + prerequisite)
        attempt = old['attempt'] + 1 if old else 0
        intent = ROOT + f'/control/{unit}_intent_{attempt}.json'
        if source.read_json(f, intent):
            raise RuntimeError('Submission intent already exists; reconcile PBS before retrying')
        name, gpu, script = UNITS[unit]
        h = {'unit': unit, 'attempt': attempt, 'job_name': name, 'gpus': gpu, 'walltime': '01:59:00',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'intent_before_qsub',
             'resume_from': old['job_id'] if resume else None}
        source.write_json(f, intent, h)
        variables = 'FW_ROLLOUT_UNIT=' + unit
        if resume:
            variables += ',FW_ROLLOUT_RESUME_FROM=' + old['job_id']
        rc, output = remote.command(t, 'qsub -N ' + shlex.quote(name) + ' -o ' + shlex.quote(ROOT + f'/artifacts/{name}_{attempt}.pbs.out')
                                  + ' -v ' + shlex.quote(variables) + ' ' + shlex.quote(ROOT + '/' + script), timeout=40)
        if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+', output.strip()):
            raise RuntimeError('qsub unknown/rejected; do not resubmit: ' + output[:500])
        h.update(job_id=output.strip(), status='confirmed')
        source.write_json(f, intent, h)
        if old: source.write_json(f, ROOT + f'/control/{unit}_previous_{old["attempt"]}.json', old)
        handles[unit] = h
        source.write_json(f, ROOT + '/control/handles.json', handles)
        save(HERE / 'handles.json', handles)
        return h


def main():
    a = argparse.ArgumentParser()
    a.add_argument('action', choices=('upload', 'submit', 'resume', 'status', 'pull', 'logs'))
    a.add_argument('--unit', choices=tuple(UNITS))
    args = a.parse_args()
    jump, t = remote.direct_connect()
    try:
        if args.action == 'upload':
            rc, out = remote.command(t, 'mkdir -p ' + shlex.quote(ROOT + '/control') + ' ' + shlex.quote(ROOT + '/artifacts'), timeout=20)
            if rc: raise RuntimeError(out)
            with remote.sftp_client(t) as f:
                previous = source.read_json(f, ROOT + '/protocol.json')
                if previous and previous != P: raise RuntimeError('Existing frozen protocol differs')
                names = ['common.py', 'protocol.json', 'prepare_sources.py', 'sources.pbs', 'run.pbs', 'prepare.py', 'worker.py']
                for name in names:
                    path = HERE / name
                    if not path.exists(): continue
                    data = path.read_bytes().replace(b'\r\n', b'\n')
                    if len(data) > remote.LIMIT: raise RuntimeError('Control file exceeds small-file limit')
                    with f.file(ROOT + '/' + name + '.tmp', 'wb') as stream: stream.write(data)
                    f.posix_rename(ROOT + '/' + name + '.tmp', ROOT + '/' + name)
            result = {'uploaded': True}
        elif args.action in ('submit', 'resume'):
            if not args.unit: a.error('--unit is required')
            result = submit(t, args.unit, args.action == 'resume')
        else:
            with remote.sftp_client(t) as f:
                handles = source.read_json(f, ROOT + '/control/handles.json') or {}
                result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'units': {
                    u: observe(t, f, u, h) for u, h in handles.items() if args.unit is None or u == args.unit}}
                save(HERE / 'last_status.json', result)
                if args.action == 'pull':
                    for name in ('comparison.json', 'sources/summary.json', 'prepare/summary.json', 'train_eval/summary.json', 'heldout/rollout_joint/summary.json'):
                        value = source.read_json(f, ROOT + '/results/' + name)
                        if value: save(HERE / 'results' / name, value)
                    for name in ('RESULTS.zh.md',):
                        try:
                            with f.file(ROOT + '/results/' + name) as stream: data = stream.read(remote.LIMIT + 1)
                            if len(data) > remote.LIMIT: raise RuntimeError('Report exceeds small-file limit')
                            (HERE / 'results' / name).write_bytes(data)
                        except OSError as exc:
                            if exc.errno != 2: raise
            if args.action == 'logs':
                if not args.unit: a.error('--unit required')
                h = result['units'][args.unit]['handle']
                rc, out = remote.command(t, 'tail -n 18 ' + shlex.quote(ROOT + '/artifacts/' + h['job_id'] + '/job.log')
                                         + ' 2>/dev/null', timeout=25)
                result = {'job_id': h['job_id'], 'query_rc': rc, 'tail': out[-12000:]}
        display = result
        if 'units' in result:
            def compact(value):
                return {k: v for k, v in (value or {}).items()
                        if k not in ('history', 'comparison', 'heldout_summary', 'update_order')}
            display = {**result, 'units': {
                u: {**row, 'progress': compact(row.get('progress')), 'summary': compact(row.get('summary'))}
                for u, row in result['units'].items()}}
        print(json.dumps(display, indent=2))
    finally:
        t.close()
        if jump: jump.close()


if __name__ == '__main__': main()
