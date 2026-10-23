"""Bounded PBS controls; workloads and tensor transfers stay in allocations."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('relational_source', HERE.parent / 'fastwam-vq-dual-kl/control.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
remote = c.remote
P = json.loads((HERE / 'protocol.json').read_text())
ROOT = P['remote_root']
JOBS = {'prepare': 'fwrelprep', 'softbin': 'fwrelbase', 'relational': 'fwreltrain', 'aggregate': 'fwrelagg'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def observe(t, f, unit, h):
    prefix = ROOT + '/artifacts/' + h['job_id']
    summary = c.source.read_json(f, ROOT + '/results/' + unit + '/summary.json') or {}
    row = {'handle': h, 'scheduler': c.source.previous.scheduler(t, h['job_id']),
           'summary': summary, 'progress': c.source.read_json(f, ROOT + '/results/' + unit + '/progress.json'),
           'exit_code': None, 'marker': False}
    if row['progress'] and row['progress'].get('pbs_jobid') != h['job_id']:
        row['previous_progress'] = row['progress']
        row['progress'] = None
    try:
        with f.file(prefix + '/exit_code.txt') as stream:
            row['exit_code'] = int(stream.read(32))
        f.stat(prefix + '/PHASE_COMPLETE')
        row['marker'] = True
    except OSError as exc:
        if exc.errno != 2:
            raise
    coverage = summary.get('complete') is True and summary.get('pbs_jobid') == h['job_id']
    if unit == 'prepare':
        coverage &= summary.get('passed') is True and summary.get('heldout_used') is False and summary.get('calibration_case_ids') == P['calibration']['case_ids']
    elif unit in P['arms']:
        coverage &= summary.get('updates') == 80 and summary.get('evaluation_queries') == 160 and summary.get('new_teacher_queries') == 0
    else:
        coverage &= summary.get('calibration_records') == 160 and summary.get('heldout_records') == 160
    q = row['scheduler']
    row['passed'] = bool(q['query_rc'] == 0 and q['job_state'] in ('F', 'X') and q['Exit_status'] == '0'
                         and row['exit_code'] == 0 and row['marker'] and coverage)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('upload', 'submit', 'repair', 'status', 'logs', 'pull'))
    parser.add_argument('--unit', choices=tuple(JOBS))
    args = parser.parse_args()
    jump, t = remote.direct_connect()
    try:
        with remote.sftp_client(t) as f:
            if args.action == 'upload':
                rc, out = remote.command(t, 'mkdir -p ' + shlex.quote(ROOT + '/control') + ' ' + shlex.quote(ROOT + '/artifacts'), timeout=20)
                if rc:
                    raise RuntimeError(out)
                old = c.source.read_json(f, ROOT + '/protocol.json')
                if old and old != P:
                    raise RuntimeError('Remote protocol is frozen')
                names = ['protocol.json', 'cpu.pbs', 'gpu.pbs', 'prepare.py', 'worker.py', 'aggregate.py', 'relational_loss.py', 'relational_selfcheck.py']
                uploaded = []
                for name in names:
                    path = HERE / name
                    if not path.is_file():
                        continue
                    data = path.read_bytes().replace(b'\r\n', b'\n')
                    if len(data) > remote.LIMIT:
                        raise RuntimeError('Only small code/control files may be uploaded')
                    with f.file(ROOT + '/' + name + '.tmp', 'wb') as stream:
                        stream.write(data)
                    f.posix_rename(ROOT + '/' + name + '.tmp', ROOT + '/' + name)
                    uploaded.append(name)
                for name in ('cpu.pbs', 'gpu.pbs'):
                    rc, out = remote.command(t, 'bash -n ' + shlex.quote(ROOT + '/' + name), timeout=15)
                    if rc:
                        raise RuntimeError(out)
                result = {'uploaded': uploaded, 'shell_syntax_passed': True}
            else:
                handles = c.source.read_json(f, ROOT + '/control/handles.json') or {}
                if args.action in ('submit', 'repair'):
                    unit = args.unit
                    if not unit:
                        parser.error('--unit required')
                    if c.source.read_json(f, ROOT + '/protocol.json') != P:
                        raise RuntimeError('Protocol differs')
                    repair = None
                    if args.action == 'repair':
                        if unit not in ('softbin', 'aggregate') or unit not in handles:
                            raise RuntimeError('Only the known endpoint-save failure and its dependent aggregation are repairable')
                        previous = observe(t, f, unit, handles[unit])
                        q = previous['scheduler']
                        if q['query_rc'] != 0 or q['job_state'] not in ('F', 'X') or previous['passed']:
                            raise RuntimeError('A verified failed terminal attempt is required; never restart a live/unknown job')
                        if unit == 'softbin':
                            error = (previous['progress'] or {}).get('error', '')
                            if q['Exit_status'] != '1' or not error.startswith('FileNotFoundError:') or '/tensors/calibration/' not in error:
                                raise RuntimeError('Failure is not the known missing tensor directory')
                        elif not handles[unit].get('afterok_job_ids') or handles['softbin']['job_id'] in handles[unit]['afterok_job_ids']:
                            raise RuntimeError('Aggregation repair requires a replaced failed dependency')
                        repair = handles[unit]
                        save(HERE / ('failed_' + unit + '_' + repair['job_id'] + '.json'), previous)
                    if unit in handles and repair is None:
                        result = handles[unit]
                    else:
                        if unit in P['arms'] and ('prepare' not in handles or not observe(t, f, 'prepare', handles['prepare'])['passed']):
                            raise RuntimeError('Successful complete C80 definition validation is required')
                        dependencies = []
                        if unit == 'aggregate':
                            if not all(u in handles for u in P['arms']):
                                raise RuntimeError('Both GPU job handles are required')
                            dependencies = [handles[u]['job_id'] for u in P['arms']]
                        intent_name = unit + ('_repair_of_' + repair['job_id'] if repair else '') + '_intent.json'
                        intent = ROOT + '/control/' + intent_name
                        if c.source.read_json(f, intent):
                            raise RuntimeError('Reconcile existing submission intent before any retry')
                        cpu = unit in ('prepare', 'aggregate')
                        h = {'unit': unit, 'status': 'intent_before_qsub', 'gpus': 0 if cpu else 1,
                             'walltime': '00:20:00' if cpu else '01:59:00', 'created_utc': datetime.now(timezone.utc).isoformat()}
                        if repair:
                            h.update(repair_of=repair['job_id'], previous_attempt=repair,
                                     repair='create missing endpoint tensor directories' if unit == 'softbin' else 'replace failed afterok dependency')
                        c.source.write_json(f, intent, h)
                        save(HERE / intent_name, h)
                        dependency = ' -W ' + shlex.quote('depend=afterok:' + ':'.join(dependencies)) if dependencies else ''
                        command = ('qsub -N ' + JOBS[unit] + dependency + ' -v ' + shlex.quote('FW_REL_UNIT=' + unit)
                                   + ' -o ' + shlex.quote(ROOT + '/artifacts/' + unit + '.pbs.out') + ' '
                                   + shlex.quote(ROOT + ('/cpu.pbs' if cpu else '/gpu.pbs')))
                        rc, out = remote.command(t, command, timeout=40)
                        if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+', out.strip()):
                            raise RuntimeError('Unknown qsub outcome; reconcile intent: ' + out[:400])
                        h.update(status='confirmed', job_id=out.strip(), afterok_job_ids=dependencies)
                        handles[unit] = h
                        c.source.write_json(f, intent, h)
                        c.source.write_json(f, ROOT + '/control/handles.json', handles)
                        save(HERE / 'handles.json', handles)
                        result = h
                else:
                    rows = {u: observe(t, f, u, h) for u, h in handles.items() if not args.unit or u == args.unit}
                    snapshot = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'units': rows}
                    save(HERE / 'last_status.json', snapshot)
                    if args.action == 'logs':
                        if not args.unit:
                            parser.error('--unit required')
                        rc, out = remote.command(t, 'tail -c 9000 ' + shlex.quote(ROOT + '/artifacts/' + handles[args.unit]['job_id'] + '/job.log'), timeout=20)
                        result = {'query_rc': rc, 'tail': out}
                    else:
                        if args.action == 'pull':
                            for u, row in rows.items():
                                if row['summary']:
                                    save(HERE / 'results' / u / 'summary.json', row['summary'])
                                for name in ('relational_selfcheck',):
                                    value = c.source.read_json(f, ROOT + '/artifacts/' + row['handle']['job_id'] + '/' + name + '.json')
                                    if value:
                                        save(HERE / 'results' / u / (name + '.json'), value)
                            for name in ('prepare/definition_validation', 'comparison', 'paired_details', 'decision'):
                                value = c.source.read_json(f, ROOT + '/results/' + name + '.json')
                                if value:
                                    save(HERE / 'results' / (name + '.json'), value)
                            try:
                                with f.file(ROOT + '/results/ANALYSIS.zh.md', 'rb') as stream:
                                    report = stream.read(remote.LIMIT + 1)
                            except OSError as exc:
                                if exc.errno != 2:
                                    raise
                            else:
                                if len(report) > remote.LIMIT:
                                    raise RuntimeError('Report exceeds bounded retrieval')
                                path = HERE / 'results/ANALYSIS.zh.md'
                                path.parent.mkdir(parents=True, exist_ok=True)
                                path.write_bytes(report)
                        result = {'checked_utc': snapshot['checked_utc'], 'units': {u: {
                            'job_id': r['handle']['job_id'], 'scheduler': r['scheduler'], 'passed': r['passed'],
                            'phase': (r['progress'] or {}).get('phase'), 'updates': (r['progress'] or {}).get('updates'),
                            'queries': (r['progress'] or {}).get('queries'), 'split': (r['progress'] or {}).get('split')}
                            for u, r in rows.items()}}
            print(json.dumps(result, indent=2))
    finally:
        t.close()
        if jump:
            jump.close()


if __name__ == '__main__':
    main()
