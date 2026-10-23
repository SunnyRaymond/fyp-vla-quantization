"""Small-file/PBS controls only; all numerical work stays in compute allocations."""
import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import importlib.util
import io
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('vq_kl_previous_control', HERE.parent / 'fastwam-vq-relational-kl/control.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
remote, source = previous.remote, previous.c.source
P = json.loads((HERE / 'protocol.json').read_text())
ROOT = P['remote_root']
JOBS = {'vq_kl': 'fwklvq', 'vq_kl_activation_kl': 'fwklva', 'aggregate': 'fwklvagg'}


class CommandFiles:
    """Bounded small-file fallback when the SSH SFTP subsystem is unavailable."""
    def __init__(self, transport):
        self.transport = transport

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    @contextmanager
    def file(self, path, mode='r'):
        if 'w' in mode:
            if not path.startswith(ROOT + '/'):
                raise RuntimeError('Writes are restricted to this experiment')
            stream = io.BytesIO()
            yield stream
            data = stream.getvalue()
            if len(data) > 65536:
                raise RuntimeError('SSH fallback permits small source/control writes only')
            encoded = base64.b64encode(data).decode('ascii')
            rc, text = remote.command(self.transport, "printf '%s' " + shlex.quote(encoded)
                                      + ' | base64 --decode > ' + shlex.quote(path), timeout=25)
            if rc:
                raise RuntimeError('Small control write failed: ' + text[:200])
        else:
            command = ('if [ -f ' + shlex.quote(path) + ' ]; then head -c ' + str(remote.LIMIT + 1)
                       + ' ' + shlex.quote(path) + '; else exit 44; fi')
            rc, text = remote.command(self.transport, command, timeout=25)
            if rc:
                if rc == 44:
                    raise FileNotFoundError(errno.ENOENT, 'Small control file absent', path)
                raise RuntimeError('Small control read failed: ' + text[:200])
            yield io.BytesIO(text.encode('utf-8'))

    def stat(self, path):
        rc, text = remote.command(self.transport, 'test -e ' + shlex.quote(path), timeout=25)
        if rc == 1:
            raise FileNotFoundError(errno.ENOENT, 'Completion file absent', path)
        if rc:
            raise RuntimeError('Completion file lookup failed: ' + text[:200])

    def posix_rename(self, old, new):
        if not old.startswith(ROOT + '/') or not new.startswith(ROOT + '/'):
            raise RuntimeError('Control rename outside experiment')
        rc, text = remote.command(self.transport, 'mv -f -- ' + shlex.quote(old) + ' ' + shlex.quote(new), timeout=25)
        if rc:
            raise RuntimeError('Atomic control rename failed: ' + text[:200])


def save(path, value):
    previous.save(path, value)


def observe(t, f, unit, handle):
    prefix = ROOT + '/artifacts/' + handle['job_id']
    summary = source.read_json(f, ROOT + '/results/' + unit + '/summary.json') or {}
    progress = source.read_json(f, ROOT + '/results/' + unit + '/progress.json') or {}
    if progress.get('pbs_jobid') != handle['job_id']:
        progress = {}
    row = {'handle': handle, 'scheduler': source.previous.scheduler(t, handle['job_id']),
           'summary': summary, 'progress': progress, 'exit_code': None, 'marker': False}
    try:
        with f.file(prefix + '/exit_code.txt') as stream:
            row['exit_code'] = int(stream.read(32))
        f.stat(prefix + '/PHASE_COMPLETE')
        row['marker'] = True
    except OSError as error:
        if error.errno != 2:
            raise
    coverage = summary.get('complete') is True and summary.get('pbs_jobid') == handle['job_id']
    if unit != 'aggregate':
        coverage &= (summary.get('updates') == (8 if unit == 'vq_kl' else 80)
                     and summary.get('evaluation_queries') == 160 and summary.get('new_teacher_queries') == 0
                     and summary.get('vq_updates') == 8
                     and summary.get('activation_updates') == (0 if unit == 'vq_kl' else 80))
    q = row['scheduler']
    row['passed'] = bool(q['query_rc'] == 0 and q['job_state'] in ('F', 'X') and q['Exit_status'] == '0'
                         and row['exit_code'] == 0 and row['marker'] and coverage)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('upload', 'submit', 'retry', 'status', 'logs', 'pull'))
    parser.add_argument('--unit', choices=tuple(JOBS))
    parser.add_argument('--reason')
    parser.add_argument('--ssh-files', action='store_true', help='Use bounded small SSH file operations instead of SFTP')
    args = parser.parse_args()
    jump, t = remote.direct_connect()
    try:
        with (CommandFiles(t) if args.ssh_files else remote.sftp_client(t)) as f:
            if args.action == 'upload':
                rc, text = remote.command(t, 'mkdir -p ' + shlex.quote(ROOT + '/control') + ' ' + shlex.quote(ROOT + '/artifacts'), timeout=20)
                if rc:
                    raise RuntimeError(text)
                old = source.read_json(f, ROOT + '/protocol.json')
                if old and old != P:
                    raise RuntimeError('Remote experiment protocol is frozen')
                names = ('protocol.json', 'gpu.pbs', 'cpu.pbs', 'worker.py', 'aggregate.py',
                         'train_banks.py', 'relational_loss.py', 'relational_selfcheck.py', 'README.zh.md')
                uploaded = []
                for name in names:
                    path = HERE / name
                    if not path.is_file():
                        continue
                    data = path.read_bytes().replace(b'\r\n', b'\n')
                    if len(data) > remote.LIMIT:
                        raise RuntimeError('Only bounded small control/source uploads are permitted')
                    with f.file(ROOT + '/' + name + '.tmp', 'wb') as stream:
                        stream.write(data)
                    f.posix_rename(ROOT + '/' + name + '.tmp', ROOT + '/' + name)
                    uploaded.append(name)
                for name in ('cpu.pbs', 'gpu.pbs'):
                    rc, text = remote.command(t, 'bash -n ' + shlex.quote(ROOT + '/' + name), timeout=20)
                    if rc:
                        raise RuntimeError(text)
                result = {'uploaded': uploaded, 'shell_syntax_passed': True}
            else:
                handles = source.read_json(f, ROOT + '/control/handles.json') or {}
                if args.action in ('submit', 'retry'):
                    unit = args.unit
                    if not unit:
                        parser.error('--unit required')
                    if source.read_json(f, ROOT + '/protocol.json') != P:
                        raise RuntimeError('Remote protocol differs')
                    prior_handle = None
                    if args.action == 'retry':
                        if unit not in handles or not args.reason:
                            raise RuntimeError('Verified failed handle and explicit repair/resume reason are required')
                        row = observe(t, f, unit, handles[unit])
                        q = row['scheduler']
                        if q['query_rc'] != 0 or q['job_state'] not in ('F', 'X') or q['Exit_status'] == '0' or row['passed']:
                            raise RuntimeError('Never retry a live, successful, or unknown job')
                        prior_handle = handles[unit]
                        save(HERE / ('failed_' + unit + '_' + prior_handle['job_id'] + '.json'), row)
                    if unit in handles and not prior_handle:
                        result = handles[unit]
                    else:
                        dependencies = []
                        if unit == 'vq_kl':
                            prep = source.read_json(f, P['source_relational_root'] + '/results/prepare/summary.json') or {}
                            if not prep.get('complete') or not prep.get('passed') or prep.get('heldout_used') is not False:
                                raise RuntimeError('Completed frozen teacher-only KL definitions required')
                        elif unit == 'vq_kl_activation_kl':
                            if 'vq_kl' not in handles:
                                raise RuntimeError('VQ job handle required')
                            dependencies = [handles['vq_kl']['job_id']]
                        else:
                            if not all(u in handles for u in P['arms']):
                                raise RuntimeError('Both stage handles required')
                            dependencies = [handles[u]['job_id'] for u in P['arms']]
                        suffix = '_retry_of_' + prior_handle['job_id'] if prior_handle else ''
                        intent = ROOT + '/control/' + unit + suffix + '_intent.json'
                        if source.read_json(f, intent):
                            raise RuntimeError('Reconcile existing submission intent before any retry')
                        handle = {'unit': unit, 'status': 'intent_before_qsub',
                                  'gpus': 0 if unit == 'aggregate' else 1,
                                  'walltime': '00:20:00' if unit == 'aggregate' else '01:59:00',
                                  'afterok_job_ids': dependencies, 'created_utc': datetime.now(timezone.utc).isoformat()}
                        if prior_handle:
                            handle.update(previous_attempt=prior_handle, repair_reason=args.reason)
                        source.write_json(f, intent, handle)
                        save(HERE / (unit + suffix + '_intent.json'), handle)
                        dependency = ' -W ' + shlex.quote('depend=afterok:' + ':'.join(dependencies)) if dependencies else ''
                        command = ('qsub -N ' + JOBS[unit] + dependency + ' -v ' + shlex.quote('FW_VQ_KL_UNIT=' + unit)
                                   + ' -o ' + shlex.quote(ROOT + '/artifacts/' + unit + '.pbs.out') + ' '
                                   + shlex.quote(ROOT + ('/cpu.pbs' if unit == 'aggregate' else '/gpu.pbs')))
                        rc, text = remote.command(t, command, timeout=40)
                        if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+', text.strip()):
                            raise RuntimeError('Unknown qsub outcome; reconcile durable intent before any retry: ' + text[:300])
                        handle.update(status='confirmed', job_id=text.strip())
                        handles[unit] = handle
                        source.write_json(f, intent, handle)
                        source.write_json(f, ROOT + '/control/handles.json', handles)
                        save(HERE / 'handles.json', handles)
                        result = handle
                else:
                    rows = {unit: observe(t, f, unit, handle) for unit, handle in handles.items()
                            if not args.unit or unit == args.unit}
                    snapshot = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'units': rows}
                    save(HERE / 'last_status.json', snapshot)
                    if args.action == 'logs':
                        if not args.unit:
                            parser.error('--unit required')
                        rc, text = remote.command(t, 'tail -c 10000 ' + shlex.quote(ROOT + '/artifacts/' + handles[args.unit]['job_id'] + '/job.log'), timeout=25)
                        result = {'query_rc': rc, 'tail': text}
                    else:
                        if args.action == 'pull':
                            for unit, row in rows.items():
                                if row['summary']:
                                    save(HERE / 'results' / unit / 'summary.json', row['summary'])
                                value = source.read_json(f, ROOT + '/results/' + unit + '/training_receipt.json')
                                if value:
                                    save(HERE / 'results' / unit / 'training_receipt.json', value)
                            for split in ('calibration', 'heldout'):
                                for unit in P['arms']:
                                    value = source.read_json(f, ROOT + '/results/' + split + '/' + unit + '/summary.json')
                                    if value:
                                        save(HERE / 'results' / split / unit / 'summary.json', value)
                            for name in ('comparison', 'paired_details', 'decision'):
                                value = source.read_json(f, ROOT + '/results/' + name + '.json')
                                if value:
                                    save(HERE / 'results' / (name + '.json'), value)
                            try:
                                with f.file(ROOT + '/results/ANALYSIS.zh.md', 'rb') as stream:
                                    report = stream.read(remote.LIMIT + 1)
                            except OSError as error:
                                if error.errno != 2:
                                    raise
                            else:
                                if len(report) > remote.LIMIT:
                                    raise RuntimeError('Report exceeds bounded retrieval')
                                path = HERE / 'results/ANALYSIS.zh.md'
                                path.parent.mkdir(parents=True, exist_ok=True)
                                path.write_bytes(report)
                        result = {'checked_utc': snapshot['checked_utc'], 'units': {u: {
                            'job_id': r['handle']['job_id'], 'scheduler': r['scheduler'], 'passed': r['passed'],
                            'phase': r['progress'].get('phase'), 'updates': r['progress'].get('updates'),
                            'split': r['progress'].get('split'), 'queries': r['progress'].get('queries')}
                            for u, r in rows.items()}}
            print(json.dumps(result, indent=2))
    finally:
        t.close()
        if jump:
            jump.close()


if __name__ == '__main__':
    main()
