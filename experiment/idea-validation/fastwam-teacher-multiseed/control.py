"""Bounded SSH control only. Numerical work and cache reads run in PBS."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('teacher_previous_control', HERE.parent / 'fastwam-vq-activation-clipping/control.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
remote = source.remote
P = json.loads((HERE / 'protocol.json').read_text())
ROOT = P['remote_root']
JOBS = {'prepare': 'fwts21prep', 'collect': 'fwts21gpu', 'analysis': 'fwts21ana'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def observe(t, f, unit, handle):
    q = source.previous.scheduler(t, handle['job_id'])
    summary = source.read_json(f, ROOT + '/results/' + unit + '/summary.json')
    progress = source.read_json(f, ROOT + '/results/' + unit + '/progress.json')
    prefix = ROOT + '/artifacts/' + handle['job_id']
    code, marker = None, False
    try:
        with f.file(prefix + '/exit_code.txt') as stream:
            code = int(stream.read(32))
        f.stat(prefix + '/PHASE_COMPLETE')
        marker = True
    except OSError as exc:
        if exc.errno != 2:
            raise
    coverage = bool(summary and summary.get('complete'))
    if unit == 'collect':
        coverage = coverage and summary.get('total_teacher_packets') == 1280 and summary.get('cases') == 80 and summary.get('training_updates') == 0
    if unit == 'prepare':
        coverage = coverage and summary.get('reused_teacher_queries') == 80 and summary.get('heldout_used') is False
    if unit == 'analysis':
        cov = (summary or {}).get('coverage', {})
        selfcheck = source.read_json(f, ROOT + '/results/analysis/analysis_selfcheck.json')
        coverage = (coverage and cov.get('case_count') == 80 and cov.get('seed_count_per_case') == 16
                    and cov.get('teacher_record_count') == 1280 and summary.get('training_updates') == 0
                    and summary.get('heldout_used') is False and selfcheck and selfcheck.get('passed') is True
                    and selfcheck.get('pbs_jobid') == handle['job_id'])
    passed = bool(q['query_rc'] == 0 and q['job_state'] in ('F', 'X') and q['Exit_status'] == '0'
                  and code == 0 and marker and coverage and summary.get('pbs_jobid') == handle['job_id'])
    return dict(unit=unit, job_id=handle['job_id'], scheduler=q, artifact_exit_code=code,
                phase_complete=marker, passed=passed, summary=summary, progress=progress)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('upload', 'submit', 'status', 'logs', 'pull', 'repair_failed_prepare'))
    parser.add_argument('--unit', choices=tuple(JOBS))
    args = parser.parse_args()
    jump, t = remote.direct_connect()
    try:
        with remote.sftp_client(t) as f:
            if args.action == 'upload':
                rc, text = remote.command(t, 'mkdir -p ' + shlex.quote(ROOT + '/control') + ' ' + shlex.quote(ROOT + '/artifacts'), timeout=20)
                if rc:
                    raise RuntimeError(text)
                old = source.read_json(f, ROOT + '/protocol.json')
                if old and old != P:
                    raise RuntimeError('Remote protocol is frozen')
                files = ['protocol.json', 'source_plan.json', 'common.py', 'collect.py', 'gpu.pbs', 'cpu.pbs', 'analyze.py', 'analysis_selfcheck.py', 'DESIGN.zh.md']
                done = []
                for name in files:
                    path = HERE / name
                    if not path.exists():
                        continue
                    data = path.read_bytes().replace(b'\r\n', b'\n')
                    if len(data) > remote.LIMIT:
                        raise RuntimeError('Only bounded small control/source files may be transferred')
                    with f.file(ROOT + '/' + name + '.tmp', 'wb') as stream:
                        stream.write(data)
                    f.posix_rename(ROOT + '/' + name + '.tmp', ROOT + '/' + name)
                    done.append(name)
                rc, text = remote.command(t, 'bash -n ' + shlex.quote(ROOT + '/cpu.pbs') + ' && bash -n ' + shlex.quote(ROOT + '/gpu.pbs'), timeout=20)
                if rc:
                    raise RuntimeError(text)
                print(json.dumps({'uploaded': done, 'shell_syntax_passed': True}))
                return
            handles = source.read_json(f, ROOT + '/control/handles.json') or {}
            if args.action == 'repair_failed_prepare':
                if set(handles) != set(JOBS):
                    raise RuntimeError('Expected exact original three-job chain')
                observed = {u: observe(t, f, u, h) for u, h in handles.items()}
                prep = observed['prepare']['scheduler']
                if prep['job_state'] not in ('F', 'X') or prep['Exit_status'] != '1':
                    raise RuntimeError('Verified failed preparation required')
                if any(observed[u]['scheduler']['job_state'] not in ('F', 'X')
                       or observed[u]['scheduler']['exec_host'] is not None for u in ('collect', 'analysis')):
                    raise RuntimeError('Dependent jobs must be terminal and never allocated')
                record = dict(reason='Source snapshot had preparing status/missing final checkpoint; all80 original C80 rows exactly match',
                              repaired_utc=datetime.now(timezone.utc).isoformat(), handles=handles,
                              observed=observed, new_training_updates=0, new_gpu_allocations=0)
                tag = handles['prepare']['job_id']
                source.write_json(f, ROOT + '/control/failed_chain_' + tag + '.json', record)
                save(HERE / ('failed_chain_' + tag + '.json'), record)
                for unit in JOBS:
                    f.posix_rename(ROOT + '/control/' + unit + '_intent.json', ROOT + '/control/' + unit + '_intent.failed_' + tag + '.json')
                source.write_json(f, ROOT + '/control/handles.json', {})
                save(HERE / 'handles.json', {})
                print(json.dumps({'archived_failed_chain': tag, 'ready_for_repaired_submission': True}))
                return
            if args.action == 'submit':
                unit = args.unit
                if not unit:
                    parser.error('--unit required')
                if unit in handles:
                    print(json.dumps(handles[unit]))
                    return
                if source.read_json(f, ROOT + '/protocol.json') != P:
                    raise RuntimeError('Frozen remote protocol differs')
                if source.read_json(f, ROOT + '/control/' + unit + '_intent.json'):
                    raise RuntimeError('Reconcile previous submission intent before retrying')
                deps = {'prepare': [], 'collect': ['prepare'], 'analysis': ['collect']}[unit]
                if any(d not in handles for d in deps):
                    raise RuntimeError('Confirmed prerequisite handle required')
                cpu = unit != 'collect'
                h = dict(unit=unit, status='intent_before_qsub', created_utc=datetime.now(timezone.utc).isoformat(),
                         gpus=0 if cpu else 1, walltime='00:20:00' if cpu else '01:59:00',
                         training_updates=0, afterok_job_ids=[handles[d]['job_id'] for d in deps])
                source.write_json(f, ROOT + '/control/' + unit + '_intent.json', h)
                save(HERE / (unit + '_intent.json'), h)
                dependency = (' -W ' + shlex.quote('depend=afterok:' + ':'.join(h['afterok_job_ids']))) if deps else ''
                script = ROOT + ('/cpu.pbs' if cpu else '/gpu.pbs')
                cmd = 'qsub -N ' + JOBS[unit] + dependency + ' -v ' + shlex.quote('TEACHER_UNIT=' + unit) + ' -o ' + shlex.quote(ROOT + '/artifacts/' + unit + '.pbs.out') + ' ' + shlex.quote(script)
                rc, text = remote.command(t, cmd, timeout=40)
                if rc or not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+', text.strip()):
                    raise RuntimeError('Unknown qsub outcome; reconcile existing intent: ' + text[:300])
                h.update(status='confirmed', job_id=text.strip())
                handles[unit] = h
                source.write_json(f, ROOT + '/control/' + unit + '_intent.json', h)
                source.write_json(f, ROOT + '/control/handles.json', handles)
                save(HERE / 'handles.json', handles)
                print(json.dumps(h))
                return
            if args.action == 'logs':
                if args.unit not in handles:
                    raise RuntimeError('Confirmed unit required')
                prefix = ROOT + '/artifacts/' + handles[args.unit]['job_id']
                rc, text = remote.command(t, 'if [ -f ' + shlex.quote(prefix + '/job.log') + ' ]; then tail -c 6500 ' + shlex.quote(prefix + '/job.log') + '; fi', timeout=25)
                print(text)
                return
            rows = {u: observe(t, f, u, h) for u, h in handles.items() if not args.unit or u == args.unit}
            state = dict(checked_utc=datetime.now(timezone.utc).isoformat(), units=rows)
            save(HERE / 'last_status.json', state)
            if args.action == 'pull':
                for u, row in rows.items():
                    if row['summary']:
                        save(HERE / 'results' / u / 'summary.json', row['summary'])
                for name in ('replay.json', 'analysis/per_case.json', 'analysis/ANALYSIS.zh.md', 'analysis/analysis_selfcheck.json'):
                    try:
                        with f.file(ROOT + '/results/' + name, 'rb') as stream:
                            data = stream.read(remote.LIMIT + 1)
                    except OSError as exc:
                        if exc.errno == 2:
                            continue
                        raise
                    if len(data) > remote.LIMIT:
                        print(json.dumps({'not_pulled': name, 'reason': 'bounded_small_file_limit'}))
                        continue
                    path = HERE / 'results' / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
            print(json.dumps(dict(checked_utc=state['checked_utc'], units={u: {k: r[k] for k in ('job_id', 'scheduler', 'passed', 'artifact_exit_code', 'phase_complete', 'progress')} for u, r in rows.items()}), indent=2))
    finally:
        t.close()
        if jump:
            jump.close()


if __name__ == '__main__':
    main()
