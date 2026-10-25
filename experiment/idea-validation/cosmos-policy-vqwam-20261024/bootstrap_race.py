"""Allocation-time race between the three bootstrap singles and a g2 bundle."""
from __future__ import annotations
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import time

from common import guard, read_json, save_json

BASELINE_FOLLOWUP = Path('control/baseline_followup_race')
BASELINE_FOLLOWUP_NAMES = {
    'perf_svdquant': 'cpvq24psvd', 'viditq': 'cpvq24vid', 'bundle': 'cpvq24c2'}
PBS_HANDLE = re.compile(r'^\d+\.[A-Za-z0-9_.-]+$')


def fields(output):
    result, key = {}, None
    for line in output.splitlines():
        match = re.match(r'^    ([A-Za-z_][\w.]*) = (.*)$', line)
        if match:
            key, value = match.groups()
            result[key] = value
        elif key and line.startswith('\t'):
            result[key] += line.strip()
    return result


def command(*args):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=35)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed rc={result.returncode}: {result.stdout[-800:]}')
    return result.stdout


def status(row):
    value = fields(command('qstat', '-xf', row['job_id']))
    if value.get('Job_Name') != row['job_name']:
        raise RuntimeError('PBS identity mismatch: ' + row['job_id'])
    return value


def started(value):
    if value.get('job_state') != 'R':
        return None
    if not value.get('stime') or not value.get('exec_host'):
        raise RuntimeError('Running candidate lacks PBS allocation/start evidence')
    return datetime.strptime(value['stime'], '%a %b %d %H:%M:%S %Y')


def winner_group(single_start, bundle_start):
    if single_start is None and bundle_start is None:
        raise RuntimeError('No candidate has an actual running allocation')
    return ('single' if single_start is not None and
            (bundle_start is None or single_start <= bundle_start) else 'bundle')


def migrate_dependencies(ready, root):
    if not ready.get('dependents'):
        return []
    parent = ready['bundle']['job_id']
    old = next(row['job_id'] for row in ready['single'] if row['stage'] == 'teacher')
    changed = []
    for row in ready['dependents']:
        value = status(row)
        dependency = value.get('depend', '')
        expected = r'afterok:' + re.escape(parent) + r'(?:@[A-Za-z0-9_.-]+)?'
        if re.fullmatch(expected, dependency):
            changed.append({'job_id': row['job_id'], 'already_migrated': True})
            continue
        if (value.get('job_state') not in ('Q', 'H') or value.get('stime')
                or value.get('exec_host') or not re.fullmatch(
                    r'afterok:' + re.escape(old) + r'(?:@[A-Za-z0-9_.-]+)?', dependency)):
            raise RuntimeError('Dependent is not an unstarted child of the old teacher: ' + row['job_id'])
        command('qalter', '-W', 'depend=afterok:' + parent, row['job_id'])
        updated = status(row)
        if not re.fullmatch(expected, updated.get('depend', '')):
            raise RuntimeError('Dependent migration was not confirmed: ' + row['job_id'])
        changed.append({'job_id': row['job_id'], 'old_parent': old, 'new_parent': parent})
        save_json(root / 'dependency_migrations.json', changed)
    return changed


def cancel(row):
    before = status(row)
    if before.get('job_state') not in ('F', 'X'):
        command('qdel', row['job_id'])
    for _ in range(45):
        after = status(row)
        if after.get('job_state') in ('F', 'X'):
            return {'job_id': row['job_id'], 'job_name': row['job_name'],
                    'before': before.get('job_state'), 'after': after.get('job_state'),
                    'Exit_status': after.get('Exit_status')}
        time.sleep(1)
    raise RuntimeError('Canceled candidate has not reached a verified terminal state: ' + row['job_id'])


def validate_registration(ready, race_directory):
    if race_directory != BASELINE_FOLLOWUP:
        return
    singles = ready.get('single')
    bundle = ready.get('bundle')
    if (not isinstance(ready.get('registration'), str) or not ready['registration']
            or not isinstance(singles, list) or len(singles) != 2
            or not isinstance(bundle, dict) or not isinstance(ready.get('dependents'), list)
            or ready['dependents']):
        raise RuntimeError('Baseline follow-up registration must contain exactly two singles and no dependents')
    rows = {row.get('stage'): row for row in singles if isinstance(row, dict)}
    if set(rows) != {'perf_svdquant', 'viditq'} or len(rows) != 2:
        raise RuntimeError('Baseline follow-up registration has an unexpected single-stage set')
    ids = []
    for stage, expected_name in BASELINE_FOLLOWUP_NAMES.items():
        row = bundle if stage == 'bundle' else rows[stage]
        job_id = row.get('job_id')
        if (row.get('job_name') != expected_name or not isinstance(job_id, str)
                or not PBS_HANDLE.fullmatch(job_id)):
            raise RuntimeError('Baseline follow-up registration has a mismatched PBS identity: ' + stage)
        ids.append(job_id)
    if len(set(ids)) != 3:
        raise RuntimeError('Baseline follow-up registration reuses a PBS job ID')


def claim(root, candidate, stage=None, race_directory='control/bootstrap_race'):
    guard()
    relative_race = Path(race_directory)
    if relative_race.is_absolute() or '..' in relative_race.parts:
        raise RuntimeError('Race directory must stay inside the task root')
    race = Path(root) / relative_race
    marker = race / 'submissions_ready.json'
    if not marker.is_file():
        # The original bootstrap singles retain their pre-registration path; new races fail closed.
        if candidate == 'single' and relative_race == Path('control/bootstrap_race'):
            return 0
        raise RuntimeError('Race candidate cannot start without registered counterpart identities')
    ready = read_json(marker)
    validate_registration(ready, relative_race)
    own = (next((row for row in ready['single'] if row['stage'] == stage), None)
           if candidate == 'single' else ready['bundle'])
    if own is None or own['job_id'] != os.environ['PBS_JOBID']:
        raise RuntimeError('Caller does not match the registered race candidate')
    if started(status(own)) is None:
        raise RuntimeError('Race caller has no actual running PBS allocation')
    winner_path, work_ready = race / 'winner.json', race / 'ready_for_work.json'
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if work_ready.is_file():
            winner = read_json(work_ready)
            if winner.get('registration') != ready['registration']:
                raise RuntimeError('Bootstrap winner belongs to another registered race')
            return 0 if winner['candidate'] == candidate else 3
        if (race / 'failure.json').is_file():
            raise RuntimeError('Bootstrap cancellation/dependency migration failed; no model may load')
        if not (race / 'claim').is_dir():
            single_starts = [started(status(row)) for row in ready['single']]
            single_starts = [value for value in single_starts if value is not None]
            bundle_start = started(status(ready['bundle']))
            single_start = min(single_starts) if single_starts else None
            winner = winner_group(single_start, bundle_start)
            if winner == candidate:
                try:
                    (race / 'claim').mkdir()
                except FileExistsError:
                    continue
                record = {'registration': ready['registration'], 'candidate': candidate,
                          'coordinator_job': os.environ['PBS_JOBID'],
                          'single_first_stime': str(single_start), 'bundle_stime': str(bundle_start),
                          'selection': 'earliest actual PBS stime; same-second tie favors singles'}
                save_json(winner_path, record)
                try:
                    migrations = migrate_dependencies(ready, race) if candidate == 'bundle' else []
                    losers = ready['single'] if candidate == 'bundle' else [ready['bundle']]
                    cancellations = [cancel(row) for row in losers]
                    save_json(work_ready, {**record, 'dependency_migrations': migrations,
                                          'cancellations': cancellations})
                    print('BOOTSTRAP_RACE_WINNER ' + candidate, flush=True)
                    return 0
                except Exception as exc:
                    save_json(race / 'failure.json', {**record, 'error': repr(exc)})
                    raise
        time.sleep(1)
    raise RuntimeError('Bootstrap winner did not finish safe cancellation/dependency migration')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['claim'])
    parser.add_argument('--candidate', choices=['single', 'bundle'], required=True)
    parser.add_argument('--stage')
    parser.add_argument('--race-directory', default='control/bootstrap_race')
    parser.add_argument('--root', default='/scratch/users/ntu/yguo017/cosmos-policy-vqwam-20261024')
    args = parser.parse_args()
    return claim(args.root, args.candidate, args.stage, args.race_directory)


if __name__ == '__main__':
    raise SystemExit(main())
