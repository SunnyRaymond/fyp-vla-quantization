"""Read-only polling of the C8 experiment's authoritative PBS handles."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
CONTROL = HERE / 'control.py'
PROGRESS_FIELDS = ('phase', 'updates', 'queries', 'last_case', 'last_seed', 'error', 'reason', 'complete')
SUMMARY_FIELDS = (
    'complete', 'passed', 'heldout_used', 'calibration_case_ids', 'updates',
    'evaluation_queries', 'new_teacher_queries', 'baseline_new_model_queries',
    'baseline_records', 'candidate_records',
)


def load_control():
    spec = importlib.util.spec_from_file_location('c8_anchor_monitor_control', CONTROL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def poll():
    c = load_control()  # Reload each round so the monitor follows current controller code.
    jump, transport = c.remote.direct_connect()
    try:
        with c.remote.sftp_client(transport) as sftp:
            handles = c.read_json(sftp, c.ROOT + '/control/handles.json') or {}
            rows = {unit: c.observe(transport, sftp, unit, handle)
                    for unit, handle in handles.items()}
            return {'checked_utc': datetime.now(timezone.utc).isoformat(), 'units': rows}
    finally:
        transport.close()
        if jump:
            jump.close()


def material_state(row):
    progress = row.get('progress') or {}
    summary = row.get('summary') or {}
    return {
        'query_rc': row['scheduler'].get('query_rc'),
        'job_state': row['scheduler'].get('job_state'),
        'Exit_status': row['scheduler'].get('Exit_status'),
        'artifact_exit_code': row.get('artifact_exit_code'),
        'phase_complete_marker': row.get('phase_complete_marker'),
        'passed': row.get('passed'),
        'progress': {key: progress[key] for key in PROGRESS_FIELDS if key in progress},
        'summary': {key: summary[key] for key in SUMMARY_FIELDS if key in summary},
    }


def event_row(unit, row):
    return {
        'unit': unit,
        'job_id': row['handle']['job_id'],
        'scheduler': row['scheduler'],
        'artifact_exit_code': row.get('artifact_exit_code'),
        'phase_complete_marker': row.get('phase_complete_marker'),
        'passed': row.get('passed'),
        'summary': {key: (row.get('summary') or {})[key]
                    for key in SUMMARY_FIELDS if key in (row.get('summary') or {})},
        'progress': row.get('progress') or {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=1200)
    args = parser.parse_args()
    if args.interval != 1200:
        raise ValueError('Frozen monitor interval is 1200 seconds')

    c = load_control()
    state_path = HERE / 'monitor_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else {}
    reported = set(state.get('reported_terminal_jobs', []))
    observations = state.get('observations', {})
    print('MONITOR_STARTED interval_seconds=1200 read_only=true', flush=True)

    while True:
        try:
            snapshot = poll()
            c = load_control()
            c.save(HERE / 'monitor_snapshot.json', snapshot)
            events, current_jobs = [], {}
            for unit, row in snapshot['units'].items():
                job = row['handle']['job_id']
                current_jobs[unit] = job
                q = row['scheduler']
                signature = material_state(row)
                previous = observations.get(job)
                terminal = q.get('query_rc') == 0 and q.get('job_state') in ('F', 'X')

                if terminal and job not in reported:
                    events.append(('MONITOR_TERMINAL', event_row(unit, row)))
                    reported.add(job)
                elif previous is not None and signature != previous:
                    if terminal or q.get('query_rc') == 0:
                        events.append(('MONITOR_UPDATE', event_row(unit, row)))
                observations[job] = signature

            state.update(
                last_checked_utc=snapshot['checked_utc'],
                reported_terminal_jobs=sorted(reported),
                observations=observations,
                jobs=current_jobs,
            )
            if events:
                for label, event in events:
                    print(label + ' ' + json.dumps(event), flush=True)

            failed = [row for row in snapshot['units'].values()
                      if row['scheduler'].get('query_rc') != 0]
            if failed:
                state['query_failures'] = state.get('query_failures', 0) + 1
                state['last_query_error'] = 'scheduler query failed for ' + ', '.join(
                    row['handle']['job_id'] for row in failed)
                if state['query_failures'] == 3:
                    print('MONITOR_QUERY_INCONCLUSIVE ' + json.dumps({
                        'consecutive_failures': 3, 'error': state['last_query_error']}), flush=True)
            else:
                state['query_failures'] = 0
                state.pop('last_query_error', None)

            aggregate = snapshot['units'].get('aggregate')
            if aggregate and aggregate['passed']:
                state['complete'] = True
                c.save(state_path, state)
                print('MONITOR_ALL_REQUIRED_STAGES_PASSED', flush=True)
                return
        except Exception as error:
            state['query_failures'] = state.get('query_failures', 0) + 1
            state['last_query_error'] = type(error).__name__ + ': ' + str(error)
            state['last_checked_utc'] = datetime.now(timezone.utc).isoformat()
            if state['query_failures'] == 3:
                print('MONITOR_QUERY_INCONCLUSIVE ' + json.dumps({
                    'consecutive_failures': 3, 'error': state['last_query_error']}), flush=True)

        c.save(state_path, state)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
