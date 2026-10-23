"""Read-only exact-handle polling every twenty minutes; never submit or change PBS jobs."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('anchor_monitor_control', HERE / 'control.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


def poll():
    spec.loader.exec_module(c)
    jump, transport = c.remote.direct_connect()
    try:
        with c.remote.sftp_client(transport) as sftp:
            handles = c.read_json(sftp, c.ROOT + '/control/handles.json') or {}
            rows = {unit: c.observe(transport, sftp, unit, handle) for unit, handle in handles.items()}
            return {'checked_utc': datetime.now(timezone.utc).isoformat(), 'units': rows}
    finally:
        transport.close()
        if jump:
            jump.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--interval', type=int, default=1200)
    args = parser.parse_args()
    if args.interval != 1200:
        raise ValueError('User-frozen monitor interval is twenty minutes')
    state_path = HERE / 'monitor_state.json'
    state = json.loads(state_path.read_text()) if state_path.is_file() else {'reported_terminal_jobs': []}
    reported = set(state.get('reported_terminal_jobs', []))
    print('MONITOR_STARTED interval_seconds=1200 read_only=true', flush=True)
    while True:
        try:
            snapshot = poll(); c.save(HERE / 'monitor_snapshot.json', snapshot)
            events = []
            for unit, row in snapshot['units'].items():
                q = row['scheduler']; job = row['handle']['job_id']
                if q['query_rc'] == 0 and q['job_state'] in ('F', 'X') and job not in reported:
                    events.append({'unit': unit, 'job_id': job, 'scheduler': q, 'passed': row['passed'],
                                   'progress': row['progress'], 'artifact_exit_code': row['artifact_exit_code']})
                    reported.add(job)
            state.update(last_checked_utc=snapshot['checked_utc'], query_failures=0,
                         reported_terminal_jobs=sorted(reported),
                         jobs={unit: row['handle']['job_id'] for unit, row in snapshot['units'].items()})
            if events:
                print('MONITOR_TERMINAL ' + json.dumps(events), flush=True)
            final = snapshot['units'].get('closedloop_aggregate')
            fixed = snapshot['units'].get('aggregate')
            if final and fixed and final['passed'] and fixed['passed']:
                state['complete'] = True; c.save(state_path, state)
                print('MONITOR_ALL_REQUIRED_STAGES_PASSED', flush=True)
                return
        except Exception as error:
            state['query_failures'] = state.get('query_failures', 0) + 1
            state['last_query_error'] = type(error).__name__ + ': ' + str(error)
            state['last_checked_utc'] = datetime.now(timezone.utc).isoformat()
            if state['query_failures'] == 3:
                print('MONITOR_QUERY_INCONCLUSIVE ' + json.dumps({'consecutive_failures': 3, 'error': state['last_query_error']}), flush=True)
        c.save(state_path, state)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
