"""Extract completed MPC rounds from logs; does not certify saved trajectories."""
import argparse
import json
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
out = a.directory
rounds = [json.loads(line) for line in (out / 'logs.json').read_text().splitlines()
          if line.strip()]
rounds = [r for r in rounds if 'mpc/success_rate' in r]
last = rounds[-1]
step = int(last['step'])
text = (out / 'job.log').read_text(errors='replace')
match = re.search(r'MPC iter ' + str(step-1) + r' Eval -------(.*?)(?=MPC iter|\Z)', text, re.S)
assert match, 'Missing completed-round console log'
block = match.group(1)
success_match = re.search(r"'success': array\(\[(.*?)\]\)", block, re.S)
distance_match = re.search(r"'state_dist': array\(\[(.*?)\]\)", block, re.S)
assert success_match and distance_match, 'Missing per-case metrics'
successes = re.findall(r'True|False', success_match.group(1))
distances = [float(x.strip()) for x in distance_match.group(1).split(',') if x.strip()]
assert len(successes) == len(distances) == 50
records = [{'case_id': i, 'success': s == 'True', 'logged_state_distance': d}
           for i, (s, d) in enumerate(zip(successes, distances))]
count = sum(r['success'] for r in records)
assert abs(count / 50 - last['mpc/success_rate']) < 1e-9
assert abs(sum(distances) / 50 - last['mpc/mean_state_dist']) < 1e-6
times = [json.loads(x) for x in (out / 'planning_times.jsonl').read_text().splitlines() if x.strip()]
times = [r for r in times if int(r['stage'].split('_')[-1]) < step]
summary = {'status': 'stopped_at_user_requested_round', 'completed_mpc_rounds': step,
           'n_evals': 50, 'successes': count, 'success_rate': count / 50,
           'paper_wall_success_rate': 0.96, 'difference_percentage_points': round(count * 2 - 96, 2),
           'max_executed_env_steps': step * 25,
           'cem_seconds': sum(r['seconds'] for r in times),
           'peak_cem_allocated_gib': max(r['peak_allocated_gib'] for r in times),
           'failed_case_ids': [r['case_id'] for r in records if not r['success']],
           'validation': '50 logged case flags and distances agree with aggregate log; no independent trajectory verification',
           'final_trajectory_and_video_export_completed': False,
           'rounds': rounds}
(out / 'stopped_cases_from_log.json').write_text(json.dumps(records, indent=2))
(out / 'stopped_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
