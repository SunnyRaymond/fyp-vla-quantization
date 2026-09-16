"""Conservative workload estimate, executed in the preflight allocation."""
import argparse
import json
import math
from pathlib import Path
from allocation_guard import require_allocation

p = argparse.ArgumentParser()
p.add_argument('--benchmark', type=Path, required=True)
args = p.parse_args()
allocation = require_allocation()
b = json.loads(args.benchmark.read_text())
assert b['status'] == 'complete' and b['benchmark_only']
pool = max(r['forward_seconds']/r['forward_count'] for r in b['repetitions'])
map8 = max(r['elapsed_seconds'] for r in b['repetitions']) + 6*pool
apply = max(r['apply_seconds'] for r in b['repetitions'])
raw = 176*map8 + b['model_load_seconds'] + 82*pool + 10*apply + 180
conservative = 1.5*raw
deadline = int(math.ceil(conservative/300)*300)
result = {'allocation':allocation,'benchmark_job':allocation['job_id'],
          'pool_seconds':pool,'cal_map8_seconds':map8,'max_unique_cal_maps':176,
          'extra_pool_forwards_reserved':82,'raw_seconds_with_io_reserve':raw,
          'safety_factor':1.5,'conservative_seconds':conservative,
          'suggested_workflow_deadline_seconds':deadline,
          'suggested_slurm_walltime_seconds':deadline+90,
          'within_two_gpu_hour_cap':deadline+90<=7200,
          'note':'Estimate includes load, full CAL search, DEV/replay and I/O; not a guarantee. No research conclusion.'}
(args.benchmark.parent/'budget_estimate.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
