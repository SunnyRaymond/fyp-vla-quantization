"""Summarize archived PBS allocation evidence, without contacting the cluster."""
import json
import re
from datetime import datetime
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    ledger = json.loads((root / 'jobs.json').read_text(encoding='utf-8-sig'))
    rows = []
    for entry in ledger['jobs']:
        path = root / 'artifacts/screen/artifacts' / entry['id'] / 'qstat_final.txt'
        if not path.exists():
            continue
        content = path.read_text()
        def field(name):
            match = re.search(r'^\s*' + re.escape(name) + r' = (.+)$', content, re.M)
            assert match, (path, name)
            return match.group(1).strip()
        assert field('job_state') == 'F'
        h, m, s = map(int, field('resources_used.walltime').split(':'))
        seconds = h * 3600 + m * 60 + s
        # PBS resources_used.ngpus is zero even for these allocated GPU jobs.
        # Read the allocation request and cross-check the executed vnode instead.
        gpus = int(field('Resource_List.ngpus'))
        assert gpus == entry['gpus'] == 1 and 'ngpus=1' in field('exec_vnode')
        row = dict(id=entry['id'], stage=entry['stage'], allocated_gpus=gpus,
                   allocated_seconds=seconds, gpu_hours=gpus * seconds / 3600,
                   exit_status=int(field('Exit_status')),
                   start=field('stime'), end=field('obittime'))
        for key, pattern in [('pbs_sampled_max_gpu_memory_MB', r'maxGpuMemoryUsed=([\d.]+)MB'),
                             ('pbs_sampled_sm_utilization_avg_percent', r'smUtilization_avg=(\d+)%')]:
            match = re.search(pattern, content)
            row[key] = float(match.group(1)) if match else None
        rows.append(row)
        entry.update(status='complete', exit_status=row['exit_status'],
                     allocated_seconds=seconds, start=row['start'], end=row['end'])
    parse = lambda value: datetime.strptime(value, '%a %b %d %H:%M:%S %Y')
    first = min(parse(row['start']) for row in rows)
    last = max(parse(row['end']) for row in rows)
    result = dict(complete=len(rows) == len(ledger['jobs']), jobs=rows,
                  gpu_hours=sum(row['gpu_hours'] for row in rows),
                  allocated_gpu_seconds=sum(row['allocated_seconds'] * row['allocated_gpus'] for row in rows),
                  first_start_cluster_local=first.isoformat(), last_end_cluster_local=last.isoformat(),
                  observed_stage_span_seconds=(last-first).total_seconds(),
                  span_includes='stage gaps, queueing, local checks and transfers between jobs; excludes initial engineering and final analysis',
                  billing_basis='allocated GPUs times PBS resources_used.walltime; excludes previous smoke')
    (root / 'artifacts/screen/resource_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (root / 'jobs.json').write_text(json.dumps(ledger, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'jobs'}, indent=2))


if __name__ == '__main__':
    main()
