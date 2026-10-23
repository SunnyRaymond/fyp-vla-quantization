"""Inspect original demonstration availability only inside a PBS CPU allocation."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import socket

job = os.environ.get('PBS_JOBID')
nodefile = os.environ.get('PBS_NODEFILE')
host = socket.gethostname().split('.')[0]
if not job or not nodefile or 'login' in host.lower() or host not in {n.split('.')[0] for n in Path(nodefile).read_text().split()}:
    raise RuntimeError('Approved PBS compute allocation required')

root = Path(os.environ['ROOT'])
p = json.loads((root / 'protocol.json').read_text())
repo = Path(p['data']['original_repo'])
tree = ast.parse((repo / 'libero/benchmark/libero_suite_task_map.py').read_text())
task_map = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in node.targets))
libraries = {name: importlib.util.find_spec(name) is not None
             for name in ('h5py', 'fsspec', 'huggingface_hub', 'scipy')}
report = {'status': 'inventory_complete', 'job_id': job, 'hostname': host,
          'protocol': p['protocol'], 'libraries': libraries, 'suites': {}, 'missing': []}
if libraries['h5py']:
    import h5py
for suite in p['suites']:
    rows = []
    for task in task_map[suite]:
        candidates = [Path(d) / suite / (task + '_demo.hdf5') for d in p['data']['dataset_roots']]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        row = {'task': task, 'path': str(path) if path else None}
        if path is None:
            report['missing'].append({'suite': suite, 'task': task, 'candidates': [str(x) for x in candidates]})
        elif libraries['h5py']:
            with h5py.File(path, 'r') as handle:
                demos = sorted(handle['data'])
                row['demo_count'] = len(demos)
                first = handle['data'][demos[0]]
                row['keys'] = list(first)
                row['states_shape'] = list(first['states'].shape) if 'states' in first else None
                row['observation_keys'] = list(first['obs']) if 'obs' in first else []
                row['attributes'] = list(first.attrs)
        rows.append(row)
    report['suites'][suite] = rows
target = root / 'source_inventory.json'
temporary = target.with_suffix('.json.tmp')
temporary.write_text(json.dumps(report, indent=2) + '\n')
temporary.replace(target)
print('SOURCE_INVENTORY_COMPLETE ' + json.dumps({'missing_tasks': len(report['missing']), 'libraries': libraries}), flush=True)
