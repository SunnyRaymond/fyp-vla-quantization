"""Read a bounded metadata sample inside a real CCDS CPU allocation."""
from pathlib import Path
import json
import os
import struct
from allocation_guard import require_allocation

allocation = require_allocation()
import pyarrow.parquet as pq

top = Path.home() / 'v100_newangles_ccds'
asset = top / 'smolvla'
out = top / 'artifacts' / os.environ['SLURM_JOB_ID']
out.mkdir(parents=True, exist_ok=True)
tasks = pq.read_table(asset / 'source_subset/meta/tasks.parquet')
episode = pq.read_table(asset / 'source_subset/meta/episodes/chunk-000/file-000.parquet')
info = json.loads((asset / 'source_subset/meta/info.json').read_text())
report = {
    'allocation': allocation,
    'tasks_schema': str(tasks.schema), 'tasks_first_rows': tasks.slice(0, 5).to_pylist(),
    'episode_schema': str(episode.schema), 'episode_first_rows': episode.slice(0, 2).to_pylist(),
    'info': {k: info.get(k) for k in ['features', 'data_path', 'video_path', 'fps', 'codebase_version']},
    'lock': json.loads((asset / 'PREPARATION.lock').read_text()),
    'model_files': sorted(p.name for p in (asset / 'model').iterdir() if p.is_file()),
    'base_files': sorted(p.name for p in (asset / 'base_vlm_metadata').iterdir() if p.is_file()),
}
normalizer = asset / 'model/policy_preprocessor_step_5_normalizer_processor.safetensors'
with normalizer.open('rb') as stream:
    length = struct.unpack('<Q', stream.read(8))[0]
    if length > 65536:
        raise ValueError('Unexpected normalizer header size')
    report['normalizer_header'] = json.loads(stream.read(length))
report['checkpoint_config'] = json.loads((asset / 'model/config.json').read_text())
report['checkpoint_preprocessor'] = json.loads((asset / 'model/policy_preprocessor.json').read_text())
(out / 'metadata_probe.json').write_text(json.dumps(report, indent=2))
