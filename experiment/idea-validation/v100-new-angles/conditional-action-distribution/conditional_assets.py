"""Inspect fixed selected video requirements on cluster CPU; no downloads."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def main():
    from allocation_guard import require_allocation
    allocation = require_allocation()
    import pyarrow.parquet as pq
    top = Path('/tc1home/UG/yguo017/v100_newangles_ccds')
    out = top / 'artifacts' / os.environ['SLURM_JOB_ID']
    spec = importlib.util.spec_from_file_location('verified_prepare', top/'artifacts/64758/smolvla_cpu_prepare.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    base = json.loads((top/'smolvla/identity.json').read_text())
    selection_path = top/'artifacts/64775/selection.json'
    selection = json.loads(selection_path.read_text())
    root = top/'smolvla/source_subset'
    records = base['dataset']['metadata_files']
    episode_records = [row for row in records if row['rfilename'].startswith('meta/episodes/') and row['rfilename'].endswith('.parquet')]
    rows = []
    for record in episode_records:
        path = root/record['rfilename']
        if helper._sha256(path) != record['downloaded_sha256']:
            raise ValueError('Episode metadata hash changed')
        rows.extend(pq.read_table(path).to_pylist())
    episodes = {int(row['episode_index']): row for row in rows}
    info = json.loads((root/'meta/info.json').read_text())
    cameras = ['observation.images.image', 'observation.images.image2']
    needed = set()
    for selected in selection['selected']:
        row = episodes[selected['episode_index']]
        for camera in cameras:
            prefix = f'videos/{camera}/'
            needed.add(helper._format_path(info['video_path'], video_key=camera,
                chunk_index=row[prefix+'chunk_index'], file_index=row[prefix+'file_index'],
                episode_chunk=row[prefix+'chunk_index'], file_chunk=row[prefix+'file_index']))
    existing = {row['rfilename']: row for row in base['dataset']['video_files']}
    missing = sorted(needed - set(existing))
    revision = 'a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4'
    payload = helper._api('datasets', 'lerobot/libero', revision)
    if payload['sha'] != revision:
        raise ValueError('Hub revision differs')
    hub = helper._sibling_records(payload)
    missing_records = [hub[name] for name in missing]
    verified_sizes = all(isinstance(row.get('size'), int) and row['size'] > 0 and row.get('lfs_sha256') for row in missing_records)
    total = sum(row['size'] for row in missing_records) if verified_sizes else None
    report = {'schema': 'conditional-fixed-assets-probe-v1', 'allocation': allocation,
        'selection_sha256': hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        'selected': selection['selected'], 'required_video_files': sorted(needed),
        'missing_records': missing_records, 'total_missing_bytes': total,
        'bounded_download_possible': bool(verified_sizes and total <= 500_000_000),
        'download_ran': False, 'revision': revision}
    (out/'asset_probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'missing_count': len(missing), 'total_missing_bytes': total, 'bounded_download_possible': report['bounded_download_possible']}))


if __name__ == '__main__':
    main()
