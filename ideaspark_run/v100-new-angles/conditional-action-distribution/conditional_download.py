"""Download only pinned missing videos selected before model outputs, on CPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def main():
    from allocation_guard import require_allocation
    allocation = require_allocation()
    top = Path('/tc1home/UG/yguo017/v100_newangles_ccds')
    asset = top/'conditional_marginal_ready'
    asset.mkdir(exist_ok=True)
    with (asset/'DOWNLOAD.lock').open('x') as stream:
        json.dump(allocation, stream)
    spec = importlib.util.spec_from_file_location('verified_prepare', top/'artifacts/64758/smolvla_cpu_prepare.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    probe_path = top/'artifacts/64776/asset_probe.json'
    probe = json.loads(probe_path.read_text())
    if not probe['bounded_download_possible'] or probe['total_missing_bytes'] > 500_000_000:
        raise ValueError('Frozen download budget exceeded')
    if probe['revision'] != 'a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4':
        raise ValueError('Wrong dataset revision')
    if helper._sha256(top/'artifacts/64775/selection.json') != probe['selection_sha256']:
        raise ValueError('Frozen selected episodes changed')
    base_path = top/'smolvla/identity.json'
    base = json.loads(base_path.read_text())
    if helper._sha256(base_path) != 'be4a49ebe588a49a29bd26ed98b8a01a648a247e66d45e12a935ac7d8d0c4e64':
        raise ValueError('Base asset identity changed')
    downloaded = []
    for record in probe['missing_records']:
        relative = helper._safe_rel(record['rfilename'])
        if not relative.startswith('videos/') or not record.get('lfs_sha256'):
            raise ValueError('Expected hash-bound video')
        destination = asset/'source_extension'/relative
        saved = helper._download(kind='datasets', repo_id='lerobot/libero',
            revision=probe['revision'], record=record, destination=destination, cap=500_000_000)
        downloaded.append({**saved, 'verified_local_path': str(destination)})
    base['dataset']['video_files'].extend(downloaded)
    base['bounded_extension'] = {'parent_identity_path': str(base_path),
        'parent_identity_sha256': helper._sha256(base_path), 'probe_sha256': helper._sha256(probe_path),
        'allocation': allocation, 'files': downloaded, 'selection_sha256': probe['selection_sha256']}
    with (asset/'identity.json').open('x') as stream:
        json.dump(base, stream, indent=2)
    report = {'status': 'complete', 'allocation': allocation, 'files': downloaded,
        'downloaded_bytes': sum(row['size'] for row in downloaded),
        'identity_path': str(asset/'identity.json'), 'identity_sha256': helper._sha256(asset/'identity.json')}
    (top/'artifacts'/os.environ['SLURM_JOB_ID']/'download.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': 'download_complete', 'bytes': report['downloaded_bytes']}))


if __name__ == '__main__':
    main()
