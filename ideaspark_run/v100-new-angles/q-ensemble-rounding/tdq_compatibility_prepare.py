"""Bounded official-checkpoint compatibility selection, without inference."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request


TOP = Path('/tc1home/UG/yguo017/v100_newangles_ccds')
REV = '73a50e2719ed8258c72c7d1fefd23b781d66e35e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    from allocation_guard import require_allocation
    allocation = require_allocation()
    out = TOP/'artifacts'/os.environ['SLURM_JOB_ID']
    dest = TOP/'tdq_compatible_checkpoint'
    if dest.exists():
        raise ValueError('Refusing to overwrite compatibility assets')
    dest.mkdir()
    (dest/'PREPARATION.lock').write_text(json.dumps(allocation))
    parent = TOP/'tdmpc2_q_coupling_ready5'
    parent_manifest = json.loads((parent/'manifest.json').read_text())
    if (parent_manifest['schema'] != 'tdmpc2-q-coupling-preparation-v1'
            or parent_manifest['task'] != 'cartpole-balance'
            or parent_manifest['asset_root'] != str(parent)
            or parent_manifest['config']['resolved_task'] != 'cartpole-balance'
            or parent_manifest['source']['commit'] != 'e9f59321933cbc8e11a002b842adc7d4ffae8ff1'):
        raise ValueError('Unexpected parent asset identity')
    spec = json.loads((out/'tdq_checkpoint_candidates.json').read_text())
    if spec['revision'] != REV or [r['seed'] for r in spec['candidates']] != [2,3]:
        raise ValueError('Candidate order or revision differs from pre-output decision')
    if spec['repository'] != 'nicklashansen/tdmpc2' or spec['subdirectory'] != 'dmcontrol':
        raise ValueError('Unexpected candidate repository')
    source = parent/'source'
    for relative, identity in parent_manifest['source']['selected_files'].items():
        if sha(source/relative) != identity['sha256']:
            raise ValueError('Pinned source file changed before compatibility check')
    sys.path.insert(0, str(source/'tdmpc2'))
    import torch
    import tdq_screen as helper
    from common.layers import api_model_conversion
    from common.world_model import WorldModel
    cfg, resolved = helper._resolve_config(source, source/'tdmpc2/config.yaml', dest/'selected.pt', out)
    report = {'allocation':allocation, 'parent_manifest_sha256':sha(parent/'manifest.json'), 'candidate_spec':spec, 'attempts':[], 'inference':False, 'CPU_model_constructed_for_strict_load_only':True, 'selected':None}
    for candidate in spec['candidates']:
        filename = f"cartpole-balance-{candidate['seed']}.pt"
        if candidate['filename'] != filename:
            raise ValueError('Candidate filename mismatch')
        if candidate['size'] > 40*1024*1024 or len(candidate['sha256']) != 64:
            raise ValueError('Unbounded checkpoint metadata')
        target = dest/filename
        url = f'https://huggingface.co/nicklashansen/tdmpc2/resolve/{REV}/dmcontrol/{filename}?download=true'
        h, size = hashlib.sha256(), 0
        with urllib.request.urlopen(url, timeout=60) as response, target.with_suffix('.part').open('xb') as stream:
            for chunk in iter(lambda: response.read(1024*1024), b''):
                size += len(chunk)
                if size > candidate['size']:
                    raise ValueError('Checkpoint exceeds pinned size')
                stream.write(chunk)
                h.update(chunk)
        if size != candidate['size'] or h.hexdigest() != candidate['sha256']:
            raise ValueError('Checkpoint hash/size mismatch')
        target.with_suffix('.part').rename(target)
        row = {'seed':candidate['seed'], 'path':str(target), 'size':size, 'sha256':h.hexdigest()}
        try:
            payload = torch.load(target, map_location='cpu', weights_only=False)
            state = helper._unwrap_checkpoint(payload, torch)
            row['original_state_shapes'] = {k:list(v.shape) if torch.is_tensor(v) else type(v).__name__ for k,v in state.items()}
            model = WorldModel(cfg).to('cpu')
            converted = api_model_conversion(model.state_dict(), state)
            incompatible = model.load_state_dict(converted, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError('Non-strict key coverage')
            row['strict_load'] = True
        except Exception as exc:
            row.update(strict_load=False, error=str(exc)[:12000])
        report['attempts'].append(row)
        (out/'compatibility.json').write_text(json.dumps(report,indent=2))
        if not row['strict_load']:
            continue
        manifest = copy.deepcopy(parent_manifest)
        manifest['asset_root'] = str(dest)
        manifest['source_asset_root'] = str(source)
        manifest['parent_manifest'] = {'path':str(parent/'manifest.json'), 'sha256':report['parent_manifest_sha256'], 'preparation_job':'64790'}
        manifest['checkpoint'].update(path=f'dmcontrol/{filename}', url=url, size=size, sha256=h.hexdigest(), expected_lfs_oid=h.hexdigest(), expected_size=size, torch_loaded=True)
        manifest['checkpoint']['compatibility_cpu_job'] = os.environ['SLURM_JOB_ID']
        observations = (parent/manifest['observations']['path']).resolve()
        if not observations.is_relative_to(parent.resolve()):
            raise ValueError('Observation path escaped parent allocation assets')
        if sha(observations) != manifest['observations']['sha256']:
            raise ValueError('Prepared observations changed')
        shutil.copy2(observations,dest/'observations.npz')
        manifest['observations']['path'] = 'observations.npz'
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
        report['selected'] = {**candidate,'checkpoint':str(target),'manifest':str(dest/'manifest.json'),'manifest_sha256':sha(dest/'manifest.json')}
        break
    report['status'] = 'complete' if report['selected'] else 'resource_blocked_no_compatible_official_candidate'
    (out/'compatibility.json').write_text(json.dumps(report,indent=2))
    if report['selected']:
        shutil.copy2(dest/'manifest.json',out/'manifest.json')
    print(json.dumps({'status':report['status'],'selected':report['selected']}))


if __name__ == '__main__':
    main()
