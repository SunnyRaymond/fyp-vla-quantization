"""Download frozen policy/tokenizer assets only inside a PBS CPU allocation."""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket
import pickle
import sys
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import GatedRepoError


def main():
    if not os.environ.get('PBS_JOBID') or 'login' in socket.gethostname().lower():
        raise RuntimeError('A real compute allocation is required')
    nodes = Path(os.environ['PBS_NODEFILE']).read_text().split()
    if socket.gethostname().split('.')[0] not in {n.split('.')[0] for n in nodes}:
        raise RuntimeError('Host is outside the approved allocation')
    root = Path(os.environ['ROOT'])
    path = root/'asset_revisions.json'
    if path.exists():
        revisions = json.loads(path.read_text())
    else:
        api = HfApi()
        revisions = {repo: api.model_info(repo).sha for repo in (
            'nvidia/Cosmos-Policy-LIBERO-Predict2-2B', 'nvidia/Cosmos-Predict2-2B-Video2World')}
        path.write_text(json.dumps(revisions, indent=2)+'\n')
    policy = snapshot_download('nvidia/Cosmos-Policy-LIBERO-Predict2-2B',
                               revision=revisions['nvidia/Cosmos-Policy-LIBERO-Predict2-2B'])
    tokenizer_source = root/'models'/'tokenizer.pth'
    if tokenizer_source.is_file():
        tokenizer_target = (Path(HF_HUB_CACHE)/'models--nvidia--Cosmos-Predict2-2B-Video2World'/'snapshots'/
                            revisions['nvidia/Cosmos-Predict2-2B-Video2World']/'tokenizer'/'tokenizer.pth')
        tokenizer_target.parent.mkdir(parents=True, exist_ok=True)
        if tokenizer_target.is_file():
            if os.path.samefile(tokenizer_source, tokenizer_target):
                pass
            else:
                tokenizer_target.unlink()
                os.link(tokenizer_source, tokenizer_target)
        else:
            if tokenizer_target.is_symlink():
                tokenizer_target.unlink()
            os.link(tokenizer_source, tokenizer_target)
        tokenizer = str(tokenizer_target)
    else:
        try:
            tokenizer = hf_hub_download('nvidia/Cosmos-Predict2-2B-Video2World', 'tokenizer/tokenizer.pth',
                                        revision=revisions['nvidia/Cosmos-Predict2-2B-Video2World'])
        except GatedRepoError:
            tokenizer = None
            print('TOKENIZER_AUTH_REQUIRED: official gated repository requires account access; preparing independent resources only', flush=True)
    # Offline resolution follows the revision selected for this task, not a later main revision.
    result = dict(policy_snapshot=policy, tokenizer=tokenizer, revisions=revisions)
    # Check whether the already-authorized policy checkpoint itself contains a tokenizer.
    # This reads only tensor metadata through mmap, inside this CPU allocation.
    import torch
    expected_env = (root/'env').resolve()
    if Path(sys.prefix).resolve() != expected_env:
        raise RuntimeError(f'Asset prep is not using the frozen environment: {sys.prefix}')
    try:
        Path(torch.__file__).resolve().relative_to(expected_env)
    except ValueError as exc:
        raise RuntimeError(f'Torch was not imported from the frozen environment: {torch.__file__}') from exc
    from collections import Counter
    checkpoints = list(Path(policy).glob('*.pt'))
    if len(checkpoints) != 1:
        raise RuntimeError('Expected one official policy checkpoint')
    checkpoint = torch.load(checkpoints[0], map_location='cpu', mmap=True, weights_only=True)
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    if not isinstance(state, dict):
        raise RuntimeError('Unsupported policy checkpoint container')
    prefixes = Counter(str(key).split('.')[0] for key in state)
    (root/'checkpoint_contents.json').write_text(json.dumps(dict(top_level=list(checkpoint)[:10], tensor_key_prefixes=dict(prefixes)), indent=2)+'\n')
    del state, checkpoint
    # Discover the real Plus instructions without rendering or loading a policy.
    plus = Path('/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005')
    os.environ['LIBERO_CONFIG_PATH'] = str(plus/'libero-config')
    sys.path[:0] = [str(plus/'libero-plus-compat'), str(plus/'LIBERO-plus'), str(plus/'deps')]
    import wand
    import skimage
    from libero.libero import benchmark
    manifest = json.loads(Path('/scratch/users/ntu/yguo017/fastwam-rotation-baselines-20261008/results/manifest.json').read_text())
    suites = {}
    instructions = set()
    for row in manifest['variants']:
        if row['suite'] not in suites:
            suites[row['suite']] = benchmark.get_benchmark_dict()[row['suite']]()
        task = suites[row['suite']].get_task(int(row['task_id']))
        if task.name != row['task_name']:
            raise RuntimeError('Frozen Plus task identity differs')
        instructions.add(task.language)
    instructions.add('open the middle drawer of the cabinet')
    protocol = json.loads((root/'protocol.json').read_text())
    source = Path(protocol['calibration']['source_root'])/'data/inputs'
    for case_id in protocol['calibration']['case_ids']:
        metadata = json.loads((source/f'case_{case_id}.json').read_text())
        description = metadata.get('description')
        if not isinstance(description, str) or not description.strip():
            raise RuntimeError(f'C8 case {case_id} has no original instruction')
        instructions.add(description)
    cache_file = Path(policy)/'libero_t5_embeddings.pkl'
    with cache_file.open('rb') as stream:
        cache = pickle.load(stream)
    missing = sorted(instructions-set(cache))
    requirement = dict(instructions=sorted(instructions), missing=missing, source_cache=str(cache_file))
    (root/'language_requirements.json').write_text(json.dumps(requirement, indent=2)+'\n')
    if missing:
        repo = 'google-t5/t5-11b'
        if repo not in revisions:
            info = HfApi().model_info(repo)
            revisions[repo] = info.sha
            path.write_text(json.dumps(revisions, indent=2)+'\n')
        filenames = HfApi().list_repo_files(repo, revision=revisions[repo])
        weights = ([name for name in filenames if name.endswith('.safetensors')]
                   or [name for name in filenames if name.startswith('pytorch_model') and name.endswith('.bin')])
        if not weights:
            raise RuntimeError('No supported official T5 weight files were found')
        patterns = weights + ['config.json', 'generation_config.json', '*token*', 'spiece.model', '*.index.json']
        result['text_encoder_snapshot'] = snapshot_download(repo, revision=revisions[repo], allow_patterns=patterns)
    result['language_requirements'] = str(root/'language_requirements.json')
    result['status'] = 'ready' if tokenizer else 'awaiting_tokenizer_access'
    # The upstream resolver requests revision main. Bind it to the recorded revision in this private task cache.
    for repo, revision in revisions.items():
        refs = Path(HF_HUB_CACHE)/('models--'+repo.replace('/', '--'))/'refs'
        refs.mkdir(parents=True, exist_ok=True)
        (refs/'main').write_text(revision)
    output = root/'asset_paths.json'
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(output)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
