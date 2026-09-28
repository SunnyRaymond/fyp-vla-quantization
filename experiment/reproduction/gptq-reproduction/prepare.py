import json
import random
from pathlib import Path
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import AutoTokenizer

root = Path(__file__).resolve().parent
api = HfApi()
manifest = {'seed': 0, 'nsamples': 128, 'seqlen': 2048, 'models': {}}
for model in ['facebook/opt-125m', 'facebook/opt-6.7b']:
    revision = api.model_info(model).sha
    path = snapshot_download(model, revision=revision, allow_patterns=['*.json', '*.txt', '*.bin'], max_workers=4)
    manifest['models'][model.split('/')[-1]] = {'repo': model, 'revision': revision, 'path': path}
    print('MODEL_READY', model, revision, flush=True)
c4rev = api.dataset_info('allenai/c4').sha
c4path = hf_hub_download('allenai/c4', 'en/c4-train.00000-of-01024.json.gz', repo_type='dataset', revision=c4rev)
c4 = load_dataset('json', data_files=c4path, split='train')
wikirev = api.dataset_info('Salesforce/wikitext').sha
files = sorted(f for f in api.list_repo_files('Salesforce/wikitext', repo_type='dataset', revision=wikirev) if f.startswith('wikitext-2-raw-v1/test-') and f.endswith('.parquet'))
assert files, 'WikiText test files missing'
wikipaths = [hf_hub_download('Salesforce/wikitext', f, repo_type='dataset', revision=wikirev) for f in files]
wiki = load_dataset('parquet', data_files=wikipaths, split='train')
manifest['datasets'] = {'c4': {'revision': c4rev, 'file': 'en/c4-train.00000-of-01024.json.gz'}, 'wikitext': {'revision': wikirev, 'files': files}}
for name, info in manifest['models'].items():
    tokenizer = AutoTokenizer.from_pretrained(info['path'], use_fast=False)
    random.seed(0)
    samples = []
    for _ in range(128):
        while True:
            row = random.randint(0, len(c4) - 1)
            tokens = tokenizer(c4[row]['text'], return_tensors='pt').input_ids
            if tokens.shape[1] > 2048:
                break
        start = random.randint(0, tokens.shape[1] - 2048 - 1)
        samples.append(tokens[:, start:start + 2048].clone())
    test = tokenizer('\n\n'.join(wiki['text']), return_tensors='pt').input_ids
    torch.save({'calibration': samples, 'test': test}, root / f'{name}-data.pt')
    info['test_tokens'] = test.numel()
    print('DATA_READY', name, test.numel(), flush=True)
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
(root / 'PREPARED').touch()
