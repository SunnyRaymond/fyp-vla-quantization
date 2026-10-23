"""Fill missing real Plus instructions once, before loading the policy in the smoke job."""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path
import pickle
import socket
import time


def main():
    root = Path(os.environ['COSMOS_ROOT'])
    from cosmos_runtime import guard
    guard(root)
    requirement = json.loads((root/'language_requirements.json').read_text())
    assets = json.loads((root/'asset_paths.json').read_text())
    target = root/'results/text_embeddings.pkl'
    target.parent.mkdir(parents=True, exist_ok=True)
    source = target if target.exists() else Path(requirement['source_cache'])
    with source.open('rb') as stream:
        cache = pickle.load(stream)
    missing = [text for text in requirement['instructions'] if text not in cache]
    import torch
    def save_cache():
        temporary = target.with_suffix('.pkl.tmp')
        with temporary.open('wb') as stream:
            pickle.dump(cache, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(target)

    if missing:
        from cosmos_policy._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder
        encoder = CosmosT5TextEncoder(model_name=assets['text_encoder_snapshot'], device='cuda', local_files_only=True)
        for index, text in enumerate(missing):
            started = float(os.environ['COSMOS_JOB_START_EPOCH'])
            if time.time()-started > 6300:
                save_cache()
                raise RuntimeError('Text preparation reached the smoke allocation save reserve; partial cache retained')
            cache[text] = encoder.encode_prompts([text], max_length=512).detach().cpu()
            # Avoid rewriting the growing full cache after every instruction.
            if (index+1) % 32 == 0 or index+1 == len(missing):
                save_cache()
            print(f'TEXT_EMBEDDING_COMPLETE index={index+1}/{len(missing)}', flush=True)
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    if not target.exists():
        save_cache()
    if any(text not in cache for text in requirement['instructions']):
        raise RuntimeError('The complete frozen language cohort is not covered')
    receipt = dict(status='complete', required_instructions=len(requirement['instructions']),
                   new_text_forwards=len(missing), cache=str(target), entries=len(cache),
                   pbs_job_id=os.environ['PBS_JOBID'])
    (root/'results/text_embeddings_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
