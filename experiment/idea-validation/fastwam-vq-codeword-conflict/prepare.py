"""CPU allocation: stratified uniform vector sample for every occupied codeword."""
import json
import os
from pathlib import Path
from common import PROTOCOL, guard, save


def main():
    guard()
    import numpy as np
    import torch
    p = json.loads((Path(os.environ['ROOT']) / 'protocol.json').read_text())
    root = Path(p['remote_root'])
    reference = json.loads((Path(p['latent_root']) / 'reference_summary.json').read_text())
    if reference['status'] != 'complete' or reference['context_count'] != 60:
        raise RuntimeError('Complete existing teacher references required')
    overlays = [torch.load(Path(p[k]) / 'selected_codebook_overlay.pt', weights_only=True, map_location='cpu')
                for k in ('distill_result', 'protected_result')]
    names = overlays[0]['module_names']
    if names != overlays[1]['module_names'] or len(names) != 614:
        raise RuntimeError('Selected checkpoint target modules differ')
    rng = np.random.default_rng(p['sample_seed'])
    modules, sampled, populations, occupied = {}, 0, 0, 0
    for i, name in enumerate(names):
        bank = torch.load(Path(p['source_root']) / 'results/full/banks' / p['initial_arm'] / f'{i:04d}.pt',
                          map_location='cpu', weights_only=True)
        if bank['module'] != name:
            raise RuntimeError('Bank identity mismatch')
        enc = bank['encoded']
        ids = enc['indices'].reshape(-1, 2).numpy()
        shape = list(enc['shape'])
        if enc['indices'].shape[1] != (shape[1] + 3) // 4:
            raise RuntimeError('Decoder vector padding differs from stored indices')
        choices, counts = [], []
        for book in range(2):
            order = np.argsort(ids[:, book], kind='stable')
            count = np.bincount(ids[:, book], minlength=256)
            offset = np.r_[0, np.cumsum(count)]
            chosen = []
            for c in range(256):
                n = min(int(count[c]), p['vectors_per_codeword'])
                chosen.append(order[offset[c] + rng.choice(int(count[c]), n, replace=False)] if n else np.empty(0, np.int64))
            choices.append(chosen)
            counts.append(count)
        union = np.unique(np.concatenate([v for b in choices for v in b]))
        books = []
        for b in range(2):
            positions = np.full((256, p['vectors_per_codeword']), -1, dtype=np.int64)
            for c, v in enumerate(choices[b]):
                positions[c, :len(v)] = np.searchsorted(union, v)
            books.append(dict(positions=torch.from_numpy(positions), population=torch.from_numpy(counts[b])))
        modules[name] = dict(flat_ids=torch.from_numpy(union), books=books, shape=shape, vectors_per_row=(shape[1] + 3) // 4)
        sampled += sum(len(v) for b in choices for v in b)
        populations += 2 * len(ids)
        occupied += sum(int((c > 0).sum()) for c in counts)
        if i % 100 == 0:
            print('SAMPLE_PLAN', i, name, flush=True)
    payload = dict(protocol=PROTOCOL, module_names=names, modules=modules, sample_seed=p['sample_seed'])
    torch.save(payload, root / 'sample_plan.pt')
    summary = dict(protocol=PROTOCOL, phase='prepare', job_id=os.environ['PBS_JOBID'], status='complete',
                   target_linears=614, occupied_codewords=occupied, sampled_book_memberships=sampled,
                   population_book_memberships=populations, vectors_per_codeword=p['vectors_per_codeword'],
                   latent_weight=reference['latent_weight'], model_forward_count=0)
    save(root / 'prepare_summary.json', summary)
    save(Path(os.environ['ARTIFACTS']) / 'summary.json', summary)


if __name__ == '__main__':
    main()
