"""Validate existing donor captures/checkpoints in a CPU allocation; no forwards."""
from pathlib import Path
import argparse
import json
import os
from common import guard, save, contexts, stem, PROTOCOL


def tensor(torch, x, shape, name):
    if not isinstance(x, torch.Tensor) or tuple(x.shape) != shape or not bool(torch.isfinite(x).all()):
        raise RuntimeError(f'Invalid {name}')


def main():
    guard()
    import torch
    from checkpointing import atomic_torch_save
    import metrics
    a = argparse.ArgumentParser()
    a.add_argument('--protocol', type=Path, required=True)
    args = a.parse_args()
    p = json.loads(args.protocol.read_text())
    root = Path(p['remote_root'])
    source_protocol = json.loads((Path(p['diagnostic_root']) / 'protocol.json').read_text())
    for key in ('calibration_cases', 'selection_cases', 'test_cases', 'evaluation_seed_indices', 'initial_arm'):
        if p[key] != source_protocol[key]:
            raise RuntimeError(f'Historical cohort/initialization changed: {key}')
    for campaign, job in [(p['diagnostic_root'], j) for j in p['diagnostic_jobs']] + [
        (p['distill_result'].split('/results/')[0], p['distill_job']),
        (p['protected_result'].split('/results/')[0], p['protected_job']),
    ]:
        artifact = Path(campaign) / 'artifacts' / job
        if not (artifact / 'PIPELINE_COMPLETE').is_file() or (artifact / 'exit_code.txt').read_text().strip() != '0':
            raise RuntimeError(f'Historical source did not complete: {job}')
    manifest = []
    for c in contexts(p):
        directory = Path(p['diagnostic_root']) / 'results' / f"shard-{c['shard']}"
        receipt = json.loads((directory / f'{stem(c)}.json').read_text())
        payload = torch.load(directory / f'{stem(c)}.pt', map_location='cpu', weights_only=True)
        if payload['context'] != c or receipt['context'] != c:
            raise RuntimeError('Cached donor context mismatch')
        captures = payload['captures']
        teacher_noise = captures['bf16']['_diagnostic']['action_initial_noise']
        for name in ('bf16', 'parent_self', 'distilled_self'):
            cap = captures[name]
            tensor(torch, cap['video_latents'], (1, 48, 3, 14, 28), name + ' future')
            tensor(torch, cap['action'], (32, 7), name + ' action')
            if not torch.equal(cap['_diagnostic']['action_initial_noise'], teacher_noise):
                raise RuntimeError('Historical donor action noises differ')
            if not torch.equal(cap['video_latents'][:, :, :1], captures['bf16']['video_latents'][:, :, :1]):
                raise RuntimeError('Historical observation clamp differs')
        manifest.append({**c, 'sampler_seed': payload['sampler_seed'], 'cache_path': str(directory / f'{stem(c)}.pt')})
    selected = {}
    for label, directory in [('action_only', p['distill_result']), ('action_latent', p['protected_result'])]:
        directory = Path(directory)
        frozen = json.loads((directory / 'frozen_checkpoint.json').read_text())
        if frozen['selected_epoch'] != 2 or frozen['updates'] != 16 or frozen['test_queries_before_freeze'] != 0:
            raise RuntimeError('Checkpoint is not the paired selected epoch2/16updates')
        overlay = torch.load(directory / 'selected_codebook_overlay.pt', map_location='cpu', weights_only=True)
        if (overlay['source_root'] != p['source_root'] or overlay['source_arm'] != p['initial_arm']
                or len(overlay['module_names']) != 614):
            raise RuntimeError('Overlay does not share the paired parent banks')
        checkpoint = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True, mmap=True)
        state = checkpoint['run_state']
        if not state['frozen'] or state['selected_epoch'] != 2:
            raise RuntimeError('Selected evaluation state is not frozen')
        table = {}
        for row in state['evaluation_rows']:
            if row['epoch'] != 2:
                continue
            key = f"{row['case_id']}:{row['seed_index']}"
            if key in table:
                raise RuntimeError('Duplicate selected action reference')
            value = torch.as_tensor(row['action'], dtype=torch.float32)
            tensor(torch, value, (32, 7), label + ' selected action')
            table[key] = value
        if set(table) != {f"{c['case_id']}:{c['seed_index']}" for c in contexts(p)}:
            raise RuntimeError('Selected references do not cover all60 contexts')
        selected[label] = table
        del overlay, checkpoint, state
    metrics.self_check(torch)
    atomic_torch_save(torch, root / 'selected_actions.pt', {'protocol': PROTOCOL, 'actions': selected})
    save(root / 'input_manifest.json', {'protocol': PROTOCOL, 'contexts': manifest})
    save(root / 'prepare_summary.json', dict(protocol=PROTOCOL, status='complete', job_id=os.environ['PBS_JOBID'],
         context_count=60, test_case_count=22, donor_cache_count=180, selected_action_count=120,
         model_forward_count=0, paired_initialization=True, selected_epoch=2))
    print('PREPARE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
