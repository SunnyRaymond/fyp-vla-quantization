"""Same-artifact full-panel engine validation, outside measured samples."""
from __future__ import annotations
import gc
import os
import re
from pathlib import Path

from common import (torch_module, read_json, save_json, save_tensors, load_tensors,
                    get_context, endpoint, cpu_tree, restore_bf16, check_tensor_parity,
                    has_time)

VERSION = 'cosmos_engine_full80_v1_output_clone'


class ValidationCheckpointed(RuntimeError):
    pass


def panel_keys(p):
    keys = [(int(case), int(seed)) for case in p['performance']['case_ids']
            for seed in p['performance']['seed_indices']]
    if len(keys) != 80 or len(set(keys)) != 80:
        raise ValueError('Engine parity requires the full frozen 40 x 2 panel')
    return keys


def identity(p, manifest, device):
    return {'format': VERSION, 'protocol': p['protocol'], 'source_job': manifest['source_job'],
            'completed_updates': manifest['completed_updates'], 'device': device,
            'panel': [list(key) for key in panel_keys(p)]}


def valid_receipt(receipt, expected):
    return (all(receipt.get(key) == value for key, value in expected.items())
            and receipt.get('status') == 'passed'
            and receipt.get('checked_queries') == receipt.get('expected_queries') == 80
            and receipt.get('failed_queries') == []
            and receipt.get('cuda_graph', {}).get('capture_errors') == []
            and receipt.get('cuda_graph', {}).get('graph_replays', 0) > 0)


def _snapshot(result):
    if result.denoiser_calls != 5:
        raise RuntimeError('Engine parity endpoint did not execute exactly five denoiser calls')
    return {'action': cpu_tree(result.action),
            'future_image_slots': cpu_tree(result.future_image_slots)}


def _compare(result, reference):
    checks = {'action': check_tensor_parity(result.action, reference['action'])}
    if set(result.future_image_slots) != set(reference['future_image_slots']):
        raise RuntimeError('Reference and engine future-view keys differ')
    checks.update({key: check_tensor_parity(value, reference['future_image_slots'][key])
                   for key, value in result.future_image_slots.items()})
    if result.denoiser_calls != 5:
        raise RuntimeError('Engine parity endpoint did not execute exactly five denoiser calls')
    return {'passed': all(row['passed'] for row in checks.values()), 'checks': checks}


def _check_time(out, expected, checked, phase):
    seconds = float(os.environ.get('COSMOS_SOFT_DEADLINE_SECONDS', 6600))
    if not has_time(seconds, estimate=90):
        save_json(out / 'summary.json', {**expected, 'status': 'checkpointed',
                  'pbs_job_id': os.environ['PBS_JOBID'], 'phase': phase,
                  'checked_queries': checked, 'expected_queries': 80})
        raise ValidationCheckpointed('Engine parity checkpointed before the allocation deadline')


def install_validated(runtime, p, root, artifact_dir):
    """Install only ours engine; reuse full-panel proof plus a live query on later workers."""
    import engine
    from runtime import install_reference
    torch = torch_module()
    root, artifact_dir = Path(root), Path(artifact_dir)
    manifest = read_json(artifact_dir / 'manifest.json')
    device = next(runtime.model.parameters()).device
    props = torch.cuda.get_device_properties(device)
    device_key = re.sub('[^A-Za-z0-9_-]', '_', props.name) + f'_sm{props.major}{props.minor}'
    out = root / 'results/ours/engine_validation' / device_key
    expected = identity(p, manifest, {'name': props.name,
                        'capability': [int(props.major), int(props.minor)]})
    receipt_path = out / 'summary.json'
    previous = read_json(receipt_path) if receipt_path.is_file() else {}
    reuse = valid_receipt(previous, expected)
    keys = panel_keys(p)

    if not reuse:
        # Reference is the same final-update8 artifact, not an additional model arm.
        install_reference(runtime, p, artifact_dir)
        anchor = load_tensors(artifact_dir.parent / 'final_anchor.pt')
        result = endpoint(runtime, get_context(runtime, int(anchor['case_id'])))
        anchor_check = _compare(result, anchor)
        del result
        save_json(out / 'training_anchor.json', {**expected, **anchor_check})
        if not anchor_check['passed']:
            raise RuntimeError('Exported artifact reference differs from the final trained anchor')
        for index, (case, seed) in enumerate(keys):
            path = out / 'reference' / f'{case}_{seed}.pt'
            if path.is_file():
                old = load_tensors(path)
                if old.get('identity') != expected or old.get('key') != [case, seed]:
                    raise RuntimeError('Stored engine reference belongs to another artifact/panel')
                del old
                continue
            _check_time(out, expected, index, 'reference')
            result = endpoint(runtime, get_context(runtime, case, seed))
            save_tensors(path, {'identity': expected, 'key': [case, seed], **_snapshot(result)})
            del result
            print(f'ENGINE_REFERENCE {index + 1}/80 case={case} seed={seed}', flush=True)

    restore_bf16(runtime)
    runtime.__dict__.pop('_ours_reference_holds', None)
    gc.collect()
    torch.cuda.empty_cache()
    engine_out = root / 'results/ours/engine' / device_key
    engine.install(runtime, p, artifact_dir, root, engine_out)
    active = runtime.ours_vqwam_engine
    engine.preflight(runtime, p, artifact_dir, root, engine_out)
    checks = []
    for index, (case, seed) in enumerate(keys[:1] if reuse else keys):
        _check_time(out, expected, index, 'engine')
        reference = load_tensors(out / 'reference' / f'{case}_{seed}.pt')
        if reference.get('identity') != expected or reference.get('key') != [case, seed]:
            raise RuntimeError('Engine reference identity/panel differs')
        try:
            result = endpoint(runtime, get_context(runtime, case, seed))
        except Exception as exc:
            save_json(out / ('failure_' + os.environ['PBS_JOBID'] + '.json'), {
                **expected, 'pbs_job_id': os.environ['PBS_JOBID'],
                'phase': 'engine_query', 'query_index': index,
                'case_id': case, 'seed_index': seed,
                'error': f'{type(exc).__name__}: {exc}',
                'cuda_graph': active.graphs.metadata()})
            raise
        row = {'case_id': case, 'seed_index': seed, **_compare(result, reference)}
        del result, reference
        checks.append(row)
        save_json(out / 'queries' / f'{case}_{seed}.json', {**expected, **row,
                  'pbs_job_id': os.environ['PBS_JOBID']})
        if not row['passed']:
            save_json(out / 'summary.json', {**expected, 'status': 'failed',
                      'checked_queries': index + 1, 'expected_queries': 80,
                      'failed_queries': [row], 'pbs_job_id': os.environ['PBS_JOBID']})
            raise RuntimeError('Packed engine action/future differs from the same artifact reference')
        print(f'ENGINE_PARITY {index + 1}/{1 if reuse else 80} case={case} seed={seed}', flush=True)
    graph = active.graphs.metadata()
    if graph['capture_errors'] or graph['graph_replays'] <= 0:
        raise RuntimeError('Full endpoint parity did not establish a working CUDA graph path')
    receipt = previous if reuse else {**expected, 'status': 'passed',
        'checked_queries': 80, 'expected_queries': 80, 'failed_queries': [],
        'pbs_job_id': os.environ['PBS_JOBID'], 'queries': checks, 'cuda_graph': graph}
    active.mark_validated(receipt)
    if not reuse:
        save_json(receipt_path, receipt)
    save_json(out / ('process_' + os.environ['PBS_JOBID'] + '_' + str(os.getpid()) + '.json'),
              {'full80_receipt': str(receipt_path), 'reused_full80_receipt': reuse,
               'live_checks': checks, 'cuda_graph': active.graphs.metadata()})
    runtime.install_metadata = active.metadata()
    return runtime.install_metadata
