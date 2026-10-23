"""Paired sampled-future and teacher-future diagnostics on frozen checkpoints."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import signal
import socket
import time
import traceback


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def guard():
    host = socket.gethostname().split('.')[0]
    nodefile = os.environ.get('PBS_NODEFILE')
    if not os.environ.get('PBS_JOBID') or not nodefile or 'login' in host.lower():
        raise RuntimeError('A genuine PBS compute allocation is required')
    nodes = {n.split('.')[0] for n in Path(nodefile).read_text().split()}
    if host not in nodes:
        raise RuntimeError('This host is outside the allocated node list')


class FrozenWeights:
    """Decode both fixed VQ checkpoints once; keep three weight sets on CPU."""
    def __init__(self, runtime, protocol):
        import codebook
        import smooth_vq as core
        from run_experiment import qinput
        torch = runtime.torch
        self.torch, self.modules = torch, runtime.q.modules
        if len(self.modules) != protocol['target_linears']:
            raise RuntimeError('Expected the frozen 614 target Linears')
        runtime.q.disable()
        self.forwards = {n: m.forward for n, m in self.modules.items()}
        self.weights = {'bf16': {n: m.weight.detach().cpu().clone() for n, m in self.modules.items()},
                        'parent': {}, 'distilled': {}}
        self.transforms, self.seen, self.current = {}, set(), None
        distill = Path(protocol['distill_result'])
        frozen = json.loads((distill / 'frozen_checkpoint.json').read_text())
        summary = json.loads((distill / 'summary.json').read_text())
        if (frozen['selected_epoch'] != 2 or frozen['updates'] != 16
                or frozen['test_queries_before_freeze'] != 0 or summary['status'] != 'complete'
                or summary['job_id'] != protocol['distill_job']):
            raise RuntimeError('The selected epoch-2 checkpoint is not the frozen completed run')
        overlay = torch.load(distill / 'selected_codebook_overlay.pt', map_location='cpu', weights_only=True)
        if (overlay['module_names'] != list(self.modules)
                or overlay['source_root'] != protocol['source_root']
                or overlay['source_arm'] != protocol['initial_arm']):
            raise RuntimeError('Selected overlay differs from the declared source banks')
        for index, (name, mod) in enumerate(self.modules.items()):
            bank = torch.load(Path(protocol['source_root']) / 'results/full/banks'
                              / protocol['initial_arm'] / f'{index:04d}.pt',
                              map_location='cuda', weights_only=True)
            if bank['module'] != name or tuple(bank['encoded']['shape']) != tuple(mod.weight.shape):
                raise RuntimeError(f'Bank identity differs at {name}')
            encoding = bank['encoded']
            self.transforms[name] = bank['transform']
            self.weights['parent'][name] = codebook.decode_vq(encoding).cpu()
            masters = overlay['master_state'][name]
            tuned = dict(encoding)
            tuned['books'] = masters['books'].to(device='cuda', dtype=torch.bfloat16)
            active = (encoding['row_scales'] > 0).float()
            tuned['row_scales'] = (masters['log_scales'].to('cuda').exp() * active).to(torch.bfloat16)
            self.weights['distilled'][name] = codebook.decode_vq(tuned).cpu()

            def forward(x, _name=name, _mod=mod):
                self.seen.add(_name)
                if self.current == 'bf16':
                    return self.forwards[_name](x)
                return torch.nn.functional.linear(qinput(torch, core, x, self.transforms[_name]),
                                                  _mod.weight, _mod.bias)

            mod.forward = forward
            if index % 100 == 0:
                print(f'FROZEN_WEIGHT_CACHE {index}/614', flush=True)
            del bank, encoding, tuned
        del overlay
        gc.collect()

    def activate(self, arm):
        if self.current != arm:
            for name, mod in self.modules.items():
                mod.weight.data = self.weights[arm][name].to('cuda')
            self.current = arm
        self.seen.clear()

    def close(self):
        for name, mod in self.modules.items():
            mod.forward = self.forwards[name]


def selected_references(rows, epoch):
    table = {}
    for row in rows:
        if row['epoch'] != epoch:
            continue
        key = row['case_id'], row['seed_index']
        if key in table:
            raise RuntimeError('Duplicate selected-epoch action reference')
        table[key] = row
    return table


def cached_actions(torch, protocol, contexts, source):
    root = Path(protocol['source_root']) / 'results/full'
    tables = {'bf16': {}, 'parent': {}, 'distilled': {}}
    for line in (root / 'actions.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        arm = {'bf16': 'bf16', protocol['initial_arm']: 'parent'}.get(row['arm'])
        if arm:
            key = row['case_id'], row['seed_index']
            if key in tables[arm]:
                raise RuntimeError('Duplicate source action reference')
            tables[arm][key] = row
    for context in contexts:
        case, seed = context['case_id'], context['seed_index']
        key = case, seed
        if key not in tables['bf16']:
            capture = torch.load(root / 'sensitivity' / f'case_{case}.pt', map_location='cpu',
                                 weights_only=True, mmap=True)
            tables['bf16'][key] = {'action': capture['baseline_action'].float().tolist()}
            del capture
    checkpoint = torch.load(Path(protocol['distill_result']) / 'checkpoint.pt', map_location='cpu',
                            weights_only=True, mmap=True)
    state = checkpoint['run_state']
    if checkpoint['schema_version'] != 2 or state['selected_epoch'] != protocol['selected_epoch'] or not state['frozen']:
        raise RuntimeError('Distillation checkpoint does not contain the frozen selected evaluation')
    # The checkpoint carries evaluations from all allocations; the last segment's JSONL does not.
    tables['distilled'] = selected_references(state['evaluation_rows'], protocol['selected_epoch'])
    del checkpoint
    for context in contexts:
        key = context['case_id'], context['seed_index']
        for arm, table in tables.items():
            if key not in table:
                raise RuntimeError(f'Missing historical {arm} action at {key}')
            action = torch.as_tensor(table[key]['action'], dtype=torch.float32)
            if tuple(action.shape) != (32, 7) or not bool(torch.isfinite(action).all()):
                raise RuntimeError(f'Invalid historical {arm} action at {key}')
    return tables


def run(args):
    guard()
    import torch
    import runner as pilot
    import run_diagnostic as source
    # The frozen old-input helper owns observation loading and metadata checks.
    import future_adapter
    import metrics
    from checkpointing import atomic_torch_save

    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if protocol['protocol'] != 'fastwam-vq-latent-diagnostic-v1':
        raise RuntimeError('Unexpected protocol')
    all_contexts = metrics.frozen_contexts(protocol)
    contexts = [c for c in all_contexts if c['shard'] == args.shard]
    if len(contexts) != 30 or len(all_contexts) != 60:
        raise RuntimeError('The frozen 60-context/two-shard scope changed')
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / 'protocol.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != protocol:
        raise RuntimeError('Existing results belong to a different numerical protocol')
    save(contract_path, protocol)
    job = os.environ['PBS_JOBID']
    started = time.monotonic()
    deadline = started + protocol['walltime_seconds'] - (time.time() - float(os.environ['FW_JOB_START_EPOCH']))
    interrupted = []
    signal.signal(signal.SIGTERM, lambda *_: interrupted.append(True))
    torch.backends.cuda.matmul.allow_tf32 = False
    stage, runtime, weights = 'source_validation', None, None
    durations, query_count = [], 0
    parity_path = args.out / 'self_future_parity.json'
    parity = json.loads(parity_path.read_text()) if parity_path.exists() else {}

    def completed_receipts():
        records = []
        for context in contexts:
            stem = f"case_{context['case_id']}_seed_{context['seed_index']}"
            path = args.out / f'{stem}.json'
            if path.exists():
                value = json.loads(path.read_text())
                if (value['context'] != context or value['protocol'] != protocol['protocol']
                        or not (args.out / f'{stem}.pt').is_file()
                        or set(value['arms']) != set(protocol['arms'])):
                    raise RuntimeError(f'Invalid completed context receipt: {stem}')
                records.append(value)
        return records

    def summary(status):
        receipts = completed_receipts()
        result = {'status': status, 'protocol': protocol['protocol'], 'job_id': job, 'shard': args.shard,
                  'context_count': len(receipts), 'expected_context_count': len(contexts),
                  'native_context_queries': sum(r['native_queries'] for r in receipts),
                  'teacher_future_queries': sum(r['teacher_future_queries'] for r in receipts),
                  'self_future_parity': parity, 'new_queries_this_segment': query_count,
                  'selected_epoch': 2, 'target_linears': 614,
                  'aggregate': metrics.aggregate([row for receipt in receipts for row in receipt['rows']]),
                  'episodes': 0, 'execution': protocol['execution'], 'elapsed_seconds': time.monotonic() - started}
        save(args.out / 'summary.json', result)
        return result

    def budget_gate():
        estimate = max(durations[-8:]) * 1.15 if durations else 150
        if interrupted or time.monotonic() + estimate + protocol['save_reserve_seconds'] >= deadline:
            summary('resumable')
            print('LATENT_DIAGNOSTIC_RESUME_READY', flush=True)
            return True
        return False

    def compare_action(actual, expected, label):
        delta = actual.float().cpu() - torch.as_tensor(expected, dtype=torch.float32)
        record = {'max_abs': float(delta.abs().max()), 'rmse': float(delta.square().mean().sqrt())}
        if (record['max_abs'] > protocol['parity_max_abs_tolerance']
                or record['rmse'] > protocol['parity_rmse_tolerance']):
            raise RuntimeError(f'{label} action parity failed: {record}')
        return record

    def ensure_self_parity(base_arm, output, datum, seed, case, seed_index):
        nonlocal query_count
        if base_arm in parity:
            return False
        if budget_gate():
            return True
        tick = time.monotonic()
        check = future_adapter.run_teacher_future(runtime, datum, seed, output['video_latents'])
        if not torch.equal(check['_diagnostic']['action_initial_noise'], output['_diagnostic']['action_initial_noise']):
            raise RuntimeError('Self-future intervention changed action noise')
        parity[base_arm] = {**compare_action(check['action'], output['action'], 'self-future'),
                            'case_id': case, 'seed_index': seed_index,
                            'video_dit_calls': check['_diagnostic']['video_dit_calls']}
        durations.append(time.monotonic() - tick)
        query_count += 1
        save(parity_path, parity)
        return False

    try:
        if len(completed_receipts()) == len(contexts):
            summary('complete')
            return
        metrics.self_check(torch)
        source_summary = json.loads((Path(protocol['source_root']) / 'results/full/summary.json').read_text())
        if source_summary['status'] != 'complete' or source_summary['pbs_jobid'] != protocol['source_full_job']:
            raise RuntimeError('Parent VQ source is not the completed declared run')
        old_root = Path(protocol['roots']['old_inputs'])
        old_plan = json.loads((old_root / 'plan.json').read_text())
        fresh_plan = {r['case_id']: r for r in json.loads((Path(protocol['source_root']) / 'plan.json').read_text())['inputs']}
        historical = cached_actions(torch, protocol, contexts, source)
        stage = 'model_loading'
        runtime = pilot.Runtime(args.out)
        runtime.arm('bf16')
        stage = 'frozen_weight_loading'
        weights = FrozenWeights(runtime, protocol)
        for context in contexts:
            case, seed_index = context['case_id'], context['seed_index']
            stem = f'case_{case}_seed_{seed_index}'
            receipt_path, payload_path = args.out / f'{stem}.json', args.out / f'{stem}.pt'
            if receipt_path.exists():
                continue
            row = source.validate_plan(old_plan, case) if case < 1000 else fresh_plan[case]
            observation, metadata = source.load_case(old_root if case < 1000 else Path(protocol['source_root']), row)
            datum = runtime.datum(observation)
            runtime.description = metadata['description']
            seed = row['sampler_seeds'][seed_index]
            payload = torch.load(payload_path, map_location='cpu', weights_only=True) if payload_path.exists() else {
                'protocol': protocol['protocol'], 'context': context, 'sampler_seed': seed, 'captures': {}}
            if payload['context'] != context or payload['sampler_seed'] != seed or payload['protocol'] != protocol['protocol']:
                raise RuntimeError('Partial context differs from frozen input/seed')
            captures = payload['captures']
            for arm in protocol['arms']:
                base_arm = 'bf16' if arm == 'bf16' else arm.split('_', 1)[0]
                if arm in captures:
                    if not arm.endswith('teacher_future') and base_arm not in parity:
                        weights.activate(base_arm)
                        if ensure_self_parity(base_arm, captures[arm], datum, seed, case, seed_index):
                            return
                    continue
                if budget_gate():
                    return
                stage = arm
                weights.activate(base_arm)
                runtime.current_arm = arm
                print('DIAGNOSTIC_QUERY ' + json.dumps({'job_id': job, 'shard': args.shard,
                      'case_id': case, 'seed_index': seed_index, 'arm': arm}), flush=True)
                tick = time.monotonic()
                if arm.endswith('teacher_future'):
                    output = future_adapter.run_teacher_future(runtime, datum, seed, captures['bf16']['video_latents'])
                    native = captures[f'{base_arm}_self']
                    if not torch.equal(output['_diagnostic']['action_initial_noise'], native['_diagnostic']['action_initial_noise']):
                        raise RuntimeError('Teacher-future replacement changed initial action noise')
                else:
                    output = future_adapter.run_capture(runtime, datum, seed)
                    if len(weights.seen) != 614:
                        raise RuntimeError(f'Native query called only {len(weights.seen)}/614 targets')
                    output['cached_action_parity'] = compare_action(output['action'], historical[base_arm][case, seed_index]['action'], arm)
                    if base_arm != 'bf16' and not torch.equal(output['_diagnostic']['action_initial_noise'],
                                                             captures['bf16']['_diagnostic']['action_initial_noise']):
                        raise RuntimeError('Native checkpoint comparison changed initial action noise')
                torch.cuda.synchronize()
                durations.append(time.monotonic() - tick)
                query_count += 1
                output['query_seconds'] = durations[-1]
                output['video_latents'] = output['video_latents'].detach().cpu()
                output['action'] = output['action'].detach().float().cpu()
                captures[arm] = output
                atomic_torch_save(torch, payload_path, payload)
                if not arm.endswith('teacher_future'):
                    if ensure_self_parity(base_arm, output, datum, seed, case, seed_index):
                        return
                save(args.out / 'progress.json', {'job_id': job, 'shard': args.shard, 'stage': stage,
                     'case_id': case, 'seed_index': seed_index, 'completed_contexts': len(completed_receipts()),
                     'query_count_this_segment': query_count, 'elapsed_seconds': time.monotonic() - started})
            records = []
            teacher = captures['bf16']
            for arm, capture in captures.items():
                record = {'case_id': case, 'seed_index': seed_index, 'split': context['split'],
                          'domain': row['domain'], 'dimension': row['dimension'], 'arm': arm,
                          **metrics.action_metrics(torch, capture['action'], teacher['action'])}
                if arm in ('parent_self', 'distilled_self'):
                    record.update(metrics.latent_metrics(torch, capture['video_latents'], teacher['video_latents']))
                records.append(record)
            payload['rows'] = records
            payload['distilled_vs_parent_latent'] = metrics.latent_metrics(torch, captures['distilled_self']['video_latents'],
                                                                         captures['parent_self']['video_latents'])
            atomic_torch_save(torch, payload_path, payload)
            save(receipt_path, {'protocol': protocol['protocol'], 'context': context, 'sampler_seed': seed,
                                'arms': list(captures), 'native_queries': 3, 'teacher_future_queries': 2,
                                'rows': records, 'distilled_vs_parent_latent': payload['distilled_vs_parent_latent']})
            summary('running')
            print(f'CONTEXT_COMPLETE case={case} seed_index={seed_index}', flush=True)
            del payload, captures, datum, observation
            gc.collect()
        if set(parity) != {'bf16', 'parent', 'distilled'}:
            raise RuntimeError('Self-future adapter parity gates are incomplete')
        summary('complete')
        print('LATENT_DIAGNOSTIC_COMPLETE', flush=True)
    except BaseException as exc:
        save(args.out / 'failure.json', {'job_id': job, 'shard': args.shard, 'stage': stage,
                                        'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        if weights:
            weights.close()
        if runtime and runtime.env:
            runtime.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    run(parser.parse_args())
