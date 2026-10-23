"""Resumable full-cohort calibration adjoints; original BF16/A4 forward unchanged."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time
from common import PROTOCOL, ARMS, guard, save, units, stem


def observed_decoder(torch, original_decode, pointers, state, target_names, ids, sampled):
    class ObservedWeight(torch.autograd.Function):
        @staticmethod
        def forward(ctx, books, logs, active, indices, dense):
            # Checkpoint's saved-tensor hook permits only one unpack in backward.
            ctx.observed_name = pointers[books.data_ptr()]
            return original_decode.forward(ctx, books, logs, active, indices, dense)

        @staticmethod
        def backward(ctx, grad):
            values = original_decode.backward(ctx, grad)
            name = ctx.observed_name
            if state['objective'] and name in target_names:
                from common import weight_vectors
                flat = weight_vectors(torch, grad.detach())
                value = flat.index_select(0, ids[name]) if sampled else flat
                value = value.float().cpu()
                if name not in state['buffers']:
                    state['buffers'][name] = torch.zeros_like(value)
                    state['book_sums'][name] = torch.zeros_like(values[0], device='cpu')
                    state['calls'][name] = 0
                state['buffers'][name].add_(value)
                state['book_sums'][name].add_(values[0].detach().cpu())
                state['calls'][name] += 1
            return values
    return ObservedWeight


def run(args):
    guard()
    import torch
    import runner as pilot
    import run_diagnostic as source
    import sampler
    import train_banks
    import latent_reference
    from checkpointing import atomic_torch_save
    from stats import FIELDS, summarize
    import fast_a4
    import transform_checks
    transform_checks.run(Path(os.environ['ARTIFACTS']) / 'transform_check.json')
    original_input = train_banks.A4Input
    train_banks.A4Input = fast_a4.A4Input
    p = json.loads((Path(os.environ['ROOT']) / 'protocol.json').read_text())
    root = Path(p['remote_root'])
    prep = json.loads((root / 'prepare_summary.json').read_text())
    plan = torch.load(root / 'sample_plan.pt', map_location='cpu', weights_only=True)
    if prep['status'] != 'complete' or prep['protocol'] != PROTOCOL:
        raise RuntimeError('Reference preparation required')
    selected = None
    if args.phase == 'exact':
        selected = json.loads((root / 'results/analysis_capture/selection.json').read_text())
        target_names = selected['module_names']
    else:
        target_names = plan['module_names']
    expected = units(p, args.shard)
    if args.phase == 'exact':
        expected = [v for v in expected if v[0] == 'action_only']
    out = root / 'results' / f'{args.phase}-{args.shard}'
    out.mkdir(parents=True, exist_ok=True)
    original_contract = out / 'protocol.json'
    if original_contract.exists() and json.loads(original_contract.read_text()) != p:
        raise RuntimeError('Resume contract changed')
    save(original_contract, p)
    job = os.environ['PBS_JOBID']
    interrupted = []
    signal.signal(signal.SIGTERM, lambda *_: interrupted.append(True))
    deadline = float(os.environ['FW_JOB_START_EPOCH']) + p['walltime_seconds']
    times, query_count = [], 0

    def finish(status):
        done = [(a, c) for a, c in expected if (out / f'{stem(a,c)}.json').exists() and (out / f'{stem(a,c)}.pt').exists()]
        s = dict(protocol=PROTOCOL, phase=args.phase, shard=args.shard, job_id=job, status=status,
                 context_count=len(done), expected_contexts=len(expected), target_linears=614,
                 completed=[stem(a,c) for a,c in done], new_queries=query_count,
                 runtime_transform_implementation=fast_a4.IMPLEMENTATION)
        save(out / 'summary.json', s)
        save(Path(os.environ['ARTIFACTS']) / 'summary.json', s)

    if all((out / f'{stem(a,c)}.json').exists() and (out / f'{stem(a,c)}.pt').exists() for a,c in expected):
        finish('complete')
        return
    references = torch.load(Path(p['latent_root']) / 'reference_cache.pt', map_location='cpu', weights_only=True, mmap=True)
    manifest = json.loads((Path(p['future_reader_root']) / 'input_manifest.json').read_text())
    entries = {(c['case_id'], c['seed_index']): c for c in manifest['contexts']}
    input_rows = {r['case_id']: r for r in json.loads((Path(p['roots']['old_inputs']) / 'plan.json').read_text())['inputs']}
    overlays = {arm: torch.load(Path(p[key]) / 'selected_codebook_overlay.pt', map_location='cpu', weights_only=True)
                for arm, key in (('action_only','distill_result'),('action_latent','protected_result'))}
    source_dir = Path(p['source_root']) / 'results/full'
    runtime = pilot.Runtime(out)
    runtime.arm('bf16')
    banks = train_banks.TrainableBanks(runtime, source_dir, p['initial_arm'])
    if list(banks.modules) != plan['module_names']:
        raise RuntimeError('Sample plan module order differs')
    if args.phase == 'exact':
        for name in banks.modules:
            for parameter in (banks.books[name], banks.logs[name]):
                parameter.requires_grad_(name in target_names)
    initial = banks.snapshot()
    ids = {n: plan['modules'][n]['flat_ids'].to('cuda') for n in target_names} if args.phase == 'capture' else {}
    pointers = {v.data_ptr(): n for n,v in banks.books.items()}
    state = {'objective': None, 'buffers': {}, 'book_sums': {}, 'calls': {}}
    original_decode = train_banks.DecodedWeight

    train_banks.DecodedWeight = observed_decoder(torch, original_decode, pointers, state, target_names, ids, args.phase == 'capture')

    def backward(loss, objective, retain=False):
        for parameter in banks.parameters():
            parameter.grad = None
        state.update(objective=objective, buffers={}, book_sums={}, calls={})
        loss.backward(retain_graph=retain)
        state['objective'] = None
        exact = {}
        for name in target_names:
            value = banks.books[name].grad
            value = torch.zeros_like(banks.books[name], device='cpu') if value is None else value.detach().cpu().clone()
            exact[name] = value
            if name in state['book_sums'] and not torch.allclose(state['book_sums'][name], value, atol=1e-7, rtol=1e-3):
                raise RuntimeError(f'Repeated-call adjoint accumulation mismatch: {name}')
        return state['buffers'], exact, state['calls']

    def parity(a, b):
        d = a.detach().float().cpu() - b.float().cpu()
        maximum, rmse = float(d.abs().max()), float(d.square().mean().sqrt())
        if maximum > p['parity_max_abs_tolerance'] or rmse > p['parity_rmse_tolerance']:
            raise RuntimeError(f'Unmodified forward differs from cached checkpoint: {maximum}/{rmse}')
        return dict(max_abs=maximum, rmse=rmse)

    current = None
    try:
        for arm, case in expected:
            key = stem(arm, case)
            if (out / f'{key}.json').exists() and (out / f'{key}.pt').exists():
                continue
            if interrupted or time.time() + max([1200] + times[-3:]) * 1.2 + p['save_reserve_seconds'] >= deadline:
                finish('resumable')
                return
            if arm != current:
                banks.restore(initial if arm == 'parent' else overlays[arm]['master_state'])
                current = arm
            tick = time.monotonic()
            entry = entries[case, 0]
            observation, meta = source.load_case(Path(p['roots']['old_inputs']), input_rows[case])
            runtime.description = meta['description']
            datum = runtime.datum(observation)
            old = torch.load(entry['cache_path'], map_location='cpu', weights_only=True)
            if arm == 'action_latent':
                prior = torch.load(Path(p['future_reader_root']) / 'futures' / f'case_{case}_seed_0.pt', map_location='cpu', weights_only=True)['output']
            else:
                prior = old['captures']['parent_self' if arm == 'parent' else 'distilled_self']
            print('GRADIENT_QUERY', args.phase, arm, case, flush=True)
            action, counts, future = sampler.differentiable_action(runtime, datum, entry['sampler_seed'], checkpointed=True, return_future=True)
            checks = dict(action=parity(action, prior['action']), future=parity(future, prior['video_latents']))
            teacher = references['teacher'][f'{case}:0']
            la = (action[:10,:6].float() - teacher['action'][:10,:6].to('cuda').float()).square().mean()
            lz = latent_reference.future_nmse_loss(torch, future, teacher['future'].to('cuda'))
            if args.phase == 'capture':
                ga, ea, ca = backward(la, 'action', retain=True)
            gz, ez, cz = backward(lz, 'latent')
            payload = dict(protocol=PROTOCOL, phase=args.phase, arm=arm, case_id=case, seed_index=0,
                           sampler_seed=entry['sampler_seed'], job_id=job, parity=checks,
                           action_mse=float(la.detach()), future_nmse=float(lz.detach()), counts=counts,
                           transform_implementation=fast_a4.IMPLEMENTATION)
            if args.phase == 'capture':
                tables, exact_a, exact_z = [], [], []
                for i, name in enumerate(plan['module_names']):
                    m = plan['modules'][name]
                    zero = torch.zeros((len(m['flat_ids']),4))
                    scales = banks.stored_encoding(name)['row_scales'].float().cpu()
                    tables.append(summarize(torch, ga.get(name,zero), gz.get(name,zero), m, scales,
                                            ea[name], ez[name], prep['latent_weight'],
                                            p['sample_seed'] + i * 1009 + case, p['null_repeats']))
                    exact_a.append(ea[name]); exact_z.append(ez[name])
                payload.update(fields=FIELDS, module_names=plan['module_names'], tables=torch.stack(tables),
                               exact_action=torch.stack(exact_a), exact_latent=torch.stack(exact_z),
                               action_calls=ca, latent_calls=cz)
                del ga, ea, ca, tables, exact_a, exact_z
            else:
                payload.update(module_names=target_names, decoded_latent_gradients={n: gz[n].reshape(plan['modules'][n]['shape'][0], -1)[:, :plan['modules'][n]['shape'][1]] for n in target_names},
                               stored_row_scales={n:banks.stored_encoding(n)['row_scales'].detach().cpu() for n in target_names},
                               exact_latent_books={n:ez[n] for n in target_names})
            atomic_torch_save(torch, out / f'{key}.pt', payload)
            row = {k:v for k,v in payload.items() if k not in ('tables','exact_action','exact_latent','decoded_latent_gradients','stored_row_scales','exact_latent_books','action_calls','latent_calls','fields','module_names')}
            row.update(action_connected_modules=len(payload['action_calls']) if args.phase == 'capture' else None,
                       latent_connected_modules=len(cz), seconds=time.monotonic()-tick)
            save(out / f'{key}.json', row)
            times.append(row['seconds']); query_count += 1
            print('GRADIENT_COMPLETE', key, row['seconds'], flush=True)
            finish('running')
            del gz, ez, cz, payload, old, prior, datum, action, future, la, lz
            gc.collect(); torch.cuda.empty_cache()
        finish('complete')
    finally:
        train_banks.DecodedWeight = original_decode
        train_banks.A4Input = original_input
        for name, module in banks.modules.items():
            module.forward = banks.forwards[name]
        if runtime.env:
            runtime.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('capture','exact'), required=True)
    parser.add_argument('--shard', type=int, choices=range(4), required=True)
    run(parser.parse_args())
