"""Two-hour, per-query resumable future interventions on a frozen paired cohort."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time
import traceback
from common import PROTOCOL, DONORS, contexts, guard, save, stem


class ReaderWeights:
    """Keep original BF16 and exactly one selected student weight set."""
    def __init__(self, runtime, p, student):
        import torch
        import codebook
        import smooth_vq as core
        from run_experiment import qinput
        self.torch, self.runtime = torch, runtime
        runtime.q.disable()
        self.modules = runtime.q.modules
        if len(self.modules) != 614:
            raise RuntimeError('Expected614 target Linears')
        self.original = {n: m.forward for n, m in self.modules.items()}
        self.weights = {'bf16': {}, student: {}}
        self.transforms, self.current, self.seen = {}, 'bf16', set()
        directory = p['distill_result'] if student == 'action_only' else p['protected_result']
        overlay = torch.load(Path(directory) / 'selected_codebook_overlay.pt', map_location='cpu', weights_only=True)
        if list(self.modules) != overlay['module_names']:
            raise RuntimeError('Overlay module order differs')
        self.student = student
        for i, (name, mod) in enumerate(self.modules.items()):
            bank = torch.load(Path(p['source_root']) / 'results/full/banks' / p['initial_arm'] / f'{i:04d}.pt',
                              map_location='cpu', weights_only=True)
            if bank['module'] != name or tuple(bank['encoded']['shape']) != tuple(mod.weight.shape):
                raise RuntimeError(f'Wrong source bank: {name}')
            enc = dict(bank['encoded'])
            state = overlay['master_state'][name]
            enc['books'] = state['books'].to(device='cuda', dtype=torch.bfloat16)
            active = (enc['row_scales'] > 0).float()
            enc['row_scales'] = (state['log_scales'].to('cuda').exp() * active.to('cuda')).to(torch.bfloat16)
            # Match the old GPU decode, including BF16 addition/multiplication rounding.
            gpu_enc = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in enc.items()}
            self.weights['bf16'][name] = mod.weight.detach().cpu().clone()
            self.weights[student][name] = codebook.decode_vq(gpu_enc).cpu()
            self.transforms[name] = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in bank['transform'].items()}

            def forward(x, _name=name, _mod=mod):
                self.seen.add(_name)
                if self.current == 'bf16':
                    return self.original[_name](x)
                return torch.nn.functional.linear(qinput(torch, core, x, self.transforms[_name]), _mod.weight, _mod.bias)

            mod.forward = forward
            if i % 100 == 0:
                print(f'READER_BANKS {i}/614', flush=True)
            del bank, enc, gpu_enc
        del overlay
        gc.collect()

    def activate(self, arm):
        if arm not in self.weights:
            raise RuntimeError('Unknown reader weight set')
        if arm != self.current:
            for name, mod in self.modules.items():
                mod.weight.data = self.weights[arm][name].to('cuda')
            self.current = arm
        self.seen.clear()

    def close(self):
        for name, mod in self.modules.items():
            mod.forward = self.original[name]


def parity(torch, actual, expected, p, label):
    actual, expected = actual.float().cpu(), torch.as_tensor(expected).float().cpu()
    if actual.shape != expected.shape or not bool(torch.isfinite(actual).all()):
        raise RuntimeError(f'{label}: wrong shape or nonfinite values')
    delta = actual - expected
    result = dict(max_abs=float(delta.abs().max()), rmse=float(delta.square().mean().sqrt()))
    if result['max_abs'] > p['parity_max_abs_tolerance'] or result['rmse'] > p['parity_rmse_tolerance']:
        raise RuntimeError(f'{label} parity failed: {result}')
    return result


def main():
    guard()
    import torch
    import runner as pilot
    import run_diagnostic as source
    import future_adapter
    import metrics
    from checkpointing import atomic_torch_save, to_cpu
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--phase', choices=('teacher', 'cross_a', 'cross_z'), required=True)
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    root = Path(p['remote_root'])
    expected = [c for c in contexts(p) if c['shard'] == args.shard]
    manifest = json.loads((root / 'input_manifest.json').read_text())
    selected = torch.load(root / 'selected_actions.pt', map_location='cpu', weights_only=True)
    if manifest['protocol'] != PROTOCOL or selected['protocol'] != PROTOCOL:
        raise RuntimeError('Prepared references belong to a different experiment')
    inputs = {stem(c): c for c in manifest['contexts']}
    out = root / 'results' / f'{args.phase}-{args.shard}'
    out.mkdir(parents=True, exist_ok=True)
    protocol_copy = out / 'protocol.json'
    if protocol_copy.exists() and json.loads(protocol_copy.read_text()) != p:
        raise RuntimeError('Resume numerical protocol changed')
    save(protocol_copy, p)
    job = os.environ['PBS_JOBID']
    deadline = float(os.environ['FW_JOB_START_EPOCH']) + p['walltime_seconds']
    interrupted = []
    signal.signal(signal.SIGTERM, lambda *_: interrupted.append(True))
    durations, calls, runtime, weights = [], 0, None, None
    torch.backends.cuda.matmul.allow_tf32 = False
    reader = {'teacher': 'bf16', 'cross_a': 'action_only', 'cross_z': 'action_latent'}[args.phase]
    donors = DONORS if args.phase == 'teacher' else ('action_only', 'action_latent')

    def records():
        rows = []
        for c in expected:
            for donor in donors:
                path = out / f'{stem(c)}_{donor}.json'
                if path.exists():
                    r = json.loads(path.read_text())
                    if (r['protocol'] != PROTOCOL or r['context'] != c or r['reader'] != reader or r['donor'] != donor
                            or r['sampler_seed'] != inputs[stem(c)]['sampler_seed'] or not path.with_suffix('.pt').is_file()):
                        raise RuntimeError('Resume row identity mismatch')
                    rows.append(r)
        return rows

    def summary(status):
        rows = records()
        generated = sum((root / 'futures' / f'{stem(c)}.pt').is_file() for c in expected)
        result = dict(protocol=PROTOCOL, job_id=job, status=status, phase=args.phase, shard=args.shard,
                      reader=reader, donors=list(donors), row_count=len(rows), expected_rows=30 * len(donors),
                      context_count=len({(r['case_id'], r['seed_index']) for r in rows}),
                      generated_protected_futures=generated, new_queries_this_segment=calls,
                      elapsed_from_pbs_start=time.time() - float(os.environ['FW_JOB_START_EPOCH']),
                      selected_epoch=2, target_linears=614, teacher_reader_unquantized=reader == 'bf16', episodes=0)
        save(out / 'summary.json', result)
        return result

    def budget(estimate=90):
        cost = max([estimate] + durations[-5:])
        if interrupted or time.time() + 1.2 * cost + p['save_reserve_seconds'] >= deadline:
            summary('resumable')
            return True
        return False

    def load_old(c):
        return torch.load(inputs[stem(c)]['cache_path'], map_location='cpu', weights_only=True)

    def datum(c):
        row = old_rows[c['case_id']] if c['case_id'] < 1000 else new_rows[c['case_id']]
        location = old_root if c['case_id'] < 1000 else Path(p['source_root'])
        observation, metadata = source.load_case(location, row)
        runtime.description = metadata['description']
        return runtime.datum(observation), row

    stage = 'startup'
    try:
        if len(records()) == 30 * len(donors):
            summary('complete')
            return
        prep = json.loads((root / 'prepare_summary.json').read_text())
        if prep['status'] != 'complete' or prep['context_count'] != 60:
            raise RuntimeError('Full CPU reference preparation is required')
        old_root = Path(p['roots']['old_inputs'])
        old_rows = {r['case_id']: r for r in json.loads((old_root / 'plan.json').read_text())['inputs']}
        new_rows = {r['case_id']: r for r in json.loads((Path(p['source_root']) / 'plan.json').read_text())['inputs']}
        runtime = pilot.Runtime(out)
        runtime.arm('bf16')
        need_generation = args.phase == 'teacher' and any(not (root / 'futures' / f'{stem(c)}.pt').exists() for c in expected)
        if need_generation or reader != 'bf16':
            student = 'action_latent' if args.phase == 'teacher' else reader
            stage = 'student_weight_loading'
            weights = ReaderWeights(runtime, p, student)
            weights.activate(student)
        # Generate only the absent fourth donor; three historical donors remain read-only.
        if need_generation:
            (root / 'futures').mkdir(exist_ok=True)
            for c in expected:
                path = root / 'futures' / f'{stem(c)}.pt'
                if path.exists():
                    continue
                if budget():
                    return
                stage = 'generate_action_latent_future'
                d, row = datum(c)
                seed = inputs[stem(c)]['sampler_seed']
                print(f'GENERATE {stem(c)}', flush=True)
                tick = time.monotonic()
                weights.seen.clear()
                generated = future_adapter.run_capture(runtime, d, seed)
                if len(weights.seen) != 614:
                    raise RuntimeError('Native protected capture omitted target Linears')
                old = load_old(c)
                if not torch.equal(generated['_diagnostic']['action_initial_noise'], old['captures']['bf16']['_diagnostic']['action_initial_noise']):
                    raise RuntimeError('Protected donor action noise differs from paired cache')
                check = parity(torch, generated['action'], selected['actions']['action_latent'][f"{c['case_id']}:{c['seed_index']}"], p, 'protected native')
                atomic_torch_save(torch, path, dict(protocol=PROTOCOL, context=c, sampler_seed=seed,
                    output=to_cpu(generated), native_action_parity=check, job_id=job))
                torch.cuda.synchronize()
                durations.append(time.monotonic() - tick)
                calls += 1
                summary('running')
                del d, generated, old
        if weights:
            weights.activate(reader)
        elif reader != 'bf16':
            raise RuntimeError('Student reader was not loaded')
        for c in expected:
            missing = [name for name in donors if not (out / f'{stem(c)}_{name}.json').exists()]
            if not missing:
                continue
            old = load_old(c)
            protected = torch.load(root / 'futures' / f'{stem(c)}.pt', map_location='cpu', weights_only=True)
            if (protected['protocol'] != PROTOCOL or protected['context'] != c
                    or protected['sampler_seed'] != inputs[stem(c)]['sampler_seed']):
                raise RuntimeError('Protected future identity mismatch')
            caps = old['captures']
            futures = dict(bf16=caps['bf16']['video_latents'], parent=caps['parent_self']['video_latents'],
                           action_only=caps['distilled_self']['video_latents'], action_latent=protected['output']['video_latents'])
            d, row = datum(c)
            seed = inputs[stem(c)]['sampler_seed']
            if int(row['sampler_seeds'][c['seed_index']]) != seed:
                raise RuntimeError('Prepared seed differs from current frozen plan')
            for donor in missing:
                if budget(60):
                    return
                stage = f'{reader}_reads_{donor}'
                if weights and weights.current != reader:
                    raise RuntimeError('Reader weight set changed between donor interventions')
                print(f'READER_QUERY reader={reader} donor={donor} {stem(c)}', flush=True)
                tick = time.monotonic()
                output = future_adapter.run_teacher_future(runtime, d, seed, futures[donor])
                if not torch.equal(output['_diagnostic']['action_initial_noise'], caps['bf16']['_diagnostic']['action_initial_noise']):
                    raise RuntimeError('Reader intervention changed paired action noise')
                replay = None
                if reader == 'bf16' and donor == 'bf16':
                    replay = parity(torch, output['action'], caps['bf16']['action'], p, 'teacher reader replay')
                if reader != 'bf16' and reader == donor:
                    replay = parity(torch, output['action'], selected['actions'][reader][f"{c['case_id']}:{c['seed_index']}"], p, 'student self-future replay')
                torch.cuda.synchronize()
                duration = time.monotonic() - tick
                durations.append(duration)
                calls += 1
                diagnostic = output['_diagnostic']
                if not diagnostic['teacher_future_injection_verified'] or diagnostic['video_dit_calls'] != 0:
                    raise RuntimeError('Future was not injected at the intended interface')
                r = dict(protocol=PROTOCOL, context=c, **c, sampler_seed=seed, reader=reader, donor=donor,
                         job_id=job, domain=row['domain'], dimension=row['dimension'], replay_parity=replay,
                         query_seconds=duration, future_injection_verified=True, video_dit_calls=0, action_dit_calls=10,
                         **metrics.action_metrics(torch, output['action'], caps['bf16']['action']))
                payload_path = out / f'{stem(c)}_{donor}.pt'
                atomic_torch_save(torch, payload_path, dict(protocol=PROTOCOL, action=output['action'].float().cpu(), row=r))
                save(out / f'{stem(c)}_{donor}.json', r)
                summary('running')
                del output
            del old, protected, caps, futures, d
            gc.collect()
        final = summary('complete')
        if final['row_count'] != final['expected_rows'] or final['context_count'] != 30:
            raise RuntimeError('Incomplete shard cannot finish')
    except BaseException as exc:
        save(out / 'failure.json', dict(job_id=job, phase=args.phase, shard=args.shard, stage=stage,
             error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if weights:
            weights.close()
        if runtime and runtime.env:
            runtime.env.close()


if __name__ == '__main__':
    main()
