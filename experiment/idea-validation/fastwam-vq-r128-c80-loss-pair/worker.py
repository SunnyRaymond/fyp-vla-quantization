"""Matched C80 Seq8 VQ losses with frozen R0 and cached BF16 teachers."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import signal
import time

PROTOCOL = 'fastwam-vq-r128-c80-loss-pair-v1'
ARMS = ('az', 'reader_kl')
CASES = list(range(2000, 2080))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_protocol(p):
    t, q, a = p['training'], p['quantization'], p['allocation']
    if (p['protocol'] != PROTOCOL or p['arms'] != list(ARMS)
            or p['calibration']['case_ids'] != CASES or p['calibration']['seed_indices'] != [0]
            or p['target_linears'] != 614 or t['vq_updates'] != 8
            or t['contexts_per_update'] != 10 or t['training_exposures'] != 80
            or t['vq_learning_rate'] != .0005 or t['rotation_updates'] != 0
            or t['rotation_learning_rate'] != 0 or t['gradient_norm_clip'] != 1
            or t['latent_weight'] != .04245452683169643
            or not t['vq_optimizer_reset_after_update1'] or t['post_update_diagnostic_forwards']
            or q['D'] != 1 or q['clipping_ratio'] != 1 or q['native_kernel']
            or a['gpus'] != 2 or a['walltime'] != '12:00:00' or a['execution_queue'] != 'g2'
            or a['soft_deadline_seconds'] != 42600
            or p['source_initialization']['protocol'] != 'fastwam-vq-r128-c80-dual-v1'
            or p['source_initialization']['job'] != '25912619.pbs101'
            or p['reader_source']['protocol'] != 'fastwam-vq-c80-frozen-reader-kl-v1'
            or not p['reader_source']['runtime_artifacts']
            or not p['numerics']['deterministic_algorithms']):
        raise ValueError('Frozen R128/C80 paired-loss recipe differs')
    return p


def beta_from_means(p, nmse, kl):
    if any(not math.isfinite(v) or v <= 0 for v in (nmse, kl)):
        raise ValueError('Initial C80 losses must be positive and finite')
    return p['reader_source']['config']['coefficient_fraction'] * p['training']['latent_weight'] * nmse / kl


class Work:
    def __init__(self, p, arm):
        import sampler
        sampler._require_compute_allocation()
        validate_protocol(p)
        if arm not in ARMS:
            raise ValueError('Unknown loss arm')
        import numpy as np
        import torch
        import runner
        import calibration_source
        import latent_reference
        import source_worker
        from checkpointing import atomic_torch_save, capture_rng, restore_rng, to_cpu, optimizer_to_device
        self.p, self.arm, self.torch, self.np = p, arm, torch, np
        self.runner, self.source, self.sampler = runner, calibration_source, sampler
        self.latent_reference, self.helpers = latent_reference, source_worker
        self.atomic_torch_save, self.capture_rng = atomic_torch_save, capture_rng
        self.restore_rng, self.to_cpu, self.optimizer_to_device = restore_rng, to_cpu, optimizer_to_device
        self.root = Path(p['remote_root'])
        self.out = self.root / 'results'
        self.unit = self.out / 'arms' / arm
        self.unit.mkdir(parents=True, exist_ok=True)
        self.job = os.environ['PBS_JOBID']
        self.started = time.monotonic()
        self.deadline = float(os.environ['FW_JOB_START_EPOCH']) + p['allocation']['soft_deadline_seconds']
        self.stopped = []
        signal.signal(signal.SIGTERM, lambda signum, frame: self.stopped.append(signum))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        source_worker.Work.reset_rng(self)
        self.plan = read(p['calibration']['plan_file'])
        self.rows = {r['case_id']: r for r in self.plan['inputs']}
        if self.plan['calibration_cohorts']['c80'] != CASES:
            raise ValueError('Original C80 plan differs')
        rotation = p['source_rotation']
        old_training = read(Path(rotation['root']) / 'results' / rotation['lane'] / 'training.json')
        self.schedule = source_worker.validate_schedule(
            [row['cases'] for row in old_training['arms'][rotation['arm']]['history']], self.rows, p['suites'])
        self.teacher_root = Path(p['c80_teacher_directory'])
        self.heldout_teacher_root = Path(p['baseline_root']) / 'results/teacher'
        teacher_receipt = read(self.teacher_root / 'receipt.json')
        if (not teacher_receipt.get('complete') or teacher_receipt.get('case_ids') != CASES
                or teacher_receipt.get('seed_indices') != [0]):
            raise ValueError('Cached C80 BF16 teacher receipt differs')
        init = p['source_initialization']
        source_summary = read(init['source_summary'])
        if (source_summary.get('protocol') != init['protocol'] or not source_summary.get('complete')
                or source_summary.get('pbs_jobid') != init['job']):
            raise ValueError('Shared R0/VQ0 source is incomplete')
        for name in ('source_receipt', 'repro_receipt'):
            receipt = read(init[name])
            if receipt.get('protocol') != init['protocol'] or not receipt.get('complete') or not receipt.get('passed'):
                raise ValueError('Shared source gate failed: ' + name)
        self.data, self.runtime, self.bank, self.rb, self.masters = {}, None, None, None, None
        self.counters = dict(student_queries=0, formal_infer_queries=0, reader_queries=0,
                             parameter_reverse_passes=0, new_teacher_queries=0)
        counter_file = self.unit / 'query_counters.json'
        if counter_file.exists():
            previous = read(counter_file)
            if previous['protocol'] != PROTOCOL or previous['arm'] != arm:
                raise ValueError('Query counter binding differs')
            self.counters.update(previous['counters'])
        self.progress('loading', updates=0)

    def progress(self, phase, **extra):
        value = dict(protocol=PROTOCOL, arm=self.arm, pbs_jobid=self.job, phase=phase,
                     seconds=time.monotonic() - self.started, **self.counters, **extra)
        save(self.unit / 'progress.json', value)
        print('LOSS_PAIR_PROGRESS ' + json.dumps(value, allow_nan=False), flush=True)

    def enough(self):
        if self.stopped or time.time() + 300 >= self.deadline:
            raise TimeoutError('Committed updates and preparation cases retained for resume')

    def count(self, key):
        self.counters[key] += 1
        save(self.unit / 'query_counters.json', dict(protocol=PROTOCOL, arm=self.arm, counters=self.counters))

    def datum(self, case):
        return self.helpers.Work.datum(self, case)

    def teacher(self, case, seed=0):
        return self.helpers.Work.teacher(self, case, seed)

    def loss(self, action, future, target):
        return self.helpers.Work.loss(self, action, future, target)

    def metrics(self, action, future, target):
        return self.helpers.Work.metrics(self, action, future, target)

    def query(self, case, seed=0, grad=False):
        self.count('student_queries')
        action, future, counts = self.helpers.Work.query(self, case, seed, grad)
        if (counts != {'video_steps': 10, 'action_steps': 10}
                or not bool(self.torch.isfinite(action).all()) or not bool(self.torch.isfinite(future).all())):
            raise ValueError('Student query shape/finite/sampling gate failed')
        return action, future, counts

    def load_model(self):
        from rotation_bank import RotationBanks, VQBanks
        torch, p = self.torch, self.p
        self.runtime = self.runner.Runtime(self.unit)
        self.runtime.model.eval()
        self.runtime.arm('bf16')
        self.rb = RotationBanks(self.runtime, p['calibration']['bank_directory'])
        rotation = p['source_rotation']
        saved = torch.load(rotation['deployment'], map_location='cpu', weights_only=True)
        if (saved['protocol']['protocol'] != rotation['protocol'] or saved['lane'] != rotation['lane']
                or saved['arm'] != rotation['arm'] or saved['updates'] != 8 or saved['target_linears'] != 614):
            raise ValueError('Actual R0 deployment binding differs')
        self.initial_rotation = saved['rotations']
        self.rb.rotations.restore(self.initial_rotation)
        self.rb.rotations.freeze()
        # The frozen reader needs the original BF16 weights, before VQ overwrites them.
        if self.arm == 'reader_kl':
            self.masters = {n: m.weight.detach().cpu().clone() for n, m in self.runtime.q.modules.items()}
        self.rb.close()
        init = p['source_initialization']
        self.bank = VQBanks(self.runtime, init['bank_directory'], self.rb.rotations, self.rb.transforms, init['protocol'])
        if (len(self.bank.parameters()) != 1228
                or any(v.requires_grad for v in self.runtime.model.parameters())
                or any(v.requires_grad for v in self.bank.rotations.parameters())
                or max(self.bank.rotations.orthogonality().values()) > 5e-5):
            raise ValueError('Only 614 VQ codebook/row-gain pairs may train')
        self.helpers.Work.reset_rng(self)

    def reader_action(self, reader, case, future, grad):
        self.enough()
        self.count('reader_queries')
        value, description = self.datum(case)
        self.runtime.description = description
        with self.torch.set_grad_enabled(grad):
            return reader(self.runtime, value, self.rows[case]['sampler_seeds'][0], future, checkpointed=grad)

    def preparation(self):
        if self.arm == 'az':
            return dict(protocol=PROTOCOL, arm='az', complete=True, beta=0., heldout_used=False)
        from reader_runtime import reader_session, compact_counts
        from reader_loss import reader_terms
        p, torch = self.p, self.torch
        out = self.out / 'reader_preparation'
        summary_path = out / 'summary.json'
        if summary_path.exists():
            prep = read(summary_path)
            if (prep['frozen_protocol'] != p or not prep['complete'] or prep['heldout_used']
                    or prep['calibration_case_ids'] != CASES):
                raise ValueError('Frozen beta preparation differs')
            return prep
        stats = {'action': p['reader_source']['action_stats']}
        (out / 'student_future').mkdir(parents=True, exist_ok=True)
        pending = []
        # Stage only 80 small futures; enter the memory-parking reader session once.
        for case in CASES:
            record_path = out / 'cases' / f'{case}.json'
            if record_path.exists():
                record = read(record_path)
                if record['protocol'] != PROTOCOL or record['case_id'] != case:
                    raise ValueError('Preparation case binding differs')
                continue
            self.enough()
            cache_path = out / 'student_future' / f'{case}.pt'
            if not cache_path.exists():
                action, future, _ = self.query(case)
                target = self.teacher(case)
                _, _, nmse = self.loss(action, future, target)
                self.atomic_torch_save(torch, cache_path, dict(protocol=PROTOCOL, case_id=case,
                    plan_row=self.rows[case], future=future.detach().cpu(), latent_nmse=float(nmse)))
                del action, future, target, nmse
            pending.append(case)
            self.progress('initial_student_calibration', staged_cases=len(pending), case_id=case)
        if pending:
            with reader_session(torch, self.runtime, self.bank, self.masters) as reader:
                for case in pending:
                    self.enough()
                    cache = torch.load(out / 'student_future' / f'{case}.pt', map_location='cpu', weights_only=True)
                    if cache['protocol'] != PROTOCOL or cache['case_id'] != case or cache['plan_row'] != self.rows[case]:
                        raise ValueError('Initial future cache differs')
                    target = self.teacher(case)
                    parity = None
                    if case == CASES[0]:
                        actual, _ = self.reader_action(reader, case, target['future'], False)
                        parity = float((actual.float().cpu() - target['action'].float()).abs().max())
                        if parity != 0:
                            raise ValueError('Frozen BF16 reader must replay teacher actions exactly')
                        del actual
                    action, counts = self.reader_action(reader, case, cache['future'], False)
                    values = reader_terms(torch, action, target['action'], stats, p['reader_source']['config'])
                    record = dict(protocol=PROTOCOL, case_id=case, sampler_seed=self.rows[case]['sampler_seeds'][0],
                        latent_nmse=cache['latent_nmse'], **{k: float(v) for k, v in values.items()},
                        counts=compact_counts(counts), pbs_jobid=self.job)
                    if parity is not None:
                        record['teacher_action_max_abs'] = parity
                    save(out / 'cases' / f'{case}.json', record)
                    del action, counts, target, values, cache
                    self.progress('initial_reader_calibration', initial_cases=case-CASES[0]+1, case_id=case)
        records = [read(out / 'cases' / f'{case}.json') for case in CASES]
        means = {k: sum(row[k] for row in records) / 80 for k in ('latent_nmse', 'reader_kl', 'reader_mse')}
        prep = dict(protocol=PROTOCOL, frozen_protocol=p, complete=True, heldout_used=False,
                    calibration_case_ids=CASES, beta=beta_from_means(p, means['latent_nmse'], means['reader_kl']),
                    stats=stats, initial_means=means, teacher_action_max_abs=records[0]['teacher_action_max_abs'],
                    pbs_jobid=self.job)
        save(summary_path, prep)
        return prep

    def kl_batch(self, batch, beta, stats, first):
        from reader_runtime import reader_session
        from reader_loss import reader_terms
        torch = self.torch
        staged, metrics = [], []
        for case in batch:
            action, future, _ = self.query(case)
            target = self.teacher(case)
            # Combine the loss cotangents in FP32, then cast once at the student boundary.
            a_leaf = action.detach().float().requires_grad_(True)
            z_leaf = future.detach().float().requires_grad_(True)
            total, _, _ = self.loss(a_leaf, z_leaf, target)
            ga, gz = torch.autograd.grad(total, (a_leaf, z_leaf))
            metrics.append(self.metrics(action, future, target))
            staged.append(dict(case=case, action=action.detach().cpu(), future=future.detach().cpu(),
                               ga=ga.detach().cpu(), gz=gz.detach().cpu()))
            del action, future, target, a_leaf, z_leaf, total, ga, gz
        reader_metrics = []
        with reader_session(torch, self.runtime, self.bank, self.masters) as reader:
            for index, row in enumerate(staged):
                case = row['case']
                leaf = row['future'].to(device='cuda', dtype=torch.float32).detach().requires_grad_(True)
                plain = None
                if first and index == 0:
                    plain, _ = self.reader_action(reader, case, leaf.detach(), False)
                    plain = plain.detach().cpu()
                action, _ = self.reader_action(reader, case, leaf, True)
                if plain is not None and float((action.float().cpu() - plain.float()).abs().max()) != 0:
                    raise ValueError('Reader grad/no-grad output differs')
                target = self.teacher(case)
                terms = reader_terms(torch, action, target['action'], stats, self.p['reader_source']['config'])
                rg, = torch.autograd.grad(beta * terms['reader_kl'], leaf)
                if (not bool(torch.isfinite(rg).all()) or bool((rg[:, :, :1] != 0).any())
                        or not bool((rg[:, :, 1:] != 0).any())):
                    raise ValueError('KL future cotangent must be finite, nonzero, and exclude frame0')
                row['gz'] += rg.detach().cpu()
                reader_metrics.append(dict(reader_kl=float(terms['reader_kl'].detach()),
                    reader_mse=float(terms['reader_mse'].detach()), future_gradient_norm=float(rg.float().norm())))
                del leaf, action, plain, terms, target, rg
        parity = 0.
        for row in staged:
            action, future, _ = self.query(row['case'], grad=True)
            error = max(float((action.detach().float().cpu()-row['action'].float()).abs().max()),
                        float((future.detach().float().cpu()-row['future'].float()).abs().max()))
            if error != 0:
                raise ValueError('Staged student replay must match full action/future exactly')
            parity = max(parity, error)
            torch.autograd.backward((action, future), ((row['ga']/10).to(action), (row['gz']/10).to(future)))
            self.count('parameter_reverse_passes')
            del action, future
        return metrics, {k: sum(v[k] for v in reader_metrics)/10 for k in reader_metrics[0]}, parity

    def train(self):
        torch, p = self.torch, self.p
        self.load_model()
        action, future, _ = self.query(CASES[0])
        initial = self.metrics(action, future, self.teacher(CASES[0]))
        expected = read(p['source_initialization']['common_endpoint'])
        error = max(abs(initial[k]-expected[k]) for k in self.helpers.METRICS)
        if error > 1e-6:
            raise ValueError('Shared R0/VQ0 initialization differs')
        del action, future
        setup = dict(protocol=PROTOCOL, arm=self.arm, pbs_jobid=self.job, passed=True,
            source_job=p['source_initialization']['job'], initial_parity_max_abs=error,
            initial_metrics=initial, target_linears=614, trainable_parameters=1228,
            frozen_rotation=True, new_teacher_queries=0)
        save(self.unit / 'setup_checks.json', setup)
        prep = self.preparation()
        beta = prep['beta']
        optimizer = torch.optim.Adam(self.bank.parameters(), lr=p['training']['vq_learning_rate'])
        state = dict(updates=0, exposures=0, history=[], lineage=[])
        checkpoint = self.unit / 'checkpoint.pt'
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if payload['protocol'] != p or payload['arm'] != self.arm or payload['beta'] != beta:
                raise ValueError('Committed training recipe differs')
            state = payload['state']
            if (state['updates'] != len(state['history']) or state['exposures'] != state['updates']*10
                    or [h['cases'] for h in state['history']] != self.schedule[:state['updates']]):
                raise ValueError('Committed training schedule differs')
            self.bank.restore(payload['module_masters'])
            optimizer.load_state_dict(payload['optimizer'])
            self.optimizer_to_device(torch, optimizer, 'cuda')
            self.restore_rng(torch, payload['rng'])
        if self.job not in state['lineage']:
            state['lineage'].append(self.job)
        while state['updates'] < 8:
            self.enough()
            update, started = state['updates'], time.monotonic()
            batch = self.schedule[update]
            self.progress('training', updates=update, exposures=state['exposures'], beta=beta)
            optimizer.zero_grad(set_to_none=True)
            extra, replay_error = {}, 0.
            if self.arm == 'az':
                metrics = []
                for case in batch:
                    action, future, _ = self.query(case, grad=True)
                    target = self.teacher(case)
                    total, _, _ = self.loss(action, future, target)
                    if not future.requires_grad or not bool(torch.isfinite(total)):
                        raise ValueError('Connected finite future loss required')
                    metrics.append(self.metrics(action.detach(), future.detach(), target))
                    (total/10).backward()
                    self.count('parameter_reverse_passes')
                    del action, future, total, target
            else:
                metrics, extra, replay_error = self.kl_batch(batch, beta, prep['stats'], update == 0)
            coverage = self.bank.gradient_report()
            if coverage['connected'] != 614 or coverage['finite'] != 614 or coverage['nonzero'] < 1:
                raise ValueError('VQ gradient coverage failed')
            if (any(v.grad is not None for v in self.runtime.model.parameters())
                    or any(v.grad is not None for v in self.bank.rotations.parameters())):
                raise ValueError('Frozen original model/R0 accumulated gradients')
            norm = torch.nn.utils.clip_grad_norm_(self.bank.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            if update == 0:
                optimizer.state.clear()
            self.bank.refresh()
            state['updates'] += 1
            state['exposures'] += 10
            state['history'].append(dict(update=state['updates'], cases=batch,
                before={k: sum(m[k] for m in metrics)/10 for k in self.helpers.METRICS},
                beta=beta, **extra, coverage=coverage, vq_gradient_norm=float(norm),
                staged_replay_max_abs=replay_error, seconds=time.monotonic()-started, pbs_jobid=self.job))
            self.atomic_torch_save(torch, checkpoint, dict(protocol=p, arm=self.arm, beta=beta, state=state,
                module_masters=self.bank.snapshot(), optimizer=self.to_cpu(optimizer.state_dict()), rng=self.capture_rng(torch)))
            save(self.unit / 'training.json', dict(protocol=PROTOCOL, arm=self.arm, **state))
            self.progress('training_committed', updates=state['updates'], exposures=state['exposures'],
                          beta=beta, last_update_seconds=state['history'][-1]['seconds'])
            gc.collect()
            torch.cuda.empty_cache()
        for value in self.bank.parameters():
            value.requires_grad_(False)
            value.grad = None
        optimizer.state.clear()
        frozen = all(torch.equal(v.detach().cpu(), self.initial_rotation[s]) for s,v in self.bank.rotations.raw.items())
        if not frozen:
            raise ValueError('R0 changed during VQ calibration')
        deployment = dict(protocol=p, arm=self.arm, pbs_jobid=self.job, updates=8, exposures=80,
            rotations=self.bank.rotations.snapshot(), module_masters=self.bank.snapshot(),
            bank_directory=p['source_initialization']['bank_directory'],
            bank_protocol=p['source_initialization']['protocol'], target_linears=614,
            frozen_rotation=True, beta=beta, D=1, r=1)
        self.atomic_torch_save(torch, self.unit / 'deployment.pt', deployment)
        action, future, counts = self.query(CASES[0])
        value, description = self.datum(CASES[0])
        self.runtime.description = description
        self.count('formal_infer_queries')
        actual, _ = self.runtime.infer(value, self.rows[CASES[0]]['sampler_seeds'][0], measure=False)
        parity = float((action.float().cpu()-actual.float().cpu()).abs().max())
        if not math.isfinite(parity) or parity > 1e-6:
            raise ValueError('Final source sampler/formal infer parity failed')
        anchor = dict(protocol=PROTOCOL, arm=self.arm, pbs_jobid=self.job, updates=8,
            exposures=80, case_id=CASES[0], seed_index=0, sampler_seed=self.rows[CASES[0]]['sampler_seeds'][0],
            plan_row=self.rows[CASES[0]], suite=self.rows[CASES[0]]['suite'], counts=counts,
            metrics=self.metrics(action, future, self.teacher(CASES[0])), target_linears=614,
            frozen_rotation=True, formal_parity_max_abs=parity)
        self.atomic_torch_save(torch, self.unit / 'anchor.pt', dict(**anchor, action=action.detach().cpu(), future=future.detach().cpu()))
        save(self.unit / 'anchor.json', anchor)
        summary = dict(protocol=PROTOCOL, arm=self.arm, pbs_jobid=self.job, status='complete', complete=True,
            updates=8, exposures=80, target_linears=614, frozen_rotation=frozen,
            initial_parity_max_abs=error, formal_parity_max_abs=parity, final_anchor=True,
            new_teacher_queries=0, beta=beta, counters=self.counters, lineage=state['lineage'],
            seconds=time.monotonic()-self.started, orthogonality=self.bank.rotations.orthogonality())
        save(self.unit / 'summary.json', summary)
        self.progress('complete', updates=8, exposures=80, beta=beta)
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    try:
        Work(p, args.arm).train()
    except BaseException as error:
        save(Path(p['remote_root'])/'results/arms'/args.arm/'failure.json',
             dict(protocol=PROTOCOL, arm=args.arm, pbs_jobid=os.environ.get('PBS_JOBID'),
                  error=type(error).__name__, message=str(error), checkpointed=isinstance(error,TimeoutError)))
        raise


if __name__ == '__main__':
    main()
