"""Reuse the frozen C80 runtime; transport probes then matched Comp8/Uncomp8."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import time

import base_worker as base

PROTOCOL = 'fastwam-vq-r128-c80-transport-v1'
CASES, HELDOUT, METRICS = base.CASES, base.HELDOUT, base.METRICS
read, save = base.read, base.save
PROBE_ARMS = ['Seq_A4', 'Seq_to_Joint_A4', 'Joint_A4', 'Joint_to_Seq_A4',
              'Seq_noA4', 'Seq_to_Joint_noA4', 'Joint_noA4', 'Joint_to_Seq_noA4']


def units(shard=None):
    calibration = CASES if shard is None else CASES[shard::2]
    heldout = HELDOUT if shard is None else HELDOUT[shard::2]
    return [('calibration', c, 0) for c in calibration] + [('heldout', c, s) for c in heldout for s in (0, 1)]


def validate(p):
    base.validate_protocol(p['base_protocol'])
    t = p['training']
    if (p['protocol'] != PROTOCOL or p['training_arms'] != ['Comp8', 'Uncomp8']
            or t['updates'] != 8 or t['exposures'] != 80 or t['contexts_per_update'] != 10
            or t['vq_learning_rate'] != .0005 or t['rotation_learning_rate'] != .00001
            or t['latent_weight'] != .04245452683169643 or t['extra_losses'] != []
            or p['allocation'] != p['base_protocol']['allocation']
            or p['source_dual']['job'] != '25912619.pbs101'
            or p['source_dual']['protocol'] != p['base_protocol']['protocol']
            or p['source_dual']['root'] != p['base_protocol']['remote_root']
            or p['probe_arms'] != PROBE_ARMS
            or p['cohorts']['calibration']['case_ids'] != CASES
            or p['cohorts']['calibration']['seed_indices'] != [0]
            or p['cohorts']['heldout']['case_ids'] != HELDOUT
            or p['cohorts']['heldout']['seed_indices'] != [0, 1]):
        raise RuntimeError('Frozen transport design differs')


class Work(base.Work):
    def __init__(self, p, stage, shard=0, arm=None, execution=None):
        self.config = p
        self.execution = execution
        if execution is not None and (execution.get('mode') != 'comp_only_reuse_joint8'
                or execution.get('resume_arm') != 'Comp8'
                or execution.get('source_cancelled_job') != '25919309.pbs101'
                or execution.get('reference') != dict(arm='Joint8', job='25912619.pbs101', kind='historical_reuse')
                or execution.get('allocation') != dict(gpus=1, ncpus=16, memory_gb=110, walltime='04:00:00', execution_queue='g1')
                or not execution.get('scientific_protocol_unchanged')):
            raise RuntimeError('Single-GPU execution amendment differs')
        validate(p)
        data_p = copy.deepcopy(p['base_protocol'])
        data_p['remote_root'] = p['remote_root']
        super().__init__(data_p, stage, shard, arm)
        self.source_dual = Path(p['source_dual']['root'])
        self.old_out = self.source_dual / 'results'
        s = read(self.old_out / 'summary.json')
        if (read(self.source_dual / 'protocol.json') != p['base_protocol'] or not s.get('complete')
                or s.get('pbs_jobid') != p['source_dual']['job'] or s.get('target_linears') != 614
                or s.get('endpoints', {}).get('Seq8', {}).get('heldout') != 80
                or s.get('endpoints', {}).get('Joint8', {}).get('heldout') != 80
                or not read(self.old_out / 'banks/receipt.json').get('passed')):
            raise RuntimeError('Completed dual-arm source binding differs')
        artifact = self.source_dual / 'artifacts' / p['source_dual']['job']
        if (artifact / 'exit_code.txt').read_text().strip() != '0' or not (artifact / 'PHASE_COMPLETE').exists():
            raise RuntimeError('Original dual job is not artifact-confirmed successful')
        self.deployments = {}
        self.student_queries, self.gradient_queries, self.formal_queries = 0, 0, 0

    def progress(self, **extra):
        value = dict(protocol=PROTOCOL, stage=self.stage, shard=self.shard, arm=self.arm,
                     pbs_jobid=self.job, seconds=time.monotonic() - self.started)
        value.update(extra)
        save(self.unit / 'progress.json', value)
        if self.shard == 0 and self.arm != 'Uncomp8':
            save(self.out / 'progress.json', value)
        print('TRANSPORT_PROGRESS ' + json.dumps(value, allow_nan=False), flush=True)

    def complete(self, count, **extra):
        result = dict(protocol=PROTOCOL, stage=self.stage, shard=self.shard, arm=self.arm,
                      pbs_jobid=self.job, status='complete', complete=True, units=count,
                      seconds=time.monotonic() - self.started, student_queries=self.student_queries,
                      gradient_queries=self.gradient_queries, formal_queries=self.formal_queries, **extra)
        save(self.unit / 'summary.json', result)
        self.progress(status='complete', units=count)
        return result

    def load_model(self):
        from transport_bank import TransportBanks
        super().load_model(vq=False)
        self.rb.close()
        self.bank = TransportBanks(self.runtime, self.config['source_dual']['banks'], self.rb.rotations,
                                   self.rb.transforms, self.config['source_dual']['protocol'])
        for arm in ('Seq8', 'Joint8'):
            d = self.torch.load(self.old_out / f'arms/{arm}/deployment.pt', map_location='cpu', weights_only=True)
            if (d.get('protocol') != self.config['base_protocol'] or d.get('arm') != arm
                    or d.get('updates') != 8 or d.get('exposures') != 80 or d.get('target_linears') != 614
                    or set(d['module_masters']) != set(self.bank.modules)):
                raise RuntimeError('Source deployment identity differs: ' + arm)
            self.deployments[arm] = d
        if not all(self.torch.equal(v, self.initial_rotation[s]) for s, v in self.deployments['Seq8']['rotations'].items()):
            raise RuntimeError('Seq8 is not anchored to the original R0')

    def query(self, case, seed=0, grad=False):
        self.student_queries += 1
        self.gradient_queries += int(grad)
        self.enough()
        torch, bank = self.torch, self.bank
        value, description = self.datum(case)
        self.runtime.description = description
        self.runtime.current_arm = self.arm or 'R0_VQ0'
        bank.seen.clear()
        bank.query_reset()
        bank.rotations.refresh()
        with torch.set_grad_enabled(grad):
            for stream in bank.rotations.raw:
                bank.rotations.matrix(stream)
                if bank.compensated:
                    # Prime outside sampler checkpoints so recomputation has the same graph.
                    bank.relative(stream)
            action, counts, future = self.sampler.differentiable_action(
                self.runtime, value, self.rows[case]['sampler_seeds'][seed], checkpointed=grad, return_future=True)
        if len(bank.seen) != 614:
            raise RuntimeError('Sampler did not reach all 614 target Linears')
        return action, future, counts

    def freeze(self):
        for parameter in self.bank.parameters() + list(self.bank.rotations.parameters()):
            parameter.requires_grad_(False)
        self.bank.rotations.refresh()
        self.bank.query_reset()

    def binding(self, arm, split, case, seed):
        return dict(protocol=PROTOCOL, arm=arm, split=split, case_id=case, seed_index=seed,
                    suite=self.rows[case]['suite'], sampler_seed=self.rows[case]['sampler_seeds'][seed])

    def endpoint(self, arm, split, case, seed, *, values=None, source_binding=None):
        path = self.out / f'endpoints/{arm}/{split}/{case}_{seed}.json'
        binding = self.binding(arm, split, case, seed)
        if path.exists():
            record = read(path)
            if any(record.get(k) != v for k, v in binding.items()):
                raise RuntimeError('Saved endpoint binding differs')
        else:
            if values is None:
                action, future, counts = self.query(case, seed)
                values = self.metrics(action, future, self.teacher(case, seed))
                del action, future
            else:
                counts = None
            record = dict(binding, **{k: values[k] for k in METRICS}, counts=counts,
                          source_binding=source_binding, pbs_jobid=self.job)
            save(path, record)
        if any(not math.isfinite(record[k]) or record[k] < 0 for k in METRICS):
            raise RuntimeError('Invalid endpoint metric')
        return record

    def original(self, arm, split, case, seed):
        r = read(self.old_out / f'endpoints/{arm}/{split}/{case}_{seed}.json')
        expected = dict(self.binding(arm, split, case, seed), protocol=self.config['source_dual']['protocol'])
        if any(r.get(k) != v for k, v in expected.items()):
            raise RuntimeError('Source endpoint binding differs')
        return r

    def set_source(self, arm):
        d = self.deployments[arm]
        self.bank.restore(d['module_masters'])
        self.bank.set_anchor(d['rotations'])
        self.bank.set_target(d['rotations'], compensated=False)
        self.freeze()

    def run_probe(self):
        from unit_checks import run
        torch = self.torch
        unit = run(self.config['gates'])
        save(self.unit / 'unit.json', unit)
        self.load_model()
        self.bank.compensated = True
        # Reuse the old bounded repeated forward/backward/Adam probe, with the new compensated bank.
        base.Work.probe(self)
        numeric = torch.load(self.unit / 'probe.pt', map_location='cpu', weights_only=True)
        numeric.update(protocol=PROTOCOL, compensated=True)
        self.atomic_torch_save(torch, self.unit / 'probe.pt', numeric)
        repeat = numeric['repeat_comparison']
        reference = torch.load(self.old_out / 'stages/common_eval_0/probe.pt', map_location='cpu', weights_only=True)
        if reference.get('protocol') != self.config['source_dual']['protocol']:
            raise RuntimeError('Original first-update probe binding differs')
        old, new = reference['probe'], numeric['probe']
        vq_names = [n for n in new['gradients'] if not n.startswith('rotation/')]
        if len(vq_names) != 1228 or set(vq_names) != {n for n in old['gradients'] if not n.startswith('rotation/')}:
            raise RuntimeError('Original initial VQ parameter coverage differs')
        maxima = dict(forward_max_abs=0., vq_gradient_max_abs=0., vq_delta_max_abs=0.)
        for name in ('action', 'future'):
            maxima['forward_max_abs'] = max(maxima['forward_max_abs'], float((new[name].float() - old[name].float()).abs().max()))
            if not torch.allclose(new[name], old[name], atol=1e-6, rtol=0):
                raise RuntimeError('Compensated initial output changed the archived probe')
        for name in vq_names:
            g, og, d, od = new['gradients'][name], old['gradients'][name], new['delta'][name], old['delta'][name]
            maxima['vq_gradient_max_abs'] = max(maxima['vq_gradient_max_abs'], float((g - og).abs().max()))
            maxima['vq_delta_max_abs'] = max(maxima['vq_delta_max_abs'], float((d - od).abs().max()))
            if not torch.allclose(g, og, atol=1e-7, rtol=1e-4) or not torch.allclose(d, od, atol=1e-7, rtol=0):
                raise RuntimeError('Compensation changed the initial VQ gradient/Adam delta')
        initial_vq = dict(passed=True, complete=True, parameters=len(vq_names), **maxima)
        save(self.unit / 'initial_vq_receipt.json', initial_vq)
        del reference, old, new
        del numeric
        selected, source_error = units(self.shard), 0.
        self.bank.cache_evaluation_weights = True
        for anchor, source_arm, target_arm in (('Seq', 'Seq8', 'Joint8'), ('Joint', 'Joint8', 'Seq8')):
            self.set_source(source_arm)
            source_alias = anchor + '_A4'
            for split, case, seed in selected:
                old = self.original(source_arm, split, case, seed)
                self.endpoint(source_alias, split, case, seed, values=old,
                              source_binding=dict(protocol=self.config['source_dual']['protocol'],
                                                  arm=source_arm, job=self.config['source_dual']['job']))
            for split, case, seed in (selected[0], next(v for v in selected if v[0] == 'heldout')):
                action, future, _ = self.query(case, seed)
                values = self.metrics(action, future, self.teacher(case, seed))
                error = max(abs(values[k] - self.original(source_arm, split, case, seed)[k]) for k in METRICS)
                source_error = max(source_error, error)
                if error > self.config['gates']['source_metric_atol']:
                    raise RuntimeError('Source A4 replay differs')
                del action, future
            self.bank.activation_bits = 16
            for index, (split, case, seed) in enumerate(selected):
                cached = self.out / f'probe_tensors/{anchor}/{split}/{case}_{seed}.pt'
                if not cached.exists():
                    action, future, _ = self.query(case, seed)
                    values = self.metrics(action, future, self.teacher(case, seed))
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    self.atomic_torch_save(torch, cached, dict(binding=self.binding(anchor + '_noA4', split, case, seed),
                        action=action.detach().cpu(), future=future.detach().cpu(), metrics=values))
                    del action, future
                else:
                    payload = torch.load(cached, map_location='cpu', weights_only=True)
                    if payload['binding'] != self.binding(anchor + '_noA4', split, case, seed):
                        raise RuntimeError('Cached noA4 output binding differs')
                    values = payload['metrics']
                    del payload
                self.endpoint(anchor + '_noA4', split, case, seed, values=values)
                self.progress(status='source_noA4', anchor=anchor, units=index + 1)
            transported = 'Seq_to_Joint' if anchor == 'Seq' else 'Joint_to_Seq'
            self.bank.set_target(self.deployments[target_arm]['rotations'], compensated=True)
            self.bank.prepare_evaluation()
            for bits, suffix in ((4, '_A4'), (16, '_noA4')):
                self.bank.activation_bits = bits
                for index, (split, case, seed) in enumerate(selected):
                    self.enough()
                    alias = transported + suffix
                    pair_path = self.out / f'pairs/{anchor}/{split}/{case}_{seed}.json'
                    endpoint_path = self.out / f'endpoints/{alias}/{split}/{case}_{seed}.json'
                    if bits == 4:
                        self.endpoint(alias, split, case, seed)
                    elif not pair_path.exists() or not endpoint_path.exists():
                        action, future, _ = self.query(case, seed)
                        values = self.metrics(action, future, self.teacher(case, seed))
                        old = torch.load(self.out / f'probe_tensors/{anchor}/{split}/{case}_{seed}.pt', map_location='cuda', weights_only=True)
                        if old['binding'] != self.binding(anchor + '_noA4', split, case, seed):
                            raise RuntimeError('Paired noA4 output binding differs')
                        da, dz = action.float() - old['action'].float(), future.float() - old['future'].float()
                        pair = dict(protocol=PROTOCOL, anchor=anchor, split=split, case_id=case, seed_index=seed,
                            suite=self.rows[case]['suite'], action_max_abs=float(da.abs().max()),
                            future_max_abs=float(dz.abs().max()),
                            action_nmse=float(da.square().sum() / old['action'].float().square().sum().clamp_min(1e-12)),
                            future_nmse=float(dz.square().sum() / old['future'].float().square().sum().clamp_min(1e-12)),
                            pbs_jobid=self.job)
                        save(pair_path, pair)
                        self.endpoint(alias, split, case, seed, values=values)
                        del action, future, old, da, dz
                    self.progress(status='transport', anchor=anchor, activation_bits=bits, units=index + 1)
        self.complete(80, source_replay_max_abs=source_error, unit=unit, repeat=repeat, initial_vq=initial_vq,
                      probe_counts={a: dict(calibration=40, heldout=40) for a in self.config['probe_arms']})

    def gate(self):
        torch = self.torch
        stages = [read(self.out / f'stages/probe_{i}/summary.json') for i in (0, 1)]
        if any(s.get('protocol') != PROTOCOL or not s.get('complete') or s.get('units') != 80
               or not s.get('unit', {}).get('passed') or not s.get('repeat', {}).get('passed')
               or not s.get('initial_vq', {}).get('passed')
               or s.get('source_replay_max_abs', 1) > self.config['gates']['source_metric_atol'] for s in stages):
            raise RuntimeError('Probe prerequisites are incomplete')
        probes = [torch.load(self.out / f'stages/probe_{i}/probe.pt', map_location='cpu', weights_only=True) for i in (0, 1)]
        if any(p.get('protocol') != PROTOCOL or p.get('compensated') is not True for p in probes):
            raise RuntimeError('Cross-device probe bindings differ')
        repeat = base.compare_probe(torch, probes[0]['probe'], probes[1]['probe'], self.p)
        for alias in self.config['probe_arms']:
            for split, case, seed in units():
                if not (self.out / f'endpoints/{alias}/{split}/{case}_{seed}.json').exists():
                    raise RuntimeError('A full-cohort probe endpoint is missing')
                self.endpoint(alias, split, case, seed)
        receipt = dict(protocol=PROTOCOL, complete=True, passed=True, source_dual=self.config['source_dual'],
                       source_replay_max_abs=max(s['source_replay_max_abs'] for s in stages))
        save(self.out / 'source_receipt.json', receipt)
        save(self.out / 'unit_receipt.json', dict(protocol=PROTOCOL, complete=True, passed=True,
             units=[s['unit'] for s in stages], initial_vq=[s['initial_vq'] for s in stages],
             within_process=[s['repeat'] for s in stages], cross_device=repeat))
        save(self.out / 'gate_receipt.json', dict(protocol=PROTOCOL, complete=True, passed=True,
             source=True, fp32_invariance=True, gradient_coupling=True, cross_device=repeat))
        if self.stage == 'gate':
            self.complete(1, source=True, fp32_invariance=True, gradient_coupling=True)

    def train(self):
        if self.execution is not None and self.arm != 'Comp8':
            raise RuntimeError('User stopped Uncomp8; only Comp8 may resume')
        if not read(self.out / 'gate_receipt.json').get('passed'):
            raise RuntimeError('Transport gates did not pass')
        self.load_model()
        torch, bank, arm = self.torch, self.bank, self.arm
        bank.compensated = arm == 'Comp8'
        bank.activation_bits, bank.cache_evaluation_weights = 4, False
        for value in bank.parameters() + list(bank.rotations.parameters()):
            value.requires_grad_(True)
        optimizer = self.optimizer(True)
        state = dict(updates=0, exposures=0, history=[], lineage=[])
        state_path = self.unit / 'state.pt'
        if self.execution is not None and not state_path.exists():
            raise RuntimeError('Single-GPU continuation requires the existing Comp8 checkpoint')
        if state_path.exists():
            saved = torch.load(state_path, map_location='cpu', weights_only=True)
            if saved.get('protocol') != self.config or saved.get('arm') != arm:
                raise RuntimeError('Training resume binding differs')
            state = saved['state']
            bank.restore(saved['module_masters'])
            bank.rotations.restore(saved['rotations'])
            optimizer.load_state_dict(saved['optimizer'])
            self.optimizer_to_device(torch, optimizer, 'cuda')
            self.restore_rng(torch, saved['rng'])
            if state['updates'] != len(state['history']) or state['exposures'] != state['updates'] * 10:
                raise RuntimeError('Training checkpoint cursor differs')
            if self.execution is not None:
                if ('25919309.pbs101' not in state['lineage'] or state['updates'] < 3
                        or any(row['cases'] != self.schedule[i] for i, row in enumerate(state['history']))):
                    raise RuntimeError('Comp8 continuation lost the authorized checkpoint/schedule')
                resume_receipt = dict(protocol=PROTOCOL, pbs_jobid=self.job, arm=arm,
                    updates_at_resume=state['updates'], exposures_at_resume=state['exposures'],
                    optimizer_restored=True, rng_restored=True, scientific_protocol_exact=True,
                    source_lineage=list(state['lineage']))
                save(self.unit / 'resume_receipt.json', resume_receipt)
                print('COMP8_RESUMED ' + json.dumps(resume_receipt), flush=True)
        if self.job not in state['lineage']:
            state['lineage'].append(self.job)

        def persist():
            self.atomic_torch_save(torch, state_path, dict(protocol=self.config, arm=arm, state=state,
                module_masters=bank.snapshot(), rotations=bank.rotations.snapshot(), anchor_rotations=self.initial_rotation,
                optimizer=self.to_cpu(optimizer.state_dict()), rng=self.capture_rng(torch)))

        if not state['updates']:
            action, future, _ = self.query(CASES[0])
            actual = self.metrics(action, future, self.teacher(CASES[0], 0))
            old = self.original('R0_VQ0', 'calibration', CASES[0], 0)
            state['initial_parity_max_abs'] = max(abs(actual[k] - old[k]) for k in METRICS)
            if state['initial_parity_max_abs'] > self.config['gates']['initial_metric_atol']:
                raise RuntimeError('Training start changed R0+VQ0 arithmetic')
            del action, future
        persist()
        source_history = read(self.old_out / 'arms/Joint8/training.json')['history']
        while state['updates'] < 8 or (state['history'] and 'after' not in state['history'][-1]):
            if state['history'] and 'after' not in state['history'][-1]:
                post = []
                for case in state['history'][-1]['cases']:
                    action, future, _ = self.query(case)
                    post.append(self.metrics(action, future, self.teacher(case, 0)))
                    del action, future
                state['history'][-1]['after'] = {k: sum(r[k] for r in post) / 10 for k in METRICS}
                if arm == 'Uncomp8':
                    old = source_history[state['history'][-1]['update'] - 1]
                    error = max(abs(state['history'][-1][part][k] - old[part][k]) for part in ('before', 'after') for k in METRICS)
                    state['history'][-1]['source_training_replay_max_abs'] = error
                    # Keep all endpoints even if reproduction differs; final attribution becomes inconclusive.
                persist()
                save(self.unit / 'training.json', dict(protocol=PROTOCOL, arm=arm, **state))
                self.progress(status='training', updates=state['updates'], exposures=state['exposures'], units=state['updates'])
                if state['updates'] == 8:
                    break
            self.enough()
            batch = self.schedule[state['updates']]
            tick, before = time.monotonic(), []
            optimizer.zero_grad(set_to_none=True)
            for case in batch:
                action, future, _ = self.query(case, grad=True)
                target = self.teacher(case, 0)
                loss, _, _ = self.loss(action, future, target)
                if not future.requires_grad or not bool(torch.isfinite(loss)):
                    raise RuntimeError('Disconnected/nonfinite training objective')
                before.append(self.metrics(action.detach(), future.detach(), target))
                (loss / 10).backward()
                del action, future, target, loss
            coverage = bank.gradient_report()
            rgrad = {s: bool(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0)
                     for s, v in bank.rotations.raw.items()}
            if coverage['connected'] != 614 or coverage['finite'] != 614 or not all(rgrad.values()):
                raise RuntimeError('Joint gradient coverage failed')
            vnorm = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1, error_if_nonfinite=True)
            rnorm = torch.nn.utils.clip_grad_norm_(list(bank.rotations.parameters()), 1, error_if_nonfinite=True)
            optimizer.step()
            if state['updates'] == 0:
                for value in bank.parameters():
                    optimizer.state.pop(value, None)
            bank.refresh()
            bank.rotations.refresh()
            state['updates'] += 1
            state['exposures'] += 10
            state['history'].append(dict(update=state['updates'], cases=batch,
                before={k: sum(r[k] for r in before) / 10 for k in METRICS}, coverage=coverage,
                rotation_gradients=rgrad, vq_gradient_norm=float(vnorm), rotation_gradient_norm=float(rnorm),
                seconds=time.monotonic() - tick))
            persist()
        save(self.unit / 'training.json', dict(protocol=PROTOCOL, arm=arm, **state))
        self.freeze()
        orth = bank.rotations.orthogonality()
        if max(orth.values()) > self.config['gates']['orthogonality_atol']:
            raise RuntimeError('Final orthogonality failed')
        self.atomic_torch_save(torch, self.unit / 'deployment.pt', dict(protocol=self.config, arm=arm,
            rotations=bank.rotations.snapshot(), anchor_rotations=self.initial_rotation, module_masters=bank.snapshot(),
            bank_directory=self.config['source_dual']['banks'], target_linears=614, updates=8, exposures=80,
            compensated=bank.compensated, execution='explicit_post_decode_BF16_weight_transport', D=1, r=1))
        bank.cache_evaluation_weights = True
        bank.prepare_evaluation()
        action, future, _ = self.query(CASES[0])
        deployed = torch.load(self.unit / 'deployment.pt', map_location='cpu', weights_only=True)
        bank.load_deployment(deployed, self.config)
        self.freeze()
        bank.prepare_evaluation()
        reloaded_action, reloaded_future, _ = self.query(CASES[0])
        reload_error = max(float((action.float() - reloaded_action.float()).abs().max()),
                           float((future.float() - reloaded_future.float()).abs().max()))
        if reload_error > self.config['gates']['formal_parity_atol']:
            raise RuntimeError('Saved transport deployment round trip changed action/future')
        del deployed, reloaded_action, reloaded_future
        datum, _ = self.datum(CASES[0])
        actual, _ = self.runtime.infer(datum, self.rows[CASES[0]]['sampler_seeds'][0], measure=False)
        self.formal_queries += 1
        parity = float((action.float().cpu() - actual.float().cpu()).abs().max())
        if parity > self.config['gates']['formal_parity_atol']:
            raise RuntimeError('Formal infer parity failed')
        del action, future, actual
        for index, (split, case, seed) in enumerate(units()):
            self.endpoint(arm, split, case, seed)
            self.progress(status='evaluating', units=index + 1, updates=8, exposures=80, last_query=[split, case, seed])
        self.complete(8, updates=8, exposures=80, calibration=80, heldout=80,
            all_parameters_frozen=all(not v.requires_grad for v in bank.parameters() + list(bank.rotations.parameters())),
            initial_parity_max_abs=state['initial_parity_max_abs'], formal_infer_max_abs=parity, orthogonality=orth,
            deployment_reload_max_abs=reload_error, compensated=bank.compensated, lineage=state['lineage'])

    def finalize(self):
        from aggregate import build_comparison, render_markdown
        self.gate()
        training_arms = ['Comp8'] if self.execution is not None else self.config['training_arms']
        if self.execution is not None:
            for split, case, seed in units():
                old = self.original('Joint8', split, case, seed)
                row = self.endpoint('Uncomp8', split, case, seed, values=old,
                    source_binding=dict(protocol=self.config['source_dual']['protocol'], arm='Joint8',
                                        job=self.config['source_dual']['job'], kind='historical_reuse'))
                if row.get('source_binding', {}).get('kind') != 'historical_reuse':
                    raise RuntimeError('Uncomp8 alias is not the authorized historical reference')
        arms = self.config['probe_arms'] + self.config['training_arms']
        records = {a: {split: [read(self.out / f'endpoints/{a}/{sp}/{c}_{s}.json')
                              for sp, c, s in units() if sp == split]
                       for split in ('calibration', 'heldout')} for a in arms}
        pairs = [read(self.out / f'pairs/{anchor}/{sp}/{c}_{s}.json') for anchor in ('Seq', 'Joint') for sp, c, s in units()]
        summaries = {a: read(self.out / f'arms/{a}/summary.json') for a in training_arms}
        if any(not s.get('complete') or s.get('updates') != 8 or s.get('exposures') != 80
               or not s.get('all_parameters_frozen') or s.get('calibration') != 80 or s.get('heldout') != 80
               or s.get('formal_infer_max_abs', 1) > 1e-6 for s in summaries.values()):
            raise RuntimeError('Training endpoints incomplete')
        training = {a: read(self.out / f'arms/{a}/training.json') for a in training_arms}
        comparison = build_comparison(self.config, records, pairs, dict(pbs_jobid=self.job,
            source_dual=self.config['source_dual'], shared_R0_VQ0=True,
            training_budget='Comp8 8x10; original Joint8 8x10 reused' if self.execution else '8x10 for both arms',
            uncomp8_reference='historical_Joint8' if self.execution else 'concurrent_retrain',
            encoding_scope='decoded_weight_transport_reference', new_teacher_queries=0,
            formal_parity_max_error=max(s['formal_infer_max_abs'] for s in summaries.values())))
        save(self.out / 'comparison.json', comparison)
        (self.out / 'RESULTS.zh.md').write_text(render_markdown(comparison), encoding='utf-8')
        save(self.out / 'training.json', dict(protocol=PROTOCOL, arms=training))
        self.complete(1)
        result = dict(protocol=PROTOCOL, complete=True, status='complete', pbs_jobid=self.job,
            target_linears=614, new_teacher_queries=0, source_dual_binding=True, allocation_gpus=1 if self.execution else 2,
            execution_amendment=self.execution,
            reused_reference=dict(Uncomp8='Joint8', job=self.config['source_dual']['job']) if self.execution else None,
            encoding_scope='decoded_weight_transport_reference',
            probe_counts={a: dict(calibration=80, heldout=80) for a in self.config['probe_arms']},
            endpoints={a: dict(calibration=80, heldout=80, updates=8, exposures=80) for a in training_arms},
            gates=dict(source=True, fp32_invariance=True, gradient_coupling=True, initial_parity=True, frozen=True,
                formal_parity_max_error=max(s['formal_infer_max_abs'] for s in summaries.values()),
                deployment_reload_max_error=max(s['deployment_reload_max_abs'] for s in summaries.values()),
                orthogonality_max_error=max(v for s in summaries.values() for v in s['orthogonality'].values())),
            comparison_file='results/comparison.json', comparison_complete=True)
        save(self.out / 'summary.json', result)
        self.progress(status='complete', units=1, complete=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--execution')
    parser.add_argument('--stage', choices=('probe', 'gate', 'arm', 'finalize'), required=True)
    parser.add_argument('--shard', type=int, choices=(0, 1), default=0)
    parser.add_argument('--arm', choices=('Comp8', 'Uncomp8'))
    args = parser.parse_args()
    if (args.stage == 'arm') != (args.arm is not None):
        parser.error('--arm belongs only to the arm stage')
    work = Work(read(args.protocol), args.stage, args.shard, args.arm,
                execution=read(args.execution) if args.execution else None)
    summary_path = work.unit / 'summary.json'
    if args.stage != 'finalize' and summary_path.exists():
        old = read(summary_path)
        if old.get('protocol') != PROTOCOL:
            raise RuntimeError('Completed stage belongs to another protocol')
        if old.get('complete'):
            old.setdefault('origin_pbs_jobid', old['pbs_jobid'])
            old['pbs_jobid'] = work.job
            old['reused_complete_unit'] = True
            save(summary_path, old)
            work.progress(status='complete', units=old['units'], reused_complete_unit=True)
            return
    {'probe': work.run_probe, 'gate': work.gate, 'arm': work.train, 'finalize': work.finalize}[args.stage]()


if __name__ == '__main__':
    main()
