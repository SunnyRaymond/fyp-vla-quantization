"""Two-GPU C80 R0 -> shared MX-VQ0 -> frozen-R Seq8 / warm Joint8."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import time

PROTOCOL = 'fastwam-vq-r128-c80-dual-v1'
CASES = list(range(2000, 2080))
HELDOUT = list(range(3000, 3040))
METRICS = ('first10motor_rmse', 'first10motor_mse', 'future_latent_nmse', 'mixed_loss',
           'full32motor_rmse', 'late22motor_rmse', 'first10gripper_rmse')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_protocol(p):
    t, q, a = p['training'], p['quantization'], p['allocation']
    if (p['protocol'] != PROTOCOL or p['calibration']['case_ids'] != CASES
            or p['heldout']['case_ids'] != HELDOUT or p['heldout']['seed_indices'] != [0, 1]
            or p['calibration']['seed_indices'] != [0] or p['target_linears'] != 614
            or p['arms'] != ['Seq8', 'Joint8'] or p['endpoints'] != ['R0_VQ0', 'Seq8', 'Joint8']
            or t['vq_updates'] != 8 or t['contexts_per_update'] != 10 or t['training_exposures'] != 80
            or t['vq_learning_rate'] != .0005 or t['rotation_learning_rate'] != .00001
            or t['latent_weight'] != .04245452683169643 or t['rho'] != 1.5
            or t['normalization_c'] != .028303017887797623 or t['gradient_norm_clip'] != 1
            or not t['vq_optimizer_reset_after_update1'] or not t['rotation_optimizer_fresh']
            or q['D'] != 1 or q['clipping_ratio'] != 1 or q['native_kernel']
            or a['gpus'] != 2 or a['walltime'] != '08:00:00' or a['execution_queue'] != 'g2'
            or not p['numerics']['deterministic_algorithms']
            or p['source_rotation']['updates'] != 8 or p['source_rotation']['training_exposures'] != 80):
        raise RuntimeError('Frozen C80 dual-arm protocol differs')


def validate_schedule(schedule, rows, suites):
    if len(schedule) != 8 or any(len(batch) != 10 or len(set(batch)) != 10 for batch in schedule):
        raise RuntimeError('C80 must use eight distinct 10-context batches')
    if Counter(case for batch in schedule for case in batch) != Counter(CASES):
        raise RuntimeError('Every C80 context must appear exactly once')
    if Counter(rows[c]['suite'] for c in CASES) != Counter({s: 20 for s in suites}):
        raise RuntimeError('C80 suite coverage differs')
    for batch in schedule:
        if sorted(Counter(rows[c]['suite'] for c in batch).values()) != [2, 2, 3, 3]:
            raise RuntimeError('C80 batch suite balance differs')
    return schedule


def endpoint_units(p, shard=None):
    units = [('calibration', c, 0) for c in CASES]
    units += [('heldout', c, s) for c in HELDOUT for s in (0, 1)]
    return units if shard is None else units[shard::2]


def prepare_output_directories(out, unit):
    for path in (out / 'moments', out / 'banks', unit):
        path.mkdir(parents=True, exist_ok=True)


def validate_teacher_binding(value, row, seed, calibration=False):
    declared = value.get('seed_index')
    # The original C80 writer stores plan_row+sampler_seed under a seed-0-only receipt.
    if declared is None and calibration and seed == 0:
        declared = 0
    if (value.get('plan_row') != row or type(declared) is not int or declared != seed
            or value.get('sampler_seed') != row['sampler_seeds'][seed]):
        raise RuntimeError(f'BF16 teacher binding differs: case={row["case_id"]}, seed_index={seed}')


class Work:
    def __init__(self, p, stage, shard=0, arm=None):
        # No model-scale IO or numerical imports before the real allocation guard.
        import sampler
        sampler._require_compute_allocation()
        validate_protocol(p)
        import torch
        import numpy as np
        import runner
        import calibration_source as source
        import latent_reference
        from checkpointing import atomic_torch_save, capture_rng, restore_rng, to_cpu, optimizer_to_device

        self.p, self.stage, self.shard, self.arm = p, stage, shard, arm
        self.torch, self.np, self.runner, self.source = torch, np, runner, source
        self.sampler, self.latent_reference = sampler, latent_reference
        self.atomic_torch_save, self.capture_rng = atomic_torch_save, capture_rng
        self.restore_rng, self.to_cpu, self.optimizer_to_device = restore_rng, to_cpu, optimizer_to_device
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        self.reset_rng()
        self.root = Path(p['remote_root'])
        self.out = self.root / 'results'
        self.unit = self.out / ('arms/' + arm if stage == 'arm' else f'stages/{stage}_{shard}')
        prepare_output_directories(self.out, self.unit)
        self.job = os.environ['PBS_JOBID']
        self.started = time.monotonic()
        self.deadline = float(os.environ['FW_JOB_START_EPOCH']) + p['allocation']['soft_deadline_seconds']
        self.stopped = []
        signal.signal(signal.SIGTERM, lambda signum, frame: self.stopped.append(signum))
        self.plan = read(p['calibration']['plan_file'])
        self.rows = {r['case_id']: r for r in self.plan['inputs']}
        if (self.plan['calibration_cohorts']['c80'] != CASES
                or Counter(self.rows[c]['suite'] for c in HELDOUT) != Counter({s: 10 for s in p['suites']})):
            raise RuntimeError('Frozen original-LIBERO plan differs')
        self.source_root = Path(p['source_rotation']['root'])
        self.source_lane = self.source_root / 'results' / p['source_rotation']['lane']
        self.source_summary = read(self.source_lane / 'summary.json')
        source_p = read(self.source_root / 'protocol.json')
        r = p['source_rotation']
        record = self.source_summary.get('arms', {}).get(r['arm'], {})
        if (source_p['protocol'] != r['protocol'] or self.source_summary.get('pbs_jobid') != r['job']
                or not self.source_summary.get('complete') or record.get('updates') != 8
                or record.get('training_exposures') != 80 or record.get('heldout_queries') != 80
                or source_p['training']['latent_weight'] != p['training']['latent_weight']
                or source_p['calibration']['c80_case_ids'] != CASES
                or source_p['lanes'][r['lane']]['arms'][0]['learning_rate'] != .0001
                or source_p['quantization']['D'] != 1 or source_p['quantization']['clipping_ratio'] != 1):
            raise RuntimeError('Frozen source rotation receipt differs')
        source_training = read(self.source_lane / 'training.json')['arms'][r['arm']]
        self.schedule = validate_schedule([row['cases'] for row in source_training['history']], self.rows, p['suites'])
        self.teacher_root = Path(p['c80_teacher_directory'])
        self.heldout_teacher_root = Path(p['baseline_root']) / 'results/teacher'
        for path, cases, seeds in ((self.teacher_root, CASES, [0]), (self.heldout_teacher_root, HELDOUT, [0, 1])):
            receipt = read(path / 'receipt.json')
            if not receipt.get('complete') or receipt.get('case_ids') != cases or receipt.get('seed_indices') != seeds:
                raise RuntimeError('Verified BF16 teacher receipt differs: ' + str(path))
        self.runtime = None
        self.bank = None
        self.rb = None
        self.data = {}
        self.progress(stage=stage, status='running', units=0)

    def reset_rng(self):
        import random
        seed = self.p['training']['random_seed']
        random.seed(seed)
        self.np.random.seed(seed)
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)

    def progress(self, **extra):
        value = dict(protocol=PROTOCOL, stage=self.stage, shard=self.shard, arm=self.arm,
                     pbs_jobid=self.job, seconds=time.monotonic() - self.started)
        value.update(extra)
        save(self.unit / 'progress.json', value)
        if self.shard == 0 and self.arm != 'Joint8':
            save(self.out / 'progress.json', value)
        print('C80_DUAL_PROGRESS ' + json.dumps(value, allow_nan=False), flush=True)

    def enough(self):
        if self.stopped or time.time() + 300 >= self.deadline:
            self.progress(status='checkpointed')
            raise TimeoutError('Committed units retained for resume')

    def complete(self, units, **extra):
        value = dict(protocol=PROTOCOL, stage=self.stage, shard=self.shard, arm=self.arm,
                     pbs_jobid=self.job, status='complete', complete=True, units=units,
                     seconds=time.monotonic() - self.started, **extra)
        save(self.unit / 'summary.json', value)
        self.progress(status='complete', units=units)
        return value

    def load_model(self, vq=False):
        from rotation_bank import RotationBanks, VQBanks
        torch = self.torch
        self.runtime = self.runner.Runtime(self.unit)
        self.runtime.model.eval()
        self.runtime.arm('bf16')
        self.rb = RotationBanks(self.runtime, self.p['calibration']['bank_directory'])
        saved = torch.load(self.p['source_rotation']['deployment'], map_location='cpu', weights_only=True)
        r = self.p['source_rotation']
        if (saved.get('protocol', {}).get('protocol') != r['protocol'] or saved.get('lane') != r['lane']
                or saved.get('arm') != r['arm'] or saved.get('updates') != 8
                or saved.get('target_linears') != 614 or set(saved.get('rotations', {})) != {'action', 'video'}):
            raise RuntimeError('Actual deployment rotation binding differs')
        self.initial_rotation = saved['rotations']
        self.rb.rotations.restore(self.initial_rotation)
        if vq:
            self.rb.close()
            self.bank = VQBanks(self.runtime, self.out / 'banks', self.rb.rotations, self.rb.transforms, PROTOCOL)
        else:
            self.bank = self.rb
        self.bank.rotations.freeze()
        if max(self.bank.rotations.orthogonality().values()) > 5e-5:
            raise RuntimeError('R0 orthogonality differs')
        if any(v.requires_grad for v in self.runtime.model.parameters()):
            raise RuntimeError('Original model weights must be frozen')
        self.reset_rng()

    def datum(self, case):
        if case not in self.data:
            observation, meta = self.source.load_observation(
                self.np, Path(self.p['calibration']['plan_file']).parent, self.rows[case])
            self.data[case] = self.runtime.datum(observation), meta['description']
        return self.data[case]

    def teacher(self, case, seed):
        torch = self.torch
        root = self.teacher_root if case in CASES else self.heldout_teacher_root
        value = torch.load(root / f'{case}_{seed}.pt', map_location='cpu', weights_only=True)
        validate_teacher_binding(value, self.rows[case], seed, calibration=case in CASES)
        if (tuple(value['action'].shape) != (32, 7) or tuple(value['future'].shape) != (1, 48, 3, 14, 28)
                or any(not bool(torch.isfinite(value[k]).all()) for k in ('action', 'future'))):
            raise RuntimeError(f'BF16 teacher tensor differs: case={case}, seed_index={seed}')
        return value

    def query(self, case, seed=0, grad=False):
        self.enough()
        torch, bank = self.torch, self.bank
        value, description = self.datum(case)
        self.runtime.description = description
        self.runtime.current_arm = self.arm or ('R0_VQ0' if hasattr(bank, 'books') else 'R0_W16A4')
        bank.seen.clear()
        bank.rotations.refresh()
        with torch.set_grad_enabled(grad):
            for stream in bank.rotations.raw:
                bank.rotations.matrix(stream)
            action, counts, future = self.sampler.differentiable_action(
                self.runtime, value, self.rows[case]['sampler_seeds'][seed], checkpointed=grad, return_future=True)
        if len(bank.seen) != 614:
            raise RuntimeError('Sampler did not reach all 614 target Linears')
        return action, future, counts

    def loss(self, action, future, teacher):
        a = (action[:10, :6].float() - teacher['action'][:10, :6].to('cuda').float()).square().mean()
        z = self.latent_reference.future_nmse_loss(self.torch, future, teacher['future'].to('cuda'))
        return a + self.p['training']['latent_weight'] * z, a, z

    def metrics(self, action, future, teacher):
        total, a, z = self.loss(action, future, teacher)
        delta = action.float() - teacher['action'].to('cuda').float()
        return dict(first10motor_rmse=float(a.detach().sqrt()), first10motor_mse=float(a.detach()),
                    future_latent_nmse=float(z.detach()), mixed_loss=float(total.detach()),
                    full32motor_rmse=float(delta[:, :6].square().mean().sqrt().detach()),
                    late22motor_rmse=float(delta[10:, :6].square().mean().sqrt().detach()),
                    first10gripper_rmse=float(delta[:10, 6].square().mean().sqrt().detach()))

    def endpoint(self, arm, split, case, seed, reused=None, source_binding=None):
        path = self.out / 'endpoints' / arm / split / f'{case}_{seed}.json'
        binding = dict(protocol=PROTOCOL, arm=arm, split=split, case_id=case, seed_index=seed,
                       suite=self.rows[case]['suite'], sampler_seed=self.rows[case]['sampler_seeds'][seed])
        if path.exists():
            record = read(path)
            if any(record.get(k) != v for k, v in binding.items()):
                raise RuntimeError('Saved endpoint binding differs')
        else:
            tick = time.monotonic()
            if reused is None:
                action, future, counts = self.query(case, seed)
                values = self.metrics(action, future, self.teacher(case, seed))
                del action, future
            else:
                values, counts = reused, None
            record = dict(binding, **{k: values[k] for k in METRICS}, counts=counts,
                          reused=reused is not None, source_binding=source_binding,
                          pbs_jobid=self.job, seconds=time.monotonic() - tick)
            save(path, record)
        if any(not math.isfinite(record[k]) or record[k] < 0 for k in METRICS):
            raise RuntimeError('Endpoint metrics invalid')
        return record

    def source_replay(self):
        case = self.p['numerics']['preflight_case']
        source_path = self.source_lane / 'calibration' / self.p['source_rotation']['arm'] / 'queries' / f'{case}_0.json'
        old = read(source_path)
        expected = dict(protocol=self.p['source_rotation']['protocol'], arm=self.p['source_rotation']['arm'],
                        lane=self.p['source_rotation']['lane'], split='calibration', case_id=case, seed_index=0,
                        suite=self.rows[case]['suite'], sampler_seed=self.rows[case]['sampler_seeds'][0])
        if any(old.get(k) != v for k, v in expected.items()):
            raise RuntimeError('Source R0 replay record differs')
        action, future, _ = self.query(case)
        values = self.metrics(action, future, self.teacher(case, 0))
        error = max(abs(values[k] - old[k]) for k in METRICS)
        if error > 2e-6:
            raise RuntimeError('C80 R0 source replay failed: ' + str(error))
        self.atomic_torch_save(self.torch, self.unit / 'source_replay.pt',
                               dict(action=action.cpu(), future=future.cpu(), metrics=values))
        return error

    def moments(self):
        self.load_model()
        from rotation_selfcheck import run as check
        self.enough()
        save(self.unit / 'selfcheck.json', check())
        replay = self.source_replay()
        rb, torch = self.rb, self.torch
        rb.collect = True
        selected = CASES[self.shard::2]
        for index, case in enumerate(selected):
            path = self.out / 'moments' / f'{case}.pt'
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if saved.get('protocol') != PROTOCOL or saved.get('case_id') != case or len(saved['moments']) != 614:
                    raise RuntimeError('Saved moments identity differs')
            else:
                rb.moments.clear()
                rb.moment_rows.clear()
                action, future, counts = self.query(case)
                if len(rb.moments) != 614 or any(v < 1 for v in rb.moment_rows.values()):
                    raise RuntimeError('Missing per-context moments')
                self.atomic_torch_save(torch, path, dict(protocol=PROTOCOL, case_id=case,
                    source_rotation=self.p['source_rotation'],
                    moments={n: (v / rb.moment_rows[n]).cpu() for n, v in rb.moments.items()}, counts=counts))
                del action, future
            self.progress(status='running', units=index + 1, last_case=case)
        rb.collect = False
        rb.close()
        self.complete(40, source_replay_max_error=replay, source_binding=True)

    def fit(self):
        self.load_model()
        torch = self.torch
        import micro_codebook as micro
        self.enough()
        moments = [torch.load(self.out / 'moments' / f'{c}.pt', map_location='cpu', weights_only=True) for c in CASES]
        if any(m.get('protocol') != PROTOCOL or m.get('case_id') != c or len(m['moments']) != 614
               or m.get('source_rotation') != self.p['source_rotation'] for c, m in zip(CASES, moments)):
            raise RuntimeError('Full C80 moment pool differs')
        count = 0
        for index, (name, mod) in enumerate(self.rb.modules.items()):
            if index % 2 != self.shard:
                continue
            self.enough()
            path = self.out / 'banks' / f'{index:04d}.pt'
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if saved.get('protocol') != PROTOCOL or saved.get('module') != name or saved.get('module_index') != index:
                    raise RuntimeError('Fitted bank identity differs')
                micro.check_encoding(saved['encoded'])
            else:
                tick = time.monotonic()
                with torch.no_grad():
                    weight = self.rb.weight(name)
                    moment = torch.stack([m['moments'][name] for m in moments]).mean(0).to('cuda')
                    encoded = micro.fit_vq(weight, second_moment=moment, seed=self.p['fitting']['seed'],
                        sample_limit=self.p['fitting']['samples'], iterations=self.p['fitting']['iterations'])
                    micro.check_encoding(encoded)
                    decoded = micro.decode_vq(encoded).float()
                    nmse = float((decoded - weight).square().sum() / weight.square().sum().clamp_min(1e-12))
                self.atomic_torch_save(torch, path, dict(protocol=PROTOCOL, cohort='c80', module=name,
                    module_index=index, source_rotation=self.p['source_rotation'],
                    encoded=self.to_cpu(encoded), weight_nmse=nmse, encoded_bytes=micro.encoded_nbytes(encoded),
                    seconds=time.monotonic() - tick))
                del weight, moment, encoded, decoded
            count += 1
            self.progress(status='running', units=count, last_module=index)
        self.rb.close()
        self.complete(307, full_context_moments=80)

    def named_trainables(self):
        result = {}
        for name in self.bank.modules:
            result['books/' + name] = self.bank.books[name]
            result['logs/' + name] = self.bank.logs[name]
        result.update({'rotation/' + k: v for k, v in self.bank.rotations.raw.items()})
        return result

    def optimizer(self, joint):
        groups = [dict(params=self.bank.parameters(), lr=self.p['training']['vq_learning_rate'])]
        if joint:
            groups.append(dict(params=list(self.bank.rotations.parameters()), lr=self.p['training']['rotation_learning_rate']))
        return self.torch.optim.Adam(groups)

    def probe(self):
        torch, bank = self.torch, self.bank
        initial = bank.snapshot()
        for value in bank.rotations.parameters():
            value.requires_grad_(True)
        results = []
        for repeat in range(2):
            self.reset_rng()
            bank.restore(initial)
            bank.rotations.restore(self.initial_rotation)
            optimizer = self.optimizer(True)
            optimizer.zero_grad(set_to_none=True)
            action, future, _ = self.query(self.p['numerics']['preflight_case'], grad=True)
            target = self.teacher(self.p['numerics']['preflight_case'], 0)
            loss, _, _ = self.loss(action, future, target)
            loss.backward()
            parameters = self.named_trainables()
            if any(v.grad is None or not bool(torch.isfinite(v.grad).all()) for v in parameters.values()):
                raise RuntimeError('Numerical probe disconnected/nonfinite gradient')
            gradients = {n: v.grad.detach().cpu().clone() for n, v in parameters.items()}
            torch.nn.utils.clip_grad_norm_(bank.parameters(), 1, error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(list(bank.rotations.parameters()), 1, error_if_nonfinite=True)
            before = {n: v.detach().cpu().clone() for n, v in parameters.items()}
            optimizer.step()
            delta = {n: v.detach().cpu() - before[n] for n, v in parameters.items()}
            bank.refresh()
            bank.rotations.refresh()
            post_action, post_future, _ = self.query(self.p['numerics']['preflight_case'])
            results.append(dict(action=action.detach().cpu(), future=future.detach().cpu(),
                post_action=post_action.cpu(), post_future=post_future.cpu(), gradients=gradients, delta=delta))
            del post_action, post_future
            del action, future, target, loss, optimizer, before
        comparison = compare_probe(torch, results[0], results[1], self.p)
        self.atomic_torch_save(torch, self.unit / 'probe.pt', dict(protocol=PROTOCOL, shard=self.shard,
            probe=results[0], repeat_comparison=comparison))
        bank.restore(initial)
        bank.rotations.restore(self.initial_rotation)
        bank.rotations.freeze()
        for value in bank.parameters():
            value.grad = None
        self.reset_rng()
        return comparison

    def common_eval(self):
        self.load_model(vq=True)
        from selfcheck import run as check
        save(self.unit / 'selfcheck.json', check())
        replay = self.probe()
        for value in self.bank.parameters():
            value.requires_grad_(False)
        for index, (split, case, seed) in enumerate(endpoint_units(self.p, self.shard)):
            self.endpoint('R0_VQ0', split, case, seed)
            self.progress(status='running', units=index + 1, last_query=[split, case, seed])
        self.complete(80, calibration=40, heldout=40, repeat_probe=replay)

    def gate(self):
        torch = self.torch
        receipts = {}
        for stage, units in (('moments', 40), ('fit', 307), ('common_eval', 80)):
            for shard in (0, 1):
                summary = read(self.out / f'stages/{stage}_{shard}/summary.json')
                if (summary.get('protocol') != PROTOCOL or not summary.get('complete')
                        or summary.get('units') != units or summary.get('stage') != stage or summary.get('shard') != shard):
                    raise RuntimeError('Common stage is incomplete: ' + stage)
        for folder, expected, names in (('moments', 80, [f'{c}.pt' for c in CASES]),
                                       ('banks', 614, [f'{i:04d}.pt' for i in range(614)])):
            if any(not (self.out / folder / name).is_file() for name in names):
                raise RuntimeError('Shared ' + folder + ' coverage incomplete')
            receipts[folder] = dict(protocol=PROTOCOL, complete=True, passed=True, expected=expected, available=expected)
            save(self.out / folder / 'receipt.json', receipts[folder])
        a, b = [torch.load(self.out / f'stages/moments_{s}/source_replay.pt', map_location='cpu', weights_only=True) for s in (0, 1)]
        source_error = max(float((a[k].float() - b[k].float()).abs().max()) for k in ('action', 'future'))
        if source_error > self.p['numerics']['replay_atol']:
            raise RuntimeError('Two-device source rotation replay differs')
        save(self.out / 'source_receipt.json', dict(protocol=PROTOCOL, complete=True, passed=True,
            source_rotation=self.p['source_rotation'], cross_device_tensor_max_abs=source_error,
            source_metric_max_error=max(read(self.out / f'stages/moments_{s}/summary.json')['source_replay_max_error'] for s in (0, 1))))
        probes = [torch.load(self.out / f'stages/common_eval_{s}/probe.pt', map_location='cpu', weights_only=True) for s in (0, 1)]
        if any(v.get('protocol') != PROTOCOL or v.get('shard') != s for s, v in enumerate(probes)):
            raise RuntimeError('Probe payload binding differs')
        comparison = compare_probe(torch, probes[0]['probe'], probes[1]['probe'], self.p)
        save(self.out / 'repro_receipt.json', dict(protocol=PROTOCOL, complete=True, passed=True,
            within_process=[v['repeat_comparison'] for v in probes], cross_device=comparison,
            deterministic_algorithms=True, cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')))
        self.complete(1, source=True, repro=True)

    def train(self):
        torch, arm = self.torch, self.arm
        for name in ('source_receipt.json', 'repro_receipt.json'):
            receipt = read(self.out / name)
            if receipt.get('protocol') != PROTOCOL or not receipt.get('complete') or not receipt.get('passed'):
                raise RuntimeError('Pre-training gate failed: ' + name)
        self.load_model(vq=True)
        bank = self.bank
        joint = arm == 'Joint8'
        for value in bank.rotations.parameters():
            value.requires_grad_(joint)
        optimizer = self.optimizer(joint)
        state_path = self.unit / 'state.pt'
        state = dict(updates=0, exposures=0, history=[], lineage=[])
        if state_path.exists():
            payload = torch.load(state_path, map_location='cpu', weights_only=True)
            if payload.get('protocol') != self.p or payload.get('arm') != arm:
                raise RuntimeError('Arm resume binding differs')
            state = payload['state']
            bank.restore(payload['module_masters'])
            bank.rotations.restore(payload['rotations'])
            optimizer.load_state_dict(payload['optimizer'])
            self.optimizer_to_device(torch, optimizer, 'cuda')
            self.restore_rng(torch, payload['rng'])
            if state['updates'] != len(state['history']) or state['exposures'] != state['updates'] * 10:
                raise RuntimeError('Arm resume cursor/history differs')
        if self.job not in state['lineage']:
            state['lineage'].append(self.job)

        def persist():
            self.atomic_torch_save(torch, state_path, dict(protocol=self.p, arm=arm, state=state,
                module_masters=bank.snapshot(), rotations=bank.rotations.snapshot(),
                optimizer=self.to_cpu(optimizer.state_dict()), rng=self.capture_rng(torch)))

        if not state['updates']:
            action, future, _ = self.query(CASES[0])
            actual = self.metrics(action, future, self.teacher(CASES[0], 0))
            expected = read(self.out / 'endpoints/R0_VQ0/calibration/2000_0.json')
            initial_error = max(abs(actual[k] - expected[k]) for k in METRICS)
            if initial_error > 1e-6:
                raise RuntimeError('Arm did not start at shared R0+VQ0')
            state['initial_parity_max_abs'] = initial_error
            del action, future
        persist()
        if state['history'] and 'after' not in state['history'][-1]:
            post = []
            for case in state['history'][-1]['cases']:
                action, future, _ = self.query(case)
                post.append(self.metrics(action, future, self.teacher(case, 0)))
                del action, future
            state['history'][-1]['after'] = {k: sum(v[k] for v in post) / 10 for k in METRICS}
            persist()
            save(self.unit / 'training.json', dict(protocol=PROTOCOL, arm=arm, **state))
        while state['updates'] < 8:
            self.enough()
            update = state['updates']
            batch = self.schedule[update]
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            before = []
            for case in batch:
                action, future, counts = self.query(case, grad=True)
                target = self.teacher(case, 0)
                total, a, z = self.loss(action, future, target)
                if not future.requires_grad or not bool(torch.isfinite(total)):
                    raise RuntimeError('Disconnected/nonfinite student future loss')
                before.append(self.metrics(action.detach(), future.detach(), target))
                (total / 10).backward()
                del action, future, target, total, a, z
            coverage = bank.gradient_report()
            if coverage['connected'] != 614 or coverage['finite'] != 614 or coverage['nonzero'] < 1:
                raise RuntimeError('VQ full-model gradient coverage failed')
            rgrad = {s: bool(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0)
                     for s, v in bank.rotations.raw.items() if joint}
            if joint and (set(rgrad) != {'action', 'video'} or not all(rgrad.values())):
                raise RuntimeError('R gradient missing/nonfinite/zero')
            vnorm = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1, error_if_nonfinite=True)
            rnorm = torch.nn.utils.clip_grad_norm_(list(bank.rotations.parameters()), 1, error_if_nonfinite=True) if joint else 0.
            optimizer.step()
            if update == 0:
                for value in bank.parameters():
                    optimizer.state.pop(value, None)
            bank.refresh()
            bank.rotations.refresh()
            state['updates'] += 1
            state['exposures'] += 10
            state['history'].append(dict(update=state['updates'], cases=batch,
                before={k: sum(v[k] for v in before) / 10 for k in METRICS},
                vq_gradient_norm=float(vnorm), rotation_gradient_norm=float(rnorm), coverage=coverage,
                rotation_gradients=rgrad, seconds=time.monotonic() - tick))
            persist()  # A committed update survives interruption during its diagnostic forwards.
            post = []
            for case in batch:
                action, future, _ = self.query(case)
                post.append(self.metrics(action, future, self.teacher(case, 0)))
                del action, future
            state['history'][-1]['after'] = {k: sum(v[k] for v in post) / 10 for k in METRICS}
            persist()
            save(self.unit / 'training.json', dict(protocol=PROTOCOL, arm=arm, **state))
            self.progress(status='running', units=state['updates'], updates=state['updates'], exposures=state['exposures'])
        for value in bank.parameters() + list(bank.rotations.parameters()):
            value.requires_grad_(False)
        bank.rotations.refresh()
        frozen_rotation = all(torch.equal(value.detach().cpu(), self.initial_rotation[s]) for s, value in bank.rotations.raw.items())
        if not joint and not frozen_rotation:
            raise RuntimeError('Seq8 frozen rotation changed')
        orth = bank.rotations.orthogonality()
        if max(orth.values()) > 5e-5:
            raise RuntimeError('Final orthogonality failed')
        self.atomic_torch_save(torch, self.unit / 'deployment.pt', dict(protocol=self.p, arm=arm,
            updates=8, exposures=80, rotations=bank.rotations.snapshot(), module_masters=bank.snapshot(),
            bank_directory=str(self.out / 'banks'), target_linears=614, D=1, r=1))
        action, future, _ = self.query(CASES[0])
        value, _ = self.datum(CASES[0])
        actual, _ = self.runtime.infer(value, self.rows[CASES[0]]['sampler_seeds'][0], measure=False)
        parity = float((action.float().cpu() - actual.float().cpu()).abs().max())
        if parity > 1e-6:
            raise RuntimeError('Final sampler/formal-infer parity failed')
        del action, future, actual
        for index, (split, case, seed) in enumerate(endpoint_units(self.p)):
            self.endpoint(arm, split, case, seed)
            self.progress(status='evaluating', units=index + 1, updates=8, exposures=80,
                          last_query=[split, case, seed])
        self.complete(8, updates=8, exposures=80, calibration=80, heldout=80,
            all_parameters_frozen=all(not v.requires_grad for v in bank.parameters() + list(bank.rotations.parameters())),
            formal_infer_max_abs=parity, orthogonality=orth, frozen_rotation_exact=frozen_rotation,
            initial_parity_max_abs=state['initial_parity_max_abs'], lineage=state['lineage'])

    def finalize(self):
        from aggregate import build_comparison, render_markdown
        self.gate()
        endpoint_counts, training = {}, {}
        for arm in ('Seq8', 'Joint8'):
            summary = read(self.out / f'arms/{arm}/summary.json')
            if (summary.get('protocol') != PROTOCOL or not summary.get('complete') or summary.get('updates') != 8
                    or summary.get('exposures') != 80 or not summary.get('all_parameters_frozen')
                    or summary.get('formal_infer_max_abs', 1) > 1e-6
                    or max(summary.get('orthogonality', {'missing': 1}).values()) > 5e-5
                    or (arm == 'Seq8' and summary.get('frozen_rotation_exact') is not True)):
                raise RuntimeError('Arm not ready for finalization: ' + arm)
            endpoint_counts[arm] = dict(calibration=80, heldout=80, updates=8, exposures=80)
            training[arm] = read(self.out / f'arms/{arm}/training.json')
        for alias, spec in self.p['reference_arms'].items():
            for case in HELDOUT:
                for seed in (0, 1):
                    source = Path(spec['root']) / 'results'
                    if spec.get('lane'):
                        source /= spec['lane']
                    source /= f"{spec['split']}/{spec['arm']}/queries/{case}_{seed}.json"
                    record = read(source)
                    expected = dict(protocol=spec['protocol'], arm=spec['arm'], split=spec['split'],
                        case_id=case, seed_index=seed, suite=self.rows[case]['suite'],
                        sampler_seed=self.rows[case]['sampler_seeds'][seed])
                    if any(record.get(k) != v for k, v in expected.items()):
                        raise RuntimeError('Historical reference binding differs: ' + alias)
                    self.endpoint(alias, 'heldout', case, seed, reused=record,
                        source_binding=dict(expected, root=spec['root'], job=spec['job'],
                            **({'lane': spec['lane']} if spec.get('lane') else {}), source_path=str(source)))
        calibration, heldout = {}, {}
        for arm in ['R0_VQ0', 'Seq8', 'Joint8', 'H_VQ8', 'R0_W16A4']:
            heldout[arm] = [read(self.out / f'endpoints/{arm}/heldout/{c}_{s}.json') for c in HELDOUT for s in (0, 1)]
            if arm in self.p['endpoints']:
                calibration[arm] = [read(self.out / f'endpoints/{arm}/calibration/{c}_0.json') for c in CASES]
        provenance = dict(source_rotation=self.p['source_rotation'], shared_initialization=True,
            warm_start=True, fixed_final_checkpoint=True, new_teacher_queries=0,
            extra_joint_rotation_updates=8, equal_walltime=False, allocation_gpus=2,
            source_receipt=read(self.out / 'source_receipt.json'), repro_receipt=read(self.out / 'repro_receipt.json'),
            training=training, pbs_jobid=self.job)
        comparison = build_comparison(self.p, calibration, heldout, provenance)
        save(self.out / 'comparison.json', comparison)
        (self.out / 'RESULTS.zh.md').write_text(render_markdown(comparison), encoding='utf-8')
        # Keep control reads small; full case-level evidence stays in comparison.json and query records.
        save(self.out / 'training.json', dict(protocol=PROTOCOL, arms=training))
        endpoint_counts['R0_VQ0'] = dict(calibration=80, heldout=80)
        arms = [read(self.out / f'arms/{a}/summary.json') for a in ('Seq8', 'Joint8')]
        summary = dict(protocol=PROTOCOL, status='complete', complete=True, pbs_jobid=self.job,
            target_linears=614, new_teacher_queries=0,
            banks=dict(expected=614, available=614, receipt=True), moments=dict(expected=80, available=80, receipt=True),
            gates=dict(source=True, repro=True, frozen=True,
                formal_parity_max_error=max(a['formal_infer_max_abs'] for a in arms),
                orthogonality_max_error=max(max(a['orthogonality'].values()) for a in arms)),
            endpoints=endpoint_counts, references={a: dict(heldout=80) for a in self.p['reference_arms']},
            comparison_file='results/comparison.json', comparison_complete=comparison.get('complete'),
            allocation_gpus=2, lineage=sorted({j for a in arms for j in a['lineage']}))
        if not summary['comparison_complete']:
            raise RuntimeError('Aggregation is incomplete')
        self.complete(1)
        save(self.out / 'summary.json', summary)
        save(self.out / 'progress.json', summary)


def compare_probe(torch, a, b, p):
    report = {}
    for key in ('action', 'future', 'post_action', 'post_future'):
        error = float((a[key].float() - b[key].float()).abs().max())
        if error > p['numerics']['replay_atol']:
            raise RuntimeError(f'Repeated {key} differs by {error}')
        report[key + '_max_abs'] = error
    for kind, atol, rtol in (('gradients', p['numerics']['gradient_replay_atol'], p['numerics']['gradient_replay_rtol']),
                              ('delta', p['numerics']['parameter_replay_atol'], 0.)):
        if set(a[kind]) != set(b[kind]):
            raise RuntimeError('Repeated parameter coverage differs')
        errors = []
        for name in a[kind]:
            x, y = a[kind][name], b[kind][name]
            if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()) or not torch.allclose(x, y, atol=atol, rtol=rtol):
                raise RuntimeError(f'Repeated {kind} differs: {name}')
            errors.append(float((x - y).abs().max()))
        report[kind + '_max_abs'] = max(errors, default=0.)
        report[kind + '_tensors'] = len(errors)
    report['passed'] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--stage', choices=['moments', 'fit', 'common_eval', 'gate', 'arm', 'finalize'], required=True)
    parser.add_argument('--shard', type=int, choices=[0, 1], default=0)
    parser.add_argument('--arm', choices=['Seq8', 'Joint8'])
    args = parser.parse_args()
    if (args.stage == 'arm') != (args.arm is not None):
        parser.error('--arm is required exactly for the arm stage')
    work = Work(read(args.protocol), args.stage, args.shard, args.arm)
    existing = work.unit / 'summary.json'
    if args.stage not in ('gate', 'finalize') and existing.exists():
        summary = read(existing)
        if summary.get('protocol') == PROTOCOL and summary.get('complete'):
            prior_job = summary.get('pbs_jobid')
            lineage = list(summary.get('lineage', []))
            for job in (prior_job, work.job):
                if job and job not in lineage:
                    lineage.append(job)
            summary.update(pbs_jobid=work.job, lineage=lineage, reused_complete_unit=True,
                           origin_pbs_jobid=summary.get('origin_pbs_jobid', prior_job))
            save(existing, summary)
            work.progress(status='complete', units=summary['units'], reused_complete_unit=True)
            print('REUSED_COMPLETE_UNIT ' + str(work.unit), flush=True)
            return 0
    try:
        getattr(work, 'train' if args.stage == 'arm' else args.stage)()
        return 0
    except TimeoutError as exc:
        print('RESUMABLE: ' + str(exc), flush=True)
        return 75


if __name__ == '__main__':
    raise SystemExit(main())
