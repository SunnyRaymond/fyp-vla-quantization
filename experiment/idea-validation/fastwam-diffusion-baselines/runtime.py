"""Four persistent models share calibration and bank preparation inside one PBS job."""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from diffusion_quant import ARMS, IMPLEMENTATIONS, bank_bytes, select_rank


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def tensor_tree(value, device):
    if isinstance(value, dict):
        return {key: tensor_tree(item, device) for key, item in value.items()}
    if hasattr(value, 'detach'):
        return value.detach().to(device)
    return value


def save_tensors(torch, path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(tensor_tree(value, 'cpu'), temp)
    temp.replace(path)


def wait_for(paths, deadline):
    while not all(Path(path).is_file() for path in paths):
        if time.monotonic() >= deadline:
            raise RuntimeError('Shared preparation barrier exceeded two hours')
        time.sleep(2)


def _runtime_type():
    import runner as pilot
    import torch
    import diffusion_quant as quant

    spec = importlib.util.spec_from_file_location(
        'diffusion_previous_adapter', Path(__file__).with_name('previous_vq_adapter.py'))
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    base = previous._runtime_type()

    class Runtime(base):
        def __init__(self, out, protocol, worker_id):
            pilot.guard()
            torch.backends.cuda.matmul.allow_tf32 = False
            pilot.Runtime.__init__(self, out)
            self.model.eval()
            self.arm('bf16')
            self.protocol, self.worker_id = protocol, worker_id
            self.arm_name = protocol['worker_assignment']['mapping'][str(worker_id)]['arm']
            self.transform_implementation = IMPLEMENTATIONS[self.arm_name]
            self.run_queries = self.linear_calls = self.low_rank_calls = self.rotation_calls = 0
            self.seen, self.forwards = set(), {}
            self.banks = SimpleNamespace(seen=self.seen)
            self.deadline = time.monotonic() + 7200
            if len(self.q.modules) != 614:
                raise RuntimeError('Expected the exact614 canonical target Linears')
            self.names = list(self.q.modules)
            self.shapes = [tuple(module.weight.shape) for module in self.q.modules.values()]
            self.budget = select_rank(self.shapes, protocol)
            self.original_parameter_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
            self.original_parameters = sum(p.numel() for p in self.model.parameters())
            self.target_parameters = sum(m * n for m, n in self.shapes)

        def stage(self, name, **extra):
            save_json(Path(self.out) / 'preparation_progress.json', {
                'protocol': self.protocol['protocol'], 'worker_id': self.worker_id,
                'arm': self.arm_name, 'stage': name, 'pbs_job_id': os.environ['PBS_JOBID'], **extra})
            print('PREPARATION ' + json.dumps({'stage': name, **extra}), flush=True)

        def observation(self, row):
            import numpy as np
            import calibration_source
            path = Path(self.protocol['sources']['calibration_plan'])
            observation, metadata = calibration_source.load_observation(np, path.parent, row)
            self.description = metadata['description']
            return self.datum(observation)

        def original_query(self, row):
            action, _ = pilot.Runtime.infer(self, self.observation(row), row['sampler_seeds'][0], measure=False)
            return action.detach().float().cpu()

        def calibration(self, results, rows):
            directory = results / 'calibration'
            receipt_path = directory / f'worker_{self.worker_id}.json'
            stats_path = directory / f'worker_{self.worker_id}.pt'
            assigned = self.protocol['calibration']['case_ids'][self.worker_id::4]
            if receipt_path.is_file():
                receipt = read_json(receipt_path)
                if (receipt.get('protocol') != self.protocol['protocol']
                        or receipt.get('case_ids') != assigned or receipt.get('modules') != 614
                        or receipt.get('complete') is not True or not stats_path.is_file()):
                    raise RuntimeError('Saved calibration shard differs from the frozen C80 protocol')
            else:
                self.stage('calibration', cases=assigned)
                maxima, hooks = {}, []
                for name, module in self.q.modules.items():
                    def capture(_module, args, _name=name):
                        value = args[0].detach().float()
                        maximum = value.reshape(-1, value.shape[-1]).abs().amax(0)
                        maxima[_name] = torch.maximum(maxima[_name], maximum) if _name in maxima else maximum
                    hooks.append(module.register_forward_pre_hook(capture))
                try:
                    for ordinal, case in enumerate(assigned):
                        self.original_query(rows[case])
                        self.stage('calibration', case_id=case, completed_contexts=ordinal + 1, expected=20)
                finally:
                    for hook in hooks:
                        hook.remove()
                if set(maxima) != set(self.names) or any(not bool(torch.isfinite(v).all()) for v in maxima.values()):
                    raise RuntimeError('Calibration does not cover614 finite raw activation maxima')
                save_tensors(torch, stats_path, {'protocol': self.protocol['protocol'], 'case_ids': assigned,
                                               'maxima': maxima})
                save_json(receipt_path, {'protocol': self.protocol['protocol'], 'complete': True,
                    'worker_id': self.worker_id, 'case_ids': assigned, 'modules': 614,
                    'pbs_job_id': os.environ['PBS_JOBID']})
            receipt_paths = [directory / f'worker_{worker}.json' for worker in range(4)]
            wait_for(receipt_paths, self.deadline)
            merged = {}
            for worker, path in enumerate(receipt_paths):
                receipt = read_json(path)
                expected = self.protocol['calibration']['case_ids'][worker::4]
                if (receipt.get('protocol') != self.protocol['protocol'] or receipt.get('case_ids') != expected
                        or receipt.get('worker_id') != worker or receipt.get('complete') is not True
                        or receipt.get('modules') != 614):
                    raise RuntimeError('Shared calibration receipt differs')
                data = torch.load(directory / f'worker_{worker}.pt', map_location='cpu', weights_only=True)
                if (data.get('protocol') != self.protocol['protocol'] or data.get('case_ids') != expected
                        or set(data.get('maxima', {})) != set(self.names)):
                    raise RuntimeError('Shared calibration tensor shard differs')
                for name, maximum in data['maxima'].items():
                    if maximum.shape != (self.q.modules[name].weight.shape[1],) or not bool(torch.isfinite(maximum).all()):
                        raise RuntimeError('Invalid shared channel maximum')
                    merged[name] = torch.maximum(merged[name], maximum) if name in merged else maximum
            return merged

        def fit(self, results, maxima):
            directory = results / 'banks'
            receipt_path = directory / f'fit_worker_{self.worker_id}.json'
            indices = list(range(self.worker_id, 614, 4))
            rank = self.budget['global_rank']
            self.stage('fitting', modules=len(indices), global_rank=rank,
                       target_bpw={arm: self.budget['costs'][arm]['bpw'] for arm in ARMS})
            if not receipt_path.is_file():
                totals = {arm: {'count': 0, 'parameters': 0, 'bytes_by_kind': Counter(),
                                'rotation_layouts': Counter(), 'rank_histogram': Counter()} for arm in ARMS}
                for ordinal, index in enumerate(indices):
                    name, module = self.names[index], self.q.modules[self.names[index]]
                    for arm in ARMS:
                        path = directory / arm / f'{index:04d}.pt'
                        if path.is_file():
                            bank = torch.load(path, map_location='cpu', weights_only=True)
                            self.validate_bank(bank, index, arm)
                        else:
                            with torch.no_grad():
                                bank = quant.fit_bank(module.weight, maxima[name], arm, self.protocol, rank, index)
                            bank.update(protocol=self.protocol['protocol'], module=name, module_index=index,
                                        global_rank=rank, pbs_job_id=os.environ['PBS_JOBID'])
                            save_tensors(torch, path, bank)
                        parts = bank_bytes(bank)
                        info = totals[arm]
                        info['count'] += 1
                        info['parameters'] += module.weight.numel()
                        info['bytes_by_kind'].update(parts)
                        info['rank_histogram'][str(bank['rank'])] += 1
                        if arm == ARMS[1]:
                            info['rotation_layouts'][bank['rotation_layout']] += 1
                        del bank
                    if ordinal % 10 == 0 or ordinal + 1 == len(indices):
                        self.stage('fitting', completed_modules=ordinal + 1, expected=len(indices))
                save_json(receipt_path, {'protocol': self.protocol['protocol'], 'worker_id': self.worker_id,
                    'complete': True, 'module_indices': indices, 'global_rank': rank,
                    'arms': totals, 'pbs_job_id': os.environ['PBS_JOBID']})
            wait_for([directory / f'fit_worker_{i}.json' for i in range(4)], self.deadline)
            receipts = [read_json(directory / f'fit_worker_{i}.json') for i in range(4)]
            for worker, receipt in enumerate(receipts):
                if (receipt.get('protocol') != self.protocol['protocol'] or receipt.get('worker_id') != worker
                        or receipt.get('global_rank') != rank or receipt.get('complete') is not True
                        or receipt.get('module_indices') != list(range(worker, 614, 4))):
                    raise RuntimeError('Shared weight preparation receipt differs')
            costs = {}
            for arm in ARMS:
                parts, ranks, rotations = Counter(), Counter(), Counter()
                for receipt in receipts:
                    info = receipt['arms'][arm]
                    parts.update(info['bytes_by_kind'])
                    ranks.update(info['rank_histogram'])
                    rotations.update(info['rotation_layouts'])
                if (dict(parts) != self.budget['costs'][arm]['bytes_by_kind']
                        or sum(r['arms'][arm]['count'] for r in receipts) != 614
                        or sum(r['arms'][arm]['parameters'] for r in receipts) != self.target_parameters):
                    raise RuntimeError('Actual encoded bank sizes differ from the predeclared shape budget')
                costs[arm] = {**self.budget['costs'][arm], 'rank_histogram': dict(ranks),
                              'rotation_layouts': dict(rotations),
                              'whole_model_parameter_bpw': 8 * (self.original_parameter_bytes
                                   - 2 * self.target_parameters + sum(parts.values())) / self.original_parameters}
            manifest_path = directory / 'manifest.json'
            if not manifest_path.is_file() and self.worker_id == 0:
                save_json(manifest_path, {'protocol': self.protocol['protocol'], 'complete': True,
                    'pbs_job_id': os.environ['PBS_JOBID'], 'modules': 614, 'names': self.names,
                    'shapes': self.shapes, 'calibration_contexts': 80, 'optimizer_updates': 0,
                    'global_rank': rank, 'costs': costs,
                    'relative_bpw_gap': self.budget['relative_bpw_gap'],
                    'original_parameters': self.original_parameters,
                    'original_parameter_bytes': self.original_parameter_bytes})
            wait_for([manifest_path], self.deadline)
            manifest = read_json(manifest_path)
            if (manifest.get('protocol') != self.protocol['protocol'] or manifest.get('complete') is not True
                    or manifest.get('names') != self.names or manifest.get('global_rank') != rank
                    or manifest.get('costs') != costs or manifest.get('calibration_contexts') != 80):
                raise RuntimeError('Final bank manifest differs from the actual calibrated representation')
            return manifest

        def validate_bank(self, bank, index, arm):
            name, shape = self.names[index], self.shapes[index]
            encoded = bank.get('encoded', {})
            if (bank.get('protocol') != self.protocol['protocol'] or bank.get('module') != name
                    or bank.get('module_index') != index or bank.get('arm') != arm
                    or bank.get('global_rank') != self.budget['global_rank']
                    or bank.get('implementation') != IMPLEMENTATIONS[arm]
                    or tuple(encoded.get('shape', ())) != shape
                    or encoded.get('group_size') != 128 or encoded.get('affine') is not (arm == ARMS[1])
                    or encoded.get('packed').dtype != torch.uint8
                    or encoded.get('scales').dtype != torch.float32
                    or tuple(bank.get('smoothing').shape) != (shape[1],)
                    or not bool(torch.isfinite(bank['smoothing']).all())
                    or not bool((bank['smoothing'] > 0).all())):
                raise RuntimeError(f'Bank identity or storage layout differs: {arm}/{index}')
            if arm == ARMS[0]:
                rank = min(self.budget['global_rank'], *shape)
                if (bank.get('rank') != rank or tuple(bank['up'].shape) != (shape[0], rank)
                        or tuple(bank['down'].shape) != (rank, shape[1])
                        or bank['up'].dtype != torch.bfloat16 or bank['down'].dtype != torch.bfloat16):
                    raise RuntimeError('High-precision low-rank branch differs')
            elif (tuple(bank['signs'].shape) != (shape[1],) or bank['signs'].dtype != torch.int8
                    or encoded['zero_points'].dtype != torch.uint8
                    or bank.get('rotation_layout') != quant.rotation_layout(shape[1])):
                raise RuntimeError('ViDiT fixed rotation or affine zero point differs')

        def install(self, results, manifest):
            arm = self.arm_name
            self.stage('installing', arm=arm)
            for index, (name, module) in enumerate(self.q.modules.items()):
                bank = torch.load(results / 'banks' / arm / f'{index:04d}.pt', map_location='cpu', weights_only=True)
                self.validate_bank(bank, index, arm)
                if bank_bytes(bank) != quant.estimate_storage([self.shapes[index]],
                        self.budget['global_rank']) [arm]['bytes_by_kind']:
                    raise RuntimeError('Loaded bank byte accounting differs')
                bank = tensor_tree(bank, 'cuda')
                with torch.no_grad():
                    module.weight.data = quant.decode_weight(bank['encoded'])
                del bank['encoded']
                self.forwards[name] = module.forward
                def forward(value, _name=name, _module=module, _bank=bank):
                    self.seen.add(_name)
                    self.linear_calls += 1
                    if _bank['arm'] == ARMS[0]:
                        self.low_rank_calls += 1
                    else:
                        self.rotation_calls += 1
                    return quant.forward(value, _module.weight, _module.bias, _bank)
                module.forward = forward
            cost = manifest['costs'][arm]
            self.install_metadata = {
                'protocol': self.protocol['protocol'], 'arm': arm, 'weight_bits': 4, 'activation_bits': 4,
                'group_size': 128, 'execution': 'decoded_bf16_reference', 'kv_cache_dtype': 'bf16',
                'native_low_bit_kernel': False, 'module_count': 614, 'calibration_contexts': 80,
                'optimizer_updates': 0, 'encoded_target_bpw': cost['bpw'],
                'encoded_target_bytes': cost['encoded_bytes'], 'target_weight_parameters': cost['parameters'],
                'low_rank_global_rank': self.budget['global_rank'] if arm == ARMS[0] else 0,
                'representation': 'svdquant_low_rank_residual' if arm == ARMS[0] else 'viditq_channel_balance_rotation',
                'transform_implementation': IMPLEMENTATIONS[arm], 'bank_manifest_pbs_jobid': manifest['pbs_job_id'],
                'encoded_bytes_by_kind': cost['bytes_by_kind'],
                'whole_model_parameter_bpw': cost['whole_model_parameter_bpw']}
            self.current_arm = arm
            torch.cuda.empty_cache()

        def prepare(self, results_root):
            results = Path(results_root)
            self.stage('component_selfcheck')
            component = quant.self_check()
            plan = read_json(self.protocol['sources']['calibration_plan'])
            if (plan.get('status') != 'complete'
                    or plan.get('calibration_cohorts', {}).get('c80') != self.protocol['calibration']['case_ids']):
                raise RuntimeError('C80 original-LIBERO plan differs')
            rows = {int(row['case_id']): row for row in plan['inputs']}
            row = rows[2000]
            if any(rows[case].get('domain') != 'original' for case in self.protocol['calibration']['case_ids']):
                raise RuntimeError('Calibration contains non-original observations')
            self.stage('bf16_teacher_reference')
            expected = torch.load(Path(self.protocol['sources']['bf16_teacher_directory']) / '2000_0.pt',
                                  map_location='cpu', weights_only=True)
            if expected.get('plan_row') != row or expected.get('sampler_seed') != row['sampler_seeds'][0]:
                raise RuntimeError('BF16 anchor is differently bound')
            bf16_action = self.original_query(row)
            bf16_error = float((bf16_action - expected['action'].float()).abs().max())
            if bf16_error > self.protocol['startup_checks']['bf16_teacher_max_abs_tolerance']:
                raise RuntimeError(f'BF16 original-input anchor differs: {bf16_error}')
            maxima = self.calibration(results, rows)
            manifest = self.fit(results, maxima)
            del maxima
            self.install(results, manifest)
            self.stage('cross_worker_action_reference')
            action, _ = self.infer(self.observation(row), row['sampler_seeds'][0], measure=False)
            reference_path = results / 'reference' / f'{self.arm_name}.json'
            if self.worker_id in (0, 2) and not reference_path.is_file():
                save_json(reference_path, {'protocol': self.protocol['protocol'], 'arm': self.arm_name,
                    'case_id': 2000, 'sampler_seed': row['sampler_seeds'][0],
                    'bank_manifest_pbs_jobid': manifest['pbs_job_id'],
                    'action': action.detach().float().cpu().tolist()})
            wait_for([reference_path], self.deadline)
            saved = read_json(reference_path)
            if (saved.get('protocol') != self.protocol['protocol'] or saved.get('arm') != self.arm_name
                    or saved.get('case_id') != 2000 or saved.get('sampler_seed') != row['sampler_seeds'][0]
                    or saved.get('bank_manifest_pbs_jobid') != manifest['pbs_job_id']):
                raise RuntimeError('Cross-worker quantized action reference differs')
            delta = action.detach().float().cpu() - torch.tensor(saved['action'], dtype=torch.float32)
            error = float(delta.abs().max())
            if (not math.isfinite(error) or error > 1e-6 or len(self.seen) != 614
                    or (self.arm_name == ARMS[0] and self.low_rank_calls <= 0)
                    or (self.arm_name == ARMS[1] and self.rotation_calls <= 0)):
                raise RuntimeError('Prepared method full-query coverage/parity gate failed')
            self.run_queries = 0
            self.stage('ready', observed_modules=614, cross_worker_action_max_abs=error)
            return {'protocol': self.protocol['protocol'], 'worker_id': self.worker_id, 'arm': self.arm_name,
                'pbs_job_id': os.environ['PBS_JOBID'], 'passed': True, 'observed_modules': 614,
                'bf16_teacher_max_abs': bf16_error, 'cross_worker_action_max_abs': error,
                'component_selfcheck': component, 'bank_manifest': {key: manifest[key] for key in
                    ('protocol', 'complete', 'pbs_job_id', 'modules', 'calibration_contexts', 'global_rank', 'costs', 'relative_bpw_gap')},
                'install_metadata': self.install_metadata}

        def episode(self, row, index):
            low, rotations = self.low_rank_calls, self.rotation_calls
            record = base.episode(self, row, index)
            evidence = record.pop('vq_evidence')
            evidence.update(low_rank_calls=self.low_rank_calls - low,
                            rotation_calls=self.rotation_calls - rotations)
            record['quantization_evidence'] = evidence
            return record

    return Runtime


class DiffusionRuntime:
    def __new__(cls, *args, **kwargs):
        return _runtime_type()(*args, **kwargs)
