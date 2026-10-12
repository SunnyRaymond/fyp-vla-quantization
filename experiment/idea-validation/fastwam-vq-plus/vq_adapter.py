"""Reuse frozen additive VQ banks and the original LIBERO episode policy."""
from __future__ import annotations

import json
from pathlib import Path
import time

import runner as pilot

ARMS = ('row_scale_vq_a4', 'micro_scale_vq_a4')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class VQRuntime(pilot.Runtime):
    def __init__(self, out, protocol, arm):
        pilot.guard()
        import torch
        import smooth_vq as core
        import codebook
        import micro_codebook
        if arm not in ARMS:
            raise ValueError('Unexpected VQ arm')
        torch.backends.cuda.matmul.allow_tf32 = False
        super().__init__(out)
        self.arm('bf16')
        self.current_arm = arm
        self.protocol, self.arm_name = protocol, arm
        self.seen, self.linear_calls, self.transforms, self.forwards = set(), 0, {}, {}
        source = protocol['sources'][arm]
        source_root, bank_root = Path(source['root']), Path(source['bank'])
        summary = read_json(source_root / 'results/full/summary.json')
        job = summary.get('pbs_jobid', summary.get('job_id'))
        if summary.get('status') != 'complete' or job != source['job_id']:
            raise RuntimeError('VQ source is not the frozen completed job')
        modules = self.q.modules
        if len(modules) != 614 or 'A100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Expected 614 target Linears and the existing A100 hardware class')
        decoder = codebook if arm == ARMS[0] else micro_codebook
        old_bank = Path(protocol['roots']['old_banks']) / 'results/full/banks/smooth05_hadamard_scalar'
        encoded_bytes, transform_bytes = 0, 0
        with torch.no_grad():
            for index, (name, module) in enumerate(modules.items()):
                saved = torch.load(bank_root / f'{index:04d}.pt', map_location='cuda', weights_only=True)
                if saved.get('module') != name or tuple(saved['encoded']['shape']) != tuple(module.weight.shape):
                    raise RuntimeError(f'Frozen VQ bank identity/shape differs: {name}')
                dense = decoder.decode_vq(saved['encoded'])
                if dense.dtype != torch.bfloat16 or not bool(torch.isfinite(dense).all()):
                    raise RuntimeError(f'Invalid decoded VQ weight: {name}')
                encoded_bytes += decoder.encoded_nbytes(saved['encoded'])
                module.weight.data = dense
                del saved, dense
                original = torch.load(old_bank / f'{index:04d}.pt', map_location='cpu', weights_only=True)
                if original.get('module') != name:
                    raise RuntimeError(f'Frozen Smooth/Hadamard transform identity differs: {name}')
                transform = {key: value.to('cuda') if torch.is_tensor(value) else value
                             for key, value in original['transform'].items()}
                transform_bytes += sum(value.numel() * value.element_size() for value in transform.values()
                                       if torch.is_tensor(value))
                self.transforms[name], self.forwards[name] = transform, module.forward

                def forward(value, _name=name, _module=module, _transform=transform):
                    self.seen.add(_name)
                    self.linear_calls += 1
                    z = core.transform_input(value, _transform).float()
                    scale = z.abs().amax(dim=-1, keepdim=True) / 7
                    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
                    operand = (torch.round(z / safe).clamp(-7, 7) * scale).to(torch.bfloat16)
                    return torch.nn.functional.linear(operand, _module.weight, _module.bias)

                module.forward = forward
                if index % 100 == 0:
                    print(f'VQ_BANK_LOADED arm={arm} modules={index}/614', flush=True)
                del original
        torch.cuda.empty_cache()
        self.install_metadata = {
            'arm': arm, 'source_bank': str(bank_root), 'source_job_id': source['job_id'],
            'module_count': len(modules), 'execution': 'decoded_bf16_vq',
            'activation_bits': 4, 'kv_cache_dtype': 'bf16', 'native_low_bit_kernel': False,
            'smooth_alpha': 0.5, 'hadamard_block': 128, 'transform_seed': 20261006,
            'encoded_bank_bytes': encoded_bytes, 'transform_tensor_bytes': transform_bytes,
            'frozen_bank_reused': True, 'new_training_steps': 0, 'new_calibration_cases': 0,
            'decoded_target_weight_dtype': 'bf16', 'indices_only_bits_per_weight': 4,
            'scale_block': 32 if arm == ARMS[1] else None,
        }
        pilot.save(out / 'runtime_vq.json', {
            'snapshot_phase': 'frozen_vq_bank_installed_before_source_replay',
            'install_metadata': self.install_metadata,
            'resident_weight_precision': 'decoded_bf16', 'kv_cache_dtype': 'bf16',
        })

    def infer(self, datum, seed, measure=True):
        self.seen.clear()
        result = super().infer(datum, seed, measure=measure)
        if len(self.seen) != 614:
            raise RuntimeError(f'Only {len(self.seen)}/614 VQ targets called in this query')
        if self.q._native_int4_gemm_calls != 0 or self.q._kv_active:
            raise RuntimeError('Frozen VQ reference unexpectedly enabled native INT4 or KV4')
        return result

    def validate_reference(self):
        import torch
        import run_diagnostic as source
        p, arm = self.protocol, self.arm_name
        row_root = Path(p['sources'][ARMS[0]]['root'])
        case, seed_index = p.get('source_parity_case_id', 1000), p.get('source_parity_seed_index', 0)
        plan = {row['case_id']: row for row in read_json(row_root / 'plan.json')['inputs']}
        row = plan[case]
        observation, metadata = source.load_case(row_root, row)
        if arm == ARMS[1]:
            record = read_json(Path(p['sources'][arm]['root']) / f'results/full/query_records/selection_{case}_{seed_index}.json')
        else:
            record = None
            with (row_root / 'results/full/actions.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    candidate = json.loads(line)
                    if (candidate['arm'], candidate['case_id'], candidate['seed_index']) == (arm, case, seed_index):
                        record = candidate
                        break
            if record is None:
                raise RuntimeError('Missing frozen row-scale action reference')
        sampler_seed = row['sampler_seeds'][seed_index]
        if record['sampler_seed'] != sampler_seed or record['arm'] != arm:
            raise RuntimeError('VQ parity reference has a different arm/seed')
        self.description = metadata['description']
        action, _ = self.infer(self.datum(observation), sampler_seed, measure=False)
        reference = torch.tensor(record['action'], dtype=torch.float32)
        max_abs = float((action.detach().float().cpu() - reference).abs().max())
        tolerance = p.get('source_parity_max_abs_tolerance', 1e-6)
        receipt = {'arm': arm, 'case_id': case, 'seed_index': seed_index,
                   'sampler_seed': sampler_seed, 'max_abs_vs_frozen_action': max_abs,
                   'tolerance': tolerance, 'passed': max_abs <= tolerance,
                   'observed_modules': len(self.seen), 'install_metadata': self.install_metadata}
        if not receipt['passed']:
            raise RuntimeError(f'Frozen VQ action replay differs: arm={arm} max_abs={max_abs}')
        return receipt

    def episode(self, row, index):
        # Keep the exact original episode loop; replace its native-INT4-only evidence gate.
        started = time.monotonic()
        calls_before = self.linear_calls
        obs = self.task(row, index)
        cap = 700 if row['suite'] == 'libero_10' else 400
        steps, replans, success = 0, 0, False
        metrics, action_sample = [], []
        while steps < cap and not success:
            action, metric = self.infer(self.datum(obs), pilot.seed_for(index, replans))
            metric['replan'] = replans
            metrics.append(metric)
            if replans < 3:
                action_sample.append(action.float().cpu().tolist())
            commands = self.command(action)
            for command in commands[:min(10, cap - steps)]:
                obs, _, done, _ = self.env.step(command.tolist())
                steps += 1
                if done:
                    success = True
                    break
            replans += 1
            if replans % 5 == 0:
                print(f'REPLAN variant={row["variant_id"]} arm={self.current_arm} index={replans} steps={steps}', flush=True)
        self.env.close()
        self.env = None
        evidence = {**self.install_metadata, 'observed_module_count': len(self.seen),
                    'quantized_linear_calls': self.linear_calls - calls_before,
                    'native_int4_gemm_calls': self.q._native_int4_gemm_calls,
                    'kv_packed_prefills': self.q._kv_packed_prefills, 'kv_layer_reads': self.q._kv_layer_reads}
        if evidence['quantized_linear_calls'] <= 0:
            raise RuntimeError('Episode did not execute frozen VQ/A4 Linears')
        return {
            'protocol': self.protocol['protocol'], 'variant_id': str(row['variant_id']), 'index': index,
            'suite': row['suite'], 'dimension': row['dimension'], 'task_id': int(row['task_id']),
            'original_task': row.get('original_task'), 'subtype': row.get('subtype'),
            'arm': self.current_arm, 'complete': True, 'success': success,
            'termination': 'success' if success else 'control_step_timeout', 'steps': steps,
            'replans': replans, 'episode_seconds': time.monotonic() - started,
            'env_seed': pilot.seed_for(index, environment=True), 'sampler_seed': pilot.seed_for(index),
            'state_id': 0, 'initial_state_path': self.init_states_path,
            'description': self.description, 'metrics': metrics, 'first_action_chunks': action_sample,
            'vq_evidence': evidence,
        }

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        for name, forward in self.forwards.items():
            self.q.modules[name].forward = forward
        self.q.close()
