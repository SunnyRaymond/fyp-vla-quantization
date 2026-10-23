"""Train or export the frozen C80 clip/joint calibration arms on a PBS GPU."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time


HERE = Path(__file__).resolve().parent
TARGET_LINEARS = 614
UPDATES = 80


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def validate_protocol(p):
    source = p['source_c80_protocol']
    calibration = p['calibration']
    source_calibration = source['calibration']
    training = p['training']
    if p.get('protocol') != 'fastwam-vq-c80-clip-joint-plus-v1':
        raise ValueError('Unexpected target protocol')
    if (source.get('protocol') != 'fastwam-vq-c80-activation-clipping-v2-original-heldout'
            or source.get('remote_root') != p['sources']['c80_root']):
        raise ValueError('Frozen C80 source protocol binding differs')
    if (calibration['case_ids'] != source_calibration['case_ids']
            or calibration['update_order'] != source_calibration['update_order']
            or calibration['seed_index'] != source_calibration['seed_index']
            or calibration['plan_file'] != source_calibration['plan_file']
            or calibration['bank_directory'] != source_calibration['bank_directory']
            or calibration['source_job_id'] != source_calibration['source_job_id']):
        raise ValueError('C80 calibration binding differs from the source protocol')
    if (len(calibration['case_ids']) != UPDATES
            or len(set(calibration['case_ids'])) != UPDATES
            or len(calibration['update_order']) != UPDATES
            or set(calibration['update_order']) != set(calibration['case_ids'])
            or calibration['seed_index'] != 0):
        raise ValueError('Expected the exact frozen 80-context C80 cohort')
    expected_training = {
        'additional_updates': 80, 'epochs': 1, 'weight_learning_rate': 0.0005,
        'clip_learning_rate': 0.01, 'optimizer': 'Adam',
        'fresh_optimizer_each_arm': True, 'optimizer_reset_after_update1': True,
        'gradient_norm_clip': 1.0, 'rho': 1.5,
        'normalization_c': 0.028303017887797623,
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError('Frozen C80 training settings differ')
    for arm, weights in (('clip_only', False), ('joint', True)):
        config = p['arm_configs'][arm]
        if (config.get('updates') != UPDATES or config.get('epochs') != 1
                or config.get('rho') != 1.5 or config.get('train_weights') is not weights
                or config.get('train_clips') is not True):
            raise ValueError(f'Frozen {arm} parameter contract differs')
    if (p['sources']['teacher_directory'] != source['remote_root'].rstrip('/') + '/results/teacher/calibration'
            or p['sources']['plan_file'] != calibration['plan_file']
            or p['sources']['clip_root'] != source['clipping_root']
            or p['sources']['baseline_root'] != source['baseline_root']
            or source['source_clipping_protocol']['remote_root'] != source['clipping_root']):
        raise ValueError('C80 teacher, plan, clipping, or baseline source path differs')
    if training != source['training'] or p['clipping'] != source['clipping']:
        raise ValueError('Training differs from the source C80 loss contract')
    backward = p.get('quantization', {}).get('weight_backward', {})
    if (backward.get('implementation') != 'bounded_vector_chunks_v2'
            or backward.get('max_vectors_per_chunk') != 1048576):
        raise ValueError('Frozen bounded-vector backward implementation differs')


def validate_deployment(deployment, p, arm):
    expected = {
        'schema_version': 1, 'protocol': p['protocol'], 'arm': arm,
        'updates': UPDATES, 'target_linears': TARGET_LINEARS, 'epochs': 1,
        'rho': p['training']['rho'], 'normalization_c': p['training']['normalization_c'],
        'case_ids': p['calibration']['case_ids'],
        'update_order': p['calibration']['update_order'],
    }
    if any(deployment.get(key) != value for key, value in expected.items()):
        raise ValueError('Deployment metadata differs from the frozen protocol')
    masters, ratios = deployment.get('module_masters'), deployment.get('ratios')
    if not isinstance(masters, dict) or len(masters) != TARGET_LINEARS:
        raise ValueError('Deployment must contain 614 FP32 module masters')
    if not isinstance(ratios, dict) or set(ratios) != set(masters):
        raise ValueError('Deployment clipping-ratio coverage differs from its masters')
    if any(set(value) != {'books', 'log_scales'} for value in masters.values()):
        raise ValueError('Each deployment master must contain books and log_scales')


def validate_receipt(receipt, p, arm):
    expected = {
        'schema_version': 1, 'protocol': p['protocol'], 'arm': arm,
        'status': 'complete', 'complete': True, 'updates': UPDATES,
        'epochs': 1, 'target_linears': TARGET_LINEARS,
        'calibration_ids': p['calibration']['case_ids'],
        'update_order': p['calibration']['update_order'],
        'normalization_c': p['training']['normalization_c'], 'rho': p['training']['rho'],
        'weights_unchanged': arm == 'clip_only',
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Training receipt differs from the frozen protocol')
    if not receipt.get('history_source') or len(receipt.get('history', [])) != UPDATES:
        raise ValueError('Training receipt must bind the exact 80-update history source')
    history = receipt['history']
    if ([item.get('case_id') for item in history] != p['calibration']['update_order']
            or [item.get('update') for item in history] != list(range(1, UPDATES + 1))
            or receipt.get('source') != source_bindings(p)):
        raise ValueError('Training receipt history or parent source binding differs')
    parity = receipt.get('reference_action_parity', {})
    max_abs = parity.get('max_abs')
    if (parity != receipt.get('actual_infer_parity')
            or parity.get('passed') is not True
            or isinstance(max_abs, bool) or not isinstance(max_abs, (int, float))
            or not math.isfinite(max_abs)
            or max_abs > p['closed_loop_reference']['max_abs_tolerance']):
        raise ValueError('Training receipt lacks passing sampler/pilot reference parity')
    validate_backward_receipt(receipt.get('backward_check', {}),
                              receipt.get('backward_check', {}).get('implementation'),
                              receipt.get('backward_check', {}).get('max_vectors_per_chunk'))


def validate_backward_receipt(receipt, implementation, max_vectors):
    if (implementation != 'bounded_vector_chunks_v2' or max_vectors != 1048576
            or receipt.get('passed') is not True
            or receipt.get('implementation') != implementation
            or receipt.get('max_vectors_per_chunk') != max_vectors):
        raise RuntimeError('Bounded DecodedWeight backward parity/performance receipt is missing or differs')


def guard():
    jobid = os.environ.get('PBS_JOBID', '')
    nodefile = os.environ.get('PBS_NODEFILE', '')
    host = socket.gethostname().split('.')[0]
    if (not jobid or not nodefile or 'login' in host.lower()
            or not Path(nodefile).is_file()):
        raise RuntimeError('A real PBS compute allocation is required')
    nodes = {line.split('.')[0] for line in Path(nodefile).read_text().split()}
    if host not in nodes:
        raise RuntimeError('Current host is outside the PBS allocation')


def initial_summary(p, arm, jobid, resume_from=None):
    return {
        'schema_version': 1, 'protocol': p['protocol'], 'jobid': jobid,
        'pbs_jobid': jobid, 'arm': arm, 'updates': 0, 'updates_this_job': 0,
        'source_updates': 0, 'status': 'starting', 'complete': False,
        'target_linears': TARGET_LINEARS, 'calibration_ids': p['calibration']['case_ids'],
        'update_order': p['calibration']['update_order'], 'checkpoint': None,
        'resume_from': resume_from, 'resume_lineage': [],
    }


def source_bindings(p):
    src = p['source_c80_protocol']
    src_clip = src['source_clipping_protocol']
    parent = src_clip['parent_root']
    return {
        'source_c80_protocol': src['protocol'],
        'source_c80_root': src['remote_root'],
        'source_c80_training_job': p['sources']['clip_only_training_job'],
        'source_c80_prepare_job': p['sources']['c80_prepare_job'],
        'baseline_root': parent,
        'baseline_arm': src_clip['parent_arm'],
        'baseline_training_job': p['sources']['baseline_training_job'],
        'baseline_validation_job': p['sources']['baseline_validation_job'],
        'clip_root': src['clipping_root'],
        'runtime_artifacts': p['sources']['runtime_artifacts'],
        'bank_directory': p['calibration']['bank_directory'],
        'teacher_directory': p['sources']['teacher_directory'],
        'plan_file': p['calibration']['plan_file'],
        'baseline_checkpoint': parent.rstrip('/') + '/results/training/' + src_clip['parent_arm'] + '/deployment.pt',
    }


def validate_resume_source(root, arm, resume_from, checkpoint_path, progress):
    if not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+', resume_from or ''):
        raise RuntimeError('Invalid source PBS job ID for checkpoint resume')
    source_artifact = Path(root) / 'artifacts' / resume_from
    summary = json.loads((source_artifact / 'calibration_summary.json').read_text())
    if ((source_artifact / 'exit_code.txt').read_text().strip() != '75'
            or not (source_artifact / 'TRAINING_CHECKPOINTED').is_file()
            or (source_artifact / 'TRAINING_COMPLETE').exists()
            or summary.get('status') != 'checkpointed'
            or summary.get('complete') is not False
            or summary.get('arm') != arm
            or summary.get('jobid', summary.get('pbs_jobid')) != resume_from
            or summary.get('checkpoint') != str(checkpoint_path)
            or not isinstance(summary.get('updates'), int)
            or not 0 <= summary['updates'] <= UPDATES
            or progress.get('status') != 'checkpointed'
            or progress.get('complete') is not False
            or progress.get('arm') != arm
            or progress.get('jobid', progress.get('pbs_jobid')) != resume_from
            or progress.get('checkpoint') != str(checkpoint_path)
            or progress.get('updates') != summary['updates']
            or not Path(checkpoint_path).is_file()):
        raise RuntimeError('Previous PBS attempt is not a verified checkpointed resume source')
    return summary


def run(p, arm, summary, resume_from=None):
    root = Path(p['remote_root'])
    out = root / 'results' / 'training' / arm
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / 'checkpoint.pt'
    artifact = Path(os.environ['ARTIFACTS'])
    state = dict(summary)
    state.update({'status': 'running', 'phase': 'validating_sources',
                  'checkpoint': str(checkpoint_path), 'updates': 0,
                  'updates_this_job': 0, 'new_updates': 0,
                  'resume_from': resume_from, 'resume_lineage': []})
    resume_source_progress = (json.loads((out / 'progress.json').read_text())
                              if resume_from else None)
    resume_source_summary = (validate_resume_source(root, arm, resume_from, checkpoint_path,
                                                    resume_source_progress)
                             if resume_from else None)

    def progress(**values):
        state.update(values)
        state['jobid'] = state['pbs_jobid'] = os.environ['PBS_JOBID']
        save_json(artifact / 'calibration_summary.json', state)
        save_json(out / 'progress.json', state)
        print('C80_CALIBRATION_PROGRESS ' + json.dumps(state, allow_nan=False), flush=True)

    progress()
    import torch
    import numpy as np
    import sampler
    import runner
    import calibration_source
    import clipping_reference as ref
    import latent_reference
    import train_banks
    from clip_banks import ClipBanks
    from checkpointing import (atomic_torch_save, capture_rng, optimizer_to_device,
                               restore_rng, to_cpu)

    sampler._require_compute_allocation()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261017)
    backward_receipt = json.loads((Path(p['remote_root']) / 'results/backward_check.json').read_text())
    validate_backward_receipt(
        backward_receipt, getattr(train_banks, 'BACKWARD_IMPLEMENTATION', None),
        getattr(train_banks, 'BACKWARD_MAX_VECTORS', None))
    progress(phase='validating_sources', backward_check={
        'passed': backward_receipt['passed'],
        'implementation': backward_receipt['implementation'],
        'max_vectors_per_chunk': backward_receipt['max_vectors_per_chunk'],
    })
    source_protocol = p['source_c80_protocol']
    source_root = Path(source_protocol['remote_root'])
    if json.loads((source_root / 'protocol.json').read_text()) != source_protocol:
        raise RuntimeError('The source C80 protocol file differs from the frozen embedded protocol')
    source_clip_protocol = source_protocol['source_clipping_protocol']
    if json.loads((Path(source_protocol['clipping_root']) / 'protocol.json').read_text()) != source_clip_protocol:
        raise RuntimeError('The source C8 clipping protocol differs from the embedded parent protocol')

    plan_path = Path(p['calibration']['plan_file'])
    plan = json.loads(plan_path.read_text())
    calibration = p['calibration']
    if (plan.get('status') != 'complete'
            or plan.get('calibration_cohorts', {}).get('c80') != calibration['case_ids']):
        raise RuntimeError('Frozen C80 input plan differs')
    rows = {int(row['case_id']): row for row in plan['inputs']}
    if not set(calibration['case_ids']).issubset(rows):
        raise RuntimeError('Frozen C80 plan is missing calibration rows')
    for case in calibration['case_ids']:
        row = rows[case]
        if (row.get('domain') != source_protocol['data']['domain']
                or len(row.get('sampler_seeds', [])) != 2):
            raise RuntimeError(f'Calibration plan row {case} has a different domain or seed binding')

    teacher_dir = Path(p['sources']['teacher_directory'])
    teacher_receipt = json.loads((teacher_dir / 'receipt.json').read_text())
    expected_prepare = p['sources']['c80_prepare_job']
    if (teacher_receipt.get('complete') is not True or teacher_receipt.get('queries') != UPDATES
            or teacher_receipt.get('case_ids') != calibration['case_ids']
            or teacher_receipt.get('seed_indices') != [0]
            or teacher_receipt.get('pbs_jobid') != expected_prepare):
        raise RuntimeError('C80 calibration teacher receipt differs')
    for case in calibration['case_ids']:
        path = teacher_dir / f'{case}_0.pt'
        if not path.is_file():
            raise RuntimeError(f'Missing frozen C80 teacher {case}_0.pt')
    teachers = {}
    for case in calibration['case_ids']:
        saved = torch.load(teacher_dir / f'{case}_0.pt', map_location='cpu', weights_only=True)
        row = rows[case]
        if saved.get('plan_row') != row or saved.get('sampler_seed') != row['sampler_seeds'][0]:
            raise RuntimeError(f'C80 teacher binding differs for case {case}')
        action, future = saved.get('action'), saved.get('future')
        if (tuple(action.shape) != (32, 7) or tuple(future.shape) != (1, 48, 3, 14, 28)
                or not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(future).all())):
            raise RuntimeError(f'C80 teacher tensor is invalid for case {case}')
        teachers[case] = saved

    source_training = source_root / 'results/train_eval'
    source_receipt = json.loads((source_training / 'training_receipt.json').read_text())
    source_deployment = torch.load(source_training / 'deployment.pt', map_location='cpu', weights_only=True)
    source_summary = json.loads((source_training / 'summary.json').read_text())
    source_training_job = p['sources']['clip_only_training_job']
    if (source_receipt.get('protocol') != source_protocol['protocol']
            or source_receipt.get('complete') is not True
            or source_receipt.get('updates') != UPDATES
            or source_receipt.get('calibration_ids') != calibration['case_ids']
            or source_receipt.get('update_order') != calibration['update_order']
            or source_receipt.get('weights_unchanged') is not True
            or source_receipt.get('trainable_ratios') != TARGET_LINEARS
            or source_receipt.get('pbs_jobid') != source_training_job
            or source_summary.get('complete') is not True
            or source_summary.get('updates') != UPDATES
            or source_summary.get('pbs_jobid') != source_training_job
            or source_deployment.get('protocol') != source_protocol
            or source_deployment.get('updates') != UPDATES
            or len(source_deployment.get('ratios', {})) != TARGET_LINEARS
            or source_deployment.get('baseline_checkpoint') != source_bindings(p)['baseline_checkpoint']):
        raise RuntimeError('Completed C80 clip_only source training binding differs')
    if ([row.get('case_id') for row in source_receipt.get('history', [])]
            != calibration['update_order']
            or [row.get('update') for row in source_receipt.get('history', [])]
            != list(range(1, UPDATES + 1))):
        raise RuntimeError('Source clip_only history does not follow the frozen C80 update order')
    progress(phase='loading_baseline', source_updates=UPDATES if arm == 'clip_only' else 0)

    data_root, parent_rows, _, baseline, baseline_reference = ref.load_inputs(source_clip_protocol, torch)
    if (len(baseline.get('module_masters', {})) != TARGET_LINEARS
            or baseline.get('arm') != 'rho1.5' or baseline.get('updates') != 8
            or baseline.get('normalization_c') != p['training']['normalization_c']):
        raise RuntimeError('C8 rho1.5 update8 baseline masters differ')
    if any(parent_rows[case] != rows[case] for case in calibration['case_ids']):
        raise RuntimeError('C80 plan rows differ from the frozen C8 parent plan')

    deadline = ref.Deadline()
    runtime = None
    try:
        runtime = runner.Runtime(out)
        runtime.model.eval()
        runtime.arm('bf16')
        banks = ClipBanks(runtime, calibration['bank_directory'], baseline['module_masters'],
                          arm, p['clipping'])
        if (len(banks.modules) != TARGET_LINEARS or banks.train_weights != (arm == 'joint')
                or banks.train_clips is not True):
            raise RuntimeError('Trainable parameter groups do not match the requested arm')
        if arm == 'clip_only':
            banks.restore_clips(source_deployment['ratios'])
            if banks.train_weights or any(banks.books[name].requires_grad or banks.logs[name].requires_grad
                                          for name in banks.modules):
                raise RuntimeError('clip_only export must keep all codebooks and row gains frozen')
        else:
            groups = banks.optimizer_groups(p['training'])
            parameters = [value for group in groups for value in group['params']]
            expected_parameters = [value for name in banks.modules for value in
                                   (banks.books[name], banks.logs[name], banks.ratios[name])]
            if (len(parameters) != len(expected_parameters)
                    or {id(value) for value in parameters} != {id(value) for value in expected_parameters}
                    or any(not value.requires_grad for value in expected_parameters)):
                raise RuntimeError('joint must optimize all books, log row gains, and 614 clip ratios')

        def datum(case):
            obs, metadata = calibration_source.load_observation(np, data_root, rows[case])
            return runtime.datum(obs), metadata

        def load_teacher(case):
            return teachers[case]

        updates, history = 0, []
        resume_lineage = []
        job_start_updates = 0
        if arm == 'joint':
            optimizer = torch.optim.Adam(groups)
            if checkpoint_path.is_file() and not resume_from:
                raise RuntimeError('A joint checkpoint exists; exact PBS checkpoint resume is required')
            if resume_from:
                if not checkpoint_path.is_file():
                    raise RuntimeError('Resume source was provided but the joint checkpoint is missing')
                if not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+', resume_from):
                    raise RuntimeError('Invalid source PBS job ID for checkpoint resume')
                source_artifact = root / 'artifacts' / resume_from
                source_summary = json.loads((source_artifact / 'calibration_summary.json').read_text())
                source_progress = resume_source_progress
                if ((source_artifact / 'exit_code.txt').read_text().strip() != '75'
                        or not (source_artifact / 'TRAINING_CHECKPOINTED').is_file()
                        or (source_artifact / 'TRAINING_COMPLETE').exists()
                        or source_summary.get('status') != 'checkpointed'
                        or source_summary.get('complete') is not False
                        or source_summary.get('arm') != arm
                        or source_summary.get('jobid') != resume_from
                        or source_summary.get('checkpoint') != str(checkpoint_path)
                        or source_progress.get('status') != 'checkpointed'
                        or source_progress.get('complete') is not False
                        or source_progress.get('arm') != arm
                        or source_progress.get('jobid') != resume_from
                        or source_progress.get('checkpoint') != str(checkpoint_path)):
                    raise RuntimeError('Previous PBS attempt is not a verified checkpointed resume source')
                resume_lineage = list(source_progress.get('resume_lineage', [])) + [resume_from]
            if checkpoint_path.is_file():
                saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                if (saved.get('protocol') != p or saved.get('arm') != arm
                        or saved.get('update_order') != calibration['update_order']):
                    raise RuntimeError('Joint checkpoint protocol or arm differs')
                updates, history = saved['updates'], saved['history']
                if (not isinstance(updates, int) or not 0 <= updates <= UPDATES
                        or len(history) != updates
                        or [item['case_id'] for item in history]
                        != calibration['update_order'][:updates]
                        or [item['update'] for item in history] != list(range(1, updates + 1))):
                    raise RuntimeError('Joint checkpoint update history is not an exact prefix')
                if resume_from and resume_source_summary.get('updates') != updates:
                    raise RuntimeError('Previous PBS summary and exact checkpoint update count differ')
                banks.restore(saved['module_masters'])
                banks.restore_clips(saved['ratios'])
                optimizer.load_state_dict(saved['optimizer'])
                optimizer_to_device(torch, optimizer, 'cuda')
                restore_rng(torch, saved['rng'])
            job_start_updates = updates
            state['resume_lineage'] = resume_lineage
            progress(phase='training', updates=updates, updates_this_job=0,
                     checkpoint=str(checkpoint_path))
            while updates < UPDATES:
                deadline.check(max([item['seconds'] for item in history[-3:]], default=300) * 1.15)
                case = calibration['update_order'][updates]
                value, metadata = datum(case)
                runtime.description = metadata['description']
                banks.seen.clear()
                teacher = load_teacher(case)
                optimizer.zero_grad(set_to_none=True)
                started = time.monotonic()
                action, counts, future = sampler.differentiable_action(
                    runtime, value, rows[case]['sampler_seeds'][0],
                    checkpointed=True, return_future=True)
                action_mse = (action[:10, :6].float()
                              - teacher['action'][:10, :6].to('cuda').float()).square().mean()
                latent_nmse = latent_reference.future_nmse_loss(
                    torch, future, teacher['future'].to('cuda'))
                loss = (action_mse + p['training']['rho']
                        * p['training']['normalization_c'] * latent_nmse)
                if (len(banks.seen) != TARGET_LINEARS or not loss.requires_grad
                        or not bool(torch.isfinite(loss))):
                    raise RuntimeError('Training coverage or loss check failed')
                loss.backward()
                coverage = banks.coverage()
                for group in ('weight', 'clip'):
                    current = coverage.get(group, {})
                    if (current.get('connected') != TARGET_LINEARS
                            or current.get('finite') != TARGET_LINEARS
                            or current.get('nonzero', 0) < 1):
                        raise RuntimeError(f'Gradient coverage failed for {group}: {current}')
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, p['training']['gradient_norm_clip'], error_if_nonfinite=True)
                optimizer.step()
                if updates == 0 and p['training']['optimizer_reset_after_update1']:
                    optimizer.state.clear()
                banks.project()
                torch.cuda.synchronize()
                updates += 1
                history.append({
                    'update': updates, 'case_id': case, 'seed_index': 0,
                    'action_mse': float(action_mse), 'future_latent_nmse': float(latent_nmse),
                    'loss': float(loss), 'gradient_norm': float(norm), 'coverage': coverage,
                    'seconds': time.monotonic() - started, 'pbs_jobid': os.environ['PBS_JOBID'],
                    **counts,
                })
                del action, future, action_mse, latent_nmse, loss
                atomic_torch_save(torch, checkpoint_path, {
                    'protocol': p, 'arm': arm, 'updates': updates,
                    'update_order': calibration['update_order'], 'history': history,
                    'module_masters': banks.snapshot(), 'ratios': banks.clip_snapshot(),
                    'optimizer': to_cpu(optimizer.state_dict()), 'rng': capture_rng(torch),
                })
                progress(phase='training', updates=updates,
                         updates_this_job=updates - job_start_updates, last_update=history[-1],
                         checkpoint=str(checkpoint_path))
        else:
            if resume_from:
                raise RuntimeError('clip_only export does not accept checkpoint resume')
            updates = UPDATES
            history = source_receipt['history']
            job_start_updates = UPDATES
            resume_lineage = []
            progress(phase='source_reused', updates=UPDATES, updates_this_job=0,
                     source_updates=UPDATES, history_source=str(source_training / 'training_receipt.json'),
                     checkpoint=None)

        deadline.check(300)
        module_masters = banks.snapshot()
        ratios = banks.clip_snapshot()
        if (len(module_masters) != TARGET_LINEARS or len(ratios) != TARGET_LINEARS
                or any(t.dtype != torch.float32 for value in module_masters.values()
                       for t in (value['books'], value['log_scales']))
                or any(value.dtype != torch.float32 for value in ratios.values())):
            raise RuntimeError('Final deployment must contain 614 FP32 masters and ratios')
        if arm == 'clip_only':
            for name, master in baseline['module_masters'].items():
                if (not torch.equal(module_masters[name]['books'], master['books'].float())
                        or not torch.equal(module_masters[name]['log_scales'], master['log_scales'].float())):
                    raise RuntimeError('clip_only changed a codebook or row gain')

        deployment = {
            'schema_version': 1, 'protocol': p['protocol'], 'arm': arm,
            'updates': UPDATES, 'epochs': 1, 'target_linears': TARGET_LINEARS,
            'module_masters': module_masters, 'ratios': ratios,
            'normalization_c': p['training']['normalization_c'], 'rho': p['training']['rho'],
            'case_ids': calibration['case_ids'], 'update_order': calibration['update_order'],
            'source': source_bindings(p),
        }
        validate_deployment(deployment, p, arm)
        atomic_torch_save(torch, out / 'deployment.pt', deployment)
        progress(phase='reference_parity', updates=UPDATES,
                 updates_this_job=updates - job_start_updates,
                 deployment=str(out / 'deployment.pt'), checkpoint=str(checkpoint_path)
                 if arm == 'joint' else None)

        reference = p['closed_loop_reference']
        case, seed_index = reference['case_id'], reference['seed_index']
        if case not in rows or case != 2000 or seed_index != 0:
            raise RuntimeError('Closed-loop reference must be frozen C80 case 2000 seed 0')
        value, metadata = datum(case)
        seed = rows[case]['sampler_seeds'][seed_index]
        runtime.description = metadata['description']
        banks.seen.clear()
        with torch.no_grad():
            action, counts, _future = sampler.differentiable_action(
                runtime, value, seed, checkpointed=False, return_future=True)
        if len(banks.seen) != TARGET_LINEARS or tuple(action.shape) != (32, 7):
            raise RuntimeError('Reference sampler did not reach 614 Linears with a [32,7] action')
        actual_action, _ = runtime.infer(value, seed, measure=False)
        max_abs = float((actual_action.detach().float().cpu()
                         - action.detach().float().cpu()).abs().max())
        if not math.isfinite(max_abs) or max_abs > reference['max_abs_tolerance']:
            raise RuntimeError(f'Reference sampler/pilot inference parity exceeds tolerance: {max_abs}')
        reference_action = {
            'schema_version': 1, 'protocol': p['protocol'], 'arm': arm,
            'case_id': case, 'seed_index': seed_index, 'sampler_seed': seed,
            'action': action.detach().float().cpu().tolist(),
            'target_linears': len(banks.seen), 'counts': counts,
            'actual_infer_parity': {'max_abs': max_abs, 'passed': True},
            'plan_row': rows[case], 'metadata': metadata,
        }
        save_json(out / 'reference_action.json', reference_action)
        if runtime.env is not None:
            runtime.env.close()
            runtime = None

        receipt = {
            'schema_version': 1, 'protocol': p['protocol'], 'arm': arm,
            'status': 'complete', 'complete': True, 'updates': UPDATES, 'epochs': 1,
            'target_linears': TARGET_LINEARS, 'calibration_ids': calibration['case_ids'],
            'update_order': calibration['update_order'],
            'normalization_c': p['training']['normalization_c'], 'rho': p['training']['rho'],
            'source': source_bindings(p), 'weights_unchanged': arm == 'clip_only',
            'backward_check': {
                'passed': backward_receipt['passed'],
                'implementation': backward_receipt['implementation'],
                'max_vectors_per_chunk': backward_receipt['max_vectors_per_chunk'],
            },
            'history_source': (str(source_training / 'training_receipt.json') if arm == 'clip_only'
                               else str(checkpoint_path)),
            'history': history, 'source_updates': UPDATES if arm == 'clip_only' else 0,
            'new_updates': 0 if arm == 'clip_only' else UPDATES,
            'resume_from': resume_from, 'resume_lineage': resume_lineage,
            'actual_infer_parity': reference_action['actual_infer_parity'],
            'reference_action_parity': reference_action['actual_infer_parity'],
            'pbs_jobid': os.environ['PBS_JOBID'],
        }
        validate_receipt(receipt, p, arm)
        save_json(out / 'training_receipt.json', receipt)
        progress(phase='complete', status='complete', complete=True, updates=UPDATES,
                 updates_this_job=updates - job_start_updates,
                 source_updates=UPDATES if arm == 'clip_only' else 0,
                 checkpoint=str(checkpoint_path) if arm == 'joint' else None,
                 reference_action=str(out / 'reference_action.json'),
                 deployment=str(out / 'deployment.pt'),
                 actual_infer_parity=reference_action['actual_infer_parity'],
                 history_source=receipt['history_source'],
                 weights_unchanged=receipt['weights_unchanged'])
        print('C80_CALIBRATION_COMPLETE ' + json.dumps(state, allow_nan=False), flush=True)
        return 0
    except TimeoutError as exc:
        progress(phase='checkpointed', status='checkpointed', complete=False,
                 reason=str(exc), checkpoint=str(checkpoint_path))
        return 75
    except Exception as exc:
        progress(phase='failed', status='failed', complete=False,
                 error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        if runtime is not None and runtime.env is not None:
            try:
                runtime.env.close()
            except Exception as exc:
                print('C80_CALIBRATION_CLEANUP_ERROR ' + type(exc).__name__ + ': ' + str(exc),
                      file=sys.stderr, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('clip_only', 'joint'))
    parser.add_argument('--resume-from')
    parser.add_argument('--protocol', type=Path, default=HERE / 'protocol.json')
    args = parser.parse_args(argv)
    p = json.loads(args.protocol.read_text())
    arm = args.arm or os.environ.get('FW_CAL_ARM')
    if arm not in ('clip_only', 'joint'):
        parser.error('select FW_CAL_ARM=clip_only|joint')
    jobid = os.environ.get('PBS_JOBID', 'missing')
    if not os.environ.get('ARTIFACTS'):
        raise RuntimeError('PBS ARTIFACTS path is required')
    artifact = Path(os.environ['ARTIFACTS'])
    summary = initial_summary(p, arm, jobid, args.resume_from)
    save_json(artifact / 'calibration_summary.json', summary)
    allocation_verified = False
    try:
        validate_protocol(p)
        guard()
        allocation_verified = True
        return run(p, arm, summary, args.resume_from)
    except Exception as exc:
        current = json.loads((artifact / 'calibration_summary.json').read_text())
        current.update({'jobid': os.environ.get('PBS_JOBID', 'missing'),
                        'pbs_jobid': os.environ.get('PBS_JOBID', 'missing'),
                        'status': 'failed', 'complete': False,
                        'error': type(exc).__name__ + ': ' + str(exc)})
        save_json(artifact / 'calibration_summary.json', current)
        if allocation_verified:
            progress_path = Path(p['remote_root']) / 'results' / 'training' / arm / 'progress.json'
            save_json(progress_path, current)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
