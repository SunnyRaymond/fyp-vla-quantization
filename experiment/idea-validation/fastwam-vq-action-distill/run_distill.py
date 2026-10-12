"""True final-action distillation, resumable across genuine two-hour PBS allocations."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import signal
import statistics
import tempfile
import time
import traceback


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['case_id'], []).append(row)
    metrics = ('motor_rmse_first10_vs_bf16', 'gripper_rmse_first10_vs_bf16', 'motor_rmse_32_vs_bf16')
    values = {str(case): {metric: statistics.mean(r[metric] for r in records) for metric in metrics}
              for case, records in groups.items()}
    return {'case_count': len(groups), 'context_count': len(rows),
            'mean_motor_rmse_first10': statistics.mean(v[metrics[0]] for v in values.values()),
            'mean_gripper_rmse_first10': statistics.mean(v[metrics[1]] for v in values.values()),
            'mean_motor_rmse_32': statistics.mean(v[metrics[2]] for v in values.values()), 'per_case': values}


def numerical_contract(protocol):
    keys = ('source_root', 'source_full_job', 'roots', 'initial_arm', 'target_linears',
            'calibration_cases', 'calibration_seed_indices', 'selection_cases', 'test_cases',
            'evaluation_seed_indices', 'teacher', 'loss', 'trainables', 'fixed', 'optimizer',
            'learning_rate', 'gradient_norm_clip', 'maximum_epochs', 'updates_per_epoch',
            'checkpoint_selection', 'training_gradients', 'checkpointing')
    return {key: protocol[key] for key in keys}


def advance_training(state, protocol):
    state['updates_completed'] += 1
    state['next_train_index'] += 1
    if state['next_train_index'] == len(protocol['calibration_cases']):
        state['epochs_completed'] = state['epoch']
        state['phase'] = 'selection'


def evaluation_pairs(cases, seeds, rows):
    expected = [(case, seed) for case in cases for seed in seeds]
    completed = [(row['case_id'], row['seed_index']) for row in rows]
    if completed != expected[:len(completed)]:
        raise RuntimeError('Saved evaluation is not a unique completed prefix of the frozen cohort')
    return expected[len(completed):]


def run(args):
    import sampler
    sampler._require_compute_allocation()
    import torch
    import runner as pilot
    import run_diagnostic as source
    from train_banks import TrainableBanks, self_check
    from checkpointing import atomic_torch_save, capture_rng, optimizer_to_device, restore_rng, to_cpu

    p = json.loads(args.protocol.read_text(encoding='utf-8'))
    if p['protocol'] != 'fastwam-vq-final-action-distill-v2':
        raise ValueError('Unexpected protocol')
    if args.out.exists():
        raise FileExistsError(f'Existing result directory: {args.out}')
    args.out.mkdir(parents=True)
    save(args.out / 'protocol.json', p)
    started = time.monotonic()
    allocation_started = float(os.environ.get('FW_JOB_START_EPOCH', time.time()))
    deadline = started + p['walltime_seconds'] - (time.time() - allocation_started)
    job_id = os.environ['PBS_JOBID']
    interrupted = []
    signal.signal(signal.SIGTERM, lambda signum, frame: interrupted.append(signum))
    source_root, old_root = Path(p['source_root']), Path(p['roots']['old_inputs'])
    source_dir = source_root / 'results/full'
    stage, state, banks, optimizer = 'source_loading', None, None, None
    payload = None
    torch.backends.cuda.matmul.allow_tf32 = False

    def progress(**extra):
        record = {'status': 'running', 'stage': stage, 'job_id': job_id,
                  'updates_completed': state['updates_completed'] if state else 0,
                  'epochs_completed': state['epochs_completed'] if state else 0,
                  'elapsed_seconds': time.monotonic() - started, **extra}
        save(args.out / 'progress.json', record)
        print('DISTILL_PROGRESS ' + json.dumps(record), flush=True)

    try:
        full_summary = json.loads((source_dir / 'summary.json').read_text())
        if full_summary['status'] != 'complete' or full_summary['pbs_jobid'] != p['source_full_job']:
            raise RuntimeError('Source summary is not the verified completed run')
        old_plan = json.loads((old_root / 'plan.json').read_text())
        fresh = {row['case_id']: row for row in json.loads((source_root / 'plan.json').read_text())['inputs']}
        cached, references = {}, {}
        for line in (source_dir / 'actions.jsonl').open(encoding='utf-8'):
            row = json.loads(line)
            if row['arm'] not in ('bf16', p['initial_arm']):
                continue
            key = (row['case_id'], row['seed_index'])
            table = references if row['arm'] == 'bf16' else cached
            if key in table:
                raise RuntimeError(f'Duplicate cached action: {row["arm"]} {key}')
            table[key] = row
        for case in p['calibration_cases']:
            key = (case, 0)
            if key not in references:
                capture = torch.load(source_dir / 'sensitivity' / f'case_{case}.pt',
                                     map_location='cpu', weights_only=True, mmap=True)
                row = source.validate_plan(old_plan, case)
                references[key] = {'case_id': case, 'seed_index': 0, 'sampler_seed': row['sampler_seeds'][0],
                                   'action': capture['baseline_action'].detach().float().clone().tolist()}
                del capture
                gc.collect()
        for cases, seeds in ((p['calibration_cases'], [0]),
                             (p['selection_cases'] + p['test_cases'], p['evaluation_seed_indices'])):
            for case in cases:
                for seed_index in seeds:
                    key = (case, seed_index)
                    expected = (source.validate_plan(old_plan, case) if case < 1000 else fresh[case])['sampler_seeds'][seed_index]
                    for table in (references, cached):
                        if key not in table:
                            raise RuntimeError(f'Missing paired cached teacher/initialization {key}')
                        action = torch.tensor(table[key]['action'], dtype=torch.float32)
                        if table[key]['sampler_seed'] != expected or action.shape != (32, 7) or not torch.isfinite(action).all():
                            raise RuntimeError(f'Invalid cached action identity/values {key}')

        stage = 'checkpoint_self_checks'
        progress()
        self_check()
        from test_checkpointing import self_check as checkpoint_test
        from test_driver import self_check as driver_test
        with tempfile.TemporaryDirectory(prefix='checkpoint-test-', dir=args.out) as directory:
            checkpoint_test(torch, directory)
        driver_test(p)
        print('CHECKPOINT_AND_CURSOR_SELF_CHECKS_PASSED', flush=True)
        if args.resume_dir:
            if args.resume_dir.parent.resolve() != (Path(p['remote_root']) / 'results').resolve():
                raise RuntimeError('Resume directory is outside this campaign')
            previous_protocol = json.loads((args.resume_dir / 'protocol.json').read_text())
            if numerical_contract(previous_protocol) != numerical_contract(p):
                raise RuntimeError('Resume changes the frozen numerical contract')
            if (args.resume_dir / 'checkpoint.pt').is_file():
                payload = torch.load(args.resume_dir / 'checkpoint.pt', map_location='cpu', weights_only=True)
                if payload['schema_version'] != 2 or payload['numerical_contract'] != numerical_contract(p):
                    raise RuntimeError('Unsupported or mismatched resume checkpoint')
                state = payload['run_state']
                if state['phase'] == 'complete':
                    raise RuntimeError('Already completed checkpoint must not be resubmitted')
            else:
                failure = json.loads((args.resume_dir / 'failure.json').read_text())
                first = json.loads((args.resume_dir / 'first_backward.json').read_text())
                parity = json.loads((args.resume_dir / 'parity.json').read_text())
                grad_parity = json.loads((args.resume_dir / 'gradient_forward_parity.json').read_text())
                if (args.resume_dir.name != 'full-25755802.pbs101' or failure['updates'] != 1 or failure['epochs'] != 0
                        or 'First gradient timing cannot fit' not in failure['message']
                        or first['case_id'] != p['calibration_cases'][0] or first['update'] != 1
                        or any(first[key] != 614 for key in ('connected', 'finite', 'nonzero'))
                        or not all(row['passed'] for row in parity['records'])
                        or grad_parity['max_abs'] > p['parity_max_abs_tolerance']
                        or grad_parity['rmse'] > p['parity_rmse_tolerance']):
                    raise RuntimeError('Legacy first-step recovery evidence is incomplete')
                payload = {'legacy_master': torch.load(args.resume_dir / 'failure_trainable_master.pt',
                                                       map_location='cpu', weights_only=True),
                           'legacy_first': first, 'legacy_parity': parity}

        stage = 'model_loading'
        progress()
        runtime = pilot.Runtime(args.out)
        runtime.arm('bf16')
        stage = 'loading_existing_vq_banks'
        progress()
        banks = TrainableBanks(runtime, source_dir, p['initial_arm'])
        parameters = banks.parameters()
        optimizer = torch.optim.Adam(parameters, lr=p['learning_rate'])
        initial = banks.snapshot()
        if state is None:
            selection = aggregate([cached[case, seed] for case in p['selection_cases'] for seed in p['evaluation_seed_indices']])
            state = {'phase': 'train', 'epoch': 1, 'next_train_index': 0, 'updates_completed': 0,
                     'epochs_completed': 0, 'parity_records': [], 'parity_validated': False,
                     'training_history': [], 'evaluation_rows': [], 'gradient_seconds': [], 'forward_seconds': [],
                     'selection_history': [{'epoch': 0, 'source': 'paired cached initialization', **selection}],
                     'selected_epoch': 0, 'best_score': selection['mean_motor_rmse_first10'],
                     'frozen': False, 'resume_lineage': [], 'prior_wall_seconds': 0.0,
                     'optimizer_reset_after_legacy_step': False}
        best, last_trained = initial, None
        if payload and 'legacy_master' in payload:
            banks.restore(payload['legacy_master'])
            advance_training(state, p)
            state['training_history'].append(payload['legacy_first'])
            state['gradient_seconds'].append(payload['legacy_first']['gradient_seconds'])
            state['parity_records'] = payload['legacy_parity']['records']
            state['parity_validated'] = True
            state['optimizer_reset_after_legacy_step'] = True
            state['legacy_source'] = args.resume_dir.name
            state['prior_wall_seconds'] = 17 * 60 + 42
            print('LEGACY_FIRST_STEP_RECOVERED optimizer_reset_once=true next_case=2', flush=True)
        elif payload:
            banks.restore(payload['master_state'])
            optimizer.load_state_dict(payload['optimizer_state'])
            optimizer_to_device(torch, optimizer, 'cuda')
            best, last_trained = payload['best_state'], payload['last_trained_state']
            restore_rng(torch, payload['rng_state'])
            print('FULL_CHECKPOINT_RESTORED ' + json.dumps({'phase': state['phase'], 'updates': state['updates_completed']}), flush=True)
        state['resume_lineage'].append({'job_id': job_id, 'resume_from': args.resume_dir.name if args.resume_dir else None})
        del payload, initial
        gc.collect()
        data = {}

        def persist():
            snapshot = {'schema_version': 2, 'numerical_contract': numerical_contract(p),
                        'master_state': banks.snapshot(), 'optimizer_state': to_cpu(optimizer.state_dict()),
                        'best_state': best, 'last_trained_state': last_trained, 'rng_state': capture_rng(torch),
                        'run_state': {**state, 'prior_wall_seconds': state['prior_wall_seconds'] + time.monotonic() - started}}
            atomic_torch_save(torch, args.out / 'checkpoint.pt', snapshot)
            receipt = {'schema_version': 2, 'job_id': job_id, 'target_linears': len(banks.modules),
                       'phase': state['phase'], 'epoch': state['epoch'], 'next_train_index': state['next_train_index'],
                       'updates_completed': state['updates_completed'], 'epochs_completed': state['epochs_completed'],
                       'evaluation_contexts': len(state['evaluation_rows']), 'selected_epoch': state['selected_epoch'],
                       'optimizer_reset_after_legacy_step': state['optimizer_reset_after_legacy_step'],
                       'checkpoint': str(args.out / 'checkpoint.pt'), 'saved_utc_epoch': time.time()}
            save(args.out / 'checkpoint_receipt.json', receipt)
            (args.out / 'CHECKPOINT_READY').touch()
            save(args.out / 'selection_history.json', {'history': state['selection_history']})
            print('CHECKPOINT_SAVED ' + json.dumps(receipt), flush=True)

        def datum(case):
            if case not in data:
                row = source.validate_plan(old_plan, case) if case < 1000 else fresh[case]
                observation, metadata = source.load_case(old_root if case < 1000 else source_root, row)
                data[case] = (runtime.datum(observation), metadata['description'], row)
            return data[case]

        def query(case, seed_index, checkpointed):
            value, description, row = datum(case)
            runtime.description, runtime.current_arm = description, 'final_action_distill'
            banks.seen.clear()
            action, counts = sampler.differentiable_action(runtime, value, row['sampler_seeds'][seed_index], checkpointed=checkpointed)
            if len(banks.seen) != p['target_linears']:
                raise RuntimeError(f'Only {len(banks.seen)}/614 targets called')
            return action, counts

        def pause_before_next_operation(training):
            timings = state['gradient_seconds'] if training else state['forward_seconds']
            estimated = max(timings[-8:]) * 1.05 if timings else (330 if training else 90)
            if interrupted or time.monotonic() + estimated + p['save_reserve_seconds'] >= deadline:
                persist()
                save(args.out / 'summary.json', {'status': 'resumable', 'protocol': p['protocol'], 'job_id': job_id,
                     'target_linears': 614, 'completed_epochs': state['epochs_completed'], 'updates': state['updates_completed'],
                     'phase': state['phase'], 'next_train_index': state['next_train_index'],
                     'evaluation_contexts': len(state['evaluation_rows']), 'resume_lineage': state['resume_lineage'],
                     'wall_seconds': time.monotonic() - started, 'estimated_next_operation_seconds': estimated,
                     'reason': 'termination signal' if interrupted else 'next operation reaches allocation end',
                     'checkpoint': str(args.out / 'checkpoint.pt'), 'complete': False})
                print('DISTILL_RESUME_READY', flush=True)
                return True
            return False

        def evaluated(split, epoch):
            return [row for row in state['evaluation_rows'] if row['split'] == split and row['epoch'] == epoch]

        def evaluate(cases, seeds, split, epoch):
            if split == 'test' and not state['frozen']:
                raise RuntimeError('Test evaluation before freeze')
            rows = evaluated(split, epoch)
            for case, seed_index in evaluation_pairs(cases, seeds, rows):
                if pause_before_next_operation(False):
                    return None
                progress(case_id=case, seed_index=seed_index, split=split, epoch=epoch)
                tick = time.monotonic()
                action, counts = query(case, seed_index, False)
                action = action.detach().float().cpu()
                duration = time.monotonic() - tick
                state['forward_seconds'].append(duration)
                target = torch.tensor(references[case, seed_index]['action'], dtype=torch.float32)
                record = {'case_id': case, 'seed_index': seed_index, 'split': split, 'epoch': epoch,
                          'arm': 'final_action_distill', **source.action_metrics(torch, action, target),
                          'forward_seconds': duration, 'action': action.tolist(), **counts}
                rows.append(record)
                state['evaluation_rows'].append(record)
                persist()
                append(args.out / 'actions.jsonl', record)
            return aggregate(rows)

        persist()
        if not state['parity_validated']:
            stage = 'cached_initialization_forward_parity'
            parity_cases = [p['calibration_cases'][0], p['selection_cases'][0]]
            completed = [row['case_id'] for row in state['parity_records']]
            if completed != parity_cases[:len(completed)]:
                raise RuntimeError('Invalid parity checkpoint cursor')
            for case in parity_cases[len(completed):]:
                if pause_before_next_operation(False):
                    return
                tick = time.monotonic()
                action, counts = query(case, 0, False)
                state['forward_seconds'].append(time.monotonic() - tick)
                delta = action.detach().float().cpu() - torch.tensor(cached[case, 0]['action'], dtype=torch.float32)
                record = {'case_id': case, 'max_abs': float(delta.abs().max()), 'rmse': float(delta.square().mean().sqrt()), **counts}
                record['passed'] = record['max_abs'] <= p['parity_max_abs_tolerance'] and record['rmse'] <= p['parity_rmse_tolerance']
                if not record['passed']:
                    raise RuntimeError(f'Cached initialization parity failed: {record}')
                state['parity_records'].append(record)
                persist()
            state['parity_validated'] = True
            persist()
        save(args.out / 'parity.json', {'records': state['parity_records'], 'reused_on_resume': bool(args.resume_dir)})

        while state['phase'] not in ('finalize', 'complete'):
            stage = state['phase']
            if stage == 'train':
                if pause_before_next_operation(True):
                    return
                case = p['calibration_cases'][state['next_train_index']]
                progress(case_id=case, epoch=state['epoch'], update_in_epoch=state['next_train_index'])
                optimizer.zero_grad(set_to_none=True)
                tick = time.monotonic()
                action, counts = query(case, 0, True)
                if state['updates_completed'] == 0:
                    diff = action.detach().float().cpu() - torch.tensor(cached[case, 0]['action'])
                    grad_parity = {'max_abs': float(diff.abs().max()), 'rmse': float(diff.square().mean().sqrt())}
                    save(args.out / 'gradient_forward_parity.json', grad_parity)
                    if grad_parity['max_abs'] > p['parity_max_abs_tolerance'] or grad_parity['rmse'] > p['parity_rmse_tolerance']:
                        raise RuntimeError('Checkpointed gradient forward differs from cached deployed forward')
                teacher = torch.tensor(references[case, 0]['action'], dtype=torch.float32, device='cuda')
                loss = (action[:10, :6].float() - teacher[:10, :6]).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite actual final-action loss')
                loss_value = float(loss.detach())
                loss.backward()
                report = banks.gradient_report()
                if report['connected'] != 614 or report['finite'] != 614 or report['nonzero'] == 0:
                    save(args.out / 'gradient_coverage_failure.json', report)
                    raise RuntimeError('Trainable codebook gradient coverage is incomplete or nonfinite')
                norm = torch.nn.utils.clip_grad_norm_(parameters, p['gradient_norm_clip'], error_if_nonfinite=True)
                optimizer.step()
                del action, teacher, loss
                banks.refresh()
                torch.cuda.synchronize()
                duration = time.monotonic() - tick
                state['gradient_seconds'].append(duration)
                record = {'update': state['updates_completed'] + 1, 'case_id': case, 'epoch': state['epoch'],
                          'actual_preupdate_motor_mse': loss_value, 'gradient_norm_before_clip': float(norm),
                          'gradient_seconds': duration, **counts, **report}
                state['training_history'].append(record)
                advance_training(state, p)
                persist()
                append(args.out / 'training.jsonl', record)
                if state['updates_completed'] == 1:
                    save(args.out / 'first_backward.json', record)
            elif stage == 'selection':
                score = evaluate(p['selection_cases'], p['evaluation_seed_indices'], 'selection', state['epoch'])
                if score is None:
                    return
                state['selection_history'].append({'epoch': state['epoch'], **score})
                if score['mean_motor_rmse_first10'] < state['best_score']:
                    state['best_score'], state['selected_epoch'], best = score['mean_motor_rmse_first10'], state['epoch'], banks.snapshot()
                if state['epoch'] == p['maximum_epochs']:
                    state['phase'] = 'freeze'
                else:
                    state['epoch'] += 1
                    state['next_train_index'] = 0
                    state['phase'] = 'train'
                persist()
            elif stage == 'freeze':
                if state['updates_completed'] != p['maximum_epochs'] * p['updates_per_epoch']:
                    raise RuntimeError('Freeze before full declared training completion')
                last_trained = banks.snapshot()
                banks.restore(best)
                state['frozen'], state['phase'] = True, 'final_calibration'
                persist()
            elif stage in ('final_calibration', 'test'):
                split = 'calibration' if stage == 'final_calibration' else 'test'
                cases = p['calibration_cases'] if split == 'calibration' else p['test_cases']
                seeds = [0] if split == 'calibration' else p['evaluation_seed_indices']
                score = evaluate(cases, seeds, split, state['selected_epoch'])
                if score is None:
                    return
                state[split + '_summary'] = score
                state['phase'] = 'test' if split == 'calibration' else 'finalize'
                persist()
            else:
                raise RuntimeError(f'Invalid resume phase: {stage}')

        atomic_torch_save(torch, args.out / 'selected_codebook_overlay.pt',
                         {'source_root': str(source_root), 'source_arm': p['initial_arm'], 'module_names': list(banks.modules),
                          'master_state': best, 'fixed_indices_and_transforms': 'reuse corresponding source bank files'})
        atomic_torch_save(torch, args.out / 'last_trained_master.pt', last_trained)
        save(args.out / 'frozen_checkpoint.json', {'selected_epoch': state['selected_epoch'],
             'completed_epochs': state['epochs_completed'], 'updates': state['updates_completed'],
             'selection_score': state['best_score'], 'test_queries_before_freeze': 0})
        changed_books, changed_scales = 0, 0
        for name in banks.modules:
            encoding = banks.stored_encoding(name)
            changed_books += int(torch.count_nonzero(encoding['books'] != banks.encodings[name]['books']))
            changed_scales += int(torch.count_nonzero(encoding['row_scales'] != banks.encodings[name]['row_scales']))
        initial_test = aggregate([cached[case, seed] for case in p['test_cases'] for seed in p['evaluation_seed_indices']])
        test = state['test_summary']
        summary = {'status': 'complete', 'protocol': p['protocol'], 'job_id': job_id, 'target_linears': 614,
                   'completed_epochs': state['epochs_completed'], 'updates': state['updates_completed'],
                   'selected_epoch': state['selected_epoch'], 'selection_history': state['selection_history'],
                   'calibration': state['calibration_summary'], 'test': test, 'cached_initial_test': initial_test,
                   'relative_test_motor_rmse_change': test['mean_motor_rmse_first10'] / initial_test['mean_motor_rmse_first10'] - 1,
                   'changed_stored_book_scalars': changed_books, 'changed_stored_row_scales': changed_scales,
                   'evaluation_contexts': len(state['evaluation_rows']), 'test_queries_before_freeze': 0,
                   'teacher_inference_queries': 0, 'teacher_source': p['teacher'],
                   'training_seconds_per_update': state['gradient_seconds'], 'forward_seconds': state['forward_seconds'],
                   'wall_seconds': time.monotonic() - started,
                   'cumulative_wall_seconds': state['prior_wall_seconds'] + time.monotonic() - started,
                   'resume_lineage': state['resume_lineage'],
                   'optimizer_reset_after_legacy_step': state['optimizer_reset_after_legacy_step'],
                   'episodes': 0, 'native_low_bit_kernel': False, 'test_independence_boundary': p['reuse_boundary'],
                   'peak_cuda_allocated_bytes': int(torch.cuda.max_memory_allocated())}
        save(args.out / 'summary.json', summary)
        print('FINAL_ACTION_DISTILL_COMPLETE ' + json.dumps(summary), flush=True)
    except Exception as exc:
        # Keep the last committed checkpoint; the failed in-flight operation is never committed.
        save(args.out / 'failure.json', {'status': 'failed', 'stage': stage, 'message': str(exc),
             'updates': state['updates_completed'] if state else 0, 'epochs': state['epochs_completed'] if state else 0,
             'checkpoint_available': (args.out / 'checkpoint.pt').is_file(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume-dir', type=Path)
    run(parser.parse_args())
