"""Small PBS self-check for train/evaluation resume cursors."""
from __future__ import annotations


def self_check(protocol):
    from run_distill import (advance_training, evaluation_pairs, latent_weight_for_rho,
                             numerical_contract)

    assert latent_weight_for_rho(0.0, 2.0) == 0.0
    assert latent_weight_for_rho(0.5, 2.0) == 1.0
    for rho, normalization_c in ((-0.1, 2.0), (0.0, 0.0)):
        try:
            latent_weight_for_rho(rho, normalization_c)
        except ValueError:
            pass
        else:
            raise AssertionError('Negative rho and nonpositive normalization must be rejected')

    state = {'phase': 'train', 'epoch': 1, 'next_train_index': 0,
             'updates_completed': 0, 'epochs_completed': 0}
    advance_training(state, protocol)
    assert state['next_train_index'] == 1 and protocol['calibration_cases'][1] == 2
    for _ in range(7):
        advance_training(state, protocol)
    assert state['phase'] == 'selection' and state['epochs_completed'] == 1 and state['updates_completed'] == 8
    state.update(phase='train', epoch=2, next_train_index=0)
    for _ in range(8):
        advance_training(state, protocol)
    assert state['updates_completed'] == 16 and state['epochs_completed'] == 2
    cases, seeds = protocol['test_cases'], protocol['evaluation_seed_indices']
    rows = [{'case_id': cases[0], 'seed_index': seed} for seed in seeds]
    pending = evaluation_pairs(cases, seeds, rows)
    assert len(pending) == 42 and pending[0] == (cases[1], seeds[0])
    try:
        evaluation_pairs(cases, seeds, rows + rows[:1])
    except RuntimeError:
        pass
    else:
        raise AssertionError('Duplicate completed evaluation must be rejected')
    changed = {**protocol, 'learning_rate': protocol['learning_rate'] * 2}
    assert numerical_contract(changed) != numerical_contract(protocol)
    assert numerical_contract({**protocol, 'save_reserve_seconds': 1}) == numerical_contract(protocol)
