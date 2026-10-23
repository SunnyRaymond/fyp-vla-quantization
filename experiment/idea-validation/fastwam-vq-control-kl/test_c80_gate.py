"""Small complete-cohort aggregation and projection checks; no model/data."""
from copy import deepcopy
import torch

from c80_gate_grad import add_gradients
from c80_gate_reduce import merge_shards, project_candidate


def main():
    keys = ['ratio::a', 'channel::a']
    originals = [torch.tensor(0.99), torch.tensor([0.98, -0.98])]
    a = {'parameter_keys': keys, 'originals': originals, 'case_ids': [1, 3],
         'gradient_sum': [torch.tensor(2.0), None], 'loss_sum': 3.0,
         'case_metrics': [{'case_id': 1, 'loss': 1.0}, {'case_id': 3, 'loss': 2.0}]}
    b = {'parameter_keys': keys, 'originals': [value.clone() for value in originals],
         'case_ids': [2, 4], 'gradient_sum': [torch.tensor(6.0), torch.tensor([4.0, -8.0])],
         'loss_sum': 7.0,
         'case_metrics': [{'case_id': 2, 'loss': 3.0}, {'case_id': 4, 'loss': 4.0}]}
    _, _, grads, loss = merge_shards(torch, [a, b], [1, 2, 3, 4])
    assert loss == 2.5 and float(grads[0]) == 2.0
    torch.testing.assert_close(grads[1], torch.tensor([1.0, -2.0]))
    total = add_gradients([None, torch.tensor([1.0, 1.0])], a['gradient_sum'])
    assert float(total[0]) == 2.0
    torch.testing.assert_close(total[1], torch.tensor([1.0, 1.0]))
    for bad in ('overlap', 'parameters'):
        changed = deepcopy(b)
        if bad == 'overlap':
            changed['case_ids'][0] = 1
        else:
            changed['originals'][0] += 0.1
        try:
            merge_shards(torch, [a, changed], [1, 2, 3, 4])
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid shard accepted: ' + bad)
    proposed, steps = project_candidate(
        torch, keys, originals, [torch.tensor(0.1), torch.tensor([0.1, -0.1])],
        1.0, {'ratio': (0.5, 1.0), 'channel': (-1.0, 1.0)})
    assert float(proposed['ratios']['a']) == 1.0
    torch.testing.assert_close(proposed['log2_scales']['a'], torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(steps[0], torch.tensor(1.0) - originals[0])
    torch.testing.assert_close(originals[1], torch.tensor([0.98, -0.98]))
    print('C80 checks passed: mean gradient, unused parameters, cohort rejection and projected actual steps')


if __name__ == '__main__':
    main()
