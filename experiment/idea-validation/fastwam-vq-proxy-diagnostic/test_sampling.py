"""Small numerical checks; run only inside an approved PBS allocation."""
from pathlib import Path
import importlib.util


def main():
    import run_experiment
    run_experiment.guard()
    import torch
    spec = importlib.util.spec_from_file_location('proxy_diagnostic_driver', Path(__file__).with_name('run_diagnostic.py'))
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    old = driver.probes(torch, 20261009, 4)
    assert old.shape == (4, 10, 6)
    assert torch.allclose(old.reshape(4, 60).norm(dim=1), torch.ones(4))
    assert torch.equal(old, driver.probes(torch, 20261009, 4))
    assert not torch.equal(old, driver.probes(torch, 20261010, 4))
    # Overlapping subsets must each keep N/2, never N/union_size.
    call = {'row_ids': torch.tensor([1, 3, 5]), 'row_sets': {'old': torch.tensor([1, 3]), 'new': torch.tensor([3, 5])},
            'total_rows': 10, 'adjoints': torch.ones(2, 3, 1)}
    result = driver.project_error(torch, call, torch.tensor([[1.], [2.], [3.]]))
    assert torch.equal(result['old'], torch.tensor([15., 15.]))
    assert torch.equal(result['new'], torch.tensor([25., 25.]))
    # Shared calls are summed before the square: +1 and -1 may legitimately cancel.
    call['adjoints'] = torch.tensor([[[1.], [0.], [0.]], [[-1.], [0.], [0.]]])
    assert torch.equal(driver.project_error(torch, call, torch.tensor([[1.], [2.], [3.]]))['old'], torch.tensor([5., -5.]))
    metric = driver.vector_metrics(torch, torch.tensor([1., -1.]), torch.tensor([2., -2.]))
    assert metric['prediction_error_rmse'] == 1. and abs(metric['cosine'] - 1.) < 1e-6
    print('PROBE_ROW_SAMPLING_CHECKS_PASSED', flush=True)


if __name__ == '__main__':
    main()
