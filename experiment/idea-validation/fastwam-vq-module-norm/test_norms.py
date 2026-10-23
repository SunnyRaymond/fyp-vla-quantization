"""One compute-only test for module matching, reuse, padding and overlay coding."""
from pathlib import Path
import tempfile
import torch
from common import guard
from prepare_norms import candidate, fit_module
from storage import write_overlay, read_overlay


def main():
    guard()
    torch.manual_seed(171031)
    # A padded final vector plus differently scaled modules expose global-only matching.
    shape = (128, 4)
    idx = torch.arange(128)[:, None].repeat(1, 2) % 8
    books = torch.randn(2, 256, 4).to(torch.bfloat16).float()
    valid = torch.ones(shape, dtype=torch.bool); valid[-1, 3] = False
    scales = torch.linspace(.3, 2., 128)
    baseline = (((books[0][idx[:, 0]] + books[1][idx[:, 1]]) * scales[:, None]) * valid).to(torch.bfloat16)
    g = torch.randn(shape) * valid
    sums, div = torch.zeros(256, 4), torch.zeros(256, 4)
    sums.index_add_(0, idx[:, 0], scales[:, None] * g)
    div.index_add_(0, idx[:, 0], scales.square()[:, None] * valid)
    d = dict(indices=idx, books=books, scales=scales, valid=valid, baseline=baseline,
             dc=-sums / div.clamp_min(1e-30), untied=-g, random=torch.randn(shape) * valid)
    target = candidate(d, 'shared', .5)[1]
    for kind in ('shared', 'untied', 'random'):
        value, actual, _ = fit_module(d, kind, target)
        assert abs(actual / target - 1) < .01 and value[-1, 3] == 0
    with tempfile.TemporaryDirectory() as folder:
        header = dict(format='full-support-bf16-overlay-v1', modules=[dict(module_index=2, book=0,
            codewords=list(range(8)), scalar_count=int(valid.sum()), shape=[1, 511])])
        values = {'toy': candidate(d, 'untied', .2)[0]}
        info = {'toy': dict(valid_coordinates=valid)}
        path = Path(folder) / 'toy.bin'
        write_overlay(path, header, values, info)
        restored = read_overlay(path, header, info, ['toy'])
        assert torch.equal(restored['toy'], values['toy'])
    print('MODULE_NORM_AND_OVERLAY_TESTS_PASSED', flush=True)


if __name__ == '__main__':
    main()
