"""PBS-side numerical smoke checks for smooth_vq.py; do not run on login nodes."""

import os
import socket


def _require_pbs_allocation():
    job_id = os.environ.get("PBS_JOBID", "").strip()
    host = socket.gethostname().strip().lower()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id:
        raise SystemExit("Refusing numerical self-check: PBS_JOBID is empty")
    if "login" in host:
        raise SystemExit(f"Refusing numerical self-check on login node: {host}")
    if not nodefile or not os.path.isfile(nodefile):
        raise SystemExit("Refusing numerical self-check: PBS_NODEFILE is missing")
    with open(nodefile, encoding="utf-8") as stream:
        allocated = {line.strip().split(".", 1)[0].lower() for line in stream if line.strip()}
    if host.split(".", 1)[0] not in allocated:
        raise SystemExit(f"Refusing numerical self-check: {host} is not listed in PBS_NODEFILE")


_require_pbs_allocation()

import torch

from smooth_vq import (
    _assign,
    decode_scalar_quantized,
    decode_vq_quantized,
    fit_vq,
    inverse_transform_input,
    make_transform,
    quant_activation,
    scalar_quantize,
    transform_input,
    transform_weight,
)


def test_transform_equivalence_and_hadamard_tail():
    torch.manual_seed(7)
    x = torch.randn(3, 131)
    w = torch.randn(9, 131)
    tr = make_transform(torch.rand(131) + 0.1, w, alpha=0.75, hadamard=True, seed=19)
    z, b = transform_input(x, tr), transform_weight(w, tr)
    torch.testing.assert_close(z @ b.T, x @ w.T, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(inverse_transform_input(z, tr), x, atol=2e-5, rtol=2e-5)
    assert z.shape == x.shape and b.shape == w.shape
    assert float(tr["scale"].min()) >= 1 / 16 and float(tr["scale"].max()) <= 16


def test_activation_zero_and_signed_q4():
    decoded, stats = quant_activation(torch.zeros(2, 5), 4)
    assert decoded.dtype == torch.bfloat16 and torch.count_nonzero(decoded) == 0
    assert stats["qmax"] == 7 and stats["zero_rows"] == 2
    x = torch.tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]])
    decoded, stats = quant_activation(x, 4)
    assert stats["saturation_count"] == 0 and torch.isfinite(decoded.float()).all()


def test_scalar_pack_decode_and_storage():
    torch.manual_seed(11)
    w = torch.randn(3, 129)
    result = scalar_quantize(w, group=128)
    decoded = decode_scalar_quantized(result["packed"], result["scales"], result["shape"], result["group"])
    torch.testing.assert_close(decoded, result["decoded"])
    assert result["packed"].dtype == torch.uint8
    assert result["scales"].dtype == torch.bfloat16
    assert result["storage_bytes"] == (w.numel() + 1) // 2 + 3 * 2 * 2
    assert result["decoded"].shape == w.shape and result["decoded"].dtype == torch.bfloat16
    reference = torch.empty_like(w, dtype=torch.bfloat16)
    for row in range(w.shape[0]):
        for start in range(0, w.shape[1], 128):
            end = min(start + 128, w.shape[1])
            part = w[row, start:end]
            scale = (part.abs().max() / 7).to(torch.bfloat16).float()
            safe = scale if float(scale) > 0 else torch.tensor(1.0)
            reference[row, start:end] = (torch.round(part / safe).clamp(-7, 7) * scale).to(torch.bfloat16)
    torch.testing.assert_close(result["decoded"], reference)


def test_vq_tail_index_decode_storage_and_toy_fit():
    # A five-column input exercises per-row d4 tails and exact stored-byte accounting.
    tail = fit_vq(torch.randn(3, 5), seed=3, sample_limit=32, iterations=2)
    assert tail["indices"].shape == (3, 2, 2) and tail["indices"].dtype == torch.uint8
    decoded = decode_vq_quantized(tail["indices"], tail["codebooks"], tail["shape"])
    torch.testing.assert_close(decoded, tail["decoded"])
    assert tail["codebooks"].shape == (2, 256, 4) and tail["codebooks"].dtype == torch.bfloat16
    assert tail["storage_bytes"] == tail["indices"].numel() + 2 * 256 * 4 * 2

    # Four repeated vector types should be represented accurately by the residual codebooks.
    patterns = torch.tensor([[1.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 3.0, 0], [0, 0, 0, 4.0]])
    toy = patterns.repeat(64, 1).reshape(16, 64)
    fitted = fit_vq(toy, seed=5, sample_limit=256, iterations=2)
    mse = (toy - fitted["decoded"].float()).square().mean()
    assert float(mse) < 1e-3


def test_diagonal_metric_changes_assignment():
    vectors = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    codebook = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.9, 3.0, 0.0, 0.0]])
    geometric = _assign(vectors, codebook, torch.ones_like(vectors))
    emphasize_first = _assign(vectors, codebook, torch.tensor([[1.0, 0.0, 1.0, 1.0]]))
    assert int(geometric[0]) == 0 and int(emphasize_first[0]) == 1


if __name__ == "__main__":
    test_transform_equivalence_and_hadamard_tail()
    test_activation_zero_and_signed_q4()
    test_scalar_pack_decode_and_storage()
    test_vq_tail_index_decode_storage_and_toy_fit()
    test_diagonal_metric_changes_assignment()
    print("smooth_vq self-checks passed")
