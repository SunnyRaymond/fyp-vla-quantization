"""PBS-only numerical self-checks for codebook.py."""

from __future__ import annotations

import os
import socket
from pathlib import Path


def _require_compute_allocation() -> None:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    host = socket.gethostname().lower()
    if not job_id or not nodefile:
        raise RuntimeError("numerical codebook checks require PBS_JOBID and PBS_NODEFILE")
    if "login" in host:
        raise RuntimeError(f"refusing numerical codebook checks on login host {host}")
    nodefile_path = Path(nodefile)
    if not nodefile_path.is_file():
        raise RuntimeError("PBS_NODEFILE must name an existing allocation node file")
    allocated = {
        line.strip().split(".", 1)[0].lower()
        for line in nodefile_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not allocated or host.split(".", 1)[0] not in allocated:
        raise RuntimeError(f"current host {host} is absent from PBS_NODEFILE")


def _engine():
    _require_compute_allocation()
    import torch
    import codebook

    return torch, codebook


def test_row_scale_decode_and_storage() -> None:
    torch, vq = _engine()
    weight = torch.tensor([[3.0, 4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 2.0, 0.0]], dtype=torch.bfloat16)
    encoding = vq.fit_vq(weight, seed=17, sample_limit=32, iterations=2)

    expected_scales = weight.float().square().mean(dim=1).sqrt().to(torch.bfloat16)
    assert torch.equal(encoding["row_scales"], expected_scales)
    assert encoding["books"].dtype == torch.bfloat16
    assert tuple(encoding["books"].shape) == (2, 256, 4)
    assert encoding["indices"].dtype == torch.uint8
    assert tuple(encoding["indices"].shape) == (2, 2, 2)

    decoded = vq.decode_vq(encoding)
    books = encoding["books"].float()
    idx = encoding["indices"].long()
    manual = books[0][idx[..., 0]] + books[1][idx[..., 1]]
    manual = manual.reshape(2, -1)[:, :5] * encoding["row_scales"].float()[:, None]
    assert torch.equal(decoded, manual.to(torch.bfloat16))
    assert vq.encoded_nbytes(encoding) == 2 * 256 * 4 * 2 + 2 * 2 * 2 + 2 * 2


def test_quantized_input_output_compensation() -> None:
    torch, vq = _engine()
    books = torch.zeros((2, 256, 4), dtype=torch.bfloat16)
    books[0, 0, 0] = 0.5
    encoding = {
        "format": vq.FORMAT,
        "shape": (1, 1),
        "block_size": 4,
        "books": books,
        "row_scales": torch.ones((1,), dtype=torch.bfloat16),
        "indices": torch.zeros((1, 1, 2), dtype=torch.uint8),
    }
    # Teacher output is ZB^T = 1.0 * 1.5; the layer receives QZ = 2.0.
    inputs_q = torch.tensor([[2.0]])
    teacher_outputs = torch.tensor([[1.5]])
    objective = lambda dense: vq.local_output_loss(dense, inputs_q, teacher_outputs, reduction="sum")
    initial_loss = float(objective(vq.decode_vq(encoding).float()).item())
    tuned, stats = vq.tune_fixed_indices(
        encoding,
        objective,
        max_steps=32,
        patience=8,
        learning_rate=0.05,
        tune_scales=False,
    )
    tuned_loss = float(objective(vq.decode_vq(tuned).float()).item())
    assert initial_loss > 0.2
    assert tuned_loss < initial_loss * 0.1
    assert stats["best_loss"] <= tuned_loss + 1e-6
    assert torch.equal(tuned["indices"], encoding["indices"])


def test_anisotropic_metric_protects_sensitive_coordinate() -> None:
    torch, vq = _engine()
    books = torch.zeros((2, 256, 4), dtype=torch.bfloat16)
    books[0, 0] = torch.tensor([0.1, 10.0, 0.0, 0.0], dtype=torch.bfloat16)
    books[0, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.bfloat16)
    encoding = {
        "format": vq.FORMAT,
        "shape": (1, 4),
        "block_size": 4,
        "books": books,
        "row_scales": torch.ones((1,), dtype=torch.bfloat16),
        "indices": torch.tensor([[[1, 0]]], dtype=torch.uint8),
    }
    target = torch.zeros((1, 4))
    isotropic, _ = vq.reassign_indices(encoding, target, second_moment=torch.ones((4,)))
    sensitive, scores = vq.reassign_indices(
        encoding,
        target,
        block_metric=torch.diag(torch.tensor([1000.0, 1.0, 1.0, 1.0])).unsqueeze(0),
    )

    assert int(isotropic["indices"][0, 0, 0]) == 1
    assert int(sensitive["indices"][0, 0, 0]) == 0
    assert scores["objective_after"] < scores["objective_before"]
    assert torch.equal(sensitive["books"], encoding["books"])
    assert torch.equal(sensitive["row_scales"], encoding["row_scales"])


def main() -> None:
    test_row_scale_decode_and_storage()
    test_quantized_input_output_compensation()
    test_anisotropic_metric_protects_sensitive_coordinate()
    print("PBS codebook self-checks passed")


if __name__ == "__main__":
    main()
