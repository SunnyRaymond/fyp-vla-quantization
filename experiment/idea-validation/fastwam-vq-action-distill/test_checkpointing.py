"""PBS-invoked checkpoint round-trip and interruption self-check."""
from __future__ import annotations

import random
from pathlib import Path


def _assert_equal(torch, left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(torch, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_equal(torch, a, b)
    else:
        assert left == right


def self_check(torch, tempdir) -> None:
    """Compare the next Adam update across uninterrupted and saved/resumed runs."""
    import numpy as np

    from checkpointing import (atomic_torch_save, capture_rng, optimizer_to_device,
                               restore_rng, to_cpu)

    random.seed(1207)
    np.random.seed(1207)
    torch.manual_seed(1207)

    def update(parameter, optimizer):
        optimizer.zero_grad(set_to_none=True)
        noise = torch.rand_like(parameter)
        target = torch.tensor(random.random() + float(np.random.random()), dtype=parameter.dtype)
        loss = ((parameter * noise - target) ** 2).sum()
        loss.backward()
        optimizer.step()

    parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    update(parameter, optimizer)
    cursor = {"updates_completed": 1, "epoch": 0, "next_case": 17,
              "evaluation_records": [{"epoch": 0, "score": 0.375, "split": "selection"}]}
    payload = to_cpu({"parameter": parameter.detach(), "optimizer": optimizer.state_dict(),
                      "rng": capture_rng(torch), "run": cursor})

    checkpoint = Path(tempdir) / "roundtrip.pt"
    atomic_torch_save(torch, checkpoint, payload)
    uninterrupted = torch.nn.Parameter(parameter.detach().clone())
    uninterrupted_optimizer = torch.optim.Adam([uninterrupted], lr=0.01)
    uninterrupted_optimizer.load_state_dict(optimizer.state_dict())
    update(uninterrupted, uninterrupted_optimizer)
    expected_parameter = uninterrupted.detach().clone()
    expected_optimizer = to_cpu(uninterrupted_optimizer.state_dict())

    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert loaded["run"] == cursor
    resumed = torch.nn.Parameter(loaded["parameter"].clone())
    resumed_optimizer = torch.optim.Adam([resumed], lr=0.01)
    resumed_optimizer.load_state_dict(loaded["optimizer"])
    optimizer_to_device(torch, resumed_optimizer, "cpu")
    restore_rng(torch, loaded["rng"])
    update(resumed, resumed_optimizer)
    assert torch.equal(resumed, expected_parameter)
    _assert_equal(torch, to_cpu(resumed_optimizer.state_dict()), expected_optimizer)

    previous_bytes = checkpoint.read_bytes()
    original_save = torch.save

    def fail_after_partial_write(value, destination):
        destination.write(b"partial checkpoint")
        raise OSError("simulated interrupted serialization")

    torch.save = fail_after_partial_write
    try:
        try:
            atomic_torch_save(torch, checkpoint, payload)
        except OSError as error:
            assert str(error) == "simulated interrupted serialization"
        else:
            raise AssertionError("Injected checkpoint write failure was not raised")
    finally:
        torch.save = original_save
    assert checkpoint.read_bytes() == previous_bytes
    assert list(Path(tempdir).iterdir()) == [checkpoint]
