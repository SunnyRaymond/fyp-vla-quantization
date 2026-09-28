#!/usr/bin/env python3
"""Small adapter algebra check, run inside the PBS allocation."""

import torch

from runner import BankAdapter


class ToyStudent:
    def __call__(self, context, actions):
        return actions


def main():
    prediction = torch.arange(60, dtype=torch.float32).reshape(3, 5, 4)
    eye = torch.eye(4)
    identity = BankAdapter(ToyStudent(), eye)(None, prediction)
    assert torch.equal(identity, prediction)
    scalar = BankAdapter(ToyStudent(), 2 * eye)(None, prediction)
    assert torch.equal(scalar[:, :4], prediction[:, :4])
    assert torch.allclose(scalar[:, -1].mean(0), prediction[:, -1].mean(0))
    assert torch.allclose(
        scalar[:, -1] - scalar[:, -1].mean(0),
        2 * (prediction[:, -1] - prediction[:, -1].mean(0)),
    )
    print("adapter self-check PASS")


if __name__ == "__main__":
    main()
