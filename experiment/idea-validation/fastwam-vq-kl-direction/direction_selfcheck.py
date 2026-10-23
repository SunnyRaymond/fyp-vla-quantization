import copy
import json
import os

import torch

from direction_utils import adam_delta, gradient_summary


def _close(actual, expected, dtype):
    tol = 1e-12 if dtype == torch.float64 else 1e-6
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


def _angle_checks():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    groups = [{"name": "ratio", "params": [p]}, {"name": "channel", "params": []}]
    a = torch.tensor([1.0, 0.0], dtype=torch.float64)
    cases = [(torch.tensor([0.0, 2.0], dtype=torch.float64), 0.0),
             (-a, -1.0), (torch.zeros_like(a), 0.0)]
    for z, cosine in cases:
        got = gradient_summary(torch, [a], [z], groups)["overall"]
        assert abs(got["cosine"] - cosine) < 1e-12
    p2 = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    groups = [{"name": "ratio", "params": [p]}, {"name": "channel", "params": [p2]}]
    got = gradient_summary(torch, [None, torch.tensor([3.0])], [torch.tensor([1.0]), None], groups)
    assert got["overall"]["connected"] == {"a": 1, "z": 1, "total": 2}
    assert got["ratio"]["norm_a"] == 0.0 and got["channel"]["norm_z"] == 0.0


def _adam_case(dtype, history, device="cpu"):
    p0 = torch.nn.Parameter(torch.tensor([0.7, -0.2], dtype=dtype))
    p1 = torch.nn.Parameter(torch.tensor([-0.5, 0.9], dtype=dtype))
    opts = [{"params": [p0], "lr": 0.003, "betas": (0.9, 0.999), "eps": 1e-8},
            {"params": [p1], "lr": 0.017, "betas": (0.9, 0.999), "eps": 1e-8}]
    trained = torch.optim.Adam(opts)
    for t in range(history):
        p0.grad = torch.tensor([0.1 + t * 0.001, -0.2], dtype=dtype)
        p1.grad = torch.tensor([-0.4, 0.25 + t * 0.002], dtype=dtype)
        trained.step()
        trained.zero_grad(set_to_none=True)

    q = [torch.nn.Parameter(p.detach().to(device).clone()) for p in (p0, p1)]
    live = [{"params": [q[0]], "lr": 0.003, "betas": (0.9, 0.999), "eps": 1e-8},
            {"params": [q[1]], "lr": 0.017, "betas": (0.9, 0.999), "eps": 1e-8}]
    reference = torch.optim.Adam(live)
    saved = copy.deepcopy(trained.state_dict()) if history else None
    if saved is not None:
        if str(device).startswith("cuda"):
            assert all(s["exp_avg"].device.type == "cpu" for s in saved["state"].values())
        reference.load_state_dict(copy.deepcopy(saved))
    grads = [torch.tensor([0.17, -0.31], dtype=dtype, device=device),
             torch.tensor([-0.08, 0.42], dtype=dtype, device=device)]
    before = [p.detach().clone() for p in q]
    frozen = copy.deepcopy(saved["param_groups"]) if saved is not None else None
    frozen_state = ({k: {n: v.clone() if torch.is_tensor(v) else v for n, v in s.items()}
                     for k, s in saved["state"].items()} if saved is not None else None)
    predicted = adam_delta(torch, grads, reference.param_groups, saved)
    for p, old in zip(q, before):
        assert torch.equal(p, old)
    if saved is not None:
        assert saved["param_groups"] == frozen
        for pid, state in frozen_state.items():
            for key, value in state.items():
                now = saved["state"][pid][key]
                assert torch.equal(now, value) if torch.is_tensor(value) else now == value
    for p, g in zip(q, grads):
        p.grad = g.clone()
    reference.step()
    for p, old, delta in zip(q, before, predicted):
        _close(p.detach() - old, delta, dtype)


def selfcheck():
    _angle_checks()
    for dtype in (torch.float64, torch.float32):
        _adam_case(dtype, 0)
        _adam_case(dtype, 79)
    if torch.cuda.is_available():
        _adam_case(torch.float32, 79, device="cuda")
        return "passed"
    print("DIRECTION_SELFCHECK_CUDA_SKIPPED")
    return "skipped"


def main():
    cuda = selfcheck()
    artifacts = os.environ.get("ARTIFACTS")
    if artifacts:
        os.makedirs(artifacts, exist_ok=True)
        result = {"status": "passed", "cuda": cuda, "pbs_jobid": os.environ.get("PBS_JOBID")}
        with open(os.path.join(artifacts, "direction_selfcheck.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print("DIRECTION_SELFCHECK_PASSED")


if __name__ == "__main__":
    main()
