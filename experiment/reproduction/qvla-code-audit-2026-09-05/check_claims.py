"""CPU-only checks of the unmodified public QVLA functions; no model download."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "QVLA" / "openvla" / "qvla"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SRC / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    sens = load("sensitivity_hessian_proxy")
    gates = load("assign_gates_from_sensitivity")
    inject = load("inject_fake_w")
    results = {"torch": torch.__version__, "device": "cpu"}

    # Exact output wrapper used by allocation main() cannot be read by injection.
    proxies = {"language_model.test": {8: torch.tensor([1., 2.]), 4: torch.tensor([3., 4.])}}
    allocation, stats = gates.greedy_allocate(proxies, [4, 8, 16], 8.)
    payload = {"proxy_pt": "out/proxy.pt", "bits": [4, 8, 16], "assign": allocation, "stats": stats}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gates.json"
        path.write_text(json.dumps(payload))
        try:
            inject._load_gates(str(path), torch.device("cpu"))
        except (TypeError, ValueError, RuntimeError) as exc:
            results["gate_schema"] = {"fails": True, "error": str(exc)}
        else:
            raise AssertionError("Expected published schema mismatch")
        path.write_text(json.dumps(allocation))
        assert len(inject._load_gates(str(path), torch.device("cpu"))) == 1

    # Counterexample for Eq. 8, retaining Appendix B's heap scheduling.
    # Both rows first reach 8 bits. At 8->4, incremental costs are 2/4 vs 50/4,
    # but the release uses absolute costs 102/4 vs 60/4 and chooses the other row.
    p = {"layer": {8: torch.tensor([100., 10.]), 4: torch.tensor([102., 60.]), 16: torch.zeros(2)}}
    actual, _ = gates.greedy_allocate(p, [4, 8, 16], 6.)
    assert actual["layer"] == [8, 4]
    paper_next = min(range(2), key=lambda i: float(p["layer"][4][i] - p["layer"][8][i]) / 4)
    assert paper_next == 0
    results["allocation_counterexample"] = {"released": actual["layer"], "equation_8": [4, 8], "same_average_bits": 6}

    # Changing only a downstream action mapping changes true sensitivity,
    # while the released local proxy cannot see this mapping.
    layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[.31, 1.]]))
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]])

    def score_and_action_error(action_gain):
        proxy = sens._HessianProxy(layer, device=torch.device("cpu"))
        hook = layer.register_forward_hook(lambda m, inputs, output: proxy.add_batch(inputs[0]))
        with torch.no_grad():
            reference = action_gain * layer(x)
        hook.remove()
        score = sens._compute_proxy_for_bits(layer, proxy.diag_hinv(), [2])[2].item()
        q = sens._quantize_row_sym(layer.weight.detach()[0], 2)
        quant_action = action_gain * (x @ q).unsqueeze(1)
        error = ((quant_action - reference) ** 2).sum(dim=1).mean().item()
        return {"proxy": score, "action_mse": error}

    one, ten = score_and_action_error(1.), score_and_action_error(10.)
    assert one["proxy"] == ten["proxy"]
    assert abs(ten["action_mse"] / one["action_mse"] - 100) < .001
    results["downstream_gain"] = {"gain_1": one, "gain_10": ten, "action_mse_ratio": ten["action_mse"] / one["action_mse"]}

    # Fake quant retains BF16 dense storage, including nominal 0-bit rows.
    m = torch.nn.Linear(4, 3, bias=True).to(torch.bfloat16)
    with torch.no_grad():
        m.weight.copy_(torch.tensor([[.1, .3, .7, 1.], [.2, .4, .8, 1.], [.3, .5, .9, 1.]]))
        m.bias.fill_(1)
    before = {"shape": list(m.weight.shape), "dtype": str(m.weight.dtype), "weight_bytes": m.weight.numel() * m.weight.element_size()}
    inject._apply_weight_only_fake_quant(m, torch.tensor([0, 2, 4]))
    after = {"shape": list(m.weight.shape), "dtype": str(m.weight.dtype), "weight_bytes": m.weight.numel() * m.weight.element_size()}
    assert before == after
    assert torch.count_nonzero(m.weight[0]) == 0
    assert m(torch.ones(1, 4, dtype=torch.bfloat16))[0, 0].item() == 1.
    results["fake_quant_storage"] = {"before": before, "after": after, "zero_bit_row_output_with_bias_1": 1.}

    missing = "run_eval_with_qvla_fakew.py"
    assert not (SRC / missing).exists()
    results["readme_eval_script_missing"] = missing
    for backend in ["openvla-oft", "UniVLA"]:
        for f in SRC.glob("*.py"):
            assert f.read_bytes() == (ROOT / "QVLA" / backend / "qvla" / f.name).read_bytes()
    results["backend_scripts"] = "Four QVLA Python files identical across all three backends"
    (ROOT / "check-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
