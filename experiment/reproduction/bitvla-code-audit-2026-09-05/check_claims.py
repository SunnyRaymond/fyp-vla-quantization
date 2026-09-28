"""CPU checks of exact quantizer definitions extracted from the public source AST.
No checkpoint, GPU, Transformers installation or remote code execution needed.
"""
import ast
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "BitVLA"
MODELS = REPO / "transformers/src/transformers/models"
NAMES = {"WeightQuant", "ActQuant", "absmean", "quantize_to_int2", "dequantize_from_int2", "BitLinear"}


def load_defs(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in NAMES]
    assert {n.name for n in nodes} == NAMES
    scope = {"torch": torch, "nn": nn, "F": F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def main():
    result = {"torch": torch.__version__, "scope": "isolated exact source definitions, CPU synthetic inputs"}
    for label, path in [("vision", MODELS / "siglip/modeling_siglip.py"), ("language", MODELS / "llava/modeling_bitnet.py")]:
        d = load_defs(path)
        w = torch.tensor([[-1., -.1, .1, 1.], [-.9, -.01, .01, .9]])
        alpha = w.abs().mean()
        qw = d["WeightQuant"].apply(w)
        assert torch.allclose(qw, (w / alpha).round().clamp(-1, 1) * alpha)
        assert set((qw / alpha).round().flatten().tolist()) == {-1., 0., 1.}
        x = torch.tensor([[.1, .5, -.7, 2.], [-2., .4, .8, .1]])
        beta = x.abs().amax(dim=-1, keepdim=True)
        qa = d["ActQuant"].apply(x)
        assert torch.allclose(qa, (127*x/beta).round().clamp(-128,127)*beta/127)
        for name in ["WeightQuant", "ActQuant"]:
            v = x.clone().requires_grad_()
            d[name].apply(v).sum().backward()
            assert torch.equal(v.grad, torch.ones_like(v))
        for n in [1, 4, 7, 32]:
            v = torch.linspace(-1, 1, n)
            packed, step, shape, count = d["quantize_to_int2"](v)
            restored = d["dequantize_from_int2"](packed, step, shape, count)
            assert packed.numel() == (n+3)//4
            assert torch.allclose(restored, d["absmean"](v))
        kwargs = {"weight_bits": 1, "input_bits": 8} if label == "vision" else {}
        layer = d["BitLinear"](4, 2, bias=False, **kwargs).to(torch.bfloat16)
        with torch.no_grad():
            layer.weight.copy_(w)
        xb = x.to(torch.bfloat16)
        out = layer(xb)
        expected = F.linear(d["ActQuant"].apply(xb), d["WeightQuant"].apply(layer.weight))
        assert torch.equal(out, expected)
        assert layer.weight.dtype == torch.bfloat16
        before_bytes = layer.weight.numel()*layer.weight.element_size()
        layer.quantize_weights()
        assert layer.weight is None
        after = layer(xb)
        assert torch.equal(out, after)
        result[label] = {"ternary_weights": True, "int8_activation_grid": True, "identity_STE_gradients": True,
                         "packing_roundtrip": "pass including padding", "default_weight_storage_bytes": before_bytes,
                         "optional_packed_weight_bytes": layer.q_weight.numel()*layer.q_weight.element_size(),
                         "optional_scale_bytes": layer.w_step.numel()*layer.w_step.element_size(),
                         "packed_forward_matches_default": True, "matmul": "floating F.linear after dequantization"}
    script = REPO / "openvla-oft/ft_script/ft_bitvla_libero_spatial.sh"
    target = script.read_text().split()[6]
    assert target == "../vla-scripts/finetune_bitnet.py"
    assert not (REPO / "openvla-oft" / target).exists()
    assert (script.parent / target).exists()
    result["finetune_script_cwd"] = "README openvla-oft cwd fails relative target; ft_script cwd resolves it"
    meta = json.loads((ROOT / "hf-metadata.json").read_text(encoding="utf-8-sig"))
    result["checkpoint_metadata"] = {"revision": meta["sha"], "safetensors": meta["safetensors"]}
    (ROOT / "check-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
