"""Small CPU chain-rule and exception-rollback checks; no model/data is loaded."""
from pathlib import Path
from types import SimpleNamespace
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'fastwam-vq-dual-kl'))
from parameter_gate import (paired_vjp, projected_step, resident_weights,
                            parked_student_weights)
from control_losses import action_response_kl, latent_control_kl


def main():
    ratio = torch.nn.Parameter(torch.tensor(0.8))
    channel = torch.nn.Parameter(torch.tensor([0.15, -0.2]))
    unused = torch.nn.Parameter(torch.tensor(0.4))
    parameters = [ratio, channel, unused]
    weight = torch.linspace(-0.6, 0.7, 14).reshape(2, 7)
    clean = torch.linspace(-1.0, 1.0, 64).reshape(32, 2)
    inputs = {'clean': clean, 'noisy': clean + 0.13 * torch.cos(clean * 7)}
    teacher = {v: 0.2 * x @ weight for v, x in inputs.items()}
    stats = {'action': {'mean': [0.0] * 6, 'scale': [1.0] * 6},
             'config': {'bins': 16, 'interior_min': -3.5, 'interior_max': 3.5,
                        'sigma': 0.375, 'std_floor': 0.02}}

    def forward(variant):
        d = torch.exp2(channel)
        u = inputs[variant] / d * ratio * 3
        q = (u.round() - u).detach() + u
        future = q / (3 * ratio) * d
        return torch.sin(future) @ weight, future

    def reader(future):
        return torch.tanh(future @ weight) * 1.1

    for objective in ('response', 'latent'):
        outputs = {v: forward(v) for v in inputs}
        if objective == 'response':
            loss = action_response_kl(torch, outputs['clean'][0], outputs['noisy'][0],
                                      teacher['clean'], teacher['noisy'], stats)
        else:
            loss = sum(0.5 * latent_control_kl(torch, reader(outputs[v][1]),
                                              teacher[v], stats) for v in inputs)
        direct = torch.autograd.grad(loss, parameters, allow_unused=True)
        values = {v: outputs[v][0 if objective == 'response' else 1].detach()
                  .requires_grad_(True) for v in inputs}
        if objective == 'response':
            leaf_loss = action_response_kl(torch, values['clean'], values['noisy'],
                                           teacher['clean'], teacher['noisy'], stats)
        else:
            leaf_loss = sum(0.5 * latent_control_kl(torch, reader(values[v]),
                                                   teacher[v], stats) for v in inputs)
        vectors = torch.autograd.grad(leaf_loss, list(values.values()))
        cotangents = {v: (vector, None) if objective == 'response' else (None, vector)
                      for v, vector in zip(inputs, vectors)}
        staged = paired_vjp(torch, parameters, forward, cotangents)
        for expected, actual in zip(direct, staged):
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    originals = [p.detach().clone() for p in parameters]
    delta = [torch.ones_like(p) for p in parameters]
    for fail in (False, True):
        try:
            with projected_step(torch, parameters, originals, delta, 0.4,
                                lambda: ratio.clamp_(0.5, 1.0)) as actual:
                assert float(ratio.detach()) == 1.0
                assert float(actual[0]) > 0
                if fail:
                    raise RuntimeError('intentional evaluation failure')
        except RuntimeError:
            assert fail
        assert all(torch.equal(p.detach(), old) for p, old in zip(parameters, originals))
    masters = {'weight': torch.tensor([0.5, -0.75], dtype=torch.bfloat16)}
    for fail in (False, True):
        try:
            with resident_weights(torch, masters, 'cpu') as weights:
                assert weights['weight'].data_ptr() != masters['weight'].data_ptr()
                assert torch.equal(weights['weight'], masters['weight'])
                if fail:
                    raise RuntimeError('intentional reader failure')
        except RuntimeError:
            assert fail
        assert not weights
        assert torch.equal(masters['weight'], torch.tensor([0.5, -0.75], dtype=torch.bfloat16))
    model = torch.nn.Module()
    model.register_parameter('anchor', torch.nn.Parameter(torch.zeros(1), requires_grad=False))
    model.layer = torch.nn.Linear(2, 2, bias=False)
    model.layer.weight.requires_grad_(False)
    model.device = torch.device('cpu')
    base = model.layer.weight.detach().clone()
    switching = SimpleNamespace(runtime=SimpleNamespace(
        model=model, q=SimpleNamespace(modules={'layer': model.layer})),
        decoded_base_weights={'layer': base})
    identity = id(model.layer.weight)
    for fail in (False, True):
        try:
            with parked_student_weights(torch, switching):
                assert model.layer.weight.data_ptr() == base.data_ptr()
                if fail:
                    raise RuntimeError('intentional parked-reader failure')
        except RuntimeError:
            assert fail
        assert id(model.layer.weight) == identity
        assert torch.equal(model.layer.weight.detach(), base)
    print('Parameter gate toy checks passed: both staged gradients match joint autograd; rollback survives evaluation failure')


if __name__ == '__main__':
    main()
