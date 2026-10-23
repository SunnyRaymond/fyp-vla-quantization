"""Small independent numerical checks, executed only inside the GPU allocation."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import smooth_vq
from rotation_bank import A4, Rotations, transformed


def run():
    import sampler
    sampler._require_compute_allocation()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261021)
    report = {'passed': False, 'fixtures': []}
    for width in (7, 128, 257):
        mod = torch.nn.Linear(width, 9, device='cuda', dtype=torch.bfloat16)
        mod.weight.requires_grad_(False)
        mod.bias.requires_grad_(False)
        modules = {'video_expert.fixture': mod}
        rotations = Rotations(modules)
        transform = smooth_vq.make_transform(torch.rand(width, device='cuda') + .1,
                    mod.weight, alpha=.5, hadamard=True, block=128, seed=43)
        x = (torch.randn(3, 2, width, device='cuda') * 2).to(torch.bfloat16)
        z = transformed(x, transform, rotations, 'video')
        w = transformed(mod.weight, transform, rotations, 'video', weight=True)
        expected_z = smooth_vq.transform_input(x, transform)
        expected_w = smooth_vq.transform_weight(mod.weight, transform)
        if not torch.equal(z, expected_z) or not torch.equal(w, expected_w):
            raise AssertionError('R0 transform differs from the original signed H arithmetic')
        qa = A4.apply(z)
        expected_a4 = smooth_vq.quant_activation(expected_z, 4)[0]
        if not torch.equal(qa, expected_a4):
            raise AssertionError('R0 A4 operands changed')
        with torch.no_grad():
            if rotations.raw:
                for raw in rotations.raw.values():
                    raw.normal_(0, .003)
            rotations.refresh()
            zr = transformed(x, transform, rotations, 'video')
            wr = transformed(mod.weight, transform, rotations, 'video', weight=True)
            before = F.linear(x.float(), mod.weight.float())
            after = F.linear(zr, wr)
            error = float((before-after).abs().max())
            if not torch.allclose(before, after, atol=2e-4, rtol=2e-4):
                raise AssertionError('Paired Smooth/R does not preserve the FP32 Linear')
        if rotations.raw:
            # Verify the actual STE through both branches against ordinary autograd.
            def loss(use_checkpoint):
                rotations.refresh()
                # The same graph must be used on initial execution and checkpoint recomputation.
                rotations.matrix('video')
                def operation(value):
                    operand = A4.apply(transformed(value, transform, rotations, 'video'))
                    weight = transformed(mod.weight, transform, rotations, 'video', weight=True).to(torch.bfloat16)
                    return F.linear(operand, weight, mod.bias).float().square().mean()
                return checkpoint(operation, x, use_reentrant=False) if use_checkpoint else operation(x)
            a = loss(False)
            normal = torch.autograd.grad(a, list(rotations.parameters()))
            b = loss(True)
            checked = torch.autograd.grad(b, list(rotations.parameters()))
            if not all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in normal):
                raise AssertionError('R gradients are disconnected/zero/nonfinite')
            if not all(torch.equal(g, h) for g, h in zip(normal, checked)):
                raise AssertionError('Checkpoint changed the R gradient')
            if max(rotations.orthogonality().values()) > 5e-5:
                raise AssertionError('Cayley lost orthogonality')
        report['fixtures'].append({'width': width, 'fp32_equivalence_max_abs': error,
                                   'initial_H_A4_exact': True, 'checkpoint_gradient_exact': True})
    report.update(passed=True, paired_weight_and_input_rotation=True, smooth_fixed=True, D=1, r=1)
    return report
