"""Numerical joint-path checks run only in the approved GPU allocation."""
def run():
    import sampler
    sampler._require_compute_allocation()
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    import micro_codebook as micro
    import smooth_vq
    from c8_train_banks import A4Input, DecodedWeight
    from rotation_bank import A4, Rotations, transformed
    from rotation_selfcheck import run as rotation_checks

    report = rotation_checks()
    mod = torch.nn.Linear(128, 9, device='cuda', dtype=torch.bfloat16)
    mod.requires_grad_(False)
    rotations = Rotations({'video_expert.fixture': mod})
    transform = smooth_vq.make_transform(torch.rand(128, device='cuda') + .1,
        mod.weight, alpha=.5, hadamard=True, block=128, seed=43)
    x = torch.randn(3, 2, 128, device='cuda', dtype=torch.bfloat16)
    books = torch.nn.Parameter(torch.randn(2, 256, 4, device='cuda'))
    logs = torch.nn.Parameter(torch.zeros(9, device='cuda'))
    active = torch.ones(9, device='cuda')
    indices = torch.randint(256, (9, 32, 2), device='cuda', dtype=torch.uint8)
    exponents = torch.full((9, 4), 127, device='cuda', dtype=torch.uint8)
    encoded = dict(format=micro.FORMAT, shape=(9, 128), block_size=4, scale_block=32,
        books=books.detach().to(torch.bfloat16), row_scales=logs.detach().exp().to(torch.bfloat16),
        indices=indices, scale_exponents=exponents)
    dense = micro.decode_vq(encoded)
    parameters = [books, logs] + list(rotations.parameters())

    def operation(value, legacy=False):
        qx = A4Input.apply(value, transform) if legacy else A4.apply(transformed(value, transform, rotations, 'video'))
        weight = DecodedWeight.apply(books, logs, active, indices, exponents, dense)
        return F.linear(qx, weight, mod.bias)

    rotations.matrix('video')
    old, joint = operation(x, True), operation(x)
    if not torch.equal(old, joint):
        raise AssertionError('Joint0 changed H+VQ0 forward arithmetic')
    old_grads = torch.autograd.grad(old.float().square().mean(), [books, logs])
    joint_grads = torch.autograd.grad(joint.float().square().mean(), parameters)
    if not all(torch.equal(a, b) for a, b in zip(old_grads, joint_grads[:2])):
        raise AssertionError('Joint0 changed original VQ gradients')
    rotations.refresh(); rotations.matrix('video')
    checked = checkpoint(operation, x, use_reentrant=False)
    checked_grads = torch.autograd.grad(checked.float().square().mean(), parameters)
    if not all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in checked_grads):
        raise AssertionError('Joint gradient is disconnected/zero/nonfinite')
    if not all(torch.equal(a, b) for a, b in zip(joint_grads, checked_grads)):
        raise AssertionError('Checkpoint changed joint gradients')
    report.update(joint0_H_VQ_forward_exact=True, joint0_VQ_gradients_exact=True,
        joint_checkpoint_gradients_exact=True, joint_rotation_and_VQ_connected=True)
    return report
