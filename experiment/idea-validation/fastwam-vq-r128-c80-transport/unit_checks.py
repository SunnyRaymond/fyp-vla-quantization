"""Numerical transport gates. Import numerical libraries only after the PBS guard."""
def run(limits):
    import sampler
    sampler._require_compute_allocation()
    import torch
    import torch.nn.functional as F
    from rotation_bank import Rotations
    from transport_bank import right_blocks, relative_input_inverse

    torch.manual_seed(20261022)
    reports = []
    for width in (128, 257):
        mod = torch.nn.Linear(width, 9, device='cuda', dtype=torch.bfloat16)
        rotations = Rotations({'video_expert.fixture': mod})
        raw = rotations.raw['video']
        with torch.no_grad():
            raw.copy_(torch.randn_like(raw) * .01)
        anchor = rotations.matrix('video').detach().clone()
        x = torch.randn(3, width, device='cuda') * .1
        weight = torch.randn(9, width, device='cuda') * .1
        with torch.no_grad():
            raw.add_(torch.randn_like(raw) * .01)
        rotations.refresh()
        current = rotations.matrix('video')
        inverse = relative_input_inverse(current, anchor)
        expected = F.linear(right_blocks(x, anchor), weight)
        actual = F.linear(right_blocks(x, current), right_blocks(weight, inverse.t()))
        output_error = float((actual - expected).abs().max())
        gradient = torch.autograd.grad(actual.square().mean(), raw)[0]
        gradient_error = float(gradient.abs().max())
        if output_error > limits['unit_fp32_output_atol'] or gradient_error > limits['unit_fp32_rotation_gradient_atol']:
            raise RuntimeError('Non-initial transport output/gradient invariance failed')
        with torch.no_grad():
            identity = relative_input_inverse(current, current.detach())
        if not torch.equal(identity, torch.eye(128, device='cuda')):
            raise RuntimeError('Centered transport did not start at exact identity')
        reports.append(dict(width=width, output_max_abs=output_error, rotation_gradient_max_abs=gradient_error,
                            noninitial_anchor=True, identity_exact=True))

    # A non-invariant old path must actually be distinguishable by this fixture.
    rotations.refresh()
    current = rotations.matrix('video')
    old = F.linear(right_blocks(x, current), weight)
    old_error = float((old - expected).abs().max())
    old_gradient = float(torch.autograd.grad(old.square().mean(), raw)[0].abs().max())
    if old_error <= limits['unit_fp32_output_atol'] or old_gradient <= limits['unit_fp32_rotation_gradient_atol']:
        raise RuntimeError('Transport fixture cannot distinguish the uncompensated path')
    # Exercise the actual BF16 decoder plus per-Linear checkpoint at a non-identity anchor.
    import micro_codebook as micro
    import smooth_vq
    from c8_train_banks import DecodedWeight
    from rotation_bank import A4, transformed
    from transport_bank import TransportBanks
    name = 'video_expert.fixture'
    mod = torch.nn.Linear(128, 9, device='cuda', dtype=torch.bfloat16)
    mod.requires_grad_(False)
    rotations = Rotations({name: mod})
    with torch.no_grad():
        rotations.raw['video'].copy_(torch.randn_like(rotations.raw['video']) * .01)
    books = torch.nn.Parameter(torch.randn(2, 256, 4, device='cuda') * .01)
    logs = torch.nn.Parameter(torch.zeros(9, device='cuda'))
    indices = torch.randint(256, (9, 32, 2), device='cuda', dtype=torch.uint8)
    exponents = torch.full((9, 4), 127, device='cuda', dtype=torch.uint8)
    encoding = dict(format=micro.FORMAT, shape=(9, 128), block_size=4, scale_block=32,
        books=books.detach().to(torch.bfloat16), row_scales=torch.ones(9, device='cuda', dtype=torch.bfloat16),
        indices=indices, scale_exponents=exponents)
    mod.weight.data = micro.decode_vq(encoding)
    transform = smooth_vq.make_transform(torch.ones(128, device='cuda'), mod.weight,
                                         alpha=.5, hadamard=True, block=128, seed=43)
    bank = object.__new__(TransportBanks)
    bank.modules, bank.rotations = {name: mod}, rotations
    bank.transforms, bank.encodings = {name: transform}, {name: encoding}
    bank.books, bank.logs, bank.active = {name: books}, {name: logs}, {name: torch.ones(9, device='cuda')}
    bank.relative_cache, bank.dense_cache, bank.seen = {}, {}, set()
    bank.anchor = {'video': rotations.matrix('video').detach().clone()}
    bank.compensated, bank.activation_bits, bank.cache_evaluation_weights = True, 4, False
    bank.legacy = {}
    x = torch.randn(3, 128, device='cuda', dtype=torch.bfloat16)

    def legacy(value):
        qx = A4.apply(transformed(value, transform, rotations, 'video'))
        w = DecodedWeight.apply(books, logs, bank.active[name], indices, exponents, mod.weight)
        return F.linear(qx, w, mod.bias)

    parameters = [books, logs, rotations.raw['video']]
    old = legacy(x)
    old_grads = torch.autograd.grad(old.float().square().mean(), parameters)
    rotations.refresh(); bank.query_reset()
    # Match the nested sampler/per-Linear checkpoints and the query's cache priming order.
    for stream in rotations.raw:
        rotations.matrix(stream)
        bank.relative(stream)
    from torch.utils.checkpoint import checkpoint
    actual = checkpoint(bank.make_forward(name, mod), x, use_reentrant=False)
    new_grads = torch.autograd.grad(actual.float().square().mean(), parameters)
    if not torch.equal(old, actual) or not all(torch.equal(a, b) for a, b in zip(old_grads[:2], new_grads[:2])):
        raise RuntimeError('Compensation changed the initial BF16 forward or VQ gradients')
    if not all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in new_grads):
        raise RuntimeError('Actual checkpointed compensation gradient is disconnected/nonfinite')
    gradient_difference = float((new_grads[2] - old_grads[2]).abs().max())
    if gradient_difference <= 1e-8:
        raise RuntimeError('Weight branch did not alter the R gradient')
    return dict(passed=True, complete=True, fixtures=reports, old_path_output_max_abs=old_error,
                old_path_rotation_gradient_max_abs=old_gradient, explicit_weight_branch=True,
                actual_checkpoint_initial_forward_exact=True, initial_VQ_gradients_exact=True,
                actual_nested_checkpoint=True,
                actual_checkpoint_all_gradients_connected=True, rotation_gradient_difference=gradient_difference)
