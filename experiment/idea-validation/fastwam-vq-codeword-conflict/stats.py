"""Statistics after summing each vector's gradients over ALL sampler calls."""

FIELDS = ('population', 'sample_count', 'latent_active_count', 'latent_cancellation',
          'latent_mean_pair_cosine', 'latent_shared_harm_fraction', 'joint_shared_harm_fraction',
          'exact_action_latent_cosine', 'exact_latent_gradient_sq', 'exact_action_gradient_sq',
          'exact_action_step_latent_derivative', 'exact_joint_step_latent_derivative',
          'random_group_cancellation', 'random_group_pair_cosine', 'sample_latent_gradient_sq')


def cancellation(torch, u, valid):
    norm = u.norm(dim=-1)
    active = valid & (norm > 0)
    count = active.sum(-1)
    summed = (u * valid[..., None]).sum(1)
    total = (norm * valid).sum(1)
    ratio = 1 - summed.norm(dim=-1) / total.clamp_min(1e-30)
    ratio = torch.where(total > 0, ratio.clamp(0, 1), 0)
    normalized = u / norm.clamp_min(1e-30)[..., None] * active[..., None]
    pair = (normalized.sum(1).square().sum(-1) - count) / (count * (count - 1)).clamp_min(1)
    return ratio, torch.where(count > 1, pair.clamp(-1, 1), 0), count


def summarize(torch, ga, gz, plan, row_scales, exact_a, exact_z, weight, seed, null_repeats):
    tables = []
    for book, part in enumerate(plan['books']):
        pick = part['positions'].long()
        valid = pick >= 0
        selected = pick.clamp_min(0)
        rows = plan['flat_ids'][selected] // plan['vectors_per_row']
        scales = row_scales[rows]
        a, z = ga[selected] * scales[..., None], gz[selected] * scales[..., None]
        a, z = a * valid[..., None], z * valid[..., None]
        ea, ez = exact_a[book].float(), exact_z[book].float()
        cz, pz, nz = cancellation(torch, z, valid)
        normz = z.norm(dim=-1)
        active = valid & (normz > 0)
        harms = ((z * ez[:, None]).sum(-1) < 0) & active
        joint = ea + weight * ez
        joint_harms = ((z * joint[:, None]).sum(-1) < 0) & active
        za, zz = ea.square().sum(-1), ez.square().sum(-1)
        cosine = (ea * ez).sum(-1) / (za * zz).sqrt().clamp_min(1e-30)
        generator = torch.Generator(device='cpu').manual_seed(seed + book)
        flat = z[valid]
        null_c, null_p = torch.zeros_like(cz), torch.zeros_like(pz)
        for _ in range(null_repeats):
            shuffled = torch.zeros_like(z)
            shuffled[valid] = flat[torch.randperm(len(flat), generator=generator)]
            c, q, _ = cancellation(torch, shuffled, valid)
            null_c += c / null_repeats
            null_p += q / null_repeats
        columns = (part['population'].float(), valid.sum(-1).float(), nz.float(), cz, pz,
                   harms.sum(-1) / nz.clamp_min(1), joint_harms.sum(-1) / nz.clamp_min(1),
                   cosine.clamp(-1, 1), zz, za, -(ea * ez).sum(-1), -(joint * ez).sum(-1),
                   null_c, null_p, (z.square().sum(-1) * valid).sum(-1))
        table = torch.stack(columns, -1)
        if table.shape != (256, len(FIELDS)) or not bool(torch.isfinite(table).all()):
            raise RuntimeError('Invalid codeword statistics')
        tables.append(table)
    return torch.stack(tables)
