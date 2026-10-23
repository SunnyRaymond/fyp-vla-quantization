"""Keep the original motor mean and add the separately weighted seventh coordinate."""


def action_losses(action, target, lambda_g):
    if lambda_g != 1 / 6 or tuple(action.shape) != (32, 7) or tuple(target.shape) != (32, 7):
        raise RuntimeError('Frozen first10 motor6 plus lambda_g=1/6 gripper experiment required')
    error = action[:10].float() - target[:10].float()
    return error[:, :6].square().mean(), error[:, 6].square().mean()
