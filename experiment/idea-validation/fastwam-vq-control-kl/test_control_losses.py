"""Toy checks only; no real-model input, parameter, or checkpoint is loaded."""
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'fastwam-vq-dual-kl'))
from control_losses import action_response_kl, latent_control_kl


def main():
    stats = {'action': {'mean': [0.0]*6, 'scale': [1.0]*6}, 'config': {
        'bins': 16, 'interior_min': -3.5, 'interior_max': 3.5, 'sigma': .375, 'std_floor': .02}}
    teacher = torch.zeros(32, 7, requires_grad=True)
    student = torch.zeros(32, 7, requires_grad=True)
    changed = torch.cat((torch.full((10, 6), .1), torch.zeros(10, 1)), dim=1)
    perturbed = torch.cat((changed, torch.zeros(22, 7)), dim=0).requires_grad_()
    loss = action_response_kl(torch, student, perturbed, teacher, teacher, stats)
    loss.backward()
    assert float(loss.detach()) > 0 and torch.isfinite(perturbed.grad).all()
    assert float(perturbed.grad[:10, :6].abs().sum()) > 0
    assert perturbed.grad[10:].abs().sum() == 0 and perturbed.grad[:, 6].abs().sum() == 0
    assert teacher.grad is None
    zero = action_response_kl(torch, student+.2, perturbed+.2, student, perturbed, stats)
    assert abs(float(zero.detach())) < 1e-6
    assert float(latent_control_kl(torch, student+.2, teacher, stats).detach()) > 0
    assert float(latent_control_kl(torch, student, teacher, stats).detach()) == 0
    print('Control KL toy checks passed: response gradient, teacher detach, prefix scope, bias blind spot')


if __name__ == '__main__':
    main()
