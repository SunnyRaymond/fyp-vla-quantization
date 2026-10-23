"""Check loss values and independent analytic gradients on an allocated GPU."""
import json
import os
from pathlib import Path


def main():
    import sampler
    sampler._require_compute_allocation()
    import torch
    from gripper_loss import action_losses
    action = torch.arange(32*7,device='cuda',dtype=torch.float32).reshape(32,7).div(100).requires_grad_()
    target = torch.zeros_like(action)
    motor, gripper = action_losses(action,target,1/6)
    original = action[:10,:6].square().mean()
    torch.testing.assert_close(motor,original,rtol=0,atol=0)
    motor_gradient, = torch.autograd.grad(motor,action,retain_graph=True)
    gradient, = torch.autograd.grad(motor+gripper/6,action)
    expected = torch.zeros_like(action);expected[:10] = action[:10].detach()/30
    torch.testing.assert_close(gradient,expected,rtol=1e-6,atol=1e-8)
    torch.testing.assert_close(gradient[:10,:6],motor_gradient[:10,:6],rtol=0,atol=0)
    assert bool((gradient[:10,6]!=0).all()) and bool((gradient[10:]==0).all())
    value={'passed':True,'pbs_jobid':os.environ['PBS_JOBID'],'lambda_g':1/6,
           'motor_gradient_unchanged':True,'first10_gripper_connected':True,'late22_gradient_zero':True,
           'analytic_gradient_max_abs':float((gradient-expected).abs().max())}
    (Path(os.environ['ARTIFACTS'])/'gripper_selfcheck.json').write_text(json.dumps(value,indent=2)+'\n')
    print('GRIPPER_LOSS_SELFCHECK_PASSED',flush=True)


if __name__=='__main__':main()
