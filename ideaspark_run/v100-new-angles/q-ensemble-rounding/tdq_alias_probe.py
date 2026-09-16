"""CPU-only state serialization versus actual TensorDict storage diagnostic."""
import json
import os
from pathlib import Path
import sys

def main():
    from allocation_guard import require_allocation
    allocation = require_allocation()
    top = Path('/tc1home/UG/yguo017/v100_newangles_ccds')
    out = top/'artifacts'/os.environ['SLURM_JOB_ID']
    source = top/'tdmpc2_q_coupling_ready5/source'
    sys.path.insert(0,str(source/'tdmpc2'))
    import torch
    import tdq_screen as h
    from common.world_model import WorldModel
    from common.layers import api_model_conversion
    checkpoint = top/'tdq_compatible_checkpoint/cartpole-balance-3.pt'
    if h._sha256_file(checkpoint) != h.CHECKPOINT_SHA256:
        raise ValueError('Checkpoint hash mismatch')
    cfg, _ = h._resolve_config(source,source/'tdmpc2/config.yaml',checkpoint,out)
    model = WorldModel(cfg).to('cpu')
    def inspect(stage):
        state = model.state_dict()
        rows = []
        for layer, leaf in [('0','bias'),('0','weight'),('2','weight')]:
            suffix = layer+'.'+leaf
            actual = [model._Qs.params[layer,leaf],model._detach_Qs_params[layer,leaf],model._target_Qs_params[layer,leaf]]
            serialized = [state[p+suffix] for p in (h.Q_PREFIX,h.DETACH_PREFIX,h.TARGET_PREFIX)]
            rows.append({'suffix':suffix,'actual_ptrs':[h._storage_token(x) for x in actual],'state_ptrs':[h._storage_token(x) for x in serialized], 'actual_live_detach_equal':torch.equal(actual[0],actual[1]),'state_live_actual_equal':torch.equal(serialized[0],actual[0])})
        return {'stage':stage,'rows':rows}
    report = {'allocation':allocation,'inference':False,'probes':[inspect('before_load')]}
    state = h._unwrap_checkpoint(torch.load(checkpoint,map_location='cpu',weights_only=False),torch)
    model.load_state_dict(api_model_conversion(model.state_dict(),state),strict=True)
    report['probes'].append(inspect('after_load'))
    (out/'alias_probe.json').write_text(json.dumps(report,indent=2))
    print('complete')

if __name__ == '__main__':
    main()
