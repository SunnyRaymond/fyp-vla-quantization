"""Independent CPU verification for a single frozen broadcast screen."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

PINS = {
    'producer':'a82cf47e7ec88a1747b01244cab386fab1cff60e95ad9344a17c2bc0d0419a77',
    'protocol':'52d57c928973ac598d27b894ba124d161f51613da1a20ed74540bebccb09bb9a',
    'freeze':'233bd7723478fb6deb99216b51d8cd15a0dd71a2776c073465f4a0f9ef1185a2',
    'teacher_helper':'6e475ff4c75dda269b6d916ac28f8d63879abe6aae6427da6955bb35d36ed559',
    'smoke_runner.py':'de5c5eb19f4b26614e71f9e2af36db21e9fd5bcc65188f3191772552bc750af9',
    'screen_runner.py':'51c2463a92a3bf84eabae735eeaa9771a7202add0b2227fe06c7fe1f2bd69d19',
}


def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def run(args,allocation):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    job = args.input
    checks = {}
    result = {'schema':'broadcast-coupling-verification-v1','allocation':allocation,
              'producer_job':job.name,'checks':checks,'decision':'implementation_inconclusive'}
    eng = read(job/'engineering.json')
    checks['producer_complete'] = eng['status']=='completed'
    if not checks['producer_complete']:
        result['producer_error'] = eng.get('error')
        if 'Timeout' in eng.get('error','') or (job/'exit_status.json').exists() and read(job/'exit_status.json')['exit_code']==124:
            result['decision'] = 'inconclusive_budget'
        return result
    for name,receipt in eng['receipts'].items():
        path = Path(receipt['path'])
        checks['receipt_'+name] = path.parent==job and sha(path)==receipt['sha256']
        if name in PINS:
            checks['pin_'+name] = receipt['sha256']==PINS[name]
    checks['required_receipts'] = set(PINS).union({'runtime','raw'}).issubset(eng['receipts'])
    required = ('teacher_helper_pin','input_manifest_pin','protocol_pin','v100','v100_capability',
        'eval_frozen','trajectories','encoded_shape','broadcast_exact','predictor_input_readback',
        'prediction_shape','rtn_before_after_input','all_model_state_unchanged','all_completed','hook_all60')
    checks['producer_gates'] = all(eng['checks'].get(k) is True for k in required)
    checks['hook_all60'] = eng['hook_checks']==[True]*60
    checks['rtn_inputs'] = eng['rtn_input_equal']==[True]*6
    checks['state_unchanged'] = eng['state_before']==eng['state_after'] and len(eng['state_before'])==64
    a = eng['allocation']
    checks['allocation'] = a.get('verified') is True and a.get('scheduler')=='slurm' and str(a.get('job_id'))==job.name and a.get('partition')=='UGGPU-TC1' and a.get('hostname','').lower().startswith('tc1n')
    checks['v100'] = 'V100' in eng['gpu']['name'] and eng['gpu']['compute_capability']==[7,0]
    checks['budget'] = eng['work_seconds']<=270
    manifest_path = Path(eng['input_manifest']['path'])
    expected_manifest = '6dae677a0e7763896d735495500aea78a9254e59b67fa47817bb9bb9bb1f5c14'
    checks['input_manifest'] = sha(manifest_path)==expected_manifest==eng['input_manifest']['sha256']
    manifest = read(manifest_path)
    checks['sample_hashes'] = eng['input_manifest']['sample_sha256']==[s['sha256'] for s in manifest['samples']]
    runtime = read(job/'runtime.json')
    identity = runtime['identity']
    checks['checkpoint'] = identity['checkpoint_sha256']=='8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b'
    checks['source_commit'] = identity['source_commit']=='0a9492fa12044b852ae9e001cc74604b79c8bb0c'
    checks['dinov2_commit'] = identity['dinov2_source_commit']=='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
    binding = identity['prepared_runtime_binding']
    checks['prepared_binding'] = binding['prepared_source_config_checkpoint_verified'] is True and binding['checkpoint_config_sha256']==manifest['checkpoint']['config']['sha256']
    for name,digest in identity['source_file_sha256'].items():
        if name == 'preprocessor.py':
            checks['source_'+name] = digest=='c5ad3949628bec91ecdc6f8e011d4cc9f0c8cc9231f2291a77e5daf7e4fd27f3'
            continue
        # The frozen CPU manifest contains these exact executable sources.
        records = [v for v in manifest['source_identity']['files'].values() if v['path'].endswith('/'+name)]
        checks['source_'+name] = len(records)==1 and records[0]['sha256']==digest
    with np.load(job/'raw.npz',allow_pickle=False) as raw:
        def gen(seed):
            return torch.Generator(device='cpu').manual_seed(seed)
        expected_uniform = torch.stack([torch.rand((20,),generator=gen(s),dtype=torch.float32) for s in (2501,2502,2503)]).numpy()
        expected_offset = ((torch.arange(196)%3)[torch.randperm(196,generator=gen(2601))]).numpy()
        checks['uniform_rng'] = np.array_equal(raw['uniform'],expected_uniform)
        checks['offset_rng'] = np.array_equal(raw['offset'],expected_offset)
        spec = importlib.util.spec_from_file_location('independent_replay',args.replay)
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        science = replay.replay(raw)
    result['scientific_replay'] = science
    result['raw_sha256'] = sha(job/'raw.npz')
    result['replay_sha256'] = sha(args.replay)
    result['work_seconds'] = eng['work_seconds']
    if all(checks.values()):
        result['decision'] = science['decision']
    return result


if __name__ == '__main__':
    from allocation_guard import require_allocation
    allocation = require_allocation()
    p = argparse.ArgumentParser()
    for name in ('input','output','replay'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    result = run(args,allocation)
    (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'decision':result['decision'],'failed_checks':[k for k,v in result['checks'].items() if not v]}))
