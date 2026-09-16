"""Independent evidence and saved-array replay; real CPU allocation only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

PINS = {
    'producer': '06b45d7343d9b99cfd40e9fce5b5e9c78642d733c6ee0f0e6eb90c7e2128b6a1',
    'protocol': '979f6b1c643ae8ce2f07d5fb1f76c4ad156ecb6eaec5e9c817e73ff089ea0f13',
    'freeze': '4a6103cd602184a020b9f0f1619b01936b7c40c29188332f2b26a0a78234a55e',
    'persistence_helper': '01059efb28661486931b935ad4a3a565456c043bc76985873a571bbf8c09c700',
    'flow_helper': 'ddf6e02ceb6b27a86ac479b54a6b24b5351342876916f39709504034efdb45a9',
}
SOURCES = {
    'lerobot_package': 'e9cf424c0656b89d285065a9344efd8149208950047f096f751db50e7d898bd7',
    'configuration_smolvla': '6c55f3dea30a3c9571ecaa3dccf599dab9240a6eaa658b3fbc507273778b49aa',
    'modeling_smolvla': '3bdbaeecbd0dd3908d08507c13ed3517e63d2a653555322e2428066efb77b5f4',
    'smolvlm_with_expert': 'b70356145870c7da1e92a2195ee4626f7e9f9387576c6bed5ba2bfefae2a38d9',
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def run(args, allocation):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    checks = {}
    result = {'schema':'euler-jacobian-verification-v1', 'allocation':allocation,
              'producer_job':args.input.name, 'checks':checks, 'decision':'implementation_inconclusive'}
    job = args.input
    eng = read(job/'engineering.json')
    checks['producer_complete'] = eng['status'] == 'completed'
    if not checks['producer_complete']:
        result['decision'] = 'inconclusive_budget' if ('Timeout' in eng.get('error','') or
            (job/'exit_status.json').exists() and read(job/'exit_status.json')['exit_code']==124) else 'implementation_inconclusive'
        result['producer_error'] = eng.get('error')
        return result
    for key, receipt in eng['receipts'].items():
        path = Path(receipt['path'])
        checks['receipt_'+key] = path.parent == job and sha(path) == receipt['sha256']
        if key in PINS:
            checks['pin_'+key] = receipt['sha256'] == PINS[key]
    checks['all_required_receipts'] = set(PINS).issubset(eng['receipts']) and all(
        k in eng['receipts'] for k in ('runtime','raw','quant_snapshot','module_aliases'))
    required = ('helper_pin','protocol_pin','input_pin','v100','source_pin','strict_checkpoint',
        'runtime','parameter_grads_disabled','expert112','quant_readback_restore','episode_binding',
        'prefix_cache_enabled','cache_unchanged','final_restore','nonexpert_unchanged','all_completed')
    checks['producer_gates'] = all(eng['checks'].get(k) is True for k in required)
    a = eng['allocation']
    checks['allocation'] = a.get('verified') is True and a.get('scheduler')=='slurm' and str(a.get('job_id'))==job.name and a.get('partition')=='UGGPU-TC1' and a.get('hostname','').lower().startswith('tc1n')
    checks['v100'] = 'V100' in eng['gpu']['name'] and eng['gpu']['compute_capability']==[7,0]
    checks['cache_all12'] = eng['cache_checks']==[True]*12
    checks['nonexpert_digest'] = eng['nonexpert_before']==eng['nonexpert_after'] and len(eng['nonexpert_before'])==64
    checks['bounded_work'] = eng['work_seconds'] <= 540
    runtime = read(job/'runtime.json')
    checks['sources'] = all(runtime['source_identity'][k]['sha256']==v for k,v in SOURCES.items())
    checks['checkpoint'] = runtime['checkpoint']['sha256']=='9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8'
    checks['runtime'] = runtime['dtype']=='float32' and runtime['compile_model'] is False and runtime['rtc_config'] is None
    manifest_path = Path(eng['input_manifest']['path'])
    checks['input_manifest'] = sha(manifest_path)=='71243c83702ada092481abb2772787ebfc01774d8b92d63c4a5812e1771a04ef' and eng['input_manifest']['sha256']==sha(manifest_path)
    payload = torch.load(job/'quant_snapshot.pt', map_location='cpu', weights_only=True)
    aliases = read(job/'module_aliases.json')
    checks['quant112'] = len(payload)==112 and set(payload)==set(aliases)
    qchecks = {'shape':True,'finite':True,'integer_range':True,'scale':True,'grid':True,'dequant':True,'readback':True,'restore':True}
    for name, entry in payload.items():
        fp = entry['fp'].numpy()
        scale = entry['scale'].numpy()
        codes = entry['codes'].numpy()
        dequant = entry['dequant'].numpy()
        qchecks['shape'] &= fp.ndim==2 and codes.shape==fp.shape==dequant.shape and scale.shape==(fp.shape[0],1)
        qchecks['finite'] &= all(np.isfinite(entry[k].numpy()).all() for k in entry)
        qchecks['integer_range'] &= codes.dtype==np.int8 and bool((np.abs(codes)<=7).all())
        maximum = np.abs(fp).max(axis=1,keepdims=True)
        expected_scale = np.where(maximum>0,maximum*np.float32(1/7),np.float32(1))
        qchecks['scale'] &= np.array_equal(scale, expected_scale)
        # CUDA divides a tensor by a tensor here. Allow only a single-code
        # tie ambiguity within float32 rounding error at exact half steps.
        normalized = fp.astype(np.float64)/scale.astype(np.float64)
        cpu_codes = np.clip(np.rint(normalized),-7,7)
        mismatch = codes != cpu_codes
        qchecks['grid'] &= bool((~mismatch | (np.abs(normalized-(np.floor(normalized)+.5))<=2e-6)).all())
        qchecks['dequant'] &= np.array_equal(dequant,codes.astype(np.float32)*scale)
        qchecks['readback'] &= np.array_equal(dequant,entry['readback_q'].numpy())
        qchecks['restore'] &= np.array_equal(fp,entry['readback_fp'].numpy())
    checks.update({'quant_'+k:bool(v) for k,v in qchecks.items()})
    del payload
    with np.load(job/'raw.npz', allow_pickle=False) as raw:
        def gen(seed):
            return torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn((1,50,32),generator=gen(2301),dtype=torch.float32).numpy()
        dirs = ((torch.randint(0,2,(2,32),generator=gen(2401)).float()*2-1)/(32**.5)).numpy()
        tail = (.1*torch.randn((49,32),generator=gen(2402),dtype=torch.float32)).numpy()
        checks['noise_seed'] = np.array_equal(raw['noise'],noise)
        checks['direction_seed'] = np.array_equal(raw['directions'],dirs)
        checks['tail_seed'] = np.array_equal(raw['tail_delta'],tail)
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
    report = run(args, allocation)
    (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'decision':report['decision'],'failed_checks':[k for k,v in report['checks'].items() if not v]}))
