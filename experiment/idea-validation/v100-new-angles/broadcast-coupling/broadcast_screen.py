"""Fixed one-step activation-broadcast diagnostic; real SLURM V100 only."""
import argparse
import hashlib
import importlib.util
import json
import time
import traceback
from pathlib import Path

TEACHER_SHA = '6e475ff4c75dda269b6d916ac28f8d63879abe6aae6427da6955bb35d36ed559'
MANIFEST_SHA = '6dae677a0e7763896d735495500aea78a9254e59b67fa47817bb9bb9bb1f5c14'


def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def run(args, allocation):
    start = time.monotonic()
    out = args.output
    eng = {'schema':'broadcast-coupling-engineering-v1','status':'running',
           'allocation':allocation,'checks':{},'receipts':{}}
    raw = None
    def check(k,v):
        eng['checks'][k] = bool(v)
        if not v:
            raise RuntimeError(k)
    def receipt(k,path):
        eng['receipts'][k] = {'path':str(path),'sha256':sha(path)}
    def deadline():
        if time.monotonic()-start > 240:
            raise TimeoutError('Frozen 240-second budget exceeded')
    try:
        check('teacher_helper_pin',sha(args.teacher_helper)==TEACHER_SHA)
        check('input_manifest_pin',sha(args.input_manifest)==MANIFEST_SHA)
        freeze = json.loads(args.freeze.read_text())
        check('protocol_pin',sha(args.protocol)==freeze['protocol_sha256'])
        for name,path in [('producer',Path(__file__)),('teacher_helper',args.teacher_helper),
                          ('protocol',args.protocol),('freeze',args.freeze)]:
            receipt(name,path)
        spec = importlib.util.spec_from_file_location('teacher_helper',args.teacher_helper)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        manifest,samples = helper._manifest_samples(args.input_manifest)
        smoke,_,helper_hashes = helper._load_helpers(args.helper_dir)
        binding = helper._bind_prepared_runtime(args.root,manifest)
        runtime = smoke._runtime(args.root)
        torch,model,device = runtime['torch'],runtime['model'],runtime['device']
        import numpy as np
        check('v100',device.type=='cuda' and 'V100' in torch.cuda.get_device_name(device))
        eng['gpu'] = {'name':torch.cuda.get_device_name(device),
                      'compute_capability':list(torch.cuda.get_device_capability(device))}
        check('v100_capability',eng['gpu']['compute_capability']==[7,0])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        check('eval_frozen',not model.training and not any(p.requires_grad for p in model.parameters()))
        _,structure = helper._runtime_structure(runtime,smoke)
        identity = helper._source_identity(args.root,runtime,smoke,helper_hashes)
        identity['prepared_runtime_binding'] = binding
        write(out/'runtime.json',{'identity':identity,'structure':structure})
        receipt('runtime',out/'runtime.json')
        for name in ('smoke_runner.py','screen_runner.py'):
            receipt(name,args.helper_dir/name)
        eng['input_manifest'] = {'path':str(args.input_manifest),'sha256':MANIFEST_SHA,
            'sample_sha256':[s['sample_sha256'] for s in samples]}
        check('trajectories', [s['underlying_trajectory_id'] for s in samples]==[1035,1534,1158,203,1837,1095])
        def state_digest():
            h = hashlib.sha256()
            for name,value in sorted(model.state_dict().items()):
                a = value.detach().cpu().contiguous().numpy()
                h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(repr(a.shape).encode()); h.update(a.tobytes())
            return h.hexdigest()
        eng['state_before'] = state_digest()
        preprocessor = smoke._preprocessor(runtime)
        def gen(seed):
            return torch.Generator(device='cpu').manual_seed(seed)
        uniform = torch.stack([torch.rand((20,),generator=gen(s),dtype=torch.float32)
                               for s in (2501,2502,2503)]).numpy()
        offset = ((torch.arange(196)%3)[torch.randperm(196,generator=gen(2601))]).numpy()
        def empty(shape):
            return np.full(shape,np.nan,dtype=np.float32)
        raw = {'initial_z':empty((6,196,404)),'base_quant':empty((6,3,20)),
            'uniform':uniform,'scale':empty((6,)),'rtn_vector':empty((6,20)),
            'seen_tail':empty((6,2,3,196,20)),'pred_visual':empty((6,2,3,196,384)),
            'fp_visual':empty((6,196,384)),'fp_copy_visual':empty((6,196,384)),
            'rtn_visual':empty((6,2,196,384)),'codes':np.zeros((6,3,20),dtype=np.int8),
            'offset':offset.astype(np.int64),'valid_indices':np.arange(124,130,dtype=np.int64),
            'trajectory_ids':np.array([1035,1534,1158,203,1837,1095],dtype=np.int64),
            'completed':np.zeros(6,dtype=bool)}
        actual_calls = []
        eng['hook_checks'] = []
        eng['rtn_input_equal'] = []
        for i,sample in enumerate(samples):
            deadline()
            obs,_ = helper._make_observations(sample,preprocessor,torch,device)
            actions = torch.as_tensor(sample['action_blocks'][:1],device=device,dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                z = model.encode(obs,actions)
                check('encoded_shape',tuple(z.shape)==(1,1,196,404))
                original = z[0,0].cpu().numpy()
                check('broadcast_exact',np.array_equal(original[:,384:],np.broadcast_to(original[0,384:],(196,20))))
                raw['initial_z'][i] = original
                vector = original[0,384:]
                maximum = np.max(np.abs(vector))
                scale = np.float32(maximum/np.float32(7)) if maximum > 0 else np.float32(1)
                normalized = np.divide(vector,scale,dtype=np.float32)
                lower = np.floor(normalized)
                codes = np.clip(lower[None,:]+(uniform < (normalized-lower)[None,:]).astype(np.float32),-7,7).astype(np.int8)
                base = codes.astype(np.float32)*scale
                rtn = np.clip(np.rint(normalized),-7,7).astype(np.float32)*scale
                raw['scale'][i] = scale
                raw['base_quant'][i] = base
                raw['codes'][i] = codes
                raw['rtn_vector'][i] = rtn
                def predict(value):
                    seen = []
                    def hook(_module,inputs):
                        seen.append(inputs[0].detach().cpu().numpy().copy())
                    handle = model.predictor.register_forward_pre_hook(hook)
                    try:
                        prediction = model.predict(value)
                    finally:
                        handle.remove()
                    match = len(seen)==1 and np.array_equal(seen[0],value[0].detach().cpu().numpy())
                    eng['hook_checks'].append(bool(match))
                    check('predictor_input_readback',match)
                    check('prediction_shape',tuple(prediction.shape)==(1,1,196,404))
                    return prediction[0,0,:,:384].cpu().numpy(),seen[0][0,:,384:]
                raw['fp_visual'][i],_ = predict(z)
                raw['fp_copy_visual'][i],_ = predict(z.clone())
                before = z.clone()
                before[0,0,:,384:] = torch.as_tensor(rtn,device=device).expand(196,20)
                after = z.clone()
                after_np = np.clip(np.rint(np.divide(original[:,384:],scale,dtype=np.float32)),-7,7).astype(np.float32)*scale
                after[0,0,:,384:] = torch.as_tensor(after_np,device=device)
                eng['rtn_input_equal'].append(bool(torch.equal(before,after)))
                check('rtn_before_after_input',all(eng['rtn_input_equal']))
                raw['rtn_visual'][i,0],_ = predict(before)
                raw['rtn_visual'][i,1],_ = predict(after)
                for arm in range(2):
                    for draw in range(3):
                        tail = np.broadcast_to(base[draw],(196,20)).copy() if arm==0 else base[(draw+offset)%3]
                        value = z.clone()
                        value[0,0,:,384:] = torch.as_tensor(tail,device=device)
                        prediction,seen_tail = predict(value)
                        raw['pred_visual'][i,arm,draw] = prediction
                        raw['seen_tail'][i,arm,draw] = seen_tail
                        deadline()
            raw['completed'][i] = True
            np.savez(out/'raw.npz',**raw)
            write(out/'engineering.json',eng)
            print(json.dumps({'state':i,'seconds':time.monotonic()-start}),flush=True)
        eng['state_after'] = state_digest()
        check('all_model_state_unchanged',eng['state_before']==eng['state_after'])
        check('all_completed',raw['completed'].all())
        check('hook_all60',len(eng['hook_checks'])==60 and all(eng['hook_checks']))
        receipt('raw',out/'raw.npz')
        eng['status'] = 'completed'
        eng['work_seconds'] = time.monotonic()-start
        eng['max_memory_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
    except Exception as exc:
        eng['status'] = 'failed'
        eng['error'] = type(exc).__name__+': '+str(exc)
        if raw is not None:
            np.savez(out/'raw.npz',**raw)
            receipt('raw',out/'raw.npz')
        traceback.print_exc()
        raise
    finally:
        write(out/'engineering.json',eng)


if __name__ == '__main__':
    from allocation_guard import require_allocation
    allocation = require_allocation()
    p = argparse.ArgumentParser()
    for name in ('root','input-manifest','teacher-helper','helper-dir','protocol','freeze','output'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args(),allocation)
