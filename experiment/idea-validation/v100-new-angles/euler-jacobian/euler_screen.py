"""One frozen SmolVLA input-Jacobian diagnostic, real SLURM GPU only."""
import argparse
import hashlib
import importlib.util
import json
import time
import traceback
from pathlib import Path

B2_SHA = '01059efb28661486931b935ad4a3a565456c043bc76985873a571bbf8c09c700'
SOURCE_SHA = {
    'lerobot_package': 'e9cf424c0656b89d285065a9344efd8149208950047f096f751db50e7d898bd7',
    'configuration_smolvla': '6c55f3dea30a3c9571ecaa3dccf599dab9240a6eaa658b3fbc507273778b49aa',
    'modeling_smolvla': '3bdbaeecbd0dd3908d08507c13ed3517e63d2a653555322e2428066efb77b5f4',
    'smolvlm_with_expert': 'b70356145870c7da1e92a2195ee4626f7e9f9387576c6bed5ba2bfefae2a38d9',
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def run(args, allocation):
    started = time.monotonic()
    out = args.output
    eng = {'schema': 'euler-jacobian-engineering-v1', 'status': 'running',
           'allocation': allocation, 'checks': {}, 'receipts': {}}
    checks = eng['checks']
    raw = None

    def deadline():
        if time.monotonic()-started > 480:
            raise TimeoutError('Frozen 480-second workload limit')

    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise RuntimeError(name)

    def receipt(name, path):
        eng['receipts'][name] = {'path': str(path), 'sha256': sha(path)}

    try:
        check('helper_pin', sha(args.persistence_helper) == B2_SHA)
        frozen = json.loads(args.freeze.read_text())
        check('protocol_pin', sha(args.protocol) == frozen['protocol_sha256'])
        receipt('protocol', args.protocol)
        receipt('freeze', args.freeze)
        receipt('producer', Path(__file__))
        receipt('persistence_helper', args.persistence_helper)
        receipt('flow_helper', args.flow_helper)
        helper = module(args.persistence_helper, 'persistence_helper')
        manifest = helper._load_manifest(args.input_manifest)
        check('input_pin', manifest['sha256'] == frozen['input_manifest_sha256'])
        eng['input_manifest'] = {'path': str(args.input_manifest), 'sha256': manifest['sha256'],
            'sample_sha256': [e['sha256'] for e in manifest['entries']]}
        flow = helper._load_flow_helper(args.flow_helper)
        import numpy as np
        import torch
        torch.set_num_threads(4)
        policy, runtime = helper._load_runtime(flow, args, torch, manifest['identity'])
        write(out/'runtime.json', runtime)
        receipt('runtime', out/'runtime.json')
        eng['gpu'] = flow._gpu_evidence(torch)
        eng['gpu']['compute_capability'] = list(torch.cuda.get_device_capability())
        check('v100', eng['gpu']['compute_capability'] == [7, 0])
        check('source_pin', all(runtime['source_identity'][k]['sha256'] == v for k, v in SOURCE_SHA.items()))
        check('strict_checkpoint', runtime['checkpoint']['sha256'] == '9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8')
        check('runtime', not policy.training and runtime['dtype'] == 'float32' and
              runtime['compile_model'] is False and runtime['rtc_config'] is None)
        for p in policy.parameters():
            p.requires_grad_(False)
        check('parameter_grads_disabled', not any(p.requires_grad for p in policy.parameters()))
        modules, aliases, alias_report = helper._module_bindings(flow, policy, torch)
        check('expert112', len(modules) == 112)
        write(out/'module_aliases.json', alias_report)
        receipt('module_aliases', out/'module_aliases.json')
        fp = {n: m.weight.detach().clone() for n, m in modules}
        nonexpert_before = flow._state_subset_digest(policy, aliases, torch)
        eng['nonexpert_before'] = nonexpert_before
        q, snapshot = {}, {}
        with torch.no_grad():
            for n, m in modules:
                maxima = fp[n].abs().amax(dim=1, keepdim=True)
                scale = torch.where(maxima > 0, maxima*(1.0/7.0), torch.ones_like(maxima))
                codes = torch.round(fp[n]/scale).clamp(-7, 7)
                q[n] = codes*scale
                snapshot[n] = {'fp': fp[n].cpu(), 'scale': scale.cpu(),
                               'codes': codes.to(torch.int8).cpu(), 'dequant': q[n].cpu()}
        helper._apply_map(modules, q, torch)
        for n, m in modules:
            snapshot[n]['readback_q'] = m.weight.detach().cpu().clone()
        helper._restore_fp(modules, fp, torch)
        for n, m in modules:
            snapshot[n]['readback_fp'] = m.weight.detach().cpu().clone()
        torch.save(snapshot, out/'quant_snapshot.pt')
        receipt('quant_snapshot', out/'quant_snapshot.pt')
        del snapshot
        check('quant_readback_restore', True)
        def rng(seed):
            return torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn((1,50,32), generator=rng(2301), dtype=torch.float32)
        directions = (torch.randint(0,2,(2,32), generator=rng(2401)).float()*2-1)/(32**.5)
        tail = .1*torch.randn((49,32), generator=rng(2402), dtype=torch.float32)
        def empty(shape):
            return np.full(shape, np.nan, dtype=np.float32)
        raw = dict(noise=noise.numpy(), directions=directions.numpy(), tail_delta=tail.numpy(),
            x_fp=empty((6,50,32)), v=empty((6,2,50,32)), jac=empty((6,2,32,50,32)),
            fd_values=empty((6,2,2,2,32)), tail_probe_v=empty((6,2,32)),
            fp_noop_velocity=empty((6,50,32)), fp_path=empty((6,6,50,32)),
            fp_path_v=empty((6,5,50,32)), completed=np.zeros((6,2), dtype=bool),
            episode_ids=np.array([105,52,84,66,7,8], dtype=np.int64),
            timestep=np.array(.5,dtype=np.float32), dt=np.array(-.1,dtype=np.float32),
            fd_epsilon=np.array(.002,dtype=np.float32))
        check('episode_binding', [int(e['episode_index']) for e in manifest['entries']] == raw['episode_ids'].tolist())
        preprocessor, prep_identity = flow._load_preprocessor(args.model_path, args.vlm_path)
        eng['preprocessor'] = prep_identity
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        model = policy.model
        check('prefix_cache_enabled', model.config.use_cache is True)
        eng['cache_checks'] = []
        eng['input_receipts'] = []
        for i, entry in enumerate(manifest['entries']):
            deadline()
            helper._restore_fp(modules, fp, torch)
            sample = flow._load_raw_sample(Path(entry['sample_path']), entry, torch)
            batch, prep_receipt = flow._prepare_batch(sample, preprocessor, torch)
            eng['input_receipts'].append(prep_receipt)
            batch = flow._move_to_device(batch, torch.device('cuda:0'), torch)
            with torch.no_grad():
                images, masks = policy.prepare_images(batch)
                state = policy.prepare_state(batch)
                embeddings, pad, att = model.embed_prefix(images, masks,
                    batch['observation.language.tokens'], batch['observation.language.attention_mask'], state=state)
                _, cache = model.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad, att),
                    position_ids=torch.cumsum(pad, dim=1)-1, past_key_values=None,
                    inputs_embeds=[embeddings, None], use_cache=model.config.use_cache, fill_kv_cache=True)
            digest = helper._nested_tensor_digest(cache, torch)
            def velocity(x, t=.5):
                return model.denoise_step(x_t=x, prefix_pad_masks=pad, past_key_values=cache,
                    timestep=torch.tensor([t], device=x.device, dtype=torch.float32))
            with torch.no_grad():
                x = noise.cuda().clone()
                raw['fp_path'][i,0] = x.cpu().numpy()[0]
                for step in range(5):
                    v = velocity(x, 1-.1*step)
                    raw['fp_path_v'][i,step] = v.cpu().numpy()[0]
                    x = x + (-.1)*v
                    raw['fp_path'][i,step+1] = x.cpu().numpy()[0]
                raw['x_fp'][i] = x.cpu().numpy()[0]
                raw['fp_noop_velocity'][i] = velocity(x).cpu().numpy()[0]
            for arm in range(2):
                deadline()
                helper._apply_map(modules, fp if arm == 0 else q, torch)
                leaf = x.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    v = velocity(leaf)
                    raw['v'][i,arm] = v.detach().cpu().numpy()[0]
                    for j in range(32):
                        grad, = torch.autograd.grad(v[0,0,j], leaf, retain_graph=j<31, create_graph=False)
                        raw['jac'][i,arm,j] = grad.detach().cpu().numpy()[0]
                        deadline()
                del v, grad, leaf
                with torch.no_grad():
                    for d in range(2):
                        delta = torch.zeros_like(x)
                        delta[0,0] = directions[d].cuda()*.002
                        raw['fd_values'][i,arm,d,0] = velocity(x+delta)[0,0].cpu().numpy()
                        raw['fd_values'][i,arm,d,1] = velocity(x-delta)[0,0].cpu().numpy()
                    perturbed = x.clone()
                    perturbed[0,1:] += tail.cuda()
                    raw['tail_probe_v'][i,arm] = velocity(perturbed)[0,0].cpu().numpy()
                cache_same = helper._nested_tensor_digest(cache, torch) == digest
                eng['cache_checks'].append(cache_same)
                check('cache_unchanged', all(eng['cache_checks']))
                raw['completed'][i,arm] = True
                np.savez(out/'raw.npz', **raw)
                write(out/'engineering.json', eng)
                print(json.dumps({'state':i,'arm':arm,'seconds':time.monotonic()-started}), flush=True)
            del cache, embeddings, batch, images, state, x
        helper._restore_fp(modules, fp, torch)
        check('final_restore', all(torch.equal(m.weight,fp[n]) for n,m in modules))
        eng['nonexpert_after'] = flow._state_subset_digest(policy, aliases, torch)
        check('nonexpert_unchanged', eng['nonexpert_after'] == nonexpert_before)
        check('all_completed', raw['completed'].all())
        receipt('raw', out/'raw.npz')
        eng['status'] = 'completed'
        eng['work_seconds'] = time.monotonic()-started
        eng['max_memory_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
    except Exception as exc:
        eng['status'] = 'failed'
        eng['error'] = type(exc).__name__+': '+str(exc)
        if raw is not None:
            np.savez(out/'raw.npz', **raw)
            receipt('raw', out/'raw.npz')
        traceback.print_exc()
        raise
    finally:
        write(out/'engineering.json', eng)


if __name__ == '__main__':
    from allocation_guard import require_allocation
    allocation = require_allocation()
    p = argparse.ArgumentParser()
    for name in ['input-manifest','model-path','checkpoint','vlm-path','flow-helper','persistence-helper','protocol','freeze','output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--lerobot-source', type=Path, default=None)
    run(p.parse_args(), allocation)
