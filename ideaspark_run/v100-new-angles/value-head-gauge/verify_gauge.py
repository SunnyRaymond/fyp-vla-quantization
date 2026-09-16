"""Independent CPU replay of the frozen final-Q gauge screen."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scientific_replay(data, np):
    q = data['member_q'].astype(np.float64)
    if q.shape != (8,4,64,5) or not np.isfinite(q).all():
        raise ValueError('Expected finite member_q[8,4,64,5]')
    probs = data['probabilities']
    if probs.shape != (8,4,5,64,101) or not np.isfinite(probs).all():
        raise ValueError('Expected finite probabilities[8,4,5,64,101]')
    no_op = bool(np.allclose(probs[:,1],probs[:,0],atol=1e-7,rtol=1e-6)
                 and np.allclose(q[:,1],q[:,0],atol=1e-5,rtol=1e-6))
    pairs = list(itertools.combinations(range(5),2))
    error = q-q[:,:1]
    pair_error = np.stack([(error[...,i]+error[...,j])/2 for i,j in pairs],axis=-1)
    A = np.square(pair_error).mean(axis=(2,3))
    weight = data['original_weight'].astype(np.float64)
    if weight.shape != (5,101,512) or not np.isfinite(weight).all():
        raise ValueError('Wrong original Q final weight')
    denominator = float(np.square(weight).sum())
    common_ratio = float(101*np.square(weight.mean(axis=1)).sum()/denominator) if denominator > 0 else None
    original_scale = data['original_scale'].astype(np.float64)
    centered_scale = data['centered_scale'].astype(np.float64)
    if original_scale.shape != (5,101,1) or centered_scale.shape != original_scale.shape:
        raise ValueError('Wrong row scale layout')
    if not (np.isfinite(original_scale).all() and np.isfinite(centered_scale).all()
            and (original_scale>0).all() and (centered_scale>0).all()):
        raise ValueError('Invalid row scale')
    scale_delta = np.abs(centered_scale-original_scale)/original_scale
    grid_changed = bool((scale_delta>1e-6).any() or
                        not np.array_equal(data['original_codes'],data['centered_codes']))
    global_binding = bool(common_ratio is not None and common_ratio>1e-8 and grid_changed)
    all_nondegenerate = bool((A[:,2]>1e-8).all())
    gains = 1-A[:,3]/A[:,2] if all_nondegenerate else None
    positive = int((A[:,3]<A[:,2]).sum())
    median_gain = float(np.median(gains)) if gains is not None else None
    effect = bool(all_nondegenerate and median_gain>=.25 and positive>=6 and A[:,3].mean()<A[:,2].mean())
    decision = ('implementation_inconclusive' if not no_op else
                'inconclusive_no_gauge' if not global_binding else
                'inconclusive_degenerate' if not all_nondegenerate else
                'scope_limited_preliminary_go' if effect else 'method_no_go')
    return {'numeric_decision':decision,'no_op_pass':no_op,
            'FP_probability_max_abs':float(np.max(np.abs(probs[:,1].astype(np.float64)-probs[:,0]))),
            'FP_decoded_Q_max_abs':float(np.max(np.abs(q[:,1]-q[:,0]))),
            'pooled_weight_common_mode_ratio':common_ratio,'grid_changed':grid_changed,
            'max_relative_scale_change':float(scale_delta.max()),
            'global_binding':global_binding,'all_states_nondegenerate':all_nondegenerate,
            'median_gain_all_states':median_gain,'positive_count':positive,
            'mean_A_by_arm':A.mean(axis=0).tolist(), 'unordered_pairs':pairs,
            'states':[{'reset_seed':5209+i,'A_by_arm':A[i].tolist(),
                       'gain':float(gains[i]) if gains is not None else None} for i in range(8)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    from allocation_guard import require_allocation
    allocation = require_allocation()
    import numpy as np
    import torch
    raw_path = args.input/'raw_gauge.npz'
    with np.load(raw_path,allow_pickle=False) as packed:
        data = {key:packed[key].copy() for key in packed.files}
    if data['schema'].item() != 'value-head-gauge-raw-v1':
        raise ValueError('Unexpected raw schema')
    arms = ['FP-original','FP-centered','W4-RTN-original','W4-RTN-centered']
    if data['arm_names'].tolist() != arms or data['completed'].shape!=(4,) or not data['completed'].all():
        raise ValueError('Wrong or incomplete arms')
    shapes = {'observations':(8,5),'candidate_actions':(8,64,3,1),
              'terminal_latents':(8,64,512),'terminal_actions':(8,64,1),
              'member_q':(8,4,64,5),'q_logits':(8,4,5,64,101),
              'probabilities':(8,4,5,64,101),'final_weight_original':(5,101,512),
              'final_weight_centered':(5,101,512),'final_bias_original':(5,101),
              'final_bias_centered':(5,101),'final_weight_arm_pre':(4,5,101,512),
              'final_weight_arm_post':(4,5,101,512),'final_bias_arm':(4,5,101),
              'quant_scales':(4,5,101,1),'quant_codes':(4,5,101,512),
              'weight_error':(4,5,101,512),'weight_error_quotient':(4,5,101,512)}
    for key,shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid {key} shape or values')
        expected_dtype = np.int8 if key=='quant_codes' else np.float32
        if data[key].dtype != expected_dtype:
            raise ValueError(f'Unexpected {key} dtype')
    data.update(original_weight=data['final_weight_original'],
                original_scale=data['quant_scales'][2],centered_scale=data['quant_scales'][3],
                original_codes=data['quant_codes'][2],centered_codes=data['quant_codes'][3])
    science = scientific_replay(data,np)
    checks = {}
    w,b = data['final_weight_original'],data['final_bias_original']
    wc,bc = data['final_weight_centered'],data['final_bias_centered']
    checks['centered_weight_definition'] = bool(np.allclose(wc,w.astype(np.float64)-w.astype(np.float64).mean(axis=1,keepdims=True),atol=2e-7,rtol=1e-5))
    checks['centered_bias_definition'] = bool(np.allclose(bc,b.astype(np.float64)-b.astype(np.float64).mean(axis=1,keepdims=True),atol=2e-7,rtol=1e-5))
    expected_pre = np.stack([w,wc,w,wc])
    checks['pristine_arm_inputs'] = bool(np.array_equal(data['final_weight_arm_pre'],expected_pre)
                                        and np.array_equal(data['final_bias_arm'],np.stack([b,bc,b,bc])))
    checks['FP_weights_exact'] = bool(np.array_equal(data['final_weight_arm_post'][:2],expected_pre[:2]))
    for arm in (2,3):
        pre = expected_pre[arm]
        maximum = np.max(np.abs(pre),axis=-1,keepdims=True)
        # PyTorch v2.6 CUDA CPU-scalar division uses a*reciprocal(b).
        # Match the frozen GPU recipe; CPU NumPy direct division can differ 1 ULP.
        scale = np.where(maximum==0,np.ones_like(maximum),maximum*np.float32(1.0/7.0))
        codes = np.where(maximum==0,0,np.clip(np.rint(pre/scale),-7,7)).astype(np.int8)
        checks[f'arm{arm}_exact_RTN_grid'] = bool(np.array_equal(scale,data['quant_scales'][arm])
            and np.array_equal(codes,data['quant_codes'][arm])
            and np.array_equal(codes.astype(np.float32)*scale,data['final_weight_arm_post'][arm]))
    expected_error = data['final_weight_arm_post']-expected_pre
    checks['weight_error_arrays'] = bool(np.array_equal(expected_error,data['weight_error']) and
        np.allclose(data['weight_error_quotient'],expected_error-expected_error.mean(axis=2,keepdims=True),atol=1e-8,rtol=1e-6))
    quotient = data['weight_error_quotient'].astype(np.float64)
    science['quotient_weight_error_MSE_by_arm'] = np.square(quotient).mean(axis=(1,2,3)).tolist()
    generator = torch.Generator(device='cpu').manual_seed(6301)
    replay_actions = torch.empty((8,64,3,1),dtype=torch.float32).uniform_(-1,1,generator=generator).numpy()
    checks['candidate_actions_CPU_replay'] = bool(np.array_equal(replay_actions,data['candidate_actions']))
    checks['native_action_ranges'] = bool(np.max(np.abs(data['candidate_actions']))<=1 and np.max(np.abs(data['terminal_actions']))<=1)
    logits = torch.from_numpy(data['q_logits'])
    probabilities = torch.softmax(logits,dim=-1)
    checks['logit_probability_consistency'] = bool(np.allclose(probabilities.numpy(),data['probabilities'],atol=2e-7,rtol=2e-6))
    # Independent pinned 101-bin [-10,10] symlog decoder; no model load.
    symlog_value = (probabilities*torch.linspace(-10,10,101,dtype=torch.float32)).sum(dim=-1)
    decoded = torch.sign(symlog_value)*(torch.exp(torch.abs(symlog_value))-1)
    decoded = decoded.permute(0,1,3,2).numpy()
    checks['independent_CPU_scalar_decode'] = bool(np.allclose(decoded,data['member_q'],atol=1e-5,rtol=5e-6))
    science['CPU_decode_max_abs'] = float(np.max(np.abs(decoded.astype(np.float64)-data['member_q'])))
    engineering = json.loads((args.input/'engineering.json').read_text())
    runtime = engineering.get('runtime_identity',{})
    source = engineering.get('source_identity',{})
    manifest_id = engineering.get('manifest_identity',{})
    manifest_path = Path(manifest_id['path'])
    manifest = json.loads(manifest_path.read_text())
    parent_ref = manifest['parent_manifest']
    parent = json.loads(Path(parent_ref['path']).read_text())
    expected_parent = '9240979105b919f6051f27261d00ff33652105f962d039728638caabb042d369'
    expected_checkpoint = '0e2c0eada8f160dd65c8faaa5e4ca2f8f496104e21433e334d716b73257a52c2'
    expected_source = 'e9f59321933cbc8e11a002b842adc7d4ffae8ff1'
    checks['complete_producer'] = engineering.get('status')=='complete' and engineering.get('schema')=='value-head-gauge-engineering-v1'
    checks['frozen_parent'] = parent_ref['sha256']==expected_parent and sha(parent_ref['path'])==expected_parent
    checks['checkpoint_pin'] = engineering.get('checkpoint_identity',{}).get('sha256')==expected_checkpoint and parent['checkpoint']['sha256']==expected_checkpoint
    checkpoint_path = Path(engineering['checkpoint_identity']['path'])
    checks['checkpoint_actual_file'] = checkpoint_path.stat().st_size==31344610 and sha(checkpoint_path)==expected_checkpoint
    checks['source_pin'] = source.get('commit')==expected_source and parent['source']['commit']==expected_source
    checks['source_file_identities'] = all(source['selected_files'].get(name,{}).get('sha256')==identity['sha256'] for name,identity in parent['source']['selected_files'].items())
    checks['source_actual_files'] = all(sha(Path(source['root'])/name)==identity['sha256'] for name,identity in parent['source']['selected_files'].items())
    checks['manifest_hash'] = sha(manifest_path)==manifest_id['sha256']
    checks['fresh_reset_identity'] = manifest.get('schema')=='value-head-gauge-preparation-v1' and manifest.get('seed_order')==list(range(5209,5217)) and manifest.get('inference_ran') is False and manifest.get('full_rollout') is False
    observations_path = manifest_path.parent/manifest['observations']['path']
    checks['observations_file_hash'] = sha(observations_path)==manifest['observations']['sha256']==manifest_id['observation_sha256']
    with np.load(observations_path,allow_pickle=False) as prepared:
        checks['raw_observations_match_prepared'] = bool(np.array_equal(data['observations'],prepared['observations']) and prepared['seeds'].tolist()==list(range(5209,5217)))
        input_meta = json.loads(prepared['metadata_json'].item())
    checks['reset_only_preparation'] = input_meta.get('model_loaded') is False and input_meta.get('checkpoint_loaded') is False and input_meta.get('env_steps')==0 and input_meta.get('render_calls')==0
    load = runtime.get('load',{})
    checks['strict_load'] = load.get('strict') is True and load.get('missing_keys')==[] and load.get('unexpected_keys')==[] and load.get('official_api_model_conversion') is True
    checks['FP32_eval_runtime'] = runtime.get('model_eval') is True and runtime.get('compile') is False and runtime.get('packages',{}).get('torch')=='2.6.0+cu124' and runtime.get('packages',{}).get('tensordict')=='0.7.2'
    checks['runtime_device_and_nograd'] = runtime.get('model_device_before_and_after_load')==['cuda:0','cuda:0'] and runtime.get('grad_enabled_for_screen') is False
    cfg = runtime.get('resolved_config',{})
    checks['decoder_model_configuration'] = all(cfg.get(k)==v for k,v in {'num_q':5,'num_bins':101,'vmin':-10,'vmax':10,'latent_dim':512,'action_dim':1,'task':'cartpole-balance','multitask':False}.items())
    gpu = engineering.get('gpu_identity',{})
    checks['V100'] = gpu.get('verified_v100') is True and 'V100' in gpu.get('name','') and gpu.get('compute_capability')==[7,0] and gpu.get('total_memory_bytes',0)>=30*1024**3
    for label,relative in {'common_math':'tdmpc2/common/math.py','common_layers':'tdmpc2/common/layers.py','world_model_module':'tdmpc2/common/world_model.py'}.items():
        record = runtime['source_identity'][label]
        checks[label+'_actual_import'] = record['sha256']==source['selected_files'][relative]['sha256'] and Path(record['path']).resolve()==(Path(source['root'])/relative).resolve() and sha(record['path'])==record['sha256']
    init_record = runtime['source_identity']['common_init']
    checks['common_init_actual_import'] = Path(init_record['path']).resolve()==(Path(source['root'])/'tdmpc2/common/init.py').resolve() and sha(init_record['path'])==init_record['sha256']=='3f085f2d11439312ce765bc0c4332048147d6540d7ca30cdeddacf2d70180ea4'
    checks['loaded_config_identity'] = sha(runtime['config_path'])==source['selected_files']['tdmpc2/config.yaml']['sha256']
    cache = engineering.get('fp_terminal_cache',{})
    checks['cache_recipe'] = cache.get('action_seed')==6301 and cache.get('policy_seed_by_state')==list(range(7301,7309)) and cache.get('fp_dynamics_roll_steps')==3 and cache.get('environment_steps')==0 and cache.get('policy_cache_calls')==8
    for key in ('candidate_actions','terminal_latents','terminal_actions'):
        checks[key+'_cache_hash'] = hashlib.sha256(data[key].tobytes()).hexdigest()==cache.get(key+'_sha256')
    restore = engineering.get('snapshot_restore',{})
    checks['exact_restore_and_bypass'] = all(restore.get(k) is True for k in ('final_restore_pass','target_unchanged','bypassed_unchanged'))
    checks['only_final_head_transaction'] = set(restore.get('transaction_names',[]))=={
        '_Qs.params.2.weight','_Qs.params.2.bias','_detach_Qs_params.2.weight','_detach_Qs_params.2.bias'}
    transactions = engineering.get('arm_transactions',[])
    checks['actual_parameter_transactions'] = len(transactions)==4 and [x['name'] for x in transactions]==arms and all(x.get('actual_parameter_readback_exact') is True for x in transactions)
    checks['head_storage_aliases'] = len(transactions)==4 and all(len(x['alias']['rows'])==2 and all(row['live_detach_alias'] is True and row['target_distinct'] is True for row in x['alias']['rows']) for x in transactions)
    checks['head_alias_identities'] = len(transactions)==4 and all({(row['suffix'],row['live_name'],row['detach_name'],row['target_name'],tuple(row['shape']),row['dtype']) for row in tx['alias']['rows']}=={
        (suffix,'_Qs.params.2.'+suffix,'_detach_Qs_params.2.'+suffix,'_target_Qs_params.2.'+suffix,shape,'torch.float32')
        for suffix,shape in [('weight',(5,101,512)),('bias',(5,101))]} for tx in transactions)
    checks['transaction_raw_hashes'] = len(transactions)==4 and all(row['weight_pre_sha256']==hashlib.sha256(data['final_weight_arm_pre'][i].tobytes()).hexdigest() and row['weight_post_sha256']==hashlib.sha256(data['final_weight_arm_post'][i].tobytes()).hexdigest() and row['bias_sha256']==hashlib.sha256(data['final_bias_arm'][i].tobytes()).hexdigest() for i,row in enumerate(transactions))
    frozen_hashes = {'freeze':'811e45b7094de973f798a6143dd29c1a4fba86af31fd6a24e6bb04ddd8d287ad',
                     'draft':'37acbac8b55f2497e763c7620e5d67bddd7bea96b3d57e07c06f4394f3f04fe0'}
    for prefix in ('freeze','draft'):
        protocol = engineering['protocol_identity']
        checks[prefix+'_protocol_identity'] = Path(protocol[prefix+'_path']).parent==args.input.resolve() and sha(protocol[prefix+'_path'])==protocol[prefix+'_sha256']==frozen_hashes[prefix]
    helper = engineering['helper_identity']
    checks['job_frozen_helper'] = Path(helper['path']).parent==args.input.resolve() and sha(helper['path'])==helper['sha256']=='3688d97e4c6e2bed4da4ff7aa552605690c3571e49dcf4c026e955b3c642eb11'
    recorded_ratio = engineering.get('global_binding',{}).get('pooled_weight_common_mode_ratio')
    checks['independent_global_binding_agreement'] = recorded_ratio is None if science['pooled_weight_common_mode_ratio'] is None else recorded_ratio is not None and np.isclose(recorded_ratio,science['pooled_weight_common_mode_ratio'],atol=1e-12,rtol=1e-10)
    checks = {key:bool(value) for key,value in checks.items()}
    report = {'allocation':allocation,'input_job':args.input.name,'raw_sha256':sha(raw_path),
              'engineering_sha256':sha(args.input/'engineering.json'),'checks':checks,**science,
              'decision':science['numeric_decision'] if all(checks.values()) else 'implementation_inconclusive',
              'scope':'Frozen 8 reset states, one pinned checkpoint, final-head fake W4 only; no return, novel centering, native acceleration or full validation claim.'}
    args.output.mkdir(parents=True,exist_ok=True)
    encoded = json.dumps(report,indent=2,allow_nan=False)
    if len(encoded.encode())>=64*1024:
        raise ValueError('Compact report exceeds 64KiB')
    (args.output/'verification.json').write_text(encoded)
    print(json.dumps({key:report[key] for key in ('decision','numeric_decision','global_binding','median_gain_all_states','positive_count')}))


if __name__ == '__main__':
    main()
