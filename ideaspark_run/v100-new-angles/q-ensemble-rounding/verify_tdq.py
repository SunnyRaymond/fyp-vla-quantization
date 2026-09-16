"""Independent all-pair scalar-Q error screen; real CPU allocation required."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from allocation_guard import require_allocation
    allocation = require_allocation()
    import numpy as np
    path = args.input/'raw_tdq.npz'
    with np.load(path, allow_pickle=False) as raw:
        if raw['schema'].item() != 'tdq-coupling-raw-v1':
            raise ValueError('Wrong schema')
        data = {key: raw[key].copy() for key in ('member_q', 'observations',
            'candidate_actions', 'terminal_latents', 'terminal_actions', 'completed',
            'seeds', 'episode_ids', 'arm_names', 'official_avg', 'official_pair', 'official_rng_state')}
    shapes = {'member_q': (8,3,4,64,5), 'observations': (8,5),
              'candidate_actions': (8,64,3,1), 'terminal_latents': (8,64,512),
              'terminal_actions': (8,64,1), 'completed': (3,4), 'seeds': (3,),
              'episode_ids': (8,), 'arm_names': (4,), 'official_avg': (8,3,4,64),
              'official_pair': (8,3,4,2)}
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f'Wrong {key} shape: {data[key].shape}')
    if data['arm_names'].tolist() != ['FP32','W4_RTN','W4_independent_SR','W4_stratified_SR']:
        raise ValueError('Wrong arm order')
    if data['seeds'].tolist() != [4101,4102,4103] or not data['completed'].all():
        raise ValueError('Wrong seeds or incomplete data')
    if data['episode_ids'].tolist() != [f'cartpole-balance:reset_seed:{seed}' for seed in range(5201,5209)]:
        raise ValueError('Wrong reset state IDs/order')
    rng = data['official_rng_state']
    if rng.ndim != 4 or rng.shape[:3] != (8,3,4) or rng.shape[-1] == 0 or rng.dtype != np.uint8:
        raise ValueError('Wrong saved CUDA RNG state layout')
    for key in ('member_q','observations','candidate_actions','terminal_latents','terminal_actions','official_avg'):
        if not np.isfinite(data[key]).all():
            raise ValueError(f'Nonfinite {key}')
    members32 = data['member_q'].astype(np.float32)
    checks = {
        'FP_repeated_exactly': bool(np.array_equal(members32[:, :, 0], np.broadcast_to(members32[:, :1, 0], (8,3,64,5)))),
        'RTN_repeated_exactly': bool(np.array_equal(members32[:, :, 1], np.broadcast_to(members32[:, :1, 1], (8,3,64,5)))),
        'actions_in_native_range': bool(np.max(np.abs(data['candidate_actions'])) <= 1),
        'terminal_actions_in_native_range': bool(np.max(np.abs(data['terminal_actions'])) <= 1),
        'same_saved_CUDA_RNG_by_state': bool(np.array_equal(rng, np.broadcast_to(rng[:, :1, :1], rng.shape))),
    }
    reconstructed = np.zeros((8,3,4,64), dtype=np.float32)
    pair_schedule_identical = True
    for ep in range(8):
        for seed in range(3):
            for arm in range(4):
                pair = data['official_pair'][ep,seed,arm]
                if pair.dtype.kind not in 'iu' or len(set(pair.tolist())) != 2 or min(pair) < 0 or max(pair) >= 5:
                    raise ValueError('Invalid official pair')
                reconstructed[ep,seed,arm] = members32[ep,seed,arm][:,pair].mean(axis=1, dtype=np.float32)
                pair_schedule_identical &= bool(np.array_equal(pair, data['official_pair'][ep,0,0]))
    official = data['official_avg'].astype(np.float32)
    checks['same_official_pair_schedule'] = pair_schedule_identical
    checks['official_avg_decode_gate'] = bool(np.allclose(reconstructed, official, atol=1e-5, rtol=1e-6))
    official_max_error = float(np.max(np.abs(reconstructed.astype(np.float64)-official.astype(np.float64))))
    members = members32.astype(np.float64)
    error = members - members[:, :, :1]
    pairs = list(itertools.combinations(range(5), 2))
    pair_errors = np.stack([(error[...,i]+error[...,j])/2 for i,j in pairs], axis=-1)
    cross_products = np.stack([error[...,i]*error[...,j] for i,j in pairs], axis=-1)
    A = np.square(pair_errors).mean(axis=(1,3,4))
    M = np.square(error).mean(axis=(1,3,4))
    C = cross_products.mean(axis=(1,3,4))
    checks['all_pair_MSE_decomposition'] = bool(np.allclose(A, (M+C)/2, atol=1e-8, rtol=1e-10))
    # Gates were frozen before any GPU output. C is an uncentered cross moment.
    records = []
    for ep in range(8):
        ai, ast, ar = map(float, (A[ep,2], A[ep,3], A[ep,1]))
        mi, ms = float(M[ep,2]), float(M[ep,3])
        nondegenerate = ai > 1e-8 and mi > 1e-8
        member_ratio = ms/mi if mi > 0 else None
        binding = bool(nondegenerate and .9 <= member_ratio <= 1.1)
        absolute_gain = ai-ast
        gain = absolute_gain/ai if ai > 0 else None
        cross_reduction = float((C[ep,2]-C[ep,3])/2)
        attribution = cross_reduction/absolute_gain if absolute_gain > 0 else None
        records.append({'episode_id': str(data['episode_ids'][ep]),
            'A': A[ep].tolist(), 'M': M[ep].tolist(), 'C_uncentered': C[ep].tolist(),
            'member_MSE_ratio_strat_ind': member_ratio, 'nondegenerate': nondegenerate,
            'binding': binding, 'gain': gain, 'absolute_gain': absolute_gain,
            'cross_term_reduction': cross_reduction, 'cross_attribution_fraction': attribution,
            'positive': bool(binding and absolute_gain > 0),
            'attribution_pass': bool(binding and absolute_gain > 0 and attribution >= .75),
            'not_worse_RTN': ast <= ar})
    bound = [row for row in records if row['binding']]
    all_nondegenerate = all(row['nondegenerate'] for row in records)
    median_gain = float(np.median([row['gain'] for row in records])) if all_nondegenerate else None
    positive = sum(row['positive'] for row in records)
    attribution_count = sum(row['attribution_pass'] for row in records)
    rtn_count = sum(row['not_worse_RTN'] for row in records)
    practical = bool(A[:,3].mean() <= A[:,1].mean() and rtn_count >= 6)
    mechanism = bool(all_nondegenerate and len(bound) >= 6 and median_gain >= .25 and attribution_count >= 6)
    numerical = ('inconclusive_degenerate' if not all_nondegenerate else
                 'inconclusive_binding' if len(bound) < 6 else
                 'preliminary_go' if mechanism and practical else
                 'mechanism_positive_practical_no_go' if mechanism else 'mechanism_no_go')
    engineering = json.loads((args.input/'engineering.json').read_text())
    runtime = engineering.get('runtime_identity', {})
    source = engineering.get('source_identity', {})
    inputs = engineering.get('input_identity', {})
    cache = engineering.get('fp_terminal_cache', {})
    restore = engineering.get('weight_restore', {})
    load = runtime.get('load', {})
    config = runtime.get('resolved_config', {})
    checkpoint_sha = '0e2c0eada8f160dd65c8faaa5e4ca2f8f496104e21433e334d716b73257a52c2'
    producer_checks = {
        'status_complete': engineering.get('status') == 'complete',
        'schema': engineering.get('schema') == 'tdq-coupling-engineering-v1',
        'strict_load': load.get('strict') is True and load.get('missing_keys') == [] and load.get('unexpected_keys') == [],
        'model_FP32_eval': runtime.get('model_eval') is True and runtime.get('compile') is False and runtime.get('grad_enabled_for_screen') is False,
        'checkpoint_pinned': engineering.get('checkpoint_identity', {}).get('sha256') == checkpoint_sha,
        'source_pinned': source.get('commit') == 'e9f59321933cbc8e11a002b842adc7d4ffae8ff1',
        'V100': engineering.get('gpu_identity', {}).get('verified_v100') is True,
        'reset_input_only': inputs.get('environment_steps') == 0 and inputs.get('render_calls') == 0 and inputs.get('model_loaded_in_preparation') is False,
        'restore_exact': restore.get('exact_elementwise') is True and restore.get('target_unchanged') is True and restore.get('bypassed_unchanged') is True and restore.get('final_full_digest') == engineering.get('transaction', {}).get('snapshot_digest'),
        'FP_cache_contract': cache.get('action_seed') == 6201 and cache.get('policy_seed_by_state') == list(range(7201,7209)) and cache.get('fp_dynamics_roll_steps') == 3 and cache.get('environment_steps') == 0 and cache.get('policy_cache_calls') == 8,
        'model_config': all(config.get(key) == expected for key, expected in {'task':'cartpole-balance','obs':'state','model_size':5,'num_q':5,'latent_dim':512,'action_dim':1,'num_bins':101,'compile':False,'multitask':False}.items()),
    }
    for key in ('candidate_actions','terminal_latents','terminal_actions'):
        producer_checks[key+'_cache_hash'] = hashlib.sha256(data[key].tobytes()).hexdigest() == cache.get(key+'_sha256')
    for label, relative in {'common_math':'tdmpc2/common/math.py','common_layers':'tdmpc2/common/layers.py','world_model_module':'tdmpc2/common/world_model.py'}.items():
        expected = source.get('selected_files', {}).get(relative, {}).get('sha256')
        actual = runtime.get('source_identity', {}).get(label, {}).get('sha256')
        producer_checks[label+'_loaded_source'] = bool(expected and actual == expected)
    init_record = runtime.get('source_identity', {}).get('common_init', {})
    init_path = Path(init_record['path']).resolve()
    producer_checks['common_init_within_pinned_tree_and_hash'] = Path(source['root']).resolve() in init_path.parents and sha(init_path) == init_record.get('sha256')
    producer_checks['loaded_config_file_identity'] = sha(runtime['config_path']) == source.get('selected_files', {}).get('tdmpc2/config.yaml', {}).get('sha256')
    producer_checks['runtime_versions'] = runtime.get('packages', {}).get('torch') == '2.6.0+cu124' and runtime.get('packages', {}).get('tensordict') == '0.7.2' and runtime.get('packages', {}).get('omegaconf') == '2.3.0'
    official_checks = engineering.get('official_avg_checks', {})
    producer_checks['eight_transactions_official_API_pass'] = len(official_checks) == 8 and all(row.get('all_pass') is True for row in official_checks.values())
    bindings = engineering.get('q_bindings', {})
    expected_weights = {f'_Qs.params.{i}.weight': shape for i, shape in ((0,[5,512,513]),(1,[5,512,512]),(2,[5,101,512]))}
    producer_checks['exact_three_live_Q_weights'] = bindings.get('q_weight_shapes') == expected_weights
    aliases = bindings.get('alias_rows', [])
    tensor_aliases = [row for row in aliases if row.get('suffix') not in ('__batch_size','__device')]
    producer_checks['live_detach_target_storage_binding'] = len(tensor_aliases) == 10 and all(row.get('live_detach_alias') is True and row.get('target_distinct') is True and row.get('binding_source') == 'actual_TensorDict_parameters' for row in tensor_aliases)
    manifest_path = Path(inputs['manifest_path'])
    producer_checks['manifest_hash'] = sha(manifest_path) == inputs.get('manifest_sha256')
    observation_path = Path(inputs['observations_path'])
    producer_checks['prepared_observation_file_hash'] = sha(observation_path) == inputs.get('observations_sha256')
    with np.load(observation_path, allow_pickle=False) as prepared:
        producer_checks['raw_observations_equal_prepared'] = bool(np.array_equal(data['observations'], prepared['observations']))
    import torch
    generator = torch.Generator(device='cpu').manual_seed(6201)
    replay_actions = torch.empty((8,64,3,1), dtype=torch.float32).uniform_(-1.,1., generator=generator).numpy()
    producer_checks['independent_CPU_candidate_action_replay'] = bool(np.array_equal(replay_actions, data['candidate_actions']))
    quantizer_path = args.input/'quantizer_transactions.json'
    quantizers = json.loads(quantizer_path.read_text())
    expected_transactions = {'FP32': ('FP32_no_quantization', None), 'W4_RTN': ('RTN', None)}
    for arm, recipe in (('W4_independent_SR','independent_SR'), ('W4_stratified_SR','stratified_SR')):
        expected_transactions.update({f'{arm} seed={seed}': (recipe, seed) for seed in (4101,4102,4103)})
    producer_checks['quantizer_transaction_set'] = set(quantizers) == set(expected_transactions)
    quantizer_consistent = producer_checks['quantizer_transaction_set']
    references = {}
    for label, (recipe, seed) in expected_transactions.items():
        transaction = quantizers.get(label, {})
        quantizer_consistent &= transaction.get('recipe') == recipe and transaction.get('seed') == seed
        records_for_arm = transaction.get('records', [])
        if label == 'FP32':
            quantizer_consistent &= records_for_arm == []
            continue
        quantizer_consistent &= len(records_for_arm) == 3 and {row['name'] for row in records_for_arm} == set(expected_weights)
        quantizer_consistent &= transaction.get('bits') == 4 and transaction.get('qmin') == -7 and transaction.get('qmax') == 7
        for row in records_for_arm:
            quantizer_consistent &= row.get('actual_parameter_readback_exact') is True
            name = row['name']
            identity = (row.get('pre_weight_sha256'), row.get('scale_sha256'))
            if name not in references:
                references[name] = identity
            quantizer_consistent &= identity == references[name] and all(isinstance(value,str) and len(value)==64 for value in identity)
            quantizer_consistent &= row.get('shape') == expected_weights.get(name)
            quantizer_consistent &= isinstance(row.get('post_weight_sha256'),str) and len(row['post_weight_sha256']) == 64
            if seed is not None:
                quantizer_consistent &= isinstance(row.get('uniform_sha256'),str) and len(row['uniform_sha256']) == 64
            if recipe == 'stratified_SR':
                quantizer_consistent &= isinstance(row.get('permutation_sha256'),str) and len(row['permutation_sha256']) == 64
    producer_checks['same_grid_original_weights_and_recorded_SR'] = bool(quantizer_consistent)
    report = {'schema': 'tdq-independent-verification-v1', 'allocation': allocation,
        'source_job_dir': str(args.input), 'raw_sha256': sha(path),
        'engineering_sha256': sha(args.input/'engineering.json'),
        'quantizer_transactions_sha256': sha(quantizer_path),
        'raw_checks': checks, 'producer_checks': producer_checks,
        'official_avg_max_abs_error': official_max_error, 'unordered_pairs': pairs,
        'numeric_decision': numerical,
        'decision': numerical if all(checks.values()) and all(producer_checks.values()) else 'implementation_inconclusive',
        'binding_count': len(bound), 'positive_count': positive,
        'attribution_pass_count': attribution_count, 'median_gain_all_states': median_gain,
        'RTN_not_worse_count': rtn_count, 'practical_gate': practical,
        'mean_A_by_arm': A.mean(axis=0).tolist(), 'episodes': records,
        'scope': '8 real environment reset states, frozen FP imagined terminal inputs. Exact mean over 10 possible two-Q pairs, not MSE of five-Q mean. Three rounding seeds are averaged within each state. No training, environment step, success or native low-bit performance claim.'}
    args.output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if len(encoded.encode()) >= 64*1024:
        raise ValueError('Compact report exceeds budget')
    (args.output/'verification.json').write_text(encoded, encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('decision','binding_count','positive_count','median_gain_all_states','practical_gate')}))


if __name__ == '__main__':
    main()
