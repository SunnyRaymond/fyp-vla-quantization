"""Synthetic results tree -> analysis.run: tables render, gap closure / paired counts / D1 medians are correct."""
import json
import math
import random
import tempfile
from pathlib import Path

import analysis

HERE = Path(__file__).resolve().parent
METRICS = ('first10motor_rmse', 'first10motor_mse', 'future_latent_nmse', 'mixed_loss', 'full32motor_rmse',
           'late22motor_rmse', 'first10gripper_rmse', 'first10gripper_disagreement')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def main():
    cfg = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
    level = {a: 0.04 for a in cfg['arms']}
    level.update(A8=0.02, W16A16=0.005, A4_g32=0.035, A4_inno_sd=0.025, A4_inno=0.03)
    rng = random.Random(0)
    with tempfile.TemporaryDirectory() as tmp:
        cfg['remote_root'] = tmp
        res = Path(tmp) / 'results'
        suites = cfg['suites']
        for arm, base in level.items():
            records = []
            for i, case in enumerate(cfg['evaluation']['case_ids']):
                for seed in (0, 1):
                    row = dict(arm=arm, case_id=case, seed_index=seed, suite=suites[i // 10], seconds=5.0, static_reuse=3)
                    row.update({m: base * (1 + 0.01 * rng.random()) for m in METRICS})
                    write(res / 'endpoints' / arm / f'{case}_{seed}.json', row)
                    records.append(row)
            by_suite = {s: {m: base for m in METRICS} for s in suites}
            write(res / 'arms' / f'{arm}.json', dict(overall={m: base for m in METRICS}, by_suite=by_suite,
                                                     seconds_per_query=5.0, static_reuse_per_query=[3],
                                                     h_w16a4_reference=dict(first_endpoint=dict(max_relative=0., bitwise_metrics=True),
                                                                            max_abs_diff_all_endpoints=dict(first10motor_rmse=0., future_latent_nmse=0.),
                                                                            historical_overall={}) if arm == 'A4' else None))
        for w in cfg['workers']:
            write(res / f'worker_{w}' / 'startup.json', dict(pbs_jobid='123.pbs101'))
        keys = {}
        fields = dict(energy=4.0, innovation_energy=1.0, absmax_ratio=0.3, residual_dot=0.5, residual_sq=1.0,
                      previous_residual_sq=1.0, has_previous=1)
        first = dict(fields, innovation_energy=0, absmax_ratio=0, residual_dot=0, previous_residual_sq=0, has_previous=0)
        for j in range(5):
            keys[f'video_expert.blocks.{j}.ffn|294'] = dict(
                meta=dict(name='v', stream='video', tokens=294, width=3072, groups=['all', 'frame0', 'future']),
                shape=[3072, 14336], calls=dict(calls=11, static=0),
                steps={str(s): {g: (first if s == 0 else fields) for g in ('all', 'frame0', 'future')} for s in range(11)})
            keys[f'action_expert.blocks.{j}.ffn|32'] = dict(
                meta=dict(name='a', stream='action', tokens=32, width=1024, groups=['all', 'first10', 'rest']),
                shape=[1024, 4096], calls=dict(calls=10, static=0),
                steps={str(s): {g: (first if s == 0 else fields) for g in ('all', 'first10', 'rest')} for s in range(10)})
        keys['video_expert.blocks.0.cross_attn.k|129'] = dict(
            meta=dict(name='c', stream='video', tokens=129, width=4096, groups=['all']), shape=[3072, 3072],
            calls=dict(calls=11, static=10), steps={'0': {'all': first}})
        write(res / 'd1' / 'raw.json', dict(contexts=10, keys=keys))
        s = analysis.run(cfg)
        assert s['complete'] and s['arms_complete'] == 13
        assert math.isclose(s['gap_closure_to_A8']['A4_inno_sd']['first10motor_rmse'], 0.75)
        assert s['d3_decision']['best_temporal_arm'] == 'A4_inno_sd' and s['d3_decision']['go'] is True
        assert s['paired_contexts']['A4']['A8']['first10motor_rmse']['better'] == 40
        d1 = s['d1']
        assert d1['counts'] == {'action_loop': 5, 'static': 1, 'video_loop': 5}
        assert math.isclose(d1['decision']['median_innovation_ratio'], 0.5)
        assert math.isclose(d1['decision']['median_residual_corr'], 0.5) and d1['decision']['go'] is False
        assert math.isclose(d1['by_step']['video_loop'][10]['bits_gain'], -math.log2(0.3))
        text = (res / 'RESULTS.zh.md').read_text(encoding='utf-8')
        assert 'D3 主表' in text and 'D1：' in text and '10（prefill）' in text
        print(text[:1500])
    print('PASS test_analysis')


if __name__ == '__main__':
    main()
