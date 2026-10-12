"""Aggregate the two VQ arms and the four existing paired baselines in PBS."""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

from evaluate import ARMS, PROTOCOL, load_manifest, read_json, recover_records, save, worker_arm, worker_indices


def load_workers(results, p, rows):
    records, metadata, setup, checkpointed = {}, {}, {}, []
    for worker in range(4):
        arm = worker_arm(worker)
        out = results / f'worker_{worker}'
        progress, summary = read_json(out / 'progress.json'), read_json(out / 'summary.json')
        fields = ('protocol', 'worker_id', 'arm', 'status', 'complete', 'episodes', 'expected', 'soft_deadline_reached')
        if any(progress.get(key) != summary.get(key) for key in fields):
            raise ValueError('Worker progress/summary disagree')
        if (summary.get('protocol') != PROTOCOL or summary.get('worker_id') != worker
                or summary.get('arm') != arm or summary.get('expected') != 140):
            raise ValueError('Worker receipt has a different protocol/assignment')
        found = recover_records(out / 'episodes.jsonl', rows, worker, p['sources'][arm]['bank'])
        if summary.get('episodes') != len(found):
            raise ValueError('Worker receipt count differs from actual saved records')
        complete = set(found) == set(worker_indices(worker))
        if complete:
            if summary.get('status') != 'complete' or summary.get('complete') is not True:
                raise ValueError('Complete worker has an invalid terminal status')
        elif (summary.get('status') == 'checkpointed' and summary.get('complete') is False
              and summary.get('soft_deadline_reached') is True):
            checkpointed.append(worker)
        else:
            raise ValueError('Incomplete worker did not reach its soft deadline')
        receipt = read_json(out / 'setup_receipt.json')
        if (receipt.get('passed') is not True or receipt.get('arm') != arm
                or receipt.get('observed_modules') != 614
                or receipt.get('max_abs_vs_frozen_action', 1) > p.get('source_parity_max_abs_tolerance', 1e-6)):
            raise ValueError('Frozen VQ action replay did not pass')
        setup[str(worker)] = receipt
        current = summary.get('install_metadata')
        if current != receipt.get('install_metadata'):
            raise ValueError('Worker install metadata differs from its setup receipt')
        if arm in metadata and metadata[arm] != current:
            raise ValueError('Workers for the same arm loaded different frozen banks')
        metadata[arm] = current
        for index, record in found.items():
            if (arm, index) in records:
                raise ValueError('Duplicate episode slot across workers')
            records[arm, index] = record
    return records, metadata, setup, checkpointed


def rate(records, indices):
    successes = sum(records[index]['success'] for index in indices)
    return {'successes': successes, 'episodes': len(indices), 'success_rate': successes / len(indices)}


def paired(candidate, reference, indices):
    wins = sum(candidate[index]['success'] and not reference[index]['success'] for index in indices)
    losses = sum(reference[index]['success'] and not candidate[index]['success'] for index in indices)
    return {'wins': wins, 'losses': losses, 'ties': len(indices) - wins - losses,
            'episodes': len(indices), 'success_rate_difference': (wins - losses) / len(indices)}


def run(args):
    import runner
    runner.guard()
    import reference_aggregate as old
    p = read_json(args.protocol)
    if p.get('protocol') != PROTOCOL or tuple(p.get('arms', [])) != ARMS:
        raise ValueError('Unexpected VQ protocol')
    manifest, rows = load_manifest(args.manifest)
    rotation_root = Path(p['references']['rotation_results'])
    old_manifest, _, shards = old.load_manifest(rotation_root / 'manifest.json')
    if manifest != old_manifest:
        raise ValueError('New evaluation differs from the original frozen manifest')
    records, metadata, setup, checkpointed = load_workers(args.results, p, rows)
    references = old.load_reference_episodes(rotation_root, rows, shards)
    scalar, _, _ = old.load_scalar_episodes(Path(p['references']['scalar_results']), rows)
    references.update({(old.ARM, index): record for index, record in scalar.items()})
    if len(references) != 1120:
        raise ValueError('Expected the four existing paired arms, 280 episodes each')
    counts = collections.Counter(arm for arm, _ in records)
    complete = len(records) == 560 and not checkpointed
    summary = {'protocol': PROTOCOL, 'status': 'complete' if complete else 'checkpointed',
               'complete': complete, 'manifest_variants': 280, 'episode_slots': len(records),
               'reference_episode_slots': len(references), 'new_arms': list(ARMS),
               'arm_episodes': {arm: counts[arm] for arm in ARMS},
               'checkpointed_workers': checkpointed, 'install_metadata': metadata,
               'setup_receipts': setup, 'calibration_refit': False, 'new_training_steps': 0,
               'claim_scope': 'Paired closed-loop success on the existing frozen cohort; decoded VQ compute; no native VQ speed claim'}
    if complete:
        expected = {(arm, index) for arm in ARMS for index in range(280)}
        if set(records) != expected:
            raise ValueError('Missing or unexpected complete VQ slots')
        by_arm = {arm: {index: record for (a, index), record in records.items() if a == arm} for arm in ARMS}
        reference_arms = list(old.REFERENCE_ARMS) + [old.ARM]
        by_arm.update({arm: {index: references[arm, index] for index in range(280)} for arm in reference_arms})

        def bucket(indices):
            return {'arms': {arm: rate(values, indices) for arm, values in by_arm.items()},
                    'paired': {arm: {other: paired(by_arm[arm], by_arm[other], indices)
                                     for other in by_arm if other != arm} for arm in ARMS}}

        summary['overall'] = bucket(list(range(280)))
        summary['suites'] = {suite: bucket([i for i, row in enumerate(rows) if row['suite'] == suite])
                             for suite in sorted({row['suite'] for row in rows})}
        summary['dimensions'] = {dim: bucket([i for i, row in enumerate(rows) if row['dimension'] == dim])
                                 for dim in sorted({row['dimension'] for row in rows})}
        summary['cells'] = {f'{suite}::{dim}': bucket([i for i, row in enumerate(rows)
                                  if row['suite'] == suite and row['dimension'] == dim])
                            for suite in summary['suites'] for dim in summary['dimensions']}
        summary['timing'] = {arm: old.summarize_timing(by_arm[arm]) for arm in ARMS}
        summary['timing_boundary'] = 'Different closed-loop trajectories/query counts; timings are not a same-input microbenchmark'
        lines = ['# Row-scale VQ 与 Microscaling VQ：LIBERO-Plus 280 episodes / arm', '',
                 '| 配置 | 成功 / episodes | Success rate |', '|---|---:|---:|']
        for arm, value in summary['overall']['arms'].items():
            lines.append(f"| {arm} | {value['successes']}/{value['episodes']} | {value['success_rate']:.2%} |")
        lines += ['', '两组VQ均复用原614层bank及Smooth0.5/H128变换，decoded BF16 compute、动态A4、KV BF16；没有重新拟合。',
                  '共560个新增episodes，与已有四组的1120个episodes采用相同冻结variants、环境与sampler seeds、IDM/replan/控制步数配置。',
                  '模型加载、显存和耗时反映当前decoded VQ实现，不作为native低位宽kernel或同输入速度结论。']
        (args.results / 'RESULTS.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save(args.results / 'summary.json', summary)
    print(f'VQ_PLUS_AGGREGATE status={summary["status"]} slots={len(records)} reference_slots=1120', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
