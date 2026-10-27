"""Aggregate the two frozen AQLM closed-loop arms on the approved compute node."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics

from evaluate import (
    ARMS, PROTOCOL, TIMING_FIELDS, load_manifest, read_json, recover_records,
    save, validate_protocol, validate_setup_receipt, worker_arm, worker_indices,
)


def rate(records, indices):
    selected = [records[index] for index in indices if index in records]
    successes = sum(row['success'] for row in selected)
    return dict(successes=successes, episodes=len(selected),
                success_rate=successes / len(selected) if selected else None)


def paired(candidate, comparator, indices):
    values = [(candidate[i]['success'], comparator[i]['success']) for i in indices
              if i in candidate and i in comparator]
    wins = sum(a and not b for a, b in values)
    losses = sum(b and not a for a, b in values)
    return dict(wins=wins, losses=losses, ties=len(values)-wins-losses,
                episodes=len(values),
                success_rate_difference=(wins-losses)/len(values) if values else None)


def stats(values):
    values = sorted(float(value) for value in values)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Nonfinite or negative timing/memory evidence')
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                p50=statistics.median(values) if values else None,
                p95=values[math.ceil(.95*len(values))-1] if values else None,
                maximum=values[-1] if values else None)


def timing(records):
    return {
        kind: {field: stats(metric[field] for row in records.values()
                           for metric in row['metrics']
                           if kind == 'all_queries'
                           or metric['first_query_in_run'] == (kind == 'first_query_in_run'))
               for field in TIMING_FIELDS}
        for kind in ('all_queries', 'first_query_in_run', 'subsequent_queries')
    }


def load_workers(results, rows, protocol):
    arms = {arm: {} for arm in ARMS}
    workers = {}
    for worker_id in range(4):
        directory = Path(results)/'workers'/f'worker_{worker_id}'
        arm, indices = worker_arm(worker_id), worker_indices(worker_id)
        setup_path, episode_path = directory/'setup_receipt.json', directory/'episodes.jsonl'
        if not setup_path.exists():
            if episode_path.exists() and episode_path.stat().st_size:
                raise ValueError('Episode records exist without their source setup receipt')
            workers[str(worker_id)] = dict(arm=arm, complete=False, episodes=0, expected=140)
            continue
        setup = read_json(setup_path)
        metadata = validate_setup_receipt(setup, protocol, worker_id, arm)
        records = recover_records(episode_path, rows, worker_id, protocol, metadata, arm)
        summary_path = directory/'summary.json'
        summary = read_json(summary_path) if summary_path.exists() else {}
        complete = summary.get('complete') is True
        if summary:
            expected = dict(protocol=PROTOCOL, worker_id=worker_id, arm=arm,
                            episodes=len(records), completed=len(records), expected=140,
                            indices=indices, completed_indices=list(records), install_metadata=metadata)
            if any(summary.get(key) != value for key, value in expected.items()):
                raise ValueError('Worker summary disagrees with validated episode records')
        if complete and (summary.get('status') != 'complete' or set(records) != set(indices)):
            raise ValueError('Worker claims completion without its full frozen shard')
        if set(arms[arm]) & set(records):
            raise ValueError('Duplicate episode across workers in the same arm')
        arms[arm].update(records)
        workers[str(worker_id)] = dict(arm=arm, complete=complete, episodes=len(records),
                                      expected=140, source_job=metadata['source_job'],
                                      startup_passed=True)
    return arms, workers


def summarize(arms, rows, workers, protocol):
    if set(arms) != set(ARMS):
        raise ValueError('Both AQLM arms are required')
    all_indices = list(range(280))
    if len(rows) != 280 or any(set(records)-set(all_indices) for records in arms.values()):
        raise ValueError('Closed-loop results exceed the frozen cohort')
    complete = (all(set(records) == set(all_indices) for records in arms.values())
                and len(workers) == 4 and all(w['complete'] for w in workers.values()))

    def panel(indices):
        return dict(expected_episodes_per_arm=len(indices),
                    arms={arm: rate(arms[arm], indices) for arm in ARMS},
                    ours_vs_reconstruction=paired(arms['ours'], arms['reconstruction'], indices))

    suites = sorted({row['suite'] for row in rows})
    dimensions = sorted({row['dimension'] for row in rows})
    result = dict(protocol=PROTOCOL, pbs_jobid=os.environ.get('PBS_JOBID'),
                  complete=complete, status='complete' if complete else 'checkpointed',
                  checked_utc=datetime.now(timezone.utc).isoformat(),
                  episodes=sum(len(r) for r in arms.values()), expected_episodes=560,
                  episodes_per_arm={arm: len(arms[arm]) for arm in ARMS},
                  overall=panel(all_indices), workers=workers,
                  by_suite={suite: panel([i for i,r in enumerate(rows) if r['suite']==suite])
                            for suite in suites},
                  by_dimension={dimension: panel([i for i,r in enumerate(rows)
                                                  if r['dimension']==dimension])
                                for dimension in dimensions},
                  by_cell={f'{suite}/{dimension}': panel([i for i,r in enumerate(rows)
                                                         if r['suite']==suite and r['dimension']==dimension])
                           for suite in suites for dimension in dimensions},
                  timing={arm: timing(records) for arm,records in arms.items()},
                  execution=protocol['quantization']['execution'], native_low_bit_kernel=False,
                  new_training_updates=0, new_teacher_queries=0, reference_episode_slots=0,
                  parent_pair_job=protocol['sources']['parent_pair_job'])
    if complete and (result['episodes'] != 560
                     or result['overall']['ours_vs_reconstruction']['episodes'] != 280):
        raise ValueError('Completed paired result has the wrong denominator')
    return result


def report(summary):
    overall = summary['overall']
    lines = ['# AQLM loss-pair 闭环结果', '',
             f"状态：{summary['status']}，完成 {summary['episodes']}/560 episodes。", '',
             '原始冻结 cohort：4 suites × 7 perturbation dimensions × 10 variants；每臂280，state0和seeds配对。', '',
             '| Arm | 成功 / episodes | 成功率 |', '| --- | ---: | ---: |']
    for arm in ARMS:
        value = overall['arms'][arm]
        percent = f"{100*value['success_rate']:.2f}%" if value['episodes'] else 'NA'
        lines.append(f"| {arm} | {value['successes']}/{value['episodes']} | {percent} |")
    pair = overall['ours_vs_reconstruction']
    lines += ['', f"ours 相对 reconstruction：{pair['wins']} 胜 / {pair['losses']} 负 / {pair['ties']} 平，配对分母 {pair['episodes']}。", '']
    for title, panels in (('Suite', summary['by_suite']), ('Perturbation dimension', summary['by_dimension']),
                          ('Suite / dimension', summary['by_cell'])):
        lines += [f'## {title}', '', '| 分组 | reconstruction | ours | 胜 / 负 / 平 |',
                  '| --- | ---: | ---: | ---: |']
        for key, value in panels.items():
            a, b, p = value['arms']['reconstruction'], value['arms']['ours'], value['ours_vs_reconstruction']
            lines.append(f"| {key} | {a['successes']}/{a['episodes']} | {b['successes']}/{b['episodes']} | {p['wins']}/{p['losses']}/{p['ties']} |")
        lines.append('')
    lines += ['## 推理实现与证据边界', '',
              '执行为 decoded BF16 dense AQLM weights + dynamic A4 + BF16 KV。时间和显存只描述本实现；不代表 native low-bit kernel 加速。',
              'summary.json 分别保存每臂 first query / subsequent query 的 denoising latency、full query wall time 和显存统计。',
              '没有额外训练、BF16 teacher forward或外部reference episodes。各臂使用固定final8部署，不按闭环结果选择checkpoint。',
              '这是固定cohort的一次配对结果；不据此宣称跨seed统计显著性。', '']
    return '\n'.join(lines)


def main():
    import runner
    runner.guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    protocol = validate_protocol(read_json(args.protocol))
    _, rows = load_manifest(args.manifest)
    arms, workers = load_workers(args.results, rows, protocol)
    summary = summarize(arms, rows, workers, protocol)
    if len(json.dumps(summary, allow_nan=False).encode('utf-8')) > 256*1024:
        raise ValueError('Compact summary exceeds the controller small-file limit')
    save(args.results/'summary.json', summary)
    (args.results/'report.zh.md').write_text(report(summary), encoding='utf-8')
    print(json.dumps(dict(complete=summary['complete'], episodes=summary['episodes'],
                          episodes_per_arm=summary['episodes_per_arm'])), flush=True)
    return 0 if summary['complete'] else 75


if __name__ == '__main__':
    raise SystemExit(main())
