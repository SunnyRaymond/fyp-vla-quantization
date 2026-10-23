"""Validate all resumed cached references in a CPU allocation before GPU inference."""
import argparse
import json
import os
from pathlib import Path
import time

from run_future import guard, cached_actions, save


def main():
    guard()
    import torch
    import metrics
    import run_diagnostic as source
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    protocol = json.loads(args.protocol.read_text())
    contexts = metrics.frozen_contexts(protocol)
    metrics.self_check(torch)
    tables = cached_actions(torch, protocol, contexts, source)
    result = {'status': 'complete', 'protocol': protocol['protocol'], 'job_id': os.environ['PBS_JOBID'],
              'phase': 'cache_preflight', 'context_count': len(contexts),
              'selected_epoch': protocol['selected_epoch'], 'tensor_metrics_self_check': True,
              'matched_references': {arm: sum((c['case_id'], c['seed_index']) in rows for c in contexts)
                                     for arm, rows in tables.items()},
              'model_forward_count': 0, 'elapsed_seconds': time.monotonic() - started}
    if result['matched_references'] != {'bf16': 60, 'parent': 60, 'distilled': 60}:
        raise RuntimeError('Preflight does not cover the complete frozen reference set')
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / 'summary.json', result)
    print('CACHE_PREFLIGHT_COMPLETE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
