"""Validate the already-completed paired reference in a CPU allocation."""
import argparse
import os
from pathlib import Path
import closedloop
import closedloop_aggregate
from anchor_selfcheck import require_allocation


def main():
    require_allocation()
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol', required=True)
    p = closedloop.read_json(parser.parse_args().protocol)
    source = closedloop.read_json(Path(p['closedloop']['source_root']) / 'protocol.json')
    closedloop.validate_protocol(p, source)
    manifest, rows = closedloop.load_manifest(p['closedloop']['manifest'])
    records, metadata = closedloop_aggregate.load_reference(p, source, manifest, rows)
    import previous_vq_adapter
    import runner
    adapter = previous_vq_adapter._runtime_type()
    if not issubclass(adapter, runner.Runtime) or adapter.episode is runner.Runtime.episode:
        raise RuntimeError('The prior decoded-VQ episode adapter is unavailable')
    result = {'protocol': p['protocol'], 'complete': True, 'passed': True,
              'pbs_jobid': os.environ['PBS_JOBID'], 'reference_episodes': len(records),
              'reference_successes': metadata['successes'], 'reference': metadata,
              'episode_adapter': 'previous_vq_adapter._runtime_type().episode',
              'model_forwards': 0, 'training_updates': 0, 'environment_steps': 0}
    closedloop.save(Path(p['remote_root']) / 'results/closedloop_prepare/summary.json', result)
    print('ANCHOR_CLOSEDLOOP_PREPARE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
