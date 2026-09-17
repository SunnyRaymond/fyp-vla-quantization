"""Bounded, single-allocation controller for smoke and same-GPU throughput."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name('smoke_runner.py')
    base = [sys.executable, str(runner), '--root', str(args.root)]
    start = time.time()
    subprocess.run(base + ['--mode', 'check', '--output', str(out / 'check')],
                   check=True, timeout=720)
    workload = out / 'check' / 'workload.pkl'
    if not workload.is_file():
        raise RuntimeError('Smoke did not save workload.pkl')
    controller = {'status': 'running', 'gpu_count': 1, 'started_at': start,
                  'benchmark_repeats': 12, 'runs': []}
    for workers in (1, 2):
        group = out / f'workers_{workers}'
        group.mkdir()
        release = group / 'start'
        processes = []
        files = []
        try:
            for worker in range(workers):
                worker_out = group / f'worker_{worker}'
                ready = group / f'ready_{worker}'
                log = (group / f'worker_{worker}.log').open('w')
                files.append(log)
                command = base + ['--mode', 'bench', '--output', str(worker_out),
                                  '--workload', str(workload), '--repeats', '12',
                                  '--warmup', '2', '--ready-file', str(ready),
                                  '--start-file', str(release),
                                  '--worker-id', str(worker), '--workers', str(workers)]
                processes.append(subprocess.Popen(command, stdout=log,
                                                  stderr=subprocess.STDOUT))
            deadline = time.monotonic() + 180
            while not all((group / f'ready_{i}').is_file() for i in range(workers)):
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError('Benchmark exited before ready; inspect worker logs')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Benchmark initialization exceeded 180 seconds')
                time.sleep(0.1)
            released = time.time()
            release.touch()
            for process in processes:
                result = process.wait(timeout=max(1, 300 - (time.time() - released)))
                if result != 0:
                    raise RuntimeError(f'Benchmark worker exited {result}')
            controller['runs'].append({'workers': workers, 'release_epoch': released,
                                       'all_workers_exit_epoch': time.time(),
                                       'status': 'completed'})
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            for log in files:
                log.close()
            (out / 'controller.json').write_text(json.dumps(controller, indent=2))
    controller.update(status='completed', ended_at=time.time(),
                      elapsed_seconds=time.time() - start)
    (out / 'controller.json').write_text(json.dumps(controller, indent=2))
    print(json.dumps(controller, indent=2), flush=True)


if __name__ == '__main__':
    main()
