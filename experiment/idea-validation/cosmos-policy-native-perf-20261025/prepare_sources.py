"""Fetch upstream kernel sources only inside a real CPU PBS allocation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tarfile
import urllib.request
from datetime import datetime, timezone

SOURCES = {
    'quarot': ('https://github.com/spcl/QuaRot.git', 'main'),
    'spinquant': ('https://github.com/facebookresearch/SpinQuant.git', 'main'),
    'viditq': ('https://github.com/thu-nics/ViDiT-Q.git', 'viditq'),
    'nunchaku': ('https://github.com/nunchux-ai/nunchaku.git', 'main'),
}


def extract_source(archive, target):
    target.mkdir(exist_ok=True)
    with tarfile.open(archive, 'r:*') as bundle:
        for member in bundle.getmembers():
            relative = Path(*Path(member.name).parts[1:])
            if not relative.parts:
                continue
            destination = (target / relative).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise RuntimeError('Unsafe upstream archive path: ' + member.name)
            if member.issym():
                linked = (destination.parent / member.linkname).resolve()
                if not linked.is_relative_to(target.resolve()):
                    raise RuntimeError('External upstream symbolic link: ' + member.name)
            if member.islnk():
                linked = Path(*Path(member.linkname).parts[1:])
                if not (target / linked).resolve().is_relative_to(target.resolve()):
                    raise RuntimeError('External upstream hard link')
                member.linkname = str(linked)
            member.name = str(relative)
            bundle.extract(member, path=target)


def guard():
    host = socket.gethostname().split('.')[0]
    nodefile = Path(os.environ.get('PBS_NODEFILE', '/nonexistent'))
    if not os.environ.get('PBS_JOBID') or 'login' in host.lower() or not nodefile.is_file():
        raise RuntimeError('Source preparation requires a PBS compute allocation')
    if host not in {row.split('.')[0] for row in nodefile.read_text().split()}:
        raise RuntimeError('Host is not listed in the PBS allocation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    guard()
    vendor = args.root / 'vendor'
    vendor.mkdir(exist_ok=True)
    receipts = {}
    for name, (url, branch) in SOURCES.items():
        target = vendor / name
        try:
            slug = url.removeprefix('https://github.com/').removesuffix('.git')
            request = urllib.request.Request(f'https://api.github.com/repos/{slug}/commits/{branch}',
                                             headers={'User-Agent': 'Cosmos-native-performance-research'})
            with urllib.request.urlopen(request, timeout=90) as response:
                commit = json.load(response)['sha']
            archive = vendor / (name + '-' + commit + '.tar.gz')
            if not (target / 'UPSTREAM_COMMIT').exists():
                if not archive.exists():
                    urllib.request.urlretrieve(f'https://codeload.github.com/{slug}/tar.gz/{commit}', archive)
                extract_source(archive, target)
                (target / 'UPSTREAM_COMMIT').write_text(commit + '\n')
            elif (target / 'UPSTREAM_COMMIT').read_text().strip() != commit:
                raise RuntimeError('Existing source has another commit; explicit restaging required')
            receipts[name] = {'url': url, 'commit': commit, 'branch': branch, 'status': 'source_ready'}
        except (subprocess.SubprocessError, OSError, RuntimeError) as error:
            receipts[name] = {'url': url, 'status': 'failed', 'error': str(error)}
        print(json.dumps({name: receipts[name]}), flush=True)
    output = args.root / 'results/preparation'
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'pbs_job_id': os.environ['PBS_JOBID'], 'host': socket.gethostname(),
               'created_utc': datetime.now(timezone.utc).isoformat(), 'sources': receipts,
               'status': 'complete' if all(r['status'] == 'source_ready' for r in receipts.values()) else 'incomplete',
               'new_training_updates': 0, 'new_closed_loop_episodes': 0}
    (output / 'sources.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if receipt['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
