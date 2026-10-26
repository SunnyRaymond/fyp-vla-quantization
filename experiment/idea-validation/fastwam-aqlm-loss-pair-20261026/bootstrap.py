"""Fetch the pinned official AQLM sources inside a CPU allocation."""
from pathlib import Path
import json
import os
import shutil
import socket
import tempfile
import urllib.request
import zipfile


def guard():
    host = socket.gethostname().split('.')[0]
    nodes = Path(os.environ['PBS_NODEFILE']).read_text().split()
    if (not os.environ.get('PBS_JOBID') or 'login' in host.lower()
            or host not in {n.split('.')[0] for n in nodes}):
        raise RuntimeError('Approved compute allocation required')


def main():
    guard()
    root = Path(os.environ['NEW_ROOT'])
    p = json.loads((Path(os.environ['ARTIFACTS']) / 'protocol.json').read_text())
    commit = p['official']['commit']
    vendor = root / p['official']['directory']
    receipt = root / 'results/bootstrap/summary.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved.get('commit') != commit or not (vendor / 'aq_engine.py').is_file():
            raise RuntimeError('Existing official source checkout differs')
        return
    vendor.parent.mkdir(parents=True, exist_ok=True)
    if vendor.exists():
        raise RuntimeError('Unreceipted vendor directory; inspect before replacing')
    with tempfile.TemporaryDirectory(prefix='aqlm-', dir=vendor.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary / 'source.zip'
        url = f'https://codeload.github.com/Vahe1994/AQLM/zip/{commit}'
        with urllib.request.urlopen(url, timeout=120) as response, archive.open('wb') as out:
            shutil.copyfileobj(response, out)
        with zipfile.ZipFile(archive) as source:
            for info in source.infolist():
                destination = (temporary / info.filename).resolve()
                if not destination.is_relative_to(temporary.resolve()):
                    raise RuntimeError('Unsafe archive path')
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError('Source archive contains a symlink')
            source.extractall(temporary)
        checkout = temporary / f'AQLM-{commit}'
        for name in ('aq_engine.py', 'src/aq.py', 'src/utils.py', 'src/kmeans.py'):
            if not (checkout / name).is_file():
                raise RuntimeError('Official source archive is incomplete: ' + name)
        checkout.rename(vendor)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(dict(protocol=p['protocol'], complete=True,
        commit=commit, repository=p['official']['repository'], pbs_jobid=os.environ['PBS_JOBID'],
        source=str(vendor), dependencies_installed=False, closed_loop_episodes=0), indent=2)+'\n')
    print('BOOTSTRAP_COMPLETE ' + commit, flush=True)


if __name__ == '__main__':
    main()
