"""Install the compiler-only CUDA components needed by upstream kernels on PBS CPU."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.request
from prepare_sources import guard, extract_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    guard()
    base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
    with urllib.request.urlopen(base+'redistrib_12.8.1.json', timeout=90) as response:
        manifest = json.load(response)
    toolkit = args.root/'toolkits/cuda-12.8.1'
    toolkit.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for component in ('cuda_nvcc', 'cuda_cudart', 'cuda_cccl', 'libcurand'):
        entry = manifest[component]['linux-x86_64']
        marker = toolkit/(component+'.ready')
        if not marker.is_file():
            archive = toolkit.parent/Path(entry['relative_path']).name
            if not archive.is_file():
                urllib.request.urlretrieve(base+entry['relative_path'], archive)
            extract_source(archive, toolkit)
            marker.write_text(manifest[component]['version']+'\n')
        receipts[component] = {'version': manifest[component]['version'], 'url': base+entry['relative_path']}
        print(json.dumps({component: receipts[component]}), flush=True)
    if not (toolkit/'lib64').exists() and (toolkit/'lib').is_dir():
        (toolkit/'lib64').symlink_to('lib')
    version = subprocess.check_output([str(toolkit/'bin/nvcc'), '--version'], text=True)
    out = args.root/'results/preparation'
    out.mkdir(parents=True, exist_ok=True)
    (out/'toolkit.json').write_text(json.dumps({'status': 'complete', 'pbs_job_id': os.environ['PBS_JOBID'],
                                             'cuda_home': str(toolkit), 'components': receipts,
                                             'compiler_version': version}, indent=2)+'\n')
    print(version, flush=True)


if __name__ == '__main__':
    main()
