#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --job-name=cem_archive_metadata
#SBATCH --output=artifacts/archive_%j.log
set -euo pipefail
TOP="$HOME/cem_update_ccds"
source "$TOP/control/allocation_guard.sh"
"$TOP/venv/bin/python" - <<'PY'
import os, zipfile, json
from pathlib import Path
root=Path.home()/'cem_update_ccds/modelroot'
with zipfile.ZipFile(root/'downloads/wall_single.zip') as archive:
    entries=archive.infolist()
    print(json.dumps({'entries':len(entries),'uncompressed_bytes':sum(x.file_size for x in entries),'compressed_bytes':sum(x.compress_size for x in entries),'first_names':[x.filename for x in entries[:15]],'host':os.uname().nodename}))
PY
