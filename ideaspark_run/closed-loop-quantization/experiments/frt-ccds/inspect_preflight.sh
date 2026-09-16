#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=frt_inspect_preflight
#SBATCH --output=artifacts/inspect_preflight_%j.log
set -euo pipefail
TOP="$HOME/frt_ccds"
source "$TOP/control/allocation_guard.sh"
export PYTHONPATH="$TOP/control" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$HOME/cem_update_ccds/venv/bin/python" - "$TOP/artifacts/64691/preflight" <<'PY'
from allocation_guard import require_allocation
require_allocation()
import json,sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1])
with np.load(p/'stage_b.npz',allow_pickle=False) as a:
    r={k:float(np.max(np.abs(a[k][0]))) for k in ('clean_error','transport_error','clean_mse','transport_mse')}
r['interpretation']='diagnostic only; preflight not a scientific result'
(p/'fp_diagnostic.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
PY
