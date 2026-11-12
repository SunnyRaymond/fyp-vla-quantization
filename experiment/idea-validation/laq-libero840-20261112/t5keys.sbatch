#!/bin/bash
# EEE CPU job: which T5 caches on EEE hold the 40 standard-LIBERO prompts, and whether Fast-WAM's text encoder is local.
#SBATCH --job-name=laq840t5
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/projects/yguo017/laq840/logs/%x-%j.out
set -euo pipefail
P=/projects/yguo017; U=/scratch/users/ntu/yguo017
find $P/scratch/fastwam-smoke/model-cache -type f -size +1M -printf '%s %p\n' | sort -n | tail -20
cat > "$P/.tmp/t5keys.py" <<'PY'
import json, pickle, glob
rows = json.load(open("/projects/yguo017/laq840/results/fastwam/manifest.json"))["rows"]
langs = sorted({r["language"] for r in rows})
paths = ["/scratch/users/ntu/yguo017/cosmos-a16-baselines-20261101/results/prep/base_extra_cache.pkl",
         "/scratch/users/ntu/yguo017/cosmos-policy-c8-w4a4-20261015/results/text_embeddings.pkl",
         "/scratch/users/ntu/yguo017/cosmos-policy-vqwam-20261024/results/language_cache/embeddings.pkl"]
for path in paths:
    try:
        c = pickle.load(open(path, "rb"))
    except Exception as e:
        print(path, "ERR", type(e).__name__, e); continue
    print(path, len(c), "covers", sum(l in c for l in langs), "/ 40; missing", [l for l in langs if l not in c][:3])
PY
srun --ntasks=1 --container-image="$P/libero-latest.sqsh" --container-mounts="$P/scratch:$U,$P:$P,$P/home:$HOME" \
  "$U/cosmos-policy-c8-w4a4-20261015/env/bin/python" "$P/.tmp/t5keys.py"
