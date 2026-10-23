#!/bin/bash
set -euo pipefail
test -n "${PBS_JOBID:-}" && test -s "${PBS_NODEFILE:-}" || exit 64
case "$(hostname -s)" in *login*) exit 64;; esac
grep -Fqx "$(hostname -s)" <(awk -F. '{print $1}' "$PBS_NODEFILE") || exit 64
ROOT=${ROOT:?}
ARTIFACTS=${ARTIFACTS:?}
REPO=$ROOT/vendor/cosmos-policy
UV=/scratch/users/ntu/yguo017/cosmos3-edge-robolab120/tools/uv/uv
COSMOS_PLUS_ROOT=/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005
COSMOS_NATIVE_ROOT=$COSMOS_PLUS_ROOT/native/imagemagick-jammy
COSMOS_NATIVE_LIB=$COSMOS_NATIVE_ROOT/usr/lib/x86_64-linux-gnu
export UV_CACHE_DIR=$ROOT/cache/uv
export UV_PYTHON_INSTALL_DIR=$ROOT/cache/python
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=180 UV_CONCURRENT_DOWNLOADS=4
export HF_HOME=$ROOT/cache/huggingface
for candidate in "$HOME/.cache/huggingface/token" "$HOME/.huggingface/token"; do
  if [[ -s "$candidate" ]]; then export HF_TOKEN_PATH="$candidate"; break; fi
done
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false GIT_TERMINAL_PROMPT=0
export LIBERO_CONFIG_PATH=$COSMOS_PLUS_ROOT/libero-config
export LD_LIBRARY_PATH="$COSMOS_NATIVE_LIB:$COSMOS_NATIVE_ROOT/usr/lib:/usr/lib/x86_64-linux-gnu:/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAGICK_HOME=$COSMOS_NATIVE_ROOT/usr
export MAGICK_CONFIGURE_PATH=$COSMOS_NATIVE_ROOT/etc/ImageMagick-6:$COSMOS_NATIVE_LIB/ImageMagick-6.9.11/config-Q16
export MAGICK_CODER_MODULE_PATH=$COSMOS_NATIVE_LIB/ImageMagick-6.9.11/modules-Q16/coders
export MAGICK_FILTER_MODULE_PATH=$COSMOS_NATIVE_LIB/ImageMagick-6.9.11/modules-Q16/filters
mkdir -p "$ROOT/vendor" "$ROOT/cache" "$ROOT/models" "$ROOT/results"
test -x "$UV"
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/NVlabs/cosmos-policy.git "$REPO"
fi
git -C "$REPO" rev-parse HEAD > "$ROOT/source_revision.txt"
printf 'STAGE=official_runtime_install\n'
"$UV" python install 3.10
export UV_PROJECT_ENVIRONMENT=$ROOT/env
"$UV" sync --project "$REPO" --frozen --no-default-groups --extra cu128 --group libero --python 3.10
printf 'STAGE=policy_checkpoint_and_tokenizer\n'
"$ROOT/env/bin/python" "$ROOT/receive_tokenizer.py"
"$ROOT/env/bin/python" "$ROOT/prepare_assets.py"
"$ROOT/env/bin/python" -m pip --version >/dev/null 2>&1 || true
"$UV" pip freeze --python "$ROOT/env/bin/python" > "$ARTIFACTS/packages.txt"
printf 'STAGE=cpu_import_check\n'
cd "$REPO"
"$ROOT/env/bin/python" - <<'PY'
import importlib, json, os, platform, socket
from pathlib import Path
import sys
root = Path(os.environ['ROOT'])
expected_env = (root/'env').resolve()
if Path(sys.prefix).resolve() != expected_env:
    raise RuntimeError(f'CPU import check is not using the frozen environment: {sys.prefix}')
import torch
try:
    Path(torch.__file__).resolve().relative_to(expected_env)
except ValueError as exc:
    raise RuntimeError(f'Torch was not imported from the frozen environment: {torch.__file__}') from exc
plus = Path('/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005')
os.environ['LIBERO_CONFIG_PATH'] = str(plus/'libero-config')
sys.path[:0] = [str(plus/'libero-plus-compat'), str(plus/'LIBERO-plus'), str(plus/'deps')]
versions = {}
for name in ('torch', 'torchvision', 'transformer_engine', 'megatron.core', 'mujoco', 'libero', 'huggingface_hub', 'wand', 'skimage'):
    module = importlib.import_module(name)
    versions[name] = str(getattr(module, '__version__', 'imported'))
assets = json.loads((root/'asset_paths.json').read_text())
ready = bool(assets.get('tokenizer'))
receipt = dict(status='ready_for_gpu_smoke' if ready else 'awaiting_tokenizer_access', pbs_job_id=os.environ['PBS_JOBID'], host=socket.gethostname(),
               python=platform.python_version(), versions=versions,
               source_revision=(root/'source_revision.txt').read_text().strip(),
               repo=str(root/'vendor/cosmos-policy'), python_path=str(root/'env/bin/python'),
               model_loads=0, gpu_forwards=0)
destination = root/('environment_ready.json' if ready else 'environment_status.json')
tmp = destination.with_suffix('.json.tmp')
tmp.write_text(json.dumps(receipt, indent=2)+'\n')
tmp.replace(destination)
print(json.dumps(receipt), flush=True)
PY
