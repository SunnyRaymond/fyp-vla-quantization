#!/bin/bash
set -euo pipefail

[[ -n "${PBS_JOBID:-}" ]] || { echo "FAIL_CLOSED: PBS_JOBID is missing"; exit 90; }
[[ -n "${PBS_NODEFILE:-}" && -r "$PBS_NODEFILE" ]] || { echo "FAIL_CLOSED: PBS_NODEFILE is missing or unreadable"; exit 91; }
HOST_SHORT="$(hostname -s)"
case "$HOST_SHORT" in asp2a-login*|*login*) echo "FAIL_CLOSED: login node $HOST_SHORT"; exit 92;; esac
awk -v h="$HOST_SHORT" '$1 == h || index($1, h ".") == 1 { found=1 } END { exit !found }' "$PBS_NODEFILE" || {
  echo "FAIL_CLOSED: current host is not in the PBS allocation"; exit 93;
}

TASK_ROOT="${TASK_ROOT:?TASK_ROOT must point to task scratch}"
ROBO="$TASK_ROOT/vendor/RoboLab"
COSMOS="$TASK_ROOT/vendor/cosmos-framework"
HF_HOME="$TASK_ROOT/cache/huggingface"
export HF_HOME HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export UV_CACHE_DIR="$TASK_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$TASK_ROOT/cache/uv/python"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=300
export UV_CONCURRENT_DOWNLOADS=8
export PYTHONUNBUFFERED=1
# Keep checkouts small and prevent credential prompts in the non-interactive job;
# the full RoboLab LFS payload is fetched explicitly in its own stage below.
export GIT_LFS_SKIP_SMUDGE=1
export GIT_TERMINAL_PROMPT=0
RECEIPT="$TASK_ROOT/logs/cpu_stage_receipt.env"
set_status() {
  local key="$1" value="$2"
  if grep -q "^$key=" "$RECEIPT"; then
    sed -i "s|^$key=.*|$key=$value|" "$RECEIPT"
  else
    printf '%s=%s\n' "$key" "$value" >> "$RECEIPT"
  fi
}

# Remove only the interrupted, task-specific Isaac venv. Shared uv caches may
# contain both retained Cosmos CUDA wheels and Isaac packages, so leave them intact.
ISAAC_VENV="$TASK_ROOT/envs/robolab-isaac51"
if [[ -d "$ISAAC_VENV" ]]; then
  [[ "$TASK_ROOT" == /scratch/users/ntu/yguo017/cosmos3-edge-robolab120 && \
     "$ISAAC_VENV" == "$TASK_ROOT/envs/robolab-isaac51" ]] || {
    echo "FAIL_CLOSED: refusing to clean unexpected simulator venv path"; exit 95;
  }
  rm -rf -- "$ISAAC_VENV"
  echo "REMOVED_INCOMPLETE_ISAAC_VENV=$ISAAC_VENV"
fi
set_status ISAAC_UV_CACHE_STATUS PRESERVED_SHARED_CACHE_NOT_TARGETED

. /etc/os-release
printf 'CONTAINER_OS=%s %s\n' "$ID" "$VERSION_ID"
if [[ "$ID" != ubuntu ]] || ! awk -F. -v v="$VERSION_ID" 'BEGIN { split(v,a,"."); exit !(a[1] > 22 || (a[1] == 22 && a[2] >= 4)) }'; then
  echo "FAIL: reused SIF is not Ubuntu 22.04+"; exit 40
fi
printf 'CONTAINER_PYTHON=%s\n' "$(python3 --version 2>&1)"
printf 'CONTAINER_FFMPEG=%s\n' "$(command -v ffmpeg || true)"

mkdir -p "$TASK_ROOT/vendor" "$TASK_ROOT/models" "$TASK_ROOT/tools" "$TASK_ROOT/cache/bootstrap" "$TASK_ROOT/logs"
for tool in git curl tar; do
  command -v "$tool" >/dev/null 2>&1 || { echo "FAIL: reused SIF is missing $tool"; exit 39; }
done
if [[ -x "$TASK_ROOT/tools/git-lfs/git-lfs" ]]; then
  export PATH="$TASK_ROOT/tools/git-lfs:$PATH"
fi
if ! git lfs version >/dev/null 2>&1; then
  LFS_ARCHIVE="$TASK_ROOT/cache/bootstrap/git-lfs-linux-amd64-v3.8.0.tar.gz"
  if [[ ! -s "$LFS_ARCHIVE" ]]; then
    curl -fL --retry 2 --connect-timeout 20 \
      https://github.com/git-lfs/git-lfs/releases/download/v3.8.0/git-lfs-linux-amd64-v3.8.0.tar.gz \
      -o "$LFS_ARCHIVE"
  fi
  mkdir -p "$TASK_ROOT/tools/git-lfs"
  tar -xzf "$LFS_ARCHIVE" --strip-components=1 -C "$TASK_ROOT/tools/git-lfs"
  export PATH="$TASK_ROOT/tools/git-lfs:$PATH"
fi
git lfs version

checkout_source() {
  local url="$1" dest="$2" revision="$3" fresh=0
  if [[ ! -d "$dest/.git" ]]; then
    git clone --no-checkout --depth 1 --filter=blob:none "$url" "$dest"
    fresh=1
  fi
  if [[ "$fresh" -eq 0 && -n "$(git -C "$dest" status --porcelain --untracked-files=no)" ]]; then
    echo "FAIL_CLOSED: tracked changes in existing checkout $dest"; return 94
  fi
  git -C "$dest" lfs install --local
  git -C "$dest" fetch --depth 1 origin "$revision"
  git -C "$dest" checkout --detach "$revision"
  [[ -z "$(git -C "$dest" status --porcelain --untracked-files=no)" ]] || {
    echo "FAIL_CLOSED: pinned checkout is dirty: $dest"; return 94;
  }
}

echo "STAGE=robolab_source_and_all_lfs_assets"
set_status REPOSITORIES_STATUS IN_PROGRESS
checkout_source https://github.com/NVlabs/RoboLab.git "$TASK_ROOT/vendor/RoboLab" ad45d4f974725d020f82c2b0d77d78533aeba2b3
git -C "$TASK_ROOT/vendor/RoboLab" lfs pull
printf 'ROBOLAB_COMMIT=%s\n' "$(git -C "$TASK_ROOT/vendor/RoboLab" rev-parse HEAD)"
printf 'ROBOLAB_LFS_OBJECTS=%s\n' "$(git -C "$TASK_ROOT/vendor/RoboLab" lfs ls-files | wc -l)"
set_status ROBOLAB_ASSETS_STATUS PASS

echo "STAGE=cosmos_framework_source"
checkout_source https://github.com/NVIDIA/cosmos-framework.git "$TASK_ROOT/vendor/cosmos-framework" cf5d68c00d97ccd2480a2320ed652b92dec63102
printf 'COSMOS_FRAMEWORK_COMMIT=%s\n' "$(git -C "$TASK_ROOT/vendor/cosmos-framework" rev-parse HEAD)"
set_status REPOSITORIES_STATUS PASS

UV_DIR="$TASK_ROOT/tools/uv"
UV_BIN="$UV_DIR/uv"
mkdir -p "$UV_DIR" "$TASK_ROOT/cache/uv" "$HF_HOME/hub"
if [[ ! -x "$UV_BIN" ]]; then
  command -v curl >/dev/null 2>&1 || { echo "FAIL: curl missing from reused SIF"; exit 41; }
  curl -LsSf https://astral.sh/uv/0.12.2/install.sh | env UV_INSTALL_DIR="$UV_DIR" sh
fi
"$UV_BIN" --version
"$UV_BIN" python install 3.13
PY313="$("$UV_BIN" python find 3.13)"
printf 'PYTHON313=%s\n' "$PY313"

echo "STAGE=cosmos_edge_policy_hf_cache"
set_status POLICY_CHECKPOINT_STATUS IN_PROGRESS
HF_CLI="$COSMOS/cosmos_framework/utils/hf_cli"
MODEL_SNAPSHOT="$(UV_PROJECT_ENVIRONMENT="$TASK_ROOT/envs/hf-cli" "$UV_BIN" run --project "$HF_CLI" --frozen --no-default-groups --python "$PY313" hf download \
  nvidia/Cosmos3-Edge-Policy-DROID --revision a7c7288f9b6ac1684e993007b0f9703dd26e58ef --cache-dir "$HUGGINGFACE_HUB_CACHE" --quiet)"
[[ -d "$MODEL_SNAPSHOT" ]] || { echo "FAIL: Edge Policy-DROID snapshot not materialized"; exit 42; }
printf 'POLICY_REPOSITORY=nvidia/Cosmos3-Edge-Policy-DROID\nPOLICY_REVISION=a7c7288f9b6ac1684e993007b0f9703dd26e58ef\nPOLICY_SNAPSHOT=%s\n' \
  "$MODEL_SNAPSHOT" | tee "$TASK_ROOT/models/policy_snapshot.env"
set_status POLICY_CHECKPOINT_STATUS PASS

echo "STAGE=wan2_2_vae_hf_cache"
set_status WAN_VAE_STATUS IN_PROGRESS
VAE_SNAPSHOT="$(UV_PROJECT_ENVIRONMENT="$TASK_ROOT/envs/hf-cli" "$UV_BIN" run --project "$HF_CLI" --frozen --no-default-groups --python "$PY313" hf download \
  Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --cache-dir "$HUGGINGFACE_HUB_CACHE" --quiet)"
[[ -s "$VAE_SNAPSHOT" ]] || { echo "FAIL: pinned Wan2.2 VAE file not materialized"; exit 43; }
VAE_LINK="$COSMOS/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"
mkdir -p "$(dirname "$VAE_LINK")"
ln -sfn "$VAE_SNAPSHOT" "$VAE_LINK"
printf 'WAN_VAE_REPOSITORY=Wan-AI/Wan2.2-TI2V-5B\nWAN_VAE_REVISION=921dbaf3f1674a56f47e83fb80a34bac8a8f203e\nWAN_VAE_PATH=%s\n' \
  "$VAE_LINK" | tee "$TASK_ROOT/models/wan_vae.env"
set_status WAN_VAE_STATUS PASS
set_status RESOURCES_STATUS PASS

echo "STAGE=robolab_simulator_environment_skipped_by_user_scope"
set_status ROBOLAB_DEPENDENCIES_STATUS SKIPPED_USER_SCOPE_A100_NON_RTX
set_status ROBOLAB_SIM_ENV_STATUS SKIPPED_USER_SCOPE_A100_NON_RTX

echo "STAGE=install_cosmos_edge_policy_server_python313"
set_status COSMOS_DEPENDENCIES_STATUS IN_PROGRESS
UV_PROJECT_ENVIRONMENT="$TASK_ROOT/envs/cosmos3-edge-policy-cu128" \
  "$UV_BIN" sync --frozen --no-default-groups --project "$COSMOS" --python "$PY313" --group=cu128 --group=policy-server

set_status COSMOS_DEPENDENCIES_STATUS PASS
set_status DEPENDENCIES_STATUS PASS
set_status HF_HOME "$HF_HOME"
set_status POLICY_SNAPSHOT "$MODEL_SNAPSHOT"
set_status WAN_VAE_PATH "$VAE_LINK"
set_status COSMOS_POLICY_ENV "$TASK_ROOT/envs/cosmos3-edge-policy-cu128"
set_status ROBOLAB_SIM_ENV_STATUS SKIPPED_USER_SCOPE_A100_NON_RTX
set_status COSMOS_STACK Python-3.13+Torch-2.10.0-cu128+cu128-inference+policy-server
set_status ENVIRONMENT_TESTED NO_MODEL_LOAD_RENDER_OR_INFERENCE
