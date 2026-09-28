# AutoDL execution guide

Current experiment state: **the physical-v2 formal campaign is complete**. The eight conditions each ran 50 episodes; all stage gates were true and native, runner, and wrapper exits were 0. Pair physical-002 bank validation passed, while its original pair summary retains a `flush` serialization error. Strict RGB v1 Pair 008 remains failed at the original threshold. Summary JSON/CSV are retrieved; the formal evidence archive has been retrieved. See [`EVAL_ADDENDUM_PHYSICAL_PAIRING_V2.json`](EVAL_ADDENDUM_PHYSICAL_PAIRING_V2.json).

The compact policy bundle is already extracted and passed its bundle gate at `/root/autodl-tmp/rolling-ball-lewm/bundle/rolling-ball-lewm-epoch100`. Do not transfer or extract it again. Empty-scene SDK passed; task-camera captures were exercised, strict-v1 RGB equality failed, the physical-v2 bank validated, and the full campaign completed. The formal evidence archive has been retrieved.

## Paths and environment

Use these paths in both SSH sessions:

```bash
ROOT=/root/autodl-tmp/rolling-ball-lewm
CONTROL="$ROOT/controls"
BUNDLE="$ROOT/bundle/rolling-ball-lewm-epoch100"
REFLEXBENCH="$BUNDLE/reflexbench"
ASSET_MIRROR="$ROOT/assets/local"
CONDA=/root/miniconda3/bin/conda
SIM="$ROOT/envs/sim"
POLICY="$ROOT/envs/policy"
LAB="$ROOT/src/IsaacLab-v2.3.1"
HOST=autodl-container-59db4f87a9-11fcb48a

test -s "$BUNDLE/BUNDLE.json"
test -s "$REFLEXBENCH/PINNED.json"
test "$(hostname)" = "$HOST"
test -s "$ROOT/omniverse/config/omniverse.toml"

export TMPDIR="$ROOT/tmp"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export PIP_NO_CACHE_DIR=1
export XDG_CACHE_HOME="$ROOT/cache/xdg"
export XDG_DATA_HOME="$ROOT/omniverse/data"
export XDG_CONFIG_HOME="$ROOT/omniverse/config"
export XDG_RUNTIME_DIR="$ROOT/tmp/xdg-runtime"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export OMNI_CONFIG_PATH="$ROOT/omniverse/config"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export CUDA_CACHE_PATH="$ROOT/cache/cuda"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
export VK_ICD_FILENAMES="$CONTROL/nvidia_headless_icd.json"
```

The task-local ICD selects NVIDIA's EGL Vulkan driver; the machine's default GLX ICD did not work. `vulkaninfo` passed with this override, and SDK smoke 003 passed 10 empty-scene steps on driver 570.124.04. Task-camera capture was exercised; strict-v1 RGB equality failed, while the physical-v2 bank passed its independent physical validation. This does not claim that 570.124.04 is an officially tested Isaac Sim driver. See [NVIDIA Vulkan ICD documentation](https://download.nvidia.com/XFree86/Linux-x86_64/575.64/README/installedcomponents.html).

The base Miniconda Python is 3.12.3 with Torch 2.7.0+cu128 and TorchVision 0.22.0+cu128. Keep it unchanged: its cp312 packages cannot serve the isolated Python 3.11 environments. Isaac Sim 5.1 uses the dedicated Python 3.11 simulator environment; the frozen policy stack uses its own Python 3.11 environment with Torch 2.8.0+cu128.

CPU preparation finished on the previous AutoDL host and its environment was cloned to the current host. StableWorldModel, image dependencies and ReflexBench are installed. The initial post-install import check failed because datasets 2.14.4 expected the removed pyarrow `PyExtensionType`; the separate repair installed datasets 5.0.1 and passed all three imports and the ten frozen policy version checks. Do not rerun either one-shot package script or reinstall ReflexBench. The old `results/cpu-packages` path was not cloned to this host; those CPU preparation reports remain in local artifacts. The USD file dependencies and Kit MDL source are present. Pair physical-002 bank validation and the three-episode capability smoke are complete; the full 50-seed campaign has completed.

## Current runtime status

The completed campaign used the verified RTX 4090 host, policy server PID `2477`, SDK smoke 003, and the validated physical-002 bank. Do not restart the server or SDK, recapture the bank, or rerun any campaign stage. The summary JSON and CSV are already retrieved locally; the summary JSON, CSV, and formal evidence archive have been retrieved locally.

Strict RGB v1 render-repeatability history is frozen in `EVAL_FREEZE.json`: Pair runs 003–005 used `rtx/post/aa/op1=TAA` and failed RGB comparison (mean about 3.4, p99 about 17) with no bank. Pair 006 under FXAA failed (mean 2.2105, p99 7); Pair 007 with 64 refresh renders failed (mean 0.38237, p99 4); Pair 008 with 256 refresh renders terminally failed the unchanged mean <= 0.25 / p99 <= 2 gate (mean 0.3431986, p99 3). Pair 008 initial physical-state checks passed, but same-first-action was not verified. Stop render-only warmup tuning; do not overwrite these results.

The strict v1 settings and failures remain frozen in `EVAL_FREEZE.json`. Physical-v2 design is documented in [`EVAL_ADDENDUM_PHYSICAL_PAIRING_V2.json`](EVAL_ADDENDUM_PHYSICAL_PAIRING_V2.json). Pair physical-001 failed because extra dynamic task attributes differed. Pair physical-002 embedded `pair_check` passed across all 50 seeds, with same-first-action physical maxdiff 0.0 and a 10,781,767-byte fixture bank. Its `pair.json` retained a top-level `flush` serialization error; do not rewrite that result as a clean PASS. The saved bank was independently loaded and validated (`results/pair-physical-50-newhost-002/fixture_bank_validation.json` = `PASS_PHYSICAL_FIXTURE_BANK_VALIDATION`) and its compressed copy was retrieved under local `artifacts/autodl/pair-physical-50-newhost-002` (2,337,583 bytes). Strict RGB remains FAIL and does not enter the physical-v2 pass decision; each policy request still receives new native RGB. No bank recapture is needed.

Sync smoke 001 failed on Gym reset order; 002 failed because `cfg.num_rerenders_on_reset` is absent in pinned Isaac Lab 2.3.1, so root mapped the native `rerender_on_reset` boolean to 0/1. Sync smoke 003 (`results/sync-smoke-physical-001/sync.json`) completed with paired-physical and closed-loop flags true, native/wrapper exits 0, but 0/3 task successes: all three ran 75 control ticks and terminated in native phase 2. Episode wall times were 18.687, 18.482, and 19.066 s; 75 requests had median server latency 644.95 ms, CEM latency 632.20 ms, wall observation age 650.97 ms, sim observation age 0 ms, and RTF 0.053347. This is a capability smoke, not the formal success-rate result.

Campaign parent PID `30529` has exited after sync-50 → fixed-delay 6×50 → true-async-50, using the same physical-002 bank, epoch-100 checkpoint, and frozen CEM/native physics. All three stage gates were true and native/runner/wrapper exits were 0. Success counts for sync/K0/K1/K2/K4/K8/K16/async were 1/1/0/2/0/4/0/1 per 50 episodes. Sync and K0 successes were on different seeds; these results do not establish a monotonic delay effect or benefit. True-async had 1262/1262 wall-deadline overruns and did not meet the 40 ms target. This is a full LeWM+CEM policy evaluation, not a one-step model experiment. Results are in `results/campaign-physical-001/`; summary JSON/CSV are retrieved locally.

Sync and fixed-delay K=0 repeats expose render/noise nuisance; physical-v2 is not strict visual pairing or pure model-causal evidence. The strict RGB diagnostic may remain false. Final power/shutdown status is tracked only in `AUTODL_RUN.json` → `shutdown_after_completion`.

The evaluator keeps all schedules on the same native `env.step()` path: one 40 ms control tick advances four 10 ms physics steps. Fixed delay blocks the simulator during planner service, then inserts exactly K hold ticks; true-async paces each tick against wall time. Native task success is reported only from ReflexBench termination handling. The exact interfaces and remaining runtime gates are in [EVALUATOR_ADAPTER.md](EVALUATOR_ADAPTER.md).

Strict RGB v1 Pair 008 remains failed. The physical-v2 capability smoke completed 0/3 successes; formal outcomes are complete and must be interpreted with the limitations above, not as evidence of a monotonic delay benefit or 40 ms hard-realtime performance.

## Power status

The authoritative shutdown state and next action are in [`AUTODL_RUN.json`](AUTODL_RUN.json), field `shutdown_after_completion`; this guide does not mirror whether the instance is powered on or off.

Version references: [Isaac Lab v2.3.1 installation](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/pip_installation.html) and [Isaac Sim 5.1 Python/EULA guidance](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html).
