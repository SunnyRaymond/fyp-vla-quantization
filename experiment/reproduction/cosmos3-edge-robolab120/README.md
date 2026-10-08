# Cosmos 3 Edge on RoboLab-120

## Exact public artifacts and frozen sources

- World model: [nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge), 4B, OpenMDW 1.1; released 2026-07-20.
- Policy: [nvidia/Cosmos3-Edge-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID), official DROID action policy, OpenMDW 1.1. Frozen revision: a7c7288f9b6ac1684e993007b0f9703dd26e58ef. Its model card reports an approximately 9.17 GB repository and BF16 as the tested precision.
- Simulator and benchmark: [NVlabs/RoboLab](https://github.com/NVlabs/RoboLab), frozen revision: ad45d4f974725d020f82c2b0d77d78533aeba2b3. The upstream repository describes RoboLab-120 and owns the task definitions/assets; policies/cosmos3/run.py is its Cosmos client.
- Policy-server source: [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework), frozen revision: cf5d68c00d97ccd2480a2320ed652b92dec63102.
- Exact pinned server/dependency source: [action policy server](https://raw.githubusercontent.com/NVIDIA/cosmos-framework/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/scripts/action_policy_server_robolab.py), [pyproject.toml](https://raw.githubusercontent.com/NVIDIA/cosmos-framework/cf5d68c00d97ccd2480a2320ed652b92dec63102/pyproject.toml), and [FAQ](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/docs/faq.md).
- Required auxiliary video tokenizer: [Wan-AI/Wan2.2-TI2V-5B pinned snapshot](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/921dbaf3f1674a56f47e83fb80a34bac8a8f203e), file `Wan2.2_VAE.pth` (about 2.82 GB). The Edge policy references this VAE through Cosmos Framework's checkpoint resolver.
- Official server/client recipe: [pinned action_policy_droid_server.md](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/docs/action_policy_droid_server.md). NVIDIA's matching [Cosmos 3 action cookbook](https://github.com/NVIDIA/cosmos/blob/3e3c6d61dc15d6517c4793beab5ec3894ffa07f1/cookbooks/cosmos3/generator/action/run_policy_with_cosmos_framework.md) documents the Edge-policy flags.

RoboLab is Apache-2.0; the Cosmos model and framework use OpenMDW 1.1. The model card lists Ampere, Blackwell, and Hopper as supported microarchitectures and BF16 as the tested precision. Separately, NVIDIA's Sparse-WAM paper reports RTX 4090 inference measurements for Cosmos3-Edge; a 4090 remains a candidate for an actual runtime test, not a result of this reproduction. RoboLab requires Ubuntu 22.04+, Python 3.11, Isaac Sim 5.0/5.1, Isaac Lab 2.2.0/2.3.2.post1, and an RTX GPU. It recommends at least 48 GB VRAM and estimates about 8 GB of disk for RoboLab assets (about 7 GB).

## Prepared resources and A100 policy-server environment

The CPU-stage installer is configured to reuse the existing Ubuntu 22.04 SIF and install Python 3.13 with the pinned Cosmos Framework base dependencies, the official CUDA 12.8 inference group (`cu128`), and the `policy-server` group. It does not install `--all-extras`, the train-only `cu128-train` group, Isaac Sim, Isaac Lab, or a RoboLab Python environment. NVIDIA's pinned framework defines `cu128` as the inference group and `cu128-train` as adding training packages; the policy server uses OpenPI's WebSocket protocol. Package installation runs on a CPU allocation; it does not validate GPU imports or inference.

At the pinned framework revision, attention dispatch labels architecture 80 as A100 and tries `flash2`, cuDNN, then NATTEN, selecting the first backend that passes its input/device checks. FlashAttention 3 is present in the dependency set but is not in the automatic architecture-80 order (it is first for architecture 90). This describes the source-level default only; no A100 kernel has been run here. See the pinned [attention backend selector](https://raw.githubusercontent.com/NVIDIA/cosmos-framework/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/model/attention/backends.py).

The full pinned RoboLab checkout and LFS assets remain staged as the complete task definitions/assets for RoboLab-120. The fixed Edge policy snapshot and its required Wan2.2 VAE are staged in the Hugging Face cache and linked at the framework's expected VAE path for offline resolution.

For the tokenizer, the pinned Cosmos Framework recognizes the bundled `Cosmos3EdgeProcessor` in this policy snapshot and rewrites the configured HF model alias to the local checkpoint directory. The future A100 server launcher therefore uses the fixed local snapshot with HF offline mode; a mutable `refs/main` alias is unnecessary.

`scripts/run_policy_server.sh` is the future A100 server entrypoint. It requires a real PBS GPU allocation, samples only `CUDA_VISIBLE_DEVICES` into `logs/policy-server-${PBS_JOBID}.log`, and starts only the WebSocket policy server. It is saved but has not been run; starting it will load the model. The server flags are `--format-prompt-as-json True --guidance-interval 960 1001`.

`scripts/run_one_episode.sh` preserves the official one-task recipe for later use, but intentionally exits until a compatible RTX-enabled machine has a RoboLab/Isaac runtime. It has not been run. The saved future command requests `BananaInBowlTask` with `--num-envs 1 --num-runs 1 --headless --instruction-type default --video-mode all`.

## Resource findings

- The CCDS-TC1 probe allocated a Tesla V100-PCIE-32GB, which is unsupported for the Cosmos policy and has no RTX ray-tracing hardware for Isaac Sim.
- ASPIRE2A is available for CPU-only staging. Its A100 GPU supports the policy's Ampere architecture but is not an Isaac Sim RTX device.
- A single read-only probe of the previously authorized AutoDL endpoint found its host key already pinned but got ConnectionRefusedError. No restart, new rental, shutdown, or change to the previous experiment was attempted.
- The official ASPIRE2A Remote Visualization Portal login page is reachable, but no portal session or A40 allocation/runtime was established.
- The local Windows RTX 4060 Laptop GPU has 8 GB VRAM and is not used for this reproduction.

## Preparation status

From PowerShell with the working directory `D:\Downloads\Final Year Project`, use the verified project Python (the default Miniconda Python is not the cluster-control runtime):

```powershell
& '.\nscc-access\.venv\Scripts\python.exe' '.\experiment\reproduction\cosmos3-edge-robolab120\scripts\submit_cpu_stage.py' --check-existing
& '.\nscc-access\.venv\Scripts\python.exe' '.\experiment\reproduction\cosmos3-edge-robolab120\scripts\submit_cpu_stage.py' --status 25660513.pbs101
```

The current CPU-only preparation is `25660513.pbs101`. Its last confirmed PBS state was `R` on `x1001c4s1b1n0`, in the pinned Cosmos Python 3.13 / CUDA 12.8 policy-server dependency installation stage; the scratch receipt still marked dependencies `IN_PROGRESS`. It had downloaded multiple large CUDA wheels and was still resolving/installing packages. SSH connectivity timed out during the latest status attempts, so the current state and final receipt are **pending confirmation**. Do not submit another preparation job while this job's terminal state is unknown. The PBS job checks `PBS_JOBID`, `PBS_NODEFILE`, and that its actual hostname is not a login node before heavy I/O. Its EXIT trap records the exit status and finish time in the scratch receipt.

The original simulator-scope attempt `25659665.pbs101` was intentionally stopped at the user's request while downloading Isaac Sim/Isaac Lab dependencies (`Exit_status=143`), before that simulator environment completed. The incomplete task-specific `robolab-isaac51` virtual environment was removed; shared uv cache and the previously staged source/assets/checkpoints were retained. Preparation `25659994.pbs101` was also actively stopped after prolonged repeated large-wheel download messages (`Exit_status=143`), not a natural installation failure. A later bounded HTTP-range probe `25660720.pbs101` completed on a different compute node: the public torch wheel returned 1 MiB at about 183 kB/s; the flash-attn wheel returned about 588 kB in the 40-second limit at about 15 kB/s. Those single-sample timings are not a measurement of the main job's node or its uv transfer rate. Its scratch log and receipt are `logs/wheel-net-probe-25660720.pbs101.log` and `logs/wheel-net-probe-25660720.pbs101.env`.

The local status snapshot is `logs/cpu_stage_latest_status.txt`; the live scratch log is `/scratch/users/ntu/yguo017/cosmos3-edge-robolab120/logs/job.log`. Once SSH access is restored, check `25660513.pbs101`'s terminal `Exit_status` and the receipt's resource/dependency gates before updating this status. The RoboLab simulator environment is deliberately skipped, and no new job should install it.

No model load, render, inference, or RoboLab episode has been run. The current CPU dependency installation has not yet been confirmed complete, and the prepared CUDA wheels/Python environment have not been tested against an allocated A100. RoboLab's simulator runtime is intentionally not installed because Isaac Sim requires a compatible RTX GPU; the episode launcher is deferred until that runtime can be prepared on suitable hardware.
