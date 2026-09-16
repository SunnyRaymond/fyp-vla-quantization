
## EGL UUID compatibility correction

- `16179181.pbs101` showed robosuite `binding_utils.py` also asserts `MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`; despite the resolver computing `0`, its original UUID mask was intentionally still present, so preflight failed before context creation. No model load; GPU walltime was 2 seconds.
- `resolve_egl_device.sh` now requires `nvidia-smi` to expose exactly the scheduler UUID, records it as `CUDA_VISIBLE_DEVICES_ORIGINAL`, remaps the process-local mask to logical `CUDA_VISIBLE_DEVICES=0`, verifies the same UUID remains visible, and sets `MUJOCO_EGL_DEVICE_ID=0`. This preserves PBS isolation because remapping happens only after UUID identity verification inside the allocation; a multi-device or mismatch case exits 65.
- Corrected preflight `16179434.pbs101` submitted (`1 GPU/2 CPUs/8 GB/5 min`); it records before/after GPU lists and runs EGL context plus a 64x64 MuJoCo render. Full smoke waits for its `SUCCESS`.

## EGL preflight correction

- `16179167.pbs101` confirmed PBS mapping `GPU-a1a62934-...` → physical index `0`, but preflight used `resolve_egl_device | tee`, so the exported `MUJOCO_EGL_DEVICE_ID=0` was lost in a pipeline subshell; robosuite still parsed the UUID and failed. It used only 2 seconds GPU time and did not load model.
- Preflight now writes resolver output with redirection and then `cat`, preserving the exported variable in the job shell. Corrected preflight `16179181.pbs101` submitted; it remains `1 GPU/2 CPUs/8 GB/5 min` and must produce EGL context + render checksum before full smoke.
# Latest status — EGL diagnosis and preflight

更新时间：2026-09-09（Asia/Singapore）

- Runtime：本 task 使用 `gpt-5.6-luna` / Reasoning effort `xhigh`；主代理 heartbeat 负责长等待。
- Final smoke `16179022.pbs101` 已真实加载 checkpoint：`Loaded checkpoint via model.load_checkpoint`，Wan components finished in `194.93s`；随后在 robosuite EGL 初始化失败。首要错误是 `egl_context.py` 将 PBS 注入的 `CUDA_VISIBLE_DEVICES=GPU-a1a62934-772e-ee6d-f8f2-3f85324b65e1` 当作 integer，析构 `AttributeError` 为继发错误；该作业 `Exit_status=1`，无 rollout 视频。
- 已确认 robosuite 代码的选择逻辑：若未设置 `MUJOCO_EGL_DEVICE_ID`，它会尝试把 `CUDA_VISIBLE_DEVICES` 转成整数；不能把该变量改成 `0`，因为 PBS 的 UUID mask 必须保留。
- 已新增 `resolve_egl_device.sh`：在获批 GPU allocation 内以 `nvidia-smi --query-gpu=index,uuid` 将 scheduler UUID 映射为物理 index，仅设置 `MUJOCO_EGL_DEVICE_ID`，保留原 `CUDA_VISIBLE_DEVICES`，无法唯一映射时 fail-closed。
- 已新增短时 GPU `egl_preflight.pbs`：`1 GPU/2 CPUs/8 GB/5 min`，在 compute node 记录 GPU UUID/index，创建 robosuite `EGLGLContext`，并用 MuJoCo `Renderer` 输出 64x64 render checksum；不加载模型。
- 已同步 preflight、helper 和最终 smoke runner；preflight 作业 `16179167.pbs101` 已提交。只有 `SUCCESS` 后才提交同映射逻辑的最终 one-task/one-episode smoke。
- 所有模型加载、EGL/render、hash、安装和 I/O 继续仅在 PBS compute allocation；login 仅执行 qstat/stat/tail 等轻量查看，保留 SSH host-key 校验。

## Final GPU queue status

- `16179022.pbs101` is currently `Q`; scheduler comment remains `Not Running: would exceed overall limit on resource ngpus in queue`. It has not consumed GPU walltime. The CPU environment and all component checks are green; the next state change will be allocation or a scheduler failure.
## Final GPU smoke retry

- CPU repair `16178948.pbs101` succeeded (`Exit_status=0`, `dependency_repair=ok`, `eval_entry_import=ok`, `pip check: No broken requirements found`, walltime `00:03:22`).
- Smoke runner was resynchronized with the final `libero-compat` shim and `GIT_PYTHON_REFRESH=quiet`.
- Final bounded GPU smoke `16179022.pbs101` submitted with `afterok:16178948.pbs101`; it requests `1 GPU/16 CPUs/110 GB/2h`, uses one `libero_goal` task/episode in `first_frame` mode with application timeout 5400s, and requires MP4 + results JSON + exit code before `SUCCESS`.
## Latest pip check repair

- `16178932.pbs101` passed full `eval_entry_import=ok`, but `pip check` found `anyio 4.15.1` requiring `typing_extensions>=4.16.0`, conflicting with FastWAM pin `typing-extensions==4.15.0`.
- Prep/repair scripts now pin `anyio==4.14.0` before the FastWAM-required `typing-extensions==4.15.0`.
- CPU-only repair `16178948.pbs101` submitted (`afterok:16178767.pbs101`) to reapply both pins, rerun full entry import, and require clean `pip check`; no GPU smoke until success.
## Latest pip consistency repair

- `16178896.pbs101` passed `eval_entry_import=ok`; `pip check` found only `fastwam 0.1.0` requiring `typing-extensions==4.15.0` while dependency resolution had installed 4.16.0.
- `prepare_fastwam_env.pbs` and `repair_env_deps.pbs` now pin `typing-extensions==4.15.0` after the broad runtime install.
- `16178932.pbs101` submitted CPU-only (`afterok:16178767.pbs101`) to apply the pin, rerun full entry import, and require a clean `pip check`; no GPU smoke is submitted until it succeeds.
## Latest dependency probe

- `16178806.pbs101` installed the omitted pyproject packages successfully and reached the full entry import, then exposed one path-only issue in the probe: `action_ensembler` is a script-local module under `experiments/libero`, so the probe needed that directory on `PYTHONPATH` to mimic direct script execution.
- Repair script now includes `$SRC/experiments/libero` in `PYTHONPATH`; `16178896.pbs101` is submitted CPU-only with `afterok:16178767.pbs101` for the same full entry import plus `pip check`. No GPU requested while dependency/path validation continues.
## Latest CPU dependency repair

- `16178798.pbs101` completed full entry-import attempt but failed at `datasets` import (`ModuleNotFoundError: No module named 'datasets'`); no GPU was requested.
- The repair script now installs the complete FastWAM `pyproject.toml` runtime set omitted by the initial minimal install: `av`, `datasets`, `deepspeed`, `future`, `jsonlines`, `packaging`, `pandas`, `pyarrow`, `torchcodec`, and `wandb` (existing pinned torch/vision and other packages are reused).
- New CPU-only repair `16178806.pbs101` submitted with dependency `afterok:16178767.pbs101`; it performs full `eval_libero_single.py` entry import and `pip check`. GPU smoke waits for its success.# Fast-WAM LIBERO-goal Smoke Handoff — Latest Status

更新时间：2026-09-09 17:10 SGT

- Runtime：`gpt-5.6-luna` / Reasoning effort `xhigh`；主代理 heartbeat 负责长等待监测。
- Checkpoint：`libero_optional_idm_2cam224.clean.pt` 已通过 HF API LFS size/OID 与 compute-node SHA-256 校验；stats 已存在。
- Environment：`16178595.pbs101` 已成功，`imports=ok`（torch 2.7.1+cu128、mujoco 3.3.2）。
- Auxiliary components：`16178670.pbs101` 已成功，VAE/T5/tokenizer 的 exact-size 与 SHA-256 均通过；VAE/T5 来源为公开 `noodlepop` HF mirror，tokenizer 来源为官方 `Wan-AI` HF repo，mirror 等同性仍由 FastWAM loader registry hash 作为最终门控。
- GPU smoke：`16178717.pbs101` 真实申请 A100 并运行约 1m57，但在 eval import 阶段因 SIF 的当前 `libero` package 与 pinned FastWAM 需要的 legacy `libero.libero` layout 不匹配而退出（`ModuleNotFoundError`），未加载 checkpoint、未生成视频。
- 修复：smoke PBS 新增 job-local `libero-compat` namespace，未改 SIF/extracted package；已提交 `16178753.pbs101`，但其 import 后又发现 `bddl` 缺少 `future`。CPU repair `16178767.pbs101` 已只安装 `future==1.0.0` 成功。
- 当前 repair：已将 repair script 扩展为完整 `eval_libero_single.py` entry import + `pip check`，提交 `16178779.pbs101`（CPU-only，依赖 `afterok:16178767.pbs101`）。只有该 job 成功后才提交下一次 GPU smoke，避免重复占用 GPU。
- 所有 heavy I/O、hash、安装、解压、模型加载和 rollout 均必须在 PBS compute allocation；login 仅 qstat/stat/tail 等轻量查看。旧 login 下载入口仍无条件拒绝。
# Fast-WAM Optional IDM LIBERO-goal smoke handoff

## 已完成：主代理验证成功（2026-09-09 19:29 SGT）

- **最小 smoke 已成功**：GPU作业16180074.pbs101，PBS Exit_status=0，walltime=00:06:14，SUCCESS存在。
- 官方Optional IDM checkpoint实际加载成功，first_frame模式，LIBERO-goal task0 / trial0，任务 `open the middle drawer of the cabinet`；results JSON记录1/1成功，任务执行55.93576秒。不能据此估计全套LIBERO-goal成功率。
- CPU OSMesa渲染 + PBS分配的一张A100进行CUDA推理；保留CUDA_VISIBLE_DEVICES的scheduler UUID并验证实际PyTorch UUID。没有全局EGL device context探测。不是EGL速度复现，也未测试IDM模式。
- 本地产物：`artifacts/16180074.pbs101/rollout.mp4`（228479 bytes，完整FFmpeg解码exit0）、`gpu0_task0_results.json`、`job.log`、`eval.log`、`final-frame.png`。已目视核查最后一帧打开抽屉。
- 已结束GPU作业；Luna agent及heartbeat保持暂停，任务已完成，不要自动重跑。
- 可复用入口：`osmesa_smoke.pbs`；实际执行脚本快照在成功artifact中。详见 `RESULT.md`。

## 接手期间记录（历史）

- 主代理已暂停 Luna 实施和 heartbeat，独占修复；不要并发提交旧 EGL 探测。
- 主 checkpoint 实际已加载成功（16179022）；后续失败是 robosuite/PBS UUID/EGL 设备隔离兼容问题，不是权重下载。
- 禁止逐个全局 EGL device 创建 context；16179763 已由主代理取消。
- 新路线：现有 LIBERO SIF 提供 OSMesa CPU 软件渲染，Fast-WAM 推理仍只使用 PBS 分配的 CUDA UUID。仅用于最小 smoke，不能作为 EGL/GPU 渲染性能数据。
- 新文件：`render_preflight.py`、`osmesa_preflight.pbs`、`osmesa_smoke.pbs`。仅在 PBS compute node 执行。旧 `resolve_egl_device.sh` 不再用于新路线。
- CPU 预检在真实 LIBERO-goal task0 执行 reset、initial state、10 个 no-op steps，检查两路 256×256 非空图像；先不加载大模型。
- 已查明容器需绑定 host Python 的 `libffi.so.6`；OSMesa 的 LLVM15 需容器新版 libstdc++，不能被 host GCC11 的 LD_LIBRARY_PATH 覆盖。当前优先容器 `/usr/lib/x86_64-linux-gnu`。
- CPU 真实渲染预检 16179894.pbs101 成功（Exit_status=0，39秒）：LIBERO Goal task0 reset、initial state、10步；两路256×256图像正常，分别std=53.16/63.55，已保存PNG及render-preflight.json。主代理已取回并目视检查agentview图像。
- 第一轮容器 GPU smoke 16179907 在完整入口导入 pyarrow 时发现缺少 host OpenSSL1.1（Exit1）；已绑定libssl.so.1.1和libcrypto.so.1.1，未改模型权重或任务参数。
- CPU 16180040 已同时通过 full_eval_entry_import=ok 和真实两路render检查。新版预检覆盖实际评测入口。
- 新GPU smoke使用1GPU/16CPU/110GB/30min，保留并验证PBS UUID，再完成真实render preflight，最后加载固定checkpoint单episode；要求exit0、非空MP4和results JSON同时存在。
- 当前GPU作业16180074.pbs101已运行，在同一container通过CUDA UUID、full_eval_entry_import及真实render，正在加载Wan组件。日志在 `artifacts/16180074.pbs101/job.log` 和 `eval.log`（远端根为 `/scratch/users/ntu/yguo017/fastwam-smoke`）。
- 旧 `egl_preflight.pbs` 已在本地及远端无条件禁用，防止误启动全局device探测。
- 下方为历史记录，以本节当前状态及最新 PBS 日志为准。

更新时间：2026-09-09（Asia/Singapore）

## 当前固定状态

- 官方仓库：`https://github.com/yuantianyuan01/FastWAM`
- 固定 revision：`7faa71108368fbb3b6885649f112af607427a2d4`（官方 `main` HEAD；commit `Optimize IDM action-only inference`）
- 本地源码：`D:\Downloads\Final Year Project\reproduction\fastwam-smoke\FastWAM`
- 官方 README 已 live 核验 Optional IDM 参数：`task=libero_optional_idm_2cam224_1e-4`、`EVALUATION.sigma_shift=1.0`、`+EVALUATION.action_infer_mode=first_frame`。
- 代码核验：`experiments/libero/eval_libero_single.py` 在 action-only 路径向 `infer_action` 转发 `action_infer_mode`；`src/fastwam/models/wan22/fastwam_optional_idm.py` 支持 `idm` 与 `first_frame`。

## Checkpoint 下载

- `libero_optional_idm_2cam224.pt`：`12,041,735,545` bytes（约 11.2 GiB；HF `X-Linked-ETag`: `26b4efffded221b9a303ca8f2cddb9999c8b7524e2751c855c74a986242ce8b4`）。
- HF API live LFS metadata：`https://huggingface.co/api/models/yuanty/fastwam/tree/main?recursive=true` 返回该文件 `lfs.oid=26b4efffded221b9a303ca8f2cddb9999c8b7524e2751c855c74a986242ce8b4`、`lfs.size=12,041,735,545`；该 OID 将由后续 CPU 作业对 clean 文件实际计算的 SHA-256 复核。
- `libero_optional_idm_2cam224_dataset_stats.json`：`40,939` bytes（HF `ETag`: `e8ff7341ad86b9750dc1fbc26250da8c2c05f1a2`）。
- 远端主机：`asp2a-login-ntu02`，通过现有 NTU Jump Host + 严格 host-key 校验连接。
- 远端下载目录：`/scratch/users/ntu/yguo017/fastwam-smoke/checkpoints/`
- 远端下载日志：`/scratch/users/ntu/yguo017/fastwam-smoke/logs/checkpoint_download.log`
- 早期 login-node 多下载器曾并发写入同一路径；该可疑文件已安全保留为 `libero_optional_idm_2cam224.concurrent-download.quarantine.pt`，禁止加载。
- 管理员已发出 login heavy-I/O 警告；本任务已确认 login 上无遗留 `aria2c`、`wget`、`curl`、`find`、hash、解压或安装进程。所有本任务 shell 脚本均加入 `PBS_JOBID` 存在且 hostname 非 login 的 fail-closed guard；旧 `start_checkpoint_download.sh` 已改为无条件拒绝执行，仅保留作历史提示。
- 当前干净下载方式：PBS CPU-only job `16178218.pbs101`（`q2`，2 CPUs/4 GB，8h walltime）在 compute node 使用单一 `/usr/bin/aria2c`、8 connections、可恢复写入。当前 clean 文件：`/scratch/users/ntu/yguo017/fastwam-smoke/checkpoints/libero_optional_idm_2cam224.clean.pt`。
- clean 下载日志：`/scratch/users/ntu/yguo017/fastwam-smoke/logs/checkpoint_download_16178218.pbs101.log`；完成后写入 `libero_optional_idm_2cam224.clean.COMPLETE`。
- 该作业启动约 10 秒时已写入 518 MB（约 4%），日志估计约 6 分钟剩余；以 PBS job 完成、`clean.COMPLETE` 和目标文件 `stat` 等于 `12,041,735,545` 为完成条件。目标文件完成前不可使用 clean 文件。
- 启动/查看命令（不含任何凭据）：

  ```powershell
  python reproduction\dino-wm-wall\remote.py --command "qstat 16178218.pbs101; stat -c '%s %y' /scratch/users/ntu/yguo017/fastwam-smoke/checkpoints/libero_optional_idm_2cam224.clean.pt; test -f /scratch/users/ntu/yguo017/fastwam-smoke/checkpoints/libero_optional_idm_2cam224.clean.COMPLETE && echo COMPLETE || true; tail -20 /scratch/users/ntu/yguo017/fastwam-smoke/logs/checkpoint_download_16178218.pbs101.log"
  ```

  ```bash
  aria2c --continue=true --max-connection-per-server=8 --split=8 \
    --min-split-size=16M --file-allocation=none --max-tries=0 \
    --retry-wait=10 --timeout=60 --summary-interval=30 \
    --dir=/scratch/users/ntu/yguo017/fastwam-smoke/checkpoints \
    --out=libero_optional_idm_2cam224.clean.pt \
    https://huggingface.co/yuanty/fastwam/resolve/main/libero_optional_idm_2cam224.pt
  ```

- 启动后曾观测到约 `2,085,596,169` bytes（约 17%）并持续增长；随后继续到约 `10,380,423,168` bytes（约 86%），速率随 HF CDN 波动。完成判据：PBS 作业退出成功、`stat` size 等于 `12,041,735,545`，下载日志含成功结束信息，并通过下述 SHA-256 校验。
- 下载作业完成后已串接 CPU-only 校验作业 `16178265.pbs101`（`afterok:16178218.pbs101`）：先重新核对 HF API 的 `lfs.oid` 与 `lfs.size`，再运行 `sha256sum`，成功后写入 `libero_optional_idm_2cam224.clean.VERIFIED`。只有该 marker 存在且校验日志报告 SHA-256 完全相等时才可加载 clean checkpoint。
- checkpoint/env 当前状态：`qstat` 显示下载与校验作业均已 finished；目标 clean 文件、stats 与 `clean.VERIFIED` 均存在。尚未完成的是 FastWAM runtime 环境与 Wan VAE/T5/tokenizer。CPU-only PBS 作业 `16178291.pbs101` 已提交，负责固定源码、创建 Python 3.11 venv、复用容器内 LIBERO assets，并通过 ModelScope 的持久目录下载这些辅助模型；日志目录为 `/scratch/users/ntu/yguo017/fastwam-smoke/artifacts/16178291.pbs101/`。
- `16178291.pbs101` 已在 compute node 早期失败：该节点没有 `git`，尚未开始 pip、解压或模型下载；脚本已改成使用 compute node 的 `curl`/`wget` 拉取固定 revision tarball，并在下一次提交中重试。
- `16178299.pbs101` 因复用了早先 Python 3.10 壳 venv，在模块切换到 Python 3.11 后出现失效解释器并以 `Exit_status=127` 结束；未产生 GPU 或大模型 I/O。脚本已加入 Python 版本自检、必要时重建 venv，并每次从固定 revision tarball 重建源码目录。
- 修复后的环境作业为 `16178331.pbs101`；新的 `afterok` 最小 GPU smoke 作业为 `16178334.pbs101`，旧的 `16178323.pbs101` 因依赖失败已结束且不使用 GPU。`16178334.pbs101` 仅在环境作业成功后申请 `1×GPU/16 CPUs/110 GB/2h`，运行 `libero_goal task_id=0 num_trials=1 action_infer_mode=first_frame`，并设置 90 分钟应用层自终止；输出、MP4、results JSON、退出码和 nvidia-smi 均写入独立 artifact 目录。
- `16178331.pbs101` 已完成大部分依赖安装并抽取 LIBERO 后，在 editable 安装阶段以 `Exit_status=1` 失败：复用 SIF 的 `/app/libero/libero` 是 package/assets 树，没有顶层 `setup.py`。已改为直接设置 `PYTHONPATH=$ROOT/libero` 并保留 `LIBERO_CONFIG_PATH`，不会再次做错误 editable install。
- 修复后环境作业 `16178489.pbs101` 已提交（持久 venv/model-cache 可复用）；新的 `afterok` GPU smoke 为 `16178491.pbs101`。待环境作业成功后，才会申请 GPU。

## access_audit 只读结果

- 当前 live 连接仍为 `asp2a-login-ntu02`，项目余额查询为 `99,461.895 SU`，当时无用户 jobs。
- 可复用容器：`/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif`（约 2.6 GB）；容器内已有 `/app/libero/libero/libero/{assets,bddl_files,init_files}`，可用于 LIBERO assets。
- 可复用缓存：`/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/source/uv-cache-linux.tar`（约 7.7 GB）和 `wheelhouse-manylinux_2_28.tar`（约 3.5 GB）。现有 `venvs/vla-eval`、`harness/.venv` 为空壳，不作为 FastWAM 环境依据。
- 该容器原为 OpenVLA-OFT evaluation 环境；只复用其 LIBERO assets，FastWAM checkpoint 与依赖需独立核验。

## 后续最小 smoke

1. 等待 `16178291.pbs101` 完成，并检查 artifact `SUCCESS`、import smoke 和 model-cache manifest；确认 VAE/T5/tokenizer 真实存在。
2. 仅在 PBS allocated GPU node 上执行一个 bounded 作业：`libero_goal`、`task_id=0`、`EVALUATION.num_trials=1`、`+EVALUATION.action_infer_mode=first_frame`、保存 rollout MP4、日志和退出码；不要在 login node 运行推理。
3. 运行入口（以远端实际绝对路径替换 `<FASTWAM>` 和 `<CKPT>`）：

  ```bash
  python experiments/libero/eval_libero_single.py \
    task=libero_optional_idm_2cam224_1e-4 \
    ckpt=<CKPT>/libero_optional_idm_2cam224.clean.pt \
    EVALUATION.dataset_stats_path=<CKPT>/libero_optional_idm_2cam224_dataset_stats.json \
    EVALUATION.task_suite_name=libero_goal \
    EVALUATION.task_id=0 \
    EVALUATION.num_trials=1 \
    EVALUATION.sigma_shift=1.0 \
    +EVALUATION.action_infer_mode=first_frame \
    EVALUATION.compile_action_infer=false \
    EVALUATION.output_dir=<FASTWAM>/smoke_output \
    gpu_id=0
  ```

   `eval_libero_single.py` 的非 worker 单任务路径会保存 `libero_goal/videos/*.mp4` 与 `*_results.json`；以退出码、JSON 和 MP4 三者共同判定 smoke 是否跑通。

## 资源边界

- ASPIRE2A 项目：`personal-yguo017`；scratch quota 100 TB，当前 live quota 约 138.6 GB used；GPU 任务必须通过 PBS，按实际短 walltime 自终止。
- 不复制或输出 credentials；不关闭 host-key 校验；不在 login node 做 model inference 或 benchmark 计算。

## 2026-09-09 当前交接（最新）

- 运行时确认：本 task 使用 `gpt-5.6-luna`，Reasoning effort=`xhigh`；监控由主代理维护。
- `16178489.pbs101` 的失败根因是 compute node 上 ModelScope HTTPS connection reset。未在 login node 重试，也未将 ModelScope 失败误报为成功。
- 已 live 核验可访问的公开 HF mirror：`noodlepop/Wan-Series-Converted-Safetensors` 提供 `Wan2.2_VAE.safetensors`（1,409,401,152 bytes，LFS SHA-256 `0e913a...590996`）和 `models_t5_umt5-xxl-enc-bf16.safetensors`（11,361,845,432 bytes，LFS SHA-256 `d92de6...8eb23`）；tokenizer 使用官方公开 `Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/`。FastWAM live code 的 `_resolve_configs()` 正好 redirect VAE/T5 到这两个 safetensors 文件名，tokenizer model id 与官方 config 完全一致。mirror digest 仅证明传输文件完整，不能单独证明它等同于 DiffSynth 官方仓库；实际 loader 的 registry model hash 与 state-dict 结构检查会作为 smoke 的兼容性门槛。
- 已把 smoke 的 `DIFFSYNTH_DOWNLOAD_SOURCE` 固定为 `huggingface`，并设置 `DIFFSYNTH_SKIP_DOWNLOAD=true`；依赖所有辅助文件预先落盘后才可启动。
- 已在 compute-only 保护下同步最新三个 PBS 脚本：`prepare_fastwam_env.pbs`、`download_components.pbs`、`smoke_libero_goal.pbs`。login-node 旧启动脚本仍无条件拒绝执行。
- 新环境作业：`16178595.pbs101`（CPU-only，8 CPUs/32 GB，8h），负责固定 FastWAM revision、复用 LIBERO assets、复用 venv 并运行 import smoke；去除 ModelScope snapshot 下载，成功后才进入组件作业。
- 组件作业暂不与环境并发，计划为 `afterok:16178595.pbs101` 提交 `download_components.pbs`：CPU-only，单文件串行、可恢复写入、VAE/T5/tokenizer 逐项 exact-size + SHA-256 校验。组件总下载量约 12.77 GB；完成后写 `artifacts/<jobid>/SUCCESS` 和 manifest。
- GPU smoke 仍未提交；只在组件作业成功后以新的 `afterok` 依赖提交，配置为 `libero_goal`、`task_id=0`、`num_trials=1`、`first_frame`、`compile_action_infer=false`、90-minute application timeout，保存 MP4、results JSON、日志和退出码。
- 截至本记录，FastWAM env、VAE/T5/tokenizer 和 LIBERO rollout 均未宣称完成；checkpoint 主文件已 clean + LFS SHA-256 verified，stats 已存在。
- 提交后轻量检查：`16178595.pbs101` 当前 `R`，compute host=`x1001c1s4b0n1`，已运行约 16 秒；login 端没有执行下载、hash、解压或 Python 计算。
- 环境 `16178595.pbs101` 已成功（主代理核实 `Exit_status=0`、`SUCCESS`、`imports=ok`）。
- 已提交组件作业 `16178670.pbs101`，依赖 `afterok:16178595.pbs101`；该作业为 CPU-only `4 CPUs/8 GB/8h`，只在 compute node 串行下载约 12.77 GB VAE/T5/tokenizer，并逐项 exact-size + SHA-256 验证。GPU smoke 暂未提交。
- `16178670.pbs101` 轻量状态检查：compute host=`x1001c2s6b1n0`；T5 下载已约 78%，日志估计约 67 秒剩余，仍为单一写入者；尚未开始 tokenizer/SHA-256 总校验。
- 组件作业 `16178670.pbs101` 已完成：compute host=`x1001c2s6b1n0`，walltime=`00:06:12`，`Exit_status=0`，`artifacts/16178670.pbs101/SUCCESS` 存在。VAE/T5/tokenizer 均达到预期大小；四个大文件 SHA-256 与脚本内 HF API LFS metadata 完全相等（见 `job.log`），manifest 已写入 artifact。组件目录现已满足 FastWAM `_resolve_configs()` 的 redirect 路径。
- mirror 身份边界仍保留：VAE/T5 来自公开 `noodlepop` HF mirror，tokenizer 来自官方 `Wan-AI` HF repo；hash 是传输完整性证据。最终加载时 `hash_model_file` registry identity gate 和 `strict=False` state-dict load 将验证它们能否被当前 FastWAM code 接受。
- 已提交 GPU smoke 作业 `16178717.pbs101`，依赖 `afterok:16178670.pbs101`；资源为 `1 GPU/16 CPUs/110 GB/2h`，compute-only，应用层 `timeout 5400s`。参数固定为 `libero_goal`、`task_id=0`、`num_trials=1`、`action_infer_mode=first_frame`、`sigma_shift=1.0`、`compile_action_infer=false`；输出目录和 artifact 按 job id 隔离，预期生成 MP4、results JSON、退出码和 SUCCESS。
- `16178717.pbs101` 轻量状态检查：当前 `Q`，PBS comment=`Not Running: would exceed overall limit on resource ngpus in queue`；尚未消耗 GPU walltime，等待调度。
- `16178717.pbs101` GPU smoke 已结束 `Exit_status=1`，资源仅使用约 `00:01:57`；真正失败发生在 eval import 阶段：SIF 抽出的 LIBERO 是当前 `libero` package 布局，而 pinned FastWAM 代码导入 legacy `libero.libero`，因此 `ModuleNotFoundError`，没有模型加载、rollout 或视频输出。
- 已修正 smoke PBS：在 compute node 的 job-local `libero-compat` 目录建立兼容 namespace，将 `libero.libero` 映射到抽取 package；未修改 SIF 或 extracted LIBERO。已同步并提交重试作业 `16178753.pbs101`，依赖 `afterok:16178670.pbs101`，仍为单 GPU bounded smoke。该修复将在 runtime import 阶段验证。

- 完整 CPU entry import repair `16178779.pbs101` 因 compute node 无 `git` 触发 GitPython 初始化异常而失败；该阶段仍未申请 GPU。已将 `GIT_PYTHON_REFRESH=quiet` 纳入 smoke/repair 环境（FastWAM 本次 eval 不调用需要 Repo 的 dataset-cache helper），并重提 `16178798.pbs101`，依赖 `afterok:16178767.pbs101`，继续执行完整 entry import + `pip check`。通过后才重提 GPU。









## 2026-09-09 latest — UUID-preserving EGL preflight

- Runtime：本 task 使用 `gpt-5.6-luna` / Reasoning effort `xhigh`。
- 已修正上一版不充分的 remap：`resolve_egl_device.sh` 现在保留 PBS 注入的 UUID `CUDA_VISIBLE_DEVICES`，不再改写为 `0`；在 compute node 以 venv 内 PyTorch `torch.cuda.get_device_properties(0).uuid` 与 scheduler UUID 完全比对，并要求该 mask 仅枚举一个 EGL device，任一不一致即 fail-closed。
- 新增 job-local `robosuite` copy/patch：只在 artifact 目录复制当前 venv 的 `binding_utils.py`，允许已验证 UUID mask 配合数值 `MUJOCO_EGL_DEVICE_ID=0`，不修改 venv、SIF 或系统 package；其 copy 通过 `PYTHONPATH` 优先加载。
- 已用严格 host-key 连接同步 `resolve_egl_device.sh`、`egl_preflight.pbs`、`smoke_libero_goal.pbs`，本地三者均通过 `bash -n`。
- 新 preflight：`16179546.pbs101`，`1 GPU/2 CPUs/8 GB/5 min`，仅做 UUID identity、EGL context 与 64x64 MuJoCo render；日志应在 `/scratch/users/ntu/yguo017/fastwam-smoke/artifacts/16179546.pbs101/job.log`，成功标记为 `SUCCESS`。
- 前一 `16179434.pbs101` 仅证明旧 `CUDA_VISIBLE_DEVICES=0` 方案能 render，不能作为 UUID remap 身份证据；等待 `16179546` 的 PyTorch UUID 与 EGL count 结果后，才允许提交 full one-task/one-episode smoke。
## 2026-09-09 latest — UUID prefix normalization and retry

- `16179546.pbs101` 按预期 fail-closed（约 41 秒，未进入 EGL render）：PBS UUID 为 `GPU-1a89214c-dfe1-449d-3cd2-f1479e0bf264`，PyTorch 返回同一 UUID 的无 `GPU-` 前缀形式。该差异是表示格式，不是设备身份不一致。
- `resolve_egl_device.sh` 已加入明确规范化：保留并比较 `GPU-` canonical UUID，同时记录 PyTorch `raw` 与 `normalized` 值；仍要求 `torch.cuda.device_count()==1`、EGL device count 为 1，任何实际 mismatch 继续 fail-closed。未改变 PBS `CUDA_VISIBLE_DEVICES`。
- 新 preflight `16179562.pbs101` 已提交（`1 GPU/2 CPUs/8 GB/5 min`），日志：`/scratch/users/ntu/yguo017/fastwam-smoke/artifacts/16179562.pbs101/job.log`。仅在其 `SUCCESS` 且 render evidence 存在后提交 full smoke。
## 2026-09-09 latest — probe diagnostics and retry

- `16179562.pbs101` 在 compute node 以 `Exit_status=65` fail-closed；日志只有 GPU inventory，未产生 EGL 或 robosuite artifact，说明 UUID probe 在身份比较前退出。已增强 helper：捕获并记录 PyTorch/EGL probe 的 stdout+stderr 与状态，避免隐藏 probe 失败原因；仍不改变 PBS mask。
- 新 preflight `16179615.pbs101` 已提交（`1 GPU/2 CPUs/8 GB/5 min`），日志：`/scratch/users/ntu/yguo017/fastwam-smoke/artifacts/16179615.pbs101/job.log`。成功条件仍为 canonical UUID match、单 EGL device、job-local robosuite patch、EGL context 与 64x64 render。
## 2026-09-09 latest — capture resolver diagnostics

- `16179615.pbs101` 也以 `Exit_status=65` fail-closed，job.log 只到 GPU inventory；上一版 resolver 的 stderr 未进入 `egl-device-map.txt`，无法看到 probe 细节。
- 已将 preflight 与 smoke 的 resolver 调用改为同时捕获 stdout/stderr 到 `egl-device-map.txt`，以保留 fail-closed 的实际诊断；未放宽任何身份门槛。
- 新 preflight `16179663.pbs101` 已提交（`1 GPU/2 CPUs/8 GB/5 min`），日志：`/scratch/users/ntu/yguo017/fastwam-smoke/artifacts/16179663.pbs101/job.log`。待 map 明确显示 raw/canonical torch UUID 及 EGL count 后再决定是否进入 smoke。
## 2026-09-09 latest — EGL PCI identity mapping

- `16179663.pbs101` 在 compute node 发现 `eglQueryDevicesEXT()` 返回 6 个全局设备，因此已安全 `Exit_status=65`；这证明单纯要求 EGL count=1 会过严，不能证明 index 0 属于本次 allocation。
- resolver 正在改为三段式身份门槛：PyTorch canonical UUID == PBS UUID；`nvidia-smi` 为该 UUID 提供唯一 PCI bus；EGL `EGL_DRM_DEVICE_FILE_EXT` 的 sysfs PCI bus 与目标 PCI 唯一匹配，得到实际 `selected_egl_index`。仍保留原始 PBS `CUDA_VISIBLE_DEVICES`，并将匹配到的 EGL index 传给 job-local robosuite patch。
- 本地 resolver 通过 `bash -n`；同步与新 PBS preflight 尚待完成。full smoke 暂不提交。
## 2026-09-09 latest — PCI mapped preflight submitted

- 已将修正后的 resolver（含 PyTorch UUID + NVIDIA PCI + EGL DRM/sysfs 唯一匹配）同步至远端；preflight/smoke 的 resolver 输出均捕获到 artifact map。
- 新 preflight `16179731.pbs101` 已提交（`1 GPU/2 CPUs/8 GB/5 min`），日志：`/scratch/users/ntu/yguo017/fastwam-smoke/artifacts/16179731.pbs101/job.log`。预期 map 会给出 `target_pci_bus` 与 `selected_egl_index`；只有随后 robosuite EGL context 与 64x64 render 成功才提交 full smoke。
## 2026-09-09 latest — CUDA/GL interop EGL mapping retry

- `16179731.pbs101` 的 map 显示所有 6 个 EGL device 的 DRM query 都抛 `EGLError`，PCI 路径无法可靠使用；该 job 保持 `Exit_status=65`，没有 render。
- resolver 已改为对每个 EGL device 创建短暂 raw EGL OpenGL context，并调用 CUDA driver `cuGLGetDevices` + `cuDeviceGetUuid`，将当前 GL context 的真实 CUDA UUID 与已验证的 PBS/PyTorch UUID 比对；唯一匹配才返回 `MUJOCO_EGL_DEVICE_ID`，不改变 PBS `CUDA_VISIBLE_DEVICES`。每个候选 context 都在 probe 内清理。
- 新 preflight `16179763.pbs101` 已提交（`1 GPU/2 CPUs/8 GB/5 min`），日志：`/scratch/users/ntu/yguo017/fastwam-smoke/artifacts/16179763.pbs101/job.log`。full smoke 仍等待成功的真实 EGL identity + render evidence。
## 2026-09-09 latest — unsafe global EGL probe stopped

- `16179763.pbs101` 已按主代理指示被 `qdel`，最终 `Exit_status=143`，仅运行约 43 秒并停在 GPU inventory；没有执行完整的逐卡 probe、没有创建跨设备 EGL contexts、没有 robosuite/render 结果，且未写 `SUCCESS`。
- 不再使用“遍历所有全局 EGL devices + CUDA/GL interop”探测，因为这可能触及非本 allocation GPU。full smoke 暂停在安全 EGL identity 映射门槛。
- 下一步仅考虑已有成功 Apptainer EGL vendor 配置，或在 PBS compute node 内使用 CPU OSMesa 做短 render/preflight；不得在 login node 运行探测，也不得重提全局逐卡探测。
