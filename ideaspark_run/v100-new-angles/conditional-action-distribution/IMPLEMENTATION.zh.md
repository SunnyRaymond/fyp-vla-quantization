# conditional-action-distribution：实现说明

本实现只负责 8 个 frozen conditions 的 raw action collection；所有 SWD、energy、
`Dpair/Dnoise`、collapse、translation sensitivity 与跨 condition science gate 均
留给 root 的独立 CPU verifier。本地没有运行模型、计算数值结果、连接 cluster 或
提交 job。

## 运行边界

`marginal_gpu.sh` 请求一张 `UGGPU-TC1` V100、4 cores、24G、15 minutes，并将
内部 runner deadline 固定为 840 seconds。它先 source
`${TOP}/control/allocation_guard.sh`，再进行脚本快照、GPU evidence、hash、模型
加载或 sample I/O。runner 自身在 guard 后才加载 NumPy、pinned `flow_screen.py`
与 torch；缺少真实 SLURM allocation、V100、资产或 identity 时 fail closed。
不下载、不安装、不使用 RTC、`compile_model`、AMP、native low-bit kernel 或
`noise=None`。

默认资产是 `${HOME}/v100_newangles_ccds/smolvla`，manifest 是
`${HOME}/v100_newangles_ccds/conditional_marginal/manifest.json`。shell 允许用
`MARGINAL_*` 环境变量指定独立 asset root、manifest、model、checkpoint、VLM、
Python 与 LeRobot source；命令行同时支持 `--manifest`/`--input-manifest`。
Manifest 只接受 `conditional-marginal-raw-input-manifest-v1` 的 8 个已准备 sample，
严格验证 base identity SHA-256、sample SHA-256、task/episode/frame、两路真实
camera、state8 与 action7，不重新选择或替换 sample。

## 模型与执行

调用已锁定的 Flow helper（SHA-256
`ddf6e02ceb6b27a86ac479b54a6b24b5351342876916f39709504034efdb45a9`）的
`_load_runtime`、`_load_preprocessor`、`_load_raw_sample`、`_prepare_batch`、
`_move_to_device`、`_eligible_modules`、`_snapshot_weights`、`_restore_weights`、
`_state_subset_digest` 和 `_quantize_locus`。不调用 helper 的 `_run`、旧
`_load_manifest` 或旧 science metrics。

每个 arm 只在 arm 切换时 restore 一次；`FP32` 不量化，`expert_W4` 对实际绑定
到 `state_dict` 的 expert transformer Linear 做 symmetric per-output-channel
RTN W4，dequantize 后仍以 FP32 前向。所有其它参数由 bypass digest 在 arm 前、
arm 后和最终 restore 后核对；restore 使用 helper 的逐元素检查，避免每个 sample
重复全模型 hash。

每个 condition 使用同一组 64 个显式 CPU-generated noise：seed 1901 和 1902
各生成 32×50×32，再拼成 64×50×32；每次推理都 `policy.reset()`、clone batch
和 noise、`torch.no_grad()`，固定 micro-batch=4。首个 condition 的前四个 draw
另做 batch4 与四次 individual 对照；FP 与 Q 的 `max_abs` 都必须 `<=1e-5`，失败
写成 `implementation_inconclusive`，不降低门槛或减少样本。

## 产物契约

`raw_marginal.npz` 保存：

- `schema='conditional-marginal-raw-v1'`；
- `actions` shape `(8,2,64,50,7)`，arm 顺序 `['FP32','expert_W4']`；
- `noise` shape `(64,50,32)`，`completed` shape `(8,2)`、`episode_ids` shape `(8,)`；
- `batch_check_batched` 与 `batch_check_individual`，均为 `(2,4,50,7)`。

输出目录会逐 condition 原子写 raw，另存完整 `runtime_identity.json`、较详细的
`engineering.json`、小于 64 KiB 的 `summary.json`、`allocation.json`、GPU
identity、runner/helper/protocol/implementation snapshots 与 `run.log`。summary
只报告工程状态、source/checkpoint/processor identity、V100/peak memory、noise
和 sample hashes，并明确 `science_gate_deferred_to_root_cpu_verifier=true`。
