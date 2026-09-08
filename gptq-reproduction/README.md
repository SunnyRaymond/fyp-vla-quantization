# GPTQ 首次复现：OPT-6.7B / ASPIRE2A

目标：自行运行官方 GPTQ，比较 WikiText-2 test PPL，验证在 1 × A100 40GB 上的实际耗时和显存。不是 pretraining，也不是推理 kernel benchmark。

## 固定设置

- 官方代码：https://github.com/IST-DASLab/gptq
- Commit：`2d65066eeb06a5c9ff5184d8cebdf33662c67faf`；`upstream/` 保留原始代码。
- 论文： https://arxiv.org/html/2210.17323v2 ，Table 3。
- 模型：`facebook/opt-125m`（流程检查）、`facebook/opt-6.7b`（正式实验）。下载时解析并记录精确 revision。
- Calibration：C4 第一 train shard，seed 0，128 × 2048 tokens，遵循官方随机选文档再随机选片段的流程。
- Evaluation：WikiText-2 raw test，用双换行连接；2048 tokens 不重叠窗口，末尾不足一个窗口部分不计入，使用官方 `opt_eval`。
- 对照：FP16、RTN 4-bit、GPTQ 4-bit、GPTQ 3-bit。
- Quantizer：asymmetric、per-row、group size -1、damp 0.01、block size 128，关闭 act-order / static-groups。
- PyTorch 2.1.2 CUDA 11.8、Transformers 4.31.0；具体环境见运行后回收的 environment.txt。不是论文历史依赖环境的逐版本复制。

## 论文参考值（不是本次实测）

| Model | FP16 | RTN 4-bit | GPTQ 4-bit | GPTQ 3-bit |
|---|---:|---:|---:|---:|
| OPT-125M | 27.65 | 37.28 | 31.12 | 53.85 |
| OPT-6.7B | 10.86 | 12.10 | 11.39 | 14.86 |

先检查 FP16 是否接近参考值，再比较 GPTQ 与 RTN；不通过调 seed 追逐论文数值。数据 revision、软件版本和浮点数值差异须披露。

## 资源控制

- `prepare.pbs`：4 CPU / 32GB，最长 1 小时；安装和下载不占 GPU。
- `smoke.pbs`：1 GPU / 16 CPU / 110GB，最长 10 分钟，每个配置最长 150 秒。
- `main.pbs`：1 GPU / 16 CPU / 110GB，最长 90 分钟，每个配置最长 20 分钟。
- 按 CPU prepare → smoke 验证 → main 顺序提交。失败立即退出，不无限自动重试。
- 正式作业跳过已有成功 JSON；完成后立即释放 GPU。首次 smoke + main 请求上限为 1.667 GPU-hours，最终以 PBS 实际 accounting 为准。
- 远程路径：`/scratch/users/ntu/yguo017/gptq-reproduction`。
- 本地 `remote.py` 复用已有 `nscc-access/aspire2a_shell.py`，保留 host-key verification，不复制凭据。

## 结果范围

`run.py` 调用未修改的官方 `opt_sequential` / `opt_eval`；额外包装仅负责离线数据、参数设置、计时和 JSON。准备数据时仅把长度恰好等于 2048 的文档跳过，以避免原始采样范围为空。

量化后的权重在本次 PPL 路径仍用浮点 Tensor 表示其量化网格值。这验证 quantization accuracy，不证明 packed checkpoint 压缩率或真实低比特推理加速。没有编译专用 3-bit CUDA kernel 是预期配置。

结果文件记录 PPL、耗时、PyTorch allocator 显存峰值、实际 GPU 型号和 evaluation token 数。allocator 峰值不等于整卡 `nvidia-smi` 总占用。

## 当前作业

- CPU preparation：`16165215.pbs101`，2026-09-08 提交。
- 本任务已通过 app 设置为 GPT-5.6 Luna xhigh 接续监控。
- 已建立每 10 分钟检查的 thread heartbeat：`gptq`；此前 subagent 已停止，避免重复提交。只在有意义的阶段变化、失败或完成时通知；完成后暂停。
- 提交正式 GPU 作业前核对 smoke 结果。定时监控使用本地 Codex app；PBS 已提交作业自行运行。
