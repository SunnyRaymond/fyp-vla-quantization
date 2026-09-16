# WM/WAM/VLA quantization feasibility audit（2026-09-12）

## 范围与判断边界

本审计只检查本地已有 source、manifest、results 和 PBS 产物，服务于“最多 4×A100-SXM4-40GB”的候选筛选；没有新 idea、模型加载、实验、集群连接或大文件操作。文中“已可复用”只表示本地存在代码/权重/验证产物；“可行”表示有既有运行证据，不等于 native INT4/VQ kernel、压缩 bytes 或 latency 已证实。

调度配置按任务要求记为 `gpt-5.6-luna / xhigh`；这是调度配置记录，不是本机 runtime 自证的模型身份。

## 基座与本地复用度

| 基座 | 已有证据与可复用物 | 实际 precision / 规模边界 | ≤4 A100 判断 |
|---|---|---|---|
| **DINO-WM Wall** | `reproduction/dino-wm-wall/source/`、`source.tar`、Wall planner/eval 脚本和配置；官方 `wall_single` epoch 65、source commit `0a9492fa...` 的 metadata 与 results；远端 PBS 已用 1×A100 完成 50-case、7 MPC rounds，47/50=94%，提前停止。实际 checkpoint（368,656,057 bytes）和 dataset 不在本地 source 根，不能称离线可重跑。 | eval metadata 为 `torch.float32`，去掉 visualization-only decoder 后 42,020,948 params；1×A100 CEM peak allocated 约 4.06 GiB。旧 `SCREEN_RESULTS.zh.md` 的 17-job fast screen 也全为 FP32 activation + weight-only symmetric per-output-channel W4/W8 emulation。 | **最稳妥的单卡基座。** 既有闭环 screen 实际 2.657 A100-hours（最多 4 jobs 并发），calibration probe peak sampled device memory 12,820 MB。可在 1×A100 做小批量 calibration/闭环；带 backward 的全模型 probe、decoder 开启或更长 MPC 仍需 smoke gate。旧 RankCal 具体 recipe 已 `no-go`，不能当作正向方法证据。 |
| **OpenVLA-OFT Goal** | 本地 checkpoint 目录 `reproduction/openvla-oft-vla-eval-libero/source/checkpoints/openvla-7b-oft-finetuned-libero-goal/`（22 files，约 15.45 GB）和 OFT/harness source 均存在；官方 revision `c2d0f9f...`。两次独立 2×A100 PBS evaluation 完成 500 unique episodes、500 videos，491/500=98.2%，验证器与 SQLite 均通过。 | checkpoint config 与 loader 均为 `torch_dtype=bfloat16`；主权重 4 个 safetensors 加 action/proprio heads。成功拓扑是每张 GPU 一个 model server；2 GPU/job 是并行分片，不是模型必须做 tensor parallel。README 的 14,890,088 KB 是节点 peak memory 记录，不能误写成 VRAM peak。 | **可作为 VLA anchor，但校准更贵。** 既有完整 inference 证明一张 40GB A100 可承载一份 BF16 server；复现实验建议 2×A100/job、每卡一副本。activation/backward calibration 可能超过 40GB，应先 batch=1、冻结非目标模块、layer-wise CPU/offload，再决定是否扩大。 |
| **Fast-WAM Optional IDM** | 本地 `reproduction/fastwam-smoke/FastWAM/` source、configs、PBS runner、结果和 smoke video；revision `7faa711...`。`first_frame` 在 1×A100、OSMesa CPU render 上 1/1 成功；另有 32 states/256 rows 的 `idm` fake-W4 diagnostic。实际 checkpoint（12,041,735,545 bytes）与 Wan components 在此前 compute allocation 中加载，但当前本地目录不保存权重，不能称离线重跑。 | source 默认 `torch.bfloat16`；日志记录 video expert 5.00B、action expert 1.02B；checkpoint load 后 action-only smoke 成功。既有结果没有记录 peak VRAM；1 episode 执行时间 55.94 s 不含完整 model load，不能当 latency benchmark。 | **可作为第二个 WAM anchor，但需严格 gated。** `first_frame` + OSMesa 是已验证路线；完整 `idm` future imagination、VAE decode 和长 suite 需先做 1×A100 smoke。联合 MoT backward/activation calibration 的显存尚未测量；优先 inference-only activation collection 或冻结分支的 layer-wise PTQ。旧 OTC 结果只支持“当前配方不扩展”：observation 占 `C_r` 约 99.77%，不是 deployment gain。 |
| **SmolVLA** | 本地 harness 只有 `configs/model_servers/lerobot/smolvla.yaml` 与通用 LeRobot bridge；配置指向 `lerobot/smolvla_base`。没有本地 LIBERO-tuned checkpoint、smoke、显存或成功率结果。 | 配置注释明确是 SO-100 pretraining base，`chunk_size=null` 使用 policy default；没有本地 runtime dtype/显存证据。 | **当前不列入无训练候选。** 若未来选择，第一道门是 compute allocation 内取得并核验 target-task checkpoint 和单卡 smoke；在此之前不把它写成可复用 VLA baseline，也不为它预留 4-GPU campaign。 |

## Quantization / VQ 的现有证据

- DINO-WM 的 `source/conf/decoder/vqvae.yaml` 为 `quantize: False`；其 VQVAE 是 decoder-side latent codebook，而本次 Wall planning 还将 decoder 置为 `None`。因此不能把该代码写成 DINO-WM backbone weight VQ 或 native compression。
- OpenVLA-OFT loader 确实暴露 `load_in_4bit`/`load_in_8bit` 参数，且主路径是 BF16；本地 `pyproject.toml` 没有已核实的 `bitsandbytes`/packed-kernel deployment 结果。参数存在只证明 integration hook，不能证明该 ASPIRE2A 环境能运行 INT4。
- Fast-WAM source 的 `quantile` 主要来自 dataset statistics；没有本地 weight-VQ/INT4 kernel path。OTC 与 DINO screen 都是 fake quant/emulation：真实低比特存储、kernel throughput、device-memory savings 均未测。
- 因而 A100 上“可能可用”的 INT4/VQ implementation 只能作为待实测项：需在真实 PBS compute allocation 中先做 tiny load/inference identity gate，再测 packed bytes、peak VRAM、数值一致性和 throughput；不能从 A100 型号、代码 flag 或 nominal parameter bytes 推断。

## 保守的 ≤4×A100 资源策略

1. **先做单卡 gates（总并发最多 4 卡）：** DINO 1×A100 做 reference + 极小 fake-quant calibration；Fast-WAM 1×A100 先验证 `first_frame`，若候选需要 IDM 再单独验证 `idm`；OpenVLA 1×A100 做 BF16 load/one-episode gate。每个 gate 必须检查真实 `PBS_JOBID`、compute hostname、allocated GPU UUID、checkpoint identity、输出和应用退出码。
2. **只保留通过 gate 的两个 anchor。** DINO 的全 suite 成本有实测依据，但旧 RankCal recipe 已 no-go；OpenVLA 的 500-episode 参考成本是两个独立 2×A100 jobs、各约 65 分钟；Fast-WAM 只按既有 one-episode smoke 估算，不能沿用作者报告的 full-suite 数字作为本地资源承诺。SmolVLA 暂停。
3. **校准与闭环分开计费。** 先 CPU 准备/缓存和小 workload，再用 1×A100 做 activation collection；需要 backward 时 batch=1、目标层局部梯度或 layer-wise offload，并设置 peak-memory/时间 stop rule。通过后才用 2×A100（每卡一份独立 replica）做 paired evaluation；最多同时两组 2×A100 jobs，禁止默认做跨卡模型切分。
4. **四卡上限不是四卡保证。** 4×A100 只表示并发上限；总 campaign 需按实际 PBS allocated GPU-hours、成功早停和重试单独记录。任何 native INT4/VQ latency/memory claim 必须有相同 workload 的 BF16/FP32 对照、packed checkpoint size、peak VRAM 和同步后的 throughput；fake quant 结果不能填入这些字段。

## 结论性分级

`DINO-WM Wall`：代码/eval 和 bounded fake-quant 证据最完整，单卡资源最稳，但具体 RankCal 方案已 no-go，且本地无 checkpoint。`OpenVLA-OFT Goal`：checkpoint、代码和完整闭环 reference 最完整，BF16 单卡 inference 已被 2×A100 分片运行间接证实，适合 VLA anchor；PTQ calibration 是主要显存风险。`Fast-WAM Optional IDM`：WAM 代码和小规模两 mode 证据可复用，优先保持 OSMesa + one-GPU gate；不能把 first-frame smoke 或 OTC diagnostic 写成 full IDM success/deployment。`SmolVLA`：目前只有通用配置，不满足无训练、可复用 checkpoint 的候选条件。

## 仍待实测（本审计未执行）

1. 每个候选真实 runtime 的 peak allocated/reserved/device VRAM、activation shape 与 backward 是否能在 40GB 内完成。
2. OpenVLA `load_in_4bit/8bit` 的依赖、权重类型、数值/闭环一致性；Fast-WAM/DINO 的 native INT4/VQ packed path 是否存在并可用。
3. native checkpoint bytes、kernel throughput、control frequency、power/energy 与 BF16/FP32 对照。
4. Fast-WAM full `idm` suite、DINO 更长 MPC、OpenVLA quantized full Goal suite；这些均需独立 PBS allocation，不从现有 smoke/diagnostic 外推。

候选目录 `ideaspark_run/vq-action-geometry/phase0` 与 `ideaspark_run/closed-loop-quantization/phase0` 在本审计时仍只有 phase-0 retrieval/material，未出现可据以改变上述 compute 结论的完成候选 artifact；本文件不等待或修改它们。
