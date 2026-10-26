# FYP: Microscaling VQ for WAMs

当前主线：**action + future latent + frozen reader KL loss 的 Microscaling VQ，以及 WAM inference engine**。仓库备份实现、冻结协议、实验记录、结果、论文和组会材料。

2026-10-26 整理后保留 83 个 Fast-WAM / Cosmos 相关实验目录，以及 FastWAM 基础复现和 Cosmos3-Edge 迁移准备。实验原路径保持不变，避免破坏代码之间的引用。

## 优先阅读

| 内容 | 入口 | 用途 |
|---|---|---|
| Fast-WAM 当前方法 | [C80/R128 loss-pair](experiment/idea-validation/fastwam-vq-r128-c80-loss-pair/README.zh.md) · [闭环结果](experiment/idea-validation/fastwam-vq-r128-c80-loss-pair-plus/RESULTS.zh.md) | 严格匹配的 A+Z vs A+Z+reader KL |
| Fast-WAM inference engine | [C + decoder + buffer](experiment/idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/README.zh.md) · [结果](experiment/idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/RESULTS.zh.md) | 冻结 checkpoint 的解码、buffer、CUDA Graph 与显存优化 |
| Cosmos 最新六臂比较 | [完整结果入口](experiment/idea-validation/cosmos-policy-vqwam-20261024/INTERIM_RESULTS.zh.md) | BF16、adapted baselines、VQ-WAM 的性能与闭环 |
| Cosmos native performance | [当前报告](experiment/idea-validation/cosmos-policy-native-perf-20261025/RESULTS_mixed.zh.md) | 逐行保留 cohort、N、计时范围和 fidelity 状态 |
| 进行中的 AQLM loss-pair | [说明](experiment/idea-validation/fastwam-aqlm-loss-pair-20261026/README.zh.md) · [状态](experiment/idea-validation/fastwam-aqlm-loss-pair-20261026/RUN_STATE.json) | 共享初始化，对比 reconstruction 与 A+Z+KL；仅 fixed-input |
| 真机任务与文献建议 | [2026-10-26 报告](research/real-robot/2026-10-26/2026-10-26_FastWAM_RealRobot_Task_Recommendations.docx) · [来源](research/real-robot/2026-10-26/sources.json) | 从 output 迁入的正式报告 |
| 完整保留目录 | [experiment_catalog.json](research/experiment_catalog.json) | 每个目录的现有说明、结果入口和粗分类 |

## 当前结果与证据范围

- **Fast-WAM matched loss ablation**：同一 C80/R128 初始化与预算，A+Z 为 **204/280**，A+Z+reader KL 为 **208/280**，BF16 reference 为 **216/280**。KL 相对 A+Z 为 14 胜、10 负、256 平；这是冻结 cohort 的配对结果，不代表统计显著性或跨训练 seeds 稳定性。
- **Fast-WAM engine**：C+decoder+buffer 的 policy-call p50 **0.887096 s**、p95 **0.894262 s**、query peak allocated **16.914255 GB**；完整 80-query panel 与原 checkpoint 的 action 差异为 0。实现采用 VQ storage、在线 BF16 decode 和 BF16 GEMM，不属于 native INT4/VQ GEMM。
- **Cosmos 六臂闭环**：2026-10-24 实验的 ours VQ-WAM 为 **230/280**，BF16 为 **233/280**。旧 C8 A+Z 的 **226/280** 留在独立目录，不能与新 checkpoint/engine 混用。
- **Cosmos native performance**：当前 sampler 报告 ours mean **435.695 ms**、peak allocated **3.3082 GB**。BF16 与量化行测量设计不同；native baseline 存在 failed/missing fidelity receipts，不把性能记录当作精度通过或 native 闭环结果。
- **AQLM loss-pair**：整理时最新本地记录为 prepare 尚未完成、pair 尚未提交；之后状态由原实验任务持续记录。本次未查询或更改集群任务。

## 对照与诊断的组织

以下目录均位于 `experiment/idea-validation/`；完整列表见目录索引。

| 类别 | 主要目录 |
|---|---|
| VQ 表示与 Microscaling | `fastwam-smooth-vq-phases12`、`fastwam-vq-microscale`、`fastwam-vq-plus`、`fastwam-prism-visualization` |
| Action / latent supervision | `fastwam-vq-action-aware`、`fastwam-vq-action-distill`、`fastwam-vq-action-latent`、`fastwam-vq-c8-action-latent-plus` |
| Latent 机制与反事实对照 | `fastwam-vq-future-reader`、`fastwam-vq-future-nmse-matched`、`fastwam-vq-latent-diagnostic`、`fastwam-vq-proxy-diagnostic` |
| Reader KL 与其他 KL ablation | `fastwam-vq-c8-frozen-reader-kl`、`fastwam-vq-c80-frozen-reader-kl`、`fastwam-vq-r128-c80-loss-pair*` 和现有 `*kl*` 诊断 |
| Rotation / activation / scale / clipping | 现有 R128、joint、activation、B32、per-channel、Q-Clip 相关目录 |
| Baselines 与校准分布 | `fastwam-diffusion-baselines`、`fastwam-rotation-baselines`、`fastwam-spinquant-rmse`、`fastwam-smooth-scalar-plus`、`fastwam-vq-calibration-suites`、AQLM loss-pair |
| Engine 组件与失败路线 | `fastwam-vq-r128-c80-runtime-abc`、`*inference-profile`、`*layerdecode-bf16`、`*fused-kernel`、`*fused-tuning`、`*c-decoder-buffer` |
| 早期 WAM 量化 / cache / profiling | `fastwam-phase-dither`、`fastwam-action-parallel`、`fastwam-libero-bottleneck`、`fastwam-idm-libero-bottleneck` 等 |
| Cosmos 跨模型证据 | `cosmos-policy-c8-w4a4`、`cosmos-policy-vqwam-20261024`、`cosmos-policy-native-perf-20261025`，独立报告 |

保留负结果以支持方法选择、机制限制和 paper ablation。`experiment/reproduction/` 仅保留：

- `fastwam-smoke/`：FastWAM 源码入口、环境和 smoke；内部 `FastWAM/` 为独立 Git repository，主仓库保存 gitlink。精确 revision 与 upstream 见 [UPSTREAM_SOURCES.json](UPSTREAM_SOURCES.json)。
- `cosmos3-edge-robolab120/`：另一 WAM 架构的迁移准备；不属于已完成的 Microscaling VQ 验证。

## 数据与备份边界

| 数据 / 产物 | 保存方式与口径 |
|---|---|
| `agilex pepper training data/` | **整目录 Git ignore，原始数据保留在本机**；含 `run_20261023_141641`、`run_20261023_150009` 两个 run。本次未读取、移动或删除其中数据文件，也未统计体积 |
| Calibration / held-out | 保留各实验 `protocol.json`、manifest、case IDs、seeds 和 checkpoint 记录；C8/C80、旧 22 cases 与 40 contexts × 2 seeds 面板分别解释 |
| LIBERO-Plus 闭环 | 主要冻结面板为 4 suites × 7 perturbation dimensions × 10 variants = 280 slots；不等于整个 LIBERO-Plus。各结果使用自身 checkpoint 与执行路径 |
| 模型、tensor caches、环境、大型二进制数据 | 按 `.gitignore` 排除；Git 不是完整磁盘备份，远端位置以实验记录为准 |
| 论文 / slides | `papers/`、`Weekly Slides/` 保留；可编辑组会文件没有改排版 |
| Credentials / keys | 继续忽略，不进入提交 |

固定输入误差、closed-loop success、压缩存储、GPU peak 和 latency 是不同证据；decoded-BF16/fake-A4、native backend、不同 WAM 和计时范围分别记录。

## 2026-10-26 清理与恢复

先完成备份提交 **`080a4c64`** 并推送，再清理工作目录；没有重写 Git history 或 force push。

移出旧 `idea/`、`ideaspark_run/`、`meeting-ready/`、`.codex-tmp/`、`output/`、`tmp/`；7 个旧 LeWM/Fast-LeWM validation 和空壳 `pulled/`；9 个无关 reproduction，包括旧 BitVLA/QVLA/OpenVLA audits、DINO-WM、LeWM/LpWM/PLDM、独立 GPTQ 和 OTC-PTQ。详见 [清理清单](research/cleanup_2026-10-26.json)。

为了保留 Git 未覆盖的 ignored files 和独立仓库内部改动，移出内容整体保存在本地 **`D:\Downloads\FYP-cleanup-recovery-20261026`**，其中有清理计划与逐目录回执。这份恢复副本不属于 GitHub 备份，也没有永久删除；Git 历史可查看清理前已跟踪材料。

## 后续备份

```powershell
git add --all
git diff --cached --stat
git commit -m "Update WAM research backup"
git push
```

新数据目录先加入 `.gitignore` 再 staging。`scripts/check_archive_size.py` 可检查单文件 50 MiB 上限，但也会对保留的 FastWAM gitlink 报 `Source snapshot missing`；该引用不包含源码快照，恢复时需从登记 upstream 检出指定 commit。凭据与大数据另行保存。
