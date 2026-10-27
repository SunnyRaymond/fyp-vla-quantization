# 接手操作：资产、环境、状态查询与恢复

返回 [总交接](../HANDOFF.zh.md)。本文给的是实际保存的路径和可复用入口，不在本次写文档期间执行提交/训练/推理。远端位置来自协议与 launcher，未逐个重新确认大文件今天仍在；恢复先做小范围存在性/receipt 检查，不先重下载、重拟合或启动模型。

## 1. 本机、login node 与 compute allocation 的分工

本机工作目录为 `D:\Downloads\Final Year Project`。适合编辑源码/文档、读小 JSON、使用现有 controller 连接与查询。模型加载、Torch tensor 数值分析、benchmark、训练、模拟器、依赖安装、大型数据处理应在获批 CPU/GPU allocation。

ASPIRE2A login 只用于轻量 SSH、PBS 提交/查询、小型控制文件与有大小上限的日志/receipt。**不能在 login 上下载/传输大文件、递归扫描大型目录、hash 模型、解压、编译/安装、加载 checkpoint 或运行 CPU/GPU 计算。** Tmux/nohup/后台/低 CPU 都不是豁免。用户曾收到管理员 heavy-I/O 正式警告，不能重启被终止的旧下载器。

每个 workload 在任何重操作前验证：真实`PBS_JOBID`、非 login hostname、可读`PBS_NODEFILE`且当前 host 在其中；严格 launcher 还核对 live PBS state=R、job name、routing queue 与 exec_host。不要手动设置 PBS 变量绕过 guard。GPU 任务运行中每 15 秒将分配 GPU 的 utilization/VRAM 写入本 job.log；收尾退出，避免空占。

CPU 数据准备/安装/后处理优先 CPU allocation。Compute 无法联网不意味着可以退回 login 下载；应保留状态，查明获准路径。禁止动用账户 banked reset；保留 SSH host-key verification。依据：[项目AGENTS](../../AGENTS.md)、[严格主launcher](../idea-validation/fastwam-vq-r128-c80-loss-pair/run.pbs)。根`ASPIRE2A_README.md`部分较旧描述不能覆盖这些更严格规则。

## 2. 本机哪些内容在备份里

| 内容 | 状态 |
|---|---|
| `experiment/`源码、protocol、报告、小 JSON/CSV/receipts | 主 Git 备份的核心 |
| `papers/`、`Weekly Slides/` | 保留文献和可编辑组会材料 |
| `agilex pepper training data/` | 整目录 ignore，两个 raw runs 仍在本机；未验证其 schema/时间同步/动作标注 |
| `*.pt/*.pth/*.npy/*.npz/*.hdf5`、SIF、venv、压缩包 | 通常 ignore，不能假设 clone 后存在 |
| `credentials.env`、keys | ignore，不能复制到交接文档或 Git |
| `experiment/reproduction/fastwam-smoke/FastWAM` | 独立 Git，主 repo 保存 gitlink，不包含完整源码/内部 history |
| 仓库外恢复副本 | `D:\Downloads\FYP-cleanup-recovery-20261026`；不在 GitHub 备份中 |

清理前/后主 repo 提交分别为`080a4c64`与`38099754`。本次新增交接文件是之后的文档工作，不把它们伪装成已存在于这些旧提交。当前 raw data、远端 checkpoints 与旧嵌套 repo 的未提交修改不能仅靠主 Git 恢复。

FastWAM 上游为`https://github.com/yuantianyuan01/FastWAM.git`，固定 revision`7faa71108368fbb3b6885649f112af607427a2d4`。具体 metadata 见[UPSTREAM_SOURCES.json](../../UPSTREAM_SOURCES.json)。若本机源码缺失可按该登记恢复；若需要在集群下载/解压源码或模型，仍必须在 compute allocation，不能照搬到 login。

## 3. 必须保留的远端依赖

以下路径**不是建议清理目标**，只是当前依赖地图：

| 路径 | 用途 |
|---|---|
| `/scratch/users/ntu/yguo017/fastwam-smoke` | FastWAM 源码、clean checkpoint、base venv、model-cache、normalizer 等 |
| `/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005` | Plus repo、compat、deps、manifest、env_status 与 ImageMagick/sensor-noise 依赖 |
| `/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif` | 多个当前 WAM 任务仍用的 SIF；名字含 OpenVLA 不代表已无关 |
| `/scratch/users/ntu/yguo017/fastwam-a4-phases12-20261006` | 原 runner/quantization/libero_state_loader 等历史源码与输入来源 |
| `/scratch/users/ntu/yguo017/fastwam-smooth-vq-phases12-20261006` | smooth_vq/codebook 及冻结 preprocessing 来源 |
| `/scratch/users/ntu/yguo017/fastwam-vq-calibration-suites-20261013` | `data/plan.json`、source receipt、C8/C80 banks 与 Original 数据恢复规则 |
| `/scratch/users/ntu/yguo017/fastwam-vq-c80-activation-clipping-20261017/results/teacher/calibration` | 主 FastWAM C80 BF16 teacher cache/receipt |
| `/scratch/users/ntu/yguo017/fastwam-vq-activation-clipping-20261017` | 共享 held-out teacher 与旧 baseline 端点等 |
| `/scratch/users/ntu/yguo017/cosmos-policy-c8-w4a4-20261015` | Cosmos vendor 源码、assets、独立`env/bin/python`与旧校准 reference |

本机删除旧实验目录**没有**清理集群这些 roots。不要据本机目录名推断 remote 可删除；例如 SIF 和旧 pilot env_status 仍被当前 launchers 引用。`/scratch`是工作存储，不应当作永久归档保证。

## 4. 当前 Fast-WAM checkpoint 与 bundle 定位

### 4.1 算法训练链

R0 deployment：

```text
/scratch/users/ntu/yguo017/fastwam-r128-improvements-20261021/results/coverage_c80/arms/C80_once_lr1e4/deployment.pt
```

共同 VQ0 banks 与验证：

```text
/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-dual-20261022/results/banks
/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-dual-20261022/results/source_receipt.json
/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-dual-20261022/results/repro_receipt.json
/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-dual-20261022/artifacts/25912619.pbs101
```

当前 Q0 与 matchedAZ 都在下列 root，选择 arm 时明确写`reader_kl`或`az`：

```text
/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-loss-pair-20261022/results/arms/reader_kl/deployment.pt
/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-loss-pair-20261022/results/arms/reader_kl/checkpoint.pt
/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-loss-pair-20261022/artifacts/25921795.pbs101
```

`deployment.pt`服务安装；`checkpoint.pt`另含 optimizer/RNG/cursor，可续训。不要用 deployment 代替 exactresume，更不要看文件名类似就把 Q-Clipfinal80 换成 Q0 final8。

Reader artifact 依赖：

```text
/scratch/users/ntu/yguo017/fastwam-vq-c80-frozen-reader-kl-20261022/artifacts/25913982.pbs101
```

当前 loss-pair 的`run.pbs`将该 reader 实现及 dual 成功 artifact 中`rotation_bank.py`、`c8_train_banks.py`、`sampler.py`、`checkpointing.py`等复制到本 job artifact，并将源 worker 命名为`source_worker.py`。这解释了当前本机 worker 里若干 import 为何不能直接在原目录独立运行。

### 4.2 Inference-only 资产

已验收的 bundle 位置：

```text
/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-inference-profile-20261023-r2/results/export/inference_bundle.pt
```

最终 buffer 候选执行 root 由本地`long/protocol.json`登记为：

```text
/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-c-decoder-buffer-20261024-r3-long
```

结果来源不是一个“全成功 GPU job”：decoderproducer25959865、bufferexecutor25960283、CPU 最终验收 25961057。恢复结果时同时保留三者和 reuse_source 绑定；只看 bufferPBS exit1 便宣布实验无效、或只看 CPUexit0 便篡改 GPU 状态，都不正确。

Bundle 不含 reader 与 trainingmasters，但 factory 仍需 FastWAMcode、cachedVAE/text/tokenizer 及 normalizer/config。Scratch/Graph/JIT 是 runtime 资产，不能把 bundle 文件大小当成 cold-loadpeak 或 querypeak。不要通过 login 下载 15GiBbundle 来“确认可恢复”。

## 5. 环境恢复原则

Fast-WAM 当前 PBS 环境脚本加载`gcc/11.4.0-nscc`、`python/3.11.7-gcc11`、`apptainer/1.5.0`、`cuda/12.2.2`，再在指定 SIF 中用`$BASE/venv/bin/python`。**Host module 版本不等于容器里的 PyTorch/CUDA runtime 版本**；以具体 job 的 runtime/environmentreceipt 为准，不用“pip 最新版本”替换旧环境。

`environment.sh`负责 env_status 别名解析、Plus repo/deps/compat、ImageMagick 库、libffi/libssl/libcrypto 绑定，OSMesa 设置，HFofflinecache 和 PYTHONPATH。保持`CUBLAS_WORKSPACE_CONFIG=:4096:8`、禁 TF32 及原 determinism 设置；使用`apptainer --cleanenv`时必须显式传进去，之前 loss-pair-plus 曾因漏传在 0 episodes 时失败。

Cosmos 使用旧 C8 root 的独立 env 和 vendor/cosmos-policy；新结果记录 Torch2.7.0+cu128/CUDA12.8。它也依赖同一 SIF/Plusassets，不能把 Fast-WAMenv 直接覆盖过去。Nativebaseline 还涉及各自 backendbuildreceipts；仅能 import 不证明实际 kernel 调用、fidelity 或性能合格。

AQLM 官方`AQEngine.quantize()`局部关闭 strictdeterministic 是明确修订；其余两臂 sampler/训练仍开启。不要全局关 determinism 去绕过检查，也不要将共同混合 heavy/light/minimalbanks 改成每臂各拟合一份。

## 6. 接手第一步：只读检查，不提交任何新作业

从项目根目录查看本地状态：

```powershell
Get-Content -LiteralPath '.\experiment\idea-validation\fastwam-aqlm-loss-pair-20261026\RUN_STATE.json'
Get-Content -LiteralPath '.\experiment\idea-validation\fastwam-aqlm-loss-pair-20261026\monitor_status.json'
```

需要 fresh 集群状态时，现有 controller 已经负责 trustedSSH 与有界读取，不要自己重写认证/关闭 host-key 检查：

```powershell
& '.\nscc-access\.venv\Scripts\python.exe' -Xutf8 -B '.\experiment\idea-validation\fastwam-aqlm-loss-pair-20261026\control.py' status
& '.\nscc-access\.venv\Scripts\python.exe' -Xutf8 -B '.\experiment\idea-validation\fastwam-aqlm-loss-pair-20261026\control.py' logs --stage prepare
```

`status`查看全部阶段，不因传`--stage`变成只验 prepare；`verify`检查整体`completion_passed`，并非 prepare 单阶段通过就返回全实验完成。`pull --stage prepare`只取 prepare 小型产物：

```powershell
& '.\nscc-access\.venv\Scripts\python.exe' -Xutf8 -B '.\experiment\idea-validation\fastwam-aqlm-loss-pair-20261026\control.py' pull --stage prepare
```

这些命令可修改本地查询/拉取记录，但不训练或提交。若 SSH/SFTP 超时，记为 unknown/inconclusive；查不到新 log 不等于 job 失败，也不授权新 attempt。历史 PBS 记录可能已过保留期，旧 job 的查询失败不能推翻已保存终态 receipt，更不能触发重新跑完整 cohort。

## 7. AQLM 已停止：接手时保留什么

**截至 2026-10-27 13:57:09 SGT 的完成 receipt**：共享 prepare `26005386` 已 614/614 banks，pair `26008761` 两臂各 8 updates/80 exposures、calibration80/held-out80、final anchors 齐全。Fixed-input 是明确负结果。闭环 `26010239` 按用户要求在 525 条取消，PBS F/141；reconstruction280、ours245，缺 35 条；CPU 分析 `26020010` F/0，0 新 training/forward。监控已停止，automatic_resume=false；原 owner 仍为 `01a17756-c858-7142-b39a-619f26b9eae0`。

接手默认是保留与阅读，不是恢复：

1. 先读[停止分析](../idea-validation/fastwam-aqlm-loss-pair-plus-20261026/STOPPED_ANALYSIS.zh.md)、[用户停止receipt](../idea-validation/fastwam-aqlm-loss-pair-plus-20261026/user_stop_receipt.json)及[CPU完成receipt](../idea-validation/fastwam-aqlm-loss-pair-plus-20261026/stopped_analysis/26020010.pbs101/completion_receipt.json)。历史 `remaining_required_work`/旧 README 执行顺序不覆盖最新停止指令；不 qsub、不恢复剩余 35 条、不新建监控。
2. 保存源 pair 两臂 final8/FP32 masters/optimizer/RNG、共同 614 初始化、80 captures、冻结 plans、完整 calibration/held-out records、四份 anchor 验证与所有 OOM attempts 来源。初始化 22heavy/12light/580minimal，不能另拟合一份替换。
3. Fixed-input 产物在 `/scratch/users/ntu/yguo017/fastwam-aqlm-loss-pair-20261026/results/`；闭环与停止分析在 `/scratch/users/ntu/yguo017/fastwam-aqlm-loss-pair-plus-20261026/`，完整参数/episode 分析为其 `user_stop_analysis/26020010.pbs101/`。本机紧凑副本在相应 `pulled/26008761.pbs101/` 与 `stopped_analysis/26020010.pbs101/`；大文件未通过 login 搬回。
4. Reconstruction 完整 202/280 单列；两臂比较只用共同 245 条 179/245 对 125/245，5 胜 59 负 181 平。Long 仅 35 配对，不能将 episode 总体值伪装四 suite 等权 280 结果；未完成 35 条不算失败。
5. 当前诊断支持优先检查 codebook 步幅，尚未因果定位。只有后续明确授权重开，才从已有初始化和固定 batch 拆 A/lambda Z/beta KL 梯度、检查小步真实 loss；现阶段不运行额外 forward、retrain 或 sweep。原 fixed-input protocol 的 closed_loop=false 只是阶段边界，后续评测确实发生过，但被用户停止。

**标准 LIBERO 已独立完成。** `fastwam-libero-engine-20261027 / 26020749` 为 391/400，PBS/四 workers/PIPELINE_COMPLETE 完整，无待续 episodes、监控已停。远端 root `/scratch/users/ntu/yguo017/fastwam-libero-engine-20261027`，本地 `pulled/26020749.pbs101/` 保存 400-episode summary、task_queue 和实际 artifacts；详见[结果](../idea-validation/fastwam-libero-engine-20261027/RESULTS.zh.md)。它复用 Q0 和原 engine bundle，不需要为接手重新校准或 decoder tuning。

## 8. 恢复和完成的具体判断

**训练恢复**需要原 job 已终态、合法 checkpoint、完整 masters/optimizer/RNG、update cursor 与同一 frozenprotocol。Exit75 只有配合 checkpointed/RESUME_READY 记录才是可续断点；exit141/143 可能是用户受控取消，不能统一自动 resume。不得删除 intent 来让 controller 再 qsub，提交结果不明先恢复原 handle。

**评估恢复**以有效完整 query/episode 为单位，保留 samecase/seed/planrow/source checkpoint 的已完成前缀；只补缺失 units。Timeout 仍是有效 policy failure。不要重抽 variants、把缺失换成容易 case 或为“凑满分母”重新跑失败 episodes。

**完成验收**根据阶段实际合同同时看：PBS 终态与 Exit、artifact exit、PHASE_COMPLETE/PIPELINE_COMPLETE、matchingprotocol/job/source、实际 record 唯一覆盖、target count、训练 updates/exposures、所需 parity 以及 worker receipts。不是每个实验都必须有同一个 marker 名字，读其 controller/finalizer 的真实 gate。

**后处理恢复**适用于数据已完整原子写出、仅报告器/schema/锁收尾出错的已知情况。用 CPU allocation 逐条绑定原 producer，修报告即可；保留 sourceGPU 失败，新的 CPU success 独立登记，new forward/episode 为 0。例如 Cosmos1120 新 records+560 复用与 Fast-WAMdecoder/buffer 跨 job receipt，都不需重跑 GPU。

**参数/数值 gate 失败**不能用后处理“补成功”。FusedGEMM 改变输出、anchor parity 失败、NaN、cohort 不足等需要修实现或保持失败；不能放宽原 tol、改 reference 或把不同 checkpoint 混到同一个 summary。

## 9. 如果只想使用当前 inference engine

先明确目标是调用当前固定 Q0、固定 geometry、单 request/单 stream，而不是已有通用生产 server。标准 LIBERO 已验证四个独立 GPU 常驻 workers 的 400-episode 闭环；这不等于同一 GPU 共享 workspace 的多 stream 并发已验证。当前交付包含导出/安装/runtime 与测量入口，尚没有通用多用户 serving 接口、动态 batch scheduler 或 robot adapter 的验证。

在获批 GPU allocation 中，沿现有 launcher 恢复 FastWAM factory 与 bundle，再安装当前 decoder/workspace/cache/Graph 实现。依次核对 bundle 的 614target manifest、source anchor、weight/activationoperand 语义、独立 anchoraction/future、fixedgeometry 下的输入刷新；正式 benchmark 不带 profiler。不要直接在 login 或本机运行`profile_worker.py`/torch.load。

现有 Fast-WAM 主 controller 与最终 runtime 只读入口分别是：

```powershell
& '.\nscc-access\.venv\Scripts\python.exe' -Xutf8 -B '.\experiment\idea-validation\fastwam-vq-r128-c80-loss-pair\control.py' status
& '.\nscc-access\.venv\Scripts\python.exe' -Xutf8 -B '.\experiment\idea-validation\fastwam-vq-r128-c80-c-decoder-buffer\control.py' status --lane long
```

它们用于查既有 run，不会给你一个交互 GPU。需要新 allocation 时必须由具体实验任务定义资源、输入、产物和数值 gate；不要为“验证文档能跑”重跑完整 80 模型 queries。现有 CPUmocktests 只验证 controller/Graph 契约，不代替 GPUparity。

## 10. 常见错误与处理依据

| 现象 | 优先处理 | 不应采取的做法 |
|---|---|---|
| Q 有 ngpus/running-limit 注释 | 保留 handle，fresh 只读查询 | 因 comment 直接重提或动 reset |
| SSH/SFTP timeout | 标 inconclusive，后续重查 | 假定 PBS 已失败/重复 submit |
| 无 PIPELINE_COMPLETE 但 records 齐 | 分清 source 数值工作与后处理；必要 CPU 恢复 | 修改旧 job exit 或直接重跑模型 |
| Stageout_status=1 | 明确 stageout 未成功；查 scratch 小 receipt | 把 stageout 异常全算成 policy failure |
| 最新代码与 artifact 不同 | 原结果用原 artifact 解释；新 attempt 明确版本 | 以当前源码“证明”旧 run 做过某功能 |
| Native 标签但调用数 0 | 看正式 querybackend/PTX/counters，区分启动 BF16 | 只凭文件名或 import 成功宣称 native |
| Full80 timings 但 fidelity missing | 分别记录性能 N 与精度 query 覆盖 | 用 720 timings 证明 80fidelity 通过 |
| BF16 tensor 转 NumPy 失败 | CPU 后处理副本按需转 FP32，保持模型 dtype | 改整模型 dtype 以迁就日志 |
| CUDA index_add/index_reduce 确定性错误 | 按已记录精确 scope 处理，保留 repro 限制 | 全局禁 determinism、放宽 parity |
| Graph 用旧 input 或共享 scratch 冲突 | 查输入 copy、cache 失效、storage/stream ownership | 跨请求共享一个 scratch 直接并发 |
| Calibrationloss 降、held-out/closed-loop 更差 | 保留全表和 incumbent，按原 gate 判断 | 删除坏 case、按已见 test 补选 checkpoint |

## 11. 交接前后保持的东西

所有已完成 sourcebanks、teacher、manifest、seeds、metrics 定义、原失败 attempt 和用户 scope amendment 应只读保留。若新方案必须变动数据/训练预算/部署 backend，另起明确 experiment identity；不要覆盖旧 remote_root 并继续沿用旧 receipt。

本机可编辑文档不等于所有远端实验已停止；但 AQLM 本轮有明确用户停止 receipt，标准 LIBERO 也已完成，两者监控均已停。不要因文档更新重建这些监控或恢复 AQLM。现有历史监控里出现不同 Luna 版本是当时配置记录，不是要求新 agent 继承所有旧 scheduler 任务。Q/R 无实质变化保持安静，只有完成/失败/合法断点/需处理才通知；query failure 不授权重提。

复现的最低充分交接包为：源代码 revision 或 artifact、frozenprotocol 及 amendments、plan/manifest+datasetrevision、checkpoint/bank 和 teacher 位置、environment/assetreceipts、controllerhandle/intent、逐 unitrecords、最终 summary/完成证据，以及本文说明的数值与公平边界。缺其中一项就明确缺口，不以“README 写了路径”冒充完整可恢复证明。
