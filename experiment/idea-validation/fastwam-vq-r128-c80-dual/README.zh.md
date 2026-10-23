# C80 R128：冻结 R 的 Seq8 与 warm-start Joint8

用户授权单一双 GPU、walltime>2h 的并行任务，优先 g2，subagent 每 20 分钟监控。本轮复用已完成的 C80 W16A4 rotation checkpoint；共同拟合一次 R0 坐标下的 VQ0，再比较冻结 R 与联合微调。

实验已完成并核实：`25912619.pbs101`（attempt2）为 g2、双 A100 40GB、PBS F/Exit0，artifact exit0、PHASE_COMPLETE 和全部阶段/两臂 receipts 完整。两臂均完成8次更新、80 exposures，分别完整评估 calibration80 与 heldout80；没有新增 teacher queries。最终结果与原始训练记录已拉取至 [results/25912619.pbs101](results/25912619.pbs101)。

## 最终结果与解释

本轮应保留 **Seq8（冻结 C80 R0，再训练 VQ）** 为后续候选。Joint8 的 calibration 拟合略好，但 heldout 明显较差；本轮联合训练没有带来泛化收益。进一步核对实际执行源码后，确认当前 Joint 的 R 只有 activation 梯度路径，weight 没有配对补偿，因而固定 VQ 时的 R-only 更新也会改变浮点模型函数。它是未补偿的 compressed-student 适配方案，不能作为“等价 R+VQ 联合优化”的完整检验。详细公式、训练证据、测试盲点与改进边界见 [Joint8 深入诊断](JOINT8_DIAGNOSIS.zh.md)。

下面均为误差，越低越好；heldout 是同一40 contexts×2 seeds，先平均 seeds，再四 suite 等权。

| Heldout endpoint | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE |
|---|---:|---:|---:|---:|---:|---:|
| C80 R0-W16A4 | 0.042443858 | 0.034349046 | 0.003288021 | 0.044778372 | 0.045477245 | 0.037716236 |
| C80 R0+VQ0，共同未微调起点 | 0.048127906 | 0.059216201 | 0.004880400 | 0.051842591 | 0.052857261 | 0.044791690 |
| C80 Seq8 | 0.044207680 | 0.051400843 | 0.004175274 | 0.047952118 | 0.049053147 | 0.038028946 |
| C80 Joint8 | 0.045749947 | 0.059470054 | 0.004682388 | 0.049379018 | 0.050440249 | 0.049313403 |
| 原 C8 H+VQ8 | 0.044296774 | 0.060877371 | 0.004598652 | 0.047500980 | 0.048528504 | 0.057054031 |

- **Joint8 vs Seq8：** motor +3.489%、latent +15.699%、mixed +12.146%；四个 suite 的这三项均回退。Mixed 在40个配对 contexts中15个改善、25个变差。因此差异不限于单个 suite 的异常值。Full32/late22 分别 +2.976%/+2.828%。
- **Seq8 vs 共同 VQ0：** motor −8.145%、latent −13.198%、mixed −14.448%，mixed 有32/40 contexts改善。固定 R 后继续训练 VQ 有明确的描述性收益。
- **Seq8 vs 原 H+VQ8：** motor −0.201%，只宜称基本持平；latent −15.567%、mixed −9.207%。Full32/late22 则 +0.950%/+1.081%，并非所有 action horizons 都改善。Motor 在19/40 contexts改善、21/40变差；latent 在27/40改善。Motor 的总均值优势主要来自 libero_10，其余三个 suite 略差。
- **VQ 压缩代价仍在：** Seq8 相比同一 R0 的 W16A4，motor +4.156%、latent +49.643%、mixed +26.984%。不能把相对 H 的改善理解为已恢复 W16A4 fidelity。
- **Gripper 必须单列：** Seq8 vs H 的平均 gripper RMSE −33.346%，但仅12/40 contexts改善；Joint8 vs Seq8 平均 +29.673%，却有35/40改善。均值和配对胜负反向，说明少数大幅回退对均值影响很大，不能把任一平均值概括成普遍改善。Gripper 没有进入训练 loss。

Calibration 与 heldout 的对照尤其关键：

| Endpoint | Calibration motor | Calibration latent | Calibration mixed | Heldout motor | Heldout latent | Heldout mixed |
|---|---:|---:|---:|---:|---:|---:|
| Seq8 | 0.043426336 | 0.047780080 | 0.003954003 | 0.044207680 | 0.051400843 | 0.004175274 |
| Joint8 | 0.043118979 | 0.046901237 | 0.003889337 | 0.045749947 | 0.059470054 | 0.004682388 |

Joint8 能把 calibration 的 motor、latent、mixed 都拟合得略低，但这种优势没有推广到独立 demonstration/heldout seeds。这证明了端点排序反转，尚不能单独识别根因。代码已确认的未补偿R函数漂移、受限VQ补偿能力、量化优化和一般泛化问题可能共同作用；不能仅以calibration/heldout差距把原因定为普通过拟合。详细诊断中的等价性条件与新增机制探针，优先于继续扩大calibration或调参。

训练记录与数值检查没有发现梯度断连或部署坐标错误：两臂每次更新的614层 VQ 梯度均 connected/finite/nonzero；Joint 的 action/video R 梯度每次均连接。首次更新前两臂的指标及 VQ 梯度 norm 相同；进程内和跨 GPU 的固定数值探针在 forward、1230组梯度和 Adam delta 上 max_abs 均为0。Seq 的最终 R 与 R0 逐元素相同；Joint 的 R 按设计发生变化。两臂训练后所有参数都冻结，formal-infer parity max_abs=0，最大 orthogonality error=2.6226e-6。上述探针与检查通过，不等于已排除所有训练/泛化问题。

当前 Joint 的实现是 compressed student 共同适配：R 改变输入坐标，codebooks/gains 同时学习补偿，但 indices/B32 exponents 固定；它没有在每次 R 更新后显式保持浮点函数等价。因此，R0 在 W16A4 上有效，并不保证这条受限 VQ decoder 下的更新路径仍有效。A4 的离散分箱与 decoder 补偿受限，是值得验证的机制假设；本轮结果不能否定所有保持等价重参数化的 R+VQ 联合训练。

对“扩大 calibration”的判断应限定为：**C80 R0 + C80 VQ fitting/训练这一组合，能在冻结 R 后获得比旧 H+VQ8 更低的 latent/mixed error；但没有证明单独增加 weight fitting 的 calibration 数量就有收益。** 旧 H 使用 C8，而本轮同时改变 R、moments coverage、训练 contexts 和 batch 梯度平均方式，无法单独识别 calibration 数量或 R 的因果效应。要回答 weight 是否也需要 C80，后续应固定 R/训练预算后只改变 fitting moments 的覆盖；要检验 R 本身贡献，应增加相同 C80 配方的 H 对照。本轮没有追加这些实验。

若后续继续研究 Joint，应先明确 R-only 更新的等价性及 d4/B32 格式约束，建立带可微 weight 补偿的参考实现，再评估重新编码/投影回原 VQ 格式的代价。VQ warm-up、R anchor 和更新频率属于此后的稳定措施；不能用它们代替配对补偿。应使用独立 calibration validation 确定规则，避免继续依据当前 heldout 调参。这些是下一轮建议，尚未运行。

完整逐 suite、七项指标和七组配对比较见 [RESULTS.zh.md](results/25912619.pbs101/RESULTS.zh.md)；[comparison.json](results/25912619.pbs101/comparison.json) 是 compute allocation 内生成的原始聚合，未在本机重新计算。终态验证见 [verification.json](results/25912619.pbs101/verification.json)。本轮是单次训练的固定输入 decoded-BF16/A4 evidence，没有显著性、closed-loop 或 native-kernel 性能结论，也不是 equal-compute 比较。

成功 allocation walltime **03:41:47**，申请08:00:00并在完成后退出；另有下述两次准备失败，walltime分别00:07:56与00:39:21，均为双 GPU allocation。训练期间按15秒记录 GPU utilization/VRAM。监控已在成功终态停止，`nextcheckUTC=null`。期间两次工具会话切换导致监控 agent 消失并出现间隔，恢复记录保存在 RUN_STATE；这些间隔没有中断 PBS 作业。

## 准备失败与恢复记录

首个作业 `25910317.pbs101` 已在 g2 运行并以 Exit1 退出，allocation walltime00:07:56；第一条 moment 写入前缺少父目录，尚无正式训练更新。修复为在 compute worker 内先建立 moments/banks 输出目录，并允许在确认原 handle 终态失败后上传修复代码再 resume，仍保持冻结协议。

第二个作业 `25911189.pbs101` 完成80 moments和614 banks，common_eval随后因旧C80 teacher schema未单独保存seed_index而退出（walltime00:39:21，正式训练更新0）。原writer对新C80 records只保存完整plan_row与实际sampler_seed，seed0由冻结receipt声明；新loader仅在这一C80 seed0约束下接受字段缺省，仍严格检查plan与实际seed，heldout继续要求明确seed_index。没有更改teacher张量。22项标准库检查通过，moments/banks全部复用。原失败日志与终态分别保留于 `results/25910317.pbs101`、`results/25911189.pbs101`；实时进展以 `RUN_STATE.json` / `monitoring_state.json` 和当前handle的PBS/artifact为准。

## 固定问题和训练臂

R0 来自 `fastwam-r128-improvements` / `25905203.pbs101` / `coverage_c80/C80_once_lr1e4`：80 个 contexts、8 次平均 10 contexts 梯度的 Adam 更新，LR1e-4。R80 指 C80 calibration，并非 80 次更新。

| 项目 | Seq8 | Joint8 |
|---|---|---|
| 初始 R | 相同保存的 R0 | 相同保存的 R0 |
| 初始 VQ | 共同新拟合的 VQ0 | 同一份 VQ0 |
| R 更新 | 冻结，结束时逐元素核对 R0 | LR1e-5，额外8次更新 |
| VQ 更新 | LR5e-4，8次 | LR5e-4，8次 |
| 数据 | 同一原 C80 的8个10-context batches | 相同 batch / 顺序 / seed |
| optimizer | 第二阶段 fresh Adam | 第二阶段 fresh Adam，不继承 W16A4 moments |
| 更新参数 | BF16 decoder 对应的 codebooks FP32 masters、log row gains | 左侧参数 + 两路 stream-shared Cayley raw A128 |

每个 context 等权，先累积 10 个 loss/10 的梯度，再按 R/VQ 两组分别 norm clip 1，然后更新一次。沿用 VQ update1 后清一次 VQ Adam state 的规则；R state 不清。固定 final8，不从 heldout 选 checkpoint 或 LR。

原模型、Smooth0.5/signs、D=I、clipping ratio=1、BF16 KV、VQ indices、B32 exponents 全程冻结。采用 d4、M2、K256 的 BF16 codebooks/row gains 和 uint8 power-of-two exponents，实际算术为 decoded BF16 dense Linear + dynamic per-token signed A4(qmax7)。没有 native low-bit kernel。

Loss 延续 `first10×6 motor MSE + 0.04245452683169643 × clean-future latent NMSE`，rho1.5 与 normalization c 固定。Latent 排除 clamped frame0，student future 保持 action 梯度连接。Gripper 不加入本轮 loss，但必须独立报告回归，不能用 mixed 改善宣称完整 action 改善。

Joint 的 forward 为 `A4((X/S) H U(A)) @ Decode(C,g,I,e).T`。A 从保存的 R0 raw parameters 开始，不重置为 I，也不在 decoded weight 上重复右乘 U。它是 compressed student 的共同适配，不要求每步保持原 FP 函数等价；训练后直接保存编码参数，不再 refit。

## 共用准备与双 GPU 安排

单个 normal routing 作业申请 2 GPUs、32 CPUs、220 GB、8h，实际进入 g2 后才允许启动。两 GPU 各绑定独立 UUID、每 worker 16 CPU threads。

1. `moments`：两 shard 各收集40个 R0-W16A4 contexts 的614层输入 second moments；每 context 先按有效 token rows 归一化，再在80 contexts间等权。
2. `fit`：共享80个 moments，两 shard 各拟合307层 `W S R0`，sample_limit8192、iterations8、seed20261006；每个 bank 只有一个写入者。
3. `common_eval`：两个独立 Runtime 各评 R0+VQ0 的40 calibration与40 heldout queries；共同覆盖完整两 cohort。
4. `gate`：核对全部80 moments、614 banks、保存的 R0 重放，以及重复数值探针。门限通过才进入正式训练。
5. `arm`：GPU0 运行 Seq8，GPU1 同时运行 Joint8，分别保存 update/Adam/RNG/checkpoint 与 query records。
6. `finalize`：完整聚合两新臂、共同 VQ0 和两个已有参考端点。

准备阶段的分片结果共享，两个正式训练进程使用独立 model/Runtime，避免 arm 间隐藏状态残留。完成或失败及时退出，不为了填满申请时长等待。

## 数值预检

此前 W16A4 与 VQ 训练都出现过相同已记录前向指标之后的更新分化。本轮显式关闭 TF32，使用 deterministic algorithms、CUBLAS_WORKSPACE_CONFIG=:4096:8、固定 seeds/hash seed。

两进程均在相同 R0+VQ0、case2000 上重复两次完整 forward/backward/Adam step，保存完整 action/future、各参数梯度、参数 delta、更新后的 action/future；探针更新全部丢弃。验证进程内重复及跨 GPU 一致性。输出 max_abs≤1e-6，gradient atol1e-7/rtol1e-4，parameter delta atol1e-7。若实际 CUDA 算子不支持确定性或检查失败，保留证据并修复，不静默放宽门限或继续正式训练。

小型 GPU 数值 selfchecks、source checkpoint schema、614层梯度连接/finite、R orthogonality≤5e-5、最终 sampler/formal-infer max_abs≤1e-6 亦为验收项。

## Cohorts、对照和结果解释

- Calibration：固定 cases2000..2079，seed index0，4 suites各20 contexts，每 task两个不同 demonstrations。
- Heldout：固定 cases3000..3039，seed indices0/1，40 contexts×2=80 queries，每 task使用独立 demonstration。
- 每个新端点 R0_VQ0、Seq8、Joint8 均完整评 calibration80 和 heldout80；保留相同输入/seed/BF16 action+future teachers，新增 teacher queries=0。
- 复用已有 R0_W16A4 和原 H_VQ8 的各80条 heldout，验证完整来源绑定。原 H_VQ8 使用 C8，属于历史 incumbent，不能将本轮差异归因为单一 R 效应。
- 聚合先 context内平均seeds，再suite内平均10个heldout contexts，最后四suite等权；独立配对单位是40 contexts。

主比较为 Joint8 vs Seq8；同时报告两臂 vs R0_VQ0 的训练收益、vs R0_W16A4 的压缩代价、vs H_VQ8 incumbent 的表现。完整报告 first10 motor RMSE/MSE、future latent NMSE、mixed、full32/late22 motor、gripper 及逐suite/context配对变化。仅 mixed 改善而 motor/latent/长horizon/gripper 回退时，必须明确报告取舍。

两臂 VQ 更新次数和 exposures 相同；Joint 额外更新 R，不能称为 equal-compute/walltime。单次训练没有 run-to-run 方差或显著性结论；既有 heldout 已被多轮查看，后续新协议确认优于最终部署结论更合适。本轮不扩大 cohort、不按 heldout 追加训练。

## 控制、恢复与交付

`control.py upload/submit/status/verify/pull/resume` 使用现有可信 SSH helper。Submission intent 在 qsub 前保存，live/unknown handle 禁止重复提交；只有确认原 handle 终态才可 resume。每个 preparation 单位与每次 optimizer 更新原子保存，完成阶段与完成 queries 可复用。8h allocation 保留 checkpoint 时间。

终态成功要求 scheduler F/Exit0、artifact exit0、PHASE_COMPLETE、root与各stage/arm完整summary、完整cohort分母和数值门限同时通过。GPU utilization/VRAM 每15秒写入 job log；gpt-6-luna/xhigh subagent每20分钟轻量查询，Q/R无实质变化保持安静。

Login node 仅用于连接、qsub/qstat、小型代码/JSON控制；所有模型、Torch数值、tensor/大文件读写、VQ fitting和聚合均在真实PBS compute allocation中进行。保留SSH host-key验证，不处理banked reset、不输出credentials。

最终交付 `results/summary.json`、`comparison.json`、`RESULTS.zh.md`、两臂deployment/checkpoints、训练记录和所有阶段/数值gate receipts。证据为固定输入 BF16/A4 fidelity，不包含环境 episodes、closed-loop success 或 native-kernel 性能。
