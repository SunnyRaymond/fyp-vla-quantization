# R128 旋转幅度、更新方式与 calibration 覆盖实验

本轮三项实验已完成并验收，均为纯 **W16A4**：固定原始BF16 weights、Smooth0.5/signs、BF16 KV、D=I和clipping ratio=1，只优化两路stream-shared Cayley U128。未引入VQ。

**主要结果：C80相对H的heldout motor/latent/mixed为-0.38%/-10.50%/-5.36%，相对原C8-R8为-4.80%/-14.86%/-12.10%；但gripper相对H为+25.08%，37/40个context变差。motor平均收益很小，不能认定整体action稳定优于H。**

第3项按用户要求只训练C80，不做C8等处理量重训。C80采用80个不同context、8次10-context平均梯度更新、LR1e-4；这是整体recipe比较，不能单独归因于calibration数量。较小LR和旋转收缩缓解部分退化，但各C8新臂仍未同时超过H。

完整解读见[最终分析](results/final/ANALYSIS.zh.md)，全臂数值与case/suite数据见[总表](results/final/RESULTS.zh.md)和[comparison.json](results/final/comparison.json)。另外发现相同首步已记录前向标量之后的gradient/update数值差异，顺序/LR比较需保持描述性；详见[首步诊断](results/optimization/25905200.pbs101/INTERPRETATION.zh.md)。

path 25905198、optimization 25905200、coverage 25905203均为g1、F/Exit0/verified；单GPU allocation合计5小时41分26秒。各臂完整评估C8 8/8和heldout80/80，C80臂另完成C80 80/80，新增teacher为0。20分钟gpt-6-luna/xhigh subagent监控已结束；[最终状态](RUN_STATE.json)。19项stdlib检查通过，模型/GPU计算均在PBS allocation内执行。

## 问题与公平条件

已有 R8 在完整 C8 上 mixed loss 下降17.98%，但收益主要来自 case2019；其余七个 context 等权 mixed loss 反而上升10.62%。完整 heldout motor RMSE 上升4.65%。本轮分别检查最终旋转幅度、更新顺序/LR，以及独立 calibration context 的多样性。

所有臂使用相同 teacher、冻结 Smooth、first10×6 action MSE + 0.04245452683169643×clean-future latent NMSE（排除clamped frame0）。保留同一40 contexts×2 seeds的完整配对评估和原 H/R8 对照。所有预先固定的臂均报告；不使用heldout选择checkpoint、LR、t或最佳臂。checkpoint固定update8。

| 实验 | 臂与控制 | 训练预算 | 要回答的问题 |
|---|---|---|---|
| 1，path | t=0、0.25、0.5、1；0/1复用已有端点并做重放检查 | 无新训练 | 沿同一已学方向减小幅度是否改变泛化表现？ |
| 2，optimization | grouped/interleaved × LR1e-3/1e-4；grouped1e-3复用原R8 | 各8 samples、8 Adam updates | 分别区分顺序和LR影响 |
| 3，coverage | 只训练C80，复用现有C8-R8与H比较 | 80个不同contexts、8次10-context平均梯度更新、LR1e-4 | C80完整recipe能否改善现有端点？ |

Path 使用 **U(t)=Cayley(t(A8−A8ᵀ))**，以原定义中的0.5系数计算；不是对U矩阵做线性插值。输入和weights成对使用同一个U(t)。

原C8顺序：2000,2019,2021,2025,2051,2057,2066,2070。
交错顺序：2000,2021,2051,2066,2019,2025,2057,2070。
Coverage每个update有10个context，四suite按3/3/2/2轮转，每个suite全程20次曝光，平均10个loss后更新Adam。按用户最新指示，C8重复曝光控制已在提交前取消。原C8-R8使用8个context、8次单context更新、LR1e-3；因此C80对照同时涉及数据覆盖、LR和梯度累积方式，不能将差异只归因于calibration数量。

## 数据与复用

- C8固定为8个contexts、seed index0；C80固定为2000..2079、seed index0，C8嵌套其中。
- Heldout固定为3000..3039、seed indices0/1；每个task使用不同于calibration的demonstration。
- Heldout按seed→context→suite聚合，四suite等权；独立单位为40contexts，不将80queries当成80独立contexts。
- Path/optimization最终评完整C8与heldout；C80训练臂另评完整C80，C8与C80重叠端点可以复用。
- 原R8来源：25902163.pbs101；完整H/R诊断来源：25903221.pbs101。
- C8/C80/heldout teacher全部复用已有BF16 action+future cache。C80 receipt来自25853617.pbs101；缺失或绑定不符直接失败，不生成新teacher。

冻结plan的实际覆盖为：

| Calibration | 独立contexts | tasks | early / middle / late | 每suite contexts |
|---|---:|---:|---|---:|
| C8 | 8 | 8 | 2 / 5 / 1 | 2 |
| C80 | 80 | 40 | 28 / 28 / 24 | 20 |

C80为每task两个不同demonstration的snapshot；heldout使用该task的第三个demonstration。seed index0是各context预先固定的第一个sampler seed，并非所有context使用同一个数值seed。

具体case、source paths、训练参数和门限以 [protocol.json](protocol.json) 为准。

## 运行与监控

实时队列规则显示g1接受1GPU、g2接受2..3GPU、g3固定4GPU，且walltime>2h；execution queue通过normal路由进入。按最终范围，本轮三个作业均申请4小时、单GPU、目标g1：path、optimization和coverage（只运行coverage_c80）。查询确认实际路由。允许调度器并行运行，完成或失败及时退出。

所有模型加载、推理、训练、tensor读取和结果聚合都在真实PBS compute allocation内进行。Login仅用于受信SSH连接、qsub/qstat、小型代码和JSON/log读取。每个作业每15秒写入GPU utilization和显存样本。运行脚本验证PBS_JOBID、nodefile、实际hostname和qstat Running状态。

三个作业使用独立handle与job name，观察超时不触发重提。只有确认原job终止并且checkpoint可恢复时才resume。每个arm分别保存rotation/Adam/RNG和已提交更新数，复用已验证完成的query records。

使用gpt-6-luna/xhigh subagent每20分钟监控；状态无变化的Q/R保持安静，发现完成、失败或需处理事件才通知主代理。

## 交付与边界

每lane产出summary/progress/training/selfcheck/comparison JSON及RESULTS.zh.md。总报告合并三个lane，保留完整配对表、suite表、context统计和C8收益集中度诊断。

这轮提供固定输入BF16/A4数值证据；未执行环境episode，不代表closed-loop success、native kernel、延迟收益或equal-walltime结论。C80为用户指定的整体recipe扩展，不要求与原C8-R8等训练样本处理量。

## 结果解释规则

| 问题 | 主要读取的证据 | 结论边界 |
|---|---|---|
| 是否偏离H过多？ | 四个固定t的完整C8/heldout曲线、U−I幅度、case2019收益占比 | 较小t改善支持沿当前方向收缩；不能仅据此断言LR就是唯一原因 |
| 更新顺序还是LR？ | 原grouped1e-3加三条新臂形成2×2；同时读同LR顺序对照和同顺序LR对照 | 同时报告交互；只有一次训练顺序实例，不作跨训练随机性的显著性结论 |
| C80方案是否改善原端点？ | C80与原H/C8-R8完整heldout对照，以及新C8和C80端点、逐context梯度/训练记录 | 用户指定的整体recipe比较；数据量、LR和平均梯度方式共同改变，不作单因素因果归因 |
| 改善是否仍由少数context支撑？ | C8逐case表、case2019对净mixed-loss变化的贡献、四suite及C8覆盖task/未覆盖task分组 | 不剔除case改变主指标；净改善很小时贡献比例不稳定，同时报告绝对变化 |

所有arm与checkpoint均固定后完整报告。Heldout不用于触发更多训练、挑选t/LR或提前停止；负面结果同样交付，后续实验需另行确定问题。
