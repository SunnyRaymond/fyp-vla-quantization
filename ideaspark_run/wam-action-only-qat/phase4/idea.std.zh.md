> **FAILED VALIDATION（验证未通过，未做实验验证）**：Phase3 post-revision 的 F1 仍为 `needs_work`。原审查指出 V_visit 缺少完整前缀时间归约和固定估计来源。Phase4 补充了对所有 native 执行前缀时间步与所选动作通道求均值的标量定义，并固定使用 pre-QAT matched-PTQ reference；这不改变也不通过原门槛。
> 发布校验还记录了 `kill_switch_integrity`：其比较 Phase2 与 Phase4 的 falsification 文本。直接比较显示 Phase3 `final_candidate.json` 与 Phase4 相同；该校验未反映获准的 Phase3 对照臂修订。保留此失败记录。

# 面向 World-Action Models 的学生路径后缀 QAT

**方法名称：** Student-Path Suffix QAT (SPS-QAT)

## 研究动机
现有低比特结果很强，因此我们仍不知道：训练 WAM 的权重，是否能在强 PTQ 和匹配教师目标之外带来额外收益。QAT 是在模拟低比特运算时更新权重；PTQ 则保持预训练权重不变。近期 WAM 工作报告了较强的低比特控制结果，所以本提案不假设 W4A8 已经失效。

初始平台是 Fast-WAM Base：约 1B 参数的 ActionDiT 可作为权重范围内的 action-only 目标，同时约 5B 参数的视频 expert 保持 BF16。向 LingBot-VA 和 Cosmos Policy 的迁移只是分别验证、明确标注范围的假设。它们使用共享 backbone；更新共享 denoiser 权重会同时影响视频和动作计算，因此属于 action-targeted shared-backbone QAT，不是权重范围内的 action-only 量化。

方法把 BF16 teacher 和 fake-quantized student 放在同一个状态继续运行。这个状态由 student 自己走到，随后从该状态分离梯度，并让两者使用相同 context、相同剩余噪声更新，完成原生 solver 的整个剩余 suffix。Teacher endpoint 是模型给出的目标，不是从原始噪声端点做 KD，也不是物理 oracle。之后用原生 action denormalization 和 executed-prefix selector 转成控制器要执行的动作。训练信号是逐通道缩放的连续命令残差，计算在 clipping 或 gripper threshold 之前；实际裁剪或阈值化后的命令另行测量。

它与 vanilla QAT 加匹配 teacher KD 的可检验区别，是使用从 student 访问状态得到的完整剩余 suffix endpoint 作为目标。这只是目标设计的假设，不能称为已证实的新颖性，也不能称为首个 WAM QAT。另一个经验预测是：相对 matched teacher-state KD 的 completion 增益，可能随 $`V_{visit}`$ 增大。$`V_{visit}`$ 是每项任务的诊断值，在一组 held-out contexts 上测量，并对所有 arm 使用同一个 QAT 前的 matched-PTQ reference；不会给每个 arm 单独做处理后排序。Phase4 对其定义的澄清没有通过此前的 gate，post_revision 状态仍是 FAILED VALIDATION。

## 方法
### M1_background
*记录各模型契约，并生成 fake-quant student 自己的 solver 路径。*

1. 为 Fast-WAM Base、共享 backbone 的 LingBot-VA 和 Cosmos Policy 记录各自的 BF16 checkpoint、原生 solver schedule、action maps、executed-prefix selector、匹配的 branch-free W4A8 modules、排除的精度范围和部署 kernel。Fast-WAM 的独立 ActionDiT 属于权重范围内的 action-only；另外两个属于 action-targeted shared-backbone scope。
   - _为什么：_ 只有 scope、schedule、精度、action interface 和部署运算都匹配，比较才能隔离权重适配带来的效果。
2. 按 episode 划分每个 LIBERO 数据集。从训练集抽取 context、原生初始噪声和 solver 更新，让 fake-quant student 按原生 schedule 运行并记录访问过的状态。Fast-WAM 的状态是 action latents；共享 backbone 模型的状态同时包含视频和动作。
   - _为什么：_ 训练目标来自量化 student 自己访问的状态；固定 context 和噪声便于配对比较，但不模拟物理动作执行后的环境观测。

### M2_suffix_target
*从 student 访问状态出发，用配对的完整 suffix endpoints 训练。*

3. 从原生 solver indices 中抽取一个位置，对 student 访问状态停止梯度，并让 BF16 teacher 与 fake-quant student 从这个相同状态重新开始。两者使用同一 context 和剩余原生噪声更新，分别运行未改动的完整剩余 suffix，并保存两个 endpoint action chunks。

*在同一学生到达状态和后续 solver innovations 下，分别运行 PTQ student 与 BF16 teacher 的剩余 suffix，再经原生动作映射和 executed-prefix 选择得到逐时刻、逐通道差值。*
$$ \delta^{\mathrm{PTQ-ref}}_{m,k,u,j}=\left[E_m\!\left(D_m\!\left(S[Q_{\mathrm{ref}}|Q_{\mathrm{ref}},m,k](x_{Q_{\mathrm{ref}},m,k},c,\eta_{>k})\right)\right)-E_m\!\left(D_m\!\left(S[\mathrm{BF16}|Q_{\mathrm{ref}},m,k](x_{Q_{\mathrm{ref}},m,k},c,\eta_{>k})\right)\right)\right]_{u,j} \tag{1} $$

   - _为什么：_ 这样测的是 student 访问状态上的 suffix response，而不只是原始噪声端点或 teacher 访问状态上的结果。
4. 使用各模型原生的 action denormalization 和 executed-prefix selector 转换两个 endpoint。在 clipping 或 gripper threshold 之前，用连续命令计算训练残差；按固定的正逐通道 controller scale 缩放，并对所有原生 prefix steps 和选定 channels 求平均。只沿 student suffix 反向传播。实际裁剪或阈值化后命令的差异另按原生单位报告。

*训练损失对完整 native executed prefix 的所有时刻和选中动作通道取均值，并按固定的模型/控制器通道尺度归一化；梯度只回传至 student suffix。*
$$ \mathcal{L}_{\mathrm{train},m}=\mathbb{E}_{c,z,\eta,k}\!\left[\frac{1}{T_{\mathrm{exec},m}|J_m|}\sum_{u=1}^{T_{\mathrm{exec},m}}\sum_{j\in J_m}\left(\frac{\tilde y_{Q|Q,m,k,u,j}-\tilde y_{\mathrm{BF16}|Q,m,k,u,j}}{s_{m,j}}\right)^2\right] \tag{2} $$

   - _为什么：_ 连续训练信号可求梯度；控制器实际执行的精确命令是另一项结果，两者不能视为相同。

### M3_validation
*比较匹配训练对照、分别报告结果，并检验有范围限定的迁移。*

5. 与匹配的 branch-free W4A8 PTQ、vanilla action-only QAT 加 teacher KD、原始噪声端点 KD、teacher-state suffix KD 和 student-state 逐步 OPD 比较。若 teacher-state 两个 arm 的目标相同，就只计作一个 arm。预算按实测 GPU-time 或 teacher 与 student 的总 NFE 对齐，并报告两类计算量、样本数、context 曝光、更新次数和 teacher suffix 工作量。共享 backbone 的 OPD 使用 action-output loss mask 选择动作通道和 prefix steps；这个 mask 不会让共享权重变成 action-only。
   - _为什么：_ 这些对照检验完整 student-state suffix endpoint 是否比普通 QAT、PTQ、端点或 teacher-state 蒸馏、逐步 student supervision 更有价值，同时如实计入 teacher 成本。
6. 在 held-out fixed contexts 和配对噪声上分别报告精确 executed-action 偏差、LIBERO closed-loop completion 和完整原生 action-query latency；latency 包含 video/context prefill，且不调用 teacher。$`V_{visit}`$ 对原生 prefix 的完整 time-by-channel response 求标量，在 held-out context 子集上估计，所有 arm 共用同一个 QAT 前的 matched-PTQ reference。要做整体 query 加速推断，先测 action-time share，再用 Amdahl's law。

*Phase4 补充定义：将 exact executed-command response 对完整 native 时间步与动作通道前缀求均值，形成标量；所有对照臂都使用同一个 QAT 前 matched-PTQ checkpoint 估计。本定义不代表 post_revision gate 已通过。*
$$ \mathcal{V}^{\mathrm{PTQ-ref}}_{\mathrm{visit},m}=\mathbb{E}_{c\sim\mathcal{D}^{\mathrm{eval}}_m,z,\eta,k\sim\operatorname{Unif}(K_m)}\!\left[\frac{1}{T_{\mathrm{exec},m}|J_m|}\sum_{u=1}^{T_{\mathrm{exec},m}}\sum_{j\in J_m}\left(\frac{\delta^{\mathrm{PTQ-ref}}_{m,k,u,j}}{s_{m,j}}\right)^2\right] \tag{3} $$

   - _为什么：_ 命令保真度、任务完成率和部署延迟回答不同问题。$`V_{visit}`$ 是固定 reference 下的诊断量；它与收益的关系仍是假设。Phase4 的澄清没有通过此前 gate，post_revision 状态仍为 FAILED VALIDATION。
7. 在 LingBot-VA 和 Cosmos Policy 上，按各自原生 scope 单独做 shared-backbone 迁移检查。明确共享 denoiser 更新会影响视频和动作计算，并将结果与 Fast-WAM 分开报告。
   - _为什么：_ 这检验有明确范围的迁移假设，不把共享权重说成权重范围内的 action-only，也不混合不同模型契约的结果。

## 可证伪性与测量

检验相对 matched teacher-state KD 的 closed-loop completion 增益，是否随任务级 `V_visit` 增大。`V_visit` 在 held-out context 子集上对完整原生 prefix time-by-channel response 求值；所有 arm 使用同一个 QAT 前的 matched-PTQ reference，不按处理后的 arm 分别排序。对照包括 matched PTQ、vanilla QAT 加 teacher KD、原始噪声端点 KD、teacher-state suffix KD 和 student-state 逐步 OPD。精确 executed-command 偏差和完整原生 action-query latency 也要分别报告。材料没有给出数值通过阈值。若相对 matched teacher-state KD 没有 completion 增益，或没有预测的 `V_visit` 关联，就不能支持这项预测；保真度或 action block 变快本身不能证明控制收益。Estimator 的澄清没有通过此前 gate，post_revision 状态仍为 FAILED VALIDATION。

## 资源与可行性

原始和当前 campaign 成本都是未知 GPU-days，因为没有测量 campaign hours、每秒样本数或 suffix-training throughput；本次没有运行 GPU 实验。资源上限是最多四张彼此独立的 A100-SXM4-40GB，显存不是合并的 160GB。Fast-WAM 的约 5B BF16 video 加 1B BF16 action core，权重约占 12GB。只把 action 权重转为 W4 后，整体约 10.5GB；理想化的 whole-core 权重缩减约为 1.14 倍，不是整个模型缩小 4 倍。这些只是权重存储估算，不代表训练显存或吞吐已验证。梯度、optimizer state、teacher 和 student suffix 计算、activations、cache 和 workspace 都会增加成本；shared-backbone 适配还可能增加 activation 和 teacher 显存需求，并同时改变视频与动作计算。LingBot-VA 和 Cosmos Policy 只是范围明确的迁移假设；不能用论文里的独立模块推断 LingBot checkpoint 大小，action-output mask 也不等于 action-only 权重。整体 query 加速需实测 action-time share，再按 Amdahl's law 推断。目前可行性仍是条件性的，不宣称 SOTA 或虚构 GPU-days。
