# AD-WM：Action-Discriminative World Models for Counterfactual Model Predictive Control

## 论文身份与本地材料

- **编号 / 版本：** #033，arXiv `2609.30264v1`，2026-09-24 版本；本文是 arXiv preprint。
- **作者：** Jiabin Qiu、Zixuan Chen（共同一作）、Hongye Cao、Jieqi Shi、Jing Huo、Yang Gao；Nanjing University。
- **本地论文：** [paper-arxiv-v1.pdf（9 页）](paper-arxiv-v1.pdf)。下文页码均指该 PDF 的页码。
- **来源：** [arXiv 摘要页](https://arxiv.org/abs/2609.30264v1) · [arXiv HTML](https://arxiv.org/html/2609.30264v1) · [arXiv v1 PDF](https://arxiv.org/pdf/2609.30264v1) · [项目入口](https://ad-wm.github.io/)。论文摘要称视频和代码可从项目入口获取；本阅读包没有验证该网页当前是否可访问，也没有确认或填写任何代码仓库地址。
- **阅读范围：** 本地完整提取文本（含正文、表格和参考文献）已读；本包不包含代码审查或实验复现。

## 先抓直觉：预测“发生了什么”与判断“做哪个动作”不同

离线训练数据记录的是实际执行过的动作及其后果。普通 world model 因而主要被要求预测这些**事实转移**。MPC 在同一个当前状态下，却要比较许多没有同时发生过的候选动作序列：哪个会更接近目标？这是**反事实动作选择**。

如果视觉 latent 大部分由稳定背景和物体外观构成，而一步动作只造成很小的变化，那么模型即使近似复制当前 latent，也可能得到不错的一步预测误差。可是在规划时，不同动作的预测会变得相似，模型就无法可靠地区分“接近黄色方块”和“离开方块”。论文把这两件事的落差称为关键问题：factual prediction accuracy 不足以保证 counterfactual action discrimination。（引言与问题形式化：[第 1 页](paper-arxiv-v1.pdf#page=1)、[第 2 页](paper-arxiv-v1.pdf#page=2)）

**一句话概括：** AD-WM 在训练时要求模型从自己预测出的状态转移中恢复动作信息；部署时仍用原有 latent-space MPC/CEM，训练用的辅助 recovery heads 会被丢弃。（方法图与训练/规划：[第 3–4 页](paper-arxiv-v1.pdf#page=3)）

## 问题设定与方法

### 1. 规划目标

图像经 encoder 得到状态 latent `z_t`，目标图像得到 `z_g`。模型对候选动作序列 rollout，并用终点 latent 距离评分：

```text
predicted terminal cost = || z_hat_(t+H) - z_g ||²
```

MPC 反复采样候选、保留低成本序列、更新分布，执行第一个 action block，再从新观测重新规划。关键不是只预测已记录到的下一个 latent，而是让不同动作的 rollout 在规划所用的 latent 空间里保留有用差异。（第 2 页）

### 2. Residual latent dynamics

把共享的当前状态内容留在 `z_t`，预测器只估计局部变化：

```text
e_t = ψ(a_t)
Δz_hat_t = f(z_t, e_t)
z_hat_(t+1) = z_t + Δz_hat_t
L_pred = || z_hat_(t+1) - z_(t+1) ||²
```

在无限制函数类里，absolute prediction 与 residual prediction 的最优解相同；residual parameterization 改变的是学习偏置，显式聚焦 latent increment。它本身并不保证模型保留动作信息。（第 3 页）

### 3. Predictor-level action recovery

默认 inverse-dynamics head 看见 `(z_t, z_hat_(t+1))`，尝试恢复动作 embedding `e_t`。目标 `e_t` 会 detach，但预测端点保持可微，因此 recovery loss 可将梯度传到 dynamics predictor 和 action encoder。这样约束的是 MPC 将要使用的**模型生成转移**，而非只在数据中的真实转移上学习 inverse dynamics。（第 3 页）

论文另加 normalized action recovery。它用 batch mean/std（detach 后）逐维标准化动作 embedding；独立 head 给出单位协方差高斯 `q_η(ē_t | z_t, z_hat_(t+1)) = N(μ_η, I)`，并以标准正态为 reference：

```text
L_MI = -log q_η(ē_t | z_t, z_hat_(t+1))
       + β KL(q_η || N(0, I))
```

在单位协方差下，忽略常数后等于 `½||ē_t - μ_η||² + β/2||μ_η||²`。固定状态/动作表征与 target normalization 时，recovery log-likelihood 对应 Barber–Agakov conditional-MI lower bound 中的变分项；联合训练另含 KL regularization，表征和归一化统计也在变化。因此 **MI 是有信息论动机的 recovery regularizer，不是完整联合训练过程严格最大化条件互信息的保证**。（第 3–4 页）

模拟训练联合优化 `L_pred + λ_sig L_sig + λ_inv L_inv + λ_MI L_MI`；使用与 LeWM 相同的 SIGReg。机器人 post-training 冻结 encoder、去掉 SIGReg。部署时丢弃两个辅助头，预测器和 MPC/CEM 接口仍需经过 AD-WM 训练；论文没有报告由此带来的推理加速。（第 4 页）

### 4. 规划诊断：看预测的 elite 是否真有用

作者在共享候选动作池上，用同一初始环境状态执行动作序列，取得真实终点成本，再与模型预测成本比较。指标包括：

- **CAD：** 全候选池 predicted cost 与 realized cost 的 Spearman 相关；它也统计不会进入 CEM elite 的候选。
- **Best-in-elite regret `R_k`：** 模型选出的最低成本 `k` 个候选中，最好那个真实成本与整池最佳真实成本的归一化差距。
- **Elite-mean regret `R̄_k`：** 模型所选 elite 的平均真实成本，相对真实 top-k 的平均成本差，按候选池真实成本范围归一化。

这类 fixed-bank 指标能检查模型挑出的候选质量，但候选池固定，未模拟 CEM 每轮更新分布后会访问到的新候选；它们**不保证 adaptive CEM 轨迹、first action 或 closed-loop 成功**。（定义与边界：第 4 页；诊断设计与结果：第 6–7 页）

## 关键创新与证据边界

1. **训练目标对准动作比较。** 在预测损失之外，对 predictor 自己生成的 latent transition 做 action recovery，以补充 factual supervision。
2. **残差动态与 recovery 联合。** residual prediction 聚焦局部 latent 变化；recovery 再要求变化携带可恢复的动作信息。两者的贡献需结合 ablation 读，不能把全部收益都归到 MI。
3. **区分全池排序和 CEM elite 质量。** 在本文固定 bank 诊断中，elite regret 与 hard-start success 的关联强于 factual MSE 或 CAD；这是诊断证据，不是对任意规划器的定理。
4. **规划器不改。** AD-WM 修改训练出的 latent dynamics；部署仍执行 MPC/CEM。辅助 heads 不参与部署。

## 主要数值结果

### Cube：匹配 LeWM 对照和 hard starts

Table I 在相同 Cube starts 上比较不同模型。P00–P04 是逐渐增大的 cube `xy` perturbation；**hard-start (HS) 是这五列的平均，P00 也不同于 Original**。匹配的 AD-WM/LeWM 均使用 CEM；表中 external methods 使用各自原生 inference，不应当视为同等受控的训练/推理对照。（Table I，第 5 页）

| 方法 | Original | P00 | P01 | P02 | P03 | P04 |
|---|---:|---:|---:|---:|---:|---:|
| LeWM | 73.3 ± 2.5 | 8.0 ± 2.8 | 4.7 ± 0.9 | 4.7 ± 1.9 | 1.3 ± 1.9 | 0.0 ± 0.0 |
| AD-WM | 90.7 ± 3.4 | 74.0 ± 2.8 | 68.0 ± 4.3 | 56.0 ± 1.6 | 36.7 ± 5.2 | 25.3 ± 3.4 |

Cube 每个 checkpoint 每个 protocol 为 50 episodes；表中主要结果对 seeds 3072/4096/6144 报 mean ± population SD。HS 从 LeWM 的 **3.7 ± 1.4%** 升至 AD-WM 的 **52.0 ± 3.1%**。（实验协议：第 4–5 页；组件表：第 5 页）

### 组件消融：默认 full model 并非最好的一格

Table II-A 的 HS：

| 变体 | Original | HS |
|---|---:|---:|
| LeWM | 73.3 ± 2.5 | 3.7 ± 1.4 |
| Absolute + Inv + MI | 83.3 ± 0.9 | 14.4 ± 2.9 |
| Residual | 82.7 ± 2.5 | 34.7 ± 1.6 |
| Residual + Inv | 83.3 ± 1.9 | 37.1 ± 4.7 |
| Residual + MI | 89.3 ± 3.8 | **54.7 ± 3.0** |
| AD-WM（Residual + Inv + MI） | **90.7 ± 3.4** | 52.0 ± 3.1 |

因此 residual 和 MI 是主要贡献项；默认权重下加入 Inv 后的 AD-WM 平均 HS 低于 Residual + MI。Table II-B 中，MI 对三种 Inv 输入都增加 HS 均值：预测端点 `37.1→52.0`、编码端点 `45.1→60.7`、预测增量 `39.9→60.3`；该 panel 改变的是 Inv 输入，MI 始终使用预测端点。

论文事先指定的模拟默认权重为 `λ_inv=0.1, λ_MI=0.01, β=0.01`（Scene 的 `λ_MI=10⁻⁴`）。后续权重敏感性中 `λ_MI=0.03` 的 HS 均值为 **65.2%**，高于预设默认值 `52.0%`；这是 sensitivity sweep 的更优一格，不是预先指定的主结果。Inv 权重效应较小且随权重变化。（Table II，第 5 页；解释，第 6 页）

### 五个模拟环境：PushT 略降

| 环境 | reproduced LeWM | AD-WM | 论文中的匹配对照变化 |
|---|---:|---:|---|
| OGBench-Cube（Original） | 73.3% | 90.7% | +17.4 pp |
| Reacher | 76.7% | 83.3% | +6.6 pp |
| PushT | **94%** | **92%** | **−2 pp** |
| TwoRoom | 90% | 98% | +8 pp |
| Scene | 35.5% | 39.5% | +4 pp |

AD-WM 的平均 success 在五个模拟环境中的四个高于 reproduced LeWM；**PushT 是下降，不是提升**。Scene 的总体差异在配对 seed 检验中未解决（`p=0.13`）；Scene 使用 balanced hard starts，其余环境为原 protocol。不要把这张图和 external reported baselines 当成统一设置下的比较。（Figure 4，第 6 页）

### Prediction、selection 与 control 并不同步

Table III 将实际 MSE 乘以 `10³` 后展示；下表的 2.72 对应实际 MSE `0.00272`。Cube 三个 seed 的均值如下：

| 变体 | One-step MSE | Local MSE | CAD | `R₃₀` | `R̄₃₀` | HS |
|---|---:|---:|---:|---:|---:|---:|
| LeWM | 2.72 ± 0.06 | 10.36 ± 0.48 | 0.460 ± 0.010 | 0.074 ± 0.006 | 0.298 ± 0.010 | 3.7 ± 1.4% |
| AD-WM | 4.27 ± 0.10 | 15.77 ± 0.20 | 0.448 ± 0.011 | 0.029 ± 0.001 | 0.229 ± 0.005 | 52.0 ± 3.1% |

AD-WM 的 factual MSE 更高、CAD 略低，但 HS 大幅更高且 elite regrets 更低。在 15 个 model–seed observations 中，HS 与 CAD 的相关是 `−0.399`，与负 `R₃₀`、负 `R̄₃₀` 分别为 `0.863` 和 `0.810`。作者使用 64 个 shared cases、每个 300 候选及三 seed；这些相关描述的是固定候选池，不证明 regret 是闭环成败的充分统计量，也不保证 adaptive CEM。（Table III 与诊断设置，第 6–7 页）

### Franka：有提升，但部署仍依赖人工 subgoals

机器人结果用 frozen V-JEPA 2 ViT-G encoder、匹配 DROID post-training、同一 planner/deployment stack；没有 lab-specific images 或 demonstrations 用于 adaptation。论文所谓 zero-shot 不等于完全没有 post-training：模型在 filtered DROID 数据上 post-train。两模型使用相同、**人工指定的** grasp/move/place 图像 subgoals；执行 82 次/模型，安全停止计失败，按两个模型分别以 non-randomized blocks 评估。（Figure 5，第 7 页；Table IV，第 8 页）

| Protocol / metric | V-JEPA 2-AC | AD-WM |
|---|---:|---:|
| Basic pick-and-place success | 19/45 (42.2%) | 32/45 (71.1%) |
| Complex object success | 2/10 | 5/10 |
| Abnormal motion | 6/10 | 2/10 |
| Requested target moved | 14/27 | 21/27 |
| Target lift-and-place completed | 9/27 | 17/27 |

这是单一 robot setup、camera 与 backbone 的 transfer 证据；复杂物体和目标跟随样本量较小，且 manual subgoals 与分块非随机顺序限制了对完全自主、广泛部署的外推。

## 局限与阅读时要守住的边界

- **不等于更低预测误差。** Cube 中 AD-WM factual/local MSE 高于 LeWM；把“规划需要的差异”与一般预测精度分开讨论。
- **MI 是动机，不是联合训练保证。** Barber–Agakov 解释有固定表征与 normalization 条件；不能据此说训练严格最大化真实 conditional MI。
- **默认配置不是消融赢家。** Residual + MI 在 Table II-A 的 HS 高于 full AD-WM；MI weight sensitivity 还有更高的事后均值，主结果仍按预设默认权重报告。
- **静态 bank 不是 adaptive planner。** Elite regret 在本文诊断样本上与 HS 关联较强，但未证明能保证 CEM 迭代过程、动作执行序列或新任务成功。
- **没有加速结果。** 论文未报告 AD-WM 相对 LeWM 的 wall-clock latency、memory、throughput 或 energy advantage；训练辅助头不部署也不能推出整个 MPC 更快。
- **PushT 没有改善。** 匹配结果为 94%→92%；整体“四胜一负”不能转写成所有 benchmark 都提升。
- **统计与覆盖有限。** Cube 主要诊断限于该环境；Scene 的 gain 未在三 seed 配对检验中解决；robot 数据来自一个 site/camera/backbone。
- **机器人并非无人工层级的 long-horizon control。** 手工 subgoals、指定目标及固定执行 stages 仍在；两个 model blocks 非随机化。
- **跨方法表要看 inference interface。** Fast-LeWM、Sub-JEPA、INTACT 保留各自 inference；其表格结果不能独立归因于 predictor 或 AD-WM 的 loss。

## 与当前 FYP（LeWM + PushT）的关系

AD-WM 是直接相关的 LeWM/JEPA planning prior：它提醒我们，若 latent 一步变化很小，单看 prediction error 可能遗漏对候选动作排序有用的信息。但对本项目的关键限定很强：本文 **PushT 为 94%→92%**，没有展示 AD-WM 的 PushT 增益；也没有 latency 或 memory 加速数据。它需要改变 dynamics 的训练方案，不能直接作为现有 LeWM checkpoint 的免训练优化，也不能从丢弃辅助 heads 推出 planner 变快。

对 LeWM + PushT 的实际启发是一个证据顺序，而非结论：先问预测转移是否保留 action-conditioned 区别，再按候选排序、elite、first action / CEM trace 和最终 closed-loop 逐层读证据。本文的 fixed-bank `R_k` 适合作为诊断灵感之一，不可替代自适应 CEM 与任务成功评估；其 300/30/30、horizon 5 设置虽接近 LeWM 系统配置，也不自动让两项目结果可直接拼接。该论文更适合作为**动作可辨识性训练思路与诊断设计的对照**，而不是 PushT 提升或加速的已验证方案。（本文训练/规划设置：第 5 页；边界与结果：第 6–7 页）

## 阅读路线

### 20 分钟：抓主张和防止过度解读

1. **5 分钟：** 摘要与 Figure 1（第 1 页），用自己的话区分 factual prediction 和 counterfactual action selection。
2. **7 分钟：** 问题公式与 residual predictor（第 2–3 页），思考“复制当前 latent”为何可低误差却无法选动作。
3. **5 分钟：** 看 Figure 4（第 6 页）和 Table II 消融（第 5 页），特别记下 PushT 与默认配置。
4. **3 分钟：** 扫读 limitations（第 8 页），写下一条你认为最影响 FYP 外推的边界。

### 90 分钟：能够复述方法、核对主结果

1. 精读第 2–4 页：把数据转移监督、Inv、normalized recovery、测试时丢弃 heads 画成一张训练/部署流程图。
2. 对照第 5 页 Table I 和第 5–6 页 Table II：标出匹配对照、hard-start 定义、默认权重和 post-hoc sweep。
3. 精读第 6 页 Figure 4：逐环境写下 matched LeWM 与 AD-WM 的方向，不把 PushT 的方向藏进总数。
4. 看第 6–7 页 Table III：分别解释 MSE、CAD、`R₃₀`、`R̄₃₀` 的对象；判断 fixed bank 与 adaptive CEM 的差别。
5. 读第 7–8 页机器人 protocol 和 conclusion：列出 zero-shot、人工 subgoals、trial order、样本量各自限定了什么。
6. 回答 Reading Questions 1–12 中任选 6 题，并用论文页码支持答案；不要在 Meeting Card 中照抄摘要。

### 180 分钟：为组会准备一张可讨论的 prior-art card

1. 先完成 90 分钟路线，再回到第 3–4 页，检查每个 loss 的梯度路径，尤其是 detach target、预测端点与 batch normalization target。
2. 重建 Table II 的最小组件矩阵：absolute/residual × Inv/MI，并区分预设默认值和 sweep 后的高点。
3. 把 Table III 画成三层证据：factual prediction、fixed-bank candidate selection、closed-loop success；标明相关分析的样本单位与适用范围。
4. 用 Table I/图 4/机器人 Table IV 制作一张结果地图，标记环境、协议、checkpoint/seed 信息，以及不能横向比较的 native inference。
5. 对 FYP 写一个可证伪的问题：若用同一观察与候选动作，何种 action-sensitive 诊断会先于 planner/closed-loop 指标变化？哪些结果会否定这个解释？
6. 最后填写 Meeting Card，选择 3 个尚未解决的问题带去讨论；Reading Questions 留给自己逐篇阅读后作答。

## Meeting Card（留白，组会前填写）

- **我对论文主张的复述：**
- **最强的一条证据及所在页/表：**
- **我认为最脆弱的一条推断及原因：**
- **与 LeWM + PushT 最直接的连接：**
- **我需要导师帮助判断的问题：**
- **若要验证一个机制，我会先观察什么：**
- **结果如何会让我放弃当前解释：**

## Reading Questions（保持未填写）

1. 什么条件下低 factual prediction error 会让不同动作的预测终点难以区分？
2. 对论文中的视觉目标成本，动作 embedding 和预测 latent 分别通过哪些路径影响 candidate cost？
3. residual parameterization 改变了函数学习偏置的哪一部分？它缺少哪种额外约束？
4. inverse head 的 target detach 与 predicted endpoint 保持梯度分别起什么作用？
5. `L_MI` 中 recovery likelihood 与 KL penalty 分别约束什么量？
6. Barber–Agakov lower bound 的解释依赖哪些固定量？联合训练改变这些量会带来什么推论边界？
7. `R_k` 与 `R̄_k` 评价的是同一种 elite 属性吗？它们各自可能遗漏什么？
8. 为什么全池 Spearman 排序好，仍可能选不到适合下一轮 CEM 的候选？
9. fixed-bank diagnostics 与 adaptive CEM 在候选集合、更新反馈和闭环状态上有哪些差别？
10. Cube P00 为何不能简单等同于 Original？P00–P04 的 HS 平均具体代表什么？
11. Table II-A 中 residual、Inv、MI 各自对应哪些对照差异？哪些效应仍混在一起？
12. 为何 full AD-WM 的默认 HS 低于 Residual + MI？这个结果对“所有组件都必要”的说法有什么影响？
13. 如何解释 `λ_MI=0.03` 的敏感性高点与预先指定 `0.01` 的主结果之间的关系？
14. 论文怎样从 Cube 的 fixed-bank metrics 推导它们与 closed-loop success 的关联？这些观测能支持多强的结论？
15. 四个模拟环境成功率提高，为什么不足以声称 AD-WM 在所有 benchmark 上都提升？
16. PushT 的 94%→92% 可能受哪些协议、波动或机制因素影响？还需要什么数据才能区分解释？
17. 机器人结果中的 “zero-shot” 指什么适配边界？post-training、人工 subgoals 和 non-randomized blocks 又如何限定它？
18. 若把本文诊断迁移到 LeWM + PushT，怎样在不混淆 predictor、planner 和闭环证据的前提下安排评估顺序？

## Citation

```bibtex
@misc{qiu2026adwm,
  title         = {AD-WM: Action-Discriminative World Models for Counterfactual Model Predictive Control},
  author        = {Qiu, Jiabin and Chen, Zixuan and Cao, Hongye and Shi, Jieqi and Huo, Jing and Gao, Yang},
  year          = {2026},
  eprint        = {2609.30264},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  version       = {v1},
  url           = {https://arxiv.org/abs/2609.30264v1}
}
```
