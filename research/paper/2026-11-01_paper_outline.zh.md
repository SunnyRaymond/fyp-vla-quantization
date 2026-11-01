# 论文大纲（CVPR 2027 投稿稿，2026-11-01 第二版）

依据：[论文写作指南笔记](../writing-guide/论文写作指南笔记.md) 的结构要求 + 已定的论文主线（C1 → C2 → C3）+ 截至 2026-11-01 的实验结果。

**第二版（按你对摘要的意见改）**：叙事不再限定 "imagine-then-execute"，问题统一为"现有 VQ 保不住预测 future 的保真度"；加入"端侧显存 → 极低比特 → VQ"的论证；② 写成"按各模块对 future 保真度的影响分配比特"的一般形式；③ 定位为初始化，理由是原生支持混合位宽；结果先报跨 WAM 的评测范围，以 AgileX 真机收尾。改动覆盖题目、摘要、引言、相关工作、预备知识、§4、§5、§6、结论和各附录。

**怎么读这份大纲**
- 每个段落按四项写：**作用**（这一段在论证里干什么）→ **主题句草稿**（英文，可直接改成正文）→ **要点与证据** → **数字状态**。
- 状态标记：✅ 已有结果（注明来源作业）；⏳ 待跑 / 待补；❓ 需要你拍板。
- 英文句子只是草稿，用来固定每段的论点，不是定稿措辞。
- 文末有四张汇总：claim–证据对照、图表清单与版面预算、待跑实验清单、需要你决定的事项。

---

## 题目与命名

方法名沿用 **RAVQ（Reader-Aware Vector Quantization）**。"Reader" 泛指 WAM 里"从预测的 future 到动作"的那段计算，任何 WAM 都有，正是 C1 的核心对象。

题目候选（按推荐顺序；第二版按"保住 future"的叙事改写，不再绑定"先想象后执行"或"比特分配"）：
1. **Don't Lose the Future: Reader-Aware Vector Quantization for World Action Models**
2. **Future-Faithful Ultra-Low-Bit Quantization of World Action Models**
3. **Quantizing World Action Models Without Losing the Future They Predict**

第 1 个直接点出问题（量化把 future 弄丢了）和方法名，最短。第 2 个更像技术论文的标题，同时点出"极低比特"这个设定。第 3 个最直白，但偏长。

---

## 摘要（Abstract）（第二版，按 11-01 意见改）

**整段草稿**（数字待填处用 XX）：

> World action models (WAMs) jointly predict how a scene will evolve and how to act, conditioning their actions on the predicted future. Deploying them on a robot's onboard computer is memory-bound (the 6B-parameter Fast-WAM needs 12 GB for its BF16 weights alone), which calls for ultra-low-bit weight quantization, a regime where scalar quantizers collapse and vector quantization (VQ) is the method of choice. However, existing VQ methods, developed for language models, fail to preserve the fidelity of the predicted future: they reconstruct each layer or the final output, and nothing in their objectives deliberately protects the future on which the action depends. We propose RAVQ, a reader-aware vector quantization framework that makes future-latent fidelity an explicit quantization target. RAVQ (i) measures future-latent error in the metric through which the policy reads the future, (ii) distributes a fixed bit budget according to each module's measured impact on future fidelity, (iii) initializes the codes with trellis-coded rounding, which natively supports the resulting mixed bit widths, and (iv) calibrates the shared decode tables toward action and future-latent fidelity at no extra storage. We evaluate RAVQ across XX WAMs, both in simulation and in real-world deployment. On 840 paired LIBERO-Plus episodes, 2-bit RAVQ raises Fast-WAM's success rate from 73.3% (QTIP, the strongest 2-bit baseline) to XX%, within XX points of BF16 (77.6%), [while XX on WAM-2]. On an AgileX PiPER arm, 2-bit RAVQ succeeds in XX/XX real-world trials across XX tasks, compared with XX/XX for QTIP and XX/XX for BF16, with about 1.5 GB of weights.

| 句 | 功能 | 草稿要点 | 状态 |
|---|---|---|---|
| S1 | 背景：WAM 是什么 | 联合预测未来与动作，动作以预测的 future 为条件。**不限定**先想象后执行，序贯式（Fast-WAM）和联合去噪式（Cosmos-Policy）都被覆盖 | — |
| S2（前半） | 背景：端侧部署 → 为什么必须极低比特 | 机载算力受显存限制；6B 的 Fast-WAM 仅 BF16 权重就要 12 GB | ✅ 6B 出自 Fast-WAM 论文（1B 动作专家，总 6B）；12 GB = 6B × 2 字节 |
| S2（后半） | 背景：为什么是 VQ | 极低比特下标量量化崩溃，VQ 成为首选 | ✅ Fast-WAM 上 GPTQ W2（不分组）15/280，GPTVQ / QuIP# / QTIP W2 为 198–204/280（26042316 等） |
| S3 | 问题 | 现有 VQ 为语言模型设计，**fail to preserve the fidelity of the predicted future**：它们重建每一层或最终输出，目标里没有任何一项有意识地保护动作所依赖的 future | 按意见只说"没有有意识保护"，不讲比特分布 |
| S4 | 方法 | RAVQ 让 future-latent 保真度成为显式的量化目标，四个部件见下 | — |
| S5 | 评测范围 | 先说覆盖范围：XX 个 WAM，仿真 + 真机 | ⏳ WAM 个数 |
| S6 | 仿真结果 | LIBERO-Plus 840 配对：RAVQ vs QTIP 73.3% vs BF16 77.6%；第二个 WAM 一句 | ⏳ RAVQ 840；第二个 WAM |
| S7 | 真机结果（收尾） | AgileX PiPER：成功次数 / 试次，对比 QTIP 与 BF16；附权重显存 | ⏳ 全部待跑；1.5 GB = 6B × 1.993 bit ÷ 8，是按 bpw 推算的权重大小，实测显存另测 |

**四个部件的写法（S4 内）**
- **(i)** 用 policy 读 future 的度量衡量 future 误差。"reader" 泛指"从预测 future 到动作"的那段计算，任何 WAM 都有，不依赖具体结构。
- **(ii)** 按意见写虚：*distributes a fixed bit budget according to each module's measured impact on future fidelity*。它不提"动作专家 → 视频专家"，所以对共享骨干的 WAM 同样成立。在 Fast-WAM 上，这条规则的结果恰好是把动作专家的比特挪给视频专家，这个结论放到 §5.2 再讲。如果还想更虚，可以写成 *allocates bits where they best preserve the predicted future*。
- **(iii)** 写成"初始化"，并给理由：*initializes the codes with trellis-coded rounding, which natively supports the resulting mixed bit widths*。"resulting" 指 (ii) 产生的混合位宽，两句因此自然衔接。
- **(iv)** 校准共享解码表，不增加存储。

**和第一版相比改了什么**
- 去掉 "imagine-then-execute" 的限定，问题改为 "fail to preserve future fidelity"。
- 新增端侧显存 → 极低比特 → VQ 的论证（S2），用 GPTQ W2 崩溃的实测数作支撑。正文 ¶1–¶2 有对应展开。
- S3（问题句）不再提"比特没给想象网络"，只说现有方法没有有意识保护 future。
- 删掉了 73% 这个数。它是 Fast-WAM 上的具体测量，放进摘要等于把故事绑在这一个模型的结构上；它留在引言 ¶3 和 §4。如果你想在摘要里保留一个定量的问题证据，可以在 S3 末尾加 *(at 2 bits, the degraded future accounts for 73% of Fast-WAM's action error)*。
- 结果部分先说评测范围（XX 个 WAM，仿真 + 真机），再给仿真数字，最后以 AgileX 真机收尾。
- 第一版里"可选第 6 句 takeaway"删掉，因为现在以真机结果收尾。

**写作要点**
- 不写 "significant improvement" 这类空话，写清楚规模、对比对象、指标和幅度。
- 结果句给具体数字（借鉴 Q-WAM、QVGen），便于和主表对照。
- 词数：现在约 250 词（含占位符），CVPR 摘要通常 150–250 词，已到上限附近。数字填进去后如果超了，先删 S6 方括号里的第二个 WAM 那半句（正文会讲），再考虑压 S4。

---

## 1. Introduction（约 1.25 页，含 Fig. 1）

按指南的四步：领域重要 → 已有工作 → gap → 本文。共 5 段，加一个贡献列表。这与 paper 库里 10 篇论文的引言（4–6 段，背景 → 痛点 → 已有方法局限 → 本文思路 → 结果概述）一致。

**第二版叙事主线（与摘要一致）**：WAM 的动作依赖预测的 future → 端侧显存逼出极低比特 → 极低比特只有 VQ 可用 → 现有 VQ 没有有意识地保护 future，future 保真度因此丢失 → RAVQ 让 future 保真度成为显式量化目标 → 跨多个 WAM、仿真加真机验证。
全文统一用 *predicted future*（中文"预测的 future"），不用 *imagine-then-execute* 作限定。Fast-WAM 的"先去噪 future 再去噪动作"只作为具体实例出现在 §3.1。

### ¶1 大背景：WAM 依赖预测的 future，但放不进机器人

- **作用**：交代研究对象，说明为什么必须压到极低比特。
- **主题句草稿**：*World action models (WAMs) jointly predict how a scene will evolve and how to act, and their actions are conditioned on the predicted future.*
- **要点**：
  - 代表工作：Fast-WAM、Cosmos-Policy、LingBot-VA、DreamZero 等。future 与动作的耦合方式不同：有的先生成 future 再解码动作（Fast-WAM 的 IDM 模式），有的在同一个去噪过程里联合生成（Cosmos-Policy）。但动作都以预测的 future 为条件。
  - 部署约束：机器人机载算力受显存限制。6B 参数的 Fast-WAM（其中动作专家 1B）仅 BF16 权重就要 12 GB。✅ 6B 出自 Fast-WAM 论文
  - 引出设定：**weight-only、2–3 bpw**，目的是把权重压进机载显存，激活保持 BF16。
- **数字**：⏳ 选一个具体的机载设备及其显存（例如 Jetson 系列），用来对比 12 GB → 约 1.5 GB。

### ¶2 为什么是 VQ，以及已有工作做到了哪里

- **作用**：论证"极低比特 → VQ"，再铺垫已有方法，为 gap 做准备。不是综述。
- **主题句草稿**：*At two to three bits per weight, vector quantization is the only family that keeps a WAM usable.*
- **要点**：
  - **VQ 的必要性（实测）**：在 Fast-WAM 上，标量 GPTQ 在 2-bit（不分组）只剩 15/280，3-bit 为 195/280；同比特的 VQ / 格点 / 格栅码（GPTVQ、QuIP#、QTIP）在 2-bit 仍有 198–204/280，3-bit 有 202–205/280。BF16 为 216/280。✅（A16 基线，26042316 等）
  - **LLM 侧的 VQ**：AQLM、GPTVQ、QuIP#、QTIP 把语言模型推到 2 bpw，它们的目标是逐层重建误差（按输入统计加权）或最终输出。
  - **WAM 侧**：QuantWAMs（选校准粒度）、Q-WAM（保护动作子空间）面向 W4A4，目标是最终动作保真。
  - **收束句**：*None of these methods deliberately protects the predicted future that a WAM's action depends on.*

### ¶3 问题：现有 VQ 保不住 future 的保真度（C1 的直觉版）

- **作用**：Introduction 的关键段（gap）。把"现有方法不够"讲得具体：丢的是什么、为什么会丢。
- **主题句草稿**：*Applied to WAMs, existing vector quantizers fail to preserve the fidelity of the predicted future, and this loss propagates to the action.*
- **另一种开头（借鉴 Q-WAM 的设问式）**：*A natural question is what a 2-bit WAM actually loses.* 先提问，再用三条观察回答。
- **要点（三条观察，每条一句，详细放 §4；数字来自 Fast-WAM）**：
  1. **future 失真是动作误差的主要来源**：把量化模型预测的 future 换回 BF16 版本，2-bit 下动作误差去掉 73%，3-bit 下去掉 85%。✅（QTIP_W2I / QTIP_W3I 的 "future 占比"，26072226）
  2. **future 的失真来自生成它的模块**：只升级生成 future 的视频专家，就能拿到升一位全部收益的 71%；只读 future 的动作专家对 future 保真度的影响恰好为 0。✅（FLAQ 模块 1，26069856 + 26070509）
  3. **动作只读 future 很窄的一部分**：reader 的敏感度集中在 37,632 维 future latent 中约 42 个有效方向上，98% 在第一个预测帧。✅（① 的结果，26073871）
- **收束句（原则）**：现有 VQ 的目标里没有任何一项告诉它哪些模块决定 future、动作读的是 future 的哪一部分，所以 future 的保真度是在无意中丢掉的。原则句草稿：*Preserving a WAM under ultra-low-bit quantization therefore requires treating the predicted future, as the policy reads it, as a quantization target in its own right.*
- **闭环后果**：在 840 个 LIBERO-Plus episode 上，三个 2-bit VQ 方法比 BF16 少 36–50 次成功（4.3–6.0 个百分点，p ≤ 3e-4），3-bit 少 21–26 次（p ≤ 0.017）。✅（面板 A+B+C，26073449）
- **避坑**：不写 "remains challenging"。三条观察就是"难在哪里"的具体答案。观察 2 讲的是"误差从哪里来"，不写成"比特该怎么分"，比特分配留给方法。

### ¶4 本文方法：RAVQ

- **作用**：直接说本文做了什么，每个部件对应 ¶3 的哪条观察。
- **主题句草稿**：*Building on this diagnosis, we propose RAVQ, a reader-aware vector quantization framework that makes future-latent fidelity an explicit quantization target.*
- **衔接写法（借鉴 QuantWAMs 和 DSAQuant 的"观察 → 原则 → 方法"）**：¶3 末尾的原则句之后，¶4 用 *Building on this diagnosis* 开头。每个部件明确对应一条观察，让读者看到方法是从诊断推出来的，不是一组技巧的拼盘。
- **要点（一句一个部件，顺序与 §5 一致，措辞与摘要一致）**：
  - (i) 对应观察 3：用 policy 读 future 的度量（reader metric）衡量 future 误差，替代均匀的 NMSE。
  - (ii) 对应观察 2：在固定的比特预算下，按每个模块对 future 保真度的实测影响分配比特。
  - (iii) 用格栅码舍入初始化各层的码，选它是因为它原生支持 (ii) 产生的混合位宽。
  - (iv) 对应观察 1：以动作和 future 保真度（A + λZ_M + βK）为目标，校准各组共享的解码表，不增加存储。

### ¶5 结果预告

- **作用**：先说评测范围，再给最重要的数字，和 ¶3 的问题对应；顺序与摘要一致。
- **主题句草稿**：*We evaluate RAVQ across XX WAMs, both in simulation and in real-world deployment.*
- **要点**：
  - 仿真：LIBERO-Plus 840 配对，2-bit RAVQ 把 Fast-WAM 的成功率从 73.3%（QTIP，最强的 2-bit 基线）提高到 XX%，补回 BF16（77.6%）差距的 XX%。W2 是主说法；W3 作补充，因为 W3 基线到 BF16 只差 21–26 次，统计功效不够（见 §6.2）。
  - 第二个 WAM 一句。
  - 真机：AgileX PiPER 机械臂，XX 个任务，2-bit RAVQ 成功 XX/XX，QTIP XX/XX，BF16 XX/XX；权重约 1.5 GB。
- **数字**：⏳ RAVQ 840；⏳ 第二个 WAM；⏳ 真机。

### 贡献列表（3 条 bullet，动词开头，具体可验证）

- **C1** *We identify future-latent fidelity as a key quantization target for WAMs.* Counterfactual swaps and module-level upgrades show that most of the action error under ultra-low-bit quantization comes from a degraded predicted future, that this degradation originates in the modules that generate the future, and that the policy reads only a narrow part of it.
- **C2** *We propose RAVQ*, a reader-aware vector quantization framework whose error metric, bit budget and calibration objective all follow from C1, at the same bits per weight as the baselines and with no extra storage.
- **C3** *We evaluate RAVQ across XX WAMs in simulation and on a real AgileX PiPER arm*: [at 2 bits per weight it is the only method that XX]。

（C3 的措辞等闭环数字出来再定。C1 的措辞按已定的 "Identify future-latent fidelity as a key quantization target"，只把限定从 "imagine-then-execute WAMs" 改成 "WAMs"。）

写法参考：
- 第一条写"发现"本身，与 DSAQuant 的第一条（"We identify and empirically localize …"）同类，诊断本身就是一项贡献。
- 第三条以结果收尾，与 Q-WAM 的第三条（"Across …, Q-WAM preserves …"）同类。
- Q-WAM、QuantWAMs、Fast-WAM、DSAQuant 都用 3 条 bullet，审稿人对这种形式最熟悉。

### Figure 1（Teaser）

指南要求：只回答两件事（问题/观察 + 本文优势），并且 self-contained。先定一句话："读者看完应该记住：极低比特量化让 WAM 丢掉了 future 的保真度，动作误差主要由此而来；把 future 当作量化目标保护起来，成功率就回来了。"

两个方案：
- **方案 A（推荐）**：左右两栏。
  - 左：通用的 WAM 示意（观测 → 预测的 future → 动作），两条误差路径用两种颜色的箭头标出：直接作用于动作的路径，以及先破坏 future、再传到动作的路径。箭头上标 "27%" / "73%"，并注明是 Fast-WAM 2-bit 的测量。
  - 右：成功率–bpw 散点图。BF16 一条水平虚线；GPTVQ / QuIP# / QTIP 在 2、3 bpw 的点；RAVQ 在 2、3 bpw 的点，用颜色突出。
- **方案 B**：具体样本驱动。同一个 LIBERO-Plus 场景下，BF16、QTIP-W2、RAVQ-W2 三行，每行是解码出的预测帧 + 实际执行结果（成功/失败）。最直接地展示"future 丢了"。直观，但需要 VAE 解码出的帧足够清楚，⏳ 待看效果。
- **方案 C（GuidedQuant 式 teaser 表）**：第 1 页放一张小表，每个宿主一行，成对列出"宿主 → RAVQ"的成功率。首页就能看到增益。缺点是表格比图更难体现 C1。适合 RAVQ 的增益在多个比特档上都整齐的情况。
- Paper 库里 QuIP#、AQLM、GuidedQuant、QuantWAMs 都把 teaser 放在第 1 页。CVPR 的双栏版面里，Fig. 1 通常放在第 1 页右栏顶部或横跨两栏。
- Caption 草稿（借鉴 QuIP#：caption 两句话，直接写结论）：*Figure 1. Under 2-bit weight quantization, a WAM loses the fidelity of its predicted future, and most of its action error follows from it (left). RAVQ makes the predicted future an explicit quantization target and recovers closed-loop success at the same bits per weight (right).*

---

## 2. Related Work（约 0.6 页，按研究方向分 3–4 个小节）

指南要求：分类 + 概括 + 代表工作 + 局限 + 本文定位；每段四步写；不要只写相似，要写差异。

### 2.1 World action models
- **概括**：视频生成模型与动作模型联合训练，联合预测场景如何演化以及如何行动，动作以预测的 future 为条件。按 future 与动作的耦合方式分两类：
  - 分专家（mixture-of-transformers，视频专家与动作专家共享注意力）：Fast-WAM、LingBot-VA；
  - 共享骨干（同一 DiT 同时去噪视频与动作 token）：Cosmos-Policy、DreamZero 等。
- **与本文关系**：本文不改训练，只研究部署时的权重压缩。C1 说明预测的 future 不仅在训练时有用，部署时它的保真度也直接决定动作。（Fast-WAM 自己的问题就是"测试时还需不需要预测 future"，这里可以呼应一句。）

### 2.2 Post-training weight quantization
- **概括**：
  - 二阶舍入：OBQ / GPTQ；
  - 激活感知缩放：AWQ；
  - 旋转去 outlier：QuaRot、SpinQuant；
  - 向量 / 格点 / 格栅码：AQLM、GPTVQ、QuIP#、QTIP，把 LLM 推到 2 bpw；
  - 混合精度分配：HAQ、HIGGS；
  - 端损失引导的层目标：GuidedQuant（用最终损失梯度给层目标加权）。
- **局限**：层目标用输入二阶矩（或端损失的 Fisher 近似）衡量误差，把模型当作一般的"输入 → 输出"网络。目标里没有"预测的 future"这个对象，也就不会有意识地保护它。
- **本文定位（Extension + Contrast）**：③ 直接沿用 QTIP 的舍入，②、④ 沿用混合精度和共享参数校准的成熟做法；新的是**信号**：哪里该给比特、校准时盯什么，都由 C1 决定。GuidedQuant 只在这里出现，作为"端损失引导"这条线的代表，不作为对立面。

### 2.3 Quantization of generative and embodied models
- **概括**：
  - 扩散 / 视频生成：PTQD（去偏去噪循环里的系统误差）、ViDiT-Q、SVDQuant、QVGen；
  - VLA：QuantVLA、BitVLA 等；
  - WAM：QuantWAMs（按合适粒度做校准与保护）、Q-WAM（保护最终动作敏感的激活子空间，W4A4）。
- **差异**：
  - 生成模型量化的目标是生成质量本身；
  - VLA / WAM 量化的目标是最终动作；
  - 本文的目标是**被动作读取的预测 future**，并且在 weight-only 2–3 bpw 的极低比特区间（与它们的 W4A4 是不同的设定）。
- 可选一张对比表（指南建议：维度必须领域公认且与核心问题相关）：维度 = 目标（生成质量 / 最终动作 / 预测的 future）、比特设定、是否 weight-only、是否混合位宽、闭环评测规模。❓ 版面紧张时删。

### 2.4（可选，一段）Task-aware coding
- **概括**：coding for machines、语义通信、不等差错保护（UEP）。共同思想是：编码误差的代价由接收端怎么用这些数据决定。
- **定位**：RAVQ 的 ① 是这一思想在 WAM 内部的对应物，"接收端"就是从预测的 future 计算动作的那部分网络。只写一段，作为方法思想来源。

---

## 3. Preliminaries（约 0.5 页）

### 3.1 World action models
- **主题句草稿**：*A WAM maps an observation o and an instruction ℓ to a predicted future latent z and an action chunk a, where the action is conditioned on z.*
- **通用形式**：z 与 a 由同一组权重 W 生成，a 对 z 有依赖。把"给定 z 计算 a"的那部分计算记为 reader R(o, ℓ, z)。不同 WAM 的区别只在于 z 和 a 是先后生成还是在同一个去噪过程里联合生成，reader 的定义对两者都成立。
- **定义 reader**：R(o, ℓ, ·) 是"从预测的 future 到动作"的函数。它是 §5 中 ① 和 K 的基础，也是方法名里 "reader-aware" 的来源。
- **实例：Fast-WAM（IDM 模式）**：
  - 视频专家 G 用 T_v = 10 步去噪得到 future latent z = G(o, ℓ, ε_z)。z 的形状为 [48, 3, 14, 28]：3 个 latent 时间位置，第 0 帧是观测（固定为输入），第 1、2 帧是预测；14×28 的网格里包含两路相机。
  - 动作专家通过共享注意力读 z 的 K/V，用 T_a = 10 步去噪输出 32×7 的动作块 a = R(o, ℓ, z, ε_a)。闭环中执行前 10 步后重新规划，第 7 维 gripper 单独后处理。
- **实例：联合去噪的 WAM（如 Cosmos-Policy）**：future 与动作 token 在同一个 DiT 中联合去噪，reader 取"从某一去噪步的 future 状态出发，跑完剩余去噪步得到动作"的那段计算。我们已经在 Cosmos-Policy 上实现过这种 reader（joint-DiT suffix reader，cosmos-policy-vqwam-20261024）。⏳ 以最终选定的第二个 WAM 为准。
- **符号表**：o、ℓ、z、a、R、W_l（第 l 层权重）、ΔW_l、H_l；Fast-WAM 专用的 G、T_v、T_a。

### 3.2 Weight-only vector quantization
- **层目标**：min tr(ΔW H ΔWᵀ)，其中 H = E[xxᵀ] 是校准集上的输入二阶矩（下文记 H_U）。
- **QTIP**：权重先经随机 Hadamard 旋转（W = diag(SV) Hᵀ W̃ H diag(SU)），旋转域中每对权重由格栅码状态查 LUT 解码，每权重 K bit。
- **bpw 计数口径**：码 + 每层 scale + 表，接口层 BF16 也按 16 bit 计入总 bpw。这个口径对所有方法一致，在这里一次说清。

---

## 4. What Does an Ultra-Low-Bit WAM Lose?（C1，约 1.25 页）

这是 analysis 章节。结构参考"先观察、后方法"的写法：每个小节给一个观察和一张图/表，最后用一段 takeaways 把观察转成设计原则，§5 直接引用。

写法参考：
- DSAQuant 是"诊断先行"：引言里先把问题追到具体的设计原因，方法的每个组件对应一条诊断。本节的三条观察对应 ④、②、①，是同一结构。
- QVGen 每个设计选择都配一张证据图。本节每个小节标题直接写成观察的结论句（下面的英文小标题已经是这种写法），对应的图就是证据。
- 小节标题用结论句，不用"Analysis of X"这类名词短语。

**节标题备选**：*What does an ultra-low-bit WAM lose?*（与引言 ¶3 的设问呼应）或 *Ultra-low-bit quantization loses the predicted future*（结论句）。

**统一设置（开头一段）**：Fast-WAM，C80 校准集以外的 40 个场景 × 2 个 seed；三个 2-bit 宿主（GPTVQ / QuIP# / QTIP）以及公平口径的 QTIP（接口 BF16 + 时间表）。指标：
- motor：前 10 步动作的 RMSE；
- future NMSE：‖z_q − z_T‖² / ‖z_T‖²，只算第 1、2 帧；
- K：reader KL，16 个 soft bin，σ = 0.375。
- **跨模型**：本节的测量在 Fast-WAM 上完成。为了让 C1 不绑定在一个结构上，第二个 WAM 上至少复现观察 1（future 占比），可放在 §6.3 或附录。⏳

### 4.1 Two paths from a weight error to the action
- **作用**：定义要测的量。
- **主题句草稿**：*A weight error can change the action either directly, or by degrading the predicted future on which the action is conditioned.*
- **定义**：future 占比 = 1 − MSE(R_q(z_T)) / MSE(R_q(z_q))。也就是把量化模型自己预测的 z_q 换成 BF16 的 z_T 后，动作误差还剩多少。分母是完整量化模型的动作误差，分子是 future 被修好之后、只剩直接路径时的误差。
- **一句白话解释**：这个比例就是"动作误差里由 future 失真造成的那一部分"。定义只用到 reader，对任何 WAM 都能算。

### 4.2 Most of the action error comes from a degraded future
- **主题句草稿**：*Restoring the predicted future alone removes most of the action error.*
- **证据（Table 2 或 Fig. 2 左）**：
  - 公平口径 QTIP：W2 73%，W3 85%。✅ 26072226
  - 三个宿主（A16 原口径）W2：换回 BF16 future 后 motor **RMSE** 分别降 24%（QTIP）、36%（GPTVQ）、36%（QuIP#）。✅ 26050310
    - ⚠ 口径要统一：上面的 73% / 85% 是 **MSE** 占比，这里是 RMSE 降幅。QTIP A16 按 MSE 算的占比是 43.7%（26072226 表中有）。用 1 −（换后 RMSE / 原 RMSE）² 推算：QTIP 42.4%，GPTVQ 58.7%（0.0259 / 0.0403），QuIP# 58.6%（0.0200 / 0.0311）。QTIP 的推算值与按样本算的 43.7% 只差 1.3 个点，所以三个宿主在原口径下的 MSE 占比约为 44% / 59% / 59%。正式数仍要按样本 MSE 重算一遍。论文里统一用 MSE 占比。
  - 两组口径的差别要解释清楚：原口径下接口层也被量化，动作专家自身误差大，future 占比被稀释；接口层存 BF16 后，future 成为绝对主导。这也是 §6 主表要给所有方法同一口径的原因。
- **❓**：这一节全部是开环测量（held-out 场景上的 motor / future 误差）。我的理解是：你说"held-out MSE 诊断不进论文"，指的是用来调 ④ 的那些迁移类诊断；C1 的现象测量属于 analysis，可以进论文。请确认（文末 ❓1）。

### 4.3 Future error originates in the modules that generate the future
- **主题句草稿**：*Upgrading the modules that generate the future recovers most of what an extra bit can buy, while upgrading the modules that only read the future leaves it untouched.*
- **叙事要点**：这一节讲"future 的误差从哪里来"，不讲"比特该怎么分"。比特分配是方法（§5.2）从这条观察推出来的结论。
- **证据（Fig. 3）**：
  - 逐组把 W2 换成 W3（64 组：每个专家 × 块 + 条件 / 头 / 接口），测 A、λZ、βK 三项的变化。
  - 视频专家合计拿到全部升位收益的 71%，主要经 Z 和 K。
  - 动作专家 30 个块全部升位，收益约为 0，对 Z、K 恰好为 0（视频 token 不读动作专家，这是结构上的必然）。✅ 26069856 + 26070509
  - 反向验证：动作主体降一档（2→1 bit），future 完全不变，动作专家自身误差 +24%，总 motor 只 +8%。✅ 26072226
- **图的形式**：横轴为组（视频块 0–29 | 动作块 0–29 | 接口 / 条件），纵轴为升位收益，A / Z / K 三色堆叠。
- **写作注意**：块与块之间的收益差异在噪声范围内，所以图上强调**模块级**（生成 future 的模块 vs 只读 future 的模块）的对比，不对单个视频块的高低做解读。这也是 ② 只在测得出差异的模块粒度上按 future 影响分配、更细的粒度改用率失真贪心的理由。

### 4.4 The action reads a narrow part of the predicted future
- **主题句草稿**：*The action is sensitive to a small, stable part of the predicted future.*
- **证据**：
  - reader 度量 M 的有效秩 41.5（维度 37,632）；前 32 个方向占迹的 52%；第 1 个预测帧占 98%；前 10% 的对角元占 62%。
  - 稳定性：奇偶两半样本估计的对角相关 0.86。✅ 26073871
- **图（Fig. 3 右，"policy gaze"）**：把 M 的对角按 14×28 网格、两路相机画成热力图（第 1 帧），叠在对应的观测图像上，看动作在"看"哪里（预期在夹爪和目标物附近）。⏳ 需要从 M 导出画图；这张图很适合 CVPR 的读者。
- **结论句**：普通 future NMSE 对所有维度一视同仁，但动作只关心其中很小的一部分。所以衡量 future 误差应该用 reader 的度量。
- **写作注意**：只报告对角的稳定性。前 32 个方向在两半样本之间的重合度只有 0.19，这一点用来说明 ① 的 M 采用"低秩 + 对角 + ρI"的形式，并在消融里比较"只用对角"（§6.4）。不在正文展开为负面结论。

### 4.5 Takeaways（一段，3 条，带编号，§5 逐条引用）
- **T1（来自 4.2）**：future 保真度是极低比特 WAM 动作误差的主要来源，量化目标必须显式包含预测的 future。→ ④ 的目标包含 Z_M 和 K。
- **T2（来自 4.3）**：各模块对 future 保真度的影响差别很大，比特预算应该跟着这种影响走。→ ②。
- **T3（来自 4.4）**：衡量 future 误差要用 reader 的度量，而不是均匀的 NMSE。→ ①。

---

## 5. Method: RAVQ（约 1.5 页，含主图 Fig. 4 + Algorithm 1）

指南要求：先 Overview（配主图），再分模块；每个模块讲清解决什么问题、输入、处理、输出、为什么这样设计；公式前后都要解释；novelty 点到即止。

写法参考：
- Q-WAM、Fast-WAM 在方法节内用 run-in 小标题（*Overview.* / *Architecture.* / *Training objective.*）分层，读者可以快速跳读。本节每个小节内部建议也用 run-in：*What it fixes.* / *Definition.* / *Estimation.* / *Why this form.*，正好对应指南要求的"解决什么、怎么做、为什么"。
- AQLM 用动词短语作为同级小节标题。本节四个小节标题可以统一成"对象 + 动作"的结构（借鉴 QuantWAMs），方便和消融表对应，例如 *Measuring the future as the policy reads it* / *Allocating bits by future impact* / *Initializing with trellis codes* / *Calibrating shared decode tables*。

### 5.0 Overview
- **主题句草稿**：*RAVQ quantizes a pretrained WAM in four steps, each answering one of the takeaways in Sec. 4.*
- **流程（与主图从左到右一致）**：① 估计 reader 度量 M → ② 在总比特约束下，按各模块对 future 保真度的影响分配位宽 → ③ 用格栅码舍入初始化各层的码 → ④ 校准各组共享的解码表。
- **一句交代不变量**：推理时的计算图、kernel 和存储格式都与 QTIP 相同；①、④ 只在离线量化阶段使用 reader，部署时不需要 reader。
- **Fig. 4 主图设计**（按指南"方法主图"的五步）：
  - Caption 一句话：*RAVQ makes the predicted future an explicit quantization target: it measures, allocates, initializes and calibrates with one signal, how the action reads the predicted future.*
  - 布局：上方一行是通用的 WAM 结构（生成 future 的部分 → 预测的 z → reader → 动作；Fast-WAM 里分别对应视频专家和动作专家），下方一行是四个模块框，用虚线连回 WAM 中各自作用的部位：
    - ① 指向 z；
    - ② 指向各模块，用格子颜色深浅表示位宽；
    - ③ 指向每个 Linear；
    - ④ 指向解码表。
  - 视觉语言：实线 = 推理数据流；虚线 = 离线量化流程；红色 = 校准信号 A / Z_M / K。总共 3–4 种低饱和配色，在顶部加 legend（参考 WAPITI 的做法）。

### 5.1 Reader metric（①，对应 T3）
- **解决什么**：普通 NMSE 对 future 的每一维一视同仁。
- **定义**：M = E_c[ J_R(c)ᵀ diag(h_c / 2) J_R(c) ]，即 reader KL 在 teacher future 处的 Gauss–Newton 矩阵。
  - J_R：reader 输出对 future 的 Jacobian；
  - h_c：teacher 动作处 soft-bin KL 的曲率；
  - Z_M = δzᵀ M δz / ‖z_T‖²。
  - 公式后说明：M = I 时 Z_M 退回普通 NMSE。
- **估计**：
  - 每个校准场景用 4 个 Rademacher 探针 c = J_Rᵀ(ε ⊙ √(h/2))，只需要 reader 的反向传播，不需要显式构造 Jacobian；
  - 每个场景先归一到单位迹再平均，防止少数场景主导；
  - 只取第 1 帧以后（第 0 帧是观测，梯度恒为 0）。
- **形式**：M = 0.9 · (D / tr M_R) · M_R + 0.1 · I，其中 M_R = 前 32 个特征方向 + 剩余对角。
- **为什么这样设计**：低秩部分捕捉主要敏感方向；对角部分稳定（§4.4）；ρI 保证 future 的任何方向都不会完全不受约束。
- **成本**：只依赖 BF16 模型，估计一次，所有位宽共用。⏳ 补一个 GPU·h 数。

### 5.2 Future-aware bit allocation（②，对应 T2）
- **解决什么**：均匀位宽默认每个模块同样重要，但各模块对 future 保真度的影响差别很大（§4.3）。
- **约束**：Σ_l n_l K_l ≤ B · Σ_l n_l，其中 B 为目标 bpw，n_l 为层 l 的权重数。
- **一般规则（正文按这个写，与摘要的 (ii) 措辞一致）**：
  1. **模块级，按 future 影响**：对每个模块组测"升一位能让 future 保真度（Z、K）改善多少"，即 §4.3 的反事实升位实验。对 future 没有影响的模块降一档，省下的比特给影响最大的模块。
  2. **更细的粒度，按率失真**：在测不出可靠差异的粒度上（§4.3 里视频块之间的差异在噪声内），按"每多 1 bit 能降低多少按权重数加权的 H_U 误差"贪心选层，升到 K + 1，直到预算用完。
- **在 Fast-WAM 上的实例**：动作专家只读 future，对 future 保真度的影响恰好为 0，所以动作专家主体降一档（K − 1）；省下的比特全部给生成 future 的视频专家主体，再在视频专家内部按率失真选层。
- **在共享骨干的 WAM 上**：每层同时处理 future 和动作 token，规则不变，只是模块组改成块或层类型。⏳ 以第二个 WAM 的实际实现为准。
- **叙事要点**：正文先写一般规则，再写 Fast-WAM 上的结果。"把比特从动作专家挪到视频专家"是规则在这个模型上的**结果**，不是方法的定义。这样方法不绑定分专家结构。
- **公平口径（所有方法一致，正文必须明说）**：
  - 接口层（action_encoder、动作 head、proprio_encoder、视频 head）存 BF16，占权重 0.01%；
  - 时间条件层只看 timestep，存成按 timestep 的输出表，输出与 BF16 完全一致，比特比 W2 还少。
  - 这两项在 §6 的所有基线上同样开启。
- **结果描述（放这里或 §6.5）**：2-bit 预算下买到 48 个视频层（占权重 16.7%），集中在视频专家后半段（block 15–29）的 self_attn.o / v 和 ffn.2。总 bpw 1.9932，与公平口径的 QTIP 完全相等。✅ 26072226
- **为什么更细的粒度不按 future 影响分配**：视频块之间的收益差在噪声内（§4.3），按它分配等于按噪声分配，所以只在模块级用 future 影响，更细的粒度用率失真。这句话要写，审稿人会问。
- **方法节内嵌一张小表（借鉴 QTIP 在方法节里直接放验证表）**：按模块列出每类层的平均位宽，让读者一眼看到预算怎么挪。数据 ✅ 26072226：

  | | 视频主体 | 动作主体 | 文本条件 | 接口 | 时间表 | 总 bpw |
  |---|---:|---:|---:|---:|---:|---:|
  | QTIP（公平口径）2-bit | 2.01 | 2.03 | 2.02 | 16 | 0.07 | 1.993 |
  | RAVQ 2-bit | 2.22 | 1.03 | 2.02 | 16 | 0.07 | 1.993 |
  | QTIP（公平口径）3-bit | 3.01 | 3.03 | 3.02 | 16 | 0.07 | 2.981 |
  | RAVQ 3-bit | 3.22 | 2.03 | 3.02 | 16 | 0.07 | 2.981 |

  版面紧张时，这张表可以和 Fig. 5（位宽分配图）二选一。

### 5.3 Trellis-coded initialization（③）
- **定位**：③ 是 RAVQ 的初始化选择，给 ④ 提供起点。正文要写清楚为什么选它。
- **为什么选格栅码（按重要性排序）**：
  1. **原生支持 ② 给出的混合位宽**：格栅码的结构对任意每权重比特数 K 都成立：不同 K 只改变 trellis 每步移入的比特数，查表解码的结构（512×2 的 LUT）不变。所以按层混用 K = 1–4 时，编码和解码方式都是同一套。对照：格点码（如 QuIP# 的 E8P）的码本是按特定比特数设计的，换位宽要换码本或叠加残差级。（⚠ 这句对照写进正文前，再对一遍 QTIP / QuIP# 原文。）这是与 ② 直接衔接的理由，摘要 (iii) 写的就是这一条。
  2. **解码表全组共享**：同一（角色，位宽）组的所有层共用一张解码表，这正好是 ④ 要校准的对象，校准不增加存储。
  3. **极低比特下最强**：2-bit 下它是闭环最好的基线（616/840）。
- **做法**：QTIP 官方 kernel 只支持 K = 2–4，我们把实现放开到 K = 1–4（K = 2 时与原实现逐位一致），逐层用 H_U 舍入。
- 这一节写短，是标准部件。novelty 不在这里。

### 5.4 Shared-table calibration（④，对应 T1）
- **解决什么**：逐层舍入只看各层自己的重建误差，看不到预测的 future 和动作的端到端保真度。
- **参数化**：
  - W = diag(SV) Hᵀ (Wscale · LUT_g[code]) H diag(SU)；
  - 只学习每组 g（按（角色，位宽）分组）的解码表偏移 ΔLUT_g，组内所有层共用；
  - 码、scale、旋转全部冻结；参数为 0 时与未校准版逐位相同。
  - ⏳ 最终版是否还包含行 / 列 scale 和输出偏置，以最后上闭环的配置为准。当前 v3 只调共享解码表。
- **目标**：L = A + λ_M Z_M + β K。
  - A：动作 MSE；Z_M：①；K：reader soft-bin KL。
  - λ_M = λ · mean(Z) / mean(Z_M)，让 λ_M Z_M 的初始量级与原来的 λZ 一致；β 按初始质量的固定比例取一次，之后冻结。
  - 公式前说明为什么需要（T1），公式后逐项解释，再说明它让量化模型在"动作 + 被读取的 future"上贴近 BF16。
- **优化**：
  - 梯度经解码表反传，不需要稠密权重梯度；
  - 步长 δ = −η g / max|g|（最大的元素移动 η）；
  - 在 FP32 解码结果上折算回 BF16 权重；
  - C80 一遍，8 次更新，每次 10 个场景。
- **为什么共享**：
  - 解码表是全组共享的，校准不增加任何存储；
  - 参数量相对校准集很小；
  - 一张表的调整同时作用于组内所有层，对应的是系统性的误差，而不是某层的个别误差。
- **Algorithm 1**：四步伪代码，变量名与正文一致（M、K_l、B、LUT_g、L）。

### 5.5 Cost
- 存储：与同 bpw 的 QTIP 相同（解码表为每组 512×2，全模型 ⏳ 个数，可忽略）。
- 量化时间：① + ② + ③ + ④ 的 GPU·h，对比 QTIP / GPTVQ / QuIP# 的量化时间（A16 已有：QTIP W2 1.86，GPTVQ W2 3.84，QuIP# W2 1.28 GPU·h）。⏳ RAVQ 各阶段耗时。
- 推理：与 QTIP 完全相同的 kernel 与格式。⏳ 混合 K（含 K = 1）的 kernel 支持情况要核实，见 §6.6。

---

## 6. Experiments（约 2.5 页）

开头一段写清实验要回答的问题（参考指南 Experiments 节"实验服务于 claim"）：
- **Q1**：同 bpw 下，RAVQ 的闭环成功率是否超过现有最强的极低比特量化器？（§6.2）
- **Q2**：换一个 future 与动作耦合方式不同的 WAM，结论是否仍成立？（§6.3）
- **Q3**：每个部件的贡献是什么？future 保真项是否必要？（§6.4）
- **Q4**：比特花在哪里，动作在读哪里，收益出现在哪些扰动上？（§6.5）
- **Q5**：省了多少显存，速度如何？真机上是否成立？（§6.6、§6.7）

写法参考：
- QuIP# 的实验小节按"问题"组织而不是按"方法"组织，读者可以按自己关心的问题跳读。上面 Q1–Q5 与 §6.2–§6.7 一一对应，就是这个做法。
- AQLM 和 QuantWAMs 的小节和段落标题直接写结论（"Pareto optimality of AQLM."），标题即论断，正文给证据。§6.4、§6.5 的 run-in 标题建议写成结论句。示例（等数字出来再定）：
  - *Future fidelity is necessary.*（消融 B 组）
  - *Allocating by future impact matters, not mixed precision per se.*（消融 C 组）
  - *Bits go to the output projections of the late video blocks.*（Fig. 5）

### 6.1 Experimental setup
- **模型**：Fast-WAM Optional-IDM（LIBERO 2-cam 224 的公开 checkpoint，上游 revision 7faa711），BF16 为参照。⏳ 第二个 WAM。
- **基准**：LIBERO-Plus。4 个 suite × 7 个扰动维度。从每个 cell 的 50 个变体中抽 3 个互不重叠的面板（A、B、C），每个面板每 cell 10 个，共 840 个 episode。所有方法用同一批变体和 seed，因此可以逐 episode 配对。
- **指标**：成功率；与最强基线逐 episode 配对的胜 / 负次数和精确 McNemar 双侧 p 值。
  - 写一句为什么用 840：单个 280 面板之间的波动与方法差距同量级（例如 W2 宿主对 BF16 的净损失在三个面板上分别为 −12…−18、−8…−15、−16…−20）。
- **基线**：GPTQ、GPTVQ、QuIP#、QTIP，各 2 / 3 bpw。所有方法同一校准集（C80）、同一公平口径（接口 BF16 + 时间表）、同一运行时。
  - GPTQ 作为标量量化的代表留在主表里：摘要和引言说"极低比特下标量量化崩溃、VQ 是首选"，主表要能直接看到这一点（2-bit 15/280，3-bit 195/280）。⏳ GPTQ 目前只有面板 A 的 280，主表若统一用 840，GPTQ W3 要补跑 B、C；GPTQ W2 已经崩溃，280 足以说明问题，表注里写明。
- **校准**：C80 = 80 个 LIBERO 场景；λ、β 的取法；更新次数。
- **硬件**：4×A100-40GB；量化时间按 GPU·h 报告。
- 细节（面板抽样方式、seed 规则、T5 缓存、运行时一致性检查）全部放附录。
- **排版**：上面各项写成 run-in 段落（*Models.* / *Benchmark.* / *Metrics.* / *Baselines.* / *Calibration.* / *Implementation.*）。这是 QVGen、DSAQuant、Q-WAM 的统一做法，比 bullet 省版面。

### 6.2 Main results（Table 1，对应 Q1）
- **表结构**：
  - 行分 3 个 block：BF16；3 bpw block（GPTVQ / QuIP# / QTIP / **RAVQ**）；2 bpw block（同上）。
  - 列：bpw | 权重显存（GB）| libero_spatial | libero_object | libero_goal | libero_10 | 总成功率（840）| 对最强宿主的胜 / 负（p）。
  - 权重显存列直接并进主表：Q-WAM、QuantWAMs 都把 BPW 和 Mem 放在主表里，省掉单独的效率表，也让"压缩"和"精度"在同一处对照。延迟仍放 §6.6。
  - 每个 block 内最优加粗、次优下划线，RAVQ 行加浅色底。在 caption 里用一句话定义加粗和下划线的含义（GPTVQ、QVGen 的做法）。
  - Caption 直接写结论：*At 2 bpw, RAVQ is the only method within X points of BF16.*
- **已有数字**（✅ 26073449，原 A16 口径，840）：
  - BF16 652（77.6%）；
  - W3：QTIP 631、GPTVQ 630、QuIP# 626；
  - W2：QTIP 616、QuIP# 609、GPTVQ 602。
  - 各 suite 数字在面板 C 的 README 里。
- **⏳ 主表必须补的**：
  - 三个宿主在公平口径下（接口 BF16 + 时间表）的 840。我们的公平口径在开环上单独就有明显收益（W2 mixed 0.84×，gripper 0.39×），不给基线就不公平。
  - RAVQ W2 / W3 的 840。RAVQU（未校准）的 840 正在跑（fwr840u 26075585）。
- **正文段落写法**（指南：告诉读者从表里看什么）：
  1. 先说 W2：RAVQ 与最强宿主的差、配对 p 值、补回了 BF16 差距的百分之几；
  2. 再说 W3：宿主到 BF16 只差 21–26 次，RAVQ 的空间有限，如实描述；
  3. 最后一句回到 C1：收益是否集中在 future 主导的 suite / 扰动维度。
- **结果有不显著的地方**：按指南"承认 + 弱化 + 转回正面"写（例如 W3）。

### 6.3 Results across WAMs（Table 2，对应 Q2）
- **作用**：摘要和 C3 说"across XX WAMs"，这一节兑现它。选一个 future 与动作耦合方式不同的 WAM，才能说明 C1 和 RAVQ 不依赖 Fast-WAM 的分专家结构。
- **候选**：Cosmos-Policy（共享骨干，future 与动作联合去噪）或 LingBot-VA。
- **② 怎么用**：按 §5.2 的一般规则（按各模块对 future 保真度的实测影响分配），在共享骨干上模块组改成块或层类型。叙事上已经不再依赖"动作专家 → 视频专家"。⏳ 实际是否测得出模块间的差异，要在第二个模型上先跑 §4.3 的反事实升位实验。
- **C1 的复现**：同时报告第二个模型上的 future 占比（§4.1 的定义），说明"动作误差主要来自 future 失真"不是 Fast-WAM 特有的。
- 已有资产：Cosmos 的 joint-DiT suffix reader 已经实现过（cosmos-policy-vqwam-20261024）。
- 表结构同 Table 1，规模可以小一些（280 或 560）。

### 6.4 Ablation study（Table 3，对应 Q3，你来跑）
建议的行（W2，闭环为主，配对对象为同口径的 QTIP；规模至少 560，最好 840）：

| 组 | 行 | 要证明什么 |
|---|---|---|
| A 逐步搭建 | QTIP（公平口径）→ +② → +②+① +④（完整 RAVQ） | 每一步的增量 |
| B 校准目标（核心） | ④ 用 A only / A + λZ（M = I）/ A + λZ_M / A + λZ_M + βK | **future 保真项是否必要**：这一组是 C1 在方法层面的直接证据 |
| C 比特分配 | 均匀 / 反向（把比特给对 future 影响最小的模块）/ RAVQ（按 future 影响）；更细粒度上随机选层 vs 率失真贪心 | 收益来自"按 future 影响分配"，而不是"任何混合精度都行" |
| D 度量形式 | M = I / 只用对角 / 低秩 + 对角 + ρI | ① 的形式选择 |
| E 位宽曲线 | 1.75 / 2 / 2.5 / 3 bpw（RAVQ vs QTIP） | 收益随比特的变化，可以画成 Fig. 6 |

- 表里只放闭环成功率和配对结果。开环指标要不要放，见 ❓1。
- 表的形式参考 Q-WAM 的 Table 3 和 QuantWAMs 的 Table 3：累积式阶梯，每一行在上一行基础上加一个部件，比特预算固定。A 组就按这个形式；B–D 组是在完整 RAVQ 上替换某一项，可以放在同一张表的下半部分，用横线分隔。
- 消融正文围绕**设计动机**写（指南）：每组先说"如果 C1 成立，应该看到……"，再说结果。

### 6.5 Analysis（对应 Q4，图为主）
1. **比特去了哪里（Fig. 5）**：横轴视频块 0–29 / 动作块 0–29，纵轴层类型（q / k / v / o / ffn.0 / ffn.2 / cross_attn），格子颜色为位宽。一眼看出比特集中在视频专家后半段的输出侧投影。✅ 数据已有。
2. **动作在读哪里（policy gaze）**：§4.4 的热力图如果放在 §4，这里可以再给 2–3 个不同任务的例子，或者放附录。
3. **按扰动维度拆分（Fig. 6 或小表）**：BF16 与量化方法的差距集中在相机视角（BF16 74 vs W2 59–64）、背景、传感器噪声；语言和光照无损失。✅ 基线数字已有。RAVQ 的收益是否也集中在这些维度 ⏳。
   - 可以加一句解释：这些扰动下，动作更依赖预测的 future。⚠ 这是推测，需要数据支持才写（例如在这些维度上单独算 future 占比）。
4. **预测 future 的定性对比（Fig. 7）**：同一场景下 BF16 / QTIP-W2 / RAVQ-W2 的解码帧 + 执行结果，选一个成功 / 失败分歧的例子。⏳
5. **失败案例**（指南：要展示失败案例）：RAVQ 仍然失败的场景，说明局限。

### 6.6 Efficiency（Table 4，对应 Q5）
- 列：方法 | bpw | 权重显存（GB）| 峰值显存 | policy call 延迟 p50 | 量化 GPU·h。
- **已知**：目前所有臂都反量化成 BF16 在同一运行时上跑，延迟（0.50–0.52 s）不反映各方法自己的 kernel；BF16 + cache + CUDA Graph 的 policy call 为 541 ms。
- **⏳ 需要**：
  - 用 QTIP 的打包格式与解码 kernel 实测显存和延迟，至少一个 GPU，最好再加一个端侧设备；
  - 核实 K = 1 层的 kernel 支持（官方 kernel 只有 K = 2–4）。
- 这一节决定 "edge memory" 动机能否闭环。

### 6.7 Real-world experiments（Fig. 8 + 小表，对应 Q5）
- **作用**：摘要以真机结果收尾，这一节是摘要最后一句的出处。
- 平台：AgileX PiPER 机械臂（6 自由度 + 夹爪，7 维关节空间动作）。已有一批采集数据：`agilex pepper training data/run_20261023_141641`，40 条，10 fps，任务如 "place the lemon to the box"；另有 run_20261023_150009。数据尚未验证。
- 前置工作：Fast-WAM 要先在 PiPER 数据上微调出 BF16 策略，再在它上面做量化。这一步工作量大，要尽早排期。⏳
- 设定：BF16 / QTIP-W2（最强 2-bit 基线）/ RAVQ-W2，XX 个任务 × 每任务 XX 次。⏳ 全部待定。可参考 `research/real-robot/2026-10-26/` 里的任务建议（Stack bowls、Hang mug、Carrot to plate 等）。
- 附带报告：真机设备上的权重显存（≈ 1.5 GB vs 12 GB），与摘要最后一句对应。
- 形式参考 Q-WAM 的 Table 2 和 QuantWAMs 的 Table 6 + Fig. 6：表里记"成功次数 / 试次"，配一张真机系统与任务的图。
- 版面不够时：正文保留一张图（场景 + 成功率条形图），细节放附录。

---

## 7. Conclusion（约 0.2 页）+ Limitations

- **结论三句**：
  1. 极低比特量化让 WAM 丢掉了预测 future 的保真度，动作误差主要由此而来；
  2. 把 future 保真度作为显式的量化目标（按 policy 读 future 的方式衡量、按 future 影响分配比特、以 future 保真度校准），就能在不加比特的情况下补回闭环性能，仿真和真机都成立；
  3. 这说明压缩 WAM 时，预测的 future 不是可以随便牺牲的中间量。
- **Limitations（真实、具体，一次说清）**：
  - ② 的模块级分配依赖反事实升位测量，在模块间差异小于测量噪声的模型上会退化为纯率失真分配（视第二个 WAM 的结果改写）；
  - reader 度量在 teacher future 处线性化；
  - 只评测了 weight-only，激活仍为 BF16；
  - 闭环评测在仿真为主，真机规模有限。

---

## 8. Supplementary Material（列提纲即可）

1. LIBERO-Plus 三个面板的构造、seed 规则、配对检验细节，各面板分别的结果。
2. 完整分 suite × 扰动维度的结果表（所有方法、两个位宽）。
3. bpw 计数细则：每类层的位宽、接口层、时间表、解码表。
4. Reader KL 的定义（16 bin，σ = 0.375，按 motor 坐标标准化）；λ、β 的取法。
5. ① 的估计细节：探针数、单位迹归一、低秩截断，以及谱和稳定性的图。
6. ② 选中的全部视频层列表。
7. ④ 的步长规则和 FP32 折算。
8. 与 W4A4 方法的参照（同一面板：SpinQuant 203，ViDiT-Q 203，QuaRot 195，均为 280）；GPTQ 的分面板结果（主表只放合计）。
9. 更多 policy gaze 和预测帧的可视化。
10. 计算开销明细。

---

## 附 A. Claim–证据对照

| Claim | 位置 | 证据 | 状态 |
|---|---|---|---|
| 端侧部署需要极低比特，而极低比特下只有 VQ 可用 | 摘要 S2，§1 ¶1–¶2，§6.2 | Fast-WAM 6B、BF16 权重 12 GB；GPTQ W2 15/280、W3 195/280 vs VQ W2 198–204、W3 202–205 | ✅ Fast-WAM 论文；A16 基线 26042316 等 |
| 极低比特下动作误差主要来自 future 失真 | 摘要 S3，§1 ¶3，§4.2 | future 占比 73%（W2）/ 85%（W3），公平口径 QTIP | ✅ 26072226 |
| 三个宿主上都成立 | §4.2 | 换 BF16 future 后 motor RMSE 降 24 / 36 / 36%（MSE 占比需统一重算） | ✅ 26050310（口径待统一） |
| future 的失真来自生成它的模块 | §1 ¶3，§4.3 | 视频专家拿走 71% 升位收益；只读 future 的动作专家对 future 的影响为 0 | ✅ 26069856 + 26070509 |
| 动作只读 future 的一小部分 | §4.4 | 有效秩 41.5 / 37,632，第 1 帧 98%，对角稳定 0.86 | ✅ 26073871 |
| 极低比特下现有方法闭环显著掉点 | §1 ¶3，§6.2 | W2 −36…−50 / 840，W3 −21…−26 / 840 | ✅ 26073449 |
| RAVQ 同 bpw 胜过最强宿主 | 摘要 S6，§6.2 | RAVQ 840 vs 公平口径宿主 840 | ⏳ |
| future 保真项必要 | §6.4 B 组 | A-only vs 完整目标的闭环 | ⏳（你来跑） |
| 按 future 影响分配比特是收益来源 | §6.4 C 组 | 反向分配 / 随机选层 | ⏳ |
| 跨 WAM 成立（C1 与 RAVQ） | 摘要 S5–S6，§6.3 | 第二个模型的 future 占比 + 主表 | ⏳ ❓2 |
| 省显存、可部署 | §1 ¶1，§6.6 | 打包格式下的显存和延迟 | ⏳ |
| 真机成立 | 摘要 S7，§6.7 | AgileX PiPER 成功次数 / 试次 | ⏳ |

## 附 B. 图表清单与版面预算（CVPR 正文 8 页）

| 编号 | 内容 | 位置 | 大小 | 状态 |
|---|---|---|---|---|
| Fig. 1 | Teaser：两条误差路径 + 成功率–bpw | §1 | 双栏宽，约 1/3 页 | ⏳ 需 RAVQ 数字 |
| Fig. 2 | future 占比（三宿主 × 两口径 × 两位宽）条形图 | §4.2 | 单栏 | ✅ 数据有，待画 |
| Fig. 3 | 左：逐组升位收益（A / Z / K 堆叠）；右：policy gaze 热力图 | §4.3–4.4 | 双栏宽 | 左 ✅ 数据有；右 ⏳ 待导出 |
| Fig. 4 | RAVQ 主图 | §5 | 双栏宽，约 1/3 页 | 待画 |
| Fig. 5 | 位宽分配图 | §6.5 | 单栏 | ✅ 数据有，待画 |
| Fig. 6 | 按扰动维度的成功率 / 位宽曲线 | §6.5 | 单栏 | 部分 ⏳ |
| Fig. 7 | 预测 future 定性对比 | §6.5 或附录 | 单栏 | ⏳ |
| Fig. 8 | 真机 | §6.7 | 单栏 | ⏳ |
| Table 1 | 主表 | §6.2 | 双栏宽 | 部分 ✅ |
| Table 2 | 跨 WAM 结果 | §6.3 | 单栏 | ⏳ |
| Table 3 | 消融 | §6.4 | 单栏 | ⏳ |
| Table 4 | 效率 | §6.6 | 单栏 | ⏳ |
| Alg. 1 | RAVQ 流程 | §5 | 单栏 | 待写 |

页数分配：

| 部分 | 页数 |
|---|---:|
| 摘要 + 引言 | 1.25 |
| 相关工作 | 0.6 |
| 预备知识 | 0.5 |
| C1 分析 | 1.25 |
| 方法 | 1.5 |
| 实验 | 2.5 |
| 结论 | 0.2 |
| 合计 | 约 7.8 |

图表多于版面能放下的量。摘要现在点名了"跨 WAM"和"真机"，所以 Table 2 和 Fig. 8 必须留在正文；Fig. 7 和 Table 4 中至少一项进附录（显存已并进 Table 1，Table 4 只剩延迟，最适合挪走）。

## 附 C. 写论文需要、目前还没有的实验

按对论文的必要性排序（第二版：摘要点名了跨 WAM 和真机，这两项上调）：
1. **RAVQ W2 / W3 的 840 闭环**（完整版，含 ④）。未校准版 RAVQU 的 840 在跑（fwr840u 26075585）。
2. **三个宿主在公平口径下的 840 闭环**（GPTVQ / QuIP# / QTIP × W2 / W3，加接口 BF16 + 时间表）。没有它，主表的比较不公平。
3. **真机（AgileX PiPER）**：先验证已采数据、在 PiPER 上微调出 BF16 的 Fast-WAM，再跑 BF16 / QTIP-W2 / RAVQ-W2。前置链条最长，要最早启动。
4. **第二个 WAM**：先在它上面复现 future 占比（C1）和反事实升位（决定 ② 怎么分），再跑主表。
5. **消融 B 组**（A-only vs 完整目标）：论证 C1 在方法层面的关键证据（你来跑）。
6. **效率实测**：打包格式下的显存、延迟；核实 K = 1 的 kernel。
7. **GPTQ W3 补到 840**（面板 B、C），让主表里"标量量化 vs VQ"的对比和其他行同一规模。
8. 画图用的导出：M 对角的热力图、预测帧解码。
9. 可选的闭环版 C1 证据：BF16 视频专家 + W2 动作专家 vs W2 视频专家 + BF16 动作专家，840 闭环。开环的 73% 用闭环再印证一次，C1 会更硬，也能回避 ❓1 的顾虑。

## 附 D. 需要你决定的事项

- **❓1 开环测量进不进正文。** C1 的三条证据（§4.2–4.4）都是 held-out 场景上的开环测量。我按"它们是现象分析，不是方法性能的代理"来处理，放进正文；方法的所有性能结论只用闭环。如果你的意思是任何 held-out MSE 都不进论文，§4 需要改成以附 C 第 9 项的闭环版证据为主。
- **❓2 第二个 WAM 选哪个。** 叙事上已经按你的意见把 ② 写成一般形式（按各模块对 future 保真度的实测影响分配），不再绑定"动作专家 → 视频专家"，所以共享骨干的 WAM 也能用。剩下的是选哪个模型：Cosmos-Policy（已有 reader 实现，耦合方式与 Fast-WAM 最不同，最能说明一般性）还是 LingBot-VA（QuantWAMs 也用了它，审稿人熟悉）。
- **❓3 题目。** §题目 的三个候选（第二版已按"保住 future"的叙事重拟）。
- **❓4 Fig. 1 用方案 A（路径 + 曲线）还是方案 B（具体样本）。**
- **❓5 W3 在论文里的分量。** W3 宿主到 BF16 只差 21–26 次，840 上显著胜出宿主约等于追平 BF16。建议主说法放在 W2，W3 作为完整性结果。
- **❓6 要不要和 Q-WAM / QuantWAMs 对比。** 两者都是 W4A4，与我们的 weight-only 2–3 bpw 不是同一设定。我的建议：正文在相关工作里说明设定不同；附录给一行同面板的 W4A4 参照（SpinQuant / ViDiT-Q / QuaRot 已有 280 的结果）。审稿人看到两条极低比特路线放在一起，就不会追问。
- **审稿预期提醒（来自 paper 库）**：Q-WAM 在 3 个 WAM 和 5 个真机任务上评测，QuantWAMs 在 Fast-WAM 和 LingBot-VA 两个 WAM、RoboTwin 和 LIBERO 两个基准以及真机上评测。审稿人很可能拿这个规模来比。所以 ❓2（第二个 WAM）和真机实验的优先级很高；只有一个模型、一个基准，在这个方向上容易被认为证据不足。

## 附 E. 写作风格参考（paper 库）

提取了 10 篇论文的章节结构：QTIP、QuIP#、GPTVQ、GuidedQuant、AQLM（LLM 极低比特量化），Q-WAM、QuantWAMs（WAM 量化），Fast-WAM（我们的主模型），QVGen、DSAQuant（视频生成量化）。

### E.1 共同结构（我们的大纲已经对齐）

| 部分 | 10 篇论文的共同做法 | 本大纲 |
|---|---|---|
| 摘要 | 5–13 句；背景 → 问题 → 方法 → 组件 → 结果 | 5 句（可加第 6 句 takeaway） |
| 引言 | 4–6 段；背景 → 痛点 → 已有局限 → 本文思路 → 结果概述 | 5 段 + 3 条贡献 |
| 贡献 | WAM 方向的论文基本都是 3 条 bullet | 3 条 bullet |
| 方法 | 先 Preliminaries / 问题形式化，再按组件或阶段分节 | §3 预备知识 + §5 四个部件 |
| 实验 | 主结果 → 消融 → 效率；细节用 run-in 小标题 | §6.2 → §6.4 → §6.6 |
| Teaser | 多数放在第 1 页 | Fig. 1 |

### E.2 具体借鉴了什么、用在哪里

| 写法 | 出处 | 用在本大纲的位置 |
|---|---|---|
| 诊断先行，方法组件与诊断一一对应；第一条贡献写"发现" | DSAQuant | §4 整体结构；贡献 C1 |
| 观察 → 原则 → 方法（"Based on this principle, we propose"） | QuantWAMs | 引言 ¶3 → ¶4 的衔接；§4.5 Takeaways |
| 设问式开头（"A natural first question is whether …"） | Q-WAM | 引言 ¶3 的备选开头；题目候选 3 |
| 摘要结尾一句 takeaway，把结果提升为一般原则 | DSAQuant | 摘要可选 S6 |
| 摘要结果句给具体区间和幅度 | Q-WAM、QVGen | 摘要 S5 |
| Teaser 放第 1 页，caption 两句直接写结论 | QuIP# | Fig. 1 |
| 第 1 页 teaser 表，成对列出"基线 → 基线 + 本文" | GuidedQuant | Fig. 1 方案 C |
| 方法节内用 run-in 小标题分层 | Q-WAM、Fast-WAM | §5 各小节 |
| 部件名统一成"对象 + 动作"的结构，方法与消融对应 | QuantWAMs | §5 小节标题 |
| 方法节内嵌一张小验证表 | QTIP | §5.2 位宽分配表 |
| 每个设计选择配一张证据图 | QVGen | §4 各小节 |
| 实验小节按"问题"组织 | QuIP# | §6 开头的 Q1–Q5 |
| 小节或段落标题即结论（"Pareto optimality of AQLM."） | AQLM、QuantWAMs | §4 小节标题；§6.4、§6.5 的 run-in |
| 实验设置用 run-in（Models. / Training. / Evaluation.） | QVGen、DSAQuant、Q-WAM | §6.1 |
| BPW、显存列并入主表 | Q-WAM、QuantWAMs | Table 1 |
| 表注里定义加粗、下划线的含义 | GPTVQ、QVGen | Table 1 |
| 累积式消融阶梯，比特预算固定 | Q-WAM、QuantWAMs | Table 3 的 A 组 |
| 真机表记"成功次数 / 试次"，配系统与任务图 | Q-WAM、QuantWAMs | §6.7 |
| 结论用 "suggest that" 一类的措辞，与证据范围一致 | Fast-WAM | §7 |

### E.3 有意没有照搬的

- **Fast-WAM 的"受控变体放在同一张主表的分组行里"**：我们的消融（Table 3）规模较大，放进主表会挤掉 suite 列，所以单独成表。
- **GuidedQuant 的贡献写成 run-in 段落**：它适合贡献之间互相依赖的情况。我们的三条贡献（发现 / 方法 / 评测）界线清楚，用 bullet 更醒目。

完整提取笔记（每篇的摘要逐句功能、章节树、引言逐段功能、图表清单）在 `C:\Users\Raymond\AppData\Local\Temp\claude\D--Downloads-Final-Year-Project\8c18cf31-2f4a-49f5-b832-7af084307091\scratchpad\style\STYLE_NOTES.md`。这是临时目录。如果你想长期保留，我可以把它移到 `research/writing-guide/` 下。
