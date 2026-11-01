# 论文大纲（CVPR 2027 投稿稿，2026-11-01 第五版）

依据：[论文写作指南笔记](../writing-guide/论文写作指南笔记.md) 的结构要求 + 已定的论文主线（C1 → C2 → C3）+ 截至 2026-11-01 的实验结果。

**第二版（按你对摘要的意见改）**：叙事不再限定 "imagine-then-execute"，问题统一为"现有 VQ 保不住预测 future 的保真度"；加入"端侧显存 → 极低比特 → VQ"的论证；② 写成"按各模块对 future 保真度的影响分配比特"的一般形式；③ 定位为初始化，理由是原生支持混合位宽；结果先报跨 WAM 的评测范围，以 AgileX 真机收尾。改动覆盖题目、摘要、引言、相关工作、预备知识、C1 分析、方法、实验、结论和各附录（指当时的章节划分）。

**第三版（结构调整，参照 SteerQuant 第 3 节）**：
- 原 Preliminaries 和 C1 分析合并为新的 §3 Preliminaries and Motivation：3.1 只放定义，3.2 是协议段加三个结论段，共用 Fig. 2 三联图。
- Takeaways 删掉，方法各部件改为引用 §3.2 (a)–(c)。
- 原 §5–§7 依次改为 §4–§6，附录 Supplementary 改为 §7。原 Fig. 2 和 Fig. 3 合并为 Fig. 2，后面的图依次前移一号。
- 2.1 压缩到两三句；Fast-WAM 的具体结构和联合去噪模型的 reader 都挪到 §5.1（第四版起，原先放在跨 WAM 小节的 reader 说明也并入 §5.1）。

**第五版（初稿已写成 LaTeX）**：
- 初稿在 `CVPR2027/`（main.tex + sec/0_abstract … 6_conclusion，X_suppl 只有大纲）。图先放空白占位框，表格已搭好，缺数据的地方写 XX。
- Table 1 去掉了胜 / 负和 p 值列。
- 按你的意见，"两种口径的差别""差异在噪声范围内""为什么用 840"这类对故事没有帮助的内容，正文一律不写；大纲里相应的条目也已删改。

**第四版（按你的意见改实验节）**：
- §3.2 的协议段压到三四句（对齐 SteerQuant），细节挪到附录 A；
- 原"主结果"和"跨 WAM 结果"合并成 §5.2，Table 1 按 WAM 分块；
- 删掉 Analysis 小节，内容并入 §4.2、§5.2 或附录（理由见 §5 开头）；
- 新写 §5.3 真机：AgileX PiPER，三个任务，Table 2 骨架，结果留 XX；
- 实验节改为 5.1 设置 / 5.2 主结果 / 5.3 真机 / 5.4 消融 / 5.5 效率，附录提纲改成 A–I 并按 SteerQuant 把初步分析放在附录 A。

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
- **(ii)** 按意见写虚：*distributes a fixed bit budget according to each module's measured impact on future fidelity*。它不提"动作专家 → 视频专家"，所以对共享骨干的 WAM 同样成立。在 Fast-WAM 上，这条规则的结果恰好是把动作专家的比特挪给视频专家，这个结论放到 §4.2 再讲。如果还想更虚，可以写成 *allocates bits where they best preserve the predicted future*。
- **(iii)** 写成"初始化"，并给理由：*initializes the codes with trellis-coded rounding, which natively supports the resulting mixed bit widths*。"resulting" 指 (ii) 产生的混合位宽，两句因此自然衔接。
- **(iv)** 校准共享解码表，不增加存储。

**和第一版相比改了什么**
- 去掉 "imagine-then-execute" 的限定，问题改为 "fail to preserve future fidelity"。
- 新增端侧显存 → 极低比特 → VQ 的论证（S2），用 GPTQ W2 崩溃的实测数作支撑。正文 ¶1–¶2 有对应展开。
- S3（问题句）不再提"比特没给想象网络"，只说现有方法没有有意识保护 future。
- 删掉了 73% 这个数。它是 Fast-WAM 上的具体测量，放进摘要等于把故事绑在这一个模型的结构上；它留在引言 ¶3 和 §3.2。如果你想在摘要里保留一个定量的问题证据，可以在 S3 末尾加 *(at 2 bits, the degraded future accounts for 73% of Fast-WAM's action error)*。
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
全文统一用 *predicted future*（中文"预测的 future"），不用 *imagine-then-execute* 作限定。Fast-WAM 的"先去噪 future 再去噪动作"只作为具体实例出现在 §5.1（实验设置）。

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
- **要点（三条观察，每条一句，详细放 §3.2；数字来自 Fast-WAM）**：
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
- **要点（一句一个部件，顺序与 §4 一致，措辞与摘要一致）**：
  - (i) 对应观察 3：用 policy 读 future 的度量（reader metric）衡量 future 误差，替代均匀的 NMSE。
  - (ii) 对应观察 2：在固定的比特预算下，按每个模块对 future 保真度的实测影响分配比特。
  - (iii) 用格栅码舍入初始化各层的码，选它是因为它原生支持 (ii) 产生的混合位宽。
  - (iv) 对应观察 1：以动作和 future 保真度（A + λZ_M + βK）为目标，校准各组共享的解码表，不增加存储。

### ¶5 结果预告

- **作用**：先说评测范围，再给最重要的数字，和 ¶3 的问题对应；顺序与摘要一致。
- **主题句草稿**：*We evaluate RAVQ across XX WAMs, both in simulation and in real-world deployment.*
- **要点**：
  - 仿真：LIBERO-Plus 840 配对，2-bit RAVQ 把 Fast-WAM 的成功率从 73.3%（QTIP，最强的 2-bit 基线）提高到 XX%，补回 BF16（77.6%）差距的 XX%。W2 是主说法；W3 作补充，因为 W3 基线到 BF16 只差 21–26 次，统计功效不够（见 §5.2）。
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
- **写法**：两三句即可，不展开分类。
- **草稿**：*World action models train a video generator jointly with an action model, so that actions are conditioned on a predicted future [Fast-WAM, Cosmos-Policy, LingBot-VA, DreamZero]. They improve generalization over VLA policies but inherit the size of video backbones. We do not modify their training and study only how to compress them for deployment.*
- 可选半句呼应：Fast-WAM 讨论过测试时是否还需要预测 future；我们从压缩的角度说明，部署时 future 的保真度直接决定动作。

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

## 3. Preliminaries and Motivation（约 1.5 页，含 Fig. 2）

**结构（参照 SteerQuant 第 3 节 "Preliminary and Motivation"）**：3.1 只放后文要用的定义，最后一句引出观察；3.2 先用一段写测量协议，再写三个加粗的 run-in 结论段，三段共用一张三联图（Fig. 2），每段最后一句落在"所以保护应该……"，然后直接进入 §4 方法。原来单独的 Preliminaries 和 C1 分析两节合并成这一节，Takeaways 小节删掉，方法里改为直接引用 §3.2 (a)–(c)。

写法参考：
- DSAQuant 是"诊断先行"：方法的每个组件对应一条诊断。§3.2 的三条观察分别对应 ④、②、①，是同一结构。
- QVGen 每个设计选择都配一张证据图。三个结论段对应 Fig. 2 的三个子图。
- run-in 标题直接写结论句，不用 "Analysis of X" 这类名词短语。

### 3.1 Quantizing a WAM

- **作用**：只放 §3.2 和 §4 要用的定义，不讲具体模型。
- **WAM 的通用形式**：
  - 主题句草稿：*A WAM maps an observation o and an instruction ℓ to a predicted future latent z and an action chunk a, and the action is conditioned on z.*
  - reader：把"给定 z 计算 a"的那部分计算记为 R(o, ℓ, z)。它是 §4.1 中 ① 和 K 的基础，也是方法名里 "reader-aware" 的来源。
  - 耦合方式只用一句话带过：*Whether z and a are generated sequentially or denoised jointly, R is well defined.* Fast-WAM 的具体结构和形状、联合去噪模型的 reader 实现，都放到 §5.1 和附录。
- **Weight-only VQ**：
  - 层目标：min tr(ΔW H ΔWᵀ)，H = E[xxᵀ] 是校准集上的输入二阶矩（下文记 H_U）。
  - QTIP 的解码形式：W = diag(SV) Φ_mᵀ (Wscale · LUT[code]) Φ_n diag(SU)，Φ 为随机 Hadamard 旋转（不用 H，避免与 Hessian 冲突；R 已用于 reader）。权重先经旋转；在旋转域中，每对权重由格栅码状态查 LUT 解码，每个权重 K bit。§4.3 的混合位宽和 §4.4 的解码表校准都基于这个形式。
  - bpw 计数口径：码 + 每层 scale + 表，接口层 BF16 也按 16 bit 计入总 bpw。所有方法用同一口径，在这里一次说清。
- **误差度量**（§3.2 和 §4.4 共用）：
  - A：前 10 步动作的误差（motor RMSE）；
  - Z：future NMSE，‖z_q − z_T‖² / ‖z_T‖²，只算预测帧；
  - K：reader KL。把 reader 的输出按 motor 坐标标准化后分成 16 个 soft bin（σ = 0.375），比较量化模型与 BF16 的分布。
- **两条误差路径与 future 占比**（原 4.1）：
  - 主题句草稿：*A weight error can change the action either directly, or by degrading the predicted future on which the action is conditioned.*
  - 定义：future 占比 = 1 − MSE(R_q(z_T)) / MSE(R_q(z_q))。也就是把量化模型自己预测的 z_q 换成 BF16 的 z_T 后，动作误差还剩多少。分母是完整量化模型的动作误差，分子是 future 修好之后、只剩直接路径的误差。
  - 一句白话解释：这个比例就是"动作误差里由 future 失真造成的那一部分"。定义只用到 reader，对任何 WAM 都能算。
- **符号表**：o、ℓ、z、a、R、W_l（第 l 层权重）、ΔW_l、H_l、A、Z、K。
- **结尾一句（引出 3.2）**：*Existing quantization objectives measure layer outputs or the final action, but never the predicted future on which the action depends.* 下一节回答：这个遗漏到底让模型丢了多少。

### 3.2 Observations

**协议段**：一段三四句，详细程度对齐 SteerQuant。只交代做了什么、在哪个模型上、测什么，最后一句指向附录。
- 草稿：*We examine how ultra-low-bit weight quantization affects the predicted future and, through it, the action. We quantize Fast-WAM with three 2-bit VQ methods and measure the action error, the future error and the future share (Sec. 3.1) on held-out LIBERO contexts. To locate where future error originates, we additionally raise one module group at a time from 2 to 3 bits. Appendix A provides the full protocol and results.*
- 挪到附录 A 的细节（正文不写）：
  - 样本：C80 校准集以外的 40 个场景 × 2 个 seed；
  - 口径：与 §5 所有方法一致（接口层 BF16 + 时间条件表）。不报告也不讨论其他口径；
  - 模块组的划分：64 组，每个专家 × 块 + 条件 / 头 / 接口；
  - K 的 soft-bin 参数；
  - 第二个 WAM 上对观察 (a) 的复现（future 占比）。⏳

**Fig. 2（三联图，三段共用）**：
- (a) future 占比条形图：三个方法 × 两个位宽（统一口径）；
- (b) 逐组升位收益：横轴为组（视频块 0–29 | 动作块 0–29 | 接口 / 条件），纵轴为升位收益，按 A / Z / K 三色堆叠；
- (c) policy gaze 热力图：把 M 的对角按 14×28 网格、两路相机画成热力图（第 1 个预测帧），叠在对应的观测图像上，看动作在"看"哪里（预期在夹爪和目标物附近）。
- Caption 草稿：*Figure 2. What an ultra-low-bit WAM loses. (a) Restoring the predicted future removes most of the action error. (b) Future error comes from the modules that generate the future; modules that only read it leave the future untouched. (c) The action is sensitive to a narrow region of the predicted future.*
- 状态：(a)、(b) ✅ 数据已有，待画；(c) ⏳ 需要从 M 导出。

**(a) *Most action error comes from a degraded future.***
- 要点：
  - 把预测的 future 换回 BF16 版本，公平口径 QTIP 的动作误差在 W2 下去掉 73%，在 W3 下去掉 85%。✅ 26072226
  - GPTVQ、QuIP# 在统一口径下的 future 占比（MSE）还没有测，⏳ 补测后填进 Fig. 2(a) 和正文的 "XX%–XX%"。
  - 正文只报告统一口径下的数字，不讨论口径差别（按你的意见：对整体故事没有帮助的内容不写进正文）。
- 末句（所以保护应该……）：*This implies that the quantization objective must include the predicted future itself, not only the action.* → ④ 的目标包含 Z_M 和 K。
- **❓1**：这一段和 (b)、(c) 都是开环测量（held-out 场景上的动作 / future 误差）。我的理解是：你说的"held-out MSE 诊断不进论文"，指的是用来调 ④ 的那些迁移类诊断；C1 的现象测量属于动机分析，可以进论文。请确认（附 D ❓1）。

**(b) *Future error originates in the modules that generate it.***
- 要点：
  - 每次把一个模块组从 W2 换成 W3，测 A、λZ、βK 三项的变化（分组方式见附录 A）。
  - 生成 future 的视频专家合计拿到全部升位收益的 71%，主要经 Z 和 K。
  - 只读 future 的动作专家，30 个块全部升位，收益约为 0，对 Z、K 恰好为 0（视频 token 不读动作专家，这是结构上的必然）。✅ 26069856 + 26070509
  - 反向验证：动作专家主体降一档（2→1 bit），future 完全不变，动作一侧自身误差 +24%，总 motor 只 +8%。✅ 26072226
- 叙事要点：这一段讲"future 的误差从哪里来"，不讲"比特该怎么分"。比特分配是方法（§4.2）从这条观察推出来的结论。
- 写作注意：Fig. 2(b) 只强调**模块级**的对比（生成 future 的模块 vs 只读 future 的模块），不对单个视频块的高低做解读，也不写"差异在噪声范围内"一类的话。
- 末句（所以保护应该……）：*This implies that the bit budget should follow each module's impact on future fidelity.* → ②。

**(c) *The action reads a narrow part of the future.***
- 要点：
  - reader 度量 M（定义见 §4.1）的有效秩为 41.5，而 future 维度是 37,632；前 32 个方向占迹的 52%；第 1 个预测帧占 98%；前 10% 的对角元占 62%。
  - 稳定性：奇偶两半样本估计的对角相关为 0.86。✅ 26073871
- 写作注意：只报告对角的稳定性。前 32 个方向在两半样本之间的重合度只有 0.19，这一点用来说明 ① 的 M 为什么采用"低秩 + 对角 + ρI"的形式，并在消融里比较"只用对角"（§5.4）。不在正文展开为负面结论。
- 末句（所以保护应该……）：*This implies that future error should be measured in the metric through which the action reads it, rather than by a uniform NMSE.* → ①。

**收束句（一两句，SteerQuant 的做法，接 §4）**：*Together, these observations call for a quantizer that treats the predicted future, as the policy reads it, as a quantization target in its own right.*

---

## 4. Method: RAVQ（约 1.5 页，含主图 Fig. 3 + Algorithm 1）

指南要求：先 Overview（配主图），再分模块；每个模块讲清解决什么问题、输入、处理、输出、为什么这样设计；公式前后都要解释；novelty 点到即止。

写法参考：
- Q-WAM、Fast-WAM 在方法节内用 run-in 小标题（*Overview.* / *Architecture.* / *Training objective.*）分层，读者可以快速跳读。本节每个小节内部建议也用 run-in：*What it fixes.* / *Definition.* / *Estimation.* / *Why this form.*，正好对应指南要求的"解决什么、怎么做、为什么"。
- AQLM 用动词短语作为同级小节标题。本节四个小节标题可以统一成"对象 + 动作"的结构（借鉴 QuantWAMs），方便和消融表对应，例如 *Measuring the future as the policy reads it* / *Allocating bits by future impact* / *Initializing with trellis codes* / *Calibrating shared decode tables*。

### 4.0 Overview
- **主题句草稿**：*RAVQ quantizes a pretrained WAM in four steps, each answering one of the observations in Sec. 3.2.*
- **流程（与主图从左到右一致）**：① 估计 reader 度量 M → ② 在总比特约束下，按各模块对 future 保真度的影响分配位宽 → ③ 用格栅码舍入初始化各层的码 → ④ 校准各组共享的解码表。
- **一句交代不变量**：推理时的计算图、kernel 和存储格式都与 QTIP 相同；①、④ 只在离线量化阶段使用 reader，部署时不需要 reader。
- **Fig. 3 主图设计**（按指南"方法主图"的五步）：
  - Caption 一句话：*RAVQ makes the predicted future an explicit quantization target: it measures, allocates, initializes and calibrates with one signal, how the action reads the predicted future.*
  - 布局：上方一行是通用的 WAM 结构（生成 future 的部分 → 预测的 z → reader → 动作；Fast-WAM 里分别对应视频专家和动作专家），下方一行是四个模块框，用虚线连回 WAM 中各自作用的部位：
    - ① 指向 z；
    - ② 指向各模块，用格子颜色深浅表示位宽；
    - ③ 指向每个 Linear；
    - ④ 指向解码表。
  - 视觉语言：实线 = 推理数据流；虚线 = 离线量化流程；红色 = 校准信号 A / Z_M / K。总共 3–4 种低饱和配色，在顶部加 legend（参考 WAPITI 的做法）。

### 4.1 Reader metric（①，对应 §3.2 (c)）
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
- **为什么这样设计**：低秩部分捕捉主要敏感方向；对角部分稳定（§3.2 (c)）；ρI 保证 future 的任何方向都不会完全不受约束。
- **成本**：只依赖 BF16 模型，估计一次，所有位宽共用。⏳ 补一个 GPU·h 数。

### 4.2 Future-aware bit allocation（②，对应 §3.2 (b)）
- **解决什么**：均匀位宽默认每个模块同样重要，但各模块对 future 保真度的影响差别很大（§3.2 (b)）。
- **约束**：Σ_l n_l K_l ≤ B · Σ_l n_l，其中 B 为目标 bpw，n_l 为层 l 的权重数。
- **一般规则（正文按这个写，与摘要的 (ii) 措辞一致）**：
  1. **模块级，按 future 影响**：对每个模块组测"升一位能让 future 保真度（Z、K）改善多少"，即 §3.2 (b) 的反事实升位实验。对 future 没有影响的模块降一档，省下的比特给影响最大的模块。
  2. **组内，按率失真**：在模块组内部，按"每多 1 bit 能降低多少按权重数加权的 H_U 误差"贪心选层，升到 K + 1，直到预算用完。这是混合精度的标准做法，正文直接写做法，不解释为什么组内不按 future 影响分配。
- **在 Fast-WAM 上的实例**：动作专家只读 future，对 future 保真度的影响恰好为 0，所以动作专家主体降一档（K − 1）；省下的比特全部给生成 future 的视频专家主体，再在视频专家内部按率失真选层。
- **在共享骨干的 WAM 上**：每层同时处理 future 和动作 token，规则不变，只是模块组改成块或层类型。⏳ 以第二个 WAM 的实际实现为准。
- **叙事要点**：正文先写一般规则，再写 Fast-WAM 上的结果。"把比特从动作专家挪到视频专家"是规则在这个模型上的**结果**，不是方法的定义。这样方法不绑定分专家结构。
- **公平口径（所有方法一致，正文必须明说）**：
  - 接口层（action_encoder、动作 head、proprio_encoder、视频 head）存 BF16，占权重 0.01%；
  - 时间条件层只看 timestep，存成按 timestep 的输出表，输出与 BF16 完全一致，比特比 W2 还少。
  - 这两项在 §5 的所有基线上同样开启。
- **结果描述（放在这里）**：2-bit 预算下买到 48 个视频层（占权重 16.7%），集中在视频专家后半段（block 15–29）的 self_attn.o / v 和 ffn.2。总 bpw 1.9932，与公平口径的 QTIP 完全相等。✅ 26072226
- **组内为什么用率失真**：正文不展开，也不写"差异在噪声范围内"。如果审稿人问，rebuttal 里再答。
- **方法节内嵌一张小表（借鉴 QTIP 在方法节里直接放验证表）**：按模块列出每类层的平均位宽，让读者一眼看到预算怎么挪。数据 ✅ 26072226：

  | | 视频主体 | 动作主体 | 文本条件 | 接口 | 时间表 | 总 bpw |
  |---|---:|---:|---:|---:|---:|---:|
  | QTIP（公平口径）2-bit | 2.01 | 2.03 | 2.02 | 16 | 0.07 | 1.993 |
  | RAVQ 2-bit | 2.22 | 1.03 | 2.02 | 16 | 0.07 | 1.993 |
  | QTIP（公平口径）3-bit | 3.01 | 3.03 | 3.02 | 16 | 0.07 | 2.981 |
  | RAVQ 3-bit | 3.22 | 2.03 | 3.02 | 16 | 0.07 | 2.981 |

  完整的逐层位宽分配图放附录。

### 4.3 Trellis-coded initialization（③）
- **定位**：③ 是 RAVQ 的初始化选择，给 ④ 提供起点。正文要写清楚为什么选它。
- **为什么选格栅码（按重要性排序）**：
  1. **原生支持 ② 给出的混合位宽**：格栅码的结构对任意每权重比特数 K 都成立：不同 K 只改变 trellis 每步移入的比特数，查表解码的结构（512×2 的 LUT）不变。所以按层混用 K = 1–4 时，编码和解码方式都是同一套。对照：格点码（如 QuIP# 的 E8P）的码本是按特定比特数设计的，换位宽要换码本或叠加残差级。（⚠ 这句对照写进正文前，再对一遍 QTIP / QuIP# 原文。）这是与 ② 直接衔接的理由，摘要 (iii) 写的就是这一条。
  2. **解码表全组共享**：同一（角色，位宽）组的所有层共用一张解码表，这正好是 ④ 要校准的对象，校准不增加存储。
  3. **极低比特下最强**：2-bit 下它是闭环最好的基线（616/840）。
- **做法**：QTIP 官方 kernel 只支持 K = 2–4，我们把实现放开到 K = 1–4（K = 2 时与原实现逐位一致），逐层用 H_U 舍入。
- 这一节写短，是标准部件。novelty 不在这里。

### 4.4 Shared-table calibration（④，对应 §3.2 (a)）
- **解决什么**：逐层舍入只看各层自己的重建误差，看不到预测的 future 和动作的端到端保真度。
- **参数化**：
  - W = diag(SV) Φ_mᵀ (Wscale · LUT_g[code]) Φ_n diag(SU)（记号同 §3.1）；
  - 只学习每组 g（按（角色，位宽）分组）的解码表偏移 ΔLUT_g，组内所有层共用；
  - 码、scale、旋转全部冻结；参数为 0 时与未校准版逐位相同。
  - ⏳ 最终版是否还包含行 / 列 scale 和输出偏置，以最后上闭环的配置为准。当前 v3 只调共享解码表。
- **目标**：L = A + λ_M Z_M + β K。
  - A：动作 MSE；Z_M：①；K：reader soft-bin KL。
  - λ_M = λ · mean(Z) / mean(Z_M)，让 λ_M Z_M 的初始量级与原来的 λZ 一致；β 按初始质量的固定比例取一次，之后冻结。
  - 公式前说明为什么需要（§3.2 (a)），公式后逐项解释，再说明它让量化模型在"动作 + 被读取的 future"上贴近 BF16。
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

### 4.5 Cost
- 存储：与同 bpw 的 QTIP 相同（解码表为每组 512×2，全模型 ⏳ 个数，可忽略）。
- 量化时间：① + ② + ③ + ④ 的 GPU·h，对比 QTIP / GPTVQ / QuIP# 的量化时间（A16 已有：QTIP W2 1.86，GPTVQ W2 3.84，QuIP# W2 1.28 GPU·h）。⏳ RAVQ 各阶段耗时。
- 推理：与 QTIP 完全相同的 kernel 与格式。⏳ 混合 K（含 K = 1）的 kernel 支持情况要核实，见 §5.5。

---

## 5. Experiments（约 2.5 页）

**第四版结构调整**：
- 原"主结果"和"跨 WAM 结果"合成一节，主表按 WAM 分块，一张表放全；
- 原 Analysis 小节删掉，内容分别并入 §4.2、§5.2 的正文或附录；
- 真机提前到主结果之后。

**为什么不设 Analysis 小节（核对过 paper 库）**：参考的 11 篇论文（QTIP、QuIP#、GPTVQ、GuidedQuant、AQLM、Q-WAM、QuantWAMs、Fast-WAM、QVGen、DSAQuant、SteerQuant）都没有独立的 Analysis 小节。
- 实验节的结构都是"设置 → 主结果 → 消融 → 效率"，WAM 论文再加真机。
- 分析类的图要么放进消融（QuantWAMs 的 Fig. 3–5 都在 §4.4 消融里），要么放附录（QTIP、GPTVQ、GuidedQuant 的消融都在附录）。

原 Analysis 小节里的五项也都有更合适的位置：
- 比特去了哪里：§4.2 已经有按模块列位宽的小表，完整的逐层位宽图放附录；
- 动作在读哪里：主例已在 Fig. 2(c)，更多例子放附录；
- 按扰动维度拆分：§5.2 正文用一句话点出收益集中在哪些维度，完整表放附录；
- 预测 future 的定性对比、失败案例：放附录。

开头一段写清实验要回答的问题：
- **Q1**：在多个 WAM 上、同 bpw 下，RAVQ 的闭环成功率是否超过现有的极低比特量化器？（§5.2）
- **Q2**：在真实机器人上是否成立？（§5.3）
- **Q3**：每个部件的贡献是什么？future 保真项是否必要？（§5.4）
- **Q4**：省了多少显存，速度如何？（§5.5）

写法参考：
- QuIP# 的实验小节按"问题"组织而不是按"方法"组织，读者可以按自己关心的问题跳读。Q1–Q4 与 §5.2–§5.5 一一对应。
- SteerQuant 的主结果节用 run-in 按模型分段（*Results on LIBERO.* / *Results on RoboLab.*），一张 Table 1 按模型分块。§5.2 照这个写。
- AQLM 和 QuantWAMs 的小节和段落标题直接写结论（"Pareto optimality of AQLM."），标题即论断，正文给证据。§5.4 的 run-in 标题建议写成结论句，示例（等数字出来再定）：
  - *Future fidelity is necessary.*（消融 B 组）
  - *Allocating by future impact matters, not mixed precision per se.*（消融 C 组）

### 5.1 Experimental setup

用 run-in 段落写（*Models.* / *Benchmarks.* / *Baselines.* / *Metrics.* / *Calibration.* / *Implementation.*）。这是 QVGen、DSAQuant、Q-WAM、SteerQuant 的统一做法，比 bullet 省版面。细节都放附录。

- **Models.** Fast-WAM Optional-IDM（LIBERO 2-cam 224 的公开 checkpoint，上游 revision 7faa711），以及第二个 WAM（⏳ ❓2）。每个模型都以自己的 BF16 为参照。
  - **Fast-WAM 的结构**：正文只写一两句。总参数 6B，其中动作专家 1B；先用 10 步去噪得到 future latent，再用 10 步去噪输出 32×7 的动作块，执行前 10 步后重新规划。latent 的形状、双相机布局、gripper 后处理放附录。
  - **联合去噪模型上的 reader**：正文一句，细节放附录。future 与动作 token 在同一个 DiT 中联合去噪，reader 取"从某一去噪步的 future 状态出发，跑完剩余去噪步得到动作"的那段计算。Cosmos-Policy 上已经实现过（joint-DiT suffix reader，cosmos-policy-vqwam-20261024）。
- **Benchmarks.** 仿真用 LIBERO-Plus（4 个 suite × 7 个扰动维度）。
  - 从每个 cell 的 50 个变体中抽 3 个互不重叠的面板，每个面板每 cell 10 个，共 840 个 episode。所有方法用同一批变体和 seed，可以逐 episode 配对。
  - 第二个 WAM 最好也在 LIBERO-Plus 上评测，这样 Table 1 两个分块的列完全一致。这一点支持选 Cosmos-Policy：它有 LIBERO checkpoint，我们之前也在 LIBERO-Plus 上跑过它。
  - 真机设置见 §5.3。
- **Baselines.** GPTQ（标量量化的代表）、GPTVQ、QuIP#、QTIP，各 2 / 3 bpw。所有方法用同一校准集、同一公平口径（接口层 BF16 + 时间条件表）、同一运行时。
  - GPTQ 留在主表里，用来直接支撑"极低比特下标量量化崩溃、VQ 是首选"（2-bit 15/280，3-bit 195/280）。
  - ⏳ GPTQ 目前只有面板 A 的 280。GPTQ W3 要补跑面板 B、C；GPTQ W2 已经崩溃，280 足以说明问题，表注里写明。
- **Metrics.** 闭环成功率（分 suite 和平均）。效率指标见 §5.5。
- **Calibration.** C80（80 个 LIBERO 场景）；λ、β 的取法；更新次数。
- **Implementation.** 4×A100-40GB；量化时间按 GPU·h 报告。面板抽样、seed 规则、T5 缓存、运行时一致性检查都放附录。

### 5.2 Main results（Table 1，对应 Q1）

- **作用**：一张表放全，兑现摘要里的 "across XX WAMs"，同时说明 C1 和 RAVQ 不依赖 Fast-WAM 的分专家结构。
- **Table 1 结构**（按 WAM 分块，参照 Q-WAM、QVGen、SteerQuant 的 Table 1）：
  - 分块：上下两块，Fast-WAM 和第二个 WAM；
  - 行：每块内 BF16 一行，3 bpw 组（GPTQ / GPTVQ / QuIP# / QTIP / **RAVQ**），2 bpw 组（同上）；
  - 列：bpw | 权重（GB）| Spatial | Object | Goal | Long | Avg（按你的意见去掉了胜 / 负和 p 值列）；
  - 权重列直接并进主表（Q-WAM、QuantWAMs 都这样做），延迟放 §5.5；
  - 标记：每个位宽组内最优加粗、次优下划线，RAVQ 行加浅色底；caption 里用一句话定义加粗和下划线；
  - Caption 直接写结论，例如 *At 2 bpw, RAVQ is the only method within X points of BF16 on both WAMs.*

  表的骨架（成功率为 840 个 episode 上的百分比，XX 待填）：

  | WAM | 位宽 | 方法 | bpw | 权重（GB） | Spatial | Object | Goal | Long | Avg |
  |---|---|---|---:|---:|---:|---:|---:|---:|---:|
  | Fast-WAM | 16 | BF16 | 16 | 12 | XX | XX | XX | XX | 77.6 |
  | | 3 | GPTQ | XX | XX | XX | XX | XX | XX | XX |
  | | 3 | GPTVQ | XX | XX | XX | XX | XX | XX | XX |
  | | 3 | QuIP# | XX | XX | XX | XX | XX | XX | XX |
  | | 3 | QTIP | 2.98 | XX | XX | XX | XX | XX | XX |
  | | 3 | **RAVQ** | 2.98 | XX | XX | XX | XX | XX | XX |
  | | 2 | GPTQ | XX | XX | XX | XX | XX | XX | XX |
  | | 2 | GPTVQ | XX | XX | XX | XX | XX | XX | XX |
  | | 2 | QuIP# | XX | XX | XX | XX | XX | XX | XX |
  | | 2 | QTIP | 1.99 | ≈1.5 | XX | XX | XX | XX | XX |
  | | 2 | **RAVQ** | 1.99 | ≈1.5 | XX | XX | XX | XX | XX |
  | WAM-2 | 16 | BF16 | 16 | XX | XX | XX | XX | XX | XX |
  | | 3 / 2 | （同上五个方法 × 两个位宽） | | | | | | | |

- **已有数字**（✅ 26073449，原口径，840）：
  - BF16 652（77.6%）；
  - W3：QTIP 631、GPTVQ 630、QuIP# 626；
  - W2：QTIP 616、QuIP# 609、GPTVQ 602；
  - 各 suite 的数字在面板 C 的 README 里。
  - 主表要的是公平口径下的数，所以除 BF16 外都要重跑（见附 C）。
- **正文写法**：用 run-in 按模型分段（*Results on Fast-WAM.* / *Results on WAM-2.*）。按指南的要求，告诉读者应该从表里看出什么：
  1. 先说 W2：RAVQ 与最强基线的差、配对 p 值、补回了 BF16 差距的百分之几。
  2. 再说 W3：基线到 BF16 只差 21–26 次，RAVQ 的空间有限，如实描述。
  3. 一句点出收益集中在哪些扰动维度。基线在相机视角、背景、传感器噪声上掉得最多，完整的按维度拆分表放附录。
     - ⚠ 如果要说"这些维度更依赖预测的 future"，需要在这些维度上单独算 future 占比，否则不写。
  4. 第二个模型一段，结构相同。
- **结果有不显著的地方**：按指南"承认 + 弱化 + 转回正面"写（例如 W3）。

### 5.3 Real-world deployment（Table 2 + Fig. 4，对应 Q2）

- **作用**：摘要以真机结果收尾，这一节是那句话的出处。
- **Setup**（正文一段，细节放附录）：
  - 平台：AgileX PiPER 机械臂（6 自由度 + 夹爪）。
  - 策略：在 PiPER 的遥操作数据上微调 Fast-WAM。数据用 LeRobot 采集，两路相机（第三人称 + 腕部），与 Fast-WAM LIBERO checkpoint 的双相机布局一致；每个任务约 80 条示教。
  - 量化：在微调后的 BF16 模型上做量化。校准场景取自 PiPER 示教（XX 个），reader 度量在微调后的模型上重新估计。⏳
  - 对比：BF16 / QTIP-W2（最强的 2-bit 基线）/ RAVQ-W2，每个任务每个方法 XX 次试验。
  - 公平性：三种方法用同一组初始摆放（在桌面上做标记），方法的执行顺序随机打乱。
- **Tasks**（正文各一句，写清它考验什么）：
  1. **Two-object pick-and-place**：把一个罐子和一个盒子先后放进篮子。长时程任务，考验量化模型能否完成多步序列。
  2. **Pouring**：把珠子从一个杯子倒进另一个杯子。连续、依赖动力学的动作。
  3. **Cloth folding**：把一块小布对折。可形变物体。⏳ 这一项考验什么，请你补一句，例如"考验对形变的预测在量化后是否还保得住"。
- **成功判据**（放附录，写清楚，避免审稿人质疑）：
  - pick-and-place：两件物体都在篮内并且夹爪已松开；同时记录"第一件放入"的阶段成功。
  - pouring：倒入的珠子占比 ≥ XX%，且没有打翻杯子。
  - folding：对折后两边对齐误差 ≤ XX cm。
  - 每次试验时限 XX 秒。
- **Table 2 骨架**（形式参照 Q-WAM 的 Table 2 和 QuantWAMs 的 Table 6：成功次数 / 试次）：

  | 方法 | bpw | 权重（GB） | Pick-and-place（第一件） | Pick-and-place（两件） | Pouring | Cloth folding | 平均成功率 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | BF16 | 16 | 12 | XX/XX | XX/XX | XX/XX | XX/XX | XX% |
  | QTIP | 2 | ≈1.5 | XX/XX | XX/XX | XX/XX | XX/XX | XX% |
  | **RAVQ（本文）** | 2 | ≈1.5 | XX/XX | XX/XX | XX/XX | XX/XX | XX% |

  - 平均成功率按三个任务的最终成功计算（pick-and-place 取"两件"）。"第一件"一列只用来展示长时程任务在哪一步失败。
  - 如果要报告机载设备上的显存或延迟，就加一列，或放到 §5.5。
- **Fig. 4**：左边是真机系统（PiPER 和两路相机的位置），右边每个任务一行关键帧。最好挑一个 QTIP 失败、RAVQ 成功的例子，说明失败发生在哪一步。
- **❓**：每个任务单独微调一个 checkpoint，还是三个任务合训一个？试验次数定多少？Q-WAM 和 QuantWAMs 都是每任务 10 次，20 次统计上更稳。

### 5.4 Ablation study（Table 3，对应 Q3，你来跑）

建议的行（W2，闭环为主，配对对象为同口径的 QTIP；规模至少 560，最好 840）：

| 组 | 行 | 要证明什么 |
|---|---|---|
| A 逐步搭建 | QTIP（公平口径）→ +② → +②+① +④（完整 RAVQ） | 每一步的增量 |
| B 校准目标（核心） | ④ 用 A only / A + λZ（M = I）/ A + λZ_M / A + λZ_M + βK | **future 保真项是否必要**：这一组是 C1 在方法层面的直接证据 |
| C 比特分配 | 均匀 / 反向（把比特给对 future 影响最小的模块）/ RAVQ（按 future 影响）；更细粒度上随机选层 vs 率失真贪心 | 收益来自"按 future 影响分配"，而不是"任何混合精度都行" |
| D 度量形式 | M = I / 只用对角 / 低秩 + 对角 + ρI | ① 的形式选择 |
| E 位宽曲线 | 1.75 / 2 / 2.5 / 3 bpw（RAVQ vs QTIP） | 收益随比特的变化。可以画成一张小图；版面不够就放附录 |

- 表里只放闭环成功率和配对结果。开环指标要不要放，见 ❓1。
- 表的形式参考 Q-WAM 的 Table 3 和 QuantWAMs 的 Table 3：累积式阶梯，每一行在上一行基础上加一个部件，比特预算固定。
  - A 组就按这个形式；
  - B–D 组是在完整 RAVQ 上替换某一项，放在同一张表的下半部分，用横线分隔。
- 消融正文围绕**设计动机**写（指南）：每组先说"如果 C1 成立，应该看到……"，再说结果。

### 5.5 Efficiency（Table 4，对应 Q4）

- 写法参照 SteerQuant 的 *Inference efficiency and memory footprint*：权重存储、峰值显存、延迟。
- 列：方法 | bpw | 权重（GB）| 峰值显存 | policy call 延迟 p50 | 量化 GPU·h。权重已在 Table 1，这里重点是延迟和峰值显存。
- **已知**：目前所有方法都反量化成 BF16，在同一运行时上跑，所以延迟（0.50–0.52 s）不反映各方法自己的 kernel。BF16 + cache + CUDA Graph 的 policy call 为 541 ms。
- **⏳ 需要**：
  - 用 QTIP 的打包格式与解码 kernel 实测显存和延迟，至少测一个 GPU，最好再加 PiPER 用的机载设备；
  - 核实 K = 1 层的 kernel 支持（官方 kernel 只有 K = 2–4）。
- 这一节决定"端侧显存"的动机能否闭环。版面紧张时，Table 4 可以缩成正文里两三句数字，表放附录。

---

## 6. Conclusion（约 0.2 页）+ Limitations

- **结论三句**：
  1. 极低比特量化让 WAM 丢掉了预测 future 的保真度，动作误差主要由此而来；
  2. 把 future 保真度作为显式的量化目标（按 policy 读 future 的方式衡量、按 future 影响分配比特、以 future 保真度校准），就能在不加比特的情况下补回闭环性能，仿真和真机都成立；
  3. 这说明压缩 WAM 时，预测的 future 不是可以随便牺牲的中间量。
- **Limitations（真实、具体，一次说清）**：
  - ② 的模块级分配依赖逐组的反事实升位测量；在各模块对 future 影响相近的模型上，它退化为纯率失真分配（视第二个 WAM 的结果改写）；
  - reader 度量在 teacher future 处线性化；
  - 只评测了 weight-only，激活仍为 BF16；
  - 真机只有三个任务、每任务 XX 次试验，规模有限。

---

## 7. Supplementary Material（列提纲即可）

顺序参照 SteerQuant：附录 A 先放正文 §3.2 初步分析的完整协议。

- **A. 初步分析（§3.2）的实验细节**：样本（C80 之外 40 场景 × 2 seed）、统一口径、64 个模块组的划分、K 的 soft-bin 参数、三个方法的完整数字、第二个 WAM 上的 future 占比复现。
- **B. LIBERO-Plus 评测细节**：三个面板的构造、seed 规则、配对检验，以及各面板分别的结果。
- **C. 完整结果表**：分 suite × 扰动维度（所有方法、两个位宽）；§5.2 正文只点结论。
- **D. bpw 计数细则**：每类层的位宽、接口层、时间表、解码表。
- **E. 方法细节**：
  - Reader KL 的定义（16 bin，σ = 0.375，按 motor 坐标标准化）以及 λ、β 的取法；
  - ① 的估计细节（探针数、单位迹归一、低秩截断）和谱、稳定性的图；
  - ② 选中的全部视频层列表，以及逐层位宽分配图；
  - ④ 的步长规则和 FP32 折算。
- **F. 真机细节**：数据采集（LeRobot、两路相机、每任务约 80 条示教）、微调设置、量化校准、每个任务的成功判据与时限、初始摆放、逐次试验记录。
- **G. 补充对比**：与 W4A4 方法的参照（同一面板：SpinQuant 203，ViDiT-Q 203，QuaRot 195，均为 280）；GPTQ 的分面板结果。
- **H. 可视化**：更多 policy gaze、预测 future 的定性对比、失败案例。
- **I. 计算开销明细**。

---

## 附 A. Claim–证据对照

| Claim | 位置 | 证据 | 状态 |
|---|---|---|---|
| 端侧部署需要极低比特，而极低比特下只有 VQ 可用 | 摘要 S2，§1 ¶1–¶2，§5.2 | Fast-WAM 6B、BF16 权重 12 GB；GPTQ W2 15/280、W3 195/280 vs VQ W2 198–204、W3 202–205 | ✅ Fast-WAM 论文；A16 基线 26042316 等 |
| 极低比特下动作误差主要来自 future 失真 | 摘要 S3，§1 ¶3，§3.2 (a) | future 占比 73%（W2）/ 85%（W3），公平口径 QTIP | ✅ 26072226 |
| 三个宿主上都成立 | §3.2 (a) | 换 BF16 future 后 motor RMSE 降 24 / 36 / 36%（MSE 占比需统一重算） | ✅ 26050310（口径待统一） |
| future 的失真来自生成它的模块 | §1 ¶3，§3.2 (b) | 视频专家拿走 71% 升位收益；只读 future 的动作专家对 future 的影响为 0 | ✅ 26069856 + 26070509 |
| 动作只读 future 的一小部分 | §3.2 (c) | 有效秩 41.5 / 37,632，第 1 帧 98%，对角稳定 0.86 | ✅ 26073871 |
| 极低比特下现有方法闭环显著掉点 | §1 ¶3，§5.2 | W2 −36…−50 / 840，W3 −21…−26 / 840 | ✅ 26073449 |
| RAVQ 同 bpw 胜过最强基线（多个 WAM） | 摘要 S5–S6，§5.2 | RAVQ 840 vs 公平口径宿主 840 | ⏳ |
| future 保真项必要 | §5.4 B 组 | A-only vs 完整目标的闭环 | ⏳（你来跑） |
| 按 future 影响分配比特是收益来源 | §5.4 C 组 | 反向分配 / 随机选层 | ⏳ |
| 跨 WAM 成立（C1 与 RAVQ） | 摘要 S5–S6，§5.2，附录 A | Table 1 第二个分块 + 附录 A 的 future 占比复现 | ⏳ ❓2 |
| 省显存、可部署 | §1 ¶1，§5.5 | 打包格式下的显存和延迟 | ⏳ |
| 真机成立 | 摘要 S7，§5.3 | AgileX PiPER 三个任务的成功次数 / 试次（Table 2） | ⏳ |

## 附 B. 图表清单与版面预算（CVPR 正文 8 页）

| 编号 | 内容 | 位置 | 大小 | 状态 |
|---|---|---|---|---|
| Fig. 1 | Teaser：两条误差路径 + 成功率–bpw | §1 | 双栏宽，约 1/3 页 | ⏳ 需 RAVQ 数字 |
| Fig. 2 | 三联图：(a) future 占比（三方法 × 两口径 × 两位宽）；(b) 逐组升位收益（A / Z / K 堆叠）；(c) policy gaze 热力图 | §3.2 | 双栏宽 | (a)(b) ✅ 数据有，待画；(c) ⏳ 待导出 |
| Fig. 3 | RAVQ 主图 | §4 | 双栏宽，约 1/3 页 | 待画 |
| Fig. 4 | 真机系统与三个任务的关键帧 | §5.3 | 单栏或双栏 | ⏳ |
| （可选）Fig. 5 | 位宽曲线（消融 E 组） | §5.4 | 单栏 | ⏳；版面不够放附录 |
| Table 1 | 主表，按 WAM 分块 | §5.2 | 双栏宽 | 部分 ✅ |
| Table 2 | 真机 | §5.3 | 单栏 | ⏳ |
| Table 3 | 消融 | §5.4 | 单栏 | ⏳ |
| Table 4 | 效率 | §5.5 | 单栏 | ⏳；可缩成正文数字 |
| Alg. 1 | RAVQ 流程 | §4 | 单栏 | 待写 |

逐层位宽分配图、更多 policy gaze、预测 future 的定性对比、按扰动维度的完整表都在附录（§7 C、E、H），不占正文版面。

页数分配：

| 部分 | 页数 |
|---|---:|
| 摘要 + 引言 | 1.25 |
| 相关工作 | 0.6 |
| 预备知识与动机 | 1.5 |
| 方法 | 1.5 |
| 实验 | 2.5 |
| 结论 | 0.2 |
| 合计 | 约 7.55 |

原来预备知识（0.5）和 C1 分析（1.25）两节合计 1.75 页，合并后约 1.5 页，正文省约 0.25 页。原 Fig. 2 和 Fig. 3 合并成一张三联图。第四版又删掉了 Analysis 小节，原来的位宽分配图和定性对比图移入附录，跨 WAM 结果并进 Table 1，所以正文的图从 8 张减到 4–5 张。

实验节 2.5 页的大致分配：设置 0.4，主结果（Table 1 双栏）0.8，真机（Table 2 + Fig. 4）0.6，消融 0.45，效率 0.25。

## 附 C. 写论文需要、目前还没有的实验

按对论文的必要性排序（第二版：摘要点名了跨 WAM 和真机，这两项上调）：
1. **RAVQ W2 / W3 的 840 闭环**（完整版，含 ④）。未校准版 RAVQU 的 840 在跑（fwr840u 26075585）。
2. **三个宿主在公平口径下的 840 闭环**（GPTVQ / QuIP# / QTIP × W2 / W3，加接口 BF16 + 时间表）。没有它，主表的比较不公平。
3. **真机（AgileX PiPER）**：三个任务（两物体 pick-and-place、倒珠子、叠布），每任务约 80 条 LeRobot 示教（两路相机）→ 微调 BF16 的 Fast-WAM → 用 PiPER 示教做量化校准 → BF16 / QTIP-W2 / RAVQ-W2 各 XX 次试验。前置链条最长，要最早启动。
4. **第二个 WAM**：先在它上面复现 future 占比（C1，进附录 A）和反事实升位（决定 ② 怎么分），再跑 Table 1 的第二个分块。
5. **消融 B 组**（A-only vs 完整目标）：论证 C1 在方法层面的关键证据（你来跑）。
6. **效率实测**：打包格式下的显存、延迟；核实 K = 1 的 kernel。
7. **GPTQ W3 补到 840**（面板 B、C），让主表里"标量量化 vs VQ"的对比和其他行同一规模。
8. 画图用的导出：M 对角的热力图、预测帧解码。
9. 可选的闭环版 C1 证据：BF16 视频专家 + W2 动作专家 vs W2 视频专家 + BF16 动作专家，840 闭环。开环的 73% 用闭环再印证一次，C1 会更硬，也能回避 ❓1 的顾虑。

## 附 D. 需要你决定的事项

- **❓1 开环测量进不进正文。** C1 的三条证据（§3.2 (a)–(c)）都是 held-out 场景上的开环测量。我按"它们是现象分析，不是方法性能的代理"来处理，放进正文；方法的所有性能结论只用闭环。如果你的意思是任何 held-out MSE 都不进论文，§3.2 需要改成以附 C 第 9 项的闭环版证据为主。
- **❓2 第二个 WAM 选哪个。** 现在 Table 1 按 WAM 分块，两个模型最好在同一个基准（LIBERO-Plus）上，这又多了一个选 Cosmos-Policy 的理由。 叙事上已经按你的意见把 ② 写成一般形式（按各模块对 future 保真度的实测影响分配），不再绑定"动作专家 → 视频专家"，所以共享骨干的 WAM 也能用。剩下的是选哪个模型：Cosmos-Policy（已有 reader 实现，耦合方式与 Fast-WAM 最不同，最能说明一般性）还是 LingBot-VA（QuantWAMs 也用了它，审稿人熟悉）。
- **❓3 题目。** §题目 的三个候选（第二版已按"保住 future"的叙事重拟）。
- **❓4 Fig. 1 用方案 A（路径 + 曲线）还是方案 B（具体样本）。**
- **❓7 真机：每个任务单独微调还是合训一个 checkpoint；每任务试验次数（10 或 20）；叠布任务考验什么（一句话）。**
- **❓5 W3 在论文里的分量。** W3 宿主到 BF16 只差 21–26 次，840 上显著胜出宿主约等于追平 BF16。建议主说法放在 W2，W3 作为完整性结果。
- **❓6 要不要和 Q-WAM / QuantWAMs 对比。** 两者都是 W4A4，与我们的 weight-only 2–3 bpw 不是同一设定。我的建议：正文在相关工作里说明设定不同；附录给一行同面板的 W4A4 参照（SpinQuant / ViDiT-Q / QuaRot 已有 280 的结果）。审稿人看到两条极低比特路线放在一起，就不会追问。
- **审稿预期提醒（来自 paper 库）**：Q-WAM 在 3 个 WAM 和 5 个真机任务上评测，QuantWAMs 在 Fast-WAM 和 LingBot-VA 两个 WAM、RoboTwin 和 LIBERO 两个基准以及真机上评测。审稿人很可能拿这个规模来比。所以 ❓2（第二个 WAM）和真机实验的优先级很高；只有一个模型、一个基准，在这个方向上容易被认为证据不足。

## 附 E. 写作风格参考（paper 库）

提取了 10 篇论文的章节结构：QTIP、QuIP#、GPTVQ、GuidedQuant、AQLM（LLM 极低比特量化），Q-WAM、QuantWAMs（WAM 量化），Fast-WAM（我们的主模型），QVGen、DSAQuant（视频生成量化）。

### E.1 共同结构（我们的大纲已经对齐）

| 部分 | 10 篇论文的共同做法 | 本大纲 |
|---|---|---|
| 摘要 | 5–13 句；背景 → 问题 → 方法 → 组件 → 结果 | 7 句（第二版） |
| 引言 | 4–6 段；背景 → 痛点 → 已有局限 → 本文思路 → 结果概述 | 5 段 + 3 条贡献 |
| 贡献 | WAM 方向的论文基本都是 3 条 bullet | 3 条 bullet |
| 方法 | 先 Preliminaries / 问题形式化，再按组件或阶段分节 | §3 预备知识与动机（定义 + 观察）+ §4 四个部件 |
| 实验 | 主结果 → 消融 → 效率（WAM 论文再加真机）；细节用 run-in 小标题；不设独立的 Analysis | §5.2 主结果 → §5.3 真机 → §5.4 消融 → §5.5 效率 |
| Teaser | 多数放在第 1 页 | Fig. 1 |

### E.2 具体借鉴了什么、用在哪里

| 写法 | 出处 | 用在本大纲的位置 |
|---|---|---|
| 诊断先行，方法组件与诊断一一对应；第一条贡献写"发现" | DSAQuant | §3.2 整体结构；贡献 C1 |
| "Preliminary and Motivation" 合成一节：3.1 只放定义、结尾一句引出观察；3.2 先写协议段，再写加粗结论段并共用一张多子图，每段落在"所以保护应该……" | SteerQuant | 新 §3（原 Preliminaries 与 C1 分析合并） |
| 主结果一节、一张 Table 1 按模型分块，正文用 run-in 按模型分段；实验节不设独立的 Analysis | SteerQuant（及其余 10 篇） | §5.2；§5 去掉 Analysis |
| 观察 → 原则 → 方法（"Based on this principle, we propose"） | QuantWAMs | 引言 ¶3 → ¶4 的衔接；§3.2 每段末句和收束句 |
| 设问式开头（"A natural first question is whether …"） | Q-WAM | 引言 ¶3 的备选开头；题目候选 3 |
| 摘要结尾一句 takeaway，把结果提升为一般原则 | DSAQuant | 摘要可选 S6 |
| 摘要结果句给具体区间和幅度 | Q-WAM、QVGen | 摘要 S5 |
| Teaser 放第 1 页，caption 两句直接写结论 | QuIP# | Fig. 1 |
| 第 1 页 teaser 表，成对列出"基线 → 基线 + 本文" | GuidedQuant | Fig. 1 方案 C |
| 方法节内用 run-in 小标题分层 | Q-WAM、Fast-WAM | §4 各小节 |
| 部件名统一成"对象 + 动作"的结构，方法与消融对应 | QuantWAMs | §4 小节标题 |
| 方法节内嵌一张小验证表 | QTIP | §4.2 位宽分配表 |
| 每个设计选择配一张证据图 | QVGen | §3.2 三个结论段共用 Fig. 2 的三个子图 |
| 实验小节按"问题"组织 | QuIP# | §5 开头的 Q1–Q5 |
| 小节或段落标题即结论（"Pareto optimality of AQLM."） | AQLM、QuantWAMs | §3.2 的结论段；§5.4 的 run-in |
| 实验设置用 run-in（Models. / Training. / Evaluation.） | QVGen、DSAQuant、Q-WAM | §5.1 |
| BPW、显存列并入主表 | Q-WAM、QuantWAMs | Table 1 |
| 表注里定义加粗、下划线的含义 | GPTVQ、QVGen | Table 1 |
| 累积式消融阶梯，比特预算固定 | Q-WAM、QuantWAMs | Table 3 的 A 组 |
| 真机表记"成功次数 / 试次"，配系统与任务图 | Q-WAM、QuantWAMs | §5.3 |
| 结论用 "suggest that" 一类的措辞，与证据范围一致 | Fast-WAM | §6 |

### E.3 有意没有照搬的

- **Fast-WAM 的"受控变体放在同一张主表的分组行里"**：我们的消融（Table 3）规模较大，放进主表会挤掉 suite 列，所以单独成表。
- **GuidedQuant 的贡献写成 run-in 段落**：它适合贡献之间互相依赖的情况。我们的三条贡献（发现 / 方法 / 评测）界线清楚，用 bullet 更醒目。

完整提取笔记（每篇的摘要逐句功能、章节树、引言逐段功能、图表清单）在 `C:\Users\Raymond\AppData\Local\Temp\claude\D--Downloads-Final-Year-Project\8c18cf31-2f4a-49f5-b832-7af084307091\scratchpad\style\STYLE_NOTES.md`。这是临时目录。如果你想长期保留，我可以把它移到 `research/writing-guide/` 下。
