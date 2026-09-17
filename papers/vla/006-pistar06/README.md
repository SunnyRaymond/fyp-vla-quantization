# π*₀.₆: a VLA That Learns From Experience

> **Reading-list role**: Core / Series — deployment experience and RL post-training  
> **Verification**: `verified-full-text`; official technical report / preprint  
> **Recommended effort**: Core read

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Physical Intelligence; Ali Amin et al. Complete canonical corporate/individual list is on the arXiv metadata and official PDF title page |
| Year / version | 2025; arXiv 2511.14759, latest checked v2 (2025-11-19); official release 2025-11-17 |
| Venue / status | arXiv technical report / preprint; no peer-reviewed venue verified as of 2026-08-22 |
| Primary source | [arXiv:2511.14759](https://arxiv.org/abs/2511.14759) · [DOI](https://doi.org/10.48550/arXiv.2511.14759) · [PI official PDF](https://www.pi.website/download/pistar06.pdf) |
| Code / project | [PI official release](https://www.pi.website/blog/pistar06) |

**Identity caveat — 汇报时必须先讲。** ReadingList 写的是 bare `pi_0.6`，但没有找到 canonical title 恰为“π₀.₆”的独立 paper。官方最合理映射是本篇 `π*₀.₆`：paper 定义 `π₀.₆` base VLA，再用 RECAP 得到带星号的 RL-trained policy。另一个 [π₀.₆-MEM / MEM paper](../007-mem/README.md) 是 2026 年 memory extension，现已作为独立 companion reading 准备；它不与本篇合并，也不计作本条。

## 2. One-sentence takeaway

π*₀.₆ 用 RECAP 把 autonomous rollouts、human corrections 与 distributional value estimate 转成 positive-advantage-conditioned VLA training，在若干真实 long-horizon tasks 上将 throughput 提高约 2×并把 strict laundry success 推到 97%。

## 3. Background and prerequisites

- **Technical lineage**：π₀ flow controller → π₀.₅ open-world hierarchy → π₀.₆ stronger base model → π*₀.₆ reinforcement-learning post-training。
- **读前知识**：behavior cloning、offline RL、advantage、Monte-Carlo return、distributional value function、AWR、PPO、human correction、batch policy iteration。
- **关键术语**：`RECAP`、advantage conditioning、positive/negative condition、autonomous rollout、throughput、failure rate。

## 4. Problem

- **Target setting**：让已有 generalist VLA 从真实 deployment experience 持续改进 long-horizon/contact-rich tasks。
- **Bottleneck**：behavior cloning 只看到 expert actions，较少学习失败状态、recovery 与哪些 trajectory 更优；从头 online RL 又昂贵且危险。
- **Why previous methods are insufficient**：普通 SFT 不能有效利用 success/failure label；task-specific RL 又没有 generalist VLA prior，sample efficiency 和跨 task reuse 较差。

## 5. Method

### 5.1 System view

`π₀.₆ base policy` → autonomous rollouts + optional teleoperation corrections → success/reward + Monte-Carlo returns → multi-task distributional value model → estimate advantage → label actions as positive/negative → condition VLA with text `Advantage: positive/negative` → deploy positive-conditioned `π*₀.₆` → repeat batch iteration。

### 5.2 Core mechanism

- **Base model**：Gemma 3 4B + 约 860M action expert；continuous flow actions + FAST discrete tokens；`Knowledge Insulation` 用 stop-gradient 减少 action learning 对 pretrained knowledge 的破坏；输出 high-level subtask 与 50 Hz chunks。
- **Value learning**：用 Monte Carlo return 训练 `pφ(V|o,l)`；value 离散为 `B=201` bins。pretraining 用 full-episode estimate `A=R−V(o)`；post-training 用 `N=50` 的 `n-step return + V(o_{t+N}) − V(o_t)`。
- **Policy learning**：用 indicator `I(A>ε)` 构造 positive/negative condition，并用 natural-language input 表示 advantage；deployment 通常固定 positive condition。
- human corrections 强制视为 positive；training 中 advantage-condition dropout 30%。
- threshold：pretraining 约 top 30% demonstrations；多数 task fine-tuning 约 top 40% rollouts，T-shirt-and-shorts laundry 约 top 10%。
- inference 是 greedy positive-conditioned policy，不在运行时做 online value search；训练是 batch offline iterations。

### 5.3 PPO、DPO 与 REINFORCE：不要把三者当成同一类算法

| Method | Core update | Data requirement | Value/reward model | 与本文的关系 |
|---|---|---|---|---|
| REINFORCE | 直接估计 `∇θJ ≈ R·∇θ log πθ(a|o)`；高-return actions 提高 probability，低-return actions 降低 probability | 通常需要 current policy 的 on-policy trajectories | 不一定需要 value function，但经常用 baseline/value 降低 gradient variance | Related Work 引用的 prior VLA method；RECAP 刻意避免这种 high-variance on-policy policy gradient |
| PPO | 用 probability ratio `r=πθ/πold` 乘 advantage，并用 clipping/trust region 限制每次 policy update | 通常交替进行 fresh rollout 与若干 optimization epochs | 通常使用 critic/value function 计算 advantage | paper 内 comparison；对 flow head 的 likelihood/trust region 难处理，且 real robots 无法每几步 gradient update 就重新收集 data |
| DPO | 从 `(preferred, rejected)` pair 直接优化 policy 相对 reference policy 的 log-odds | fixed preference pairs；training 本身不需要 environment rollout | 不显式训练 reward model/value function | 只在 Related Work 中作为 prior VLA approach；不是 paper 的 experimental baseline，也不是 RECAP 本身 |

三者的最短记忆法：`REINFORCE = raw policy gradient`；`PPO = constrained/clipped policy gradient`；`DPO = preference-pair classification-like optimization`。PPO 与 REINFORCE 都需要把 policy 产生 action 的 likelihood 放进 gradient estimator；这对 flow-matching VLA 不方便。RECAP 则把 RL problem 转为 advantage-labeled supervised learning，避免直接计算完整 continuous-action policy gradient。

### 5.4 Section III 的 regularized RL 与 KL divergence

普通 RL 只最大化 expected return：

`maxπ J(π) = Eτ~π[Σt r_t]`。

regularized RL 还要求 learned policy 不要离 reference/behavior policy `πref` 太远：

`maxπ E[Σt r_t] − β E_o[D_KL(π(·|o) || πref(·|o))]`。

`KL divergence` 定义为：

`D_KL(P||Q) = E_{x~P}[log(P(x)/Q(x))]`。

它衡量：从 `P` 采样时，`P` 与 `Q` 给同一 event 的 probability 有多不一致。它恒为非负、两者完全相同时为零，但不对称，因此不是 mathematical distance。如果新 policy 给一个 reference policy 几乎从不选择的 action 很高 probability，`log(π/πref)` 会很大，受到强 penalty。

这个 regularization 在本文 setting 中有四个好处：

1. **Stay inside data support**：offline dataset 没有覆盖的 actions 无法可靠评估；靠近 behavior policy 可减少 out-of-distribution exploitation。
2. **Stable repeated optimization**：在同一批 robot data 上训练很多 gradient steps 时，防止 policy 一次跳得太远。
3. **Preserve pretrained competence**：reward/success label 很稀疏，强行只追 reward 容易破坏原有 dexterity、language following 与 safe motion prior。
4. **Controlled improvement**：regularized optimum 具有 `π̂(a|o) ∝ πref(a|o) exp(A(o,a)/β)` 的形式，即提高 high-advantage actions 的 probability，同时仍由 `πref` 提供 support。

在这个 standard KL objective 中，较大的 KL coefficient `β` 表示更强 regularization、更保守的 update。注意本文后续 Eq. 2/CFG 又用 exponent `β` 表示 sharpening strength；那里较大的 `β` 会强化 positive-conditioned distribution。两处来自相关 derivation，但 operational direction 不应简单混为同一个 knob。RECAP 默认主要依靠 task-specific threshold `ε_l`，而不是用很高 CFG weight；authors 指出过高 guidance 可能产生 aggressive actions。

RECAP 最终并没有直接把一个 explicit KL term 加到 flow-matching policy loss。它通过在 behavior dataset 上做 conditional maximum likelihood，学习 `πref(a|I,o,l)`，形成 **implicit behavior regularization**：只重新分配 dataset actions 的 probability，而不是凭空优化 dataset 外的 actions。这也是它能绕开 flow-policy exact log-likelihood 问题的原因。

### 5.5 Offline RL 与 online RL 在这里具体指什么

- **Offline RL**：optimization 时只使用一个已经收集好的 fixed dataset；policy 不能边训练边向 environment 查询新 transitions。dataset 可由 humans、旧 policies 或其他 behavior policies 产生，所以通常是 off-policy。
- **Online RL**：current policy 与 environment interaction，持续获得 fresh trajectories，并用这些 data 更新 policy；经典 PPO/REINFORCE 通常要求较新的 on-policy samples。
- **On-policy/off-policy 与 online/offline 不是同一维度**：一次 rollout 可以由 current policy 收集，因此 collection 时是 on-policy；但如果把它放入 replay dataset，之后与 demonstrations、旧-policy data 一起训练，optimization 就是 mixed off-policy/offline。

RECAP 最准确的描述是 **iterated batch offline RL**：

`deploy π^{k−1} → collect one batch → stop collection → train V^k offline → train π^k offline → redeploy`。

所以它有 real-world online data acquisition，但不是 concurrent online PPO。paper 也明确说不能每几个 gradient steps 就向 real robots 收一批 fresh data。

### 5.6 Value function 怎样训练；它是否也会 fine-tune？

先看 training target。episode 只有 success/failure label，authors 把它变成：

```text
r_t = -1       for every non-terminal step
r_T = 0        if the episode succeeds
r_T = -C_fail  if the episode fails
```

因此 successful trajectory 中的 return roughly 等于“距离成功还剩多少 steps”的负数；越接近成功越接近 0。failed episode 得到很大的 negative target。不同 task 按 maximum episode length normalize 到 `(-1,0)`。

value model 是独立的 smaller model：`Gemma 3` initialized 670M VLM backbone + value head，输入 observation `o_t` 与 language/task metadata `l`，输出 201-bin distribution `pφ(V|o_t,l)`；authors 还混入少量 multimodal web data co-training 以减少 overfitting。training procedure 是：

1. 从完整 trajectory 计算 empirical Monte Carlo return `R_t`。
2. 把 `R_t` discretize 到 201 bins。
3. 对正确 return bin 做 cross-entropy training。
4. inference 时用 distribution expectation `Σ_b pφ(b|o,l)v(b)` 还原 scalar `V(o,l)`。
5. 用 return/value difference 估计 advantage，再 threshold 成 `Advantage: positive/negative`。

你关于 fine-tuning 的理解需要分成三个阶段：

| Stage | Value function 发生什么 | Policy 发生什么 |
|---|---|---|
| RECAP pretraining | 先从 smaller pretrained VLM 训练 `Vpre`，使用整个 multi-task demonstration dataset | 再从 VLA base 训练 `πpre`；`Vpre` on-the-fly 产生 advantage labels |
| Target-task SFT / Iteration 0 | 从 `Vpre` fine-tune 得到 `V_l^0` | 从 `πpre` fine-tune `π_l^0`；demonstrations 的 indicator 固定 positive |
| Iteration `k` | 将所有 accumulated target-task data 合并，从 `Vpre` 重新 fine-tune `V_l^k` | 同样从 `πpre` 重新 fine-tune `π_l^k`，而不是从上一轮 `π_l^{k−1}` 接着训练 |

所以答案是：**value function 会在 broad pretraining dataset 上训练，也会在每个 downstream iteration 用 task data fine-tune。** 但它不是 VLA 内共享的一颗 value head；它是 separate model。Algorithm 1 把 `train V` 与 `train π` 写成 sequential subroutines，policy loss 没有定义一条 gradient path 回 value model。更严谨地说，paper 没有逐字写 “freeze V during policy training”，但 method/objectives 表明它在 policy step 中作为 label-producing critic 使用，而不是由 VLA loss jointly update。

还有一个不寻常但重要的 design：每一轮 `V_l^k` 和 `π_l^k` 都分别从 `Vpre`、`πpre` 重启 fine-tuning。authors 报告这样能减少 iterative drift；代价是不能直接累积上一轮 model parameters，只累积 dataset。

### 5.7 What is actually new

真正的新意是 `distributional value → advantage label → language-conditioned VLA` 的 RECAP recipe，能把 autonomous failure data 和 corrections 接到 generalist VLA post-training。Gemma、flow matching、Monte-Carlo return、AWR/PPO 都是已有 component。

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| RECAP 提高 throughput、降低 failures | authors state diverse laundry 与 espresso throughput 比 offline-RL+SFT **more than doubles**；failure rate roughly halves；除 diverse laundry 外 tasks 超过 **90% success**。Plot-read throughput roughly simple laundry 60、diverse laundry 8.4、espresso 29、box 13.3 successes/hour | Figs. 7–8 + Sec. 5.2, PDF p. 9 | exact text 与 plot-read 数字必须分开；within-system comparison，不是跨论文 speedup |
| Batch iterations 持续改善 | laundry：2 iterations，每次在 4 robots 收集 **300 trajectories**，overall throughput **+50%**；box：每 iteration **600 autonomous + 360 corrections**，iteration 2 后约 **2× throughput** | Figs. 9–10 + Sec. 5.3, PDF p. 10 | data volume 与 RECAP mechanism 没有完全解耦 |
| 针对特定 failure mode 可达高 success | strict one-shirt laundry：2 iterations、每次 **600 trajectories**，最终 **97% success** | Fig. 12 + Sec. 5.4, PDF p. 11 | narrow task/failure definition；不是 general capability 的 97% |
| RECAP 优于 paper 内 AWR/PPO variants | RECAP curve 最强；PPO trust-region coefficient `η=0.01` | Fig. 11, PDF p. 10 | 主要为 comparative trend；没有 external replication |

## 7. Limitations

### Authors' stated limitations

- 并非 fully autonomous：仍需 human success labels、intervention/correction 与 environment resets。
- exploration 基本是 greedy，可能无法摆脱局部最优。
- 训练是 batch offline iterations，不是 concurrent online update。
- real-robot RL 的 sample collection 与 operations 仍昂贵。（Sec. 7, PDF p. 11）

### My critique

- **Internal validity**：更多 rollouts、advantage filtering、condition token、corrections 同时变化，RECAP 的独立 causal contribution 难完全隔离。
- **External validity**：主要是 PI 自定义 household/industrial tasks，且 success judgement 有人工环节。
- **Systems validity**：throughput 同时受 policy speed、task duration、reset/logistics 影响，不等于 neural-network inference throughput。
- **Reproducibility**：base model/data 不公开；threshold、CFG/correction labels 引入多个 tuning degrees of freedom；仍是 preprint。

## 8. Why it matters for this project

- 它把 VLA learning loop 从 demonstrations 扩展到 deployment experience，说明实际部署速度会反过来决定能收集多少 RL data。
- 对 compression/edge deployment，这是一个 data-flywheel 问题：更低 latency 不只改善 control，也可能提升 experience collection throughput。
- 它是讨论 post-training、failure recovery 与 value-guided data selection 的核心 paper，但不能把 proprietary within-lab success 当成通用 benchmark 结论。

## 9. How to read it

### 20-minute route

1. 先读本页 Identity caveat，分清 π₀.₆、π*₀.₆ 与 π₀.₆-MEM。
2. Figure 2/RECAP overview + Sec. 4（PDF pp. 5–7）：追 rollout → value → advantage condition。
3. Fig. 9/10（p.10）或 Fig. 12（p.11）：核对 iteration data volume 与 success/throughput。
4. Sec. 7（p.11）：读 human-in-loop、greedy exploration、batch training。

### 60-90-minute route

1. 复习 advantage `A=Q−V`、Monte-Carlo return、distributional classification。
2. 画 RECAP loop，标出每一步需要 human input 的位置。
3. 解释为什么 `B=201` value bins 与 binary advantage condition 可以同时成立。
4. 对照 Figs. 7–12，分清 throughput、success、failure rate 与 task-specific substage metric。
5. 比较 RECAP/AWR/PPO：哪些 optimizer/constraint/data 条件 matched，哪些没有？
6. 写一个 counterfactual：只加同量 rollout、做 SFT 而不 advantage conditioning，预期会怎样？

## 10. Reading questions

1. binary positive/negative condition 丢失多少 advantage magnitude information？
2. value calibration error 会怎样污染 positive subset？
3. human corrections 强制 positive 是否引入 intervention bias？
4. gain 中多少来自更多 data，多少来自 RECAP objective？
5. greedy exploration 对 rare-but-useful recovery trajectory 有什么影响？
6. compressed/quantized policy 会怎样改变 value estimate、rollout distribution 与 experience flywheel？

### 10.1 My answers

#### 1. Binary condition 丢失多少 advantage magnitude information？

会丢掉 threshold 两侧的全部内部 ranking：刚超过 `ε` 的 action 与极高-advantage action 都是 positive；轻微错误与 catastrophic action 都可能是 negative。它也无法表达 value uncertainty。收益是 output 不需要改变，flow action expert 只多接收一个 text condition，而且所有 positive/negative data 都能保留。task-specific `ε` 与 CFG 只能调 coarse selection/sharpness，不能恢复被丢掉的 magnitude。值得测试的 extension 是 multi-bin advantage tokens 或 uncertainty-aware condition。

#### 2. Value calibration error 怎样污染 positive subset？

overestimate 会产生 false positive，让坏 action 被 positive-conditioned policy imitation；underestimate 会产生 false negative，压低 useful recovery action。若 error 在某个 task/state region 系统性偏移，per-task percentile threshold 只能控制 positive 比例，不能修正 ranking。distributional output 提供了 uncertainty information，但 RECAP 最终主要取 expectation 后 threshold，没有显式利用 uncertainty。应报告 calibration、ranking accuracy，以及 threshold 附近 label flip rate。

#### 3. Human corrections 强制 positive 是否引入 intervention bias？

会。corrections 集中在 policy 即将失败的 unusual states，operator timing/style 也不是随机样本；“human action 一定优于 autonomous action”只是 working assumption。强制 positive 的好处是把 rare recovery transitions 注入 data support、帮助 exploration；风险是错误/迟到的 intervention 也被无条件提升。应加入 correction quality labels、operator-stratified analysis，或用 value estimate 与 human-positive label 的 disagreement 做 audit。

#### 4. Gain 中多少来自更多 data，多少来自 RECAP objective？

论文不能完全分离。`offline RL + SFT → final RECAP` 同时增加 autonomous/correction data 并改变 training labels。AWR/PPO comparisons 使用同一批 on-robot data，能部分支持 advantage-conditioned extraction 优于这些 alternatives；但缺少一个最干净的 matched baseline：使用完全相同 accumulated data、相同 steps、仅做 unconditional SFT。因而“RECAP recipe 有贡献”有 evidence，“约 2× gain 全由 objective 导致”没有被证明。

#### 5. Greedy exploration 怎样影响 rare recovery trajectories？

greedy positive-conditioned policy 主要重复当前 high-probability behaviors；如果成功 recovery 从未出现在 demonstrations/rollouts 中，behavior regularization 反而会阻止跳到该 action region。human corrections 在这里相当于 targeted exploration，但仍依赖 operator 看见并及时介入。更强方案可以是 uncertainty-triggered intervention、safe action perturbation，或保守的 ensemble exploration。

#### 6. Compression/quantization 怎样改变 experience flywheel？

至少有三条路径：policy numerical error 改变 closed-loop state distribution，使原 value model miscalibrated；latency 改变 successes/hour，从而改变单位时间可收集的 RL data；threshold 附近的小 value/action perturbation 会翻转 positive/negative labels。最稳妥的 protocol 是先 deploy compressed policy 收集 calibration rollouts，再 fine-tune/recalibrate value function，并分别报告 policy-only compression、value-only compression 与 both-compressed 的 success、value calibration、P99 latency、power 和 data-collection throughput。

## 11. Weekly meeting card

- **Problem**：怎样让 generalist VLA 从真实 rollout 的成功与失败中继续学习？
- **Key idea**：distributional value 估计 advantage，再用 `Advantage: positive/negative` condition 训练/部署 VLA。
- **Best evidence**：strict one-shirt laundry 经两轮、每轮 600 trajectories 后达 97% success（Fig.12, p.11）；box 两轮后约 2× throughput（Figs.9–10, p.10）。
- **Biggest limitation**：human labels/resets/corrections 仍不可缺，base model/data proprietary，且为 preprint。
- **Question for the group**：若严格控制新增数据量，advantage conditioning 还能贡献多少？

## 12. Evidence boundary

- **Source claim**：base model/RECAP/objectives 来自 Sec. 3–4；数字来自 Figs. 7–12 与相邻正文。
- **My interpretation**：π*₀.₆ 是“VLA deployment experience post-training layer”，不是替代 π₀ pretraining。
- **Open question**：RECAP 相对等量-data SFT 的纯 gain、value-model calibration 与跨 task transfer 尚未充分确认。
- **Primary links**：[arXiv](https://arxiv.org/abs/2511.14759) · [Official PDF](https://www.pi.website/download/pistar06.pdf) · [PI release](https://www.pi.website/blog/pistar06)

