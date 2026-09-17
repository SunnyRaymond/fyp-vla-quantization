# 原总结文献对应与版本纠正

核验日期：2026-09-05 · [返回总览](README.md)

核对了 [VLA_idea.pdf](../../../idea/VLA_idea.pdf) 的 **31 个相关文献条目**。其中 24 项进入 numbered reading route（BarrierNet 用明确标注的前身 author-linked preprint）；另补 6 项，形成 30 个条目。OAT 同目录附一份新扩展稿，所以共 31 份 PDF。余下 7 项保留为扩展入口，未声称已下载或完成全文阅读。

## 方向一的 13 个原条目

| 原条目 | 本次处理 | 核验与阅读理由 |
|---|---|---|
| FAST | [本地](01-control-aware-action-tokenization/01-fast/README.md) | DCT + BPE 基础 |
| OAT | [本地，含扩展稿](01-control-aware-action-tokenization/03-oat/README.md) | 指定稿 2602.04215v2；新增 2607.21670v1，两个 arXiv records 分开 |
| FASTer | [本地](01-control-aware-action-tokenization/05-faster/README.md) | 完整标题是 *Toward Efficient Autoregressive Vision Language Action Modeling via Neural Action Tokenization* |
| X-Tokenizer | [本地](01-control-aware-action-tokenization/07-x-tokenizer/README.md) | semantic interface |
| ActionCodec | [本地](01-control-aware-action-tokenization/04-actioncodec/README.md) | VLA optimization-aware design |
| SA-VLA | [本地](01-control-aware-action-tokenization/08-sa-vla/README.md) | state-conditioned decoder |
| OmniSAT | [arXiv:2510.09667](https://arxiv.org/abs/2510.09667) | 扩展：B-spline + multi-stage residual quantization；第一轮用 BEAST / NAC 分别学习两个基础机制 |
| NAC | [本地](01-control-aware-action-tokenization/06-nac/README.md) | neural codec 路线 |
| BEAST | [本地 NeurIPS 2025 final](01-control-aware-action-tokenization/10-beast/README.md) | 原 OpenReview link 可检索到对应稿；使用正式 proceedings，完整标题加 *Action Sequences for Imitation Learning* |
| VQ-VLA | [本地](01-control-aware-action-tokenization/02-vq-vla/README.md) | ICCV 2025 work；实际标题为 *Improving … via Scaling …* |
| Wall-OSS-0.5 | [arXiv:2605.30877](https://arxiv.org/abs/2605.30877) | 扩展：完整 VLA system / pretraining report，含 vision-aligned action interface；第一轮以 X-Tokenizer 拆解语义对齐 |
| FlashVLA / Think Twice, Act Once | [本地](01-control-aware-action-tokenization/09-flashvla/README.md) | 2505.21200，不是主库 2608.27384 的 streaming FlashVLA |
| Bridging the Semantic-Action Gap… | [arXiv:2511.16449](https://arxiv.org/abs/2511.16449) | 扩展：VLA-Pruner，关注 visual-token pruning，区别于 action-token code budget |

## 方向二的 7 个原条目

| 原条目 | 本次处理 | 核验与阅读理由 |
|---|---|---|
| ActiveVLA | [本地](02-active-probing-and-dual-control/04-activevla/README.md) | active view / 3D zoom-in 对照 |
| Predictive Visuo-Tactile… | [本地](02-active-probing-and-dual-control/03-predictive-visuo-tactile/README.md) | learned interaction model + information gain |
| Push to know! | [本地](02-active-probing-and-dual-control/02-push-to-know/README.md) | IROS 2023 active physical parameter inference |
| PI-VLA | [本地 publisher final](02-active-probing-and-dual-control/06-pi-vla/README.md) | AURD 应按 replanning mechanism 阅读，不能由名称推断 physical probing |
| Perturbation-Based Uncertainty… | [本地](02-active-probing-and-dual-control/07-perturbation-failure-detection/README.md) | 正式标题含 *Epistemic*；failure detection 不是 probing planner |
| PhysReflect-VLA | [本地](02-active-probing-and-dual-control/08-physreflect-vla/README.md) | online feasibility / reflection 对照 |
| LaWAM | [本地，归方向三](03-control-distillation-and-world-models/08-lawam/README.md) | 主要贡献是 latent predictive subgoals；方向二仍提供 cross-link |

## 方向三的 11 个原条目

| 原条目 | 本次处理 | 核验与阅读理由 |
|---|---|---|
| WorldVLA | [本地](03-control-distillation-and-world-models/09-worldvla/README.md) | unified action / image modeling |
| RynnVLA-002 | [arXiv:2511.17502](https://arxiv.org/abs/2511.17502) | 扩展：同类 unified VLA / world-model 系统；第一轮先读 WorldVLA |
| GPC | [本地](03-control-distillation-and-world-models/10-gpc/README.md) | 当前 manuscript 标注 RA-L 2026；inference-time planning |
| World Model for Robot Learning survey | [arXiv:2605.00080](https://arxiv.org/abs/2605.00080) | 扩展导航：survey 用来查分支，不当作一个新控制算法 |
| DiffOG | [本地](03-control-distillation-and-world-models/11-diffog/README.md) | arXiv manuscript；原总结的 T-RO 2025 标签本次未独立闭合，不标作 journal final |
| Differentiable Predictive Control with Safety Guarantees… | [本地](03-control-distillation-and-world-models/04-dpc-cbf/README.md) | CDC 2022；DPC 为直接 policy optimization，安全保证涉及 online CBF |
| Differentiable MPC | [本地](03-control-distillation-and-world-models/03-differentiable-mpc/README.md) | NeurIPS 2018；可微求解不等于移除 MPC |
| BarrierNet | [本地前身稿与正确 journal link](03-control-distillation-and-world-models/05-barriernet/README.md) | **原 ID 2003.09140 错误**：实际是 *Tactic Learning and Proving for the Coq Proof Assistant*。T-RO 2023 正确 DOI 为 [10.1109/TRO.2023.3249564](https://doi.org/10.1109/TRO.2023.3249564)；本地使用作者主页提供的 2111.11277，不冒充 journal binary |
| SafeFlow | [arXiv:2504.08661](https://arxiv.org/abs/2504.08661) | 扩展：test-time Flow Matching Barrier Functions；与 VLSA 一起看 online safety，但不是 solver-off student |
| VLSA | [本地](03-control-distillation-and-world-models/12-vlsa/README.md) | 完整标题含 *Safety Constraint Layer*；IROS 2026 accepted |
| Diffusion Policy | [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) | 扩展基础：原 RSS 2023，当前 arXiv 已是 extended journal version，引用时注意版本 |

## 本次新增的 6 个条目

| Paper | 补足的空缺 |
|---|---|
| [Shielding-Aware Dual Control](02-active-probing-and-dual-control/01-shielding-aware-dual-control/README.md) | dual effect、belief、exploration–exploitation 的严格 formulation |
| [CoMe-VLA](02-active-probing-and-dual-control/05-come-vla/README.md) | 原总结漏掉的 VLA active-perception / memory 工作 |
| [MPC-Guided Policy Search](03-control-distillation-and-world-models/01-mpc-guided-policy-search/README.md) | sensor-based neural policy 从 MPC 学习的早期机器人先例 |
| [MPC-Net](03-control-distillation-and-world-models/02-mpc-net/README.md) | control Hamiltonian guidance 与 real robot evidence |
| [ABNet](03-control-distillation-and-world-models/06-abnet/README.md) | closed-form explicit barrier 与迭代 optimizer 的区别 |
| [TD-MPC2](03-control-distillation-and-world-models/07-td-mpc2/README.md) | decoder-free control-oriented world model 基础 |

## 第一个 report 的两篇

- [WaterSIC #23](../reading-guide/papers/23-watersic/README.md)：ICML 2026 官方条目和 arXiv v2 一致；Spotlight 等级未独立核实。0.255-bit result 需要对应理论条件，不能泛化为任意 VLA 的 guarantee。
- [GRACE #24](../reading-guide/papers/24-grace/README.md)：ICML 2026，Yanlong Chen / Amirhossein Habibian / Luca Benini / Yawei Li；QAT + KD 的 VLM weight-only 工作。原 report 未署名，未独立确认师兄本人作者身份。

## 资料未覆盖的部分

这不是 exhaustive systematic review，也没有证明某个具体 research idea 必然 novel。数据库查询、指定引用的 metadata 核验、下载与重点章节阅读分别记录；选进本库表示值得读，不能替代独立复现、代码审计或你自己的方法判断。
