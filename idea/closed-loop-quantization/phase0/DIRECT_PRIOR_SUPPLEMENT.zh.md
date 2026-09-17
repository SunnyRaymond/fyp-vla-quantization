# 直接近邻补检（bounded web full-text supplement）

日期：2026-09-12。Phase 0 connector 已因 `feedparser`/`openreview` 缺依赖及 Semantic Scholar 429/OpenAlex 504 降级；本文件只补取少量 primary web full-text/official repository，不重跑整池、不安装依赖、不下载模型。它用于新候选的 prior boundary，不能把 bounded search 当作 canonical exhaustive review。

## 取到的全文要点

| 来源 | 机制（按网页全文/摘要） | 对本候选的边界 |
|---|---|---|
| [AdaRound, arXiv:2004.10568](https://arxiv.org/abs/2004.10568) | 用数据和 task loss 学习 rounding；把 rounding 近似为 QUBO，并用 soft relaxation 做 layer-wise local loss；不需要 fine-tuning。 | 新候选可借用 AdaRound 式硬 rounding/materialization，但 load-bearing loss 是固定 self-induced residual 方向上的 transport，不是 layer-local reconstruction/QUBO。 |
| [QDrop, arXiv:2203.05740](https://arxiv.org/abs/2203.05740) 与 [official code](https://github.com/wimh966/QDrop) | PTQ reconstruction 时随机 drop activation quantization，以提升 low-bit model flatness；论文覆盖 vision/NLP，仓库给出 W/A 低比特实现。 | 作为同 calibration/eval budget 的 input-noise/activation-drop baseline；不把随机噪声当作真实 Q0 residual direction。 |
| [PD-Quant, arXiv:2212.07048](https://arxiv.org/abs/2212.07048) | 用 FP 与 quantized prediction difference 做 global prediction metric，并加 block-output regularization/activation distribution correction；正文明确讨论 scaling factor 与 rounding，PD-only 在小 CAL 上会 overfit。 | 新候选应报告 clean reconstruction、fresh residual DEV 与 overfit；transport term 测 `F(x+δ)-F(x)` 的 finite difference，区别于 final prediction difference/activation distribution correction。 |
| [Sobolev Training, arXiv:1706.04859](https://arxiv.org/abs/1706.04859) | 在 pointwise output 外匹配 target derivatives，保留 teacher 的局部函数行为；包含 compression/distillation 语境。 | 说明 derivative/JVP alignment 并非新原则；新卡只声称 finite-difference transport on a frozen, deployment-induced Q0 residual direction，并与 random same-norm/GAD-like control 比较，不宣称首次 sensitivity matching。 |
| [GAD, arXiv:2606.01651](https://arxiv.org/abs/2606.01651) 与 [official code](https://github.com/Hannah1102/GAD) | T2I distillation 以 JVP 对齐初始 noise sensitivity；仓库是训练式 geometric alignment。 | 这是关键 collateral baseline：random same-norm `delta` 和真实 Q0 `delta` 必须并列；若两者等效，候选 no-go。新卡是 frozen PTQ scalar calibration、无 teacher/student distillation training、无 JVP claim。 |

## 其他碰撞与检索限制

`RPIQ`（OpenAlex `W7119233972`）的摘要已提到 residual-projected closed-loop compensation、single-instance calibration 和 Gauss-Seidel iterative quantization；`When Can Depth Replace Precision?`（OpenAlex `W7171748342`）讨论 quantized residual system、increment error feedback 与 bounded carry。这两项说明 residual/closed-loop/error-feedback 的宽泛叙事已有直接碰撞；新候选不实现 error-feedback accumulator、Gauss-Seidel compensation 或 mixed-bit allocation，只测试冻结 Q0 residual direction 的 finite-difference transport。

本补检没有验证 `RPIQ`/`Depth Replace Precision` 的完整方法，故仅作 collision warning；没有检出 exact match 也不等于不存在。Phase 0 的直接近邻 DA-PTQ、QuantWAMs、Feedback World Model、MARR 仍以原表和全文为准。
