# 补充直接先例

2026-09-13，采集模型输出前补充检索。

[Zero-Shot Quantization for Vision-Language-Action Models via Trajectory Curvature and Attention Guidance](https://openreview.net/pdf/dcfe09ea1e77595f612807a5ebc8a92b39c6456b.pdf) 已将flow velocity的curvature用于量化校准数据生成。其式8使用velocity相对均值的平方偏差，和attention覆盖项共同优化合成observation。检索可读取原文索引片段，直接PDF访问遇到OpenReview challenge；未核验作者身份或接收状态。

因此“curvature + quantization”不构成新颖性主张。本候选保留的窄问题是：固定真实输入、冻结RTN recipe后，Q accel能否排序FP/Q action drift，并超过FP accel与Q action norm两个对照；无合成数据、无优化、无新curvature指标。该区别只是实验问题的不同，不能保证论文级novelty。结果无论正负均停止最小screen。
