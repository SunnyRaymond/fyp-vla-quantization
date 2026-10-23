# C8 anchor KL 权重、优化与方向：完整机制诊断

本报告仅覆盖固定 C8、两个参数点和预先声明的小步干预。沿用 teacher 和缓存端点，原 trained baseline 保留既有结果。各 case 固定一个 sampler seed；两个参数点及同一输入的多次 probe 均不是独立训练重复。

## 参数梯度

| 参数点 | base 与 KL cosine | KL/base 梯度 norm | action 与 KL cosine | action 与 NMSE cosine | KL 跨 case 负 cosine pairs / 28 |
|---|---:|---:|---:|---:|---:|
| initial | 0.891232 | 0.116418 | 0.394648 | 0.500428 | 4 / 28 |
| final_kl | 0.563759 | 0.101563 | 0.253142 | 0.495003 | 14 / 28 |

完整逐 case、books 与 log-row-gains 的梯度几何保存在 JSON。跨 case 负方向是共享参数下的局部干扰证据，不单独证明历史遗忘；STE 梯度也不保证实际量化 forward 下降。

将加权 KL 参数梯度写成 a × 加权 NMSE 梯度 + 正交部分，JSON 报告 a、对应的局部有效 rho，以及正交部分与 action 梯度的关系。这只是同一点的 surrogate 梯度分解，不等于两个 loss 全局等价或 Adam 实际更新等价。全 C8 平均梯度的正交部分同时包含逐 case 方向差异与隐含样本权重变化，不能全归因于新的结构方向；JSON 另给逐 case 投影。另逐 case 对齐输出 latent 与参数空间的 cosine，避免把聚合差异误认为模型反传的作用。

## 减小 beta 的真实 C8 效果

表中差值为 beta=0.1 倍减去原 beta，均为同一点、同 Adam 状态、同一步长；负值表示较小 beta 更好。

| 点 | Adam | 步长比例 | Action MSE 平均差 | Action 改善/变差 | 原 MSE+NMSE 平均差 |
|---|---|---:|---:|---:|---:|
| initial | fresh | 0.25 | +0.0005476259 | 2/6 | +0.001227593 |
| initial | fresh | 1.0 | -0.0001989218 | 5/3 | +2.715188e-06 |
| final_kl | fresh | 0.25 | -0.0006237428 | 6/2 | -0.001491539 |
| final_kl | fresh | 1.0 | +0.0001022654 | 4/4 | +8.52491e-05 |
| final_kl | saved | 0.25 | +0.0002079792 | 1/7 | +4.554905e-06 |
| final_kl | saved | 1.0 | -0.0001363921 | 5/3 | -0.0001704113 |

beta=0 对照、配对中位数和全部 8-case 差值保留在 JSON；不将本组条件用作新 checkpoint 选择。

## 真实 finite-step 与纯 KL 方向

| 点 | 步长比例 | 纯 KL 更新的实际 KL 差 | Action MSE 差 | KL 降、action 升 |
|---|---:|---:|---:|---|
| initial | 0.25 | -0.001009569 | +0.0001774093 | True |
| initial | 1.0 | -3.562549e-05 | +0.0001825568 | True |
| final_kl | 0.25 | -0.002724171 | +9.634621e-05 | True |
| final_kl | 1.0 | -0.003454471 | -7.789506e-05 | False |

## 部署量化与 Adam 几何的补充分析

这一部分在观察到初始化 probe 的 BF16 row gains 无变化后追加，只重读既有梯度、原 C8 banks 和 checkpoint。它不改变任何 GPU 条件，不新增 forward。CPU 重建的 norm/g-dot-delta 与 GPU probe 保存值的偏差记录在 JSON。

| 点 | 条件 | Base 预测差：books | Base 预测差：row gains | 部署 gain 改变数 | 实际 base 差 |
|---|---|---:|---:|---:|---:|
| initial | fresh_beta1.0_step0.25 | -0.0004462158 | -0.0006355122 | 0 | -0.0008897766 |
| initial | fresh_beta1.0_step1.0 | -0.001784863 | -0.002542049 | 0 | -0.0006920003 |
| final_kl | fresh_beta1.0_step0.25 | -0.0002178747 | -0.0003119126 | 55332 | +0.0008981694 |
| final_kl | fresh_beta1.0_step1.0 | -0.0008714986 | -0.001247651 | 202194 | -0.0008744879 |
| final_kl | saved_beta1.0_step0.25 | -1.960616e-05 | -2.848057e-05 | 18402 | +0.0007174934 |
| final_kl | saved_beta1.0_step1.0 | -7.842466e-05 | -0.0001139223 | 76343 | -0.0003510068 |

非零 FP32 master delta 可以在 BF16 存储后完全消失。按参数组分解的一阶预测用于定位这种落差；它仍不能单独把全部真实-loss 变化归因于 rounding，因为 books 更新和 A4 的 STE 误差也同时存在。


22 个参数 probe panels 中，surrogate 预测下降但真实均值上升的计数：{'action': 11, 'base': 5, 'kl': 2}。这些条件相关，不是 22 次独立实验。步长、量化 rounding 与 STE 方向近似可能共同影响结果，不能仅凭符号不一致认定 autograd 实现错误。

各 panel 相对原参数点的逐 case action/base 改善数、配对中位数和差值也保存在 JSON，不以单个均值代表全部 C8。

## 等 NMSE 的 latent 方向干预

两种修复从同一个保存的 candidate future 出发，实际 BF16 注入后匹配 raw NMSE；同一原 BF16 reader、同一 action noise，teacher-future reader 重放通过。

| 修复 | Future NMSE | Anchor KL | Reader first10 motor RMSE | Reader action MSE |
|---|---:|---:|---:|---:|
| original | 0.04956889 | 0.006222381 | 0.01682647 | 0.0003697968 |
| repair_visible | 0.04567593 | 0.003099024 | 0.01601541 | 0.0003351868 |
| repair_null | 0.0456758 | 0.00622368 | 0.01614563 | 0.0003385737 |

修复可见部分 − 修复 null 部分，reader action MSE 平均差 -3.386898e-06，中位数 -1.98107e-06，前者更好 4/8、后者更好 4/8。

Nullspace 维数很大，不能只看其包含的梯度能量比例：完整 future 有 37632 个坐标，KL 可见空间约 1248 维。JSON 同时报梯度能量按维数归一化后的集中程度。修复比较控制最终 NMSE，但两种 repair 的位移长度未必相同；它是误差结构干预，不是等幅梯度方向测试。

## 完成与边界

源 GPU 作业 25905057.pbs101；CPU 汇总作业 25905075.pbs101。16/16 参数梯度 records、22/22 参数 panels（176 case queries）、8/8 reader records（32 reader queries）。
累计 student full queries=192，reader-only queries=32，参数 reverse passes=48，reader reverse passes=8。内部 checkpoint recomputation 不作为额外逻辑 query。
原 trained baseline 0 新 queries、0 新 teacher full queries、0 持久 training updates、0 closed-loop episodes。没有通过上述数据证明独立泛化或闭环优劣。权重与方向证据可能同时成立；最终解释需结合这几张表，不把任一单个 cosine 或平均数当作机制证明。
