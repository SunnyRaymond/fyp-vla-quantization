# C8 bilevel rho pilot：完成，No-Go

2026-10-20（Asia/Singapore）。8轮 bilevel 搜索返回 rho≈2，lambda≈0.05660604。按预先冻结的规则，从共同初始状态以该常数 rho 重训16步，未通过独立 confirmation 与历史 held-out 均改善的推进条件。暂不替换 development 选定的固定 rho=1.5，也不扩展到 C80 或 activation clip。

## 主要结果

指标为相对 BF16 teacher 的 first10 motor RMSE，越低越好；变化率相对 development 预先选出的固定 rho=1.5。

| 数据 | Cases × inference seeds | 固定 rho=1.5 | bilevel 选出的 rho≈2 | 变化 |
|---|---:|---:|---:|---:|
| 独立 confirmation | 8 × 2 | 0.043139591 | 0.054976732 | +27.44% |
| 历史 held-out | 40 × 2 | 0.045663730 | 0.044399541 | -2.77% |

confirmation 上 full32 motor RMSE +23.91%，late22 motor RMSE +22.13%，future latent NMSE +37.29%；这些也超过预设的最多退化5%门槛。历史 held-out 的五项指标均改善，不能抵消独立 confirmation 未通过。

按case先平均两个 inference seeds，confirmation 改善 3/8 cases，历史 held-out 改善 25/40 cases。两个 inference seeds 不是两次独立训练。

预设 gate：两份数据的 first10 motor RMSE 均至少下降1%，且其他四项指标在两份数据上均不得退化超过5%。最终 gate=False。固定对照在任何 confirmation/historical 评测提交前已冻结；未按最终数据改选对照、rho 或 checkpoint。

## 数据划分与公平性

- Inner 使用原 C8 的8个 contexts、seed0，固定顺序重复两轮，共16个真实 Adam updates。
- Outer development 新增8个 contexts，每个 LIBERO suite 2个，只用于 rho 更新与固定对照选择；seed0。
- 独立 confirmation 另8个 contexts，每个 suite 2个，每case使用相同的 seed0/1。两份新增数据按 demonstration 与原 C80 calibration、原40 held-out 及彼此隔离；保留原 task/phase 配对规则。
- 完整保留历史40 cases ×2 seeds。其结果在过去已被观察，因此作为回顾性证据，独立 confirmation 才是新增的泛化检查。
- 每个case内先平均 seeds，再在suite内平均cases，最后四个suite等权；保留全部cases。
- 三个最终arms为固定 rho=0.1、固定 rho=1.5、选出的常数 rho≈2；均从共同未训练 C8 banks 出发，在相同 deterministic_v2 设置下训练16步，lr=0.0005、gradient norm clip=1，update1后同样重置 Adam。614个目标参数组在每一步均有连接且有限的梯度。
- VQ-only、dynamic fake A4、decoded BF16 与 BF16 KV；本轮没有 activation clip。损失固定为 first10×6 motor MSE + rho×c×future global NMSE（排除conditioned frame0）。
- c 固定为 0.028303017887797623，不与 rho 联合学习；lambda_hat=0.05660603577559524。所有最终checkpoint均取update16，未择优挑中间checkpoint。

## rho 更新及数值预检

在 u=log(rho) 空间，初始 rho=1.5，范围[0.03,3]。每轮从完全相同的参数、Adam与RNG快照出发，在下一个 C8 context 分别以 rho/1.5 和 rho×1.5（边界截断）做一次虚拟 Adam 更新；分别在4个 outer contexts（每suite1个，两个batch交替）计算四suite等权 first10 motor RMSE。用两个端点的有限差分方向更新 u，每次乘/除1.5；相对对比不超过0.1%时保持不变。恢复原状态后，用更新的rho提交2个真实updates。总共8轮、16个虚拟updates、16个真实updates。

实际 rho 轨迹：1.5 → 2.25 → 3 → 2 → 3 → 3 → 2 → 1.333333 → 2。最终用于比较的是随后独立重训的常数rho模型，搜索过程中的动态rho模型不作为最终endpoint。

首次预检的参数/Adam/RNG恢复和输出恢复通过，但重复虚拟分支的development RMSE差约0.011756，有限差分信号不可信，故没有据此启动完整pilot。随后启用严格 deterministic algorithms、固定 CUBLAS 配置、关闭TF32并统一三个最终arms的数值设置；第二次预检在initial与非空Adam状态均通过恢复、重放及信号门槛：恢复输出最大差0、重复更新状态完全相同、重复分支RMSE差0、teacher sentinel最大差0。未将问题归因到某个未经隔离验证的kernel。详情见[数值修复记录](../NUMERICS_REPAIR.zh.md)。

旧的已训练 rho=0.1/1.5 checkpoints 仅只读用于预检诊断；其数值设置不同，不能直接作为公平最终对照。最终三个arms因此全部从共同初始状态重训。原始banks、C8与历史teacher缓存继续复用；新增outer/confirmation teacher写到本实验目录。

## 实际 GPU 成本

以下按 PBS resources_used.walltime × GPU数量核算，包含模型加载、checkpoint I/O、退出开销及所有已确认尝试，未计入历史来源作业。每个GPU作业为1张 NVIDIA A100-SXM4-40GB；全部请求 walltime=01:59:00。

| 阶段 | 实际 GPU·h |
|---|---:|
| 两次preflight，包含首次失败 | 0.887222 |
| 8轮 bilevel搜索 | 1.541389 |
| 固定rho=0.1重训 | 0.550000 |
| 固定rho=1.5重训 | 0.808333 |
| 选出rho≈2后重训 | 0.815278 |
| 三组outer评测、冻结固定对照 | 0.322778 |
| rho=0.1独立/历史评测，含新增confirmation teacher | 0.488056 |
| rho=1.5独立/历史评测 | 0.196111 |
| rho≈2独立/历史评测 | 0.477778 |
| **合计** | **6.086944** |

搜索本身为 1.541 GPU·h；选权重后从头重训另需 0.815 GPU·h。两次预检另用 0.887 GPU·h。总计6.087 GPU·h还包括两个固定对照和全部评测，不能把它全部当成纯搜索开销。共享outer作业的加载与I/O无法按arm精确分摊，因此不声称一个反事实的固定rho方案会恰好节省多少整作业GPU时间。

实际新增计算操作共 90 次 forward/backward（含虚拟与重放）及 449 次 inference（含teacher与预检）。操作本身累计 3.739 小时，不含加载及checkpoint I/O，不能替代整作业GPU成本。复用8份C8 teacher、80份历史teacher；outer新增8份、confirmation新增16份，另做一次teacher parity重放。

## 完整指标

| Arm | Split | first10 motor RMSE | full32 motor RMSE | late22 motor RMSE | gripper RMSE | future NMSE |
|---|---|---:|---:|---:|---:|---:|
| low | outer | 0.044550663 | 0.051674944 | 0.053829258 | 0.094639482 | 0.047535293 |
| low | confirmation | 0.051451345 | 0.052784431 | 0.053097858 | 0.018203939 | 0.058528276 |
| low | historical | 0.043238646 | 0.048830249 | 0.050546825 | 0.048629354 | 0.055671664 |
| current | outer | 0.041933100 | 0.048189431 | 0.050100268 | 0.017021486 | 0.043017250 |
| current | confirmation | 0.043139591 | 0.045638952 | 0.046445387 | 0.016883708 | 0.041401048 |
| current | historical | 0.045663730 | 0.052534169 | 0.054423920 | 0.047925009 | 0.065219072 |
| selected | outer | 0.046343835 | 0.048399811 | 0.049165943 | 0.095403515 | 0.057514507 |
| selected | confirmation | 0.054976732 | 0.056550184 | 0.056722655 | 0.017521921 | 0.056839021 |
| selected | historical | 0.044399541 | 0.047463754 | 0.048398186 | 0.032110753 | 0.056466288 |

工程推进gate：False。

## 解释与证据边界

本次验证了有限差分和状态恢复可以可靠执行，但没有证明这种8轮、单步unroll、最终常数rho重训方案能找到泛化更好的权重。局部单步目标与最后16步常数rho模型之间可能存在偏差，development样本量及训练轨迹也可能影响结果；这些只是待验证的解释，当前不能确定原因。

该pilot仅有一条C8训练轨迹、两个配对inference seeds；结果是fixed-input teacher fidelity，使用decoded BF16/fake A4。它不等于closed-loop成功率、native低比特kernel性能、统计显著性，也不能否定所有bilevel方法。没有执行新增episode或模型预测动作。下一轮若继续研究，需要先提出能区分原因的方案并获得实验授权，本轮到此结束。

## 完成记录

- 11个当前阶段作业全部满足 PBS终态/Exit0、artifact Exit0、PHASE_COMPLETE、匹配完整summary；原首次预检虽作业退出0，但科学gate失败，已单独保留并计入成本。
- CPU aggregate `25881659.pbs101` 校验全部312个endpoint records：三组各8个outer、16个confirmation、80个historical queries，且checkpoint/rho/update16绑定匹配。
- 10个GPU作业均留有每15秒采集的GPU利用率与显存记录；小文件证据保存于telemetry_receipt.json。gpt-6-luna/xhigh subagent按600秒间隔只读监控，现已停止。
- 所有模型、checkpoint、HDF5及重操作在PBS compute allocation内完成；login仅用于轻量控制与小文件查询。无C80扩展、activation clip或额外训练提交。

机器可读结果：[comparison.json](comparison.json)、[GPU_COST.json](GPU_COST.json)、[final_verification.json](final_verification.json)。
