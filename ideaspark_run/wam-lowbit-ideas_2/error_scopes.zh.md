# 推导时必须分开的误差层级

这是局部线性化的组织框架，不是已验证的 WAM 定理，不预设新算法。

## 1. 同一 observation 下的 denoising 误差

令一个完整 inference request 的最终 action chunk 为 a，量化计算点 r 的局部误差为 e_r。若在 BF16 trajectory 附近一阶展开，且量化扰动未跨过不连续分支，则 delta_a ≈ sum_r J_r e_r；r 可同时包含 layer、stream、denoising step。

其平方误差包含对角项 sum_r e_r^T J_r^T J_r e_r，以及不同 r,u 的交叉项 sum_{r!=u} e_r^T J_r^T J_u e_u。Q-WAM 的 App. A.2 舍弃后者，SteerQuant 的单区域评分亦不能直接估计它们。是否存在稳定、可控制、可跨episode迁移的交叉项，需要实际诊断；看到非零交叉项本身不构成一个有效算法。CTEC/TAC/TCEC 已研究 diffusion 的跨步累积误差，因此不能把这条展开本身当 novelty。

## 2. 机器人实际执行的 commands

机器人可能只执行 chunk 的 prefix，并经 action unnormalization、clipping、gripper threshold、temporal ensembling 等固定 controller operations，写为 u=C(P_h a)。若 C 可微，可局部线性化为 delta_u≈D_C P_h delta_a；若有 threshold，则应显式评估相同 controller 产生的 commands 或边界翻转，不能把全chunk MSE等同于执行影响。P_h、C 必须从实际已固定 evaluator 来，不可为制造收益而改 controller。

## 3. 多个控制轮次的 feedback

即使同一 observation 上 delta_a 较小，下一轮输入仍会被已执行动作改变。光滑的局部状态模型给出 delta_x_(k+1)≈(A_k+B_k K_k)delta_x_k+B_k delta_u_k，其中 K_k 是 policy 对状态/observation 的局部反馈响应。这不提供无条件 closed-loop 保证；接触切换、渲染、controller discontinuity 或严重 distribution shift 都可能使局部模型失效。

## 4. 性能主张

上述内层误差、执行 command 偏差、outer closed-loop success，以及 native operator/端到端 latency 是不同证据。算法的初步证伪只需一个最小、可命名的机制实验；其通过不代表任务成功或加速。提议需说明部署是否仍调用 teacher、是否新增高精度分支、是否只优化 fake quantizer，及 W4A4/W4A8 的真实覆盖范围。
