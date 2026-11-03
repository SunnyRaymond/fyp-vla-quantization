# Cosmos-Policy 上的 LAVQ（cosmos-lavq-20261102）

把 LAVQ 的四个模块搬到 Cosmos-Policy-LIBERO-Predict2-2B 上，W2 和 W3 两个位宽；闭环和 Cosmos A16 基线
（`cosmos-a16-baselines-20261101`，840 集 LIBERO-Plus）逐集配对。

## 和 Fast-WAM 的差别，以及这里怎么处理

- **① reader metric**：Cosmos 用一个 DiT 联合去噪 9 个 latent slot（4 = 动作，6/7 = 未来 wrist / 第三人称图像）。
  z 定义为第 c 次 denoiser 评估之前求解器状态里的 slot 6/7；R = 剩余求解步 + 最后一次去噪。切点 c 从 {2,3,4} 里按
  C80 上的 future share ρ（把未来换成 teacher 的之后，QTIP_W2/W3 动作误差下降的比例）取最大。Z、Z_M、K 都在这个状态上算；
  最终输出的未来 NMSE 只做诊断。实现沿用 `cosmos-policy-vqwam-20261024/reader.py` 的切点续跑（teacher 恒等 1e-6，
  输入梯度连通）。
- **② 分配**：Cosmos 没有"只读未来、不生成未来"的权重，所以不存在 Δ_g = 0 的组。改成交换规则：9 组（3 个深度段 ×
  {self-attn, cross-attn, MLP}），每组测升一位的收益和降一位的代价（都用完整 L = A + λ_M Z_M + βK），代价密度最低的组整组降位，
  省下的比特按收益密度给接收组、组内按 H-proxy 每比特下降排序；前缀 0..J 在 C80 上实测，取 L 最低的。前缀 0 = 均匀 QTIP。
- **③ trellis 初始化**：K = 2/3 用基线保存的原生 QTIP trellis 码；K = 1/4 本作业用同一移植（`ravq_qtip.py`）、同一种子和
  C80 Hessian 量化。所有层的码都是原生的，不需要重新量化（Fast-WAM 的 W2 base-point 问题在这里不会出现）。
- **④ 共享表校准**：ravqc v3 规则（只调表、最大值归一化步长、从 FP32 解码 fold），表按 (层类型, K) 分组；梯度经过 5 步
  res_multistep 采样器的可微 endpoint，K 的 cotangent 来自冻结的 BF16 后半段 reader，staged replay 要求 max-abs ≤ 1e-4。
- **协议项**：Cosmos 没有 action/proprio encoder；patch embed / final layer（接口）和 adaLN / 时间嵌入（只依赖噪声水平，
  等价于逐步查表）在所有臂里都是 BF16，和基线闭环完全一致。

## 门槛

- LAVQU（只做 ②③）：held-out 40×2 上 L ≤ QTIP_Wb，否则该位宽退回前缀 0。
- LAVQ（加 ④）：held-out L ≤ 它的未校准底座（Fast-WAM 同一规则），否则闭环跑未校准臂。
- 最终臂若和均匀 QTIP 完全相同，就不再跑闭环（基线闭环即结果）。

## 作业

| 作业 | 脚本 | 队列 | 内容 |
|---|---|---|---|
| lvct | cputest.pbs | CPU | test_lavq.py + gpu_lock 跨容器互斥 |
| lavq1 | run.pbs | g2，2 卡，7 h | quant → teacher（含切点）→ metric → sens → alloc → heldout → codes → calib → final |
| lavql | run_loop.pbs | g3，4 卡，5 h，afterok lavq1 | DECISION.json 里的臂 × 840，配对分析 |

`control.py upload / cputest / submit --jobs pipeline loop --after cputest / status / pull`；
`poll.py --until <key>` 后台监控。

## 结果（流水线 26095412，闭环 26095413）

- ① 切点 4（最后一个求解步之前），C80 future share ρ：W2 0.022、W3 0.004；metric 有效秩 152.5 / 25088，
  半数对角相关 0.85，top-r 子空间重叠 0.05。λ = 12.24，β = 0.4255。
- ② W2 选前缀 3（b1_cross、b0_cross、b0_mlp 降到 W1，62 层升到 W3），W3 选前缀 3（b0 三组降到 W2，93 层升到 W4），
  比特数与均匀 QTIP 相同。held-out L：LAVQU_W2 = 0.417 × QTIP_W2，LAVQU_W3 = 0.406 × QTIP_W3。
- ④ W2（η = 0.02，8 次更新，重放误差 0）held-out L = 0.712 × LAVQU_W2，通过；W3（η = 0.005）1.024 ×，未通过。
  闭环臂：LAVQ_W2、LAVQU_W3。
- 闭环 840 集（与基线逐集配对）：LAVQ_W2 676（QTIP_W2 671，39 胜 / 34 负，p = 0.64；QuIP# 649，p = 0.0045；
  GPTVQ 610，p = 1.1e-9；BF16 701，p = 0.0035）；LAVQU_W3 695（QTIP_W3 695，21 / 21；BF16 701，p = 0.44）。
  详见 `pulled/results/RESULTS.zh.md`、`pulled/results/LOOP_RESULTS.zh.md`。
