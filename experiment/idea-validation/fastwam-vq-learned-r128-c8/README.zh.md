# 固定 Smooth：H+VQ8 与 learned R8+VQ8

本轮结果：**R8+VQ8 没有超过 H+VQ8，当前保留 H+VQ8 作为 baseline。** 完整配对结果来自成功作业 `25902163.pbs101`；每臂 40 contexts × 2 seeds，共 80 queries。

| Heldout 指标（越低越好） | H+VQ8 | R8+VQ8 | 变化 |
|---|---:|---:|---:|
| First10 motor RMSE | 0.044296774 | 0.046408990 | +4.768% |
| Future latent NMSE | 0.060877371 | 0.065187050 | +7.079% |
| Mixed loss | 0.004598652 | 0.004980800 | +8.310% |

Motor RMSE 与 mixed loss 各有 24/40 个 contexts 退化，四个 suites 的 mixed loss 均上升。该结论针对当前 C8、R8→refit→VQ8 recipe；没有评估更多 R updates、joint R/VQ training 或 D，因此不能据此否定 learned rotation 的其他训练方式，也不能推断 D 的收益。

H deployment 重放和最终 R deployment 的 sampler/formal-infer parity 均为 0；全部 614 层与 R/VQ 参数已冻结。完整表见 [RESULTS.zh.md](results/25902163.pbs101/RESULTS.zh.md)，协议与复用来源见下文及 `BASELINE_SOURCE.json`。

CPU 作业 `25902658.pbs101` 已核验两个 stream 的 R 确实变化：`max_abs(U-I)` 为 action `0.0162067`、video `0.0151227`，保存的最大正交误差约 `1.91e-6`。8 个 moments、614 个 fitted banks、两阶段各 8 次 updates 与全部 88 个评估 queries 的耗时记录完整；这次核验新增 model forwards 为 0。

成功 GPU 作业 walltime 为 **44分49秒**；R 学习 `504.816s`、moments 收集 `134.640s`、VQ refitting `509.468s`、VQ 训练 `264.117s`。这些是阶段操作计时，不能当作 native kernel latency。两次训练前启动失败分别耗时 `3分14秒`、`12分39秒`，另有一次 CPU 环境启动失败 `11秒`，均保留记录，不计入成功结果。成功 CPU 核验耗时 `2分50秒`。完整成本与来源见 [COSTS.zh.md](results/25902163.pbs101/COSTS.zh.md)。

用户授权的范围是两个 arm：复用 H+VQ8，并训练和评估 R8+VQ8。没有 H+VQ16。

固定 original LIBERO 四-suite C8：2000、2019、2021、2025、2051、2057、2066、2070，每个取 seed index 0。Smooth0.5 scales、原 signs、D=I、clipping ratio r=1、KV BF16、原模型参数和 inference 10+10 steps 均保持固定。

Loss 为 first10×6 motor MSE + 1.5×0.028303017887797623×clean future latent NMSE；latent 排除 clamped frame0，student future 保持完整 action 梯度路径。共享 BF16 teacher 不重新生成。

## 执行顺序

1. 在真实 GPU allocation 中通过小型 numerical selfcheck，并重放已训练 H+VQ8 的保存动作及正式 infer，确认复用对象。
2. 在 W16A4 下只学习每个 eligible stream 共享的 Cayley residual U：R=H128 U，U 从 I 开始。8个 C8 contexts 各更新一次，Adam lr=.001，clip norm=1，不清空首步 Adam 状态。所有原 signs 和不足128的 tail 保留。
3. 冻结 R，在 BF16W+A4 下重新收集8个 contexts 的614层 input moments。每个 context 先除有效 token rows，再在 contexts/suites 间等权。
4. 从原始模型的 W 构造 W S R 并重新拟合 d4/M2/K256、B32 VQ，sample_limit=8192、iterations=8、seed=20261006。新的 indices/exponents 不沿用旧坐标。
5. 固定 R、indices、B32 exponents，训练 codebooks/log row gains 8 updates。Adam lr=.0005，clip norm=1，并保留原C8配方中首步后清空一次 Adam state 的规则。
6. 冻结全部参数，检查 sampler/正式 infer parity，评估完整 calibration8 与 heldout40×2=80 queries；使用已有 H+VQ8 的80条记录配对汇总。

训练后 rotation 仍以原 signed FWHT 后接 U 的 factorized 算术执行，避免更改原 H 的初始数值顺序。它数学上是 learned R128；本轮不进行 folded/native kernel 性能声明。

## 公平性与证据边界

H+VQ8 来自同一冻结 C8 的 rho1.5/final8 deployment。`BASELINE_SOURCE.json` 记录训练来源、共享 teacher 和可复用80条记录。原 baseline GPU 在保存完整结果后发生已知清理错误，独立 CPU validation 已从全部80条记录重算通过；不能将原GPU exit1伪写成exit0。

主指标 first10 motor RMSE，同时报告 future latent NMSE、同定义 mixed loss 与 full32/late22/gripper 诊断。先在case内平均两个seeds，再suite内平均10cases，再四suite等权。固定最终R8/VQ8，不通过heldout挑checkpoint。

该比较允许 candidate 多出R8学习与新moments/VQ fitting成本，按阶段记录耗时；不能称为等算力比较。本轮为 decoded BF16 weight + fake A4 固定输入 fidelity 证据，没有执行环境episodes，不支持闭环成功率、native low-bit速度或真实压缩显存结论。

## PBS 和恢复

1 GPU/16 CPU/110 GB、normal、12h；所有模型、Torch数值检查、checkpoint读取/写入、VQ fitting与汇总都在真实 compute allocation。Login只做轻量状态、小代码传输和qsub。每15秒将GPU利用率与显存写入job log。subagent每20分钟只读监控；Q/R不变保持安静。

每个update、moment context、fitted layer和完整endpoint原子保存。达到soft deadline退出75并保留可恢复状态；只有证实前handle终止后才可续提，查询超时或queue comments不授权重提。不得使用banked reset或绕过SSH host-key验证。
