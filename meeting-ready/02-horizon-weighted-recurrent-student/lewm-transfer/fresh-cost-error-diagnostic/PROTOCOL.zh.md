# Fresh-bank cost-error 诊断：冻结协议

本作业复用 terminal-response-loss job `25549480.pbs101` 的 8 个 episode、两份 step-3000 checkpoint 与 official LeWM teacher，并以相同 seeds 和 builder 确定性重建 48 个 candidate banks。用途是描述“动作响应 MSE 小幅改善、elite ranking 仍 NO-GO”时的误差路径；不重新训练、不重新选择 episode、不运行 official CEM 或 closed-loop。精确定义和身份路径在 [FREEZE.json](FREEZE.json)。上一作业没有保存逐 candidate 原始张量或运行时源码 revision，因此指标对齐是当前代码的重建 sanity check，不能证明张量或代码 bitwise 一致。

## 身份先决条件

在 PBS compute node 上用上一实验的 fresh-row builder 和冻结 seeds 重建同一候选 banks。逐 bank 将两臂的 pairing key、Spearman、Top-30 recall、standardized elite regret、normalized response MSE 和 teacher contrast energy 与上一作业的保存结果对齐。Top-30 recall 必须完全相同，连续量最大绝对差不超过 `1e-4`；任何不匹配即停止，不输出诊断结论。新输出不保存大张量。

## 数学分解

对一个 bank 的第 `i` 条动作，令 teacher/student 第五步 latent 为 `zT_i/zS_i`、goal latent 为 `g`。误差 `e_i=zS_i-zT_i` 分为 `b=mean_i e_i` 与 `q_i=e_i-b`，满足 `mean_i||e_i||²=||b||²+mean_i||q_i||²`。定义 teacher/student 的 bank-centered 动作响应 `t_i/s_i`。令 `D=sum_i||t_i||²`，把 `s` 在 `t` 方向上的投影系数记为 `a=sum_i(s_i·t_i)/D`，剩余正交分量记为 `o_i=s_i-a t_i`；于是 `q_i=(a-1)t_i+o_i`。两部分误差能量都除以 `max(D,300×192×1e-6)`，其和精确等于沿用 floor 的 normalized response MSE。若 teacher response mean squared energy `<1e-6`，标记为低能量，不对该 bank 的归一化几何量作强解释。

对 terminal squared goal cost，记 `r_i=zT_i-g`，则 score error 为 `2r_i·b+||b||²+2r_i·q_i+2b·q_i+||q_i||²`。逐 candidate 验证等式闭合，分别报告共同项与动作对比项、以及三个动作对比子项的去均值 RMS；**RMS 数值本身不可相加**。计算 student 原评分以及四个 teacher-oracle 反事实（移除 `b`、移除 `q`、修正沿 teacher 动作响应方向的投影系数、移除正交动作响应）对 Top-30 recall、标准化 teacher-elite regret 和边界反序的影响。反事实只用于定位问题，不是可部署方法。

每个 episode 的 6 个 banks 为重复测量；配对差值先在同 bank 计算，再取 episode median，最后汇总 8 个 episode。另列继承绝对门槛失败的 banks、最差 banks。这些失败 bank 已由上一作业指标确定，不是独立验证样本。结果是描述性诊断，没有新 GO gate，也不改变上一实验的 NO-GO。

## 运行边界

模型、HDF5、checkpoint 与 bank 数据只在经过 `PBS_JOBID`、`PBS_NODEFILE` 和非 login-host 检查的获批 PBS compute allocation 内读取；login node 只传小控制文件、提交与查状态。GPU 使用率/显存每 5 秒进入作业 `job.log`。不查询或使用 banked reset credits。
