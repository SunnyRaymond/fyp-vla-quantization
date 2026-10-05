# Fast-WAM denoising 时间方向并行实验协议

本协议只定义固定模型上的 action denoising 时间并行实验。固定使用父目录 `PROTOCOL.zh.md` 中的 FastWAMOptionalIDM `first_frame` 基线、revision `7faa71108368fbb3b6885649f112af607427a2d4`、BF16、sigma shift 1、compile=false 和原生 10 个离散 Euler 节点。不得改模型、训练或蒸馏，不减少或重排原生节点，不改变 action mask；时间点作为独立 batch 维输入，不能拼成 token 轴。视频 cache 和 batch broadcast 由 runner 处理。

## 算法定义

原生离散更新为

\[
A_{k+1}=A_k+\Delta_k v(A_k,t_k),\quad k=0,\ldots,9.
\]

对一个长度为 \(m\) 的非重叠窗口，左端 \(A_s\) 固定，窗口内未知 action state 以 \(A_s\) 复制初始化，禁止用 teacher/native 中间状态初始化。每轮对所有猜测并行调用 `denoise(actions, timesteps)`。

- **prefix-Picard（主研究）**：
  \[
  A_{s+q}^{(r+1)}=A_s+\sum_{j=0}^{q-1}\Delta_{s+j}v(A_{s+j}^{(r)},t_{s+j}),\quad q=1,\ldots,m.
  \]
  每轮结束后用新轨迹的前 \(m\) 个节点作为下一轮猜测；最后一个节点成为下一窗口左端。
- **triangular（舍入/fidelity 对照）**：
  \[
  A_{s+j+1}^{(r+1)}=A_{s+j}^{(r)}+\Delta_{s+j}v(A_{s+j}^{(r)},t_{s+j}).
  \]
  每轮只把正确依赖向右传播一个节点。相同 batch 输出且执行 \(R\ge m\) 轮时，数学上恢复原生 Euler 路径；batch-dependent kernel 的数值差异仍须测量。它是 fidelity control，不作为加速候选。

主实现 `picard_windowed(denoise, initial_action, timesteps, deltas, width, iterations, *, method="prefix", arithmetic="native", collect_trace=False)` 接收 `[32,7]` 初始 action；`denoise` 接收 `[W,32,7]` guesses 与 `[W]` timesteps。默认不保留 trace。性能计时时保持 `collect_trace=False`。

## 冻结候选与计算量

每组都执行全部 10 个原生节点。只测以下固定组合：

| Window width | Picard rounds R | Serial denoise batch calls | Scalar NFEs |
| ---: | --- | --- | --- |
| 2 | 1, 2 | 5, 10 | 10, 20 |
| 5 | 1, 2, 3, 5 | 2, 4, 6, 10 | 10, 20, 30, 50 |
| 10 | 1, 2, 3, 5, 10 | 1, 2, 3, 5, 10 | 10, 20, 30, 50, 100 |

原生对照是 10 次 batch-size-1 forward、10 scalar NFEs。候选计时包含全部去噪调用、窗口更新和窗口边界传递；报告 batch call 数与 scalar NFE，不能把 teacher/native 轨迹计算、缓存准备或诊断 batch probe 算作候选免费成本。原生轨迹上的 batch-forward probe 单列为离线并行能力诊断，不用于部署加速结论。

Prefix 默认 `arithmetic="native"`：乘法、prefix 累加和左端相加均保留 action dtype（BF16），但 `left + sum(increments)` 的分组舍入不等于逐步 native Euler 舍入。BF16 prefix 用逐项 action-dtype tensor additions 构造，含每窗口每轮的更新开销；这些更新和 kernel launch 都计入完整求解时间，不能称为免费。`arithmetic="fp32"` 只用于单独标记的数值敏感性检查；如要纳入候选，必须通过同一 fidelity gate 和 strict native reference，不得与 BF16 主结果混报。Triangular 只执行 action dtype 更新，用于验证原生舍入依赖。

## 配对与判定

- 同一 captured observation、初始 action noise、CFG、schedule、seed 和 context 下配对比较。每个 context 先 3 次 warmup，再做 4 对 AB/BA，逐对交替原生和候选顺序；各观测是 block，4 次计时是技术重复，不能当独立 task。
- 候选完整求解计时在 CUDA 前后同步。另测 wall time；allocated/reserved 基线及峰值另列。GPU utilization / memory 每 15 秒采样到当前 PBS job log。
- 记录完整 32×7 normalized action 的 finite、max absolute error 与 relative L2。Pilot screening gate 固定为 `max_abs <= 1e-3` 且 `relative_L2 <= 1e-3`；另记录首 10 个 action 的 max absolute error 和后处理 gripper 符号差异。gate 只筛选计算近似，不能证明 closed-loop task preservation。
- strict native reference 是最后 fidelity validation。任何 gate 失败、OOM 或没有净加速都按当前配置 NO-GO 记录，不降低阈值、不删节点、不加训练，也不扩任务或搜索轮数。
- 主 latency 比较只用原生串行 action core 对 prefix 候选，不叠加固定 context cache 线的优化收益。完整 infer wrapper 若只执行一次候选 solver，其 conditioning、cache、solver 和 CPU 返回成本都计入；不能用剩余原生循环作 dummy call。

所有 model load、inference、toy `self_check()` 和结果生成只在具备真实 `PBS_JOBID`、nodefile 与非 login hostname guard 的已获批 PBS compute allocation 中运行。不得在本机执行模型或 toy inference；本文件和 Python 源码可以在本机编辑、做语法检查。Login node 仅用于提交、轻量状态查询及小型摘要读取。
