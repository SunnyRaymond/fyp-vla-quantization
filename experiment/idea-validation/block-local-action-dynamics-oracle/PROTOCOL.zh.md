# 已知正确坐标的 toy oracle 对照

## 问题

第一轮的 predictor 同时要学习坐标变换 `Q` 和 dynamics 函数 `f`。如果预测误差较高，我们无法知道主要障碍是坐标没有学对，还是分块后的 `f` 本身不够好。本对照将坐标学习从训练中拿掉：直接给模型生成器使用的正交矩阵 `M`，冻结它，只训练原来的 predictor/message。这样可测出“已知正确分块时，这些 predictor 能做到什么程度”。

## 已冻结的设计

- Controlled systems：`independent`、`lowrank_coupled`、`dense_coupled`。
- 主实验：`oracle_block`、`oracle_global4`、`oracle_local4` × 3 conditions × seeds `1101/1102/1103`，共 27 runs。
- 附加对照：`lowrank_coupled/oracle_global16` × 同三个 seeds，共 3 runs；总计 30 runs。`global16` 同时增大 message 和 predictor 输入维数，参数量也增大，因此不能单独识别通信带宽的作用。
- 数据、训练、seed、batch draws、loss、1500 steps、最后 checkpoint、horizons `1/5/10/20`、action-response 和计时配置沿用第一轮。数据按 episode 划分，train/dev/test 为 `512/128/128`。
- 用第一轮的 `M` 定义 `z=x@M.T`，模型固定使用 `s=(z-mean_z)@M`，因此 `s=x-mean_z@M`。`M` 不接收梯度且不会更新。每个 oracle predictor/message 的初始化逐张量匹配对应第一轮 learned arm 同 seed 的初始化；只替换 `Q`。
- 在真实 PBS compute allocation 内重载第一轮全部对应的 learned 与 dense checkpoints，以及三个 `learned_global16` checkpoints。重新生成相同数据，复算全部 held-out per-episode horizon errors，与第一轮保存的 arrays 比较；对齐的 oracle 与 learned、dense 进行 episode-paired bootstrap。
- 对照速度时，在同一 PBS allocation 重测新 oracle 与原 dense checkpoint 的完整 rollout：B1/B300、H10、warmup 20、repeat 60，同时报告每步还原的延迟。沿用第一轮质量门槛 `error <= dense + max(0.1*dense, 0.02)` 和速度门槛 `>=20%`；门槛不随结果调整。

## 必须通过的机制检查

1. `z=x@M.T` 与 `s=(z-mean_z)@M` 的坐标映射可逆，固定 `Q` 与生成器 `M` 一致。
2. 固定坐标没有梯度且 optimizer 不会更新它；第一轮对应 learned arm 与 oracle arm 的 MLP/message 初始化逐元素相同。
3. `independent` 系统在已知坐标中的真实跨块 Jacobian 为零，`oracle_block` 的分块更新也没有跨块路径。
4. `oracle_global4` 的消息可读取全部四个块；`oracle_local4` 每块消息只读取本块。
5. 逐 checkpoint 复算的原始 per-episode errors 与第一轮数组在冻结容差内一致，且数据、归一化量、seed 和 episode 数相符。

## 解释边界

这是 oracle 上界式诊断，不是可部署的方法：模型得到了通常未知的生成器坐标。`oracle_block` 若接近或超过 dense，说明第一轮 learned 分块差距主要来自坐标学习；若已知坐标下仍明显落后，则应优先检查局部 predictor 的容量/函数形式或优化；`global4/global16` 的结果再显示信息交换和容量的增益。所有差异仍需结合 action-response、horizon 和同 allocation latency 解读。每组只有三个训练 seeds，保持为描述性 pilot；不据此作训练随机性总体推断。

## 运行

数值机制检查、训练、baseline checkpoint 加载和结果报告必须在已分配的 PBS compute node 执行。登录节点只用于小文件 staging、提交和状态查询。根任务维护的 `run_oracle.pbs` 单次调用 runner；runner 内部先执行机制检查，再训练与复现原 checkpoint，最后生成报告和 `DONE.json`：

```text
python oracle_run.py --config FREEZE.json --output "$OUT"
```

`oracle_run.py` 使用 global16 job `25563721.pbs101` 的第一轮 `source_snapshot`（包含所有 arm 的 model factory），复用第一轮的 dataset、训练、评估、action-response、benchmark 和 bootstrap 函数，并将实际使用的 helper 复制到输出快照。checkpoint 与 per-episode arrays 只在 PBS compute allocation 内读取；大型工件留在远端，结果摘要和报告可按小文件规则取回。`test_oracle.py` 与 `write_oracle_report.py` 也保留可单独调用的 guarded entry point，PBS 主流程仍只调用 `oracle_run.py`。
