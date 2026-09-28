# LeWM 分块 pilot

2026-09-26 已完成 PBS `25566877.pbs101`：15/15 fits，终态 F，Exit_status=0，运行11分42秒。主相对质量结论为 **NO-GO**；按冻结协议停止，Stage B/C未运行。

阅读 [结果与解释](results/25566877.pbs101/RESULTS.zh.md)，数值依据见 [小型摘要](results/25566877.pbs101/summary_brief.json)。[PROTOCOL.zh.md](PROTOCOL.zh.md) 与 [FREEZE.json](FREEZE.json) 记录执行前固定的设计和门槛。

模型：192D保维正交换坐标、6×32D局部更新与16D共享消息；对照包括同预算Flat、旧balanced_base、Local通信消融和identity坐标诊断。固定三个training seeds、每臂3000updates、512个训练contexts；新held-out包含16个episode与96个嵌套候选blocks。

所有模型/data I/O、训练、tensor checks、计时和bootstrap均在真实PBS allocation执行；GPU utilization/显存记录见结果目录job.log。原始checkpoints、fresh_rows.pt、逐arm/block数据与source snapshots保留在远端：

`/scratch/users/ntu/yguo017/lewm-block-local-pilot/runs/25566877.pbs101`

早期直接g1提交被明确拒绝，随后normal成功路由到g1；拒绝attempt保留，未出现重复job。没有追加训练、修改门槛或执行超出门槛的CEM/闭环评价。
