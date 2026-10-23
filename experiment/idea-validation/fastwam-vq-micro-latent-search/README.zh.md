# Microscaling + latent-aware VQ：两轮 λ 搜索

用户批准的执行配置（2026-10-14）：旧 calibration 8 contexts、旧 selection 4 cases×2 seeds、旧 test 22 cases×2 seeds；第一轮只跑 `rho=0.1/0.5/1`；guardrails 放宽到20%；搜索两轮后报告 optimal λ；暂不做 microscaling/latent 互补实验。每个 rho 使用独立 PBS job、intent、handle与checkpoint。

用户随后补充 `rho=0.7/1.5` 两个独立粗搜点。当前第一轮共5个候选，第二轮仍3个，总计8个；两项新增身份保存在 protocol 顶层 `additional_round1_candidates`，原 `search` 数值合同保留，供已保存的三个候选 checkpoint 接续。提交后等待用户通知，不开启主动监控。

新增作业已确认提交：rho0.7→`25808524.pbs101`，rho1.5→`25808525.pbs101`；两者均从相同未训练 micro 初始化开始，复用冻结 `c=0.03033027132999191`，各自完成2epochs/16updates。

## 表示与 loss

直接复用原 B32 Microscaling VQ `25755896.pbs101` 的614 banks；固定 indices/exponents，训练 FP32 master books 与 log row gains，forward仍使用真实存储BF16值、A4和BF16 KV。

共享 reference 在真实GPU allocation生成micro calibration8+selection8，共16个parent forward，复用旧BF16 teacher60 contexts；不为每个rho重跑teacher，不在准备阶段生成parent test futures。

冻结 `c=mean_8cal_micro_initial_action_MSE / mean_8cal_micro_initial_future_NMSE`；各任务的实际 `lambda=rho*c`。Loss为first10×6 motor MSE + lambda×future NMSE，future排除clamped frame0。Student使用自己的future。

## 两轮与选择

第一轮：rho0.1、0.5、0.7、1、1.5，各自相同初始化、Adam初态、lr0.0005、clip1、2epochs/16updates、冻结calibration顺序与seeds。匹配旧baseline的update1后一次Adam reset，其后resume完整保留Adam/RNG/cursor。

第二轮仍为三个点，其中rho0是micro初始化上的action-only对照，另外两个由第一轮selection胜出区间确定：

| 第一轮选中点 | 第二轮三个rho |
|---|---|
| rho0.1，或选中epoch0 | 0、0.03、sqrt(0.1×0.5)≈0.223607 |
| rho0.5 | 0、sqrt(0.1×0.5)、sqrt(0.5×0.7) |
| rho0.7 | 0、sqrt(0.5×0.7)、sqrt(0.7×1) |
| rho1 | 0、sqrt(0.7×1)、sqrt(1×1.5) |
| rho1.5 | 0、sqrt(1×1.5)、3 |

每轮候选均完成2epochs，不按epoch1提前淘汰。保存epoch0/1/2参数，CPU chooser对全部候选×epochs选点。完整两轮共八个训练候选，不添加非micro训练或互补判断。

20% guards以同selection的未训练micro parent为参考：late22平均RMSE、gripper平均RMSE、first10 motor per-case P90均不超过1.2×parent。P90使用四个seed-averaged case值上的线性插值。先过滤guards，再以first10 motor最低分为准；在最低分1%内优先较小rho，再较早epoch。另报告严格最小点，说明近似tie取舍。epoch0允许胜出，届时不声称发生latent训练增益。

搜索阶段只执行selection，不执行任何训练候选test。两轮完成后CPU冻结`optimal_selection.json`，最终GPU只恢复选中的epoch，评测calibration8与完整test44；不再训练。Parent test actions复用既有micro结果，selected future与BF16比较；没有parent test future时明确缺失，不额外生成44个baseline-only forward。

## 当前状态与证据边界

真实 CPU allocation 自检已通过（`25800841.pbs101`，Exit0）。共享 GPU reference 作业 `25802014.pbs101` 已完整通过 parent16/teacher60 与 selection action parity，冻结 `c=0.03033027132999191`。此前 `25800862.pbs101` 在 model forward 前因 JSON action 读取兼容问题失败，已修正。第一轮三个初始 allocation 均以 Exit0 保存 resumable checkpoint，分别完成12/16、11/16、11/16 updates；当前续跑 jobs 为 rho0.1→`25805582.pbs101`、rho0.5→`25805512.pbs101`、rho1→`25805536.pbs101`。两轮搜索与完整候选训练尚未结束。实际阶段和后续 job 记录见 [RUN_STATE.json](RUN_STATE.json)。当前权威配置是[protocol.json](protocol.json)，实现交接是[CONTRACT.md](CONTRACT.md)。初始DESIGN中的新增selection、5%guards、八点或互补实验建议已被上述用户配置替代。

数值/torch/.pt/model/绘图只在真实guarded PBS CPU/GPU allocation，login只轻量控制。GPU利用率/显存每15秒写job log；2h训练allocation可按已提交update/evaluation context安全接续，不重新初始化rho任务。Host-key verification保留、credentials不输出、banked reset绝不动用。不创建定时任务。

这是历史4-case selection上的超参选择、已见22-case test上的paired recovery；decoded-BF16 VQ/fake A4，不是native kernel、独立泛化或closed-loop证据。数字λ须等待micro calibration实测，不使用旧row-output parent的归一化系数。
