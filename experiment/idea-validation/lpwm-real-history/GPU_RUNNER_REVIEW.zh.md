# GPU 阶段冻结执行记录（首个技术失败；最多一次经审阅的 retry）

## 首次提交终态

`25577811.pbs101` 已保留为技术失败：native capture 完成，16个context的history1 parity通过；随后runner因把packed candidate action末维预设为15而拒绝native实形状 `(300,5,10)`。该维度来自 pinned `dataset.action_dim=2 × frameskip=5`。未运行raw physics cost gate、primary或次级机制指标，因此不是机制NO_GO。Root授权按冻结bank与指标修正后审查，最多一次retry；不会自动重提。PBS源码请求8CPU/64GB，scheduler实际接受16CPU/110GB（同为1 GPU、90分钟），资源差异原因未核实。

资源提案：单节点1 GPU、8 CPUs、64 GB、90分钟。若 cost-only calibration 显示不足，停止并向 root 汇报实测资源预算，不自动延长、加卡、扩任务或缩小 bank。候选 bank 固定为300/context。

拟议 runner 先在 PBS 隔离目录写入 capture 与校准报告；机制质量结果只在成本门通过后产生：

1. 使用 allocation guard；核对 frozen commit/checkpoint/config 与 task identity。cohort 固定为原 formal 29 个 final-failure task indices 中升序前8个，planning seed 为99，`eval_seed=99*i+1` 中 i 是原始 0-based task index。config 使用顶层 `embed_dim=384` 与 `predictor._target_=LinearDynamicsPredictor, predictor.mode=mlp_var`。
2. 在相同 frozen task states/goals/checkpoint 上新跑 native cold-start diagnostic capture。参数不变：H=5、300 candidates、top30、30 iterations、`eval_every=1`、每轮 mean sequence 真实 environment evaluation 和原全成功早停。只记录 zero-based replan 1、2 且仍 active 的 contexts。每个 fixed bank=300：从 iteration0 按确定索引取100 early proposals；从最后实际评估 population 的270个 non-elites里按确定索引取170；加全部30 native objective elites。保留重复序列与 provenance。Instrument 只复制已生成张量/索引，不额外消耗 RNG；capture 是新诊断轨迹，不声称复现旧 outcome。
3. history1 parity gate通过后，选第一个有效 context 做唯一一次300-bank raw physics replay，以及cold/history predictor与真实未来encoder的分项计时。truth从原始init state/seed重放完整executed prefix后接candidate；不从缺失速度项的7D anchor reset。先落盘 `OUT/results/cost_calibration.json`。估算为 `1.2*(elapsed_through_calibration + 15*physics_context_seconds*anchor2_prefix_multiplier + 16*(cold_forward+history_forward+truth_encoder))+300s`；若首context是anchor1，对更长anchor2 prefix乘1.5。elapsed已经包含模型加载、native capture（包括 `eval_every=1` mean环境调用）和接口门；首context raw truth成本复用，模型前向不复用。若估算超过80分钟内部上限，写 `INCOMPLETE_RESOURCE_LIMIT` 并停止，不计算机制质量量，也不缩小bank。
4. 仅成本门通过后才从raw endpoints推导 primary cost=`(norm(goal_state[:4]-endpoint_state[:4])/20)^2+(wrapped_angle_error/(pi/9))^2`，另单独报告 T-block object-only 2D position error。两臂对相同candidate actions评分，报告模型选中top30真实平均cost与相对bank oracle-top30的elite regret。按task平均可用anchors，再计算 `(R_cold-R_history)/max(R_cold,0.01)`；cold regret<0.01为at-floor。Support条件为非at-floor任务median reduction≥10%且至少 `ceil(5/8*n_nonfloor)` 个task改善；样本不足12 contexts/6 tasks/6 nonfloor tasks记incomplete。closed-loop不在此job运行。GPU UUID/util/VRAM每5秒追加到该job的 `job.log`。

## 成本核算

不能把正式39分钟视为纯 CEM/model solve：native `eval_every=1` 会在每次 replan 的每个 CEM iteration 都把当前 mean action sequence送入真实 environment。成本外推包含该 native capture 的实测总耗时、唯一首context的300条prefix physics replay、冷/历史预测器和truth encoder三项计时、15个剩余物理context并用1.5倍因子覆盖anchor2较长prefix，以及16个未保留的quality forward组、20%和300秒stageout/teardown余量。校准仅返回 raw endpoint/future frame并做必要 shape/finite gates；不计算 position/angle cost、candidate ranking、regret、success fraction或response质量。超过内部80分钟上限时只报告资源不够，不进入机制评分，绝不偷偷减少bank。

## 接口、环境身份与停止门

- History=1 必须与 official cold path 在 action window、rollout shape、输出索引和预测数值上一致。真实 model encoder 当前对 CEM batch encode；接口比较保持 native batch geometry，禁止未经验证地编码单帧再 broadcast。
- 记录 encoder/projector/predictor/action encoder 的实际 `.training` flags；沿用 native checkpoint/constructor mode，不额外 `.eval()` 后声称完全 native。两臂 flags 一致。
- History=3 用三个 frameskip=5 决策帧、两段已执行 action blocks、5个未来 blocks；总输入7 actions、模型输出8帧，取 `rollout[:,2:]` 作为 current+5 future；物理执行仍只有5个未来 blocks。
- CPU prep 重建的 dataset sampling 与 `env_info.shape` 必须和 frozen target 的 init/goal observations/states一致。GPU capture 在每个 context 核验重放 prefix 后 observation/state 与 capture；真值从同 init/seed 重新 replay prefix 后接 candidate。保存的7D task state含 agent x/y、block x/y、angle、agent vx/vy，不含 block linear/angular velocity。
- 同一 task 的 seeds 按原50-task index映射，不能按抽样批次重新编号。`state_0/state_g`、sidecar、goal 与 source/checkpoint不一致，history1 parity失败，past-action alignment错误，instrument改变 RNG/行为，physics replay不能重现prefix，或 timing 超cap时即停止。
- 每个context的NPZ保存完整prefix actions、past action blocks、history indices、三帧raw history observations、current state与eval seed。最终汇总固定为 `OUT/results/run_summary.json`；PlanWorkspace的cwd相对target dump和native log都留在OUT，不写共享checkout。

root 已审阅当前 PBS wrapper 与 runner，并授权且只授权提交这一次机制 GPU job。没有授权自动重提或 closed-loop；即使机制 primary gate 支持，也须另行提交完整50-task paired closed-loop protocol 和资源申请。
