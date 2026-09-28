# LpWM sparse 单臂结果与 insights

2026-09-27，用户通知 sparse 完成后补录。**`25571461.pbs101` 完整运行有效：原生评估成功 21/50（42%）。** 这是一个官方 sparse cell 的端到端运行结果；dense 已取消，论文的 sparse-over-dense 比较主张仍未检验。原 goal 保持结束，不重提 dense 或 PLDM，不启动新实验。

## 运行与证据

| 项目 | 结果 |
|---|---|
| PBS | `F`；`Exit_status=0`；`Stageout_status=1`；walltime `08:07:51` |
| Train / plan | 两个命令均退出 0；run manifest 为 `PASS` |
| 完整训练 | 2 epochs，共 61,930 updates；每轮 34 validation batches；latest 与 epoch-2 checkpoints 存在 |
| 原生评估 | 50 ordered outcomes，21 success、29 failure；final rate 与布尔向量一致 |
| Task identity | 原生 `state_0/state_g` sidecars 均为 `[50,7]`，已保存 |
| GPU | NVIDIA A100-SXM4-40GB；作业内约每 5 秒记录利用率与显存；启动 UUID 与 Torch `cuda:0` 对应 |
| Torch peak allocated | train（包含 validation）30,329,685,504 bytes；plan 6,015,282,688 bytes；均非 predictor 独占显存 |

源码固定为 `YilunKuang/lpworldmodel@bdd812d9432cccda8c350086006401b436f91982`。从零训练 ViT CLS encoder，`mlp_var`、D=384、history=3、frameskip=5、training seed=0；sparse recipe 为 RepReLU、RDMReg rectified Laplace (`p=1,mu=0`)、regularizer weight=0.1、muP LR=5e-4。该 cell 对应论文的中间容量 MLP ∘ LTI(k) 路线。

评估使用官方 `scripts/plan.sh plan_lewm.yaml <run> latest 50 10`，planning seed base=99、dataset goals、goal H=5。每次 CEM 为 300 candidates、top-30、30 inner steps；外层 MPC 最多 10 次，每次执行 5 model actions（25 primitive steps）。**外层 MPC 的 10 与内层 CEM 的 30 是不同预算。**

训练日志：epoch 1 train/validation loss 为 0.0156/0.0132，epoch 2 为 0.0102/0.0115。训练目录首条日志到最后 checkpoint 约 7 小时 24 分钟；native planning 输出目录创建到作业结束标记约 39 分 28 秒。后者包含环境执行、评估、绘图及 I/O，不是纯 CEM solve latency。单 case 校准的 50×成本外推曾约 2.84 小时；它没有准确预测此次完整评估时间，也不能把两者比值称作 speedup。

## 与论文对应 cell 的描述性参照

[论文 arXiv v1](https://arxiv.org/pdf/2608.22764v1) Appendix H.1、Figure 9（印刷 p.23）中，**closed-loop / D=384 / MLP ∘ LTI(k) / sparse LpWM / λ=0.1 / μP LR=5e-4** 的格子为 **0.31 ± 0.02**，即 31% ± 2%；caption 的 ± 是三个 planning seeds 的 std。对应 dense tuned cell（λ=0.01、LR=5e-5）为 13% ± 2%，仅为论文报告，不是本轮运行结果。可查看 [对应图格截取](artifacts/25571461.pbs101/paper_fig9_closedloop_D384.png)；超参来自 [pinned reproduction script](https://github.com/YilunKuang/lpworldmodel/blob/bdd812d9432cccda8c350086006401b436f91982/scripts/reproduce_pusht.sh)。

本轮单 planning seed 的 42% 比论文三 seed 均值高 11 个百分点，但未复现该三 seed 均值；论文未公布那三个 planning seed ID 或逐项 task states，无法确认本轮 seed 99 是否在其中或任务逐项一致。该差异不能称为算法改进，也不能仅凭此认定复现冲突或一致。它说明应按这一个小 predictor cell 的报告范围判断，而不拿另一模型/harness 约 95% 的成功率作直接基准。差异来源仍未识别。

## 这次新增的机制信号

| 外层 MPC 次数 | 成功任务数 | native mean_state_dist | native mean_div_visual_emb |
|---:|---:|---:|---:|
| 1 | 15/50 | 128.16 | 29.28 |
| 2 | 20/50 | 144.22 | 46.25 |
| 3 | 20/50 | 148.74 | 63.29 |
| 4 | 20/50 | 160.55 | 77.45 |
| 5 | 21/50 | 165.37 | 88.14 |
| 6 | 21/50 | 169.06 | 93.23 |
| 7 | 21/50 | 174.00 | 102.52 |
| 8 | 21/50 | 192.18 | 111.14 |
| 9 | 21/50 | 191.90 | 118.84 |
| 10 / final | 21/50 | 203.51 | 128.80 |

**1. 外层追加预算在这条轨迹上出现平台期。** 前两次达到 20 个成功任务，第 5 次新增 1 个，后五次没有新增成功。它支持把“识别长时间无进展的失败任务”列为效率诊断方向。但这是同一次运行的事后前缀观察，不能声称减少 MPC 上限在其他 seeds 无损，也没有测得省时比例；更不能推导减少内层 CEM steps 同样无损。这里保持原预算，未做提前停止实验。

**2. One-step 训练进步没有建立长 rollout / 控制充分性。** 两轮 validation loss 降低，完整任务成功率仍是 42%。执行过程中 imagined-vs-real latent discrepancy 随更长前缀增大，这提示应检查失败轨迹的 action response、长 rollout 累积误差与 goal-cost 对齐，而不是仅凭 validation loss 判断模型可用于规划。尚未区分预测失真、latent objective、搜索不足、接触状态及 seed variation 的贡献，不能断言某一个是根因。

这里引用的是 trainer 的日志 loss，未把 prediction / regularizer 等 loss components 单独取出，也未评估 epoch-1 checkpoint 的任务成功率；不能从两轮 loss 变化推出控制能力在训练中提升或下降。

进一步核对 pinned source 后，训练 loss 的具体形式为：4 帧真实观测经同一个 encoder 编码，对前3帧的每个 causal position 做 next-step latent MSE，再加 `0.1 * RDMReg`；`detach_target=false`，target 侧 encoder 也回传梯度，没有 decoder loss，额外 variance/covariance weights 均为0。本轮是 joint representation/dynamics training，而不是固定 LeWM teacher 的 distillation。

**新增、尚未实验验证的 history 诊断点：** `num_hist=3` 是 predictor 的最大 history。`plan.py` 创建的 initial observation 只有1帧；`planning/mpc.py` 每次重规划同样只传入执行序列最后1帧。`_rollout_adaln` 因此从1个真实 latent 开始，以自己的预测逐步补出2/3帧 history。该路径只使用 visual latent，不将 dataset proprio/7D state 喂入 predictor。训练的3个 causal positions本来就包含有效history为1/2/3的情况，因此不能仅凭此称它为bug、证明分布错误或解释42%的失败。可检验的假设是：重规划时保留已有的真实近期观测及正确对齐的历史 action blocks，是否改善相同执行状态下、相同未来H=5候选的预测和真实任务结果。直接prepend观测会改变现有action/输出索引，不能作为有效对照；未来动作窗口与执行步数必须保持一致。当前仅作源码解释，没有运行该变体。

证据：[本次实际 Hydra config](artifacts/25571461.pbs101/hydra.yaml)、[world model](artifacts/25571461.pbs101/visual_world_model.py)、[predictor](artifacts/25571461.pbs101/infojepa_modules.py)、[target preparation](artifacts/25571461.pbs101/plan.py)、[MPC](artifacts/25571461.pbs101/mpc.py)。

指标定义必须保留：`mean_div_visual_emb` 实际为 `torch.norm(encoded_real_visual - imagined_visual).item()`，是整个 batch embedding tensor 的 norm，并不是每 task 的 mean，也不是到 goal 的 CEM objective。不同 MPC 阶段的 action prefix 长度和 success masks 不同，因此这些数字不能当作固定 horizon 下严格配对的预测误差。`state_dist` 对包含位置、角度和速度的 7D state 直接取 norm；它不等于 success predicate，也不适合单独作为提前停止规则。官方 success 要求 agent + T-block 的前四个位置坐标合并距离 <20，且角度差 <π/9；不能将这里的 42% 与其他 PushT harness 的 coverage-based 成功率直接比较。

**3. 现在有一个可用于后续诊断的官方小 predictor checkpoint，但没有 sparsity gain 或 runtime gain 证据。** Dense control `25571463.pbs101` 排队期间已取消，不能估计同 cell 的 sparse/dense 差值，也不能把 42% 与本项目 LeWM/FastLeWM 的约 94–97% 直接相减：目标、history、模型、训练 recipe 和评估 harness 不同。论文的 representation sparsity 用来降低所需 predictor complexity，不会自动使 dense tensor kernels 更快。官方训练代码会计算并记录 `l0_frac`（非零坐标比例）；本次尚未提取分析这些 active-fraction 结果，也未分析 support stability，未跑稀疏 kernel 或配对 latency benchmark。

如果未来继续，最有信息量的是利用这 29 条失败任务，先核对模型认为更好的 actions 是否在真实环境中更接近目标，并分开衡量固定 horizon 的 rollout discrepancy 与 task-relevant support changes；这些是候选诊断，不是本轮已完成的实验，也不是已确认的新 research gap。当前不扩展实验。

## 原始证据与结论边界

- [Terminal PBS](artifacts/25571461.pbs101/pbs_terminal.txt)、[run manifest](artifacts/25571461.pbs101/run_manifest.json)、[native outcomes](artifacts/25571461.pbs101/final_eval_outcomes.json)、[actual target states](artifacts/25571461.pbs101/final_eval_target_states.json)。
- [Native MPC metrics](artifacts/25571461.pbs101/native_plan_logs.jsonl)、[完整 planning log](artifacts/25571461.pbs101/plan.log)、[training metrics](artifacts/25571461.pbs101/training_metrics.log)、[结果摘要](artifacts/25571461.pbs101/SUMMARY.json)。
- 指标语义来自 pinned source 的 [MPC](artifacts/25571461.pbs101/mpc.py)、[evaluator](artifacts/25571461.pbs101/evaluator.py)、[PushT wrapper](artifacts/25571461.pbs101/pusht_wrapper.py)。完整 GPU telemetry 和 checkpoint 留在远端，未下载模型或大日志。

这是一个 training seed 和一个 planning seed base 的结果；50 个 evaluation tasks 来自只有 21 条轨迹的 validation split，可能共享轨迹，因此不作 50 IID trials 的总体置信区间。尚未复现完整论文网格、三 planning-seed 均值、多 training-seed 稳定性或 sparse/dense 因果效应。终态 PASS 表示该 sparse 单臂完整跑通，不表示论文主要比较 claim 已通过。

远端作业目录：`/scratch/users/ntu/yguo017/experiment/reproduction/lpwm/formal_runs/25571461.pbs101/sparse`。Dense 的取消和此前 calibration 历史保留在 [SUBMISSION.json](SUBMISSION.json)。
