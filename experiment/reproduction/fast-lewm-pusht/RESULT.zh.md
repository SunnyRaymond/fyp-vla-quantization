# Fast-LeWM pretrained PushT 复现结果

**官方 base checkpoint 已跑通完整 50-task evaluation：49/50 成功，98%。** 数值与接口检查全部通过，可以将它作为后续本地比较的 pretrained baseline。论文 base Fast-LeWM 的 PushT 报告为 96%；本次单 seed、现有兼容环境的 98% 不能解释为显著优于论文，也不证明精确重现了作者的任务名单或完整实验环境。

## 实际复现的范围

使用官方 `eval.py` 的 task selection、pixel transform、whole-dataset scalers、native checkpoint loader、CEM、buffered-action fast path 和 `World.evaluate_from_dataset`。只关闭视频、增加结果与 finite 检查、提供 Hydra output context，并保存逐 solve 的 cost arrays。没有重新训练，没有修改 learned model、task seed、candidate budget、goal offset 或 task success definition。未运行 self-consistency variant、其他三个环境或低预算 CEM sweep。

| 项目 | 实际值 |
|---|---|
| GPU job | `25569241.pbs101` |
| 硬件 | NVIDIA A100-SXM4-40GB，host `x1000c0s1b0n1` |
| 官方 Fast source | `de3e9dac539f5bbe6ff1656a2fb00938d62a3c7d`，固定 GitHub archive |
| 官方 HF release | `naiverer/fast-leworldmodel`，revision `f95379fe193c8bfc6a59c9d8437d5052bd72ff71` |
| Checkpoint | `Fast-lewm_pusht_object.ckpt`，71,897,531 bytes，17,913,184 parameters |
| Dataset | 现有 `pusht_expert_train.h5`，由 job-private symlink 引用，未重新下载 |
| Task selection | 官方 seed 42，50 个 start rows，本次来自 50 个不同 source episodes |
| Goal / evaluation | goal offset 25，eval budget 50，world max steps 100 |
| CEM | 300 candidates，top-30，30 rounds，batch size 1，solver seed 42 |
| Action interface | outer horizon 1 × 5 blocks × 5 steps；预测并缓冲执行 25 primitive actions |
| Self-consistency | weight 0，base model |

唯一未成功项是 task index 5：source row 221571、episode 1820、start step 13。其余 49 项成功。任务逐项身份和 outcomes 保存在原始 JSON 中，没有重新抽样或补跑失败任务。

## 运行证据

- `stage1_gate.json=PASS`，`run_summary.json=COMPLETED/PASS`；完整 50 项 `episode_successes`，native `success_rate=98.0` 与计数一致。
- 3000 次 candidate cost 调用，共评分 900,000 个候选；所有 cost 与 solver actions finite。
- 实测候选 shape `(1,300,1,50)`，cost shape `(1,300)`，predicted latent shape `(1,300,2,192)`。最后一个 latent 是 25-step terminal prediction；另一个位置是 initial latent。
- 两次 solve 均实际处理 50 environments，输出 actions shape `(50,1,50)`。最终 `PlanConfig` 为 horizon 1、receding horizon 1、action block 25、history 1、warm start true。官方 fast path 启用，loader/evaluator 不需要兼容替换。
- PBS `F / Exit_status=0`；`job_status.txt` 和 `final_exit_status.txt` 均为 wrapper/runner exit 0。PBS 同时报 `Stageout_status=1`，具体原因未查明；scratch 内结果、状态和日志完整可读，小型证据已取回本地。这不是模型或 runner 失败，也不应省略 stage-out caveat。

每五秒采集一次 GPU utilization/显存，job log 共 80 条采样；采样最大利用率 48%，最大显存 627 MiB。这是采样值，不能作为精确瞬时 peak。

## 延迟边界

两次同步 `solver.solve` 时间为 **36.09 s / 23.32 s**，native evaluation 耗时 **68.98 s**，runner 内计时 **260.59 s**，PBS walltime **6 min 40 s**。runner 内时间包括数据与模型准备；PBS 还包含 heavy imports 等启动工作。

这些是单次运行的描述性时间：第一个 solve 含冷启动，每次 candidate cost 还增加 finite 检查与同步开销。没有 concurrent LeWM control、平衡 arm order 或重复稳态时间样本，所以本次**不声称复现论文的 3.9× dynamics 或 48% full-CEM speedup**。两次 solve 都处理 50 environments，也不能把它们直接当作单机器人 latency。

## 依赖与泛化边界

`stable-worldmodel==0.0.6` 已按官方要求安装在独立 overlay，共享环境保持不变。其 native evaluator 与 loader 已运行验证。其余关键版本为 Torch 2.8.0+cu128、torchvision 0.23.0+cu128、stable-pretraining 0.1.7、transformers 4.57.6、numpy 2.4.6、pymunk 7.3.0、gymnasium 1.3.0；这些并非作者 requirements 全部 exact pins。因此这是现有兼容环境中的 pretrained PushT reproduction，未证明 clean-environment dependency reproduction。

这是一个 solver/task-selection seed 的 dataset-source evaluation；task starts 和 goals 来自 expert training dataset，scalers 也沿用 whole dataset。它不支持独立 test generalization、多 seed 稳定性、四任务平均成功率或 training reproduction 的结论。候选、CEM rounds、GPU samples 不是额外独立任务。

## Preparation 失败与修复

| CPU job | 结果 | 说明 |
|---|---|---|
| `25569105.pbs101` | F / exit 1 | compute node 没有 git，在源码/模型 I/O 前退出 |
| `25569145.pbs101` | F / exit 1 | fixed source 和 isolated backend 已备齐，preparation 脚本的 shutil variable scope 错误导致退出 |
| `25569206.pbs101` | F / exit 0，CPU PASS | 复用已备材料，下载固定 HF checkpoint，验证 object load、官方 config 与 runtime interfaces |

下载、解包、安装、CPU model load、GPU 推理和 simulator evaluation 全部位于获批 allocation 内；没有退回 login node 做这些工作，没有重新训练或扩大原失败方案。CPU 成功作业也报 `Stageout_status=1`，scratch 证据已取回。

## 下一步

本次已满足“先试跑官方 pretrained Fast-LeWM + PushT”的目的。后续应另行冻结同硬件、相同任务和 actual active-load 下的 LeWM/Fast-LeWM 比较，再讨论低 CEM budget 的质量与 latency frontier。本次不自动启动那组实验。

## 原始证据与来源

- [GPU run summary](artifacts/25569241.pbs101/run_summary.json)、[validity gate](artifacts/25569241.pbs101/stage1_gate.json)、[tasks](artifacts/25569241.pbs101/selected_tasks.json)、[runtime contract](artifacts/25569241.pbs101/runtime_contract.json)、[job log](artifacts/25569241.pbs101/job.log)。
- [CPU checkpoint/runtime preparation](artifacts/25569206.pbs101/prepared.json)、[frozen protocol](PROTOCOL.zh.md)、[submission record](SUBMISSION.json)。
- Gao & Xu (2026), [Fast LeWorldModel, arXiv:2606.26217v1](https://arxiv.org/html/2606.26217v1)，[official code](https://github.com/Yuntian-Gao/Fast-LeWorldModel)，[official release](https://huggingface.co/naiverer/fast-leworldmodel/tree/f95379fe193c8bfc6a59c9d8437d5052bd72ff71)。
- 实验设计参考所用 `experimental-design` skill：Kassis et al. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [DOI](https://doi.org/10.48550/arXiv.2609.00065)，current v2 metadata 已核对。
