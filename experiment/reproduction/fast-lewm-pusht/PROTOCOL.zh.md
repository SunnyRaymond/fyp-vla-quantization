# Fast-LeWM pretrained PushT 初次复现

目标：用官方 released checkpoint 跑通 base Fast-LeWM 的完整 PushT dataset-source evaluation，核实实际动作接口和 closed-loop outcomes。主入口沿用官方 `eval.py::run`，不训练新模型。源码固定 `de3e9dac539f5bbe6ff1656a2fb00938d62a3c7d`，论文固定 arXiv:2606.26217v1；checkpoint 的 HF revision 和下载来源由 CPU preparation 记录。

## 冻结设置

- 官方 seed 42、50 个 dataset-source tasks、goal offset 25、eval budget 50、world max steps 100；图像 224。
- CEM 每轮 300 candidates、top-30、30 rounds、batch size 1、solver seed 42。
- outer horizon/receding horizon 1，5 action blocks，每 block 5 primitive actions；预计预测和缓冲执行均为 25 primitive steps。运行时核实 `PlanConfig`。
- base consistency weight 0；不运行 self-consistency variant 或 10/30 预算比较。
- task 选择、scalers、cost、CEM 更新和 buffered-action fast path 沿用官方代码。数据为现有 `pusht_expert_train.h5`，没有下载第二份数据。使用 whole-dataset scaler 是官方约定，不能宣称评估完全独立于训练数据。

## 兼容和记录

`run_reproduction.py` 包装官方入口：记录最终 config、模型来源、实际预测/执行长度、task 列表、逐 task success、每次 solve 的实际 active environment 数和同步时间。每个 candidate cost 检查 finite，因此这些耗时包含记录开销，不用于声称 speedup。

使用官方 requirements 中 `stable-worldmodel==0.0.6` 的隔离安装，保持共享环境不变。CPU job `25569206.pbs101` 已验证该 release 的 native `AutoCostModel`、`evaluate_from_dataset`、单 deque buffer 和官方 checkpoint CPU load。完整官方 checkout 自带 launcher/solver config groups；直接 compose 官方 config 并提供其结果路径所需的 Hydra runtime context，运行时逐项断言 frozen settings。最终 runner 沿用 native loader/evaluator/fast path，未启用 historical backend 的兼容适配。

本次复用环境的 Torch 为 2.8.0+cu128、torchvision 0.23.0+cu128、stable-pretraining 0.1.7、transformers 4.57.6、numpy 2.4.6、pymunk 7.3.0、gymnasium 1.3.0；它们并非官方 requirements 全部 exact pins。CPU config/load 已通过，GPU runtime 尚待验证；这是现有兼容环境中的 pretrained reproduction，不是 clean-environment exact dependency reproduction。

视频关闭。官方跨 replan cost 汇总使用 `np.stack`，在 active environments 减少时可能不成立，因此仅保留各 solve 的 cost 数组；这些 costs 不是物理 realized action quality。

CPU preparation 在 guarded CPU PBS allocation 内下载唯一 checkpoint、准备固定源码、检查依赖和 CPU object load；共享旧 venv 和源码不修改。GPU evaluation 在 guarded GPU PBS allocation 内加载数据、推理与 simulator evaluation，每五秒将 GPU utilization/显存采样写入 job log。Login 仅提交/状态/小控制文件。无需模型或数据 hash。

## 判据和解释

有效复现要求 checkpoint 可加载、所有评分/动作 finite、25-step 接口一致、完整 50-task outcomes，且 PBS/wrapper/runner 终态无未解释错误。论文 base PushT 报告 96%，此次结果与其做描述性比较。低成功率不会通过改 seed、tasks、budget 或 checkpoint 调参修复；应先查 runtime/protocol 差异。

这是单 seed 的 pretrained PushT 复现；不代表四任务论文复现、training reproduction、独立 test generalization 或 paper latency reproduction。50 source tasks 是 task 单位，300 candidates、30 rounds 和 telemetry samples 都不是额外独立样本。与历史 LeWM 结果仅做有边界的上下文对照，不能当作并发配对 timing baseline。

## 来源

- Gao & Xu (2026), [Fast LeWorldModel, arXiv:2606.26217v1](https://arxiv.org/abs/2606.26217v1)。[官方实现](https://github.com/Yuntian-Gao/Fast-LeWorldModel)，[官方权重](https://huggingface.co/naiverer/fast-leworldmodel)。
- 实验设计参考所用 `experimental-design` skill；Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, current v2 metadata checked 2026-09-26. https://doi.org/10.48550/arXiv.2609.00065
