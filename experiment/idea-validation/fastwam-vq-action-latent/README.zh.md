# Fast-WAM VQ：action + future latent 第一步验证

只改变一个训练目标：在原 final-action distillation 的 first10 motor MSE 上增加 BF16 future preservation。Student 仍使用自己生成的 future，不在训练输入中注入 teacher future。保持旧实验、captures 和结果不变。

## 冻结的比较协议

- 从同一 `row_output_vq_a4` parent 初始化，训练全部 614 个 target Linear 的 codebooks 和 row scales；indices、Smooth/符号/H128、A4、biases 和其他参数固定。
- Calibration cases `[0,2,4,6,10,12,14,16]`，seed index `[0]`。2 epochs，每 epoch 8 updates，共 16 updates。
- Selection cases `[1000,1001,1002,1003]`，seed indices `[0,1]`。保留 epoch 0 初始化与完整 epochs，按 case-mean first10 motor RMSE 选 checkpoint，tie 优先较早 epoch。
- Test cases `1004..1025`，seed indices `[0,1]`，全部 22 cases / 44 contexts；先冻结 checkpoint 再访问 student test forwards。最后也保留完整 calibration 8 contexts。
- 与原实验相同的 observations、sampler seeds、32 action horizon、9 video frames、10 video + 10 action denoising steps、sigma shift 1、CPU RNG 和 compile=false。
- Adam lr `0.0005`、clip norm `1`。旧 action-only 实验第一 update 后因历史断点恢复清空 Adam 一次；本轮明确匹配该 schedule：第一 committed update 后清空一次，所有后续 resumes 完整恢复 moments/RNG/cursor，避免额外 optimizer confound。

## Loss 与 references

`L = first10×6 motor MSE + w × future NMSE`。Future 指 video denoising 的最终 clean latent endpoint，排除 observation-clamped frame 0，不是只监督视频最后一帧。第一轮不加 temporal-delta loss 或 λ sweep。

BF16 actions、BF16/parent final future latents 复用已完成 `fastwam-vq-latent-diagnostic-20261013` 的 60 个 paired contexts。CPU allocation 准备 cache，核对 input/seed/action parity、finite values、形状和两 shards 完成凭证，不跑模型。实际 `w = 0.1 × mean_8cal_initial_motor_MSE / mean_8cal_initial_future_NMSE`，在更新前冻结；其含义是初始化 calibration 均值上 latent 项的 loss 数值为 action 项的 10%，不是梯度贡献的 10%。不使用 selection/test 调整 w。

梯度 sampler 的默认 action 返回接口保留；latent 使用与原 forward 相同的最后 video scheduler output 保持 autograd 连接。训练前检查初始化 action/future parity，首次 checkpointed gradient forward 再检查 parity。训练 history 分开保存 action MSE、future NMSE、scaled latent loss、total loss 和实际 w。

## PBS 与监控

Remote root：`/scratch/users/ntu/yguo017/fastwam-vq-action-latent-20261014`。

先运行 CPU `prepare.pbs`，通过 reference cache/self-check 后才提交 GPU `gpu.pbs`。GPU 每段两小时，单 GPU、16 CPU、110 GB host memory；每 15 秒在 job log 记录利用率和显存。每完成一个 update/evaluation context 原子保存参数、Adam/RNG/cursor/evaluation history，临近上限写 `RESUME_READY`，只从核实的 terminal checkpoint 接续。Queue comments 或查询失败不授权重提。

2026-10-14 已按用户要求删除固定监控的 heartbeat，不再修复或重建它。原对话继续负责剩余评测和最终解释；查询失败只表示状态不确定，不据此重提作业。

CPU reference job `25793095.pbs101` 已完整通过，60 contexts 的 BF16/parent actions 与原实验 max_abs/RMSE 均为 0，model forwards 为 0；冻结 `w=0.0029918822946277566`。首段 GPU `25793387.pbs101` 完成 2 epochs /16 updates，并在 epoch2 selection 阶段保存安全断点。接续作业 `25798439.pbs101` 已 F/Exit0 完整完成 selection/calibration/test：选中 epoch 2，calibration 8 contexts、test 22 cases /44 contexts 全部齐全，completion marker、artifact exit 与冻结协议验证通过。

最终 CPU `analysis.pbs` 汇总同 cohort 的全部 7 个 source reference arms，并对 parent、action-only、新 action+latent checkpoint 做配对 action/future 比较。读取完整 checkpoint 的跨 allocation `evaluation_rows`，不以最后一段 actions.jsonl 代替全部评测；汇总不新增 model forward。

CPU 汇总 `25799647.pbs101` 已 F/Exit0 完整通过。详细结果在 [REPORT.zh.md](retrieved/results/analysis-25799647.pbs101/REPORT.zh.md)，逐 case 数值在 [paired_cases.csv](retrieved/results/analysis-25799647.pbs101/paired_cases.csv)，结构化来源与梯度记录在 [analysis.json](retrieved/results/analysis-25799647.pbs101/analysis.json)。这里的“0 new model forwards”专指 CPU 后处理；GPU 作业已完成实际训练和 student 评测。

易懂解释见 [INTERPRETATION.zh.md](INTERPRETATION.zh.md)。额外 CPU 后处理 `25799776.pbs101` 的 [尾部敏感性表](retrieved/results/tail-25799776.pbs101/tail_sensitivity.zh.md) 表明：相对 action-only 的 full32/late22 平均改善很依赖 case 1016；省略它后的 21-case 视图均值反而更差。敏感性视图不替换完整 22-case 主结果，也未用于训练或 checkpoint selection。该后处理无新 model forwards。

新版本 test first10/full32/late22 motor RMSE 为 `0.052951010 / 0.061331654 / 0.063866255`，均低于 parent 和旧 final-action distilled 均值；future latent NMSE 为 `0.071786944`，低于两者的 `0.084435442 / 0.093010997`。但 late22 相对旧 action-only 只有 10/22 cases 改善，gripper 平均误差略升，Scalar W4A4 与 row-scale VQ 的 full32/late22 仍更低。因此本轮支持把 latent preservation 作为辅助目标，不能声称所有 action 指标全面胜出，也不能用平均 future NMSE 替代 action checkpoint selection。

Login node 仅轻量状态和小控制文件操作。所有 torch、模型、数值计算及 `.pt` cache/checkpoint 读写只能在真实 PBS compute allocation；host-key verification 保留，绝不输出 credentials 或使用 banked reset。

## 证据范围

评估 action first10/full32/late22/gripper，以及 BF16 future RMSE/NMSE、latent temporal-delta；同 case 先平均 seeds，再对 cases 等权。与已有 VQ、parent 和 action-only epoch2 做 frozen paired 比较。已有 test cohort 已看过，结果是 recovery diagnostic，不能声称新的独立泛化证据。执行仍为 decoded BF16 VQ、BF16 KV；无 closed-loop episodes 或 native-kernel 加速结论。
