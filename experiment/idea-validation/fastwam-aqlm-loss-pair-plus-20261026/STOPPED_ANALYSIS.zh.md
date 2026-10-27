# AQLM loss 消融：停止后的结果与训练诊断

已按用户要求停止闭环，保留全部已完成 episodes、共享初始化和两臂 checkpoints。可以确认 ours 在校准数据上的拟合也明显退化，问题已出现在训练阶段；目前尚未定位到一个得到验证的梯度实现 bug。最有证据支持的排查方向是 codebook 更新步幅与参数尺度的匹配。

## 1. 停止状态与证据范围

- 闭环作业 `26010239.pbs101` 已取消并确认 PBS `F / Exit_status=141`，实际运行 `09:33:11`。这是用户主动停止，不是模型报错。
- Reconstruction 完成 280/280；ours 完成 245/280；合计保留 525/560 episodes。剩余 35 episodes 未执行完，不计为失败。
- 监控已停止，不自动恢复剩余 episodes，不提交新训练。
- 本次分析使用 CPU PBS `26020010.pbs101`，已确认 `F / Exit_status=0`，运行 `00:02:37`。只读取缓存预测、episode 记录和已有参数：**新增训练更新 0，新增模型 forward 0**。

这里的 reconstruction 是使用 pinned official AQLM 初始化/decoder、迁移到 Fast-WAM 的 Linear-output reconstruction 对照，不是原封不动运行官方 LLM 训练流程。两臂均为 decoded BF16 dense weights、dynamic fake A4、BF16 KV；不涉及 native low-bit kernel 性能结论。

两臂共享全部 614 个 AQLM g8/M4/K256 banks、冻结 rotation/indices、batch schedule、八次 Adam 更新及 LR=5e-4。共享初始化按用户要求保留了已完成拟合：22 个 heavy、12 个 light、580 个 minimal banks；它们在两臂完全相同，不能单独解释两臂分化。

## 2. 闭环结果

完整 reconstruction：**202/280 = 72.14%**。

严格使用双方都完成的相同 245 个 episode 比较：

| 指标 | Reconstruction | Ours |
|---|---:|---:|
| 成功数 / 配对 episode 数 | 179/245 | 125/245 |
| 成功率 | 73.06% | 51.02% |
| Ours 相对成功率差 | — | **−22.04 个百分点** |

逐 episode 配对：ours **5 胜、59 负、181 平**。其中“胜”表示只有 ours 成功，“负”表示只有 reconstruction 成功。

| Suite | 共同完成数 | Reconstruction | Ours | Ours 差值 |
|---|---:|---:|---:|---:|
| libero_spatial | 70 | 55/70 = 78.57% | 46/70 = 65.71% | −12.86 pp |
| libero_object | 70 | 53/70 = 75.71% | 42/70 = 60.00% | −15.71 pp |
| libero_goal | 70 | 46/70 = 65.71% | 25/70 = 35.71% | −30.00 pp |
| libero_10 | 35 | 25/35 = 71.43% | 12/35 = 34.29% | −37.14 pp |

这是中途停止后共同完成子集上的描述性结果，245 条整体成功率按 episode 汇总。libero_10 只完成一半，不能将其解释为完整四个 suite 等权的 280-episode 结果，也不作显著性或无偏总体效应推断。

## 3. 不是只有泛化差：校准集本身就没有拟合好

从训练前已经缓存的共享初始化预测恢复完整 C80 指标，再与 final8 的相同 80 个 context / seed0 比较。四个 suite 各 20 条；数值均为误差，越低越好。

| C80 指标 | 共享初始化 | Reconstruction final8 | Ours final8 | Ours 相对初始化 |
|---|---:|---:|---:|---:|
| first10 motor RMSE | 0.048906 | 0.048310 | 0.118843 | **+143.00%** |
| first10 motor MSE | 0.002532 | 0.002418 | 0.017793 | +602.64% |
| future latent NMSE | 0.057176 | 0.061066 | 0.250011 | **+337.27%** |
| mixed loss：A + λZ | 0.004960 | 0.005010 | 0.028407 | **+472.76%** |
| full32 motor RMSE | 0.050079 | 0.050391 | 0.121508 | +142.63% |
| first10 gripper RMSE | 0.049137 | 0.034631 | 0.154005 | +213.42% |

Ours 相对初始 checkpoint：**80/80 context 的 mixed loss 变差，79/80 的 first10 motor RMSE 和 future NMSE 变差**。Reconstruction 则总体接近初始化，first10 motor RMSE 略有改善。

Held-out 的 40 contexts × 2 seeds 也一致退化：

| Held-out 指标 | Reconstruction | Ours | Ours 相对变化 |
|---|---:|---:|---:|
| first10 motor RMSE | 0.051755 | 0.116874 | +125.82% |
| future latent NMSE | 0.069989 | 0.239592 | +242.33% |
| mixed loss：A + λZ | 0.005763 | 0.025813 | +347.93% |

Held-out 先在每个 context 内平均两个 seeds，再按 suite 等权汇总；这三项 ours 均为 0/40 context 胜。结果支持“当前训练方案破坏了原有初始化”，而不只是 held-out 过拟合或闭环环境不适配。

## 4. 第一次更新后就出现大幅漂移

历史每个 update 使用不同的 10-case batch，记录的是更新前 loss，所以不能把八个原始数值直接画成同一批数据的收敛曲线。本次改为：对每个 batch，比较该批输入在初始 checkpoint 的缓存指标与其实际更新前指标。

| 即将执行的 update | 已完成更新数 | 含 KL 总 loss / 相同 cases 初始值 | Motor MSE / 相同 cases 初始值 |
|---:|---:|---:|---:|
| 1 | 0 | 1.000× | 1.000× |
| 2 | 1 | **6.030×** | **8.335×** |
| 3 | 2 | 4.543× | 5.995× |
| 4 | 3 | 4.988× | 5.477× |
| 5 | 4 | 3.943× | 4.760× |
| 6 | 5 | 2.736× | 2.355× |
| 7 | 6 | 3.700× | 3.392× |
| 8 | 7 | 5.155× | 5.017× |

总 loss 为 `A + 0.04245452683169643 Z + 0.005507108948679781 KL`。在第一次更新之后，第二批 10 个 inputs 上的总 loss 从其初始值 **0.004973 升到 0.029988**，motor MSE 从 **0.002427 升到 0.020230**。

这证明大幅漂移很早就发生，且不是拿不同 batch 的原始 loss 硬比较造成的。不过它不是第一批训练样本的逐步 loss 曲线，不能据此声称第一步沿真实 loss 的上升方向更新；真实方向还需要固定 batch 的验证。

## 5. 参数变化：主要异常在 codebooks

下表相对变化定义为 `||final FP32 master − initial||₂ / ||initial||₂`。它衡量参数漂移，不直接等于 decoded weight 或输出误差。

| 参数漂移 | Reconstruction | Ours |
|---|---:|---:|
| 全部 codebooks 相对 L2 | 2.057% | **8.903%** |
| 每层 codebooks 相对 L2 的中位数 | 2.795% | **13.279%** |
| 每层 codebooks 相对 L2 的最大值 | 8.455% | **37.139%** |
| 全部 raw row scales 相对 L2 | 0.0364% | **0.1022%** |
| BF16 部署 scales 相对初始值的 L2 | 0.00326% | **0.03656%** |

Ours 的 codebook 全局漂移是 reconstruction 的 **4.33 倍**。最大两层是 `video_expert.blocks.0.ffn.2`、`video_expert.blocks.1.ffn.2`，相对变化约 37.14% / 36.91%。

Ours codebook 更新 RMS 为 **0.001530**，master 到 BF16 部署的舍入误差 RMS 为 **0.00002824**。后者远小于实际参数更新，单靠最终 codebook 存储舍入难以解释退化。激活 fake quant / STE 的梯度近似仍未得到真实方向验证。

BF16 部署的 row scales 没有符号翻转，FP32 masters 的整体变化也很小，现有证据不支持将“scale 爆炸”列为首要解释。

两臂都实际完成 8 updates、80 case exposures。最终 optimizer 的 1,228 个参数状态均为 Adam step=7，与“第一步之后清空 Adam state，再执行七步”一致；源码也明确实现该 reset。Reconstruction 的续跑正确保留了已提交的第一步，不能把差异简单归因于作业重启或少训练一步。

## 6. 已排查的实现与仍未证明的原因

源码审查没有发现明显的 loss 符号、batch 除法、reader cotangent 重复添加或 decoder 索引错误：

- A 是 first10 motor MSE，Z 是排除 frame0 的 future NMSE，系数均为正；KL 使用 teacher-detached 的 teacher→student 方向。
- A/λZ 的 cotangents 与 βKL 对 future 的 VJP 合并后，只在最终 backward 除以 batch size 10 一次。
- 614 个目标模块均有 finite/connected gradients；staged student replay 的 action/future 最大误差为 0。
- 闭环四个 worker 与对应 final8 source anchor 的 action/future replay 均为 0 误差，支持评估部署加载的是预期参数。
- Frozen reader 使用独立 BF16 masters，成功路径恢复 student 权重；此前 OOM 的未提交更新没有进入最终 checkpoint。

这些检查只排除若干具体错误，**不证明完整大模型参数梯度的数值方向正确，也不证明 STE 能提供有效下降方向**。

目前按证据排序：

1. **优先检查 codebook 的有效更新步幅。** 同一个 Adam LR=5e-4 不能保证两种目标具有相同参数漂移。Ours 在第一步后已明显恶化，八步后 codebook 漂移大得多；固定绝对 LR 对初始 RMS 较小的 codebooks 尤其值得检查。这是得到数据支持的候选原因，尚非因果定位。
2. **检查完整梯度方向与各 loss 项贡献。** λ/β 平衡的是初始标量 loss，不是 codebook 参数梯度。当前仅有合并梯度范数，不能分辨 A、λZ、βKL 谁主导，也不能把梯度范数比例当作 Adam 更新比例。
3. **不能据此判定 loss 本身不可迁移。** 当前结果是这套 LR、参数化、fake quant/STE、八步更新及 final8 选择下的负结果。校准拟合已失败，先证明优化能正常下降，才适合讨论训练轮数是否足够或 loss 的最终效果。

若以后重开，最小诊断应从现有共享初始化和一个固定 batch 开始：拆分 A/λZ/βKL 的 books/scales 梯度，核对小步更新前后真实 loss 与 decoded weight 变化，再决定 LR 或参数组策略。**本次未执行这些额外 forward 或训练，也未启动 sweep。**

## 7. 产物

- 紧凑完整分析：[summary.json](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-aqlm-loss-pair-plus-20261026/stopped_analysis/26020010.pbs101/summary.json>)。
- 分项记录位于同目录的 `closedloop_stopped.json`、`calibration_drift.json`、`parameter_summary.json`。
- 完整 614 层参数明细和原始 episode 副本保留于 compute 端 `/scratch/users/ntu/yguo017/fastwam-aqlm-loss-pair-plus-20261026/user_stop_analysis/26020010.pbs101/`，未经 login node 搬运大文件。
- 原始 fixed-input 最终报告：[report.zh.md](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-aqlm-loss-pair-20261026/pulled/26008761.pbs101/results/report.zh.md>)。

原 280×2 闭环目标尚未完成；按本次“先停止”指令暂停，保留已完成的 525 条记录和全部模型产物。
