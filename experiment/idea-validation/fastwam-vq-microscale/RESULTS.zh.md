# Microscaling + additive VQ：最小数值验证结果

B32 power-of-two scaling 在本轮固定输入配对诊断中带来正向结果：test 前 10 步 motor RMSE 从 **0.05705206** 降至 **0.05240395**，相对下降 **8.15%**，**14/22** 个 cases 改善，达到事先规定的“平均下降至少 5%，至少 12/22 cases 改善”工程门槛。Selection 的同一指标则增加 **1.11%**。这支持继续研究 block scaling，但不能据此认定它在所有场景都稳定优于 row scaling。

## 完成与复现证据

- PBS：`25755896.pbs101`，状态 F，Exit_status=0；请求 walltime `02:00:00`，实际 `01:25:21`。本次一个 allocation 内完整完成，无需再提交续跑。
- Compute device：NVIDIA A100-SXM4-40GB；沿用 Fast-WAM Optional-IDM、原 Apptainer/venv、checkpoint、sampler 和输入环境。
- 覆盖：614 个目标 Linear；8 个 calibration replay contexts；selection 4 cases × 2 seeds = 8 queries；test 22 cases × 2 seeds = 44 queries；共保存 52 个新 arm actions。
- 8 次 BF16W+A4 calibration replay 与原缓存 actions 的 max absolute difference 全部为 0。所有新 queries 都记录 614 个目标模块和提前冻结的 banks。
- 完成标记、退出 receipt、summary 的 phase/job/protocol 均一致。已核对实际运行的 protocol/runner 与保存源码一致，以及 52 个原始 actions 和 query records 的唯一配对、完整 shape 与有限值。
- 主要证据：[完整 summary](retrieved/results/full/summary.json)、[原始 actions](retrieved/results/full/actions.jsonl)、[query records](retrieved/results/full/queries.jsonl)、[PBS 终态证据](full_terminal_evidence.json)、[结果验收记录](RESULT_VERIFICATION.json)。

## 冻结方案与指标

保持 d=4、两本 256-entry BF16 additive codebooks、原 BF16 row RMS gain、Smooth alpha=0.5、Hadamard-128、dynamic signed A4 和 BF16 KV。每个 Linear 共享两本 books；以输入维度 B=32 的 block 共享一个 uint8 exponent，即八个 d=4 vectors 共用 scale。使用 `S = row_RMS * 2**round(log2(block_RMS/row_RMS))`；scales 在拟合时固定，assignment/centroid metric 包含实际 `S**2` 和原 A4 输入 diagonal second moment。Samples=8192、iterations=8、seed=20261006。

本轮是 **input-moment-weighted VQ 的 block-scale 验证**。`local_tuning_steps=0`、`action_lambda=0`，没有加入 action-aware local fine-tuning 或 final-action distillation。Row-scale VQ 与 Scalar W4A4 数值结果复用原成功作业 `25743693.pbs101` 的同输入、同 sampler seed 缓存；不是在本作业中重新运行这两个 arms。

主指标对每个 query 计算与 BF16 reference 相比的前 10 步、6 维 motor RMSE，再先平均每个 case 的两个 seeds、后等权平均 cases。以下是 test 的 22 cases / 44 contexts；所有 RMSE 越低越好。

| Arm | Motor RMSE，first10 | Motor RMSE，32 steps | Gripper RMSE，first10 |
|---|---:|---:|---:|
| Row-scale VQ + A4，对照 | 0.05705206 | 0.05970920 | 0.03179796 |
| B32 Microscaled VQ + A4 | **0.05240395** | **0.05726171** | 0.02152694 |
| 历史 Scalar W4A4，缓存参考 | 0.05499417 | 0.05934098 | **0.01251066** |

Microscaled VQ 的 test motor 两项指标优于当前 row-scale VQ 与历史 Scalar reference；gripper 优于 row-scale VQ，但仍差于 Scalar。Selection 的 first10 motor RMSE 为 0.04792927，对照为 0.04740377，只有 2/4 cases 改善。

## 存储与局部诊断

| 目标权重编码 payload，含 transforms | Row-scale VQ | Microscaled VQ |
|---|---:|---:|
| Effective bits/weight | 4.02087306 | 4.27087821 |
| Bytes | 3,023,856,089 | 3,211,869,881 |

额外 **188,013,792 bytes** 全部来自 block exponents，约增加 **0.25 bits/weight**。这些数字包含 indices、codebooks、row gains、exponents 和保存的 transforms；不是整个模型的磁盘文件大小或 GPU 峰值显存。实际 execution 仍把 VQ weights 解码成 BF16。

| 局部诊断 | 改善的 Linears / 614 | 每个 Linear 的 Micro/Control ratio 均值 | Ratio 中位数 |
|---|---:|---:|---:|
| Weight MSE | 515 | 0.949976 | 0.992695 |
| Input-moment-weighted weight MSE | 537 | 0.936985 | 0.990908 |
| 保存的 teacher inputs/outputs 上的局部 output MSE | 468 | 0.967810 | 0.994565 |

Ratio 越低越好。这是逐 Linear ratios 的非加权统计，不是把全模型所有元素汇总后的 MSE。多个局部诊断与 action 指标方向一致，支持 block scaling 有帮助的解释；仍没有定位哪一层造成某个 case 的 action 改善。

Exponent 范围为 -7 到 2，非零 exponents 共 10,291,730 个，clipped count 为 0。当前存储的 row gain × exponent scale 属于 Microscaling 风格的表示；本轮没有实现标准 MXFP4/NVFP4 或 native INT4 execution。

## 差异与结论边界

最大改善来自 plus/robot case 1017：motor RMSE 0.14466911 → 0.09453254。明显退步包括 plus/camera case 1015：0.05745678 → 0.07454431，以及 plus/language case 1020：0.04560701 → 0.05776731。平均收益包含较大 case 差异；8.15% 不能解释为每个场景都改善。

本轮通过预设工程筛选门槛，因此“power-of-two scaling 没有收益时才追加 G32 BF16 block-scale diagnostic”的条件未触发。本次最小验证已完成，未自动扩大实验。

这是既有固定输入上的配对机制诊断：原 test 已被看过，不能作为新的独立泛化验证，也没有统计显著性声明。0 个 closed-loop episodes、0 个预测 actions 被环境执行。本轮没有验证机器人成功率、action-aware fitting 叠加收益、原生低位 kernel、延迟或端到端加速；实际跨 allocation 续跑也未发生。

完整 fitted banks、moments 与逐 query 断点保留在远端 `/scratch/users/ntu/yguo017/fastwam-vq-microscale-20261011/results/full`；本地保存上述小型原始结果及本报告，没有通过 login node 传输大型 `.pt` banks。

## 全部 test cases 的配对结果

下表直接整理 PBS summary 中已计算的 case 均值。Δ = Micro − Row；负值表示 Microscaling 改善。

| Case | Domain / dimension | Row RMSE | Micro RMSE | Delta |
|---|---|---:|---:|---:|
| 1004 | original / unperturbed | 0.03878170 | 0.03948933 | +0.00070762 |
| 1005 | original / unperturbed | 0.04348471 | 0.04473771 | +0.00125300 |
| 1006 | original / unperturbed | 0.06415404 | 0.05183965 | -0.01231438 |
| 1007 | original / unperturbed | 0.04084834 | 0.03716631 | -0.00368203 |
| 1008 | original / unperturbed | 0.05590371 | 0.05110232 | -0.00480139 |
| 1009 | original / unperturbed | 0.04350794 | 0.04522412 | +0.00171617 |
| 1010 | original / unperturbed | 0.04588899 | 0.04148913 | -0.00439986 |
| 1011 | original / unperturbed | 0.05703247 | 0.05598997 | -0.00104249 |
| 1012 | original / unperturbed | 0.04274952 | 0.03932556 | -0.00342396 |
| 1013 | original / unperturbed | 0.04473266 | 0.04369286 | -0.00103981 |
| 1014 | plus / camera | 0.06153837 | 0.06031269 | -0.00122569 |
| 1015 | plus / camera | 0.05745678 | 0.07454431 | +0.01708753 |
| 1016 | plus / robot | 0.09248477 | 0.08140318 | -0.01108159 |
| 1017 | plus / robot | 0.14466911 | 0.09453254 | -0.05013657 |
| 1018 | plus / layout | 0.04948272 | 0.04007922 | -0.00940350 |
| 1019 | plus / layout | 0.04269511 | 0.04419407 | +0.00149895 |
| 1020 | plus / language | 0.04560701 | 0.05776731 | +0.01216030 |
| 1021 | plus / language | 0.04280426 | 0.04350381 | +0.00069955 |
| 1022 | plus / light | 0.04093938 | 0.04358900 | +0.00264962 |
| 1023 | plus / light | 0.04876332 | 0.04793545 | -0.00082787 |
| 1024 | plus / background | 0.05510489 | 0.04060348 | -0.01450141 |
| 1025 | plus / background | 0.09651545 | 0.07436480 | -0.02215065 |
