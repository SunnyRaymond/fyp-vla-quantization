# C80 clip-only / joint：完整闭环结果

两组各完成280 episodes，共560个新episodes；复用2240个既有reference slots。四suite × 七扰动维度 × 每cell十variants，沿用冻结的manifest、初始state和配对environment/sampler seeds。两组额外C80 calibration均为80 contexts、80次更新，均从同一C8 rho1.5 update8初始化。

C80 clip-only为207/280（73.93%），joint为202/280（72.14%）。相对C8初始化206/280，分别净增1个和净减4个成功；joint比clip-only少5个。这轮闭环结果没有显示出joint相对clip-only的成功率收益。

## 整体成功率

| 配置 | 成功 / episodes | 成功率 |
|---|---:|---:|
| BF16 | 216/280 | 77.14% |
| QuaRot Adapted W4A4 | 195/280 | 69.64% |
| SpinQuant Adapted W4A4 | 203/280 | 72.50% |
| Smooth0.5 + H128 Scalar W4A4 | 200/280 | 71.43% |
| Row-scale VQ + A4 | 202/280 | 72.14% |
| Micro VQ + A4 | 204/280 | 72.86% |
| C8 rho0.1 | 201/280 | 71.79% |
| C8 rho1.5（两组初始化） | 206/280 | 73.57% |
| C80 clip-only | 207/280 | 73.93% |
| C80 joint | 202/280 | 72.14% |

原Micro VQ到C80 clip-only的差异包含此前C8 weight训练和本次C80 clipping两步；C80 clipping本身的增量应对照同一C8 rho1.5初始化（206→207）。

## 配对结果

Win表示candidate成功、reference失败；Loss相反。

| Candidate | Reference | Win | Loss | Tie | 净成功数 |
|---|---|---:|---:|---:|---:|
| C80 clip-only | C8 rho1.5（两组初始化） | 12 | 11 | 257 | +1 |
| C80 joint | C8 rho1.5（两组初始化） | 8 | 12 | 260 | -4 |
| C80 joint | C80 clip-only | 9 | 14 | 257 | -5 |
| C80 clip-only | BF16 | 9 | 18 | 253 | -9 |
| C80 joint | BF16 | 9 | 23 | 248 | -14 |

## 按suite（每列分母70）

| 条件 | BF16 | SpinQuant | Micro VQ | C8 rho1.5 | C80 clip-only | C80 joint |
|---|---:|---:|---:|---:|---:|---:|
| libero_10 | 54 | 49 | 51 | 53 | 52 | 49 |
| libero_goal | 51 | 48 | 46 | 48 | 50 | 48 |
| libero_object | 55 | 51 | 51 | 54 | 54 | 53 |
| libero_spatial | 56 | 55 | 56 | 51 | 51 | 52 |

## 按扰动维度（每列分母40）

| 条件 | BF16 | SpinQuant | Micro VQ | C8 rho1.5 | C80 clip-only | C80 joint |
|---|---:|---:|---:|---:|---:|---:|
| background_textures | 26 | 23 | 24 | 27 | 26 | 27 |
| camera_viewpoints | 21 | 16 | 19 | 17 | 17 | 12 |
| language_instructions | 38 | 36 | 38 | 36 | 37 | 37 |
| light_conditions | 35 | 39 | 35 | 36 | 37 | 37 |
| objects_layout | 35 | 33 | 34 | 34 | 34 | 34 |
| robot_initial_states | 34 | 31 | 30 | 32 | 31 | 30 |
| sensor_noise | 27 | 25 | 24 | 24 | 25 | 25 |

joint相对clip-only的五个净下降全部出现在camera_viewpoints：12/40对17/40，配对Win=0、Loss=5。background_textures多1个、robot_initial_states少1个，其余四个维度净差为0。按suite看，joint相对clip-only在LIBERO-10少3个、goal少2个、object少1个、spatial多1个。此处定位的是结果差异，未据此归因具体训练机制。

## 实际资源与完成证据

| 环节 | PBS ID | 资源与申请walltime | 实际walltime | 完成情况 |
|---|---|---|---|---|
| clip-only导出与backward检查 | 25857126.pbs101 | 1 GPU / 01:59:00 | 00:21:51 | F/exit0；复用既有80次更新，新训练updates=0 |
| joint训练与导出 | 25858027.pbs101 | 1 GPU / 01:59:00 | 01:07:25 | F/exit0；80/80，614 targets，动作重放max_abs=0 |
| clip-only闭环 | 25858029.pbs101 | 4 GPU / 24:00:00 | 04:26:02 | 四worker各70/rc0，之后仅汇总失败；原PBS exit1保留 |
| clip-only纯CPU汇总恢复 | 25860507.pbs101 | 1 CPU / 0 GPU / 00:20:00 | 00:00:10 | F/exit0、PIPELINE_COMPLETE，完整280/2240，新forward=0 |
| joint闭环及最终合并 | 25858606.pbs101 | 4 GPU / 24:00:00 | 04:07:54 | F/exit0、PIPELINE_COMPLETE；280/2240与共享560/2240全部通过 |

GPU任务均每15秒将utilization和显存写入job.log。gpt-6-luna/xhigh subagent按20分钟间隔监控，已在终态完成后结束。所有模型、数值检查和原始记录汇总均在PBS compute allocation执行。

## Query耗时与显存

两组均为batched_h128_v2、decoded BF16 VQ weights + dynamic A4、KV BF16。下表统计各worker首个正式query之后的queries；轨迹和query数随策略成功/失败而变化。

| 配置 | 后续query数 | full action-query均值 ms | 中位数 ms | p95 ms | allocated峰值 GB | reserved峰值 GB |
|---|---:|---:|---:|---:|---:|---:|
| C80 clip-only | 7016 | 4989.43 | 4962.59 | 5387.02 | 28.4475 | 28.7163 |
| C80 joint | 7146 | 4955.28 | 4920.63 | 5335.52 | 28.4475 | 28.7163 |

这些是当前decoded BF16执行路径的测量。原生低位宽kernel速度、完整压缩存储收益和新seeds下的重复实验不在本轮证据范围。

## 修复与原始记录

汇总恢复修复了两个报告路径错误：单arm normalization检查仍索引第二个arm，以及复用C8 reference时漏初始化每worker query状态。原代码、失败PBS状态和日志保留；episode、权重、clipping ratios、manifest和seeds均未更改。joint未来汇总代码同步修复，并在完整真实数据上通过。单arm、双arm normalization与完整560-slot C8 loader回归均已通过。

机器可读完整统计：[summary.json](summary.json)，包含suite × dimension cells及全部配对win/loss/tie；最终逐项证据：[FINAL_COMPLETION_EVIDENCE.json](FINAL_COMPLETION_EVIDENCE.json)。clip-only恢复证据：[clip_only_recovered_evidence.json](clip_only_recovered_evidence.json)。

远端结果目录：`/scratch/users/ntu/yguo017/fastwam-vq-c80-clip-joint-plus-20261017/results`。
