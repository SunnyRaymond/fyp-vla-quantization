# C8 action+latent-aware VQ：LIBERO-Plus闭环结果

作业：`25822971.pbs101`（fwale1015），PBS F/Exit_status=0，artifact exit0，PIPELINE_COMPLETE=true；实际walltime 08:02:26，4 GPUs。新增560/560个episodes，两arm各280；复用六组baseline共1680个slots，没有重跑baseline。

比较口径：原冻结4 suites × 7扰动维度 × 每cell 10个variants，环境状态及sampler seeds配对；使用已在compute allocation内生成的summary，以下只整理现有聚合结果。

| 配置 | 成功 / 280 | Success rate |
|---|---:|---:|
| BF16 | 216/280 | 77.14% |
| C8 action+latent rho1.5 (1 epoch) | 206/280 | 73.57% |
| 既有 Microscaling VQ A4 | 204/280 | 72.86% |
| SpinQuant-adapted W4A4 | 203/280 | 72.50% |
| Row-scale VQ A4 | 202/280 | 72.14% |
| C8 action+latent rho0.1 (2 epochs) | 201/280 | 71.79% |
| Smooth0.5 + H128 Scalar W4A4 | 200/280 | 71.43% |
| QuaRot-adapted W4A4 | 195/280 | 69.64% |

rho1.5在本轮量化配置中成功数最高：相对既有Micro多2个、SpinQuant多3个，仍比BF16少10个；rho0.1比既有Micro少3个。差异需要结合逐episode的win/loss解释。

## 配对结果

Wins指候选成功、参照失败，losses相反，ties包括双方都成功或都失败。

| 候选 | 参照 | Wins | Losses | Ties | 成功率差 (pp) |
|---|---|---:|---:|---:|---:|
| rho0.1 | C8 action+latent rho1.5 (1 epoch) | 10 | 15 | 255 | -1.79 |
| rho0.1 | 既有 Microscaling VQ A4 | 14 | 17 | 249 | -1.07 |
| rho0.1 | SpinQuant-adapted W4A4 | 14 | 16 | 250 | -0.71 |
| rho0.1 | BF16 | 10 | 25 | 245 | -5.36 |
| rho1.5 | 既有 Microscaling VQ A4 | 15 | 13 | 252 | +0.71 |
| rho1.5 | SpinQuant-adapted W4A4 | 16 | 13 | 251 | +1.07 |
| rho1.5 | BF16 | 10 | 20 | 250 | -3.57 |

## Suite成功数（各70 episodes）

| 分组 | rho0.1 | rho1.5 | 既有Micro | SpinQuant | BF16 |
|---|---:|---:|---:|---:|---:|
| libero_10 | 49/70 | 53/70 | 51/70 | 49/70 | 54/70 |
| libero_goal | 48/70 | 48/70 | 46/70 | 48/70 | 51/70 |
| libero_object | 54/70 | 54/70 | 51/70 | 51/70 | 55/70 |
| libero_spatial | 50/70 | 51/70 | 56/70 | 55/70 | 56/70 |

## 扰动维度成功数（各40 episodes）

| 分组 | rho0.1 | rho1.5 | 既有Micro | SpinQuant | BF16 |
|---|---:|---:|---:|---:|---:|
| background_textures | 26/40 | 27/40 | 24/40 | 23/40 | 26/40 |
| camera_viewpoints | 16/40 | 17/40 | 19/40 | 16/40 | 21/40 |
| language_instructions | 37/40 | 36/40 | 38/40 | 36/40 | 38/40 |
| light_conditions | 37/40 | 36/40 | 35/40 | 39/40 | 35/40 |
| objects_layout | 33/40 | 34/40 | 34/40 | 33/40 | 35/40 |
| robot_initial_states | 29/40 | 32/40 | 30/40 | 31/40 | 34/40 |
| sensor_noise | 23/40 | 24/40 | 24/40 | 25/40 | 27/40 |

rho1.5相对既有Micro，suite净变化为libero_10 +2、Goal +2、Object +3、Spatial -5。Object是3胜0负；Spatial是2胜7负，新增能力与退步同时存在。扰动维度的净变化为背景+3、机器人初始状态+2、光照+1、相机-2、语言-2，物体布局与传感器噪声净值不变。

rho1.5相对rho0.1多5个成功，主要来自libero_10多4个和Spatial多1个；扰动上机器人初始状态多3个，其余背景/相机/布局/噪声各多1个，语言与光照各少1个。这些是固定cohort的描述性分解。

## Query耗时与显存

| Arm | 非首query数 | Full action query中位数 | GPU denoising interval中位数 | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|
| rho0.1 | 7080 | 5.045 s | 4.990 s | 28.45 GB | 28.72 GB |
| rho1.5 | 6933 | 5.069 s | 5.015 s | 28.45 GB | 28.72 GB |

每组两个worker的首个正式query单独分组，不混入上表。实现为batched_h128_v2；两组轨迹及query次数不同，耗时只描述当前decoded BF16实现，不能作为native INT4加速或同输入kernel对照。上述显存为PyTorch allocated/reserved口径，GB为十进制。

## Calibration与结论边界

当前两组共享C8 source job25789685.pbs101、case IDs [2000,2019,2021,2025,2051,2057,2066,2070]，原始LIBERO四suite各两个；normalization_c=0.028303017887797623。rho0.1=2epochs/16updates，rho1.5=1epoch/8updates，均为闭环前固定的完整配置，因此两组差异同时含rho和epoch因素。

既有Micro baseline204/280来自fastwam-vq-microscale-20261011 bank，其protocol calibration cases是[0,2,4,6,10,12,14,16]，不是本轮C8 source bank的未训练闭环对照。配对cohort允许比较完整配置，但204→206不能单独归因于action+latent训练；当前也没有同C8、同epoch的action-only闭环arm来隔离latent loss的作用。

本轮每个task×扰动选一个variant，每cell只有10个episodes，且同一task可能跨维度出现。这里报告固定cohort的成功数和paired win/loss；没有把2个episode的小差距解释为统计显著优势，也不据此宣称未见任务/seed上的稳定提升。

权重为decoded BF16 VQ、activation动态A4、KV BF16；不是native low-bit kernel测评。完整原始聚合字段及suite×dimension cells见summary.json。worker、C8 reference、source-action parity与H128/A4预检证据见../completion_evidence.json。
