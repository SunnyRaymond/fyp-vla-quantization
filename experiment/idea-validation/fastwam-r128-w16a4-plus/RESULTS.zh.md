# H-W16A4 / C80-R128-W16A4：LIBERO+ 闭环结果

作业 `25910102.pbs101` 正常完成，PBS/artifact exit0，PIPELINE_COMPLETE和完整coverage门槛均通过。实际运行08:04:43。四worker各140，两组各280，共560个新episodes；13组旧reference共3640条。

| 配置 | 成功 / episodes | 成功率 |
|---|---:|---:|
| H-W16A4 | 204/280 | 72.86% |
| C80-R128-W16A4 | 203/280 | 72.50% |
| BF16 reference | 216/280 | 77.14% |
| SpinQuant adapted W4A4 | 203/280 | 72.50% |
| Microscaling VQ A4 | 204/280 | 72.86% |
| 原C8 rho1.5 | 206/280 | 73.57% |
| C80 clip-only | 207/280 | 73.93% |

C80相对H为16 wins、17 losses、247 ties，净少1个成功，差−0.36个百分点。本次固定配对cohort没有显示总体成功率收益；两组有33个episode的成功/失败结果不同。

| Suite | H | C80-R128 | C80−H 成功数 |
|---|---:|---:|---:|
| libero_10 | 50/70 | 48/70 | -2 |
| libero_goal | 51/70 | 46/70 | -5 |
| libero_object | 50/70 | 54/70 | +4 |
| libero_spatial | 53/70 | 55/70 | +2 |

| 扰动维度 | H | C80-R128 | C80−H 成功数 |
|---|---:|---:|---:|
| background_textures | 25/40 | 24/40 | -1 |
| camera_viewpoints | 14/40 | 19/40 | +5 |
| language_instructions | 38/40 | 35/40 | -3 |
| light_conditions | 37/40 | 35/40 | -2 |
| objects_layout | 35/40 | 36/40 | +1 |
| robot_initial_states | 30/40 | 31/40 | +1 |
| sensor_noise | 25/40 | 23/40 | -2 |

四worker的源7项metric重放误差、uncached/cached完整action/future差、正式infer action差均为0，覆盖614个Linears。固定变换后的BF16 weights缓存一次；没有新增训练、teacher forward、VQ或weight quantization。C80直接复用原80-context/8-update/LR1e-4 checkpoint，H使用U=I。

全部suite×dimension及与13组reference的paired对照、分实现timing见 `summary.json`；退出、coverage、启动检查和GPU遥测证据见 `FINAL_COMPLETION_EVIDENCE.json`。

执行边界为原BF16 weights配对Smooth/sign/H/U变换、dynamic A4、BF16KV。固定输入RMSE/latent NMSE改善与闭环成功率分别解释；本轮不证明native low-bit kernel性能或多次训练随机性下的统计显著性。
