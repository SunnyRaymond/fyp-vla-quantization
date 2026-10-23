# VQ LIBERO-Plus 完整评测结果

作业 `25771275.pbs101`（`fwvqplus1012p1`）正常完成，PBS F / Exit_status=0，四worker均140/140、rc=0，PIPELINE_COMPLETE成立。优化续跑allocation实际运行7h12m16s，申请上限为4GPU/24h。

两组VQ各280个episodes，均采用既有4 suites × 7扰动维度 × 10 variants、配对环境和sampler seeds及同一episode配置。原四组baseline的1120个episodes直接复用，本轮新增560个；其中45个完整episode来自旧逐block实现，515个来自batched_h128_v2，数值等价门全部通过。

| 配置 | 成功 / episodes | Success rate | 相对BF16 |
|---|---:|---:|---:|
| BF16 | 216/280 | 77.14% | +0.00 pp |
| QuaRot adapted W4A4 | 195/280 | 69.64% | -7.50 pp |
| SpinQuant adapted W4A4 | 203/280 | 72.50% | -4.64 pp |
| Smooth/Hadamard Scalar W4A4 | 200/280 | 71.43% | -5.71 pp |
| Row-scale VQ + A4 | 202/280 | 72.14% | -5.00 pp |
| Microscaling VQ B32 + A4 | 204/280 | 72.86% | -4.29 pp |

Row-scale VQ为202/280，microscaling VQ为204/280，SpinQuant为203/280。两组VQ分别比BF16少14和12个成功episodes。Microscaling比row-scale净多2个，差0.71个百分点；本轮支持整体表现接近，不能据此断言microscaling稳定优于row-scale或SpinQuant。

逐variant配对：wins表示当前VQ成功而对照失败，losses表示当前VQ失败而对照成功，ties包括双成功和双失败。

| 当前配置 | 对照 | Wins | Losses | Ties | 净成功差 |
|---|---|---:|---:|---:|---:|
| Row-scale VQ + A4 | Microscaling VQ B32 + A4 | 11 | 13 | 256 | -2 |
| Row-scale VQ + A4 | BF16 | 10 | 24 | 246 | -14 |
| Row-scale VQ + A4 | QuaRot adapted W4A4 | 19 | 12 | 249 | +7 |
| Row-scale VQ + A4 | SpinQuant adapted W4A4 | 14 | 15 | 251 | -1 |
| Row-scale VQ + A4 | Smooth/Hadamard Scalar W4A4 | 13 | 11 | 256 | +2 |
| Microscaling VQ B32 + A4 | Row-scale VQ + A4 | 13 | 11 | 256 | +2 |
| Microscaling VQ B32 + A4 | BF16 | 9 | 21 | 250 | -12 |
| Microscaling VQ B32 + A4 | QuaRot adapted W4A4 | 21 | 12 | 247 | +9 |
| Microscaling VQ B32 + A4 | SpinQuant adapted W4A4 | 13 | 12 | 255 | +1 |
| Microscaling VQ B32 + A4 | Smooth/Hadamard Scalar W4A4 | 16 | 12 | 252 | +4 |

每个suite的分母均为70。

| Suite | BF16 | QuaRot adapted W4A4 | SpinQuant adapted W4A4 | Smooth/Hadamard Scalar W4A4 | Row-scale VQ + A4 | Microscaling VQ B32 + A4 |
|---|---:|---:|---:|---:|---:|---:|
| libero_spatial | 56/70 | 51/70 | 55/70 | 53/70 | 54/70 | 56/70 |
| libero_object | 55/70 | 52/70 | 51/70 | 52/70 | 53/70 | 51/70 |
| libero_goal | 51/70 | 45/70 | 48/70 | 47/70 | 47/70 | 46/70 |
| libero_10 | 54/70 | 47/70 | 49/70 | 48/70 | 48/70 | 51/70 |

每个扰动维度的分母均为40。

| 扰动维度 | BF16 | QuaRot adapted W4A4 | SpinQuant adapted W4A4 | Smooth/Hadamard Scalar W4A4 | Row-scale VQ + A4 | Microscaling VQ B32 + A4 |
|---|---:|---:|---:|---:|---:|---:|
| background_textures | 26/40 | 24/40 | 23/40 | 26/40 | 24/40 | 24/40 |
| camera_viewpoints | 21/40 | 12/40 | 16/40 | 14/40 | 14/40 | 19/40 |
| language_instructions | 38/40 | 36/40 | 36/40 | 37/40 | 38/40 | 38/40 |
| light_conditions | 35/40 | 36/40 | 39/40 | 35/40 | 36/40 | 35/40 |
| objects_layout | 35/40 | 36/40 | 33/40 | 34/40 | 35/40 | 34/40 |
| robot_initial_states | 34/40 | 30/40 | 31/40 | 30/40 | 29/40 | 30/40 |
| sensor_noise | 27/40 | 21/40 | 25/40 | 24/40 | 26/40 | 24/40 |

Microscaling相对row-scale在camera_viewpoints多成功5个（19/40 vs 14/40），robot_initial_states多1个；sensor_noise少2个，light_conditions和objects_layout各少1个，其余维度持平。Suite上Spatial多2个、Long多3个，Object少2个、Goal少1个。camera_viewpoints仍是两组最弱的维度。

下列耗时来自PBS内已完成的聚合，按实现分组，分别排除两个worker本run的首个正式query；旧实现和优化实现没有混成一个均值。

| VQ配置 | FWHT实现 | 完整episodes | 后续queries | Query均值(s) | Query median(s) | Query p95(s) | Denoising均值(s) | Peak allocated(GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Row-scale VQ + A4 | source_per_block_v1 | 22 | 463 | 46.639 | 46.359 | 48.777 | 46.575 | 25.116 |
| Row-scale VQ + A4 | batched_h128_v2 | 258 | 6788 | 4.731 | 4.698 | 5.108 | 4.675 | 25.116 |
| Microscaling VQ B32 + A4 | source_per_block_v1 | 23 | 442 | 47.235 | 46.451 | 51.875 | 47.174 | 25.116 |
| Microscaling VQ B32 + A4 | batched_h128_v2 | 257 | 6669 | 4.783 | 4.746 | 5.169 | 4.726 | 25.116 |

优化后row-scale平均query为4.731秒，microscaling为4.783秒；旧实现分别为46.639秒和47.235秒，分组均值比约9.9倍。孤立transform在同卡同输入测试中约快22倍。旧组只有22/23个episode，且闭环轨迹与优化组不同，端到端分组均值比不是严格同输入benchmark。

两组优化后的CUDA peak allocated均为25.116 GB，baseline allocated均为24.851 GB；peak reserved分别为25.426和25.443 GB。此处GB是记录中的CUDA allocator口径，不等于nvidia-smi进程/设备显存；显存没有随FWHT加速出现明显下降。

两组均复用冻结bank、Smooth alpha=0.5/H128变换与动态per-row A4，不重新训练/校准；GPU端使用decoded BF16 weights和operands、KV BF16。614个目标Linear的四worker action参考重放max_abs均为0，218项GPU变换/A4 fixtures逐位一致。这是完整配置的闭环成功率比较，不是native INT4速度或编码存储即GPU显存的证据。

未新增随机seed重复实验或显著性检验；0.36–0.71个百分点的小差异应按本轮配对结果描述。完整suite/dimension/cell和配对数据保存在summary.json。

- [完整汇总JSON](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-vq-plus/results/summary.json>)
- [完成核验记录](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-vq-plus/completion_verified_25771275.json>)

## Microscaling VQ、SpinQuant、BF16：逐任务与扰动对照

以下数据在CPU PBS作业25784750.pbs101内读取已完成的原始episodes并对齐，仅做结果分析，没有新推理。共280个variant × 3组 = 840个配对outcome slots，40个原始任务，每任务7条选中扰动variant（每个维度一条），不是每任务7个独立随机seed。

三组186个variant都成功，54个都失败，40个有分歧，涉及28个原始任务。总成功数相近不代表成功/失败发生在同一组案例上。

| 当前组 | 对照 | 当前独赢 | 当前独输 | 相同结果 | 净成功差 |
|---|---|---:|---:|---:|---:|
| Microscaling VQ | SpinQuant | 13 | 12 | 255 | +1 |
| Microscaling VQ | BF16 | 9 | 21 | 250 | -12 |
| SpinQuant | BF16 | 6 | 19 | 255 | -13 |

### 每suite分母70

| 分组 | BF16 | SpinQuant | Microscaling VQ |
|---|---:|---:|---:|
| libero_10 | 54/70 | 49/70 | 51/70 |
| libero_goal | 51/70 | 48/70 | 46/70 |
| libero_object | 55/70 | 51/70 | 51/70 |
| libero_spatial | 56/70 | 55/70 | 56/70 |

### 每扰动维度分母40

| 分组 | BF16 | SpinQuant | Microscaling VQ |
|---|---:|---:|---:|
| 背景/纹理 | 26/40 | 23/40 | 24/40 |
| Camera viewpoint | 21/40 | 16/40 | 19/40 |
| 语言指令 | 38/40 | 36/40 | 38/40 |
| 光照 | 35/40 | 39/40 | 35/40 |
| 物体布局 | 35/40 | 33/40 | 34/40 |
| 机器人初始状态 | 34/40 | 31/40 | 30/40 |
| Sensor noise | 27/40 | 25/40 | 24/40 |

Microscaling相对SpinQuant在camera维度5胜2负、language维度2胜0负；light维度0胜4负。SpinQuant相对BF16在light维度4胜0负，而camera维度1胜6负、背景纹理维度1胜4负。BF16在机器人初始状态和sensor noise下的成功集合覆盖了两种量化配置的成功集合；量化组在这两个维度没有反向独赢。

### Suite × 扰动维度（每cell分母10）

| Cell | BF16 | SpinQuant | Microscaling VQ |
|---|---:|---:|---:|
| libero_10::background_textures | 4/10 | 3/10 | 4/10 |
| libero_10::camera_viewpoints | 5/10 | 4/10 | 4/10 |
| libero_10::language_instructions | 10/10 | 9/10 | 10/10 |
| libero_10::light_conditions | 7/10 | 9/10 | 8/10 |
| libero_10::objects_layout | 9/10 | 8/10 | 9/10 |
| libero_10::robot_initial_states | 10/10 | 9/10 | 9/10 |
| libero_10::sensor_noise | 9/10 | 7/10 | 7/10 |
| libero_goal::background_textures | 8/10 | 5/10 | 6/10 |
| libero_goal::camera_viewpoints | 4/10 | 4/10 | 3/10 |
| libero_goal::language_instructions | 10/10 | 9/10 | 9/10 |
| libero_goal::light_conditions | 9/10 | 10/10 | 9/10 |
| libero_goal::objects_layout | 6/10 | 6/10 | 6/10 |
| libero_goal::robot_initial_states | 8/10 | 8/10 | 8/10 |
| libero_goal::sensor_noise | 6/10 | 6/10 | 5/10 |
| libero_object::background_textures | 7/10 | 7/10 | 7/10 |
| libero_object::camera_viewpoints | 7/10 | 4/10 | 6/10 |
| libero_object::language_instructions | 9/10 | 9/10 | 9/10 |
| libero_object::light_conditions | 9/10 | 10/10 | 8/10 |
| libero_object::objects_layout | 10/10 | 9/10 | 9/10 |
| libero_object::robot_initial_states | 7/10 | 6/10 | 6/10 |
| libero_object::sensor_noise | 6/10 | 6/10 | 6/10 |
| libero_spatial::background_textures | 7/10 | 8/10 | 7/10 |
| libero_spatial::camera_viewpoints | 5/10 | 4/10 | 6/10 |
| libero_spatial::language_instructions | 9/10 | 9/10 | 10/10 |
| libero_spatial::light_conditions | 10/10 | 10/10 | 10/10 |
| libero_spatial::objects_layout | 10/10 | 10/10 | 10/10 |
| libero_spatial::robot_initial_states | 9/10 | 8/10 | 7/10 |
| libero_spatial::sensor_noise | 6/10 | 6/10 | 6/10 |

### 40个原始任务（每task分母7）

每条差异注明对应variant index及扰动，符号顺序为BF16 / SpinQuant / Microscaling VQ；✓为成功，✗为控制步数超时。任务总成功数相同但差异列非空，表示胜负在不同扰动下互相抵消。

| Suite | 原始任务 | BF16 | SpinQuant | Microscaling VQ | 分歧的扰动与结果 |
|---|---|---:|---:|---:|---|
| libero_10 | KITCHEN SCENE3 turn on the stove and put the moka pot on it | 5/7 | 3/7 | 5/7 | #214 物体布局 ✓/✗/✓；#224 Camera viewpoint ✓/✗/✓ |
| libero_10 | KITCHEN SCENE4 put the black bowl in the bottom drawer of the cabinet and close it | 4/7 | 4/7 | 4/7 | #264 背景/纹理 ✗/✗/✓；#275 Sensor noise ✓/✓/✗ |
| libero_10 | KITCHEN SCENE6 put the yellow and white mug in the microwave and close it | 5/7 | 4/7 | 4/7 | #274 Sensor noise ✓/✗/✗ |
| libero_10 | KITCHEN SCENE8 put both moka pots on the stove | 5/7 | 4/7 | 4/7 | #239 机器人初始状态 ✓/✓/✗；#273 Sensor noise ✓/✗/✓ |
| libero_10 | LIVING ROOM SCENE1 put both the alphabet soup and the cream cheese box in the basket | 6/7 | 6/7 | 6/7 | 无分歧 |
| libero_10 | LIVING ROOM SCENE2 put both the alphabet soup and the tomato sauce in the basket | 6/7 | 7/7 | 6/7 | #252 光照 ✗/✓/✗ |
| libero_10 | LIVING ROOM SCENE2 put both the cream cheese box and the butter in the basket | 7/7 | 6/7 | 6/7 | #222 Camera viewpoint ✓/✓/✗；#232 机器人初始状态 ✓/✗/✓ |
| libero_10 | LIVING ROOM SCENE5 put the white mug on the left plate and put the yellow and white mug on the right plate | 5/7 | 5/7 | 5/7 | #250 光照 ✗/✓/✓；#268 背景/纹理 ✓/✗/✗ |
| libero_10 | LIVING ROOM SCENE6 put the white mug on the plate and put the chocolate pudding to the right of the plate | 5/7 | 4/7 | 5/7 | #249 语言指令 ✓/✗/✓ |
| libero_10 | STUDY SCENE1 pick up the book and place it in the back compartment of the caddy | 6/7 | 6/7 | 6/7 | 无分歧 |
| libero_goal | open the middle drawer of the cabinet | 6/7 | 6/7 | 5/7 | #158 Camera viewpoint ✓/✓/✗ |
| libero_goal | open the top drawer and put the bowl inside | 4/7 | 3/7 | 2/7 | #174 语言指令 ✓/✗/✗；#188 光照 ✓/✓/✗ |
| libero_goal | push the plate to the front of the stove | 3/7 | 3/7 | 4/7 | #184 光照 ✗/✓/✓；#197 背景/纹理 ✓/✗/✓ |
| libero_goal | put the bowl on the plate | 6/7 | 6/7 | 6/7 | 无分歧 |
| libero_goal | put the bowl on the stove | 5/7 | 5/7 | 5/7 | 无分歧 |
| libero_goal | put the bowl on top of the cabinet | 5/7 | 4/7 | 4/7 | #191 背景/纹理 ✓/✗/✗ |
| libero_goal | put the cream cheese in the bowl | 3/7 | 3/7 | 4/7 | #196 背景/纹理 ✗/✗/✓ |
| libero_goal | put the wine bottle on the rack | 6/7 | 6/7 | 6/7 | 无分歧 |
| libero_goal | put the wine bottle on top of the cabinet | 7/7 | 6/7 | 5/7 | #199 背景/纹理 ✓/✗/✗；#206 Sensor noise ✓/✓/✗ |
| libero_goal | turn on the stove | 6/7 | 6/7 | 5/7 | #194 背景/纹理 ✓/✓/✗ |
| libero_object | pick up the alphabet soup and place it in the basket | 6/7 | 5/7 | 5/7 | #85 Camera viewpoint ✓/✗/✓；#119 光照 ✓/✓/✗ |
| libero_object | pick up the bbq sauce and place it in the basket | 6/7 | 5/7 | 5/7 | #87 Camera viewpoint ✓/✗/✗ |
| libero_object | pick up the butter and place it in the basket | 5/7 | 5/7 | 5/7 | #88 Camera viewpoint ✓/✗/✗；#110 光照 ✗/✓/✓ |
| libero_object | pick up the chocolate pudding and place it in the basket | 6/7 | 5/7 | 5/7 | #94 机器人初始状态 ✓/✗/✗ |
| libero_object | pick up the cream cheese and place it in the basket | 7/7 | 7/7 | 7/7 | 无分歧 |
| libero_object | pick up the ketchup and place it in the basket | 4/7 | 4/7 | 3/7 | #118 光照 ✓/✓/✗ |
| libero_object | pick up the milk and place it in the basket | 4/7 | 4/7 | 4/7 | 无分歧 |
| libero_object | pick up the orange juice and place it in the basket | 5/7 | 5/7 | 5/7 | #78 物体布局 ✓/✗/✗；#84 Camera viewpoint ✗/✓/✓ |
| libero_object | pick up the salad dressing and place it in the basket | 6/7 | 6/7 | 6/7 | 无分歧 |
| libero_object | pick up the tomato sauce and place it in the basket | 6/7 | 5/7 | 6/7 | #86 Camera viewpoint ✓/✗/✓ |
| libero_spatial | pick up the black bowl between the plate and the ramekin and place it on the plate | 7/7 | 7/7 | 7/7 | 无分歧 |
| libero_spatial | pick up the black bowl from table center and place it on the plate | 6/7 | 6/7 | 7/7 | #14 Camera viewpoint ✗/✗/✓ |
| libero_spatial | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 6/7 | 5/7 | 4/7 | #29 机器人初始状态 ✓/✗/✗；#57 背景/纹理 ✓/✓/✗ |
| libero_spatial | pick up the black bowl next to the cookie box and place it on the plate | 5/7 | 5/7 | 5/7 | 无分歧 |
| libero_spatial | pick up the black bowl next to the plate and place it on the plate | 5/7 | 5/7 | 4/7 | #23 机器人初始状态 ✓/✓/✗ |
| libero_spatial | pick up the black bowl next to the ramekin and place it on the plate | 4/7 | 4/7 | 5/7 | #37 语言指令 ✗/✗/✓ |
| libero_spatial | pick up the black bowl on the cookie box and place it on the plate | 6/7 | 6/7 | 6/7 | 无分歧 |
| libero_spatial | pick up the black bowl on the ramekin and place it on the plate | 6/7 | 5/7 | 6/7 | #17 Camera viewpoint ✓/✗/✓ |
| libero_spatial | pick up the black bowl on the stove and place it on the plate | 5/7 | 6/7 | 6/7 | #59 背景/纹理 ✗/✓/✓ |
| libero_spatial | pick up the black bowl on the wooden cabinet and place it on the plate | 6/7 | 6/7 | 6/7 | 无分歧 |

### 全部40个分歧variant

以下保留冻结variant ID；图像内容、抓取/放置失败原因需另外回放轨迹才能判断，此处不做机制归因。

| Index | Variant ID | 原始任务 | 扰动 | BF16 | SpinQuant | Microscaling VQ |
|---:|---|---|---|---|---|---|
| 14 | libero_spatial:0679 | pick up the black bowl from table center and place it on the plate | Camera viewpoint | 失败 | 失败 | 成功 |
| 17 | libero_spatial:0809 | pick up the black bowl on the ramekin and place it on the plate | Camera viewpoint | 成功 | 失败 | 成功 |
| 23 | libero_spatial:0538 | pick up the black bowl next to the plate and place it on the plate | 机器人初始状态 | 成功 | 成功 | 失败 |
| 29 | libero_spatial:0409 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 机器人初始状态 | 成功 | 失败 | 失败 |
| 37 | libero_spatial:1051 | pick up the black bowl next to the ramekin and place it on the plate | 语言指令 | 失败 | 失败 | 成功 |
| 57 | libero_spatial:0087 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 背景/纹理 | 成功 | 成功 | 失败 |
| 59 | libero_spatial:0223 | pick up the black bowl on the stove and place it on the plate | 背景/纹理 | 失败 | 成功 | 成功 |
| 78 | libero_object:2122 | pick up the orange juice and place it in the basket | 物体布局 | 成功 | 失败 | 失败 |
| 84 | libero_object:1028 | pick up the orange juice and place it in the basket | Camera viewpoint | 失败 | 成功 | 成功 |
| 85 | libero_object:0663 | pick up the alphabet soup and place it in the basket | Camera viewpoint | 成功 | 失败 | 成功 |
| 86 | libero_object:0860 | pick up the tomato sauce and place it in the basket | Camera viewpoint | 成功 | 失败 | 成功 |
| 87 | libero_object:0768 | pick up the bbq sauce and place it in the basket | Camera viewpoint | 成功 | 失败 | 失败 |
| 88 | libero_object:0892 | pick up the butter and place it in the basket | Camera viewpoint | 成功 | 失败 | 失败 |
| 94 | libero_object:0582 | pick up the chocolate pudding and place it in the basket | 机器人初始状态 | 成功 | 失败 | 失败 |
| 110 | libero_object:2311 | pick up the butter and place it in the basket | 光照 | 失败 | 成功 | 成功 |
| 118 | libero_object:2396 | pick up the ketchup and place it in the basket | 光照 | 成功 | 成功 | 失败 |
| 119 | libero_object:2260 | pick up the alphabet soup and place it in the basket | 光照 | 成功 | 成功 | 失败 |
| 158 | libero_goal:0698 | open the middle drawer of the cabinet | Camera viewpoint | 成功 | 成功 | 失败 |
| 174 | libero_goal:1221 | open the top drawer and put the bowl inside | 语言指令 | 成功 | 失败 | 失败 |
| 184 | libero_goal:2421 | push the plate to the front of the stove | 光照 | 失败 | 成功 | 成功 |
| 188 | libero_goal:2382 | open the top drawer and put the bowl inside | 光照 | 成功 | 成功 | 失败 |
| 191 | libero_goal:0148 | put the bowl on top of the cabinet | 背景/纹理 | 成功 | 失败 | 失败 |
| 194 | libero_goal:0272 | turn on the stove | 背景/纹理 | 成功 | 成功 | 失败 |
| 196 | libero_goal:0162 | put the cream cheese in the bowl | 背景/纹理 | 失败 | 失败 | 成功 |
| 197 | libero_goal:0094 | push the plate to the front of the stove | 背景/纹理 | 成功 | 失败 | 成功 |
| 199 | libero_goal:0266 | put the wine bottle on top of the cabinet | 背景/纹理 | 成功 | 失败 | 失败 |
| 206 | libero_goal:1579 | put the wine bottle on top of the cabinet | Sensor noise | 成功 | 成功 | 失败 |
| 214 | libero_10:1956 | KITCHEN SCENE3 turn on the stove and put the moka pot on it | 物体布局 | 成功 | 失败 | 成功 |
| 222 | libero_10:0702 | LIVING ROOM SCENE2 put both the cream cheese box and the butter in the basket | Camera viewpoint | 成功 | 成功 | 失败 |
| 224 | libero_10:0753 | KITCHEN SCENE3 turn on the stove and put the moka pot on it | Camera viewpoint | 成功 | 失败 | 成功 |
| 232 | libero_10:0350 | LIVING ROOM SCENE2 put both the cream cheese box and the butter in the basket | 机器人初始状态 | 成功 | 失败 | 成功 |
| 239 | libero_10:0628 | KITCHEN SCENE8 put both moka pots on the stove | 机器人初始状态 | 成功 | 成功 | 失败 |
| 249 | libero_10:1351 | LIVING ROOM SCENE6 put the white mug on the plate and put the chocolate pudding to the right of the plate | 语言指令 | 成功 | 失败 | 成功 |
| 250 | libero_10:2443 | LIVING ROOM SCENE5 put the white mug on the left plate and put the yellow and white mug on the right plate | 光照 | 失败 | 成功 | 成功 |
| 252 | libero_10:2411 | LIVING ROOM SCENE2 put both the alphabet soup and the tomato sauce in the basket | 光照 | 失败 | 成功 | 失败 |
| 264 | libero_10:0062 | KITCHEN SCENE4 put the black bowl in the bottom drawer of the cabinet and close it | 背景/纹理 | 失败 | 失败 | 成功 |
| 268 | libero_10:0208 | LIVING ROOM SCENE5 put the white mug on the left plate and put the yellow and white mug on the right plate | 背景/纹理 | 成功 | 失败 | 失败 |
| 273 | libero_10:1846 | KITCHEN SCENE8 put both moka pots on the stove | Sensor noise | 成功 | 失败 | 成功 |
| 274 | libero_10:1910 | KITCHEN SCENE6 put the yellow and white mug in the microwave and close it | Sensor noise | 成功 | 失败 | 失败 |
| 275 | libero_10:1637 | KITCHEN SCENE4 put the black bowl in the bottom drawer of the cabinet and close it | Sensor noise | 成功 | 成功 | 失败 |

这是固定cohort上的探索性配对描述，单个task×扰动只有一条选中variant；未进行额外seed复测、轨迹回放、显著性或多重比较检验，不能把小分项差异解释为稳定机制。

- [CPU分析原始汇总](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-vq-plus/results/triplet_analysis.json>)
- [分歧variant与控制步数](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-vq-plus/results/triplet_cases.json>)
