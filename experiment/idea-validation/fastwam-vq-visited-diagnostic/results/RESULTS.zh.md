# 40 visited states：同输入动作诊断

原C80 final80 clip-only、joint与BF16；40个既有C8-student-visited states，四suite各10，原seed_index0。
三组共享每state预处理后的输入tensor与采样seed。40×3×5=600端点，566次新查询、39份BF16 base缓存；训练0、rollout0。
RMSE/MAE先逐case按前10步计算，再对40case等权；max_abs取跨case最大。原80-query held-out未改动。

| Motor | clip normalized RMSE | joint normalized RMSE | clip command RMSE | joint command RMSE |
|---|---:|---:|---:|---:|
| dx | 0.041703435 | 0.042312395 | 0.039096967 | 0.039667868 |
| dy | 0.046906956 | 0.041265435 | 0.043975268 | 0.038686343 |
| dz | 0.051390226 | 0.051777728 | 0.048178333 | 0.048541617 |
| d_rotation_x | 0.040549871 | 0.039565109 | 0.012121515 | 0.011827141 |
| d_rotation_y | 0.043139166 | 0.041611906 | 0.016177187 | 0.015604465 |
| d_rotation_z | 0.04103177 | 0.039880766 | 0.015167101 | 0.014741641 |

| Base指标 | clip-only | joint |
|---|---:|---:|
| First10 motor RMSE | 0.045833804 | 0.044575345 |
| Gripper normalized RMSE | 0.10021956 | 0.058089548 |
| Gripper开合不一致 /400 | 6 | 3 |
| 至少一次开合不一致的states /40 | 5 | 2 |
| Chunk内切换位置不一致 /360 | 7 | 3 |

Gripper -1=open、+1=close、0=阈值零。以下confusion行是BF16、列是student，顺序[-1,0,+1]：

- clip_only: `[[123, 0, 4], [0, 0, 0], [2, 0, 271]]`
- joint: `[[125, 0, 2], [0, 0, 0], [1, 0, 272]]`

Switch只比较step1–9各位置是否发生切换；不比较切换方向，也不包含上一chunk到step0的切换。

| Motor | clip mean abs case bias | joint mean abs case bias | clip global signed bias | joint global signed bias |
|---|---:|---:|---:|---:|
| dx | 0.017737122 | 0.018799133 | -0.0076953125 | -0.004940033 |
| dy | 0.018512554 | 0.01770628 | 0.0032783318 | 0.0028922844 |
| dz | 0.024125366 | 0.023039093 | -0.0068411255 | -0.0090953064 |
| d_rotation_x | 0.016424751 | 0.014017792 | 0.0045658493 | 0.0020527649 |
| d_rotation_y | 0.018627605 | 0.014573193 | -0.0030210686 | 0.0058186531 |
| d_rotation_z | 0.015581436 | 0.01369545 | -0.0013954926 | 0.0054523468 |

Mean abs case bias先在每个state内平均10步误差，再取绝对值并平均states；避免不同state的正负偏置相消。它仍不是闭环漂移测量。

| Suite | clip motor RMSE | joint motor RMSE | clip gripper mismatch /100 | joint gripper mismatch /100 |
|---|---:|---:|---:|---:|
| libero_10 | 0.040437964 | 0.039931649 | 0 | 0 |
| libero_goal | 0.050936744 | 0.045634068 | 4 | 2 |
| libero_object | 0.03971741 | 0.038686204 | 1 | 1 |
| libero_spatial | 0.052243098 | 0.054049459 | 1 | 0 |

## 局部方向响应：两相机brightness ±1 uint8

| Geometry case均值 | clip-only | joint |
|---|---:|---:|
| student_rms | 6.1330839 | 6.0728414 |
| teacher_rms | 0.5374692 | 0.5374692 |
| error_rms | 6.117905 | 6.0706756 |
| gain | 16.534596 | 16.758045 |
| cosine | -0.010316387 | 0.035075077 |

| Motor | BF16 derivative RMS | clip derivative error | joint derivative error | joint improved/worse/tied states |
|---|---:|---:|---:|---|
| dx | 0.48438144 | 6.0360468 | 5.6704092 | 28/12/0 |
| dy | 0.46214956 | 5.8854447 | 6.1242567 | 13/27/0 |
| dz | 0.56077256 | 6.7185795 | 6.6656543 | 21/19/0 |
| d_rotation_x | 0.37432419 | 5.3560731 | 5.6675339 | 16/24/0 |
| d_rotation_y | 0.55521647 | 5.7007848 | 5.69171 | 19/21/0 |
| d_rotation_z | 0.37427366 | 5.8850437 | 5.3981005 | 22/18/0 |

Geometry是60维motor方向响应的RMS、相对BF16误差、幅度gain及方向cosine，再对case等权；gain/cosine的defined case数见JSON。
不能跨image/proprio比较导数绝对大小，两者输入单位不同；响应越小不自动越好。

## 局部方向响应：EEF xyz沿单位对角方向±5mm

| Geometry case均值 | clip-only | joint |
|---|---:|---:|
| student_rms | 4.1804275 | 4.3376002 |
| teacher_rms | 0.19419306 | 0.19419306 |
| error_rms | 4.1871251 | 4.3588012 |
| gain | 24.393568 | 25.089995 |
| cosine | 0.060660625 | -0.012856102 |

| Motor | BF16 derivative RMS | clip derivative error | joint derivative error | joint improved/worse/tied states |
|---|---:|---:|---:|---|
| dx | 0.24818276 | 4.0770512 | 4.4279786 | 18/21/1 |
| dy | 0.19164865 | 3.9698917 | 4.1490611 | 16/23/1 |
| dz | 0.24705111 | 4.6316349 | 4.8783807 | 19/20/1 |
| d_rotation_x | 0.11766322 | 3.8921018 | 3.7462699 | 22/17/1 |
| d_rotation_y | 0.12616701 | 3.8336147 | 4.1340137 | 15/24/1 |
| d_rotation_z | 0.10556304 | 4.0692831 | 3.8911652 | 21/18/1 |

Geometry是60维motor方向响应的RMS、相对BF16误差、幅度gain及方向cosine，再对case等权；gain/cosine的defined case数见JSON。
不能跨image/proprio比较导数绝对大小，两者输入单位不同；响应越小不自动越好。

## 证据范围

这是40个既有baseline短程visited states的固定输入诊断；不是joint自身完整trajectory，也没有真实camera viewpoint扰动。
两条预先固定的有限差分方向不能代表完整Jacobian或闭环稳定性，不能单独证明280-episode闭环差异的因果机制。
更完整的逐维MAE/max、正负单侧响应、扰动gripper翻转、suite/case统计见comparison.json和diagnostic_details.json。
有效BF16模型输入变化及饱和比例保留在input_perturbations.json；输出为decoded-BF16 VQ + fake A4，不是native INT4速度评测。
