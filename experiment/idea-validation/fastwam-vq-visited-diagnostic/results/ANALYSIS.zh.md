# 40 visited states：joint 的单点精度与局部响应

本轮完整比较了原 C80 final80 clip-only、joint 与 BF16。使用已有的40个C8 baseline短程visited states，四个LIBERO suites各10；每个state使用原seed_index0。每个state的base、image ±、proprio ±在三个模型间共享预处理后的输入tensor与采样seed。没有重新训练、采集新rollout或改变原80-query held-out。

GPU job 25863989.pbs101：F/exit0、artifact exit0、PHASE_COMPLETE、匹配summary；walltime 00:55:07。600/600端点、566次新model queries，其中39份BF16 base直接复用已有teacher，另160份扰动BF16参考新计算。六项teacher/reference/repeat parity的max_abs全部为0，所有参数冻结。CPU报告job25864600.pbs101用时8秒、F/exit0，无新增forward。gpt-6-luna/xhigh约每10分钟监控，GPU utilization/VRAM每15秒记录，监控已停止。

## 单点motor：总体改善，但不是每维、每state都改善

以下都是相对同输入、同seed BF16的normalized first10×6 motor误差。每个case先计算RMSE，再平均40个cases；不把400个commands或200个输入变体当独立case。

| 指标 | clip-only | joint |
|---|---:|---:|
| 整体motor RMSE | 0.045833804 | 0.044575345 |
| dx | 0.041703435 | 0.042312395 |
| dy | 0.046906956 | 0.041265435 |
| dz | 0.051390226 | 0.051777728 |
| d_rotation_x | 0.040549871 | 0.039565109 |
| d_rotation_y | 0.043139166 | 0.041611906 |
| d_rotation_z | 0.041031770 | 0.039880766 |

joint整体改善约2.75%，但dx、dz略差；40个states中22个整体motor改善、18个变差。三suite均值改善，spatial从0.052243098升到0.054049459。不能把平均收益理解成每个控制方向都更好。

为避免不同states的正负偏置相消，还计算了“逐state先平均10步误差，再取绝对值，最后平均states”。joint在六维中的五维改善，只有dx从0.017737122升到0.018799133。没有证据把整体退步解释成所有维度都出现更大的持续偏置。这也不是实际闭环漂移测量。

## Gripper：整体更好，但存在motor改善、释放时机反而退步的具体案例

采用实际部署的denormalize、符号翻转、np.sign路径：-1=open、+1=close；本轮只计算commands，没有在环境中执行这些新输出。

| Base指标 | clip-only | joint |
|---|---:|---:|
| Gripper normalized RMSE | 0.100219560 | 0.058089548 |
| 开合与BF16不一致的commands /400 | 6 | 3 |
| 存在开合不一致的states /40 | 5 | 2 |
| Chunk内switch位置不一致 /360 | 7 | 3 |

因此，“joint gripper整体变差”不成立。但case2045（libero_goal）提供了一个很具体的局部反例：

| 同一个输入，step为0-based | BF16 | clip-only | joint |
|---|---|---|---|
| Motor RMSE | reference | 0.074669043 | 0.066585624 |
| 首次由close转open | step8 | step9 | 前10步没有切换 |
| 与BF16开合不一致的steps | — | [8] | [8,9] |

也就是说，joint在同一输入上motor更接近BF16，却没有在实际执行的前10步内输出BF16所要求的释放。这证明平均motor误差不能代替关键动作时机，但不证明这个state导致了之前camera闭环的失败；两者不是同一批失败观测。

case2037则是两种量化模型都在step7提前open，BF16在step8才open。joint剩余3条错误commands的BF16阈值前绝对幅度为0.992–1.000，不能把这些错误简单描述为teacher恰好位于阈值附近。

Switch位置不一致只是比较step1–9各位置“是否发生切换”，不比较方向，也不直接测量延迟。case2045中joint完全漏掉chunk内切换，switch mismatch反而比clip少一个，因此该计数必须与command序列一起解释，不能单独用来排名动作时机。

## 局部输入响应：两种量化配置都明显偏离BF16，joint的proprio响应进一步变差

图像扰动为两相机raw uint8 brightness±1；proprio扰动为raw EEF xyz沿单位对角方向±5mm。其他传感器不变。这是输入敏感性测试，proprio变化不伴随物理移动与重渲染；图像brightness也不等于真实camera viewpoint变化。

对每个state计算前10×6输出的central finite difference，再比较student与BF16的响应。以下是先逐state计算再取均值，不是把所有steps直接pool：

| 方向 | clip响应误差RMS↓ | joint响应误差RMS↓ | clip相对BF16幅度gain | joint gain |
|---|---:|---:|---:|---:|
| Image brightness | 6.117905008 | 6.070675587 | 16.534596 | 16.758045 |
| EEF proprio | 4.187125051 | 4.358801250 | 24.393568 | 25.089995 |

两种输入的导数单位不同，不能跨行比较误差绝对大小。gain是每state的student/teacher响应RMS之比再平均，BF16本身响应较小，不能只凭倍数作判断；绝对响应误差、方向和正负单侧结果都已保留。

joint的image响应误差略降约0.77%，proprio响应误差却增加约4.10%。proprio的dx、dy、dz与rotation_y响应误差均变差，rotation_x/z改善。方向cosine的case均值也接近0：image为clip -0.0103、joint 0.0351；proprio为clip 0.0607（37个有定义cases）、joint -0.0129（36个）。这不等于每个case都正交，但说明平均响应方向对齐很弱，不能用“响应越小越好”替代teacher匹配。

有效性检查：image正负扰动在全部40个cases中都改变了实际BF16 image tensor；proprio正负扰动在全部40个cases中都改变了实际BF16 proprio tensor。没有因BF16 cast把整个预定扰动吞掉的case。同输入重复完全一致，因此不能将这些输入差分归因于本次重复推理漂移。

Gripper也出现输入扰动触发的翻转，BF16四种扰动均为0/400；clip在image+/image-/proprio+/proprio-分别为4/4/9/8，joint为1/2/6/1。joint在这项描述上总体更好，不能把全部局部响应指标统称为joint更差。

## 对下一步的判断

这次直接支持两点：joint能改善固定观测的动作数值，却不保证局部输入响应一起改善；平均motor收益也可能与个别关键gripper时机退步并存。它没有证明之前280个闭环episodes的净少5次成功由哪一个机制导致。

下一步优先检查并约束量化后的邻域响应，而不是仅增加相似demo数量。Calibration需要覆盖关键阶段及小扰动邻域；teacher必须在对应输入上生成共享目标。可考虑在calibration侧增加teacher-referenced action-difference loss：比较student的a(x+delta)-a(x)与BF16的相同差分，约束实际执行first10。不能只惩罚student变化大小，否则可能把不响应误当稳健；也不要除以接近零的teacher响应来构造不稳定loss。此项尚未训练或验证收益。

若引入channel scaling，第一轮应从匹配的父checkpoint出发，冻结books/indices/row gains，让连续D从1开始并限制偏移，保留同预算clipping继续优化的对照。先不要同时释放D与全部codebooks。需共同看weight-VQ误差、activation误差、Linear输出误差，再看action端点及本轮teacher响应；不能假定当前的响应失真全部来自activation，也不能假定连续D可无损折入现有B32 exponent。

Gripper至少应保留开合与释放时机检查，不根据总体gripper RMSE就宣布问题解决。原80个held-out paired queries、原seeds和teacher继续作为正式固定输入比较；不新增development set，不用本轮病例挑checkpoint或将测试失败直接搬入训练。本轮没有执行上述任何训练或channel scaling。

全部逐维normalized/command RMSE、MAE、bias、max_abs、四suite结果及正负单侧响应见RESULTS.zh.md、comparison.json；逐case补充、阈值margin和有效模型输入变化见diagnostic_details.json与input_perturbations.json。证据仍限于两个局部方向、40个既有短程states和decoded-BF16 VQ + fake A4，不是完整Jacobian、native-kernel收益或新的闭环成功率。
