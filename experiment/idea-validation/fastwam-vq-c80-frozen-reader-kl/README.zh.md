# C80 A+Z 与 frozen-reader KL 同预算对照

2026-10-22 用户修改范围：取消 C80 A+Z 与依赖双组的原 aggregate，先分析已完成的 C80 KL，然后结束 goal。KL 已完成80次更新及160个 student、160个 frozen-reader endpoints。A+Z 已终止，不能用其部分更新充当同预算对照；原 aggregate 未执行。监控 subagent 已停止。另用 CPU allocation 汇总现有缓存，与历史 C8 A+Z 和 C8 reader-KL 比较，不新增模型 forward 或训练。下文保留原始预定设计，实际完成结论以 results/CONCLUSIONS.zh.md 为准。

缓存分析已完成。相对历史 C8 A+Z，C80 KL 的 held-out first10 motor RMSE +0.33%（20/40改善），future NMSE −10.70%；calibration reader KL 相对共同原始初始化 +12.10%，标准化 reader MSE +9.25%。本版未显示所需 motor 收益，停止后续实验，保留原 C8 A+Z incumbent。缺少匹配 C80 对照，不能识别 KL 的独立贡献，也不能判定原同预算 gate。

本轮先提交 KL 组，再提交 A+Z 组。两组从同一个原始 C8 VQ banks 初始化，使用原 C80 cases2000–2079、每例原 seed0、四 suite 交错顺序，一遍共80次更新；Adam lr5e-4，update1 后清 optimizer state，norm clip1，rho1.5 与原 normalization_c 不变。固定 final80，不按 held-out 选择 checkpoint。

只训练原614个 Linears 的 FP32 master codebooks/log row gains，indices、B32 exponents、Smooth0.5/H128、普通 dynamic A4、BF16 KV 和 backbone 固定。两组训练主目标都是 first10 motor6 MSE + rho*c*future global NMSE，排除 observed frame0；KL组只通过 student future 增加冻结原始 BF16 reader 的 soft-bin KL。无 activation clip、无 activation per-channel scaling。

复用现有 calibration80/held-out80 BF16 teacher，以及已完成 C80 teacher first10×6 population mean/std。KL 的 bins16、sigma0.375、边界[-3.5,3.5]不变；beta仅在训练前用完整 C80 的初始 reader KL 和 NMSE 配平一次，辅助项初始均值为原加权NMSE的10%。原8个初始化 future可复用；其余72个初始 student queries计入新成本。C80 与 C8 同时改变样本覆盖、更新数和标准化统计，本轮比较的是完整 C80 recipe，不能单独归因于数量。

每组最终评估完整 calibration80×seed0 和 held-out40×2seeds，均保存实际 student action/future及相同 observation/noise 下的 frozen-reader action。报告实际 motor RMSE（first10/full32/late22）、gripper RMSE、future NMSE、reader KL、标准化 reader MSE、reader motor RMSE，以及量化reader差异与future影响的交叉项。Reader自身指标仅用于机制解释，不能替代实际student主指标。

主判断：KL相对匹配 C80 A+Z 的 held-out first10 motor均值和case配对中位差均下降，至少21/40 cases改善，且future NMSE均值不退步。仅完整cohort可作结论；case内平均两个seeds，再在suite内平均，最后四suite等权。两组各一次训练，两个sampler seeds不是独立训练复现；原held-out是已反复查看的开发/回归集合。

未通过则停止本配方，不扫参数；通过也不自动开展ablation或280闭环。原C8 A+Z rho1.5继续作为历史参照，不能把本轮KL相对旧C8的变化全部归给KL。

复用现有分阶段VJP和BF16 reader实现。必要检查为：真实PBS allocation、现有teacher/统计绑定、同allocation loss/链式法则检查、共同原始初始化parity、每步student重放parity、首步实际部署变化、最终infer parity和teacher-reader parity；训练每步保存checkpoint，评估逐query保存。所有tensor/模型计算只在真实compute allocation内；login只执行轻量控制与有界小文件操作。

GPU任务申请12h、单GPU，normal路由后启动时要求实际execution queue为g1；GPU利用率与显存每15秒写入job.log。只读监控subagent使用gpt-6-luna/xhigh，每600秒检查指定handles；排队/运行无实质变化保持安静。出现终态、失败或需要处理时通知主agent；监控本身不重提作业、不更改训练。

本地控制入口：`nscc-access/.venv/Scripts/python.exe experiment/idea-validation/fastwam-vq-c80-frozen-reader-kl/control.py`。顺序为prepare(CPU)→先提交reader_kl(GPU)→提交az(GPU)→aggregate(CPU afterok双组)。调度等待不是实验结论，提交后必须继续核对当前权威状态。
