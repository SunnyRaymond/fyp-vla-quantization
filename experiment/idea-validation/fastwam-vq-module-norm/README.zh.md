# 逐模块匹配更新范数：latent protection 的方向与分配归因

**已完成（2026-10-17）**：完整720条件、60contexts及成本分析完成。小步untied的motor收益依赖模块预算，相对零新增bpw的Z边际改善小且不确定；大步在两个预算下均失败。完整解释和存储判断见 [FINAL_REPORT.zh.md](FINAL_REPORT.zh.md)。

问题：上一轮小步 untied 的改善，在逐模块匹配实际 decoded-BF16 更新范数后是否仍存在？若存在，保留独立 weights 的存储与 effective bpw 是否值得？

实验在原 action-only epoch2 checkpoint、原 calibration-only 选中的4个模块/32个 book0 codewords上进行。复用原8个 calibration contexts 的完整成员梯度，不新增 backward，不重新选择位置。Indices、book1、scales及其余权重固定，H128/fake A4与原部署数值路径一致。

采用 2 allocation profiles × 3 directions × 2 epsilons 的配对实验。S-budget 为上一轮 shared 在各模块实际产生的更新范数；U-budget 为上一轮 untied 的实际更新范数。每一种 profile/epsilon 下，shared、untied、random 在每个模块分别匹配实际 BF16 更新范数：相对目标误差不超过0.5%，三方向最大相对差不超过1%。两个 profile 都由已有 calibration 梯度和预定范数产生，不按 selection/test 结果选择。随机方向复用上一轮固定方向；它不是随机方向分布的统计样本。

主分析 epsilon=0.005；保留 epsilon=0.015 作为步长稳定性检查。两组 profile 的总更新范数由原实际 anchor 决定，约为原epsilon，报告所有实际值，不假称逐 bit相同。所有 profile/方向采用同一 support。在同一 profile内比较 untied-shared 检查更新几何/共享自由度的剩余影响；同一方向跨 profile 比较检查模块间分配影响。有限非线性可能产生交互，不强行把总收益分为唯一百分比。

完整60 contexts：8 calibration seed0、4 selection cases×2seeds、22 test cases×2seeds。先平均每个case内的seeds，再对cases等权平均，test n=22。当前test已经在上一轮读过，本轮属于内部开发归因，不作为新的独立泛化验证。报告future NMSE、固定完整BF16 reader的first10/full32/late22/gripper RMSE及native action。均为teacher fidelity，无closed-loop episodes。

12 arms共720条件。至少S-budget/shared的120条件须先通过CPU实际权重bitwise一致才能复用旧记录；其他条件也仅在完全一致时复用。各方向先匹配共同可实现的BF16范数，实际复用数量由CPU验证决定，剩余条件全部新native generation及固定BF16 teacher-reader replay。复用不缩减任何cohort或reference arm。每个case的两个seeds同分片，四个GPU分片各负责192/192/168/168最终条件，新条件数由实际bitwise复用数决定。

分析同时报告两种预算中的untied-shared、untied-random及固定方向的U-budget−S-budget，使用case配对均值/中位数/胜负数和描述性case bootstrap区间。Calibration与selection完整呈现，不依据test挑选成功条件。更换profile后，shared若达到原untied收益，支持优先调整更新分配；untied若仍改善，支持继续考察共享子空间，但不直接证明同bits assignment有效。

存储核算采用明确可序列化表示：shared将现有BF16 codebook数值原位替换，参数表示尺寸不增加。Untied/random保留全部基础VQ indices/codebooks/scales，以模块ID和book0 codewords作为selector恢复位置，额外保存完整选中support的BF16绝对weights；写入实际二进制overlay并进行bitwise round-trip。统计payload和metadata，不隐去位置/selector成本。报告完整614目标weights、选中4个模块和选中support三个分母的额外bpw，同时列出基础transform负担。整个policy的非目标参数/偏置、训练master/optimizer状态及torch.save容器不纳入该target-weight bpw。训练更新预算不需要保存到推理模型。

最终的价值判断比较同预算下的改善、逐case稳定性和实际新增bytes。若零新增bytes的shared分配调整已能获得同等收益，不优先增加untied存储；若方向收益保留而完整模型增量很小，可保留稀疏保护的候选，但closed-loop与实际decoder/kernel成本仍须后续验证。无论结果如何，不把存储压缩推导成当前decoded-BF16实现的VRAM/速度收益。

所有tensor读取、计算、模型推理、序列化与重IO仅在真实PBS compute allocation。Login仅小控制文件、PBS查询和已知有界日志/报告；保留SSH host-key验证，不输出credentials，不使用banked reset。CPU30min，GPU每作业最多2h，逐query保存，只有经核实成功RESUME_READY才能续交。GPU利用率和显存每15s写入job log。独立gpt-6-luna/xhigh subagent每约10min只读监控，无变化的Q/R静默，不提交/取消任务。

使用experimental-design和statistical-analysis skills组织配对设计与描述性分析。程序知识参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065 （已核实2026-09-02修订版v2，无已列出的journal reference）。

存储价值比较另保留原action+latent checkpoint Z的完整固定BF16 reader记录，作为零新增bpw参考；不新增模型查询，未为它补算future NMSE或native action，缺失指标不填零。

## 新模型forward前的数值可行性修订

CPU prepare `25848072.pbs101` 的toy数学/overlay测试通过；实际ubudget/shared小步的text模块沿固定方向可达到的最近BF16更新范数仍偏离原目标0.9749276%，超过0.5%容差，主动退出，尚无新model forward。为保持逐模块范数匹配，把每个原calibration-derived模块预算投到同一shared方向的最近可实现BF16范数（与原预算的变化不超过2%），再匹配untied/random到这个共同目标；逐arm0.5%和pairwise1%容差保持不变。完整记录原预算、共同预算和差值；不按future/action结果调整。可能减少旧untied条件的bitwise复用，所有未完全相同的条件重新评估；cohort和12arms均不变。
