# WAM W4A8 相位量化：最小反证结果

**结论：有小而一致的真实模型正信号，但本粗粒度 recipe 不达到事前设定的扩展门槛，记为 `NO_GO_FOR_EXPANSION`。停止本 recipe 的进一步实验，不降低门槛。**

用户随后额外授权的输出误差分解与单task配对闭环已完成，见 [独立follow-up报告](../paired-followup/RESULT.zh.md)。后续结果不改写本轮已冻结的NO-GO；以下“未开展闭环”等范围声明仅指本次原始mechanism screen。

learned phase 在8条留出轨迹上均优于预注册主比较中最强的 independent dither；平均完整 action-chunk MSE 降低 **5.27696%**，没有达到继续投入要求的 **10%**。10% 是本轮事前冻结的资源投入门槛，不是相位量化“有效/无效”的数学分界。这里的 NO-GO 不能被解释为“数学机制不存在”。

## 数学依据与实际检验

固定输入、固定 action 投影、无 clipping 的理想条件下，subtractive-dither residual 的相关函数为

$$R(\tau)=\frac1{12}-\frac{\tau(1-\tau)}2,\quad 0\le\tau<1.$$

相位可以改变跨位点协方差；只有在 observations 上平均后仍非零的 action-projected cross-spectrum，才会令局部相位目标非恒定。此时存在优于 independent dither 的相位表，但不保证优于 RTN。真实全路径的下游输入和 row scales 依赖共享 draw，局部证明不能直接外推。

源码给出具体入口：首个 time-conditioning Linear 的输入由固定 scheduler 的 sinusoidal timesteps 决定，跨 observations 不变。这使“归一化输入相位跨 observations 自动均匀化”的反例不必然适用于这些位点；平均 action 投影内积仍需实测。完整推导及源码位置见 [MATH_GATE.zh.md](MATH_GATE.zh.md) 和 [ARCHITECTURE_GATE.zh.md](ARCHITECTURE_GATE.zh.md)。

本轮检验完整 `first_frame` action 推理路径的 W4A8，而非单层 toy：全部实际执行的 denoiser/proprio Linear 都经量化，覆盖每次 inference 的 **6,446 次 Linear 调用**。为了实现有界校准，相位参数绑定为8个自由 groups；不能用它的失败否定不受限的逐 site phase 最优解。量化采用 FP32 activation reconstruction 后 cast BF16 的 fake-quant proxy，没有验证 packed INT4×INT8 GEMM 或 fused FP32 column-sum epilogue。

## 冻结协议与执行证据

| 项目 | 实际执行 |
|---|---|
| 模型 | 官方 Optional IDM clean checkpoint，`first_frame`，BF16 reference |
| 源码 | 固定 FastWAM snapshot；allocation 内复用入口源码/config identity 检查 |
| 配置 | sigma_shift=1；20 inference steps；compile=false；32-action chunk；输入224×448 |
| 量化 | groupwise signed W4 [-7,7]，G=128；逐 token row A8 maxabs/126；非 Linear、encoder/VAE 保持原精度 |
| 数据 | LIBERO-goal task0；新 BF16 短轨迹；每轨迹2次固定 observations |
| 划分 | CAL IDs4,5；DEV6,7；锁定 TEST8..15 |
| 校准 | phase与direct PTQ各2 sweeps、64 endpoint候选、**512 full forwards** |
| DEV | 两种方法各16 full forwards，选择一次后锁定参数 |
| TEST | 7 arms ×8轨迹×2 observations×4 paired draws = **448 rows**；每arm64 rows |
| 实际 precision | `torch.bfloat16`；BF16 repeat与hook-disabled identity最大误差均0 |
| 执行 | A100-SXM4-40GB；job `25666715.pbs101`；walltime **01:49:36** |
| 终态 | PBS `F`，`Exit_status=0`，作业 `exit_code.txt=0`，`PIPELINE_COMPLETE` 存在 |

完整推理调用总计1,554：BF16 50、phase 720、RTN/direct 656、independent 64、W4诊断64。相位和 direct PTQ 的校准预算分别经两份 calibration history 核对。TEST 中 phase/independent/shared0/permuted 均无 activation code 越界。每15秒 GPU utilization/显存采样保留在作业日志。

**交付 caveat：PBS `Stageout_status=1`。** 标准 stageout 异常原因未判定；scratch 下的 result、完整 job.log、退出码、完成 marker 及关键配置已直接取回本地。不能把它描述成所有 scheduler delivery checks 均通过；它没有造成本次分析所需记录缺失。

## 留出结果

指标是反归一化前32×7完整 action chunk 对 BF16 reference 的 MSE。每条轨迹先平均内部 observations/draws，再平均8条轨迹；没有把 actions 或 draws 当作独立样本。

| Arm | 平均 MSE ×10⁻⁴ | 作用 |
|---|---:|---|
| W4 + 原激活 | 3.61510 | 权重量化诊断，非同位宽比较基线 |
| 普通 RTN W4A8 | 3.85256 | 主比较基线 |
| Independent dither | 3.78977 | **预注册主比较中最强基线** |
| Shared U，phase全0 | 3.75674 | 无学习的共享 dither control |
| **Learned phase** | **3.58978** | 本方法 |
| Learned phase，site permutation | 3.68984 | phase-site 对应关系 control |
| 同预算 direct endpoint PTQ | 3.88731 | 主比较基线 |

learned相对independent的绝对 MSE 改善为 **1.99985×10⁻⁵**，trajectory bootstrap 95% CI 为 **[1.34562×10⁻⁵, 2.69200×10⁻⁵]**；8/8轨迹改善。四个冻结 TEST draws 在 observations 间复用，因此该区间是给定这组 draws 的条件性区间，不覆盖重新抽取整组 seeds 的方差。

Shared0 的均值略好于 independent。learned相对这个更简单的control只有约 **4.44%** 改善；不能把 independent 称为所有control中的最佳者。置换phase-site对应关系后，8/8轨迹的误差升高，平均收益被削弱；置换arm仍优于independent，所以不能写成“置换完全消除了收益”。

| 冻结门槛 | 结果 |
|---|---|
| 对最强主比较基线平均改善≥10% | **失败：5.27696%** |
| 至少6/8 TEST轨迹改善 | 通过：8/8 |
| trajectory bootstrap绝对改善下界>0 | 通过 |
| site permutation非共同平移gauge | 通过 |
| permutation MSE更高且至少6/8轨迹劣化 | 通过：8/8 |

## 值不值得继续

这次证据支持“相位表能改变真实联合量化路径，特定表在这些留出 observations/draws 上有小幅稳定收益”。它不支持把来源唯一归因于 cross-term cancellation：draw-dependent bias和BF16 reconstruction rounding也可能参与。

**当前建议是停止这一粗粒度 recipe 的扩展。** 独立dither已提供接近的结果，learned表在此基础上增益约5%，且没有达到事前决定投入后续阶段的标准。保留这个正信号和可复验材料，不凭它进入W4A4、native kernel或闭环 campaign。若将来提出针对未覆盖自由度的新假设，应单独授权并冻结新协议；本轮不据此继续搜索。

这一结论限定于单个 checkpoint、first_frame route、LIBERO-goal task0、固定observations与冻结seeds、8组相位及本次forward预算。没有证明整个phase家族无效，也没有证明 task success、SOTA、native latency/memory或加速收益。

## 结果材料

- [原始结果与448 rows](pilot_result.json)
- [作业完整日志与GPU采样](pilot_job.log)
- [实际配置](pilot_runtime_config.json)
- [BF16 repeat/identity gate](pilot_reference_gate.json)
- [Phase校准history](pilot_phase_calibration.json)、[Direct PTQ校准history](pilot_rtn_calibration.json)
- [冻结协议](PREREG.zh.md)、[当前运行记录](RUN_STATE.json)
- 远程完整原始 observations 与大 coverage 表保留在 `/scratch/users/ntu/yguo017/wam-phase-screen-20261003/artifacts/25666715.pbs101/`，不经login传输大文件。

## 来源

数学推导为本轮分析；文献出发点是 [Q-WAM Appendix A.2](https://arxiv.org/html/2609.33269v1)，其被忽略的跨点项不等于已证明有可用收益。固定版本只用于定位研究假设，不把论文报告当作本地实验。

研究设计流程使用 `experimental-design` skill，其来源引用为 Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv](https://arxiv.org/abs/2609.00065), [DOI](https://doi.org/10.48550/arXiv.2609.00065)。本轮已核对最新记录v2；该引用是流程来源，不是量化机制证据。
