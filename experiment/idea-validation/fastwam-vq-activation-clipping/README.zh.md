# C8 rho1.5 activation clipping 四组对照

## 已完成：未达到 channel scaling 推进门槛

四组均完成同一 held-out 的80 queries；后三组均完成固定额外8 updates。相对原 C8 rho1.5 checkpoint，weight_only 的 first10 motor RMSE / future latent NMSE 分别增加2.410% / 3.179%，clip_only 增加2.569% / 12.884%，joint 增加0.517% / 0.211%（均为误差增加）。因此预设 gate 为 false，未提交 learnable channel scaling。

joint 优于等预算 weight_only，但没有超过原 checkpoint；这说明本轮联合 clipping 缓解了继续 weight 训练的退化，尚未带来净收益。该结论只适用于本次 clipping 参数化、学习率及8-update预算，不能据此否定其他 activation 优化路线。

完整数值见 [RESULTS.zh.md](results/RESULTS.zh.md)，各 suite 与原始精度汇总见 [comparison.json](results/comparison.json)。[completion_evidence.json](completion_evidence.json) 记录覆盖、训练预算、初始 parity、梯度连接、GPU selfcheck、作业时长与补验证来源。本地最终检查没有新增模型 forward 或 hash 检查。

CPU aggregation `25850887.pbs101` 已以 PBS/artifact exit0 和完成 marker 结束。三个训练作业也均为 exit0；GPU作业实际时长最长1h16m55s，均低于2h，运行期间每15秒将 GPU utilization/显存采样写入各自 job.log。gpt-6-luna/xhigh 的600秒监控已停止。

## 冻结方案

四组统一从已完成 C8 rho1.5、update8 的 FP32 masters 初始化，实际 forward 使用 BF16 stored codebooks/row gains。源 GPU job 为 `25821658.pbs101`，完成补验证为 `25822474.pbs101`。源 checkpoint 与旧实验只读。

| Arm | 额外 weight 更新 | 额外 clipping 更新 |
|---|---:|---:|
| baseline | 0 | 0 |
| weight_only | 8 | 0 |
| clip_only | 0 | 8 |
| joint | 8 | 8 |

后三组均用完整 C8、同一顺序与 seed、fresh Adam、update1 后同样清空 Adam state，最后固定 update8，不按 held-out 选择 checkpoint。Weight LR=5e-4；clipping LR=1e-2，均在运行前固定。rho=1.5、normalization_c=0.028303017887797623。Loss 保留 first10 六维 motor MSE 与排除 clamped frame0 的 clean future global NMSE。

每个 target Linear 一个 FP32 clipping ratio，初始1，投影到[0.5,1]。A4 step 为 ratio×per-token absmax/7，输出仍为 BF16 dequantized operands。使用 LSQ 风格 scale surrogate，输入梯度仅通过 clipping 范围，absmax 视作常量。冻结 Smooth0.5/H128、VQ indices、B32 exponents 和非目标参数。

Held-out 为原 frozen original-only 40 contexts（3000..3039），四 suite 各10、每case两seeds，共80 queries/arm；demonstration 与 calibration 分离。复用 C8 calibration teacher，单独生成共享 held-out BF16 teacher action+future，每个完整 query 持久化。按 seed→case→suite→四suite均值聚合，不把tokens或frames当独立样本。

先提交一张 GPU 的 teacher+baseline preparation，完成后提交三张单GPU训练/评估任务。每个 GPU allocation 为1h59m，6840s软截止，逐update和query持久化，可合法续跑，不缩小cohort或训练预算。GPU utilization与显存每15s写入该job的job.log。最后在CPU allocation聚合。

进入 channel scaling 的预设工程筛选：clip_only 的 held-out first10 motor RMSE、future latent NMSE 同时优于baseline；或joint同时优于baseline及等预算weight_only。结果须完整80queries。该gate仅用于下一阶段探索，不构成显著性、闭环成功或unseen-task证据；后续选择与确认需区分。所有实现仍是decoded BF16/A4，不声明native INT4加速。

本地只做标准库控制逻辑/语法检查。A4 ratio1逐位一致性、H128反向一致性、clipping梯度与零范围检查均放在真实GPU allocation。Login node仅小文件上传/控制、提交、查询；禁止模型、计算、依赖安装、重I/O或大文件读取。禁止使用banked reset。

监控使用gpt-6-luna/xhigh subagent，600秒间隔；Q/R未变保持安静，query失败视为unknown，不自行重交。以当前handles、PBS终态、artifact exit、完整summary交叉确认完成；soft-deadline由主任务决定续跑。

## 首阶段收尾修复

GPU作业`25848310.pbs101`已完成共享teacher80和baseline80，初始checkpoint action parity max_abs=0，clipping GPU selfcheck通过。Baseline first10 motor RMSE=0.04429677443113178、future latent NMSE=0.060877370962407445。数值结果写完后，错误的`runtime.close()`调用触发AttributeError，故原PBS/artifact真实exit仍为1，未改写成功标记。

已改为原C8 trainer使用的“env存在时env.close()”。CPU-only `25848931.pbs101`（10min）独立补验证已通过：PBS/artifact exit0、完成marker、从80个query records重新聚合并核对summary、teacher覆盖与selfcheck齐全，零模型加载、零新forward。原GPU失败记录保持不变，独立validation.json绑定原job。

2026-10-17 12:27 SGT提交三臂：weight_only=`25848988.pbs101`、clip_only=`25848989.pbs101`、joint=`25848990.pbs101`。每臂1GPU/1h59m、固定额外8updates与完整80queries，现已全部完成；Luna subagent按600s间隔监控至CPU汇总完成后停止。
