# Analytic Padding Flow：冻结的最小 screen

2026-09-13，未采集本候选输出。裁剪 pipeline 保留 primary prior、实际接口审查、C02 受控干预、独立反证及一次 bounded screen。此处选择可直接计算的 analytic replacement；先前文档中的双模型 hybrid oracle 不实施。

研究假设：SmolVLA 的未执行 action coordinates 虽然不发给 robot，却参与 32 维 flow recursion；它们的 quantization error 可能反馈至前 7 维。若 padded target 恒为零，则当前时间约定下 `x_pad(t)=t*z_pad`，理想 padded velocity 恒为本次初始 `z_pad`。将预测的 padded velocity 替换为该已知值，可能减少 physical action 的 FP/Q drift。不是把 padded coordinates 当成没有训练过，也不声称这是新 flow 理论。

## 固定身份与样本

使用既有 `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de` 和 `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`、LeRobot 0.4.4、原始 stored weights 转 FP32。当前 metadata 确认 physical action 7、max action 32，state 保留 8 后按 official processor 补到 32；两个真实 cameras，official rename/normalization/tokenization。

固定 task IDs 0..3，每 task 排除 Flow64763 已用的 3 个 episodes 后，按 episode_index 取最早 2 个未用 episodes；frame=floor(length/4)。总计 8 个新 episodes。只用现有下载文件；若所选 video 不可用，不替换样本，记 resource_blocked。选择由 metadata 决定，与模型输出无关。noise seeds 固定 1801、1802；每 seed 一个 `(1,50,32)` standard-normal tensor，跨 episodes/arms 相同，先逐 episode 平均两 seeds 的指标。

## 四臂与干预

arm order：`FPnative, Qnative, FPanalytic, Qanalytic`。Q 仅使用原 flow screen 的 **expert transformer Linear W4**，signed [-7,7]、per-output RTN dequant FP32；language backbone、vision、action input/output projections 与其他参数保持 FP32，不搜索 bit/layer。

同一 official sampler、10 个 Euler steps、50-step action chunk，不启用 RTC、compile、AMP 或训练。每次 denoise_step 先在当前完整 `x_t` 上调用本 arm 的模型；analytic arms 仅将返回 velocity 的 `[...,7:32]` 替换为保存的本次初始 noise 对应 slice，`[...,0:7]` 原样返回。不能拼接另一个 branch 的预测，不调用 FP oracle 来修复 Q，不改 physical velocity，不在最终输出后单独置零伪装过程干预。

保存每个 episode/seed/arm 的全部 10 步原始 predicted velocities、实际 used velocities、step input states、sampler 返回的 physical actions，以及初始 noise。native arms 原始与 used velocity 完全相同；analytic arms physical slice 完全相同，padding slice 等于初始 noise。前 7 维首次 velocity 必须在 FPnative/FPanalytic、Qnative/Qanalytic 各对内逐元素相等。记录 actual time 值；analytical padded path 对 `t*z_pad` 的最大绝对差须 <=1e-5，仅作 FP32 工程容差。

## 预先定义的数值 gate

只用前 8 chunk positions、7 physical coordinates 的 normalized action MSE，不反归一化成 physical success proxy。令每 episode 在两 seeds 内平均后：

- `E0=MSE(Qnative, FPnative)`；
- `S=MSE(FPanalytic, FPnative)`；
- `E1=MSE(Qanalytic, FPnative)`；
- `E2=MSE(Qanalytic, FPanalytic)`。

episode binding：`E0>1e-6`、`S<=0.01*E0`，且第一步同一输入状态下 Qnative 与 FPnative 的 padded velocity MSE `P0>1e-10`（两 seeds 内平均、完整50×25 slice）。这要求原 Q drift 和量化对 padded field 的扰动非退化，且 analytic 操作对 FP reference 的改变足够小。只有至少 6/8 episodes binding，才解释保真收益；否则 `inconclusive_binding` 并列明原因，不得降低门槛。

binding episodes 上计算 `G1=1-E1/E0`、`G2=1-E2/E0`。两种 gain 的中位数均须 >=0.25，且全部 8 episodes 中至少 6 个满足 binding、G1>0、G2>0，才为 `preliminary_go`；binding 足够但收益未过为 `method_no_go`。1e-12 只用于比较容差。所有 episodes 的四个误差及 gain 保留，不能只报通过的子集。两个 reference 同时使用是为了避免仅靠移动 FP reference 制造改善。

阈值是经济筛查，不是统计显著性、普适性或方法 novelty 认证。即使通过，也只支持当前 expert W4 recipe 的 offline FP-fidelity potential；不推断机器人 success 或速度提升。

独立方法审查在采集前确认推导可执行。这里称 analytic padded-coordinate intervention/sampler ablation：它不需要 FP oracle forward，但其实际成功率与部署收益均未验证。RePaint/已知坐标约束是邻域 prior，不能声称首次提出 constrained denoising。

## 工程与停机

真实 CCDS SLURM allocation 和 V100；先 guard 后全部数据/模型/hash/IO；单写者 CPU preparation，无新下载/安装。记录 checkpoint/base/dataset/helper/actual runtime source hashes。strict checkpoint load，exact weight restore，官方 sampler no-op wrapper gate，四臂独立初始 noise clone，所有 raw finite/shape 完整性。每阶段先持久化小 engineering.json，full detail 独立存档；不能因 summary 大小上限丢掉完整结果。

最多一次 10min CPU preparation、一次 15min 单 V100 screen（内部 12min deadline）、一次 5min CPU raw replay。若出现实际 implementation failure，保留原 job、定位后只修同协议接线问题。科学 method_no_go 不换 bit、sample、seed、threshold、padding slice 或 calibration；positive 也不扩验。所有结论保留。
