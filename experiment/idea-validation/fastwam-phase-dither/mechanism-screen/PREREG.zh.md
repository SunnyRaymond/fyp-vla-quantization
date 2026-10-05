# 冻结的最小反证：FastWAM full-path W4A8 phase pilot

冻结时间：2026-10-03，首次模型结果产生前。研究问题是：一个廉价、可部署的共享相位表，是否有足够 held-out action 收益值得扩大搜索？本轮只检验以下具体粗粒度 recipe，不能否定不受限的逐 site 相位最优解。

## 模型与干预

- 官方 released Optional IDM clean checkpoint，FastWAM 源码版本 `7faa71108368fbb3b6885649f112af607427a2d4`；复用已验证 checkpoint，不重新 hash 12 GB 文件。远程源码为无 .git 的固定 snapshot，在 allocation 内复用已有小型 source identity 检查。
- `first_frame`，BF16 reference，sigma_shift=1，20 inference steps，compile=false，action normalization 与原 evaluator 一致。主指标是反归一化前完整 action chunk MSE。
- 所有实际执行的 video/action denoiser Linear 和 proprio Linear 使用 W4 grouped input columns，G=128，signed [-7,7]，zero point=0，max/7 scale；不够128的尾组按实际长度处理。这是 fake quantization，未验证 packed kernel。
- A8 全路径逐 token row `Δ=maxabs/126`。RTN 与 phase 使用同一 Δ；subtractive dither `xhat=Δ(round(x/Δ+frac(U+φ))-frac(U+φ))`，zero rows 直接为0。算术FP32后cast BF16。非 Linear、encoder、VAE 等保持原精度。
- 不改变干预范围，只将 phase 参数绑定为 video/action 各自 condition、early/middle/late 三个 block 段，另加 proprio。video.condition 相位固定0去掉共同平移 gauge。末层 video 无 action 因果路径的 Q/O/cross-attn/FFN phase 固定0，不参与搜索，但仍 W4A8。

## 数据与配对

只使用 LIBERO-goal task0 的新 BF16 first_frame 短轨迹。每个轨迹记录 warmup30 dummy steps 后及随后10 BF16 actions 后两次 observations。CAL initial-state IDs4,5；DEV6,7；锁定TEST8..15，共8独立轨迹。每次记录相同 images/proprio/prompt 后离线比较所有量化 arm；量化输出不驱动后续 observations。

配对 action sampler seeds2026,2027。CAL dither seeds101,102；DEV201,202；TEST1101..1104，交替配对两 sampler seeds。每个 arm 的随机相关结构按其定义实现；不能把帧、actions 或 dither draws 当独立样本。数据只代表本 task 的固定 observations，不是量化 policy 的闭环 history。

## 校准与对照

1. 先完成全部 BF16 labels、重复一致性和 hook-disabled identity gate，再一次性 inplace W4，不常驻 BF16 weight 副本。
2. phase 从全0开始，按固定排序的非 gauge group 做2轮周期坐标搜索；每坐标候选0,.25,.5,.75；每候选完整重跑4 CAL observations×2配对 draws。不使用 STE/local oracle 代替 endpoint。
3. direct endpoint PTQ 使用同样 group、同样 sweep/candidate/forward 数，搜索 A8 scale multiplier `.94,.98,1,1.02`，其 RTN clipping 在实现中明确记录。W4 weights 与其它设置固定。
4. 每种方法从两 sweep checkpoint 中只用 DEV 选择一次；TEST 不更新 phase、scale、搜索预算或阈值。
5. TEST arms：W4+原激活（权重误差诊断）；普通RTN；independent subtractive dither；共享U全零phase；learned phase；保持 learned phase 值多重集的非 gauge site permutation；同预算 direct endpoint PTQ。

## 决策与停止

以 TEST 轨迹为统计单位，先平均各轨迹内的两个 observations/paired draws。最强比较基线是 RTN、independent dither、direct endpoint PTQ 中 TEST 平均 MSE 最低者。仅当 learned phase 同时满足以下条件才扩大搜索：平均 MSE 至少降低10%；至少6/8轨迹改善；轨迹 bootstrap 95% CI 的绝对 MSE 改善下界>0；permutation 消去优势（permuted MSE 更高且至少6/8轨迹有同方向劣化）；permutation 不是 gauge。否则本粗粒度 recipe 不扩大实验，报告结果的统计不确定性，不声称整个 phase 家族不可能有效。

preflight 只检查加载、hook identity、finite/range/zero-row、full-path coverage、两种 phase 是否改变 endpoint；不据它修改本协议。工程 gate 失败则修复实现后重跑，不将其计为机制 NO-GO。GPU pilot 上限2h；超时只算未完成。每15秒记录 allocated GPU utilization/显存到 job.log。所有模型/数据 I/O、计算在真实 PBS allocation 内，login 只轻量控制。

本轮不运行 W4A4、native kernel、闭环、其它 task，且不作 acceleration 或 SOTA claim。成功只支持开展更充分的独立复验。

## 流程来源

配对与 trajectory 分组留出的研究设计参考 `experimental-design` skill。流程引用：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv](https://arxiv.org/abs/2609.00065), [DOI](https://doi.org/10.48550/arXiv.2609.00065)。2026-10-03核对最新记录为v2；这条引用是工具流程来源，不是 WAM 相位方法有效性的证据。
