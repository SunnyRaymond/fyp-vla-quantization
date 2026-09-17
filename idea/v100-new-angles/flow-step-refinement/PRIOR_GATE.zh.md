# PRIOR GATE：SmolVLA numerical PTQ × flow integration step refinement

审查日期：2026-09-13。当前实验基线固定为 LeRobot v0.4.4，不是 main。
范围限定为已有 lerobot/smolvla_libero 与 lerobot/libero 资产；本轮不下载、不
运行模型、不提交 cluster job，也不修改 flow-geometry-drift。

## 三个独立判断

| 维度 | 判断 | 含义 |
|---|---|---|
| Prior novelty status | **novelty no-go** | QuantWAMs 已有 step-specific、state-conditioned 的 PTQ intervention；本切面的 5/10/20 endpoint decomposition 在本次审阅的 primary sources 中未见，但不足以构成新 recipe 或已认证 novelty。 |
| Narrow diagnostic identifiability | **conditional-go** | 固定输入与初始 noise 的 endpoint pattern 可以测；要把它解释成 quantization bias accumulation，必须有 intermediate state/velocity 或共同路径 field evidence。 |
| 是否值得本轮 GPU | **不单独申请；已有 allocation 时可 piggyback** | 只有 preflight 证明 v0.4.4 接口、W4 target 和 intermediate capture 后，才做 12-sample 静态 screen；不做额外 rollout 或大实验。 |

## Primary prior 与重叠边界

QuantWAMs 的 [primary paper](https://arxiv.org/abs/2607.28405) 不是可忽略的
“未找到”结果：其 fixed-intervention rollout auditing 在记录的 FP16 reachable
states 上按 inner denoising step 比较 unprotected low-bit field 与 FP field，
恢复 immutable snapshot、匹配 stochastic seed，并据此选择 denoising-step
protection schedule。它与本切面直接重叠于“量化误差依赖 denoising step 和可达状态”
这一机制问题；但它保持 step count 和 precision budget，研究 protected index
placement，未做同一初始 noise 下 FP/W4 的 K=5/10/20 Euler refinement，也未用
每个 arm 的 20-step endpoint 做数值 anchor。因此本切面仍有窄 diagnostic 差异，
没有独立方法 novelty。

[QuantVLA](https://arxiv.org/abs/2602.20309) 及其
[official run script](https://github.com/AIoT-MLSys-Lab/QuantVLA/blob/main/run_quantvla.sh)
已经把 denoising-step 数作为 quantized inference 的操作旋钮：脚本默认 8，
并提示 20 以及在精度下降时使用更高 step count。[Ω-QVLA](https://arxiv.org/abs/2605.28803)
及其 [official code](https://github.com/UCMP13753/Omega-QVLA)
对 DiT action head 使用跨 denoising step 的 activation-scale table。
QVLA 的 [primary paper](https://arxiv.org/abs/2602.03782) 讨论 temporal
action drift，但没有把 solver discretization error 与 quantized field bias
分解。因而不能把本提案包装成首次发现“steps 与 PTQ 有关”，也不能据此说
QuantWAMs 已经直接做了本提案的完整 endpoint experiment。

## 精确假设与 claim boundary

对固定 observation、instruction 和初始 noise，令 A_m^K 是 arm
m∈{FP32,W4} 用 K 个 Euler steps 得到的 action-chunk endpoint。假设是

    ||A_FP32^10 - A_FP32^20|| < ||A_FP32^5 - A_FP32^20||
    ||A_W4^10   - A_W4^20|| >= ||A_W4^5   - A_W4^20||

这里 20-step 只是每个 arm 自己的 numerical anchor，不是真实 action、ground
truth 或连续流的“真解”。即使方向成立，也只能说明两条 vector field 的
discretization behavior 不同；不能推出 W4 action 更差、task success 改善，
或存在可部署收益。

至少应区分三项：

* arm 内数值残差 I_m(K)=||A_m^K-A_m^20||，描述该 arm 相对自身 20-step
  endpoint 的 refinement；
* 同一 K 的 cross-arm drift B(K)=||A_W4^K-A_FP32^K||，描述 quantized
  与 FP32 endpoint 的差异；
* 共同状态路径上的局部 field discrepancy，例如
  D=mean_t ||v_W4(x_FP32^20(t),t)-v_FP32(x_FP32^20(t),t)||。

把 ||A_W4^K-A_FP32^20|| 当成单一 quantization error 会混入两种 arm 的
integration error。即使 B(10)>B(5)，也可能只是 Q field 在不同路径上的
exposure；没有中间 x_t、v_t 或共同路径 field evaluation，不能称为
quantization bias accumulation。

## LeRobot v0.4.4 的实际接口

当前 pin 的官方 [v0.4.4 modeling_smolvla.py](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py)
在同一个文件的 sample_actions 内读取 self.config.num_steps，设置
dt=-1/num_steps，逐 step 计算 t=1+step·dt，并执行 x_t=x_t+dt·v_t；
loop 位于约第 755--787 行。v0.4.4 的
[configuration_smolvla.py](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/configuration_smolvla.py)
将 num_steps 默认设为 10。

因此，v0.4.4 没有 main 版本所用的
src/lerobot/policies/common/flow_matching.py 路径；此前引用该路径不适用于
当前实验。v0.4.4 的 sampler 返回最终 x_t，不默认返回 intermediate
trajectory；只有 rtc_processor debug track 分支会记录中间值。任何 screen
都必须先证明 capture hook/wrapper 能保存每步 state 和 velocity，并且不会启用
RTC、adaptive behavior、dropout 或改变 KV-cache 语义。

## 条件式最小 paired screen（建议，不是冻结 protocol）

若 preflight 通过，可在已有 GPU allocation 中使用 12 个已准备 sample，逐
sample 跑六个条件：FP32/W4 × K={5,10,20}。所有条件固定同一 raw image、
state、instruction、normalization、action mapping、checkpoint、预先冻结的
W4 recipe 和同一个 initial noise；不调参、不训练、不做 rollout。保存六个
endpoint，并尽量保存各 K 的 x_t、v_t；按 sample 配对汇报
I_FP(5), I_FP(10)、I_Q(5), I_Q(10)、B(5), B(10), B(20)，以及共同 FP20
path 上的 D。12 个 sample 是 paired measurements，不应拆成 72 个独立样本，
也不应把 20-step 当 oracle。

一个可证伪的最小方向 gate 是：12 个 sample 中至少 9 个满足
I_FP(10)<I_FP(5)，且至少 9 个满足 I_Q(10)>=I_Q(5)；任一方向不满足即停止
该假设。方向通过后仍只能称为 arm-specific endpoint pattern。若没有
intermediate state/velocity 与共同路径 field discrepancy，应标记为
non-identifiable，不能升级为“量化偏置累积”或新机制。

## 最终边界

本文件的 **novelty gate 是 no-go**，**narrow diagnostic gate 是 conditional-go**，
而**本轮 GPU 决策是不单独启动，最多在已有 allocation 中 piggyback**。这三个
判断分别对应 prior 重叠、可识别性条件和资源决策；缺少新 quantization recipe
不等于诊断本身不可做。若 preflight 无法证明严格 paired noise、实际生效的 K
和 intermediate/common-path evidence，则不运行，并只保留 descriptive endpoint
结论。
