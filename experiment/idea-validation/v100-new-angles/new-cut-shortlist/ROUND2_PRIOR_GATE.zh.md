# ROUND2 PRIOR GATE：新的 WM/WAM/VLA numerical-quantization 切面

审查日期：2026-09-13。范围限于本地 campaign `INDEX.zh.md`、
`PRIOR_NO_GO_AUDIT.zh.md`、相关 RESULT/PRIOR 文件、DINO-WM source 与 primary
prior。没有运行模型、访问 cluster、下载或修改既有 pipeline。结论是 **1 个
conditional narrow diagnostic，1 个 engineering-only no-go**；不据此冻结正式
protocol。

## 1. DINO-WM predictor residual-branch interaction

**切面。** 本地 [`vit.py`](../../../reproduction/dino-wm-wall/source/models/vit.py)
的 predictor 每个
Transformer block 明确执行 `x = attn(x) + x`、再执行 `x = ff(x) + x`；两条
update branch 都含自己的 `LayerNorm` 与 `Linear`。候选不是按 layer reconstruction
挑敏感层，而是问：同一 hidden state 上 attention 与 FF 的 W4 perturbation 是否
存在可测的 cross-branch covariance，使完整 predictor 的 drift 不能由两个独立
branch drift 的平方和解释。

### 可证伪预测与最小 screen

对固定输入和同一 FP current state，记 `δA_l=A_W4-A_FP`、
`δF_l=F_W4-F_FP`，并定义四臂输出漂移
`E_A=||y_{W4-A,FP-F}-y_FP||`、`E_F=||y_{FP-A,W4-F}-y_FP||`、
`E_Q=||y_{W4-A,W4-F}-y_FP||`，`N=E_Q²-E_A²-E_F²`。预测是：至少 4/6 个
fresh paired observations 中，`|<δA_l,δF_l>|/(||δA_l||||δF_l||)` 的跨
block 聚合绝对值不少于 0.20，且 `|N|` 相对于一个破坏该配对关系、保留每个
branch 边缘 norm 的 zero-covariance oracle null 至少为 10%。若 branch
covariance 与 endpoint non-additivity 不同向，或 4/6 不成立，机制 no-go；不把
正负方向事后择优。

未来若有 allocation，最多使用 6 个新的同输入 snapshots、一次加载的 FP/W4
predictor 和四个轻量分支条件：FP/FP、W4/W4、W4-attention+FP-FF、
FP-attention+W4-FF。branch delta 必须在同一 current state hook 取得，不能把
不同 rollout 的 velocity 拼起来。再用预注册的 token-level sign/permutation
null 打散 cross-branch pairing、保留每个 delta 的 norm，作为外部 oracle；它
只回答 covariance 是否有下游后果，不是可部署 quantizer。四个主臂均使用同一
完整 W4 byte budget；若增加 branch-only cost control，则按 branch family 预先
挑选总量化 bytes 相等（差异不超过 1%）的 treatment/control，不能拿未匹配的
attention/FF 参数量直接比较。无法 byte match 就停止。最多 36 次 paired
forward/hook（6 observations × 4 主臂，加固定
null/byte-match 对照），目标是单张 V100 15 分钟内完成，保存 FP64 covariance、
`E_A,E_F,E_Q,N` 和 null 原值。

**Gate。** 先过工程 gate：FP/FP 与 no-op hook 相对误差 `<=1e-6`，所有 arm
输入、checkpoint、RTN recipe 相同，branch byte 差 `<=1%`。科学
`diagnostic-go` 同时要求上面的 4/6 covariance gate、4/6 endpoint-vs-null
gate，以及 W4/W4 的 endpoint non-additivity 在 FP objective 上不是纯数值
roundoff（FP64 denominator `>1e-12`）。否则 `no-go` 或 `inconclusive`；即使
通过也只说明 branch interaction 值得进一步研究，不推出 task success、native
kernel 或 branch-specific deployment recipe。

**与已做项目的边界。** 这不是 action candidate ranking、action-gradient、
noise marginal、flow step 数、padding coordinate、semantic input scale 或
batch scale；也不改 residual transport、LoRA 或 rollout solver，因此不重演
FRT/PRR。QuantWAMs 已覆盖 reachable state/denoising-step 的保护审计，但没有
在同一 current state 做 attention-vs-FF residual-branch covariance 分解。
DynamicPTQ 研究的是 LLM 的 W4A4 activation residual-stream dynamics，
`The Quantization Benefits of Residual-Free Transformers` 比较的是 residual 与
residual-free architecture；二者使本候选只有窄 diagnostic 差异，不支持宽泛
novelty claim。若 root 不需要一个内部机制诊断，应直接不申请 GPU。

Primary：

- [DynamicPTQ](https://arxiv.org/abs/2606.12487)：residual-stream update 的
  cross-layer dynamics 与 4-bit activation instability。
- [The Quantization Benefits of Residual-Free Transformers](https://arxiv.org/abs/2605.25880)：
  residual mixing、non-Gaussianity 与 low-bit degradation 的 controlled comparison。
- [QuantWAMs](https://arxiv.org/abs/2607.28405)：WAM PTQ 的 state/step/granularity
  邻域边界。

## 2. Hidden-basis permutation invariance：engineering-only no-go

在 DINO-WM MLP 中，第一 `Linear` 的 hidden rows、对应 bias 与第二 `Linear` 的
hidden columns 做同一 permutation，GELU 是逐元素的，所以这是严格的 FP
function-preserving reparameterization。当前 per-output absmax RTN 也应对该
permutation equivariant：配对量化后的输出只能发生对应坐标 permutation。预测是
FP 与 W4 两者都应保持不变；若 Q 输出变化超过 `1e-6` relative，首先说明
quantizer、weight restore 或 hook 实现错误，而不是一个科学 effect。

控制是未变换的同一模型/同一输入/同一 W4 snapshot；每次只做一次 paired
forward，预先固定 permutation，检查 FP no-op 与 Q equivariance。`<=1e-6`
通过只得到 null，失败则 engineering stop，绝不升级为 numerical-quantization
机制。它与 semantic input scale 和 branch treatment 不同，但没有可部署收益或
可解释 planner estimand；[RepQ-ViT](https://arxiv.org/abs/2212.08254) 与
[RepQuant](https://arxiv.org/abs/2402.05628) 已把 function-equivalent scale
reparameterization 作为 PTQ 方向，因此不值得独立 GPU screen。

## 决策记录

本轮不提出第三个候选。残差 branch 项若要做，只能作为已有 DINO-WM V100 能力
上的小型 C02/C04 诊断，并且不得沿用旧 DEV 结果选择输入；hidden-basis 项保留
为实现审计想法，不进入 campaign candidate 表。所有 broader claims、训练/QAT、
完整 rollout 与 Phase 4 均超出本文件范围。
