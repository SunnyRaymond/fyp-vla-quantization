# Value-head gauge：protocol 最终静态审查

## 审查边界与结论

本审查只读取 `PROTOCOL_DRAFT.zh.md`、同目录 prior 文档、现有 TDQ verifier 约定和 pinned TD-MPC2 source；没有读取模型输出，没有运行数值、模型、集群任务，也没有修改其他文件。结论是：**机制设计可 conditional-go，但当前草案不能直接冻结**。softmax gauge、row-wise RTN、no-op 检查和 8-state 统计门槛的主方向成立；下面四项执行语义必须在实验前写死，否则会出现 NaN、分母指错、或把上一臂的量化结果再次中心化。

## 已核对且成立的部分

1. pinned `WorldModel` 的 Q ensemble 最后一层是 `nn.Linear(512, 101)`，五个 member 由 `Ensemble` 向量化；官方 `Q(return_type="all")` 返回各 member 的 101-bin logits，`Q(return_type="avg")` 对两个 member 各自调用 `two_hot_inv` 后再取平均。[world_model.py](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py) [layers.py](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/layers.py)

2. 对每个 member 令 `W∈R^(101×512)`、`b∈R^101`，并明确 `μ_W = mean(W, dim=0)∈R^512`、`μ_b = mean(b)∈R`，中心化应写成

   `Wc = W - 1_101 μ_W^T`，`bc = b - μ_b 1_101`。

   对同一 hidden input `h`，`Wc h + bc = W h + b - (μ_W^T h + μ_b)1_101`。因此在 exact arithmetic 中 softmax probabilities 不变，官方 `two_hot_inv` 后的 scalar 也不变；这部分不是把各 bin 独立减一个常数。官方解码顺序见 [math.py](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/math.py)。文档中的“精确”应理解为代数等价；FP32 求均值、矩阵乘和减法仍有 roundoff，必须由 no-op gate 实测确认。

3. 对当前权重按 output row 定义 `s_j=max_i|W[j,i]|/7`，再用 `clamp(round(W[j,:]/s_j),-7,7)*s_j` 做 fake dequant，确实是 symmetric per-output-row W4 RTN。centered arm 应以 centered 的当前 `Wc` 重算 row scale；bias 保持 FP 是一个明确的设计选择，不会破坏 FP gauge 对照。`torch.round` 的 tie rule、clamp 和 dequant dtype 应在实现中保留，不应换成 raw-logit MSE 或其他 quantizer。

4. `FP-original` 与 `FP-centered` 的 probabilities 使用 `atol=1e-7, rtol=1e-6`，decoded member Q 使用 `atol=1e-5, rtol=1e-6`，作为 implementation gate 是可证伪且方向合理的。前提是两臂使用 bit-identical cached `z/action`、相同 FP32 dtype、`eval()`、无 dropout/随机路径，并调用同一官方 `two_hot_inv`；这些条件需要显式写入冻结版。失败应保持 `implementation_inconclusive`，不当作科学 no-go。

5. 8 个 state 上预先固定的 `6/8` 方向、median 相对改善 `≥25%` 和 aggregate mean 严格下降，配合 `A_original>1e-8` 的全量 non-degenerate gate，能够在不看结果调 seed/bit 的情况下被反驳。global binding 失败标成 `inconclusive_no_gauge`、state 退化不删除样本，这个证据边界是合适的。

## 冻结前的明确 blocker

### 1. 零 output row 和 global 分母未定义

`scale=absmax/7` 在某一 row 全零时会除以零；center 后也可能产生全零 row。另一个分母 `sum(W²)` 可能为零。官方初始化确实把 Q 最后一层 weight 置零，但训练后 checkpoint 是否有非零权重要由 live evidence 决定。[world_model.py](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py) [init.py](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/init.py)

冻结版至少要写死：`absmax==0` 时使用固定 `scale=1`、integer code 全为 0、dequant 仍为 0；global norm 分母 `≤0` 或非 finite 时直接 `inconclusive_no_gauge`。不能让 NaN 进入 `A`，也不能把 zero row 当作一个可改善的量化 row。

### 2. global binding 没有定义五个 member 的聚合

`101×sum(m²)/sum(W²)` 中的 `m,W` 没有 member 轴，也没有明确 `m` 必须从 pristine original `W` 计算。`“至少一行”` 的 row 也未说明是否跨五个 member。若由实现者看到结果后选择某个 member，binding 就不是预注册 gate。

建议冻结为一个确定规则，例如对原始 `W_m` (`m=0..4`) 逐 member 计算 `μ_m=mean(W_m,dim=0)`，令 `r=max_m 101||μ_m||²/||W_m||²`；`r>1e-8` 且 `any(m,row)` 满足 scale 相对变化 `>1e-6` 或 integer code 不完全相等才通过。若采用 pooled Frobenius ratio 也可以，但公式、zero-denominator 行为和跨-member `any` 必须在输出前固定。scale/code 变化应比较每 member 的 original-vs-centered 两套量化，并保存原始整数 grid。

### 3. 四个 arm 的 pristine snapshot 语义不够明确

当前草案写了“恢复 exact”，但没有说恢复发生在每个 arm 的**开始**。正确的事务是：四臂都从同一份 FP snapshot 出发；`FP-original` 直接评估，`FP-centered` 从 snapshot 同时中心化 `W,b`，`W4-RTN-original` 从 snapshot 量化原始 `W`，`W4-RTN-centered` 从 snapshot 先中心化 `W,b` 再按 centered `W` 计算 scale 和 RTN。每臂结束恢复，下一臂不得继承上一臂的 quantized/centered weight。否则最后一臂可能实际测的是 `center(quantize(W))`，不再是协议声明的 comparison。

### 4. `A_original` 的命名可能造成致命分母错误

“相对 FP-original 的误差平方”本身没有问题，但后面使用 `A_original`、`A_centered` 时必须明确它们是两个 **W4 arm** 的误差，而不是 `FP-original` 和 `FP-centered` 两个 FP arm。建议冻结以下定义。对 state `e`、candidate `c`、unordered pair `p=(i,j)`，先按官方 `two_hot_inv` 得到 member scalar `q`，令

`S_arm(e,c,p)=(q_arm[i]+q_arm[j])/2`，

`A_arm(e)=mean_{c=1..64,p=1..10}(S_arm(e,c,p)-S_FP-original(e,c,p))²`。

然后在 effect gate 中只使用 `A_original=A_W4-RTN-original` 和 `A_centered=A_W4-RTN-centered`；FP-centered 只用于 no-op，不能成为 primary 分母。当前文字若被字面实现为 `A_FP-original=0`，8-state gate 将必然退化，因此这是冻结前 blocker。

## 门槛的剩余边界

修正上述四点后，global binding、8-state non-degeneracy 和 effect gate 都是预先可证伪的；不需要增加 full rollout 或额外 baseline。仍应报告每 state 的 raw `A` 和 gain，因为 `A>1e-8` 只是防止零分母，不保证 ratio 对极小误差完全稳定。现有门槛允许最多两个 state 变差，只要另外六个方向、median 和 aggregate mean 通过；这与当前 `scope_limited_preliminary_go` 的窄 claim 一致，但不能扩写成所有 state 都改善。

最终建议：**完成四项冻结修正后保留一次 bounded diagnostic 的 conditional-go；未修正时 implementation/protocol no-go。** 即使通过，也只能支持该 pinned checkpoint 上的 representation-sensitive W4 RTN diagnostic，不支持新 centering 方法、native low-bit deployment、task success 或作者 policy reproduction。
