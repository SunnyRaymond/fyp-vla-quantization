# Semantic Input Scales：独立审查

**审查结论：`revise`（修订后才可执行）；不是 `novelty_no_go`，但方法级新颖性主张不能成立。**

本审查只覆盖当前 `IDEA_AND_PROTOCOL.zh.md`、本地 DINO-WM source/config 与当前目录快照；未连接集群、未提交作业、未运行计算。当前快照中该目录只有 protocol，未见 semantic-specific implementation，因此 hook、observer、source/fingerprint 与 gate 尚不能验收。

## 1. 机制是否可识别

在以下条件真的由代码冻结时，`Semantic-A8` 对 `Permuted-A8` 是一个可解释的最小对比：W4 权重、A8 位宽、3 个 scale、384/10/10 组大小、CAL bank、DEV candidate pool 与 tensor coordinate 均相同，只有 partition 的坐标归属改变。`FullInput-A8` 可区分“一套 scale”与“三套 scale”的一般收益，`PerChannel-A8` 可标出更细粒度背景。

但当前 estimand 应写成“**固定 DINO-WM concatenated predictor interface 上的 coordinate-aware partition 增量**”，不能写成因果的“语义本身”增量：每个 partition 的 CAL absmax 不同，结果同时反映坐标归属和该组观测范围。单一 permutation seed=1501 只能支持一个固定随机对照，不能代表所有随机分组；这是可记录局限，无需扩跑。

CAL/DEV 必须用真实 episode/fingerprint 做不相交校验，并证明所有 arm 使用同一冻结 CAL bank 规则；否则 DEV 泄漏或重新估计 scale 会破坏识别。

## 2. 与已知先例的边界

- [QuantWAMs (arXiv:2607.28405)](https://arxiv.org/html/2607.28405v1) 已把 PTQ calibration 放在 model structure、rollout distribution 与 task objective 的交叉处，并要求只有坐标可对应时才共享 activation evidence；它还明确把无共同 coordinate system 的 pooling/splitting 视为 structural mismatch。它是强近邻，足以否定“按结构/模态分组 calibration”作为一般方法新颖性。
- [QuantWM (arXiv:2602.02110)](https://arxiv.org/html/2602.02110v1) 直接研究 DINO-WM 的 activation granularity 与 planning，并报告细粒度 activation 收益不稳定；它没有覆盖本协议的 384/10/10 与 matched permutation 对照。因此它压低广泛 novelty，却没有完全否定这个窄接口诊断。
- [SmoothQuant](https://arxiv.org/abs/2211.10438) 已建立 channel-wise activation outlier scaling；因此 PerChannel/多 scale 的一般收益不是新算法证据。

可保留的最窄 claim 是：在一个已知 concat 坐标接口上，matched permutation 是否显示额外的 coordinate-aware fidelity；结果应表述为 preliminary diagnostic，不宣称新量化原理或压缩优势。

## 3. 必须修复的执行阻断

1. **实现缺失/不可验收。** 在执行前必须有代码硬性检查 checkpoint runtime `D=404`、hook 位于 predictor positional addition 前、输入 shape `[B,P,404]`，并在错误 source/shape 时退出。不能用 source default 猜尺寸。
2. **Primary 语义不一致。** 当前 primary 是 score-selected top6 的“第一 model-action mean MSE”，并非 sequence action choice 或 terminal-objective regret。最小修订是把假设和结果名改为 *first-action fidelity proxy*；若继续声称 action choice，则必须把 one-shot FP-score regret 预先设为 non-worsening sanity gate，而非事后 secondary 解释。
3. **5%/4-of-6 gate 不可复核。** “平均 primary”须明确为先逐 episode 算 MSE、再等权平均；定义每个 control 的相对改善公式、严格/非严格比较及数值 tie epsilon，并补充 primary 近零时的不可辨识规则。当前只对 baseline regret 写了 `<=1e-12`，不足以决定 primary gate。
4. **CAL observer 仍有语义歧义。** 固定“W4A32 输入”的 dtype、reduction axes、absmax 统计单位，以及三套 scale 是否分别从同一 CAL bank 的对应 coordinate 计算；将 bank/scales/hash 写入结果并禁止 DEV 重估。`[-127,127]`（不用 `-128`）与 round-to-nearest-even 需由实现直接固定。
5. **no-op gate 要检查可比对象。** 除输出数值等价外，须在同一 candidate pool 上比较 score、top6 indices 与 action shape；hook 必须可恢复，且不改变 batch/candidate 排序。未满足即 `implementation_failure/inconclusive`，不得下科学 no-go。

## 最小验收措辞

仅当上述工程检查通过，且 Semantic 相对 FullInput、Permuted 的 episode-mean *first-action fidelity proxy* 均达到预注册 5% 改善并分别至少 4/6 episode 改善，才记 `preliminary_go`；否则记 `mechanism_no_go`，同时报告 W4A32 与 PerChannel 背景。该门槛是经济性筛选，不是统计显著性；不得据此声称真实环境 success 提升。

