# Value-head gauge：最小机制screen草案

状态：尚未冻结、未提交；需TD-MPC2官方checkpoint compatibility先解除。这里承认centering本身已有直接先例，只检验distributional value-head中一个不同于rounding-coupling的数值前提。

使用与TDQ相同task/official source，但单独fresh DMControl reset seeds5209..5216、无env step/render。相同checkpoint按预先固定2→3顺序选第一个strict-compatible者；选择只看加载schema，不看Q值或表现。全部source与checkpoint identity在最终冻结版补全。FP imagined cache：seed6301生成8×64×3×1 Uniform[-1,1] actions；各state从FP encode起用FP dynamics运行H3；seed7301+state_index只调用一次FP pi，cache供所有arms共用。

四臂顺序FP-original、FP-centered、W4-RTN-original、W4-RTN-centered。只改五个critics最后一个Linear（101×512）的weight与精确bias centering，前两个Linear、LN、encoder/dynamics/reward/policy/target均FP。每member沿101个output bins取weight均值m，Wc=W−m，bc=b−mean(b)。RTN两臂只量化其当前最后weight，per-output-row absmax/7、signed[-7,7]、torch.round，bias保持FP。没有额外member/forward/部署补偿参数；只是fake quantization，不测native kernel。

Primary对全部10个unordered pairs，先官方two_hot_inv解码每个member，再取pairavg相对FP-original的误差平方，64candidate内平均为每state A。不能用raw-logit MSE替代。Primary只比较RTN-centered与RTN-original；FP-centered仅是数学no-op的数值工程对照。

No-op：全部softmax probabilities满足allclose(atol=1e-7,rtol=1e-6)，decoded member Q满足allclose(atol=1e-5,rtol=1e-6)。失败=implementation_inconclusive，不解释成科学差异。所有weight/bias恢复exact，target与bypass hash不变。

Binding分为global与state两层。global必须是**weight** common mode非零，不能用bias common mode凑数：101×sum(m²)/sum(W²)>1e-8；至少一行scale相对变化>1e-6或整数code改变；同时保存centered与original量化后的quotient weight error（沿output-bin减去误差均值）作为描述。row-scale/code是同一checkpoint的global量，不按8个state重复计数。state层要求8个state全部A_original>1e-8；否则inconclusive_degenerate，不删除state。global无binding则inconclusive_no_gauge，不称科学no-go。

冻结时拟采用的效应门槛：全集8-state median(1−A_centered/A_original)≥25%，至少6/8state严格改善，且全8state mean A_centered<mean A_original，才给scope_limited_preliminary_go；binding足够但未满足则method_no_go。不调centering系数、bits、task、seed或calibration loss挽救结果。该positive仅是checkpoint-specific representation-sensitive PTQ前提，不认证新方法、环境success或原作者policy reproduction。

一次single V100≤10min，CPU准备≤5min、raw复算≤5min；无训练、无完整闭环，不扩大验证。当前草案不允许直接提交；root须先补source/asset身份并发布冻结标记。
