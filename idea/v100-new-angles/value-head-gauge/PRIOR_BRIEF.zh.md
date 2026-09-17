# Value-head common-mode removal：待独立审查

2026-09-13。仅primary retrieval与符号推导；未读取本候选模型输出，未冻结、未运行。

研究假设：TD-MPC2的101-bin Q head使用softmax再解码为scalar。对每个member的最后Linear，沿output-bin轴去除weight/bias均值，在exact arithmetic中不改变probabilities和decoded Q，但会改变W4 RTN的row absmax与格点。问题是现有checkpoint是否含有足够大的无效common-mode，使这种无额外forward/参数的预处理降低decoded Q误差。这里改变参数代表形式，不是增加Q members、couple rounding draws或为latent施加simplex projection。

已有证据：官方[math.py](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/math.py)的two_hot_inv先softmax、再取bin expectation、最后symexp；[world_model.py](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py)的Q final layer初始化为zero。训练后common-mode是否仍近零未知；softmax梯度和为零本身不能保证Adam逐元素更新后的weight均值不变。

最直接的先例是[TransformerLens unembed centering](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.weight_processing.html)，已公开相同translation-invariant weight transformation；[Output Embedding Centering for Stable LLM Pretraining](https://arxiv.org/abs/2601.02031)也已将output-embedding centering用于training stability。因此不能声称发现softmax gauge或新centering方法。[Softmax Bias Correction](https://arxiv.org/abs/2309.01729)针对softmax activation量化后的bias，不等同于最后Linear的weight-only gauge选择，但仍构成邻近先例。

最大反对：这可能只是把现成unembed centering应用到小Q head；common-mode也可能本来就可忽略。即使weight MSE下降，仍须看decoded value而不是raw-logit MSE；scalar value经过symexp非线性，概率损失与value fidelity不等价。单task reset附近结果不能证明控制收益。泛化novelty目前不成立，是否有足够独立的model-based数值诊断价值需外部审查。

若审查接受，只考虑新的reset seeds5209..5216、固定H3/64actions、四臂FP-original/FP-centered/RTN-original/RTN-centered，仅最后Q Linear weight W4、bias可做精确centering但不量化，其余模块FP。FP-centered对decoded Q必须先通过数值no-op；初始common-mode量、row-scale变化与非零RTN误差为binding。Primary用official random-two-Q avg的10pairs有限期望误差，不能拿raw-logit MSE充数。具体阈值、成本与input协议尚未冻结，不能直接运行。本brief不是新增实验授权要求，root可在独立反证后按用户既有授权作决定。

检索限制：完成softmax/logit/weight centering、gauge、output head+quantization的组合检索；没有发现exact TD-MPC2 weight-centering PTQ结果不等于novelty认证。所有未接受候选仍保留。
