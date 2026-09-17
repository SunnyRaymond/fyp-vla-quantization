# Semantic Input Scales：量化 conditioning 接口而不是换校准目标

2026-09-13，待审查、未执行。基于裁剪后的 ResearchStudio-Idea：已有文献+实际C04/C02 pattern、独立审查、冻结最小实验；不生成Phase4卡片。

## 假设与已有证据

DINO-WM `models/visual_world_model.py` 的concat_dim=1路径把visual、proprio、action embeddings按channel拼接，`predict`再reshape并调用predictor。现有Wall adapter报告384+10+10=404维，实际尺寸必须由checkpoint在compute预检，不能使用source defaults。这里改变 **predictor输入activation的固定scale布局**；不改变weight quantizer、损失、planner、动作或目标。

假设：统一activation scale受较大幅值模态支配，降低低维conditioning的有效分辨率；按真实语义边界分开三套scale，相对同数目但打乱语义的partition，能改善 first-action fidelity proxy。本screen识别固定坐标分组及其range的联合增量，不将其解释为抽象语义的因果效应。

这不是新量化原理。[QuantWM](https://arxiv.org/html/2602.02110v1) 已比较activation粒度，且细粒度并不保证规划改善；[SmoothQuant](https://arxiv.org/abs/2211.10438) 已处理activation outliers；[QuantWAMs](https://arxiv.org/abs/2607.28405) 是多模态calibration强近邻。剩余问题仅是此DINO-WM接口上的语义分组是否携带额外有用信息。无语义增量或仅通用分组收益则不推进。

## 机制与对照

共同W4：24个predictor Linear做signed[-7,7] per-output-channel RTN，encoder及其他weights FP32。共同activation为signed A8[-127,127]，s=CAL absmax/127，零range用s=1，round-to-nearest-even并clip。使用symmetric而非原pre-study建议的affine observer以缩小实现自由度，此选择在任何新数据前冻结；不是QuantWM完整复现。

在predictor forward的输入（positional addition之前，形状[B,P,404]）挂只读输入替换hook，输入每次都按同一固定scales量化；不改输出latent缓存，也不量化每个Linear activation。不改FP图的结构或action replacement。五个arm：

1. W4A32：不量化输入，作为仅weight损失背景；另有原始FP32 reference。
2. FullInput-A8：全部404channels共用一个scale。
3. Semantic-A8：visual384/proprio10/action10各一scale。
4. Permuted-A8：用固定seed1501随机permute404channel index，仍分384/10/10三块，仅更换partition归属，张量坐标不重排。
5. PerChannel-A8：404个scale，作为标准细粒度成本背景对照，不调参。

唯一机制负对照是Permuted-A8，它维持3个scale和分组尺寸，仅删除语义对齐。CAL范围分别按各自group同一raw input统计；不得从DEV重新估计scale。

Observer冻结：输入保持FP32。对每个CAL predictor调用，先沿batch与patch/token轴取absmax，获得404个channel范围；再对所有CAL episodes、candidates和rollout calls取max，得到唯一共享404维统计向量。每组scale为此向量在对应channel集合上的max/127。保存该充分统计量、调用数、分组membership、scale及hash；不需要保存大型input bank。DEV仅记录误差/clip计数，不更新任何range。Permuted只有一个预先固定分组，结论不外推到全部随机分组。

Pattern：C04 `heterogeneous_decomposition`将已知模态坐标分解并区别处理；C02 `controlled_diagnostic_design`控制scale数量与组尺寸，隔离语义对齐。它们作为启发，不是方法新颖性的证据。

## 冻结最小实验

- 预留CAL source indices108–111、DEV112–117；namespace CAL950000、DEV960000，candidate各加10000。此前0–95使用/保留，96–101归Antithetic，102–107归batch-scale研究，均不重用。检查真实底层episode/fingerprint。
- 固定model epoch65/source identity，同一CAL4episodes每组32个H5 normalized standard-normal actions，只收集W4A32输入range，不更新weights；一个共享CAL bank用于所有partition。
- DEV6episodes，每组64个H5、actiondim10 candidates，FP与各arm同pool。先在首个CAL pool执行no-op hook与FP32等价检查：同score shape、torch.equal/array_equal scores、相同stable top6、相同action shape；移除hook后检查无残留。全部forward保持同batch shape，不把batch变化与quantization混淆。
- Primary：按每组原始terminal objective取stable top6（同score按candidate index），比较其mean action sequence的第一model-action向量，与FP top6 mean的MSE。top6是本小pool约10%elite比例的实验规模设置，非原完整CEM5复现。
- Secondary：全H5 elite-mean MSE、one-shot top6 FP-score regret、score NMSE、各模态input quant MSE/clip比例。FP是fidelity reference，不是环境成功oracle。
- 先按episode统计再等权平均，不把candidate数量当独立N。无训练、无参数搜索、无闭环suite。

## 事前停止规则

工程失败/尺寸或identity不符/no-op不通过/未完整6episodes => implementation_failure或inconclusive，不产生科学no-go。

工程通过后，令m[e,a]为每episode的primary MSE，M[a]=sum_e m[e,a]/6。对两个control c分别计算(M[c]-M[Semantic])/M[c]，均须>=0.05；分别须至少4/6个episode满足m[e,Semantic]<m[e,c]-1e-12。若任一control的M[c]<=1e-12，没有足够headroom识别要求的双重增量，记no_binding_locus，不用极小分母放大改善。相对W4A32的primary额外误差亦需据实报告。5%与4/6为自主冻结的经济性筛选门槛，不是统计显著性。此primary不声称sequence-level action choice或terminal-objective改善，因此regret仍如实列为secondary，不事后改gate。

若上述条件未满足，mechanism_no_go；通过仅为preliminary_go。若Semantic与PerChannel无优势，保留“通用fine-grained quantization足以解释”的局限；不声称新算法或真实压缩优势。所有结果到此停止，不切A4、换permutation、增加CAL或调整阈值挽救no-go。代码缺陷才允许定位修复后原样重跑。

## 资源

CCDS单V10032GB，job<=45min，max workload2400s；预期远低于1GPU-hour但不作时间保证。全部数据、hash、模型、hooks、数值计算在真实SLURM compute allocation；head只轻控制，保留host-key，不输出credentials。复用cem_update_ccds环境，无本地推理或新模型下载。
