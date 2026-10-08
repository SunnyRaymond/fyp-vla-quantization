# Fast-WAM A4 activation 量化机制诊断结果

2026 年 10 月 6 日完成的固定输入诊断表明，当前 dynamic per-row absmax A4 recipe 的主要敏感区域是 video expert，其次是 action expert。原始 LIBERO-Spatial 输入上也出现了很大的 action 偏差；独立 Linear reference 同样复现这一偏差。逐层 native INT4 与 reference 的误差很小，因此现有证据支持优先改进 activation 量化方式。两条完整推理轨迹之间仍有可见差异，需要保留数值敏感性的结论边界。

**实验范围与比较方法。** 使用 released Optional-IDM clean checkpoint、idm mode，video/action 各 10 个 denoising steps。冻结 10 个原始 LIBERO-Spatial tasks 的 state0 输入，以及 LIBERO-Plus Spatial 的 12 个输入：Camera、Light、Background、Layout、Robot、Language 各 2 个 variants；SensorNoise 未纳入。Plus 输入按固定 selection seed 选取，不依据成功率挑选。每个输入在 30 个 no-op settling steps 后冻结，使用 2 个 sampler seeds，共 22 个输入、44 个 contexts。

Phase 1 完成 176 个 queries；Phase 2 完成全部 352 个 factorial cells，其中复用 88 个端点、新增 264 个 mixed queries。实际总数为 440 个 full queries，44 个 all-A4 traces。所有 GPU case 和 CPU 汇总均成功完成。evaluation episodes 为 0，query 后环境步数为 0。

量化 arms 共享同一套 packed W4 G128 weights，只改变 video/action/proprio 三个 scope 的 activation bits；KV 保持 BF16。量化范围为这三个 scope 的 Linear，text encoder 和 VAE 保持原精度。Primary 为前 10 步 × 6 个 motor 坐标的 RMSE，坐标取模型的 normalized action，每个 arm 与同 case、同 sampler seed 的 BF16 比较；gripper 单独统计。这里的 primary RMSE 没有再除以 BF16 action RMS，也不是物理位移误差。

**Video A4 已产生接近全 A4 的 action 偏差。** 下表是 primary RMSE 的 context 均值，Original 为 20 contexts，Plus 为 24 contexts；所有行的 weights 均为 W4，KV 均为 BF16。

| Video A | Action A | Proprio A | Original RMSE | Plus RMSE |
|---:|---:|---:|---:|---:|
| 8 | 8 | 8 | 0.02034 | 0.02852 |
| 4 | 8 | 8 | 0.32548 | 0.31233 |
| 8 | 4 | 8 | 0.11299 | 0.12854 |
| 8 | 8 | 4 | 0.02097 | 0.02837 |
| 4 | 4 | 8 | 0.34847 | 0.34513 |
| 4 | 8 | 4 | 0.32391 | 0.31301 |
| 8 | 4 | 4 | 0.11359 | 0.12958 |
| 4 | 4 | 4 | 0.34717 | 0.34535 |

完整 2³ factorial 在每个 context 内配对，并对另两个 factor 的四种设置取平均。Video 从 A8 改为 A4 的 mean effect 为 +0.25888，在全部 44 contexts 中为正，范围 +0.13126 至 +0.32867。Action 的 mean effect 为 +0.06263，范围 +0.01305 至 +0.10984，同样全部为正。Proprio 的 mean effect 为 +0.0000574，范围 −0.00894 至 +0.00718，在这批输入上没有表现为主要损失来源。

Video × Action 的 mean interaction contrast 为 −0.06871，说明最终 RMSE 的影响不能按两个 scope 简单相加。这些是 22 个输入、每个 2 个 seeds 的描述性结果；44 contexts 不能视为 44 个独立环境样本，也没有进行显著性检验。

原始输入的全 A4 mean RMSE 已达 0.34717，说明当前 recipe 在未扰动输入上也明显改变了动作。Original 与 Plus 的均值接近，但本次并未将每个 Plus variant 与同 task、同 seed、同初态的原始输入逐一匹配，因此不能用两列的均值差估计扰动的因果效应，也不能据此评价某个 Plus dimension 的难度或成功率。

**当前 per-row absmax A4 会把大量 activation 编码为零。** 对一行 activation，令 M = max(|x|)，当前量化使用 scale = M/qmax，qmax 在 A4 为 7、A8 为 127，再做 round-to-nearest。因此同一输入上的 A4 步长是 A8 的 127/7，约 18.14 倍。对于 M > 0，|x| < M/14 的值在 A4 中落入零编码区间，A8 对应区间为 |x| < M/254。较大的行内最大值会扩大其他通道的量化步长。

以下为 native all-A4 轨迹中的同层、同输入诊断；均值按真实 Linear 调用计算，每次调用等权，不能理解为所有 tensor 元素合并后的比例。

| 实际调用区域 | A4 零编码比例 | A8 零编码比例 | A4 输入相对 RMSE | A8 输入相对 RMSE |
|---|---:|---:|---:|---:|
| Video conditioning prefill | 71.52% | 8.52% | 41.54% | 2.97% |
| Video denoising step 0 | 69.37% | 8.45% | 38.69% | 2.88% |
| Video denoising step 9 | 65.39% | 7.97% | 37.39% | 2.79% |
| Action denoising step 0 | 55.98% | 6.07% | 31.49% | 2.03% |
| Action denoising step 9 | 58.77% | 6.32% | 31.49% | 1.97% |
| Proprio conditioning | 13.64% | 0.57% | 6.50% | 0.33% |

零编码中包含输入本来已有的零；A4/A8 对照显示粗量化额外扩大了零编码区域。Video 第 0 步同层 A4/A8 reference output NRMSE 均值为 0.27870，Action 第 0 步为 0.21552。损失从 conditioning 和较早的 denoising steps 就已经可见。本次 video factor 同时覆盖 conditioning prefill 与 video denoising，尚不能将两个子阶段的作用分开。

**独立 reference 同样产生很大的相对 BF16 动作偏差，但完整轨迹并不完全一致。** Reference 独立解包同一套 W4 weights，在 FP32 中计算 Linear，并输出 BF16；不调用 native INT4 GEMM。其 primary RMSE 为 Original 0.34826、Plus 0.34650，分别接近 native all-A4 的 0.34717、0.34535。

在同一层的同一输入上，video/action denoising 各 step 的 native/reference output NRMSE 均值约为 2.5×10⁻⁵ 至 3.0×10⁻⁵，远低于 A4/A8 量化造成的局部差异。这支持当前 per-row A4 recipe 是主要问题来源的判断。

直接比较 native all-A4 与 independent reference 的最终动作，44 contexts 的 first10 motor RMSE 均值为 0.04551，中位数 0.04533，最大值 0.05697。First10 gripper RMSE 均值为 0.12305，中位数 0.03512，最大值 0.75796。这说明完整推理对剩余数值差异仍然敏感；该差值衡量两条完整轨迹的输出差异，不能当作可相加的 kernel 误差贡献。后续多层、多步传播或量化舍入边界可能放大细小差异，本次没有通过干预实验区分这些机制。

**后续实验的重点。** 要修复当前 A4 recipe，应先针对 video activation 的尺度与粗量化问题，在相同固定输入上验证更细的 activation 分组或其他 outlier 处理；action expert 次之。现有结果不支持通过降低 proprio 精度解决主要误差，也不证明该 mixed 配置能带来明显加速。固定输入上的改善仍需要后续闭环验证，才能判断控制成功率。本次阶段 1 和 2 已完成，未启动新闭环评测或训练。

这次的 21 个 remaining shards 保持原样直至完成。下次应把多个输入放在同一个作业中复用模型加载，按 preflight 实测耗时设置 walltime 和余量；由于 W4 转换是一次性的，可先完成批内所有 BF16 queries，再复用同一 packed W4 bank 运行其余 arms。

完整逐维度统计见 [自动汇总报告](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-a4-phases12/artifacts/25701688.pbs101/analysis/analysis.zh.md>)，数值来源见 [summary.json](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-a4-phases12/artifacts/25701688.pbs101/analysis/summary.json>)，完成证据见 [CPU 汇总观察记录](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-a4-phases12/aggregate_watch_observation_b.json>)。
