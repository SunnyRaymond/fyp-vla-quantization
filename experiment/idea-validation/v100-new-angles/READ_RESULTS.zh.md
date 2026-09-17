# 本轮结果怎么读

最终状态：实时额度used99% / remaining1%，停止新增工作。14个经验screen均STOP，无待运行作业。

本campaign已完成14个新的bounded empirical screens，另有多项在源码、prior或资产gate停止的候选。每个screen都在初步结论处停止；未进行完整task validation。历史RankCal、OTC、CEM-Update、FRT、PRR作为出发点，未计入这14项。

## 当前留下什么

| 结果 | 已有证据 | 研究贡献与实际使用边界 |
|---|---|---|
| [Broadcast activation coupling](broadcast-coupling/RESULT.zh.md) | 相同输入MSE与每patch三draw集合，5/6状态的一步visual MSE改善至少10% | 可保留DINO-WM受控诊断；generic correlated rounding已有prior，application novelty未确证；未证明task或压缩收益 |
| [Temporal rounding persistence](rounding-persistence/RESULT.zh.md) | 相同逐step draw集合，6/6状态通过forcing与endpoint联合gate | 数值signal成立；De-biasing Diffusion已覆盖stochastic weights减少跨步相关性的宽方法主张；未验证snapshot切换成本 |
| [Antithetic rounding pairs](antithetic-rounding/RESULT.zh.md) | 有限recipe中coupling有机制signal | 两套W4相对单W8不具实用优势，practical no-go，停止 |
| [Flow geometry](flow-geometry-drift/RESULT.zh.md) | backbone子结果相关排序有signal，expert没有 | 正式状态inconclusive_provenance；原GPU最终summary丢失，不能把控制流推断当作保存的runtime证据 |
| [Value-head gauge](value-head-gauge/RESULT.zh.md) | W4数值改善看似很大 | FP no-op未通过，implementation-inconclusive；不能跳过gate把它列为方法go |

其余实测结果及全部prior-only候选见[总表](INDEX.zh.md)。No-go是针对冻结recipe与最小假设的结论，不证明整类WM/WAM/VLA quantization方向不可做。

需要按实验查找时，使用[14项实验目录](SCREEN_REGISTRY.json)及[目录审查](../../../idea/v100-new-angles/REGISTRY_AUDIT.zh.md)。每项列出protocol、RESULT、GPU producer、CPU verifier、经验结论和novelty状态；两种状态不能互相替代。

实际V100运行规模见[资源证据](../../../idea/v100-new-angles/RESOURCE_SCOPE.zh.md)，其中单独保留失败重试成本，避免把主要producer时长当作完整campaign成本。

## 不同状态不能混用

`mechanism_no_go / method_no_go`：完整运行、工程与识别gate允许解释，但预设科学门槛没通过。

`inconclusive_binding / statistical_inconclusive`：比较的识别、误差预算或检验灵敏度不足；没有足够证据做科学正负结论。

`implementation_inconclusive / inconclusive_provenance / inconclusive_budget`：实现、当次证据或完整运行不足。失败job、partial outputs与修复过程继续留档；不能默默替换成后来的成功记录。

`prior / structural / resource no-go`：未进行该候选的模型实验，由已有方法、实际结构或资产约束停止；不得混入经验no-go数量。

`scope_limited_preliminary_go`：达到一次小screen的预注册门槛；不等同于统计上充分验证、论文novelty、机器人success或可部署低比特系统。

## 可复查证据

每个候选从自己的PROTOCOL/IDEA、RESULT以及所链接的GPU/CPU receipt开始读取。完整数组与checkpoint留在CCDS compute storage，本地保留小型JSON、日志、源码与明确SHA。当前数值PTQ以FP32 dequantized operators做fake quantization，没有native INT4/A4 latency或memory claim。

ResearchStudio-Idea采用经用户授权裁剪的流程：相关pattern grounding、独立反证、冻结最小实验、真实SLURM、CPU复核、结论与停止。未为了阶段齐全生成论文、render或完整benchmark。结构/novelty审查中被root纠正的建议也保留了说明。

全部已完成screen继续STOP。只有定位到具体实现问题且原切面仍可做时，才允许另立repair gate；不能通过换指标、seed、阈值或增加任务“修好”一个no-go。

最后的[Reference-branch coupling](reference-branch-coupling/RESULT.zh.md)为mechanism no-go：6/6可识别、0/6通过事前10%联合门槛；小幅改善保留，不降低门槛重命名为go。
