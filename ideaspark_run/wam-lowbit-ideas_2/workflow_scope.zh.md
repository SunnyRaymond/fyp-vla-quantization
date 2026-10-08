# 本轮 ResearchStudio-Idea 范围

用户请求的是几个 WAM W4A4/W4A8 research ideas。采用已有的裁剪偏好：共用真实文献 grounding 与 Phase 1，然后分别生成三个候选，并独立检查程序逻辑、简单对照和 prior collision。保留每个候选及其失败或保留判定。

此目录不是完整 manuscript pipeline 的 DONE 产物；本轮不运行 Phase 4 的论文扩展、双语三卡和 PDF 渲染，不运行模型校准、GPU 实验或闭环 benchmark。最终中文讨论包由审阅后的候选和明确的限制汇总而成。

## 检索来源与限制

Phase 0 使用安装技能自带的 arXiv / OpenAlex connector，记录真实 metadata。Semantic Scholar 遇到 timeout / HTTP 429，OpenReview 本轮未启用；run-local wrapper 显式跳过这两者，未修改已安装 skill 或 credentials。Host 补充的论文标题仍经 connector 的标题匹配核验。全文由 mandatory fulltext 阶段取得；失败抓取会保留 warning。

Phase 3 使用相同的可用 connector，signature 与 alias 两类词分别覆盖近期和多年 prior。有限检索未找到相同机制不构成完整 novelty 证明。

## 证据分层

- 文献报告：论文自身的结果与假设；不是本地复现。
- 数学与程序检查：小型 synthetic 输入只验证规则能否计算，不建立真实 WAM 收益。
- 新候选：尚未验证的研究假设；数值 action fidelity、closed-loop task success、native kernel latency 各自需要实测。
- 历史失败：来自既有项目记录，本轮不重跑；只约束相同 recipe，不扩展成整个 PTQ 方向的否定。

低比特范围必须列出目标 Linear 的 W/A 精度及所有高精度例外、额外算子和持久状态。硬件和总预算尚未由用户指定，不能套用 skill 的 factory compute envelope 当作已获资源。
