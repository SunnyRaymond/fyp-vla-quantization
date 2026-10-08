# 本次研究约束

用户目标：探索 training-free efficient WAM；方法需有推广到至少两种 baseline 的具体路径，并具备真机复现路径。重点对照 FastWAM-Joint、Cosmos3-Edge-Policy-DROID 和 Sparse-WAM。

- 输出多个候选的比较、取舍和最小 falsification；至少一个候选经过 ResearchStudio-Idea 的完整机制与 prior-art 审查。其他候选的审查深度须明确标注。
- Training-free 指加速方法不更新 baseline 权重、不蒸馏、不以额外训练换取推理节省；配对实验共享同一 frozen baseline checkpoint。若新机器人/任务需要 baseline 原本就有的 post-training，应把它作为复现前置条件单列预算和可用性，不混入加速方法收益。离线 profiling / 参数选择的成本另计。
- 两种 baseline 的架构适配、现有 checkpoint / code、相同计算口径必须逐项说明；尚未运行的实验不得写成已证实的泛化。
- 真机复现路径须指定已有硬件平台、数据与代码接口。机器人实际可用性未知；不声称已完成真机复现。
- 先解释机制和可证伪预测，再列扩展验证。按 frozen gate 处理 no-go。
- 不预设用户有 150 GPU-days 或付费 API 预算；本次仅文献、代码可行性检查和小型机制推导，不提交 GPU 实验。
- 不读取或输出 credentials；不使用外部模型 transport；不动用 banked reset。
- ASPIRE2A login node 仅控制。模型/数据加载、推理、benchmark、重 I/O、下载、解压、安装和编译均须进入真实 PBS allocation，并记录 GPU 利用率/显存。
- 主额度监测：2026-10-02 首次 usedPercent=50，切换模型后 usedPercent=51（remaining=49）。备用 gpt-reserve 初始 usedPercent=0 不算突然重置。主额度一旦从已使用状态突然变为 usedPercent=0（remaining=100），立即停止所有研究和子任务，保留断点。
- 最新实际观测：root 的 `get_usage_limits` 返回 `rateLimitsByLimitId.codex.primary.usedPercent=57`、remaining=43、resetsAt=1791047429；没有发生已确认的重置。
- **停止条件是条件句，不是事件记录。** 只有实际工具返回主 codex bucket 的 usedPercent=0 才触发；不可把“若/一旦/突然变为0”的文字当作已观察到0。Phase1 曾因误读条件句报告停止，该代理确认没有调用 quota tool，故该报告不算重置证据；root 实测57后继续。
