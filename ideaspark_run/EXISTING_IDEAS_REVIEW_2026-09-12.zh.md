# 本轮新 idea 的旧证据审阅

核对日期：2026-09-12。只读审阅本地已有成果；没有重新运行实验、连接集群或查询旧作业。旧目录全部保留。

| 已有方向 | 本轮核对的证据 | 对新候选的约束 |
|---|---|---|
| RankCal | `world-model-quantization/experiments/dino-wm-wall/SCREEN_RESULTS.zh.md`：Wall held-out 24 targets，RankCal 19/24，ScoreError 19/24，两个固定 random mappings 20/24 与17/24；全候选排序改善没有同时带来 elite/action fidelity 优势；原配方 no-go | 不再把单点排序敏感度或混合精度重要性打分换名重做。新的中间指标必须有能否传递到实际动作/任务结果的筛选门槛 |
| OTC-PTQ | `../reproduction/otc-ptq-phase1/RESULT.zh.md`：Fast-WAM Optional IDM，BF16 reference；默认分数 observation 项占 CAL 99.769%，与 Local MSE 选出相同 top-2；CHECK 排名不稳定；未跑联合量化完整 suite | 不再微调 observation/action 权重来延续旧配方；单 site sensitivity 不等于联合配置收益 |
| CEM-Update PTQ | `world-model-quantization/CEM_UPDATE_PTQ_HANDOFF.zh.md` 及 `experiments/cem-update-ptq/PHASE_A_RESULT.zh.md`：改为 elite mean/std fidelity，旧数据表明可测；`RUN_STATUS.zh.md` 最后记录新搜索已提交但没有最终结果 | 这是已有、尚未证实的方法，不计为本轮新 idea；本地旧状态不代表今日远端 job 状态 |
| Future-Effect PTQ | 旧交接背景曾提出比较 Fast-WAM `idm` 与 `first_frame` 的 action difference，未选作当前验证 | 如果采用此分支，必须披露已有构想并证明 future influence 本身可靠，不能把两种模式差异直接当成有益因果效应 |

## 已有工程起点的边界

- DINO-WM Wall 的现有数值量化筛选实际累计约 2.657 allocated A100 GPU-hours；这是具体小实验的记录，不能外推为新方法耗时保证。
- `../reproduction/fastwam-smoke/RESULT.md` 记录单 A100-SXM4-40GB、OSMesa + CUDA 的 `first_frame` smoke；OTC 记录则覆盖 `idm` 测量。两种运行模式、reference dtype 和 checkpoint 必须分开锁定。
- 旧 artifacts 的 fake quant / logical bytes 不构成 native low-bit kernel、真实显存或速度收益。

## 本轮采用的 ResearchStudio-Idea 精简范围

使用安装的 `idea-spark` skill（ResearchStudio-Idea 的生成 pipeline）。两条分支分别保留有界结构化检索、关键近邻全文、corpus pattern 支撑的瓶颈与机制推导、可证伪预测、当前碰撞检索、独立审查与中文研究卡。用户已允许省略不必要步骤，因此省去弱相关文献的大规模 tagging、三语重复出卡与 PDF 排版；各分支必须单独记录实际完成、失败和省略步骤，不将精简运行标作 canonical navigator DONE。

硬件约束是最多同时 4×A100，显存型号未指定；方案应以 40GB 为保守 pilot 起点，并单独说明 80GB 可选配置。总 GPU-hours 未由用户指定，文档中的预算只能是建议上限或待 smoke 修正的规划值。

本次只生成与审查方案，不启动训练或评测。未来集群重 I/O、下载、校验、环境准备、模型加载与计算均须在真实 PBS allocation 中执行，并核验 `PBS_JOBID`、实际非 login hostname 与 allocation；login node 仅轻量控制。保留 host-key verification，不记录 credentials，同一目标文件只允许一个写入者。
