最终状态：账户used99% / remaining1%，已按用户限额停止。14个经验screen全部STOP；最后Reference-branch GPU64840/CPU64841为mechanism_no_go（6/6binding、0/6joint）。无待运行job，不再按下方历史步骤自动续跑。入口READ_RESULTS.zh.md及SCREEN_REGISTRY.json。

# 当前续跑入口

2026-09-13：用户已恢复 GlobalProtect VPN，CCDS SSH/SLURM 已实际恢复。旧续跑步骤2–6均已完成：原64815由CPU64823归档inconclusive_budget；B2 GPU64825/CPU64831为scope_limited_preliminary_go，STOP；Teacher GPU64828/CPU64830为mechanism_no_go，STOP。不得重复提交这些完成的实验。最新待办为policy-prior-support，已冻结PROTOCOL.zh.md，CPU/GPU实现与独立verifier准备中，尚未提交。最新quota remaining18%，达到<=1%停止新工作。详见ACTIVE_STATE与INDEX。

下面仅保留历史网络中断时的计划，不能作为当前待执行清单。

# 网络恢复后的最小续跑顺序

2026-09-13。当前CCDS TCP22连接超时；未确认原因。用户已授权以下最小实验，不需要再次请求实验许可。必须先有真实网络连接；禁止关闭host-key verification、改凭据、伪造allocation或退回本地/login node计算。当前没有提交成功的新CPU/GPU job，64815已确认退出。

1. 读取实时Codex quota；达到remaining<=1%即停止新工作并保存结果。未达到门槛也不得空烧tokens或盲重试连接。先检查本任务已记录jobs的真实状态，避免重复提交。
2. 上传 `rounding-persistence/verify_persistence.py` 和 `verify_persistence_cpu.sh`，等待上传命令明确成功，再提交CPU归档64815。它遇到11/12 partial receipt即输出inconclusive_budget，不加载partial科学数组。读取小型verification.json并补充RESULT，不推断科学正负。
3. 上传B2独立runner、launcher、protocol alias，再提交 `persistence_batch2_gpu.sh`，最多10min allocation/540s。完整重做相同6state×2noise，不能与64815拼接。无论结果如何只做独立CPU replay，然后停止；B2失败不再调batch、样本、容差、solver或walltime。原64815证据保留。
4. B2 CPU verifier为 `verify_persistence_batch2.py/.sh`。launcher故意没有默认GPU job；只有取得本次B2 job ID后，才把该明确路径冻结到launcher默认值并上传，或使用轻量scheduler显式export。不得误指向64815。CPU最多5min，不加载模型。
5. Teacher先上传 `prepare_teacher_bias.py/.sh`、`teacher_bias_protocol.zh.md`，提交5min CPU准备。读取小型manifest/status，检查六条Wall124–129映射与源身份，再用下载的manifest原始bytes记录SHA256到 `teacher_bias_input_freeze.json` 的 `manifest_sha256`。该文件尚不存在，禁止先凭猜测生成或直接提交GPU。
6. Teacher GPU仅在CPU准备成功且输入freeze完成后提交 `teacher_bias_gpu.sh`；single V1005min/内部240s，三个固定arm。完成或失败都由 `verify_teacher_bias.py` 的CPU replay给出对应结论。不得改选样本、补seed或扩大成真实task验证。

所有上传文件每个<=64KiB且为小型控制源码/JSON/Markdown；大数组、checkpoint、视频及其hash校验均留在已获批compute allocation。`ccds_campaign_control.py`只负责小文件与scheduler操作。每次工具返回session_id时必须等待该命令结束，不能把“已发出上传”当成“上传成功”。

本轮所有已完成screen已STOP；失败与inconclusive的区别见INDEX和独立CAMPAIGN_CONCLUSION_AUDIT。没有授权继续Value-head FP64变体、TDQ seed扩展或Flow补全验证。Camera-redundancy identifiability_no_go保留，GPU0。ResearchStudio-Idea使用用户允许的C02/C04裁剪流程，不补做无关Phase4内容。
