# FRT A/B：运行状态

2026-09-12。范围：CCDS V100 的 A 工程验证与 B 最小机制筛选。用户已授权 A 通过后执行 B；不执行 C 或 TEST。

**当前最终状态：A 工程验证通过；B 工程核验通过，但机制门未全部通过（mechanism_no_go）。A/B 已完成，无待运行作业，监控已暂停。** 详见 STAGE_A_RESULT.zh.md 与 STAGE_B_RESULT.zh.md。

## 已完成

- 阅读 CCDS User Guide，实时查询账号 QoS：1 GPU、20 CPUs、64G、MaxJobsPU=2、MaxSubmitPU=2、MaxWall=6h。
- CPU 环境检查作业 64685 COMPLETED 0:0，实际分配 20 秒，0 GPU。旧 venv、checkpoint、数据与缓存可复用；见 RESOURCE_REUSE.zh.md。
- 新远端目录为 `/tc1home/UG/yguo017/frt_ccds`；旧 `/tc1home/UG/yguo017/cem_update_ccds` 保留。

## A、B 已完成

- A：完整 GPU 作业 64688（41 秒）全部工程检查通过；CPU 独立核验 64689（1 秒）通过。初步 probe 64687 另用 23 秒。结果见 STAGE_A_RESULT.zh.md。
- B：Clean-only / RandomSameNorm / FRT，6 CAL + 6 DEV，3 fit seeds，每组 1000 updates、batch 2、Adam lr 0.01、λ=4；配置冻结于 manifest_b.json。固定共同验证残差集，不读取 TEST、不报告闭环成功收益。
- 工程调试历史：64690 启动 identity 字段错误，12 秒后停止；64691 在 preflight 阶段检测到 batch-1 FP target 与 batch-12 评测的浮点差异后停止。CPU 64693 完成诊断。统一同 batch FP reference 后完整 preflight 通过，没有清零 error 或放宽检查。
- 64694 正式 1000-update × 9 fits 与完整 common-bank 评测完成，19分34秒。CPU 独立核验 64696 完成，14秒；原始数组/episode gates/checkpoint hashes/hard W4 grid 的工程检查通过。
- FRT 相对 Random 的 fresh_union transport error 仅改善约0.30%，未达到预设5%，只有1/3seeds达标；其余4个机制门通过。本轮不自动推进 C。
- 累计GPU allocation含失败与预检为22分59秒（约0.383 GPU-hours）。heartbeat `frt-b-v100` 已暂停。
- 两名 gpt-5.6-luna/xhigh subagents 分别实现 runner 和协议/独立验证器，root 审查并负责所有集群提交。

## 执行限制

A 已独立通过并记录。B 每次作业最多 1 小时，A+B 总 GPU 分配上限 6 V100-hours；失败、预检、启动成本也计入，不因未完成而判机制 no-go。作业与失败原因记录于 jobs.json。

仅在真实 SLURM allocation 内做模型、计算、hash、数据缓存和数组验证；先验证 job、TC1N hostname、RUNNING/UserId/NodeList，GPU 工作再验证分配 GPU。Login 只提交、查询、读取不超过 64 KiB 小结果及上传小控制文件。保留 host key verification。无 ASPIRE2A 连接、无凭据上传或日志、无重复作业。
