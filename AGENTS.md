# Agent Instructions

## 语言
- 默认使用中文回复。
- Proper nouns 和 technical terms 默认保留 English，不强行翻译。
- 用户明确指定其他语言时，遵循用户要求。

## 工作风格
- 使用 `ponytail` skill，并遵循其 `SKILL.md`。
- 优先采用简单、完整、易维护的方案，避免不必要的复杂性。

## Subagent 配置
- 监控或周期性任务：使用 `gpt-5.6-luna`，Reasoning effort 设为 `xhigh`。
- 对于指令详尽、能够独立拆分执行的工作，可派遣 `gpt-5.6-luna` subagent：默认 `xhigh`，需要更深入推理时使用 `max`。
- 委派时明确任务范围、上下文、约束和验收标准；核实实际运行型号与 Reasoning effort，避免意外继承主模型。

## ASPIRE2A 管理员警告与长期操作约束

### 已发生的事件
- 2026-09-09，管理员正式警告：在 `asp2a-login-ntu02` 上运行的 `aria2c` 大文件下载违反 login node 的 fair share usage policies。通知列出 PID `1684555` 在 15:00–15:35 的记录，即使 CPU 使用率显示为 `0.0`，heavy I/O 仍属于违规。
- 管理员表示已终止相关 login-node 用户进程，并警告重复违规超过 3 次可能导致账号自动封禁。不得将通知中的多条采样记录自行解释为已累计的违规次数。
- 此规则适用于本项目所有 agent、subagent、定时任务和后续 ASPIRE2A 工作，不限于 Fast-WAM。

### 必须遵守
- Login node 仅用于轻量连接、作业提交、状态查询和小型控制文件操作。
- 严禁在任何 login node 上执行 compute-intensive jobs 或 heavy I/O，包括大文件下载/传输/复制、大文件 hash 校验、批量解压、依赖安装与编译、模型加载、推理和 benchmark。避免递归扫描大型目录。
- `tmux`、`nohup`、后台运行、低 CPU 使用率或不占 GPU 都不是豁免理由。不得自动在 login node 重启被管理员终止的 workload。
- 所有重 I/O 和计算必须放在已获批的 PBS compute-node allocation 内；调试可使用获批的 interactive compute session。若 compute node 无法联网或传输，不得退回 login node 执行，应暂停相关步骤并查明管理员允许的路径。
- 下载、校验和环境准备优先使用 CPU-only allocation；只有需要 GPU 的工作才申请 GPU。申请合理 walltime，任务结束或失败后及时退出，避免空占资源。
- 工作脚本必须在重操作前检查 `PBS_JOBID` 非空、实际 hostname 不是 login node，并确认处于获批 allocation；不满足即退出。不得手动伪造环境变量绕过检查。
- 清理遗留进程时只做轻量检查，并只终止确认属于本任务的进程。Login 节点可能负载均衡，检查一个节点不能代表其他节点；无法核查时明确记录未验证。
- 同一目标文件只能有一个下载写入者。曾被并发写入的文件需隔离，不能只凭 apparent size 判断下载完成；完整性校验也必须在 compute allocation 内执行。
- 保留 SSH host-key verification，不输出或复制 credentials 到日志、代码、交接文档或 Git。
- 委派和定时任务必须显式携带上述限制；恢复旧任务时先审查旧脚本，禁用曾在 login node 执行重操作的入口。
