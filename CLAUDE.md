# Project Instructions

## 严禁动用账户里的banked reset重置卡！

## 语言
- 默认使用中文回复。
- 用户明确指定其他语言时，遵循用户要求。

## 夜间自主执行
- 按用户本地时间（Asia/Singapore），每日 23:00 至次日 09:00，任务运行过程中不要向用户提问；遇到疑问自行判断、采用合理选项并继续处理，不要停下来。

## Subagent 配置
- 监控或周期性任务：使用 haiku 5.5 subagent。
- 对于指令详尽、简单、不需要更高级模型的insight、能够独立拆分执行的工作，可派遣 subagent，使用 haiku 5.5 ：
  - 默认使用 `xhigh`。
  - 需要更深入推理时使用 `max`。
- 委派时明确任务范围、所需上下文、约束和验收标准，以降低 token 成本并保证交付质量。
- 监控 task 定期检查状态；排队或运行状态没有实质变化时保持安静。检测到完成、失败或需要用户处理时，唤醒原任务，由原任务恢复 goal 并继续后续工作；确认不再需要监控后停用定时 task。

## 计算集群操作约束（autodl不用遵循此约束）
- 背景：2026-09-09 ASPIRE2A 管理员因 login node 上 `aria2c` 大文件下载正式警告（CPU 0% 的 heavy I/O 也算违规），重复违规超过 3 次可能封号。tmux/nohup/后台都不是豁免。
- PBS 脚本在重操作前检查 `PBS_JOBID` 非空且 hostname 不是 login node；compute node 无法联网时不要退回 login node 下载。
- 保留 SSH host-key verification；不要把 `credentials.env` 内容输出到日志、代码、文档或 Git。
- Login node 仅用于连接、作业提交、状态查询和轻量控制；禁止在其上执行计算、模型推理、benchmark、重 I/O、下载、传输、解压、编译或依赖安装。
- 能放在集群 CPU/GPU 上运行的计算任务，不要放在本机运行；优先使用合适的 PBS allocation，数据准备和环境安装优先使用 CPU allocation。
- 提交 GPU 实验时，须在作业运行期间定期采集 GPU 利用率和显存占用，并将记录写入该作业的 job log。
