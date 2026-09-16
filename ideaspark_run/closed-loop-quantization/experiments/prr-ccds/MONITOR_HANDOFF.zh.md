# 当前状态：已完成，请勿重新提交

64706 GPU、64707独立CPU核验、64708描述性明细汇总全部完成。当前recipe为mechanism_no_go；工程核验通过。结果已写RESULT.zh.md，jobs.json已更新。暂停prr-v100监控，不自动R2/R3/TEST，不恢复以下历史执行计划。

---
以下为历史执行交接，仅供审计。

# PRR 最低完整实验执行交接

用户已授权直接完成 engineering + R1 四格各3seeds，允许 subagents 和长任务定时监控。无需再问授权。工程64705已通过，35 GPU秒；正式64706于2026-09-12提交并确认RUNNING TC1N03，45min上限。详见 jobs.json、RUN_STATUS.zh.md、ENGINEERING_RESULT.zh.md。任务仍在执行，不能把正式提交写成实验通过。

## 路径与范围

本地根 D:/Downloads/Final Year Project；本实验 ideaspark_run/closed-loop-quantization/experiments/prr-ccds。
远端 /tc1home/UG/yguo017/prr_ccds/artifacts/64706。正式冻结 manifest_r1.json，采集后的 records/manifest.json。bootstrap为旧FRT64694，collector为64688；原始模型与venv在cem_update_ccds。不要覆盖旧FRT结果或重启其旧监控。

CAL72–77、DEV78–83，各6个episode×2windows；历史0–71保留，TEST84–95锁定。四格 q_local/q_recovery/l_local/l_recovery，各seeds1201/1202/1203、1000updates、batch2、H2。任何一格先失败都不能把其余未跑格自动略去而宣称完整no-go。禁止DEV调参、自动R0/R2/R3/TEST。当前总上限6 V100 allocated GPU-hours，所有本轮失败/预检/正式GPU作业均计入。当前已知64705=35sec；64704是CPU10sec，不计GPU。

## 集群硬约束

仅使用 CCDS SLURM；ASPIRE2A不可用，不连接。Head仅轻量连接、提交、状态、小控制文件（每个≤65536B）。禁止在head或本机执行模型/训练/benchmark、数据数组分析、大文件hash/传输/解压/下载/安装。重操作必须真实compute allocation，先allocation_guard检查SLURM_JOB_ID、RUNNING、UserId、实际TC1N hostname和NodeList，GPU再验证分配。不要伪造PBS_JOBID。保留SSH host-key verification；凭据仅controller本地读取credentials.env，禁止打印/上传。禁止同目标多个写入者、盲重启或杀未知任务。

最多1GPU、2个running/submitted jobs。详细独立监控/核验可委派 gpt-5.6-luna xhigh；说明上述全部限制并核实配置。已有 /root/frt_monitor_check 是Luna/xhigh verifier子任务，可用followup_task；不要创建用户新任务。

## 轻量检查

使用 prr_control.py status 64706 先查原job，再读具体小文件；controller read会保存至给定本地路径。命令行为 upload/submit/status/read，全部有scope/size限制。不得修改controller放宽限制去拉raw/checkpoints。

- progress.json：completed列出已完成arm:seed；仅作为进度，不代表独立核验通过。
- raw_final_summary.json：12fit完成后的compactsummary。
- verification.json 或后续CPU核验job指定summary：独立核验结果。
- 日志 artifacts/r1_64706.log 可仅在失败时读取；超过64KiB则在CPUallocation摘要，不能放宽head读限。

若RUNNING/PENDING且无可行动变化，保持安静，不重复提交、不高频poll。若失败，先读原job状态/小日志定位；工程故障为inconclusive，保留计费记录，修复后仅补必要工作，不擅自重调参数。正式若部分fit保存，可在compute中按identity安全恢复，不能直接改旧output导致manifest/source不一致。

## 完成核验与记录

独立verify_prr.py在CPU allocation执行（--artifact-dir正式目录 --manifest正式records/manifest.json --output指定verification.json），不得本地读取NPZ/PT。以其独立重算raw指标及integer/scale证据为准；完整工程条件失败是inconclusive。预算用SLURM实际elapsed×gpu_count汇总，不把requested walltime或训练函数时间当实际总消耗。

每family recovery对local必须平均H2误差改善至少5%，≥2/3seeds通过且各≥4/6episodes方向改善，并clean退化≤10%。q/l任一family通过=conditional_signal，两family都失败且完整=mechanism_no_go；门限模糊/工程失败=inconclusive。四格结果和interaction分别报告，不把两family都通过设为必要条件。

核验后写 RESULT.zh.md、更新RUN_STATUS.zh.md和jobs.json，说明做得好/不好、primary/secondary、每seed/episode一致性、实际GPU时间及假量化范围。不声称native W4加速/显存收益。保存raw远端位置及SHA。用户通知一次最终结论和文件链接，然后暂停本PRR heartbeat；不要archive当前研究任务。

PRR heartbeat 已创建：automation id `prr-v100`，每15分钟检查；完成后暂停此id。CPU核验64707已提交，afterany:64706，确认PENDING Dependency。结果在artifacts/64707/verification.json；先查64706和64707原状态。root完成最终verifier schema/runtime/fingerprint修正，原verifier子任务已interrupt，不需要再等待或让其覆盖文件。

记录格式勘误：正式冻结manifest_r1.json的artifact_contract.raw_shapes.wz遗留了[1,P,D]，实际与同manifest的weighting.shape一致为[P,D]=[196,404]。正式计算未改；不要改已运行manifest或其SHA。verifier按实际二维广播重算。原raw metadata仅有row IDs，initial fingerprints须从同job canonical records按episode_id/record_index/dataset_index/window_start连接，verifier已实现。
