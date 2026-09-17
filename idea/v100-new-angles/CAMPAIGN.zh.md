# V100 新切面最小验证 campaign

开始日期：2026-09-12。用户要求使用 ResearchStudio-Idea 工作流生成多个 WM / WAM / VLA + numerical quantization ideas，并在 CCDS V100 执行最小可行性验证。全部保留，无论 go 或 no-go 均不扩展成完整验证。

## 工作流裁剪

根据用户本轮明确授权，保留现有证据审查、真实文献 grounding、corpus pattern 启发、候选机制与 naive baseline、独立 coherence / collision 审查、预注册最小实验和结果结论。跳过与本次筛选无关的 Phase 4 三语卡、论文式扩写、PDF/LaTeX 排版和渲染。此为 adapted pipeline，不声称 canonical navigator DONE。

优先复用既有文献全文和实验资产。新检索若 connector 降级则保留警告，用原始来源补核验，不能声称检索完整或 novelty 已认证。

## 预算与停机

- 启动时 Codex 主额度 usedPercent=26，remaining=74%；按主额度及任何适用主模型窗口的最低 remaining 检查。工具只给整数百分比，无法保证恰好停在 1.000%。不兑换 reset credit。
- 到 remaining <=1% 时停止新增模型任务/实验，记录已提交作业和证据状态并安全收束；不能通过空跑消耗额度。
- 每候选先做最小机制 screen；默认单 V100、最多 1 GPU-hour 的第一道 gate。实际 job walltime 依预检调整并明确记录。
- 不因初步 go 自动增加 seeds、任务集、闭环规模或进行完整验证。
- no-go 仅在有具体实现缺陷证据且切面仍有独立支持时允许一次有针对性的修订；重新冻结假设、对照、数据和预算。不能在 DEV 上不断调参直到成功。

上面的修订边界针对已经产生科学no-go的recipe。环境未能准备、加载或写出结果的工程失败不属于经验no-go；可在审查原job后修正具体缺陷，始终保留原失败证据与科学协议，不能以修环境为由修改已看过的科学阈值。

## 实验与证据边界

- CCDS 使用 SLURM compute allocation；所有计算、模型加载、下载、依赖安装、hash 和重 I/O 在实际 compute 节点执行。作业必须检查 SLURM_JOB_ID、实际 hostname、scontrol allocation 与 GPU 状态，禁止伪造环境变量。
- ASPIRE2A 如确需使用，login 仅用于轻控制，所有重操作必须 PBS allocation，检查 PBS_JOBID、实际 hostname 和获批 allocation。当前使用 CCDS，不另行连接 ASPIRE2A。
- 保留 SSH host-key verification；不输出或上传凭证。大文件单写入者，禁止仅凭 apparent size 判断完整。
- 新鲜 CAL / DEV episode 或 action-pool identity 与旧数据区分；旧数据可用于诊断，不能重标独立确认。
- 配对 episode 是比较单位；同一 episode 内 action candidates 和 rollout steps 不当成独立样本。
- 明确区分 `preliminary_go`、`mechanism_no_go`、`implementation_failure`、`inconclusive`、`novelty_no_go`、`resource_blocked`。未执行不能标经验 no-go。
- fake quantization 不代表 native W4 kernels、真实加速或显存节省。FP reference 保真不等同环境成功。

## 排除约束（以最新本地审查为准）

现有 RankCal、OTC-PTQ、CEM-Update、FRT、Paired-Rollout Recovery 和 TR-PVQ 必须先审查。禁止仅换 mixed-bit scoring loss 或重命名 temporal residual / recovery。每个新 idea 应写清与历史机制的实质差异。

## 保留结构

每候选独立目录保存 IDEA、PROTOCOL、source/pattern provenance、脚本、冻结 manifest、job IDs、原始小结果、RESULT；完整数组保留在 CCDS compute storage。总表在结果产生时更新。原有实验与用户未提交更改保持完整。
