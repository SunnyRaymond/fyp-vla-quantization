# CCDS 正式实验定时交接

**已完成：64670搜索、64676 CPU核验均结束。工程通过、机制门槛失败；最终RESULTS.zh.md已保存。不要重复提交核验或扩展Stage C，暂停heartbeat。**

用户要求正式实验确认正常运行后结束前台工作，定时派 Luna subagent 检查。主 Astra6 不开启 fast；当前 subagent API 没有 fast 参数，不声称已设置。委派 gpt-5.6-luna / xhigh / fork none，并明确携带下列限制。

已创建thread heartbeat `ccds-cem-update`，每15分钟检查。若AUDIT_PENDING.zh.md存在，恢复时也应读取。

## 当前事实

- 正式搜索 job **64670**，TC1N05，1 Tesla V100-PCIE-32GB。已核实 RUNNING，且本地 `artifacts/64670/search_trace_start.jsonl` 包含配置0至7的成功评分，有限数值、每配置8 pools，约14.6秒。不是只提交未启动。
- 预检64668 COMPLETED，294秒、1 GPU；CPU准备64666 COMPLETED，1302秒、无GPU。预检包含新FP32采集及搜索/核验self-tests。CAL/DEV各4 episodes/8 pools，旧目标重叠为零。
- 预算在正式搜索前冻结：BUDGET_FREEZE.json，workflow4500秒、SLURM4590秒，1GPU。所有方法同一起点、相同W8quota、完整两轮，未根据DEV改设计。
- root credentials.env 路径已用于连接代码，所有secret只在本地读取。ASPIRE2A禁止连接，保留cem-update-ptq目录；旧job17078256结果与最终用量未知。

## 约束和控制路径

遵循项目AGENTS和ponytail。禁止login/head节点计算、重I/O、传输、下载、hash、安装、解压或模型加载。所有重工作必须在真实SLURM allocation，检查SLURM_JOB_ID、实际TC1Nxx主机、RUNNING/用户/NodeList；不得伪造PBS环境变量。能在集群完成的计算不放本机。保持SSH host-key verification，不复制或输出credentials。不盲目重启、不取消不明进程；一个文件一个写入者。VPN/连接失败停止重试并让用户处理。

本机控制器：`nscc-access/.venv/Scripts/python.exe nscc-access/ccds_control.py`。只允许 inventory/init/upload/submit/status/read，上传与读取每文件最多64KiB；大型数组留在compute，不绕过限制。

远端 TOP=/tc1home/UG/yguo017/cem_update_ccds；BASE=TOP/run；模型ROOT=TOP/modelroot；搜索目录TOP/artifacts/64670（不是BASE/artifacts）。VENV=TOP/venv。

## 每次检查

1. 首先查看本地本文件、MIGRATION.zh.md、search_job.json及已有verification/RESULTS，避免重复提交。通过控制器 `status 64670` 单次查询。RUNNING/PENDING且无异常则安静结束本次，不长时间轮询。大trace可能超过64KiB，不读完整trace来绕过限制。
2. 完成后取回小文件：`artifacts/64670/summary.json`、`dev_metrics.json`、`selected_maps.json`、`allocation.json`，保存本实验artifacts/64670。若失败、超时或summary不完整，明确工程未完成，不判idea失败，不自动重跑搜索。
3. 若搜索完整成功，先核对是否已存在本地verification_job.json；不存在时用控制器 `submit verify_results.sh` 提交一次CPU核验，立即把返回job ID写入本地verification_job.json及MIGRATION。脚本、search_job.json和verifier已上传到TOP/control，搜索job固定64670。核验为2CPU/8G/10分钟、无GPU，真实SLURM guard后执行。不要在本机运行verify_stage_b.py。
4. 核验完成后取回TOP/artifacts/核验ID/verification.json及allocation.json和status；独立验证原始数组、搜索轨迹、预算、targets、数值和冻结门槛。依据独立核验工程状态及mechanism gates判断，不能只引用搜索summary。
5. 写RESULTS.zh.md，给四方法DEV L_update、L_mu、two-step final mu MSE及门槛失败原因；计算统计与GPU hours如需程序运算应在CPU allocation。准确区分研究假设、已有证据、待定实现。若Stage B no-go则停止当前配方并暂停该定时任务；不是所有CEM-update PTQ不可能。若通过，只报告机制初步支持；闭环Stage C尚未设计/执行，不声称成功率提升，通知用户决定下一步。没有用户要求就不新增大规模实验。

## 冻结机制门槛

完整工程核验通过；CEM-Update CAL有稳定接受改进；与MeanOnly最终map不同。相对shared_start，DEV平均L_update及two-step final_mu MSE均至少降低5%，各至少3/4 episodes改善（delta<-1e-10）。相对MeanOnly/ScoreError/Rank，每项上述均降低至少5%；L_mu(Update)<=1.05*L_mu(MeanOnly)+1e-10。实现以原PROTOCOL和verify_stage_b.py为准；不得修改门槛或使用DEV重新选择map。四个DEV episodes仅快速screen，不是统计显著性结论；two-step replay不是闭环成功率。

## 通知与结束

状态不变或无可操作变化时保持安静；只在完成、失败或需要用户处理时通知。完成结果报告后暂停该heartbeat，避免重复核验/通知。所有ASPIRE2A记录继续保留，待用户明确告知access恢复再检查。
