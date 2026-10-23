# C8 bilevel rho pilot

本任务按用户已确认方案执行。先准备独立 demonstration development，再验证有限差分信号和虚拟更新的状态恢复；只有 preflight gate 通过后才提交完整 pilot。

- Inner：原 C8 8 contexts、seed0；16 committed updates。
- Outer：新增8 contexts、每suite2个；confirmation另8 contexts、每case2 seeds，全部 demonstrations 与原C80及原40 heldout隔离。
- Outer：8 rounds，两个相同起点的单步虚拟Adam分支，rho乘/除1.5，按四suite等权development first10 motor RMSE的有限差分方向更新；相对差异≤0.1%则不更新。
- rho范围[0.03,3]，初值1.5；c固定为0.028303017887797623。
- 最后固定round8 rho，从共同未训练C8 banks重训16 updates，比较固定rho0.1和rho1.5的16-update模型。
- 优先复用原始banks和teacher缓存；原rho0.1/update16、rho1.5/update8只读用于预检诊断。若确定性修复通过，最终三个arm均从共同初始状态训练16步，理由见[数值修复记录](NUMERICS_REPAIR.zh.md)。
- 最终先冻结D_rho上的固定对照，再评估confirmation 8×2与完整历史40×2。
- 全部PBS walltime 01:59:00，GPU每15秒记录利用率及显存；仅compute allocation允许数据准备、checkpoint加载和模型工作。
- gpt-6-luna/xhigh subagent每600秒只读监控；Q/R无实质变化保持安静，不因查询失败或队列comment重提。

证据边界：单一C8训练轨迹，paired inference seeds；fixed-input decoded-BF16/fake-A4。历史test已被观察；不作closed-loop、native kernel或统计显著性声明。

## 控制入口

本地Python：`nscc-access/.venv/Scripts/python.exe`。`control.py upload`仅上传小型代码/控制文件；`submit --unit data_prepare`、`submit --unit preflight`先后提交。`status`只查询PBS和小JSON；`pull`保存这些结果；`resume --unit ...`仅接受确认terminal exit75/checkpointed的同一单元。

所有完成判据都要求PBS terminal/exit0、artifact exit0、PHASE_COMPLETE及匹配的完整summary。preflight门槛另外要求state restore、输出parity、非空Adam重复分支和有限差分信号通过。

## 当前执行与后续顺序

2026-10-19 CPU data_prepare 25879274.pbs101已通过。原preflight 25879403.pbs101恢复检查通过、分支重放失败；确定性修复预检25879908.pbs101已通过全部terminal、恢复、重放和有限差分gate。已提交search 25880277.pbs101、train_low 25880278.pbs101、train_current 25880279.pbs101，全部walltime01:59:00。后续必须依次推进：

1. data_prepare verified complete后提交preflight；preflight失败不得提交完整pilot。
2. preflight gate与terminal gates均通过后提交search、train_low、train_current。search每段不足2小时，exit75/checkpointed后才可resume；累计完成8rounds/16updates，不能缩减。两个固定对照也从共同初始状态各训练16步，统一数值设置。
3. search与两个固定对照均完成后提交train_selected，从initial_state.pt重训；若rho与固定arm相同则复用相同endpoint。
4. 两个训练单元完成后提交outer_evaluation，完整24queries并冻结固定对照；之后evaluate_low生成/复用confirmation teacher并完成96queries。
5. evaluate_current与evaluate_selected各完成完整96queries，再用CPU aggregate汇总312endpoint records并计算预设gate。

所有来源checkpoint及teacher只读；本实验的continuation、选权重与evaluation结果写本目录的新remote_root。controller拒绝未知提交intent盲目重试，Q/R或查询失败都不能触发重提。

## 最终结果（2026-10-20）

完整pilot已完成，312个endpoint records；rho_hat≈2、lambda_hat≈0.05660604。相对development冻结的rho=1.5，confirmation first10 motor RMSE +27.44%，历史held-out -2.77%，未通过预设gate，不推进C80或activation clip。全部新增GPU成本6.086944 GPU·h，包含首次失败预检。监控已停止。详见[完整结果](results/RESULTS.zh.md)。

## 阶段1 Adaptive 终点诊断（2026-10-20）

用户仅授权阶段1：读取既有search的8轮16步adaptive checkpoint，用search作业冻结的Engine和runtime评估outer8×seed0、旧confirmation8×seed0/1，共24个queries。24份teacher与固定rho1.5/约2的48个endpoint结果均只读复用，不训练、不更新rho、不生成teacher。输出写入新的results/stage1_adaptive；原pilot结果不覆盖。

入口为`control.py submit --unit stage1_adaptive`。walltime01:59:00，GPU利用率/显存每15秒写入job.log；gpt-6-luna/xhigh subagent每600秒只读监控。完整完成需PBS terminal/Exit0、artifact Exit0、PHASE_COMPLETE、匹配summary、adaptive24queries与72条对比记录，以及master参数未改变。阶段2与3没有授权，不提交。

阶段1作业25883753.pbs101已完成上述全部检查：新增24F、0G、0teacher，实际GPU运行00:17:09（0.285833 GPU·h）。Adaptive相对固定rho约2的outer/旧confirmation motor RMSE改善7.63%/17.27%，但相对固定rho1.5仍退化2.09%/5.42%；旧confirmation future NMSE相对rho1.5退化40.13%。监控已停止，阶段2/3未提交，等待用户决定。详见[阶段1完整结果与逐case对比](results/stage1_adaptive/RESULTS.zh.md)。

## 阶段2与条件性阶段3授权（2026-10-20）

用户现已授权阶段2，以及在出现合格正向信号后由agent判断进入阶段3。冻结的stage2_protocol.json规定：rho1.0与2.25从同一initial_state.pt各训练16步，共32G；每支在update1/8/16评估完整outer8×seed0，共48F。沿用search作业冻结的Engine/runtime，c、C8、Adam、训练顺序和数值设置不变。每次评测后完全恢复参数、Adam与RNG，确认恢复精确后才继续训练；保留三个milestone checkpoints，支持exit75后恢复。

两个GPU入口为stage2_rho1、stage2_rho225，完成后CPU stage2_compare检查三个horizon的候选排序与log-rho割线方向。固定rho1.5只复用已有final16对照，不虚构1/8步对照值。所有作业walltime01:59:00；GPU记录每15秒采样，subagent每600秒只读监控。

阶段3仅在某候选final16完整outer first10 motor至少改善1%、其他四指标退化均不超过5%时启动；保留incumbent rho1.5。合格候选之间按final16主指标选优；若没有合格候选，本轮停止，不生成新confirmation或提交阶段3。旧confirmation和历史heldout不参与阶段2选择。阶段3若启动，须先冻结新独立demonstration confirmation，再使用配对teacher、输入和seeds评测，保持原历史40×2完整。无C80或activation clip扩展授权。
