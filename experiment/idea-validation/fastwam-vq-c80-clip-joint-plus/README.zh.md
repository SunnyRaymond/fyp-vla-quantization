# C80 clip-only / joint：LIBERO-Plus闭环

本实验执行两个固定配置，各280个episodes，共560个新episodes。沿用原四suite × 七扰动维度 × 每cell十个variants、原state和配对环境/sampler seeds。按用户后续安排，clip_only准备完成后先提交一个四GPU、24小时、normal路由g3的evaluation-only任务，与joint calibration并行；joint完成80次更新后再提交自己的280-episode评测。

## 两组配置

| 配置 | 共同初始化 | C80额外更新 | 开放参数 |
|---|---|---:|---|
| clip_only | 已完成C8 rho1.5 update8；ratio=1 | 80 | 每个Linear一个clipping ratio；weights冻结 |
| joint | 同一baseline；ratio=1 | 80 | codebooks、log row gains与clipping ratios |

这里的C80是原始LIBERO四suite各20个calibration contexts，case IDs 2000–2079，沿用来源实验的suite交错update order；不是更换为另一个C80 PTQ初始bank。两组都保留同一初始bank、teacher、seeds与loss：first10六个motor坐标MSE + 1.5 × 0.028303017887797623 × future latent NMSE（排除clamped frame0）。Late22、full32和gripper不加入loss，也不据此筛选checkpoint。

joint严格采用原实验的weight+clipping联合优化定义。本轮不引入channel/group scaling。学习率沿用weight 0.0005、ratio 0.01；Adam仅在全局update1后清空一次state；clip gradient norm=1。固定最终update80，不按held-out或闭环结果挑checkpoint。

## 复用与资源

来源对话：`codex://threads/01a147a4-f04e-7243-914a-ed8f7801aad3`（探索VQ中的activation联合优化）。来源C80目录：`fastwam-vq-c80-activation-clipping`。

来源prepare `25853617.pbs101`、clip_only80 `25854128.pbs101`、aggregate `25854912.pbs101`均已实时核对为PBS/artifact exit0、PHASE_COMPLETE且覆盖完整；见`source_evidence.json`。clip_only复用已训练80次的ratios，不重复训练，仅用独立短单卡任务统一导出并验证实际动作重放。已取消的development数据准备分支不作为本次前置任务；不新增development set或teacher。

calibration任务均为单GPU、16CPU、110GB、01:59:00，目标gdev。joint实际在单次任务`25858027.pbs101`完成80次更新、导出和动作重放，PBS/artifact exit0、TRAINING_COMPLETE，allocation总用时01:07:25，无需续跑；80个training steps的中位数33.09秒、均值33.86秒，step计时不含datum/teacher加载和checkpoint写盘。旧实现约279秒/update只作历史背景，不作为配对端到端提速测量。断点逻辑仍保存同一checkpoint、Adam state、RNG与global update order。

正式评测任务为clip_only `25858029.pbs101`和joint `25858606.pbs101`，各四GPU、24小时、280个episodes；实际进度与完成证据见`RUN_STATE.json`及各arm的原始结果。

每个正式评测任务运行同一个arm的四个persistent workers，各70个固定indices（index % 4 == worker_id）。结果写入results/evaluations/{arm}/worker_{worker_id}；两组各280完成后合并560个新episodes。复用六组既有baseline的1680 slots及C8 rho0.1/rho1.5的560 slots，共2240 reference slots，不新增baseline forward。

用户另要求检查DecodedWeight.backward的32行循环开销。共享train_banks.py现改为每chunk最多1,048,576个4D向量、按输入宽度决定行数；保留原来的BF16 stored values、梯度公式、padding与indices语义。这样减少Python循环和index_add_调用，同时限制临时tensor规模。FP32 codebook累加顺序可能变化，不承诺bitwise相同；clip_only导出allocation先运行backward_check.py，检查旧/新实现的梯度误差、代表shape的实际耗时与临时显存，三项通过后才允许joint使用。不得将独立backward提速直接解释为训练端到端提速。

实际GPU检查`25857126.pbs101`全部通过，见`backward_check_result.json`：三个代表shape的独立backward提速7.61–25.98倍，临时tensor额外峰值最高130,142,208 bytes（约124 MiB）；五组case中codebook梯度最大relative L2为1.82e-6，log row-gain梯度与原实现相同。clip_only复用80次更新、joint最终80次更新的实际动作重放max_abs均为0。

所有allocation先验证真实PBS_JOBID、非login hostname、nodefile与运行状态；GPU utilization与memory每15秒写入job log。所有模型、数值检查与重I/O仅在compute allocation执行；login仅做轻量提交、查询和小控制文件操作。沿用batched_h128_v2，正式评测前检查H128前向/逆向/A4 bitwise一致性与每worker614个目标Linear的reference action replay。

监控按用户要求使用gpt-6-luna/xhigh subagent，每20分钟查询一次。普通Q/R无实质变化保持安静；查询失败为unknown；只在完成、真实失败或合法断点需要续跑时通知主任务，不因排队注释重交作业。

## 完成与证据边界

两组calibration必须80 updates、完整C80来源和614模块，deployment/reference/receipt一致且实际动作重放通过。每个arm闭环完成须PBS与artifact exit0、PIPELINE_COMPLETE、四worker各70/rc0、新episodes280及reference2240同时成立；完整目标还要求两arm全部完成并生成560的合并结果。完整episode逐条保存，可在合法断点后复用。

clip_only的GPU任务`25858029.pbs101`实际完成四worker各70且rc0，之后仅汇总失败；原PBS/artifact exit1及日志如实保留。修复单arm normalization检查与C8 reference每worker query状态初始化后，CPU恢复任务`25860507.pbs101`在10秒内PBS/artifact exit0、PIPELINE_COMPLETE，重新全量校验280新episodes与2240 reference slots，recovery_receipt绑定原GPU任务且记录new_model_forwards=0/new_episodes=0。最终clip_only为207/280；见`clip_only_recovered_evidence.json`、`clip_only_summary.json`。恢复完成证据由原四worker完整结果、CPU成功marker与同源receipt共同构成，不将原失败PBS记录改写为成功。joint运行中的未来汇总代码同步修复，原代码另存备份，修复仅涉及汇总。

本轮是decoded BF16 VQ weights + learned clipping / dynamic A4、KV BF16的闭环成功率比较，不是native INT4 kernel或压缩存储等同于GPU显存收益的证据。

## 最终完成

2026-10-18已逐项验收两组各280 episodes与2240既有reference slots，合并560/2240完整，20分钟subagent监控已结束。C80 clip-only为207/280（73.93%），C80 joint为202/280（72.14%）；初始化C8 rho1.5为206/280。joint评测`25858606.pbs101`为PBS/artifact exit0、PIPELINE_COMPLETE，四worker各70，实际walltime04:07:54。完整说明及suite、扰动维度、配对差异见[RESULTS.zh.md](RESULTS.zh.md)，机器统计见[summary.json](summary.json)，当前完成证据见[FINAL_COMPLETION_EVIDENCE.json](FINAL_COMPLETION_EVIDENCE.json)。
