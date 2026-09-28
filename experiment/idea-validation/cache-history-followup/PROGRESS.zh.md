# FastLeWM cache / LpWM real-history 后续实验

用户授权：进行推荐 1 和 2，作为两个独立任务，遵循实验规则。原先取消的 LpWM dense 和 PLDM formal 不恢复。新 goal 不改变旧 goal 的结束记录。

两项实验均已按冻结停止条件结束，最终结论与insights见 [结束记录](RESULT.zh.md)。下列内容保留阶段进展，不应将较早的“尚未完成”视作当前状态。

## 任务 1：FastLeWM cache 在实际负载与闭环中的验证

- Owner：`next_direction_review`，已配置 `gpt-6-luna / xhigh`。
- Local：`../fastlewm-cache-b50/`；remote：`/scratch/users/ntu/yguo017/fastlewm-cache-b50`。
- 固定来源：沿用完成的 FastLeWM paired 实验的 source、checkpoint、backend、50-task manifest 和预处理。
- Fixed-observation：实际 active=50，native batch_size=1，300/top30/30、H1/block25/history1/beta0；配对 native/cache 的完整 solve。首次 cache 构建计时，trace 独立于计时，保留 native cost logging，平衡运行顺序。
- Closed-loop：同 50 tasks，solver seeds 42/43/44；native/cache 成对，报告逐 solve 的实际 active 数、actions、costs、outcomes 和完整评估耗时。三个 seeds 是同任务重复，不是 150 IID tasks。
- 正确性：各 environments 与每次 solve 的 cache 必须隔离；逐轮 candidates/costs、returned actions/costs byte-exact，finite。不得以 B1 两槽实现、调用奇偶、shape 或不稳定 pointer 代替未验证的环境身份。
- 性能门：matched median solve 耗时下降至少 10%，peak allocated 比例不超过 1.10。性能门失败仍完成可有效执行的 closed-loop；fidelity 失败停止，不放宽精度。实际 OOM/资源不足保留 incomplete。
- 当前阶段：FREEZE/PROTOCOL/cache primitive/CPU preflight 已落盘；CPU-only `25577045.pbs101` 在 x1001c6s0b1n0 实际运行后终态 F / Exit_status=3 / Stageout_status=1 / walltime=00:01:33。root 已取回并读到实际 preflight PASS 报告及4份 native source：batch_size=1 时 CEM 是 env-major，每env连续30轮；goal/current实际encode批形为B1，native选择使用torch.topk。旧wrapper错误检查了 OUT/source_reference/preflight.json，实际报告在 OUT/preflight.json，已修正且保留首attempt，不把该PBS job改写为exit0。
- 缓存实现已加入 shape/dtype/device 与 uint8 byte 比较、signed-zero 检查、env-switch eviction；最大驻留2项。更新后的dummy将与 GPU entry selfcheck 合并，不为旧路径错误单独重跑整份CPU预检。
- Root 已审阅实际 GPU runner/PBS，并修正入口 repetition/backend 字段与 FREEZE 不匹配的问题，以及把 deepcopy warmup/benchmark solver 误记为闭环 replan 的问题。真实 solver 现按 pinned wrapper 闭包捕获的对象身份区分，entry dummy 验证原实例与复制实例。技术错误记 FAILED，实际科学门失败才记 NO_GO。
- 已由 owner 唯一提交 GPU `25577481.pbs101`：1GPU/16CPU/110GB/2h，内部6300s；[SUBMISSION](../fastlewm-cache-b50/SUBMISSION.txt) 记录提交。先 Q / gdev（queue overall ngpus limit），后 root 与 owner 均实时确认 R，exec_host=x1000c1s5b0n0/2*16。allocation 内5秒 GPU telemetry已运行；root 已直接读取实际 entry_self_check.json 为 PASS，包含当前 cache 身份/eviction/signed-zero/device 与真实/复制 solver 身份检查，且发生在 model/checkpoint/dataset loading 前。不重复提交；入口自检只证明它覆盖的接口与缓存检查，科学阶段结果另报。
- 三组 fixed seeds 42/43/44 已完成，并由 root 取回[seed42](CACHE_FIXED-seed42.json)、[seed43](CACHE_FIXED-seed43.json)、[seed44](CACHE_FIXED-seed44.json)实际报告核对：每seed active50、4 warmup measurements、12 timed measurements、1500组 candidate-action/cost trace byte-exact、每arm450000 scores、最终actions/costs exact且finite。Native/cache median分别为24.0428/10.4745s、24.2628/10.5888s、25.0270/10.8713s；matched reduction median分别为56.4195%、56.3103%、56.5793%，三seed四门全部PASS。Peak allocated ratio均1.01193。每solve 3000次encode请求变为100次实际计算+2900 hits，最多2项resident，payload1205760B。Owner已确认进入closed-loop；闭环结果与完整评估耗时尚未完成，不能据fixed timing宣称闭环加速。

- 闭环seed42两臂raw report已取回：两臂49/50且逐task outcome相同；第一次真实solve actions/costs相同，但第二次真实solve的11个tasks出现差异，550/2500个action值不同，最大绝对差1.482602。这不是闭环fidelity PASS；现有记录未保存每次solve的input pixels与RNG state，因此不能归因于cache或环境噪声。按照冻结stage rule，fixed fidelity通过后收集全部3对，闭环科学门失败不提前终止、不放宽门。保留[native报告](CACHE_CLOSED-seed42-native.json)和[cache报告](CACHE_CLOSED-seed42-cache.json)。
- 任务1已终态完成，root独立qstat与实际run_summary/job_status/job.log核对：`25577481.pbs101` F / Exit_status=4 / wrapper EXIT_STATUS=4 / Stageout_status=1 / walltime00:33:37，host x1000c1s5b0n0。Exit4是全部冻结配对完成后的科学NO_GO，非技术中断。三seed逐task outcomes分别49/50、49/50、48/50且各配对相同；各seed solve0 exact，solve1分别550、1000、500/2500动作值不同以及11、20、10/50 costs不同。Overall/closed_loop均NO_GO，fixed三组门独立PASS。完整原始报告与telemetry已取回，见[最终结果](../fastlewm-cache-b50/RESULT.zh.md)和[精简汇总](../fastlewm-cache-b50/artifacts/25577481.pbs101/compact_summary.json)。不自动重跑或改门；不能把包含benchmark overhead的raw eval wall称为部署加速。

## 任务 2：LpWM 真实 history / action response

- Owner：`lpwm_result_context`，已配置 `gpt-6-luna / xhigh`，已接收执行任务。
- Local：`../lpwm-real-history/`；remote：`/scratch/users/ntu/yguo017/lpwm-real-history`。
- 固定 checkpoint：正式 sparse `25571461.pbs101` 的 latest；不重新训练 encoder/predictor。
- 比较 official cold-start 与真实近期 3 帧 history。相同 current state/goal/future H5/candidates；过去 observations/action blocks 对齐，输出和执行仍只有 5 future blocks（25 primitive steps）。初始无历史的状态保持共同 cold-start。
- CPU preflight：窄目录确认 native 输出是否另存 actions；验证 target pickle、时间索引与 state sidecars；deterministic dummy alignment check。数据/physics/replay 均在 allocation。
- 已核对的来源限制：native MPC 的 planned_actions 留在内存，plan/evaluator 没有写出它；plan_targets.pkl 的 gt_actions 是 expert actions，logs 仅 scalar，均不能当 native trajectory/proposal bank。若实际目录也没有独立 native actions，则在 GPU allocation 用同一旧 task states/goals/checkpoint 新采集诊断轨迹与 proposals。新采集不冒称旧 29 条失败轨迹的重放；子集改变 RNG 消费后也不要求复制旧 outcomes。
- 物理 reference 要保留完整执行上下文：7D state 不含 block linear/angular velocity；不能仅 reset 到 anchor 的7D state 就当作相同动态状态。候选真值应从原 seed/init_state 重放实际 prefix 后接 future，或使用已验证的完整 simulator state clone。先核对 prefix replay 与捕获 anchor 一致，成本包含 prefix 重放。原 target pickle 还未保存 env_info.shape，fresh file-target 路径需核对 initial render / expert replay goal identity；expert actions 在这里只作环境身份检查。
- Mechanism：冻结候选来源、contexts、任务分层、指标和成本上限。覆盖 native CEM proposal 分布；参考真实终点任务相关位置/角度，不把 cold-start 排序或混合 7D norm 当作 truth。报告 fixed-horizon latent error、candidate ranking/elite regret 与 action response，按 task/context 汇总。
- Root已在新机制数据产生前审定：primary score使用官方前4位置坐标norm/20与wrapped angle/(pi/9)的平方和，object-only距离另报；task先平均paired anchors的regret，再计算相对差，cold regret<0.01单列at-floor。支持门为非at-floor task median reduction>=10%且ceil(5/8*n)改善；至少12contexts/6tasks/6nonfloor，否则incomplete。Primary cohort改为旧formal29失败中按ID升序取前8，避免旧成功组过早终止造成可预见缺样本；不根据新机制结果补task。
- 条件升级：只有预声明机制门支持才运行配对 closed-loop；否则有效 NO_GO，不能以延长训练、改结构或换测试集追门。机制通过不等于 closed-loop 或加速成立。
- CPU preparation 首 attempt `25577337.pbs101`：F / Exit_status=1 / Stageout_status=1 / walltime=00:02:04，host x1001c2s6b0n1。失败发生在调用 compute PATH 中不存在的 git CLI，尚未进入 physics/data replay。修复读取 checkout 既有 SOURCE_COMMIT marker、identity/cohort 门失败时非零退出，以及 PBS 从 snapshot 执行脚本；另在重交前修正 dummy 把单帧 initial history 与 predictor max-history=1 混淆的问题，cold 两入口均保留 num_hist=3，真实模型 parity 仍须在 GPU allocation 验证。
- 相同范围重试 `25577523.pbs101` 已由 root 独立确认 F / Exit_status=0 / Stageout_status=1 / walltime=00:02:26，host x1001c1s5b1n0/5*4（4CPU/32GB/25min，无GPU），wrapper EXIT_STATUS=0。[实际 CPU report](CPU_PREFLIGHT-25577523.json) 的四门全 true；50 tasks 的 obs0/obsg/state_g expert replay exact fraction全部1、最大差0。cohort已确定为旧29失败的前8 IDs [0,2,3,4,5,8,9,12]，env_identity.pkl为339B。CPU没有模型加载或新机制指标；此 PASS 只建立任务/环境/索引准备，不证明history收益。GPU runner正在编写，尚未授权提交。
- GPU runner/PBS已由root最终审阅并授权唯一提交：1GPU/8CPU/64GB/90min、内部80min。修复输出目录/summary契约、checkpoint与实际config字段、scalar goal_H、共享source cwd写入、None log path、重复truth replay和提前quality计算等问题。校准纳入实耗capture、完整physics、两臂forward与truthencoder，并按anchor2更长prefix保守外推；有效shape/finite和单帧native parity为接口门。保存prefix/past actions/history indices/seed/current state/history obs以支持重建。提交记录与实际执行结果仍待owner回报；此授权不涵盖full50 closed-loop。
- Owner已唯一提交 `25577811.pbs101`（2026-09-27 07:09UTC）；root独立qstat核对Job_Name、Submit_arguments与请求来源正确，状态Q/gdev，原因是queue overall ngpus limit。Scheduler接收后实际select为1GPU/16CPU/110GB、walltime90min，而源码为1GPU/8CPU/64GB；变更原因未独立核实，root批准维持同一job的实际资源，不取消/修改/重提，80min内部cap与冻结实验设计不变。[提交记录](../lpwm-real-history/SUBMISSION/attempts.md)。
- Job已从Q转R，root独立qstat与小型job.log核对：exec_host=x1000c0s0b0n0/0*16；allocation入口已输出GPU identity，GPU UUID为GPU-3a19ca4c-cee2-9ec9-2c49-db98b388a9ca，A100-SXM4-40GB，5秒telemetry连续落盘。初期setup utilisation为0，暂没有native capture/parity或机制结果，不把入口/运行状态当作科学PASS。
- GPU初attempt `25577811.pbs101` 已结束，root独立确认 F / Exit_status=1 / Stageout_status=1 / walltime00:08:54，runner/wrapper均1且summary存在。Native capture完成、日志报告16 contexts的history1 parity通过；随后bank构造因脚本假定packed action dim15而实际为10失败，尚未进入cost gate/质量计算。保留为技术ERROR，不是研究NO_GO。Root授权仅修native PushT动作维度（2×frameskip5=10）并在allocation入口检查，同一冻结科学设计修复后重试须先审阅；不改候选数、指标、任务或训练。
- 修复经root审阅后已唯一重试 `25578046.pbs101`：Q/gdev，因queue overall ngpus limit等待；仍1GPU/90min、内部80min，scheduler实际16CPU/110GB。新增literal (300,5,10) synthetic bank检查与dset/workspace/subplanner实际动作维度检查；secondary predicted-zero的ratio按已有定义置0。此前未产生质量指标，科学候选/任务/primary门保持冻结。初attempt保留，不重新训练。
- 重试已R，owner报告host x1000c0s0b0n0/3*16、GPU UUID GPU-625e6467-111b-2889-184d-b934010b17d8，5秒telemetry持续，已进入native capture。Root取回并实际核对[runner_preflight](HISTORY_RUNNER_PREFLIGHT-25578046.json)：literal synthetic (300,5,10) bank构造通过，dataset action dim2、frameskip5、workspace/CEM packed dim10、source marker匹配。仅证明入口/维度修复，尚不是mechanism或资源门PASS。
- 重试 `25578046.pbs101` 已终态，root独立确认 F / Exit_status=1 / Stageout_status=1 / walltime00:06:53，wrapper/runner1、summary存在。16 contexts history1 parity再通过；进入task0/anchor1/300候选的cost calibration后，真实future编码调用transform_obs时缺proprio字段而KeyError；cost_calibration.json未生成，primary/secondary质量未计算。仍是技术ERROR，不是研究NO_GO。Root已读取实际pinned preprocessor与VWM接口：visual-only AdaLN编码不读取proprio，而现有transform_obs_visual正是完整transform_obs的图像分支。正修复future编码入口以复用此图像分支，并在allocation的native capture前增加同一真实future-encoder路径的接口检查；提交仍须具体修复审阅。

## 执行约束与完成判据

- 接口修复经root审阅后已授权唯一第三次GPU提交 `25578290.pbs101`。Root实时独立确认R、host x1000c0s1b0n1/2*16、1GPU/16CPU/110GB/90min，内部80min上限不变；已取回[实际runner_preflight](HISTORY_RUNNER_PREFLIGHT-25578290.json)：visual-only入口probe输出(300,6,384)、finite，RNG/buffers恢复且training flags不变；literal bank (300,5,10) 与native dataset2×frameskip5=CEM10检查通过。Probe是task0初始图像重复的接口检查，不能当真实future质量证据。已进入native capture，5秒GPU telemetry持续；尚无cost gate或mechanism结果。前两次技术ERROR保留，不训练、不改冻结任务/候选/primary门、不自动追加重试。

- Login nodes 仅小控制操作；所有模型/数据 I/O、重放、计算、分析、benchmark 在真实 PBS allocation。入口核对 PBS_JOBID、非 login hostname、PBS_NODEFILE membership。
- GPU 每 5 秒记录 UUID、utilization 和 VRAM 到 job.log；共享环境、旧源码与旧 artifacts 只读，输出隔离；不做额外哈希扫描。
- `25578290.pbs101` 已通过成本门。Root取回[实际cost calibration](HISTORY_COST-25578290.json)与[capture manifest](HISTORY_CAPTURE-25578290.json)：8 tasks/16 contexts齐全、missing为空；单anchor1完整300候选physics重放44.307s，cold/history/truth encoder配对1.757s，capture360.954s，保守全程估计2095.838s低于4800s上限。日志16-context history1 parity通过，已进入机制指标计算；不根据部分contexts推断收益或调整设计。
- 每个阶段开始前保留账户 quota 条件；最近实际查询 ordinary usage allowed，remaining=89%。严禁使用 banked reset。
- 完成要求：两任务各有冻结协议、真实执行及必要的终态证据、结果解释与未支持的 claim 边界。测试/manifest 只能证明其实际覆盖的范围。执行错误不等于研究 NO_GO；未执行的范围不能用代码检查或意图替代。
- History最终 `25578290.pbs101` 已由root独立确认 F / Exit0 / wrapper0 / runner0 / Stageout1 / wall00:24:27，summary存在并已取回。16 contexts/8 tasks/8 nonfloor满足样本门；primary median regret降幅5.63497%、4/8改善，未达到10%/5任务门，科学NO_GO。16个history1 parity目标差全0。按冻结规则停止，不跑history closed-loop；未追加训练/任务/候选。完整结果见[实际summary](HISTORY_SUMMARY-25578290.json)与[结束记录](RESULT.zh.md)。
- 本文件保留过程记录；具体 job IDs 和最终结果以各任务 SUBMISSION / artifacts 及上述结束记录为准。
