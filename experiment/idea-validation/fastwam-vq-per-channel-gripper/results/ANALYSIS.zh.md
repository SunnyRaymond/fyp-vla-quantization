# Per-channel + gripper loss：完整结果与闭环决定

完成时间：2026-10-19，Asia/Singapore。GPU `25869007.pbs101`、CPU `25869987.pbs101` 均 F / Exit 0，完成标记和完整覆盖检查通过。

**结论：本轮不启动闭环，也不向被暂停的“LIBERO-Plus 闭环”对话发送恢复消息。** 只加 gripper loss 改善了原 held-out 的 gripper RMSE，却没有改善最需要修复的 visited-state 实际开合：错误从 6/400 增至 9/400，漏释放仍为 2 次，motor 和 latent 同时付出代价。这不足以支持再花闭环预算验证当前 checkpoint。

## 实验只改了什么

原 C80、80 updates、update order、seeds、共享 BF16 teacher、optimizer、首次 update 后的状态 reset、rho=1.5 和冻结的 normalization_c 全部沿用。从原 baseline 的 D=1/r=1 开始，训练 614 个 ratios 和 1,806,863 个 channel log2D；books、indices、row gains、原 microscaling exponents 和模型参数冻结。

唯一训练改动为：

`L = mean(first10 motor6 error²) + (1/6) * mean(first10 gripper error²) + rho*c*L_future_latent`

motor 的 mean 有 60 个元素，gripper 的 mean 有 10 个元素。因此 lambda_g=1/6 使第七维与每个 motor 维度具有相同的单元素系数，同时保留原 motor 和 latent 项的系数。没有新增 late22、sign、margin、bias 或局部响应损失。评估固定 final80，没有按 held-out 选择 checkpoint。

原80个 held-out 为 cases3000–3039 × 两个原 seeds；calibration80 也保留原协议。另用原40个 C8-student visited states × base/image±/proprio± 五个输入，复用原 BF16、clip_only、joint、per_channel_clip 的800个端点，只新增当前模型的200个端点。两个面板分别报告。

## 原80个 held-out：gripper 改善，motor/latent 回退

以下为原协议的 query RMSE/NMSE 均值；原量化 baseline 与用户之前的 0.044297 / 0.060877 完全对应。

| 配置 | First10 motor RMSE | First10 gripper RMSE | Future latent NMSE |
| --- | ---: | ---: | ---: |
| baseline，D=1/r=1 | 0.044297 | 0.057054 | 0.060877 |
| clip_only | 0.043848 | 0.059317 | 0.054848 |
| 原 per-channel | 0.031641 | 0.078308 | 0.055378 |
| per-channel + gripper loss | 0.034101 | 0.062996 | 0.057303 |

相对原 per-channel：motor **+7.77%**，gripper **−19.55%**，latent **+3.48%**。相对原 baseline：motor −23.02%，gripper +10.42%，latent −5.87%。加损失以后，第七维仍未回到 baseline 或 clip_only 的水平。

先对同一 case 的两个 seeds 取均值，再作40个配对比较：gripper 改善36/40，motor 改善11/40、退化29/40；latent 改善18/40、退化22/40。gripper 的改善有一定普遍性，并非仅由一个极端 query 拉低均值。

calibration 上也发生了相同方向的取舍：

| 配置 | Motor RMSE | Gripper RMSE | Latent NMSE |
| --- | ---: | ---: | ---: |
| 原 per-channel | 0.029801 | 0.054312 | 0.044685 |
| + gripper loss | 0.032399 | 0.047993 | 0.049945 |

这意味着 motor/latent 的代价已经出现在 calibration 上，不能全部归因于 held-out 过拟合。它与目标之间的取舍相符；本轮没有测 gradient cosine，因此不将其直接断言为严格的梯度冲突。

新模型在原80个 held-out 上有5/800个实际开合命令与 teacher 不同，涉及4/80 queries；switch-location mismatch 为8/720，first10 内漏 open/close 的事件计数均为0。旧 original80 controls 只保留标量误差，不能据此宣称实际开合相对它们改善。跨模型的命令比较见下面 visited 面板。

## Visited40：当前 checkpoint 没有修复控制相关问题

每个 case 的 first10 共400个命令位置、360个相邻位置。Motor 为40个 case 的 RMSE 均值；actual command 使用既有 denormalize→invert→sign，−1=open、+1=close，指控制命令而非物理夹爪运动。

| 配置 | Normalized motor | Command motor | Gripper RMSE | 错误开合 /400 | 出错 cases /40 | Switch mismatch /360 | 漏 open |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_only | 0.045834 | 0.033880 | 0.100220 | 6 | 5 | 7 | 1 |
| joint | 0.044575 | 0.033010 | 0.058090 | 3 | 2 | 3 | 1 |
| 原 per-channel | 0.034658 | 0.027482 | 0.115396 | 6 | 4 | 4 | 2 |
| + gripper loss | 0.037354 | 0.029804 | 0.128591 | 9 | 5 | 7 | 2 |

新模型仍保留相对 clip_only 的 motor 优势，但相对原 per-channel，normalized motor 仅12/40 cases改善，command motor 仅15/40改善。额外 open 事件仍为2次，额外 close 从0变为1次；有序事件方向不一致的 cases 仍为4个。

新模型的2个可配对事件平均绝对延迟为0.5步，原 per-channel 的2个为0步。这里排除了方向序列不一致的 cases，必须与“仍漏2次 open”一起看，不能把可匹配事件的延迟当作完整释放表现。

按 suite 看，base 的错误开合为：Spatial 1→5，Object 1→1，Goal 4→3，LIBERO-10 0→0。Goal 有局部收益，但不足以抵消 Spatial 的退化。

### 具体发生了什么

所有 step 均从0开始，只描述当前 first10 prefix；step0 不推断跨 chunk 的事件。

| Case | Teacher | 原 per-channel | + gripper loss |
| --- | --- | --- | --- |
| 2011，Spatial | 全程 close | 正确 | step3–5 误 open，step6重新 close |
| 2013，Spatial | 全程 open | step0误 close，step1恢复 | step0–1误 close，step2才恢复 |
| 2031，Object | step9 open | 漏释放 | 仍漏释放；motor 0.02984→0.04873 |
| 2037，Object | step8 open | 正确 | 保持正确 |
| 2043，Goal | 全程 close | step8–9误 open | 修复，motor也改善 |
| 2045，Goal | step8 open | 漏释放 | step9才 open，改善为晚一步 |
| 2049，Goal | step8 open | 正确 | 新增漏释放；motor反而0.07927→0.05293 |

所以“漏 open 仍为2”并不表示具体问题没有变化：原来是2031、2045，本轮变成2031、2049。2049再次说明 motor RMSE 下降可以与关键释放失败同时发生。

还逐个读取了9个错误位置的原始连续 gripper 输出。例如2011 step3输出−0.8984、teacher为+1；2031 step9输出+0.9844、teacher为−1；2049 step8输出+0.9883、teacher为−1。**这些错误远离 sign 的0阈值，是相反的开合模式，不是零附近数值抖动。** 记录保存在 `base_gripper_failure_values.json`；简单改阈值无法同时修复两种方向的错误。

### 五个输入的表现

每格均为同样40 cases × first10 的错误开合数 /400。这五组存在配对相关性，不能当作200个独立 states。

| 输入 | clip_only | joint | 原 per-channel | + gripper loss |
| --- | ---: | ---: | ---: | ---: |
| base | 6 | 3 | 6 | 9 |
| image+ | 6 | 4 | 4 | 9 |
| image− | 6 | 1 | 4 | 6 |
| proprio+ | 7 | 7 | 6 | 6 |
| proprio− | 10 | 4 | 4 | 5 |

相对原 per-channel，四组更差、一组持平。proprio+ 的漏 open 由3降至1，但它并未形成整体稳定的开合收益。

### Motor 偏置也没有一并解决

使用 `mean_cases(abs(mean_first10(student−teacher)))`，避免相反方向的 case 互相抵消。以下为 normalized action 的六维结果。

| 配置 | dx | dy | dz | drx | dry | drz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_only | 0.01774 | 0.01851 | 0.02413 | 0.01642 | 0.01863 | 0.01558 |
| 原 per-channel | 0.01994 | 0.01851 | 0.01509 | 0.01143 | 0.01492 | 0.01467 |
| + gripper loss | 0.01948 | 0.01747 | 0.02541 | 0.01120 | 0.01376 | 0.01327 |

五维略有改善，但 dz 明显恶化；实际 command 的 dz 偏置也从0.01415升至0.02382。Goal 的 dz 从0.01915升至0.03502，Spatial 从0.01282升至0.03140。不能用“五维下降”概括为整体控制偏置改善，也不能只看跨 case 的 signed mean。

局部有限差分也有代价：image 的 teacher-relative derivative error RMS 从4.87009升至5.01410，proprio从3.69482升至3.99485。相对原 per-channel，image只有20/40改善，proprio只有18/40改善。两者仍优于 clip_only 的6.11791和4.18713，但没有新增的整体响应优势。

Teacher 在这些小扰动下实际 gripper flips 均为0；新模型相对 base 的 flips 在 image± 为6/5（原4/4），proprio±为9/4（原4/2）。这里只测了两个规定方向，不能解释为完整 Jacobian 或闭环稳定性证明。

## 这对原因判断意味着什么

1. **遗漏第七维监督确实是原目标的缺口，但补这一项并不充分。** 新 loss 在 calibration 和原 held-out 都降低了 gripper RMSE，却没有迁移到 visited states 的正确开合。不能再把问题简单归结为“加上 gripper MSE 就会转化为闭环收益”。
2. **状态覆盖不足比单纯增加训练步数更值得优先检验。** 原 held-out 的 gripper 改善36/40，而另一组 visited states 上出现高幅度的相反开合模式。这与 calibration 主要覆盖原始输入、缺少策略执行后接触/抓取/释放状态的情况相符。但当前没有更换 calibration 做因果对照，因此这是下一步假设，而非已证明的唯一原因。
3. **同时存在目标取舍和 motor 偏置问题。** motor/latent 在 calibration 与 held-out 都回退，visited dz 偏置明显增大；只改善第七维平均误差无法约束这些控制代价。更多参数、更多 updates 或更大的 lambda 都缺乏当前结果支持。
4. **本轮所检查的 loss 与部署路径是接通的。** 独立解析检查确认原 motor 梯度不变、first10 gripper 有新增梯度、late22无新增梯度；614个模块的 ratio/channel 梯度均连接、有限、非零。D1/r1与原baseline、cached/uncached action+future、实际 infer、BF16 visited replay 和 visited repeat 的 max_abs 均为0。冻结参数检查通过。没有证据将本轮失败归因于漏加 loss、用错 checkpoint 或缓存/推理路径不一致；这不等于对全部实现作无条件正确性保证。

先前 per-channel 的280条闭环是203成功，clip_only为207；motor改善不自动意味着成功率改善。原 visited 的LIBERO-10没有开合错误，却仍有该 suite 的闭环回退，因此 gripper 也不是已证实的唯一原因。旧闭环与本轮 fixed-input 面板的证据边界见此前 `fastwam-vq-per-channel-visited-diagnostic/results/ANALYSIS.zh.md`。本轮没有新增 rollout、环境 step 或 native-kernel 性能测量。

## 建议的下一步：先改变 calibration 覆盖，保持 per-channel 实现

建议下一轮先做一个数据组成因素：保持当前 per-channel + lambda_g=1/6、从D1/r1开始、80 updates及其它超参不变，用训练侧独立轨迹中的80个状态，替换目前 C80 的状态组成。按 approach、pre-grasp、closed holding、transport、pre-release、post-release 分层，保留四个 suites 的覆盖；teacher 标注 open/close 转换，并覆盖转换前后进入 first10 执行区间的 chunk。

这是“同样80个状态/80 updates，改变覆盖”的对照，先与本轮比较。原80 held-out和这40个visited states继续固定用于评估，不回灌训练，不新增 development set，也不按这些评估结果选中间 checkpoint。新训练轨迹应与评估轨迹保持隔离；需要新增的 teacher targets 应在下一轮预算中单独记账。

如果覆盖改进后仍出现同类模式错误，再单独比较均匀 gripper MSE 与训练侧预先定义的 transition-balanced gripper MSE，保持 open/close 两种方向与持有阶段的对照，保留原 motor6/latent 系数。不要在同一轮同时更换数据、提高 lambda、增加 bias loss，否则无法判断改进来自哪里。dz 的 chunk-mean bias 可作为随后独立约束的候选；本轮不追加训练。

数据量扩大可以放在覆盖因素之后，并用相同总 updates 的“重复旧数据”control区分新状态收益与更多优化步数收益。当前结果不支持只重复原 C80 更久，也不支持仅根据原 held-out gripper RMSE 推进当前 checkpoint 的闭环。

## 完成与资源记录

- 一份 A100 GPU 作业：`25869007.pbs101`，PBS walltime **01:32:30**，小于2小时；CPU汇总 `25869987.pbs101`，walltime00:01:07，0 GPU。
- 新模型查询445：训练80 + original evaluation新增159 + visited200 + parity6。Original evaluation保存160条记录，calibration首条复用folding parity端点；parity不计作新增teacher targets。新增teacher targets和rollouts均为0。
- Visited汇总1000条端点：复用800、新增200；四个students ×40 states ×2输入方向族，共320条响应记录。Original汇总640条：160新记录、480旧controls。
- `gpt-6-luna / xhigh` 每10分钟只读监控已结束，终态快照 `monitor_state.json` 标记complete。作业每15秒向job log记录GPU利用率及显存；本地保留末段日志。
- 当前决定为 `do_not_run_closed_loop`；被暂停对话未收到本线程的恢复消息。当前checkpoint及全部结果保留，便于后续作为control。

完整数值：`comparison.json`；配对case与事件：`diagnostic_details.json`；原80新模型逐query：`heldout_details.json`；训练/部署证据：`train_eval/training_receipt.json`、`train_eval/summary.json`；最终覆盖：`completion_evidence.json`。
