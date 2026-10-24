# C80 R128 A+Z+KL：runtime A/B/C 与 torch.compile

实验已结束，推荐 C（A 的共享 workspace + B 的条件缓存 + 手动 CUDA Graph）。A/B/C 各完成完整80条并输出对齐；额外 torch.compile 两次试跑均未完成，作为失败候选记录，不采用。

## 完整 policy-call 结果

| 版本 | Mean s | p50 s | p95 s | Query peak allocated GB | 与原量化输出 |
|---|---:|---:|---:|---:|---|
| BF16 reference，复用 | 1.451284 | 1.450439 | 1.505931 | 25.130502 | 原模型参考 |
| 原量化 eager，复用 | 6.547441 | 6.527626 | 6.695142 | 25.139422 | 量化参考 |
| 输入融合 dense，复用 | 3.520413 | 3.503950 | 3.724216 | 25.139422 | 一致 |
| 原逐层解码，复用 | 3.593032 | 3.532909 | 3.936028 | 16.550766 | 一致 |
| A：共享 H/U/A4 workspace | 3.676168 | 3.693649 | 3.770497 | 16.592381 | 80/80 action 一致 |
| B：A + query 内条件缓存 | 3.054290 | 3.047032 | 3.243571 | 16.688547 | 80/80 action 一致 |
| C：B + CUDA Graph | 0.956361 | 0.956914 | 0.959222 | 16.949808 | 80/80 action 一致 |
| B + torch.compile | 无有效结果 | 无有效结果 | 无有效结果 | 未完成测量 | 准备阶段崩溃，不采用 |

GPU GB 为10^9 bytes。各候选完整40 contexts（3000–3039）× seeds 0/1；batch1、32×7 actions、9frames、10video+10action steps、3warmups。计时范围为 prepared GPU datum 到 infer_action 返回并同步，包含内部text/image latent编码与denoising，排除外部observation preprocessing、环境与机器人执行。

主对照复用 `25954757.pbs101 / layer_decode_fp32`，其余三个参考复用 `25943391.pbs101`。均为A100-SXM4-40GB；旧参考来自不同allocation，候选A/B/C在同一新allocation内依次运行，不把小幅跨作业差异当作严格因果效应。未重跑任何旧版本。

## 每步到底带来了什么

A保持FP32 Smooth/H/U/A4算术、BF16 cuBLAS Linear及原weight decoder，增加42,147,840bytes共享activation workspace。614层首次operand bitwise检查全部通过；full80 action max/RMSE均0，独立anchor action/future均0。本次A未获得独立E2E收益，peak增加约41.6MB。诊断fill_从旧路径50,027次降为37,155次，但没有修改的decode/GEMM等GPU时间也比旧allocation更长，不能仅由当前跨作业比较断言预分配本身必然使计算变慢。A提供了C所用的稳定buffer地址。

B在每个query内缓存video/action text_embedding结果及每层cross-attention K/V，query间失效并重算。额外cache payload为96,165,888bytes，恰对应B−A约96.2MB的query peak差。正式每query目标Linear从6,436降至5,258，少1,178次：1,140个K/V Linear和38个text projection Linear。该缓存包含当前proprio，因此不跨query直接复用。full80 action及独立future anchor均精确一致。B相对本次A p50减少约17.5%，相对复用的逐层解码版本减少约13.8%。

C保持B同样的5,258个逻辑目标Linear/query，将video_step、video_prefill、action_step三段记录为CUDA Graph。正式80个query中回放次数分别800、80、800，所有capture/replay gate通过；其中5130个逻辑目标Linear/query在graph内执行，128个在外。full80 action及独立future anchor均0误差。C相对B p50下降约68.6%，相对旧逐层解码下降约72.9%。

C相对现有未Graph优化BF16 reference约1.52×加速、query peak约下降32.6%；相对原量化eager约6.82×加速。它是当前完整runtime组合相对现有参考的结果，不能单独归功于VQ编码，也不代表已击败同样经过缓存/Graph优化的BF16实现；这种新参考尚未测量。

## 显存、准备耗时与存储

| 新模式 | Load peak GB | Setup peak GB | Query peak GB | Warm allocated GB | 验证/准备/warmup s |
|---|---:|---:|---:|---:|---:|
| A | 16.294098 | 16.592381 | 16.592381 | 16.327653 | 59.063449 |
| B | 16.294098 | 16.688547 | 16.688547 | 16.423819 | 14.200389 |
| C | 16.294098 | 16.949808 | 16.949808 | 16.684367 | 9.540991 |

Setup覆盖anchor、workspace验证、capture/compile及3warmups，独立于load和主latency；不同模式共享机器上的JIT缓存状态，不能将该列当作隔离测量的cold compile时间。C三个region的setup累计约4.48s，包含对应eager/side-stream warmup与capture。Graph使B→C整体query peak增加261.3MB，尚未逐storage归因；这部分运行时成本已计入主表。

所有模式复用同一个inference bundle：目标quantized weights含Smooth/sign/U为2.991408997GiB，未量化parameters为11.902287906GiB，整个unique tensor payload为14.893702424GiB，实际archive为14.895613806GiB。没有新训练、额外weight archive或更低bit的GEMM；main Linear继续是BF16 operands/cuBLAS，weight按层解码到113,246,208bytes共享scratch。

## 目前剩余瓶颈

C额外诊断query中，最大单个kernel类别是weight decode：5,258次，累计约229.6ms；主BF16 GEMM、FP32 U、H/A4、normalization等仍有GPU成本。该query的H约51.7ms、A4约38.1ms；attention主kernel约32.1ms。这些来自intrusive profiler，不作为主latency严格分解，也不混加ATen与CUDA leaf的重复记录。

B→C在逻辑Linear数不变时大幅降时；profiler中host侧addmm从5,258次降到128次、mm从5,415次降到295次，支持原先大量时间消耗在提交/调度链条。C平均policy-call约956.4ms，denoising区间外约66.8ms；该区间统计仍包含GPU等待，不等于active kernel求和。

后续若继续优化，先考虑保持现有数值的独立decoder向量化/有限tiling，或用双buffer重叠decode与计算；当前结果不支持重新采用此前明显更慢的通用fused VQ GEMM。BF16 U若再试，应融合dtype转换并保持H的FP32中间计算；当前已验证的C不依赖降低这些中间精度。

## torch.compile 试跑结果

首轮 `25956803.pbs101` 在完成A/B/C后尝试对video_step/video_prefill/action_step做Inductor编译，关闭自动CUDA Graph。compile anchor阶段长时间没有GPU活动；连续两段约4分钟的qstat记录中CPU累计仅增长6秒、随后不再增长。为避免空占GPU，确认当前job身份、状态和阶段后精确qdel，终态F/Exit141，A/B/C收据完整保留。

第二轮仅重试compile：单编译进程、独立job cache、保留BF16中间cast的 `emulate_precision_casts=True`，并配置每180秒调用栈诊断和1200秒setup上限。A/B/C只复制旧收据，在compute allocation内离线复核80条action，不重新forward或计时。

重试 `25958068.pbs101` 在约3分钟准备阶段打印调用栈，主线程位于Python `inspect._signature_from_function`，随后进程segmentation fault，worker exit139。1200秒上限没有触发；不能把这次崩溃简单称为超时，也尚未定位到可修复的底层根因。没有compile版的完整80条输出、有效latency或query peak，故不将它列为可用推理版本。

PBS wrapper最终Exit0仅表示失败收据和汇总正常写出。原始四模式成功判定仍是 `complete=false / deliverable_complete=false / verify=false`，没有把失败改为成功。用户要求的A→B→C主线全部完成，额外“试一下torch.compile”得到上述负结果。

## 执行与证据

| Round | Long winner | 实际allocation walltime | Short sibling | 终态 |
|---|---|---:|---|---|
| R1 | 25956803.pbs101 | 40m48s | 25956841.pbs101 | A/B/C完整；compile停滞后long F/141；short在Q时qdel0后F |
| R2 | 25958068.pbs101 | 3m25s | 25958070.pbs101 | 仅compile重试失败139；wrapper F/0；short在Q时qdel0后F |

两轮均为long6h与short1h59竞争，compute allocation内原子选胜者，另一份精确取消。两个short均未分配compute host。新GPU allocation合计44m13s；GPU利用率/显存每15秒写入job log。gpt-6-luna/xhigh监控已停止，四个PBS job均已重新查询并确认为F。

A/B/C实测原始收据位于 `pulled/25956803.pbs101/`；R2的独立离线复核、明确source-job绑定及compile失败收据位于 `pulled/25958068.pbs101/`。本地 `completion_evidence.json`逐项记录full80、614层operand、capture/replay、任务终态及失败候选；原始四模式pipeline失败状态保留。完整actions.pt和trace.json留在compute侧，本地只拉取有界小型收据。

验证范围为冻结输入形状、batch1及单请求串行；共享scratch和Graph buffer不能直接用于并发请求。没有新训练或closed-loop episode，full80一致支持当前固定输入面板及anchor的数值对齐，不替代新的闭环结果。
