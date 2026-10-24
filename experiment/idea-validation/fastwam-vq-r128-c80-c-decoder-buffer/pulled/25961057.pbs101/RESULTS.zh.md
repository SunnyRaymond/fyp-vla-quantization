# C：decoder、buffer 与未量化参数分析

两个GPU候选均完成完整80条，数值stage全部通过。推荐decoder+buffer：相对复用C，p50下降7.30%，query peak减少35.55MB。原GPU wrapper因汇总脚本把跨job复用误要求为同一job而退出；该记录规则已修复；CPU-only作业25961057已F/Exit0，重算后的完整交付门通过。原始GPU wrapper失败状态完整保留。

## 完整 policy-call

| 实现 | Mean s | p50 s | p95 s | Query peak allocated GB | 与原量化 checkpoint 输出 |
|---|---:|---:|---:|---:|---|
| C，复用25956803 | 0.956361 | 0.956914 | 0.959222 | 16.949808 | 完整80条一致 |
| C + decoder，25959865 | 0.890745 | 0.890160 | 0.898470 | 16.949808 | 完整80条action max/RMSE=0，anchor action/future=0 |
| C + decoder + buffer，25960283 | 0.886964 | 0.887096 | 0.894262 | 16.914255 | 完整80条action max/RMSE=0，anchor action/future=0 |

每个候选完整40 contexts（3000–3039）× seeds 0/1=80，batch1、32×7 action、9 frames、10 video + 10 action denoising steps、3 warmups。计时范围为 prepared GPU datum 到 infer_action 返回并同步，包含内部 text/image-latent 编码和 denoising，排除 observation preprocessing、环境及机器人执行。主计时没有 profiler；加载、GPU预检及Graph capture不计入主latency。

已有 BF16/eager/input_fused/layer_decode_fp32/C 全部复用，没有重跑完整对照。所有GPU均为A100-SXM4-40GB；跨allocation比较不能把小幅差异当作严格配对因果估计。

## Decoder 改动与证据

独立decoder按4维codeword和B32 scale block复用indices、exponent及row scale，在5个有界tile配置中选择；保留原FP32 book sum、ldexp、multiply及BF16写出顺序，主Linear仍调用原cuBLAS。17种weight矩阵形状中，13种选block512/4warps，1种选block256/4warps，3种回退原实现（1024×7、1024×256、4096×8）。

全部614个真实weight的raw BF16 bitwise检查通过；614个activation operand检查通过。完整80条action误差0，独立anchor action/future误差0。没有训练、calibration或closed-loop。

在25959865的同一allocation上，按候选实际Graph replay与外部调用次数加权，原decoder估计264.144ms/query，新选择173.622ms/query，下降34.27%。这是decoder-only微基准估计，不能写成完整policy-call加速。完整policy-call相对复用C的p50下降约7.0%。

逻辑decoder调用仍为5258/query。按各weight shape与真实调用频次计算，每query累计写出约107.159GiB的BF16 decoded weights，其中video约91.422GiB、action约15.738GiB；这是重复解码的累计写出量，不是常驻显存或模型文件大小。

## Buffer 实现

原C的独立H/U/A4 workspace为42,147,840bytes。buffer候选让FP32 H与U的两个中间buffer复用已有113,246,208bytes的weight scratch；A4输出仍用独立BF16 buffer。顺序为H→U→A4→weight decode→cuBLAS，所以weight decode覆盖H/U时，两者已不再使用。

独立activation workspace降为8,429,568bytes，设计减少33,718,272bytes。实际query peak比decoder版减少35,553,280bytes（35.55MB）；其中33.72MB可由H/U复用直接解释，其余约1.84MB未做逐storage归因。p50从0.890160s到0.887096s，仅差0.34%，跨allocation应视为基本持平。该方案仍要求单请求、单CUDA stream，延续C已有的执行限制。

## 11.902287906GiB究竟是什么

参数清单生成于25959495，后续作业复用；完整3547个state keys按底层storage去重后为1475个storages、12,779,984,326bytes。所有remaining parameter storage均为BF16。2072个额外key引用来自aliases，不能再计作多份常驻参数。

| 组件 | Unique bytes | GiB | 说明 |
|---|---:|---:|---|
| Text encoder | 11,361,820,672 | 10.581520 | 未进入当前614个VQ Linear覆盖范围 |
| VAE | 1,409,377,336 | 1.312585 | 包含encoder与decoder |
| Video主干剩余参数 | 5,990,784 | 0.005580 | bias、normalization等小参数 |
| Action主干剩余参数 | 2,787,342 | 0.002597 | bias、normalization、modulation等小参数 |
| Proprio bias | 8,192 | 0.000008 | weight已量化，bias保留 |

Text encoder中FFN为5.625GiB、attention为3.000GiB、token embedding为1.956055GiB，其余约488KiB。VAE decoder为1,110,098,456bytes（约1.034GiB），encoder为299,255,552bytes，其余23,328bytes。

新候选anchor+3warmups的探针记录显示text encoder的242个direct parameter owner全部执行4次；VAE的124个owner中观察到52个执行。探针在正式80条计时前移除。模块可能通过direct method/parameter reads绕过forward hook，因此未观察项不自动等于所有路径都不用；完整aliases及owner调用明细已分片保存。

目标VQ含transforms存储仍为2.991408997GiB，完整inference archive仍为14.895613806GiB；本轮没有另存权重或改变编码。Archive不包含外部cached tokenizer/base assets的完整部署目录用量。

## 执行记录

R1：25959495在生成完整参数清单后，decoder首次GPU编译因tile缺少constexpr声明退出，尚未执行候选anchor/full80。短任务25959497在Q时取消并核实F。

R2：25959865完成decoder完整80条与全部数值检查；后续gate因旧reference summary没有新cohort字段而退出。完整候选数据保留，未重跑。短任务25959870在Q时取消并核实F。

R3：25960283先按旧reference来源protocol核对完整geometry，并离线复核R2的80条action；decoder gate全部通过后，仅执行buffer候选。两个stage均passed，buffer额外profiler完成。GPU job F/1，实际14m28s：最终汇总错误要求C1与C2来自同一PBS job；正确规则应为C2的reuse_source_pbs_jobid等于C1的producer。该规则已修并用实际收据做回归测试。短任务25960284在Q时取消并核实F。CPU后处理25961057已使用缓存actions重新汇总并通过全部门；没有重跑GPU forward，原job的F/1状态保留。

本地控制/边界接口测试及bash语法检查通过；另以实际跨job收据补充producer identity回归测试。GPU预检在compute allocation执行；GPU利用率与显存每15秒写入job log。没有login-node模型计算、重复对照forward、训练或闭环扩展。

## 剩余latency

以下来自额外intrusive诊断query，非headline latency；只比较CUDA leaf kernel类别，不把ATen与其子kernel重复相加。C的已有诊断复用，新候选只增加1条buffer诊断query。

| CUDA kernel类别 | 原C ms | decoder+buffer ms |
|---|---:|---:|
| Weight decode | 229.634 | 150.871 |
| BF16 GEMM类 | 227.548 | 226.973 |
| FP32 SGEMM类 | 99.850 | 99.951 |
| Smooth/H | 51.711 | 51.726 |
| A4 | 38.139 | 37.948 |
| Fill类 | 58.139 | 58.083 |
| Copy类 | 48.231 | 47.739 |
| 主要attention类 | 约32.1 | 约32.2 |

decode下降约34.3%，而没有改动的GEMM、H/A4、fill等基本维持原量级，支持decoder优化是约7.3%完整latency收益的主要来源。目前最大的GEMM类别已是BF16；FP32 SGEMM仍约100ms，不能称为完全没有优化空间。

buffer版mean policy-call为886.964ms，CUDA event denoising区间约825.032ms，差约61.933ms；区间包含GPU等待，不能当作严格active-kernel求和。

## 加载和显存分项

| 候选 | Load s | Load peak GB | 验证/准备/warmup s | Setup peak GB | Warm allocated GB | Query peak reserved GB |
|---|---:|---:|---:|---:|---:|---:|
| decoder | 167.647 | 16.294098 | 64.542 | 16.949808 | 16.684367 | 17.307795 |
| decoder+buffer | 534.107 | 16.260380 | 49.418 | 16.913894 | 16.649535 | 17.274241 |

加载包含软件初始化和权重恢复；两个候选来自不同allocation，日志亦出现RPC连接超时。没有隔离加载变慢的根因，不能把这列视为buffer的cold-start因果效应。准备阶段含614层GPU预检、anchor、Graph capture和3warmups，不是单独的生产冷编译测试。上述成本不混入主policy-call，但当前原型启动仍需进一步优化。

## 后续方向

1. 优先验证固定指令text embedding缓存和text encoder按需驻留。它影响约10.582GiB参数，且本轮确认每query确实执行；新指令的首次编码/加载成本必须另报。不能跨变化的proprio或图像错误复用。
2. 检查action-only、返回future latent的路径是否可按需加载VAE decoder。该部分约1.034GiB；当前hook窗口未观察不能单独证明所有使用路径都不需要，需结合代码调用与精确parity验证。
3. Latency侧保留本轮decoder，并考虑融合转换后的BF16 U小原型，或VQ→INT8 scratch→真实INT8 GEMM。当前BF16 GEMM约227ms、decode约151ms是更大的GPU成本。不要把codebook改INT8的约1.3MB存储收益误当成整模型GB级省显存，也不要把decoder微基准加速写成端到端加速。

相对现有未同等Graph/条件缓存优化的BF16 reference，本轮组合约1.635×、peak下降32.69%。这是当前runtime组合的比较，不能独归于VQ，亦不证明击败同等优化后的BF16实现。没有新增closed-loop，future只验证独立anchor。


CPU postprocess短任务25960939（qdev、20min）排队；长任务25961057（q1、1CPU/8GB、02:00:01）先启动后，短任务在Q时精确取消并核实F。长CPU作业实际1m52s、F/Exit0、artifact PIPELINE_COMPLETE与post controller verify均通过；new_model_forwards=0。修正后的报告位于pulled/25961057.pbs101/summary.json，原始GPU失败汇总仍在pulled/25960283.pbs101/summary.json。

本轮GPU allocation合计31m07s（R1 8m46s + R2 7m53s + R3 14m28s），CPU后处理1m52s；所有8个PBS句柄终态已核实，4个较晚启动的sibling都在Q时取消，没有额外GPU allocation。监控gpt-6-luna/xhigh已结束。
