# C80 R128 A+Z+KL fused VQ inference

源checkpoint不变，无重训。主goal：input fusion → packed VQ tile decode/BF16 GEMM → direct loader；同GPU完整40 contexts ×2 seeds，各80 queries。

PBS job `25942669.pbs101` (`vqfu0003`)，1 GPU /16 CPU /120 GB，walltime `06:00:00`，normal →g1。监控agent `/root/fused_kernel_watch`，gpt-6-luna/xhigh。

实现：`fused_input.py`（FP32 Smooth/sign/H128及whole-K A4融合）、`fused_vq.py`（uint8 lookup→FP32 decode→BF16 MMA/FP32acc）、`kernel_runtime.py`（packed安装及直接empty/meta加载）、`kernel_worker.py`（GPU预检、614 decoder验证、12代表shape microbench、五个full80面板）、`finalize.py`（端到端误差、latency、memory、storage汇总）。

`protocol.json`固定模式和数值语义；r3已通过synthetic输入精度及614层decoder BF16 bitwise检查，12个代表shape的microbench已生成；完整policy结果尚未完成。每15秒写GPU telemetry。所有编译、模型I/O和推理均在PBS allocation中。控制器仅使用bounded小文件和SSH/PBS控制。

操作：`python control.py status|logs|verify|pull`。提交采用durable intent，禁止未知qsub结果下重复提交。源export bundle引用原完整文件，不下载/复制大checkpoint。

验收：input fusion full80 exact parity；614层decoder BF16 bitwise；packed GEMM改变累加顺序则完整报告action/future误差；五模式同GPU80query及实际resident/query/load peak；输出storage压缩与端到端speedup，不能从microbench推导整模型收益。不新增closed-loop准确率声明。

首轮25942469在synthetic H128编译时失败（constexpr tuple索引），无model query；已采用scalar参数/static butterfly修复，旧artifacts保留。

第二轮25942630在VQ kernel普通全局常量编译时失败，无modelquery；已使用kernel字面量修复。当前r3的完整面板继续执行。VQ布局/tile改进在独立文件准备，不修改已运行artifacts。
