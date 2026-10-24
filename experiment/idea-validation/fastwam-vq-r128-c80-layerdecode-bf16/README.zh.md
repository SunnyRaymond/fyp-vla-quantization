# C80 R128 A+Z+KL：逐层快速解码与 BF16

已完成。第一轮p50 3.533s、peak allocated16.551GB，完整80条action误差0；BF16 U为4.166s，更广BF16 input为3.982s且均改变输出。推荐第一轮。详细结果见long/pulled/25954757.pbs101/RESULTS.zh.md。

直接复用 checkpoint `25921795.pbs101 / reader_kl` 和既有 inference bundle，不重训。

第一轮：压缩权重常驻；单个可复用 BF16 scratch；当前层由 Triton 一次解码后交给 cuBLAS `F.linear`。保留已验证的输入融合及 direct/meta loader。

完整固定面板为 contexts 3000–3039 × seeds 0/1，共 80 queries。已有 BF16、eager、input_fused 的完整面板来自 `25943391.pbs101`，按用户要求直接复用。它们与新增结果来自 A100 的不同 allocation，因此比较会注明历史对照来源。

只有完整第一轮的 action/source future anchor 对齐、mean/p50 latency 下降、query/load peak allocated memory 下降，才进入第二轮。比较基准为既有 eager；input_fused 与原 BF16 也保留在最终表中。

第二轮：先测试 U 的原生 BF16 GEMM，再测试更广的 BF16 input-transform 数据流。BF16 MMA 保留 FP32 accumulation；除法、reduction、rounding 仍有 FP32 helper，不能声称整模型全部原生 BF16。weight 解码保持原舍入。

长任务 `25954757.pbs101`（6h）与短任务 `25954759.pbs101`（1h59）竞速。计算节点上的原子锁只允许一份进入模型计算，监控取消另一份。GPU 利用率与显存每15秒写入 job log。

监控使用 gpt-6-luna / xhigh。当前状态见 RUN_STATE.json、race_status.json、monitor_status.json。

命令入口：

```text
python control.py race
python control.py status --lane long
python control.py logs --lane long
python control.py verify --lane long
python control.py pull --lane long
```

如短任务胜出，将命令中的 long 改为 short。所有模型/tensor I/O、编译和推理均只在核实的 PBS compute allocation 内运行。控制器只传输有大小上限的源码、JSON、文本收据；完整 action/trace 留在集群。

CPU-only 检查：`python -B -m unittest discover -s . -p "test_*.py"`。它们检查条件 gate 与脚本契约，GPU 正确性由作业内614 decoder及12种 Linear shape 验证。

旧尝试：r1 在用户要求复用对照后取消；r2 因提交目录命名问题在模型计算前退出。来源与取消收据保存在 attempts/ 和 superseded_r*.json。
