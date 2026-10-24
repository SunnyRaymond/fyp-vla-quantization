# C80 R128 A+Z+reader KL inference export 与 profiling

最终结果：job `25938083.pbs101` 已成功完成，F/Exit0；三组主要模式各80次，source与export action/future parity为0；短任务已取消并核验。未重训，监控已停止。

- [最终结果报告](pulled/25938083.pbs101/FINAL_REPORT.zh.md)
- [Kernel实现方案](KERNEL_DESIGN.zh.md)
- [Storage明细](EXPORT_STORAGE.zh.md)
- [机器可读reviewed kernel plan](pulled/25938083.pbs101/REVIEWED_KERNEL_PLAN.json)

记录中的原始 `kernel_plan.json` 是自动初稿；最终采用reviewed方案的输入融合→packed VQ GEMM→loader顺序。Raw analyzer中单个exact-name proprio行的分类已在reviewed JSON纠正，未来analyze.py也已修复；未因此新增GPU推理。

固定 source calibration `25921795.pbs101` 的 `reader_kl` final8 endpoint，继承已完成闭环 `25925636.pbs101` 的运行源码。新训练更新为0，不更改 codebooks、indices、B32 exponents、row gains、Smooth、H128、U、A4、KV 或采样步骤。

单个 `normal → g1` PBS allocation：1 GPU、16 CPUs、120 GB host memory、4小时 walltime。每15秒记录分配GPU的利用率和显存。作业完成后立即退出。Login node只上传小型源码和查询/提交；所有模型加载、导出、数据读取和profiling均在allocation内执行。

运行顺序为四个独立进程：

1. `legacy`：安装当前已验证的模型，重放保存的完整action/future anchor，导出inference-only bundle，测量完整40 held-out contexts ×2 seeds。
2. `bf16`：原始BF16 checkpoint，同样的80输入/采样种子及10+10 denoising steps。
3. `inference_dense`：从导出配置构造模型，从bundle加载部署状态，无需训练checkpoint或source banks；完整80 queries与legacy逐一比较。
4. `stream_decode`：anchor及3000/seed0共2个诊断queries。只常驻VQ编码，每个Linear临时完整解码weight；用于compressed residency诊断，不是fused kernel或主要speedup结果。

Factory仍依赖FastWAM源码以及缓存的base VAE、text encoder/tokenizer资产。Bundle包含未量化model state、614模块最终VQ编码、Smooth/sign、固定FP32 U、normalizer与resolved config；不包含量化模块的dense BF16 weight、FP32 training masters、optimizer或frozen reader。

每个主要模式3次warmup，再运行80个prepared-GPU-datum→action-output的wall/GPU计时。Observation读取和预处理另计；不计环境step或机器人运动。加载瞬态peak、warm resident、query peak分开报告。详细CUDA event attribution与Torch trace用独立单次query，绝不放入主延迟样本。Profiler中的stage、Linear、component、GEMM是嵌套范围，不能直接相加。

`load_seconds/load_transient_memory` 仅作各模式启动诊断：legacy计入source恢复和`prepare`的两次anchor验证，首个export writer还计入bundle写盘；BF16/export loader的计时不含随后anchor验证。因此不能据这些字段计算公平的cold-start speedup。Warm resident与80次query peak/latency才是本轮跨模式主要对比。

结果保存于remote root的`results/`：`export/inference_bundle.pt`、`storage.json`、分片`per_layer/*.json`、各模式`samples.json/actions.pt/detail.json/operators.json/trace.json`、`parity.json`、`summary.json`、`kernel_plan.json`和`RESULTS.zh.md`。仅小型JSON/Markdown通过controller拉回本机；大bundle和trace留在compute filesystem。

控制命令：

```powershell
& 'C:\Users\Raymond\miniconda3\python.exe' -Xutf8 -B control.py upload
& 'C:\Users\Raymond\miniconda3\python.exe' -Xutf8 -B control.py submit
& 'C:\Users\Raymond\miniconda3\python.exe' -Xutf8 -B control.py status
& 'C:\Users\Raymond\miniconda3\python.exe' -Xutf8 -B control.py verify
& 'C:\Users\Raymond\miniconda3\python.exe' -Xutf8 -B control.py pull
```

固定job名`vpinf0003`；durable intent后重复submit只恢复已有handle，不重复qsub。完成要求PBS终态/Exit0、matching summary的export/parity/profile全部完成、new_training_updates=0、artifact exit0与PIPELINE_COMPLETE。失败时保留已有export与measurements；修复不得重训，恢复必须复用数值及协议兼容的已有产物。

这轮输出的是固定checkpoint的部署/性能证据与下一步kernel方案，不新增closed-loop成功率结论，也不声称已经实现native VQ low-bit arithmetic。

首轮25937187在完整config解析Hydra输出目录时失败；source action/future anchor均为0误差。第二次运行使用-r2 root，保留原checkpoint、完整80-query面板和4h预算，仅修复部署配置导出。

第三次运行使用 `-r3` root，修复 formal infer CPU action 与 sampler CUDA action 的比较。复用第二轮已完成的 bundle 和 storage receipts，不复制大 bundle；所有主要计时模式在新 allocation 的同一 GPU 上重新跑完整80次，避免跨GPU混合比较。

用户授权双队列竞速：长job `25938083.pbs101`（g1/4h），短job `25938321.pbs101`（gdev/1h59m）。短job通过 `short_control.py` 管理，独立root与结果；完整80-query面板、checkpoint及数值源码不变。`queue_race.py`每30秒由monitor检查两个精确handle，先R者保留，另一job执行qdel并验证F；若stime同秒，以先提交长job为确定性tie-break。仅监控winner继续完成，不合并两张GPU的主性能样本。
