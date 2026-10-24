# C：decoder 与 buffer 优化

基于 `25956803.pbs101 / runtime_c`，冻结 `25921795.pbs101 / reader_kl` checkpoint。

本轮只执行两个新候选：`c_decoder` 对独立 weight decoder 做有界调优；`c_buffer` 在相同 decoder 上让 H/U FP32 中间值复用原有 decoded-weight scratch。A4 operand 使用独立 BF16 buffer，避免 weight decode 覆盖其输入。仍为单请求、单 CUDA stream。

所有已有 BF16、量化 eager、input_fused、layer_decode_fp32 和 C 的完整面板结果直接复用，不重新运行。新候选各保留 contexts 3000–3039 × seeds 0/1，共80条；batch1、32×7 actions、9 frames、10+10 denoising steps、3 warmups。要求614个decoder/activation operand raw BF16检查通过，完整action及独立anchor action/future误差为0。

主指标为无 profiler 的完整 policy-call p50/p95/mean、query peak allocated/reserved、load/setup peak及weight storage。独立诊断query的kernel计时仅用于定位，不作主latency。保持原 deterministic/TF32/precision flags。

单独生成未量化parameters的storage去重清单，归因11.902287906GiB到实际组件、tensor和aliases。执行探针复用新候选anchor/warmup，在正式80条计时前移除；未观察到调用不等于全局无用。

同时提交 normal→g1 六小时和gdev 1h59两份作业。验证allocation后原子选winner，只取消确认的sibling；GPU利用率/显存每15秒写入job log。所有tensor I/O、GPU编译和模型计算均在已验证PBS compute allocation执行。

没有训练、calibration、closed-loop、BF16 U、INT8或torch.compile扩展。

已完成，推荐 `c_buffer`：p50 0.887096s、query peak 16.914255GB，完整80条action和独立anchor精确对齐。详见 [RESULTS.zh.md](RESULTS.zh.md)。原始GPU收据在 `pulled/25959865.pbs101/` 与 `pulled/25960283.pbs101/`；修复汇总的CPU验收在 `pulled/25961057.pbs101/`。
