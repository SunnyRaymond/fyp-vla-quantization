# 阶段 0 速度检查：INT8 vs BF16（Fast-WAM-IDM）

路线 3（VQ 存储 + INT8 计算）的速度前提检查。回答两个问题：

1. 在 Fast-WAM-IDM 推理里真实出现的 Linear 形状上，INT8 dense GEMM（cuBLASLt，`torch._int_mm`）比 BF16 cuBLAS 快多少？
2. 整次 policy call：BF16 加上与 engine 相同的 condition cache + CUDA Graph 后是多少？与 released BF16 eager、Q0 c_buffer engine 在同一 allocation、同一 80 个固定 query、同一计时口径下比较。

不训练、不跑闭环、不改任何 checkpoint。单张 A100（g1），walltime 3 h，预计 30–40 分钟。

## 五个模式（每个独立子进程，顺序执行；某个失败不影响其余）

| 模式 | 做什么 |
|---|---|
| `shapes` | BF16 runtime 跑 case 3000 的一个 query，记录 614 个目标 Linear 每次调用的 (M, K, N, bias)；分别记录 eager（约 6436 次/query）和装上 condition cache 后（约 5258 次/query）的分布 → `results/shapes/shapes.json` |
| `gemm` | 对每个唯一 (M, K, N, bias) 测三种：BF16 `F.linear`；INT8 纯 GEMM `_int_mm`；INT8 含外围（per-token 量化 + `_int_mm` + 乘 scale 加 bias，未融合 W8A8 的现实下界）。CUDA Graph 捕获 24 次调用，权重多份轮换避开 40 MB L2，5 次 warmup 后取 25 次 replay 中位数。`_int_mm` 要求 M>16、K/N 为 8 的倍数，不满足时补齐并标记 |
| `bf16_eager` | released BF16 pipeline，复现历史 1.450 s 口径 |
| `q0_c_buffer` | 现有推荐 engine，安装方式与标准 LIBERO 评测（26020749）完全相同：复用 decoder 选择、startup anchor 逐位一致检查；复现 0.887 s 口径 |
| `bf16_cache_graph` | 原 BF16 权重不量化，装上 engine 的 condition cache 和三个 CUDA Graph 区域（video_step / video_prefill / action_step）。这是路线 3 应该比较的“同等优化 BF16” |

三个 runtime 模式共用同一个测量函数：held-out cases 3000–3039 × seeds 0/1 = 80 个 query，case 3000 上 3 次 warmup，计时边界是从 prepared GPU datum 到 `infer_action` 返回并同步（runner 的 QueryMeter），不开 profiler。数值设置与历史 profile 相同（deterministic、TF32 关、`CUBLAS_WORKSPACE_CONFIG=:4096:8`）。GEMM 微基准关闭 deterministic 和 `fill_uninitialized_memory`，避免每个输出分配都多一次 NaN 填充。

## 输出

- `results/summary.json`、`results/RESULTS.zh.md`：汇总。
- RESULTS 里的表：逐形状 GEMM 时间与加速比；按 cache 后调用次数加权的每次 policy call 纯 GEMM 时间（总计、按 M、按 stream）；三个 runtime 的 p50/mean/p95、去噪时间、peak allocated/reserved；输出一致性。
- 一致性检查（只报告，不作为失败条件）：`bf16_cache_graph` vs 本作业 `bf16_eager`；`bf16_eager` vs 历史 25943391；`q0_c_buffer` vs 历史 25960283。期望全部为 0。

## 怎么读

- **GEMM 表**里 M=294 的 video 形状看 INT8 是否接近 2×（compute-bound），M≈32 的 action 形状看 INT8 是否因读一半字节而变快。INT8 纯 GEMM 是融合 kernel 的上限，“含外围”是不融合时的下限。
- **每次 call 的 GEMM 总时间**乘上 INT8 加速比，就是路线 3 的融合 kernel 在 GEMM 部分最多能省多少；再对照 `bf16_cache_graph` 的整次 call 时间，看 GEMM 占多大比例。
- **`bf16_cache_graph` vs `q0_c_buffer`** 给出当前 engine 的 decode 真实开销；路线 3 的速度结论以 `bf16_cache_graph` 为分母。

## 操作

```
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-int8-speed-20261027/control.py upload
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-int8-speed-20261027/control.py submit
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-int8-speed-20261027/control.py status
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-int8-speed-20261027/control.py pull
```

依赖：engine 源码从 `fastwam-vq-r128-c80-c-decoder-buffer-20261024-r3-long/artifacts/25960283.pbs101` 复制；Q0 engine 的安装函数从 `fastwam-libero-engine-20261027/artifacts/26020749.pbs101/worker.py` 复制为 `libero_engine_worker.py`。全部复制都在 compute allocation 内完成。
