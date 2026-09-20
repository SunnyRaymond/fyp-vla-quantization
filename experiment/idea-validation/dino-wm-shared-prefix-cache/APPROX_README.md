# Decision-Safe Approximate Factorized Cache：bounded joint screen

这是 exact Stage-A 之后的 bounded approximate screen。它只新增 `decision.jsonl` 和 `system_summary.json` 两个 gate 输入，不修改旧 exact Stage-A 文件，也不把 approximate 结果写入 exact gate。

## 固定实验

- 官方 `wall_single` checkpoint、Wall validation data/environment 和既有 `runtime-complete.tar`；
- 10 个 independent observations；每个 observation 使用 `K=300`、`H=5`、`topk=30`、`opt_steps=10` 的 chained CEM；
- paths：`baseline` 与 `factorized_cache`；
- 每个 path/observation 在 planner call 前重置同一个 `noise_schedule_seed`；官方 `torch.randn` 实际使用 CPU RNG stream，因此该字段表示完整 planner-call noise schedule，而不是单独的 CUDA draw；后续 candidate distribution 由该 path 自己的 `mu/sigma` 递归产生；
- factorized cache 的生命周期严格限定为一个 planner call/observation：`encode_obs` 对 batch=1 执行一次，再 broadcast 到每轮的 300 candidates；action encoder、predictor、objective 和 rollout suffix 保持 `cache_core.py` 的语义；
- decision runner 使用继承自官方 `CEMPlanner` 的 instrumented `plan()`，逐行复制官方 chained-CEM 语义并记录每轮 top-k、更新后的 `mu/sigma`；
- latency 使用同一个完整 `plan()` wrapper，从 preprocessing 到返回 action 计时，CUDA 前后同步。

## 输出

`decision.jsonl` 恰好每个 observation 一行；每行的 `rounds` 包含 10 个 CEM iterations，包含：

- baseline/factorized 的 top-k candidate index 集合；
- 两条 path 的 `input_mu`/`input_sigma` 与更新后 `mu`/`sigma`；
- top-k overlap、`mu`/`sigma`/first-action 的 max-absolute diff 与 L2 diff；
- observation-level 的最终 returned first action max-absolute/L2 diff。

`system_summary.json` 只使用前 2 个 observations 做 latency：每条 path 5 次 warm-up 和 10 次 technical repeats，采用 `path_order_policy=interleaved_seeded` 及固定 scalar `path_order_seed`；同一 observation 的所有 repeats 使用同一个 `planner_call_id`，记录 `latency_ms`、`peak_memory_mib`、`noise_schedule_seed` 和 pairing。

system gate 的 timing boundary 固定为：`CEMPlanner.plan` entry through returned first action；包含 preprocessing，不包含 environment interaction。

## PBS 与 gate

远端资产必须已存在：

```text
/scratch/users/ntu/yguo017/dino-wm-wall/ASSETS_READY
/scratch/users/ntu/yguo017/dino-wm-wall/runtime-complete.tar
/scratch/users/ntu/yguo017/dino-wm-wall/data/
/scratch/users/ntu/yguo017/dino-wm-wall/checkpoints/outputs/wall_single/
```

并将 `run_approx_screen.py`、`approx_screen.pbs`、`APPROX_README.md` 以及主线程提供的 `cache_core.py`、`verify_approx_screen.py` 放到：

```text
/scratch/users/ntu/yguo017/dino-wm-wall/idea-validation/dino-wm-shared-prefix-cache/
```

PBS 在任何模型加载、runtime 解包和重 I/O 前检查 `PBS_JOBID`、非 login hostname、官方资产、`cache_core.py` 和 approximate verifier；不下载、不安装、不编译、不 hash，也不自动提交 `qsub`。预计 walltime 为 60–90 分钟，申请上限 90 分钟、单张 A100。

手动提交：

```bash
qsub /scratch/users/ntu/yguo017/dino-wm-wall/idea-validation/dino-wm-shared-prefix-cache/approx_screen.pbs
```

作业结束后才调用：

```bash
python verify_approx_screen.py \
  --decision decision.jsonl \
  --system system_summary.json
```

只有 approximate verifier exit 0 才创建 `APPROX_SCREEN_PASS`。本 screen 不执行 closed-loop；任何后续 closed-loop 结论都必须在本联合 screen 通过后另行授权和运行。
