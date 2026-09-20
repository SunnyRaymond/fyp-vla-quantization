# LeWorldModel PushT iteration cache

这是 LeWM PushT 上的最小 planner-call cache 实现。LeWM 原生已经把一个
initial observation 编码后沿 candidate axis 扩展；本实现不重复做这个
factorization，也不改变 candidate batch shape。它只把官方 `JEPA.get_cost()`
在每个 CEM iteration 中重复执行的 initial-observation encode 和 goal encode
移到一次 solve-call 的 cache build 中。

## 精确的实验边界

冻结的是官方 `config/eval/pusht.yaml` 和 `config/eval/solver/cem.yaml`：
`horizon=5`、`num_samples=300`、`n_steps=30`、`topk=30`、`batch_size=1`、
`seed=42`。每个 observation/goal 和 CEM RNG seed 组成一个 paired unit；每条
路径 5 warmups + 10 technical repeats，两个 observation，path order 交错。

full solve latency 从本 runner 的 planner-call wrapper 入口到返回结果，包含
输入 tensor clone 这一小段 preprocessing、baseline 的重复 encode 或 cache build、
30 次 CEM iteration、top-k/mean/variance 更新及 solver overhead；因此它对应
public planner/solve call 的完整边界。secondary `plan_section` 在 preprocessing
完成后开始，到 action return 结束，包含 cache setup/goal encode 和完整 CEM loop。
inner `cem_loop` 是同一次 solve 中所有同步的
`model.get_cost()` 区间之和：包含 cache build、candidate rollout 和 criterion，
但不包含 candidate sampling、top-k/update 和 solve setup。两种路径使用相同
candidate RNG、checkpoint、输入和 planner settings。

记录 preprocessing、full_planner、plan_section、cem_loop 和 cache_setup latency，
以及 final actions、first actions、每 iteration 的 top-k indices/values、cost、
mean/variance，并记录 CUDA-synchronized full solve latency、inner cost latency、
peak allocated memory。结果中的 `improvement` 定义为
`(baseline - iteration_cache) / baseline`，分别报告 full solve 和 inner cost section。

## 资产边界

runner 只接受已经存在的官方 `STABLEWM_HOME` 资产：

```text
$STABLEWM_HOME/pusht/lewm/config.json
$STABLEWM_HOME/pusht/lewm/weights.pt
$STABLEWM_HOME/pusht_expert_train.h5
```

不下载、不解压、不安装、不编译。若资产不在远端，应先在 CPU PBS allocation
中准备；A100 job 只做加载、推理和 benchmark。

## 缓存正确性边界

`IterationCacheModel` 只缓存独立的 detached `initial_embedding` 和
`goal_embedding`，不缓存 mutable `info_dict`、candidate action embedding、
predicted embedding 或 CEM distribution state。`action_encoder`、predictor、
criterion 和 CEM update 继续调用官方路径。

## 运行方式

```bash
qsub lewm_pusht_iteration.pbs
```

提交前只需设置 `LEWM_REPO_ROOT`（已 staged 的官方 source）和
`STABLEWM_HOME`（已 staged 的官方 assets）。PBS 脚本会在缺少任一资产时
fail closed，不会退回 login node 做准备。
