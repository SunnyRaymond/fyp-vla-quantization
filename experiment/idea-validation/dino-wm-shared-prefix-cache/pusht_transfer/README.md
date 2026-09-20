# DINO-WM PushT planner-call prefix-cache transfer

> **结果：FAIL。** `23986275.pbs101` 的 decision gate 未通过，full-plan paired median latency reduction 为 `4.8150% < 10%`；不进入 LeWorldModel PushT。详见 [RESULT.zh.md](RESULT.zh.md)。

这是在 Wall GD portability gate 通过后冻结的下一阶段 transfer 设计。范围只覆盖官方 DINO-WM PushT checkpoint 上的 direct `CEMPlanner.plan()` paired timing；不运行 `MPCPlanner` outer loop、不执行动作、不做 closed-loop success evaluation，也不开始 LeWorldModel。

## 官方边界

官方配置是 `experiment/reproduction/dino-wm-wall/source/conf/plan_pusht.yaml`：主线是 `MPCPlanner`，其 sub-planner 是 `CEMPlanner`。本 transfer 保留 sub-planner 的官方参数：`horizon=5`、`topk=30`、`num_samples=300`、`var_scale=1`、`opt_steps=30`、`eval_every=1`；保留 `seed=99`、`goal_source=dset`、`goal_H=5`、objective `alpha=1/base=2/mode=last`。只把官方 `n_evals=50` 收缩为冻结的两个 observation，目标仍由官方 `PlanWorkspace` 的 dset 路径生成。

官方模型身份是 `model_name=pusht`、`checkpoints/outputs/pusht/checkpoints/model_latest.pth`。官方 dataset 路径是 `data/pusht_noise/{train,val}`，每个 split 至少需要 `states.pth`、`rel_actions.pth`、`seq_lengths.pkl`、`velocities.pth` 和 `obses/episode_000.mp4`，完整数据集仍需保持官方目录结构。官方来源记录在 [DINO-WM README](../../../reproduction/dino-wm-wall/source/README.md)；OSF metadata 来源为 checkpoint bundle [xvzs4](https://osf.io/download/xvzs4/)（953,204,628 bytes）和 PushT dataset [k2d8w](https://osf.io/download/k2d8w/)（2,785,304,515 bytes）。

### Target-preparation dependency boundary

官方 `PlanWorkspace.prepare_targets()` 的 `goal_source=dset` 会采样长度为 `frameskip * goal_H + 1` 的 validation trajectory，再通过 `PushTEnv` replay action 得到初始/最终 render。该路径会导入 `pymunk`、`pygame`、`shapely` 和 `skimage`，因此 CPU-only `pusht_deps_prep.pbs` 会在 compute allocation 中用 module Python 创建 job-local bootstrap venv，再把 binary wheels `pymunk==6.8.0`、`pygame==2.5.2`、`shapely==2.0.3`、`numpy<2`、`scikit-image==0.21.0` 安装到 versioned stable PYTHONPATH overlay，并由实际 bundled runtime smoke-import `pymunk`、`pygame`、`shapely`、`cv2`、`skimage`。官方 `scikit-image==0.19.3` 没有 CPython 3.11 wheel，因此这里只在 untimed target preparation dependency 上使用 API-compatible 0.21.0，避免 source compilation。GPU job 使用该 overlay，runner 保持官方 `PlanWorkspace` target preparation；环境交互仍完全在 timed direct `CEMPlanner.plan()` 边界之外。

当前 ASPIRE2A 远端 staging 只有 `wall_single` 和 Wall data，尚未有 `outputs/pusht` 或 `pusht_noise`。资源准备改为两阶段，但 benchmark 仍只提交一个 A100 PBS job：

1. 先提交 CPU-only `pusht_assets_prep.pbs`。它在 PBS/non-login guard 后只读复用旧 job 已下载的两个 ZIP：`/scratch/users/ntu/yguo017/dino-wm-wall/artifacts/pusht-assets-23952308.pbs101/downloads/official-checkpoints.zip` 和 `official-pusht-noise.zip`。目标目录必须事先不存在，然后用系统 `unzip` 直接提取 `outputs/pusht/*` 到稳定 `$ROOT/checkpoints`、`pusht_noise/*` 到稳定 `$ROOT/data`；只测试 frozen required files，最后写小的 `$ROOT/PUSHT_ASSETS_READY` marker。
2. 提交 CPU-only `pusht_deps_prep.pbs`，用 job-local bootstrap venv 安装 overlay、再由 bundled runtime smoke-test，生成 `$ROOT/artifacts/pusht-python-overlay-pusht-v5` 和 `$ROOT/PUSHT_DEPS_READY_V5`。旧失败 job 留下的 partial directories 不会被覆盖，也不要求 login-node 清理。
3. 在两个 prep job 都成功后，以 `qsub -W depend=afterok:<assets_jobid>:<deps_jobid> pusht_transfer.pbs` 提交单个 A100 job。它只检查 stable checkpoint/data、dependency overlay 和 markers，直接解包 runtime 并运行 frozen benchmark；不下载、不解压、不安装、不调用历史 Python preparer。

如果旧 archive 缺失，asset prep 立即 fail closed，不重新下载；两个 CPU-only prep 与后续 A100 job 都不会在 login node 下载、解压、模型加载或 benchmark，不计算 checksum/hash，也不安装或编译依赖。历史 `prepare_pusht_assets.py` 保留供追溯，但新 PBS job 不调用。

## Cache boundary

`baseline` 使用官方 `wm.rollout()`，每个 CEM iteration、每个 observation 都对 candidate-repeated `obs_0` 重算 `encode_obs`。

`factorized_cache` 在一个完整 `CEMPlanner.plan()` call 内，对每个 fixed observation 只做一次 `VWorldModel.encode_obs(transformed_obs_0)`，再以 read-only expanded view 提供给每个 candidate batch。`encode_act`、predictor、`replace_actions_from_z`、objective、`argsort/topk`、`mu/sigma` update 和 random candidate generation 保持官方顺序。实现复用共享的 `cache_core.rollout_with_cache(mode="factorized_cache")`。

## Paired protocol and gate

- 两个独立 observation：`pusht_obs_00`、`pusht_obs_01`。
- 每个 observation/path：5 warmups + 10 technical paired repeats。
- 独立单位是一个 fixed observation/goal pair + one CEM RNG seed 的完整 planner call；candidate、CEM iteration 和 technical repeat 都不是独立 observation。
- 每个 timed call 前后 `torch.cuda.synchronize()`；timing 包含 preprocessing、goal encoding、cache setup、完整 CEM iterations 和 trace logging，不包含 environment interaction。
- 记录 final actions、final first actions，以及每轮每个 trajectory 的 elite/topk indices/losses、`mu`、`sigma`、first action、best loss 和 peak memory。
- path order 按 observation/repeat 交错；两条 path 使用相同 seed。

只有以下条件全部满足，才允许继续 LeWorldModel PushT：

1. decision trace bitwise exact；若不是，则 final actions、final first actions、每轮 elite/topk trace、`mu`、`sigma` 和 loss 的 max-abs 全部 `<=1e-5`；
2. technical repeats 的 paired median full `CEMPlanner.plan` latency reduction `>=10%`；
3. cached/baseline maximum peak-memory ratio `<=1.10`。

任一条件失败都停止，不调阈值，不增加 repeats，不进入 LeWorldModel。`verify_pusht_transfer.py` 是 read-only verifier；它不会修正结果或改变 gate。

## Files

- `PUSHT_TRANSFER_FREEZE.json`：冻结 source、assets、CEM settings、paired protocol 和 gate。
- `pusht_assets_prep.pbs`：CPU-only 两阶段资源准备 job；复用旧 archives、直接 `unzip` 到 stable assets 并写 marker。
- `pusht_deps_prep.pbs`：CPU-only PushT binary-wheel overlay prep 和完整 environment import smoke test。
- `prepare_pusht_assets.py`：历史 selector/preparer，保留但不由新 PBS job 调用。
- `run_pusht_transfer.py`：compute-node-only model/runtime runner。
- `verify_pusht_transfer.py`：read-only structure/decision/latency/memory verifier。
- `pusht_transfer.pbs`：single A100 PBS job definition；本阶段不提交。

本地验证边界仅为 Python syntax/`py_compile`、JSON parse 和 PBS shell syntax；不会在本机下载资产、加载模型、运行 inference 或 benchmark。
