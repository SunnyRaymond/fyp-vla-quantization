# DINO-WM Wall CEM-Residual JEPA：Frozen Stage A

Frozen run `24464560.pbs101` 已完成，scientific decision 为 **NO-GO**；见 `RESULT.zh.md`。

这是一个 predictor-only 的最小验证入口：对每个固定 CEM pool，用 frozen Wall teacher 在 `mu_before` 做一次 exact base rollout，再以

`epsilon = (action - mu_before) / sigma_before`

和 `sigma_before` 为条件，预测每个 candidate 的 dense visual/proprio latent residual。`mean_only_base_repeat` 与 `shuffled_epsilon_negative_control` 是固定对照。

数据来自既有 `rankcal-wall-screen-workload-v1` 的 26 个真实 pools、8 个 episode。顶层 `workload.pkl` 提供 `candidates/reference_scores/obs_0/obs_g`；对应 episode 的 `pools.npz` 提供 exact `mu_before/sigma_before`。Train 是 episode `000–004`（18 pools），held-out 是 `005–007`（8 pools）；candidate 不是 replicate。

Runner 会在 GPU 上重新计算 dense teacher rollout，不运行 environment、full MPC、CEM 更新或 closed loop。计时边界是 `300 × teacher rollout` 对比 `1 × teacher base + student batch(300)`，每次计时前后使用 CUDA synchronization。`stage_a.pbs` 要求有效 `PBS_JOBID/PBS_NODEFILE`、compute-node hostname、GPU，并每 30 秒记录 `nvidia-smi` telemetry；不执行下载、安装或 hash。

## PBS 运行

在已有 Wall checkout/runtime 上设置 `ROOT`；必要时覆盖 `WALL_ROOT`、`WORKLOAD`、`POOLS_ROOT`、`PYTHON_BIN` 和 `OUT`，然后提交：

```bash
qsub -v ROOT=/path/to/existing/wall-checkout stage_a.pbs
```

结果写入 `OUT/summary.json` 与 `OUT/student_checkpoint.pt`。只依据 held-out pool 的 paired rank/top-30 结果判断 frozen mechanism gate；latency 是 diagnostic，不是 full-planner speedup claim。

## 纯静态 contract check

```bash
python test_contract.py
```

该检查只解析本目录、读取 frozen workload shape，并用 tiny tensors 检查 causal mask；不会加载 Wall checkpoint，也不会运行 SSH/PBS/model rollout。
