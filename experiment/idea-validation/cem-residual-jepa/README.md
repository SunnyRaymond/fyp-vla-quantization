# CEM residual JEPA：Stage A

这是冻结的 DINO-WM predictor-level screen。训练只学习：

`teacher(mu + sigma * epsilon) future latent - teacher(mu) future latent`。

student 输入 frozen base future native latents、normalized `epsilon`、`sigma` 和
candidate-independent context summary，使用 causal temporal mask 输出 residual，最终
预测为 `base + delta`。

评估固定为 8 个 held-out contexts × 2 fresh seeds × 300 candidates；统计单位是
context×seed block，candidate 不是 replicate。三臂是 mean-only base repeat、trained
residual、shuffled-epsilon negative control。训练覆盖 initial `(mu=0,sigma=1)` 和
一次 frozen teacher-elite refined distribution (`M=64,K=8`)。

`run_stage_a.py` 复用 `jepa-action-prefix-compiler` 的 manifest 校验、trajectory
loader、native pre-encode、official DINO-WM loader、teacher rollout 和 objective helper。
`stage_a.pbs` 只允许在 PBS compute allocation 执行，含 login/head/submit、GPU、
`PBS_JOBID`/`PBS_NODEFILE` guard，并每 30 秒记录 utilization/VRAM。没有下载、安装、
SSH、hash、full CEM、encoder、environment 或 closed-loop。

本机静态检查：

```powershell
python -m py_compile run_stage_a.py test_contract.py
python test_contract.py
```

Wall fixed-pool workload 已被发现，但本版按原请求保留 PushT + 既有 JEPA manifest
contract；Wall pivot 需要另行冻结 loader/latent contract，不能把两种 backend 混入本次
结果。
