# EBMF-CEM Stage A

这是 `Elite-Band Multi-Fidelity CEM` 的首轮 calibration/feasibility runner。
本目录只实现冻结的 Stage A，不运行 held-out Stage B、native system timing 或
closed-loop。

## 固定设置

- `cheap_L4`：现有六层 `ViTPredictor` 的 positional embedding、dropout、前四个
  Transformer blocks 和现有 final `LayerNorm`。
- DINO-WM `wall_single` checkpoint；`K=300`、`H=5`、`topk=30`、CEM `10`
  rounds。
- Calibration observations 固定为 `wall_case_00` 至 `wall_case_05`。
- Reference path 是 `factorized_full`：每一轮先编码一个 observation prefix，再
  expand 到 300 candidates；cheap 与 full 都从同一 prefix 独立构造自己的
  autoregressive rollout。full 不复用 cheap 的后续 activation。
- 每轮只用 `J_full` 的 stable top-30 更新 `mu/sigma`；`J_tilde` 只用于误差和
  interval 记录。

## 文件与结果

- `stage_a_runner.py`：仅在 PBS compute node 上加载 model/data 并产生结果。
- `stage_a.pbs`：单 A100、1 小时 PBS allocation，包含 `PBS_JOBID` 和 hostname guard；不
  在 login node 运行模型、benchmark、下载、安装或 hash。
- `calibration.jsonl`：每行一个 observation，含十个 chained CEM rounds。每轮保存
  `J_tilde`、`J_full`、`abs_error`、cheap/full top-30、full cutoff、输入/输出
  `mu/sigma` 与 first action。Stage A 完成后追加 `epsilon_round`、`L`、`U`、
  `tau`、`A`、ambiguity ratio 和 `full_top30_in_A`。
- `stage_a_summary.json`：冻结设置、runtime identity、每轮 epsilon、ambiguity
  summary、coverage summary 与 `NOT_RUN` 边界，供独立 verifier 使用。
- `verifier.json`：read-only verifier 的 gate 结果；只有 verifier exit 0 时才创建
  `EBMF_CEM_STAGE_A_PASS`。

## 运行

在 ASPIRE2A 上从 login node 只提交作业，不直接运行 runner：

```bash
qsub stage_a.pbs
```

可通过环境变量指定已准备好的 compute-node runtime 路径：
`DINO_WM_ROOT`、`DINO_PREFIX_ROOT`、`DINO_EBMF_ROOT` 和 `DINO_EBMF_OUT`。

本地允许的检查只包括 Python syntax、PBS shell syntax 和 JSON 静态解析；不要在
本机或 login node 加载 checkpoint、运行 model inference 或 benchmark。

## Evidence boundary

Stage A 只回答 calibration completeness、empirical epsilon provenance、calibration
ambiguity 和 calibration 内的 containment。即使 calibration gate 通过，也不等价
于 held-out decision equivalence、planner latency gain、memory gain 或 closed-loop
success；这些必须由后续冻结 stage 单独授权。

实验设计记录参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M.
(2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
