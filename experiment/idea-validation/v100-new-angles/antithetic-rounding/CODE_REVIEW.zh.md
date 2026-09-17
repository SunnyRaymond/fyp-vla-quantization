# Ensemble screen code review

审查范围：`ensemble_screen.py`、`gpu_screen.sh`，并核对关联的 raw-score verifier；只做静态审查，没有提交或运行作业。

## 结论

**BLOCKED / revise before execution**。固定 pool、score averaging 和 quantizer 主流程基本可执行，但当前快照仍有验收语义与 allocation 安全阻断。

## 已确认

- `(300, H=5, action_dim=10)` 展平为 50；每个成员输出 300 个 score，ensemble 在成员轴 `mean(axis=0)` 后按稳定 top30 选择，FP32 regret 与 elite-mean action MSE 的定义正确。
- SR 使用同一 FP32 per-output-channel grid；independent 用独立 seed，antithetic 用同一 midpoint grid 的 `1-u`。`_pair_error_metrics` 的 `E[ab]-E[a]E[b]` 是正确的 pooled empirical covariance；它不是按 scale 归一化的 scalar 理论值，应按诊断量记录。
- `ROOT=${HOME}/cem_update_ccds/modelroot`、`PY=${OLD}/venv/bin/python`、`TORCH_HOME=${OLD}/cache/torch` 和 `root/source` 路径一致，未见下载入口。

## 提交前阻断

1. `gpu_screen.sh` 仍 source/copy 旧 guard；其 host regex 与集合比较大小写敏感。已知真实 hostname 为 `tc1n03`、SLURM NodeList 为 `TC1N03`，合法 allocation 可能在模型前被拒绝。须在保留 job/ownership/NodeList 检查下统一 `casefold`，不能伪造变量。
2. `ensemble_screen.py` 的 `overall` 只检查 aggregate 5%、action MSE 和工程完成；缺少冻结的 `>=4/6 episodes`、`>=2/3 seeds`、primary 不劣于 RTN。verifier 虽计算这些，GPU `summary.json` 仍可能先写出 `preliminary_go`，必须只有一个权威 gate/decision。
3. W8 只写成 cost context；`verify_ensemble.py` 仅输出 `w8_numerically_dominates`，没有按 protocol 形成 `practical_no_go`。应使最终 decision 与 W8 规则一致。
4. fresh manifest 只保证 `96..101` 及六者内部 fingerprint 唯一，没有扫描已有 registry 的 fingerprint；需在 execution gate 中明确 cross-registry overlap 检查。
5. `num_hist` 仅事后写入 `model_structure`，未冻结/验证（当前 checkpoint config 为 1）；manifest 也未记录 `num_pred/frameskip`。source helper/cache 仅记录运行后路径/hash，未强制 expected helper identity。至少在 preflight fail closed 并记录这些字段。

## 限制

24-bit midpoint grid 与 protocol 的连续 `Uniform[0,1)` 不完全相同；应在 protocol/报告中明确这是 finite grid，scalar covariance 结论只作诊断。无 native W4、latency、memory、environment success 或 full-CEM claim。
