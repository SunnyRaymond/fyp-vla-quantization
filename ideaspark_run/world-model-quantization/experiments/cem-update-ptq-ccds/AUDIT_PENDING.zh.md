# CCDS 独立核验修复待 CPU allocation 检查

2026-09-12。已在 `verify_stage_b.py` 增加两项完整性检查：

- CAL/DEV 每个 pool 的 search `reference.npz` 与 workload `reference_scores` 使用 `rtol=0`、`atol=1e-6` 回链比较。
- 各 method 的 `score2_ref` 按 pool 做跨 alias exact array consistency 检查。

`verify_results.sh` 现在会在完整核验前，于真实 SLURM CPU allocation 执行 verifier `--self-test`。self-test 覆盖 reference tolerance 边界与跨 alias mismatch；本机未运行计算/test，研究门槛与 budget interpretation 未改变。待 CPU allocation 完成 self-test 和完整核验后，再解释 Stage B gates。
# 已完成CPU核验

2026-09-12：修复版本已上传，job64676在真实CPU allocation完成self-tests及完整核验，工程通过。以下待测记录为历史状态；最终见RESULTS.zh.md。
