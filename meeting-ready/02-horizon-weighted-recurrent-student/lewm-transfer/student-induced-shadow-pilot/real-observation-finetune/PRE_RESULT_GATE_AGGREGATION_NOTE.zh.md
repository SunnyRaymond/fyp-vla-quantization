# Predictor gate 聚合核对（结果出现前）

在 `25543873.pbs101` 已提交且尚未收到训练/validation 结果时，复核发现已提交 runner 对每个 episode 的 t0/t25 window 先分别算 `1−terminal_MSE/initial_MSE`，然后平均这些 improvement；而冻结协议要求先分别平均同一 episode 可用窗口的 initial 与 terminal relative MSE，再算 `1−mean(terminal_MSE)/mean(initial_MSE)`。两者在有两个窗口且初始 MSE 不同时不相等。

作业继续完成固定训练，不在运行中改代码或重训。runner 的 `result.json` 保留原始每个 window 的 initial/terminal relative MSE；收到结果后，按冻结定义从这些已记录数值只读重算 episode-level gate，并将原作业判决与校正判决并列报告。若两者不一致，以冻结定义的重算结果为 predictor gate，且不在同一 validation split 上调参或重训。
