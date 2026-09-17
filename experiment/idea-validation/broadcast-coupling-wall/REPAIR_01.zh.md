# Repair 01：移除结构上不可用的 H=10

ASPIRE2A job `22697093.pbs101` 在 allocation、A100、checkpoint 和 dataset load 后 fail-closed：`episode 130 is too short for H=10, frameskip=5`。`per_state_records.jsonl` 为 0 bytes，未产生或读取 scientific values；实际 walltime `00:00:54`，Exit 64。

修正仅将 horizons 从 `[1,5,10]` 改为官方 Wall action/planner 范围内的 `[1,5]`。fresh episodes `130..135`、所有 arms、quantizer、seeds、H=5 primary gate 和停止规则不变。不补采样、不换样本、不调 threshold。若第二次仍出现 engineering failure，停止并报告，不继续消耗 GPU。
