# Wall-only broadcast-coupling staged validation

`exp1_pre_gate.pbs` 在 ASPIRE2A 单张 A100 PBS allocation 内运行 fresh 6-state multi-step pre-gate。login node 只允许上传这些小型 control files、`qsub` 和轻量 `qstat/tail`。

本目录不修改历史 `ideaspark_run/v100-new-angles/broadcast-coupling/`。结果写到远端：

`/scratch/users/ntu/yguo017/dino-wm-wall/broadcast_coupling_wall/artifacts/<PBS_JOBID>/`

`verification.json` 是是否继续扩展的唯一 gate。若为 `mechanism_no_go_stop`，不运行实验 2/3；若为 `inconclusive_binding`，先修工程问题且不得改变样本、seed 或 threshold。
