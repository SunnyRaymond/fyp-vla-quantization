# Job 24897213.pbs101 无效记录

该 job 于 2026-09-21 18:21 在 ASPIRE2A compute node 启动并结束，`Exit_status=1`。失败原因是 `run_lewm_dense_rank.py` 在准备 shared train bank 时引用未定义的 `HORIZON`，触发 `NameError`；三臂均未进入训练和评测，因此不属于实验结果。

修复内容仅将 RNG parity 补偿行改为已有基线命名空间中的 `base.HORIZON` 与 `base.ACTION_DIM`。修复后的唯一正式 handle 为 `24908446.pbs101`；本 job 不得与正式结果合并或比较。

GPU telemetry 仅证明作业运行期间持续采样到 GPU/显存，不能作为模型实验结果：训练尚未开始。
