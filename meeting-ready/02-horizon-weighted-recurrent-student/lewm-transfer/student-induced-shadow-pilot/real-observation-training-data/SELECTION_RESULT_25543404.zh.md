# 新 student-driven 真实观测采集：选样结果

CPU-only PBS `25543404.pbs101` 已完成，PBS `Exit_status=0`，manifest 状态 `COMPLETE_METADATA_ONLY_SELECTION`。按事先冻结的 seed 和排除规则选出 80 个不同 episode：64 个 collection-train、16 个 collection-validation；与全部排除任务交叠 0，两个 split 交叠 0。此次只读取 HDF5 小型索引元数据，没有读取 pixels/actions、载入模型或运行 simulator。

此结果只冻结下一项真实 poststep latent 数据采集的任务身份，不是预测质量、CEM 排序或闭环成功结果。采集需消费这个精确 manifest，不能补抽或重跑原先 16 个 diagnostic tasks。

证据：[selection manifest](results/25543404.pbs101/selection_manifest.json)、[job log 尾部](results/25543404.pbs101/job.log.tail.txt)、[采集 freeze](FREEZE.json)。
