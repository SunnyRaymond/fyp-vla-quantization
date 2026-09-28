# Student-driven real-observation data

`FREEZE.json` 冻结了80个全新 episode 的选择与 capture protocol。先运行 CPU-only metadata selector，生成 selection manifest；GPU collector 只接受这个 manifest，且必须在 PBS compute allocation 内运行。不得在 manifest 出现前启动任何环境 rollout。

Selector 仅读取 HDF5 的 `ep_len`、`ep_offset` 和被选 row 的 `episode_idx`/`step_idx`，以及 freeze 指定的小型 exclusion manifests。它按冻结的 `random.Random(20260925)` 生成64 train/16 validation tasks；输出位于 `.../real-observation-training-data/selection/<PBS_JOBID>/selection_manifest.json`。它不读 pixels、actions 或模型，也不选择替补任务。

旧 shadow diagnostic 的16个 collection 与16个 reserved IDs、ranker validation/untouched IDs、baseline closed-loop 50 IDs、prior valid-prefix 600 IDs 全部排除。任何源文件缺失、schema/count 不符、ranker validation 不属于 reserved 集、发生 ID overlap 或剩余 episode 不足80，都会 fail closed。

本地只做源码、JSON、shell 静态检查；不打开 HDF5、不运行 selector/collector，也不提交 PBS。选择和采集脚本在正式作业中各自检查 PBS compute-node allocation；GPU capture wrapper 将利用率与显存每5秒写进该作业的 `job.log`。

若 capture job `25543623.pbs101` 的 PBS exit 1 仅由最终状态打印的 `json.dumps(..., flush=True)` 参数错误触发，可在 CPU compute allocation 提交独立的 `recover_real_observation_training_data_cpu.pbs`。Recovery 只读该 job 的小型 summary、NPZ 和已冻结 selection manifest；仅当错误签名、80个 episode 身份、64/16 split、141条样本、数组形状/finite、动作误差门与 sample-to-manifest 对齐全部通过时，才新增 `collection_summary_recovered.json`。原 summary、NPZ 和 GPU job 历史保持原样；不加载模型、不读 HDF5、不重跑环境。该 PBS wrapper 目前只供静态审查，尚未提交。
