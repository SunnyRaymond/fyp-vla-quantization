# padded-coordinate-feedback CPU preparation

本目录新增的 padded_prepare.sh 只做一个 bounded CPU job，资源固定为
UGGPU-TC1、4 cores、16G、10 minutes，不申请 GPU。提交后的第一项 scheduler
操作是 source $TOP/control/allocation_guard.sh；其后 Python 再调用
allocation_guard.require_allocation()，必须同时通过真实 hostname、SLURM
owner、partition、JobState 和 NodeList 校验。本文档和脚本当前只写入本地工作区，
未上传、未提交、未执行。

远端固定根目录为 /tc1home/UG/yguo017/v100_newangles_ccds。脚本只复用
$TOP/smolvla/identity.json、$TOP/smolvla/source_subset 和已存在的
$TOP/artifacts/64758/smolvla_cpu_prepare.py 中的 _task_rows、
_format_path、_decode_frame helper；不会调用旧脚本的 main 或 download。
不下载、不安装、不创建新 environment、不读取 model weights，不看任何模型输出。
Python 使用已有 $TOP/smolvla/venv/bin/python；缺失即失败。

固定选择严格为：task_index 0、1、2、3；读取旧 Flow manifest 的 12 个
task/episode pairs，每个 task 排除旧 manifest 的 3 个 episode，再按
episode_index 升序取最早 2 个未用 episode；每个 frame 为 zero-based
floor(length/4)。选择在任何 data/video 可得性检查前冻结，不能因文件缺失换样本。
旧 manifest、task mapping、episode metadata 的映射必须唯一；没有足够候选立即失败。

只允许读取 base identity 已列出的 metadata、task-mapping parquet、data parquet
和 video 文件。所选 data/video 若不在 identity，或本地不存在、size/SHA-256 与
identity 不符，写小型 status.json，phase=resource_blocked 后退出，不挑替代。
实际读取的 source files 会在 manifest 的 source_file_sha256 中记录。base identity
不复制，manifest 记录绝对 base_identity_path 和实际 SHA-256。

产物目录为 $TOP/padded_feedback，job 小产物为 $TOP/artifacts/$SLURM_JOB_ID。
stable manifest schema 为 padded-coordinate-raw-input-manifest-v1，固定 8 条
sample record。每个 NPZ 使用旧 smolvla-raw-input-sample-v1 schema：
两路真实 camera key 的 uint8 CHW3、准确 observation.state 的 finite
float32[8]、task/episode/frame/timestamp、video timestamps 和 scalar
metadata_json；不补第三 camera，不对 state 或图像做 GPU processor 处理。
manifest 同时保留 sample path、size/SHA-256、task text、数据/视频 source mapping，
供后续 GPU runner 将 raw NPZ 交给 checkpoint processor。

PREPARATION.lock 使用显式 job ID、owner、hostname、partition 的单写者记录；
已有 lock、manifest 或 sample 时拒绝覆盖。status 始终小于 64 KiB，complete 仅表示
8 个 raw sample 已准备并通过 source/hash/decode checks，不表示模型推理或科学结果。
