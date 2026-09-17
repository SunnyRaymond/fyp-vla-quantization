# conditional-action-distribution CPU preparation

本目录只准备后续 offline screen 的 8 个 raw condition，不运行模型、不读取
checkpoint output、不下载或安装依赖。`conditional_prepare.sh` 是一次
UGGPU-TC1 CPU allocation：4 cores、16G、10 minutes；提交后的第一个
scheduler-dependent action 是 source `${TOP}/control/allocation_guard.sh`，随后
Python 再验证真实 job、hostname、owner、partition 和 NodeList。CPU job 必须
保持 `CUDA_VISIBLE_DEVICES` 不可见。

默认根目录为 `/tc1home/UG/yguo017/v100_newangles_ccds`，已有 Python 为
`${TOP}/smolvla/venv/bin/python`；base assets 来自 `${TOP}/smolvla/identity.json` 和
`${TOP}/smolvla/source_subset`。新资产只写入 `${TOP}/conditional_marginal`，job
审计写入 `${TOP}/artifacts/${SLURM_JOB_ID}`；它不会写入或覆盖
`padded_feedback`。目录有显式 `PREPARATION.lock` 单写者锁，已有 lock、manifest
或 NPZ 时拒绝覆盖。

## 冻结选择

选择按 `(task_index, episode_index)` 进行，与视频或模型输出无关。先读取并记录
两个排除 manifest：

1. `${TOP}/smolvla/manifest.json` 的 Flow raw manifest，要求 12 个 pair、每个
   task 3 个；
2. `${TOP}/padded_feedback/manifest.json` 的 padded manifest，要求 8 个 pair、
   每个 task 2 个。该文件应与已完成 CPU preparation 的 `artifacts/64771`
   manifest 保持一致；脚本保存实际路径、SHA-256、schema 和 pair/episode IDs。

两个排除集合必须互不重叠，合计每 task 5 个 pair。对 task 0、1、2、3，各自从
episode metadata 中排除这 5 个 pair 后，按 `episode_index` 升序取最早 2 个，
每个 frame 固定为 `floor(length/4)`。脚本先把这个选择及两个排除 manifest 的
hash/IDs 写入 job `selection.json`，之后才解析所选 data/video 资产；缺失、未列
入 base identity、size/SHA-256 不符或 video 解码失败均写 `resource_blocked`
status 并退出，不替换 episode。

## 资产与产物

只允许读取 base identity 已列出的 metadata、task-mapping parquet、data parquet
和两路 video。纯 helper 使用已审查的
`${TOP}/artifacts/64758/smolvla_cpu_prepare.py` 中 `_task_rows`、`_format_path`、
`_decode_frame`，不调用其 preparation main。每个 sample 是
`smolvla-raw-input-sample-v1`：两路真实 camera 的 uint8 CHW3、finite float32
`observation.state[8]`、task/episode/frame/timestamp、video timestamps 和
scalar `metadata_json`。GPU processor 留给后续 runner，CPU preparation 不补图像
或动作维度。

最终 manifest schema 为 `conditional-marginal-raw-input-manifest-v1`，固定 8 条
sample，包含：

- `base_identity_path` 与 `base_identity_sha256`；
- Flow 与 padded 两个 `excluded_manifests` 的实际路径、hash、schema、pairs 和
  episode IDs；
- frozen selection rule、全部排除/选中 IDs；
- 所有实际读取 source files 的 identity-expected SHA-256；
- state/camera mapping、decode contract 和 8 条 sample path/hash/metadata。

`status.json` 的 `complete` 只表示 8 个 raw NPZ 已通过 selection、identity、
source/hash/decode 检查，不表示模型推理或 conditional distribution 的科学
结果。任何资源阻断都保留 job status；不下载、不安装、不启动 GPU、不连接旧
screen，也不修改历史 manifest 或 padded preparation。
