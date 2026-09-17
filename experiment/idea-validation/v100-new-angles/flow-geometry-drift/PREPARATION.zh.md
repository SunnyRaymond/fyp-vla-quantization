# SmolVLA CPU preparation contract

本目录的 `smolvla_cpu_prepare.sh` 只负责一个 bounded CPU preparation job。资源为 `UGGPU-TC1`、4 cores、24G、35 minutes，未申请 GPU。脚本先 `source $HOME/v100_newangles_ccds/control/allocation_guard.sh`，Python 入口随后再次调用 `allocation_guard.require_allocation()`；实际 hostname、SLURM owner、partition 和 NodeList 不满足时立即退出。依赖安装、Hub 下载、hash、metadata 读取、parquet 处理与 video decode 都发生在已验证的 compute allocation，login/head 只做 controller 操作。

远端目标为 `$HOME/v100_newangles_ccds/smolvla`，作业小产物为 `$HOME/v100_newangles_ccds/artifacts/$SLURM_JOB_ID/`（因此可由 controller 的 `artifacts/$SLURM_JOB_ID/<small-file>` 读取）。同一目标由 `PREPARATION.lock` 单写者保护；已有 `.part`、sample 或 manifest 会 fail-closed，避免并发或不完整文件被复用。下载走公开 HTTPS、逐块计算 SHA-256、先写 `.part` 再原子 rename，不使用 credential 或 HF token。venv 由已知 `$HOME/cem_update_ccds/venv/bin/python`（Python 3.10 bootstrap）创建，避免猜测系统 Python。

固定身份如下：

- checkpoint：`lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`；checkpoint 中所有 safetensors（包括官方 processor normalizer/unnormalizer tensors）必须有 Hub LFS SHA-256，并在 `identity.json` 同时记录 LFS 与下载后 SHA-256。
- base VLM：`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`；只保存其 config/tokenizer/processor 小文件，明确 `weights_downloaded=false`，不下载 base weights，不导入或实例化 policy。
- dataset：`lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`，要求 metadata 明确为 `v3.*` 或 `3.*`（官方 `v3.0` 与无 `v` 写法都接受）。先只取 `meta/`，再取冻结选择实际引用的 data/video 文件。metadata 上限 200MB，选定 data/video 总上限 3GB，checkpoint、base metadata 与选定 data/video 合计上限 5GB。
- 依赖：`torch==2.6.0+cu124`、`torchvision==0.21.0+cu124`、`lerobot[smolvla]==0.4.4`、`torchcodec==0.2.1`。venv 位于 `$HOME/v100_newangles_ccds/smolvla/venv`，不触碰旧 `cemenv`。

选择规则在任何模型输出前固定：按 `task_index` 升序取前 4 个 task；每个 task 按 `episode_index` 升序取前 3 个 distinct episodes；每个 episode 取一个**zero-based** `frame_index=floor(length/4)`。总计 12 个 episode。字段缺失、task 映射不唯一、state shape 不匹配、pointer/template 不明、下载大小/hash 不符或 video 解码非 CHW 都停止，不静默 wrap、切换 task、拼造观测或按结果挑选样本。

样本契约供后续 `new_angles_brief` GPU runner 直接消费：

- `samples/sample_*.npz`：每个 episode 一个独立文件。checkpoint 与 dataset 交集中的每个**真实** camera key 作为同名 NPZ key，原始 frame 为 `uint8`、`[C,H,W]`；不强制 3 cameras，不写缺失 camera 的零图。原始 state 使用准确的 checkpoint state key（通常 `observation.state`），为 finite `float32` 一维数组；另有 `task_index`、`episode_index`、`frame_index`、`timestamp`、每 camera 的 `video_timestamp__...` 和 `metadata_json` scalar keys。
- `$ASSETDIR/manifest.json`：12 条记录，含相对ASSETDIR的sample路径、sample SHA-256/size、task/episode/frame/length、timestamp、真实 camera keys、state key、数据与视频 source paths；这是GPU读取的唯一稳定入口。`artifacts/$SLURM_JOB_ID/manifest.json`只是同内容的审计副本，sample路径仍以ASSETDIR为根，不能直接把该副本传给GPU runner。
- `identity.json`：checkpoint/base-VLM/dataset revision、每个下载文件的 size/LFS SHA-256/downloaded SHA-256、实际 feature mapping、`empty_cameras`、decode tolerance 和 selection rule。
- 后续 GPU runner 必须让 checkpoint processor 处理 normalization/tokenization；CPU 样本保持 raw input。本 preparation 不读取 policy weights 做加载，不做 inference，不产生 flow geometry scientific result。

官方 `prepare_images` 的 camera 语义由 GPU 端保留：本脚本只保存 checkpoint image features 与 dataset 真实 image/video features 的交集；缺失 key 写入 mapping，只有 checkpoint 自身配置的 `empty_cameras` 才可由官方 preprocessor mask。CPU 端绝不以重复、黑图或伪造数组补 camera。

`status.json`（作业目录和 assetdir 各一份）小于 64KB，阶段包括 `guard_verified`、`dependencies`、`model_downloaded`、`selection`、`subset_downloaded`、`samples_written`、`complete/failed`。`run.log` 仅供 compute-side diagnosis；没有模型输出或成功率含义。

本地 `jobs.json` 保留早期提交记录；后续root接管后的修复与job IDs以 `PREPARATION_FAILURE_AUDIT.zh.md`、本目录artifacts和campaign scheduler记录为准。不能将旧jobs.json的状态当作当前状态。

2026-09-13 root修复：视频使用LeRobot v0.4.4明确支持的pyav backend，同一timestamp nearest-frame与tolerance验证；返回float01后round恢复uint8。TorchCodec的libpython加载失败记录在64757，未改变视频或采样。identity记录实际av版本和decode conversion。
