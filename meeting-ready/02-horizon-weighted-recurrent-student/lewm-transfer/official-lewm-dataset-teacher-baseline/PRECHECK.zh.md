# Preflight 记录

- 远端 staging roots 存在：`le-wm`、`stable-worldmodel`、同一 staging 下的 `venv`。运行日志确认 `stable_pretraining` 安装在该 venv 的 `site-packages`，不是 `le-wm` 源码目录；revision 2 检查 `Path(sys.prefix).resolve()` 精确等于 staging venv，并检查该 package 位于 venv 内。`stable_worldmodel` 仍必须位于 staged source root。
- 既有 teacher 文件存在：`stablewm_home/pusht/lewm_object.ckpt`，大小 72,345,781 bytes；HDF5 数据集存在：`stablewm_home/pusht_expert_train.h5`，大小 46,300,921,856 bytes。这里只核路径与文件 metadata，没有在 login node 打开、读取或复制模型/HDF5。
- 本地已有的 `load_official_checkpoint` 在 Stage 1/2 使用过，加载同一官方对象格式；新 runner 显式记录它是唯一 loader 例外。不得使用 HF cache 或下载。
- pinned `eval.py`、PushT YAML 与 CEM YAML 的轻量源码已核对；新 runner 保留官方 sampling 的 `len(valid_indices)-1` 语义、三组 scaler、ImageNet/224 transform、solver 配置、state/goal callables 与 50-step budget。100 是 world episode hard cap，50 是本评测有效 rollout budget。
- runner 在 PBS compute guard 后显式导入 `hdf5plugin`，注册 compressed HDF5 需要的 filter，再创建数据集对象。
- `video=None` 是唯一额外 evaluator output 差异；任务和 metrics 都保存。数据/teacher 只在 compute allocation 内由 job-private cache symlink 访问。
- 样本量固定为 50 个 upstream seed-42 选中 rows；目标门槛为至少 5 个 success 才具备后续同任务 paired 对照的工程依据。没有统计显著性声明或自动扩样。
- 原始 `25531848.pbs101` 在 import-root guard 失败；无 selected-task 或 summary，因此没有评估 outcome。保留该作业的小型日志与状态文件及 revision-1 freeze 快照。
- revision 2 是该 guard 的最小启动修正。静态核验通过：所有 JSON 可解析、runner AST/`--help` 通过、PBS `bash -n` 通过；与 revision 1 比较后确认除 guard/provenance说明与修订记录外，科学设置完全相同。按既有授权仅重提一次 teacher-only baseline；不训练、不提交四臂。
- 目标队列预检：`normal` 路由目标含 `gdev`；`gdev` 标记 `from_route_only=True`，并允许 1 GPU、2 小时 walltime。脚本经 `normal` route 提交，资源请求匹配 gdev。
