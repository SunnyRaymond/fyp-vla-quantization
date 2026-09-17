# FRT A/B：CCDS 环境与资源复用

2026-09-12。用户授权先完成 A 并记录结果，通过后执行有界 B；不执行 C/TEST，不连接 ASPIRE2A。

## 已核实

- User Guide 第 2、14、16–20 页要求所有程序在 SLURM compute allocation 内执行。第 23 页列有共享环境，但优先复用已验证的个人环境。
- 实际账号 QoS `normal`：总 CPU 20、GPU 1、内存 64G，最多 2 个运行中作业，单作业最长 6 小时。初次检查无已有作业。GPU 任务串行；CPU 核验可以在资源合规时并行。不批量提交多份 GPU 作业。
- CPU inventory job `64685` 在 `TC1N07` 检查成功，检查程序耗时 15.77 秒，未使用 GPU、未安装依赖、未下载。
- 复用 `/tc1home/UG/yguo017/cem_update_ccds/venv`、`modelroot` 与 `cache/torch`。版本：torch 2.2.0+cu121、torchvision 0.17.0+cu121、numpy 1.26.4、hydra 1.3.2。
- checkpoint 368656057 bytes；本次 compute-node SHA256：`8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b`。epoch 65 来自旧准备记录，A 将在加载时重新验证。
- source HEAD `0a9492fa12044b852ae9e001cc74604b79c8bb0c`；存在旧准备留下的 `env/__init__.py`、`models/dino.py` 修改，分别用于 Wall-only import 与固定 DINOv2 revision。A 还需验证运行语义。
- 数据目录含 actions/states/layout metadata 与 observations；旧准备记录为 192 valid trajectories。本次未重载全部数据，A 将验证所选 episodes 可读取。
- 新控制与结果放在独立 `/tc1home/UG/yguo017/frt_ccds`，旧实验数据保持不变。

## 边界

环境复用检查不等于 A 通过，也不证明 FRT 拟合可行。全部模型加载、backward、缓存生成、hash 和数组核验必须先验证真实 `SLURM_JOB_ID`、`TC1Nxx` hostname、scontrol RUNNING/UserId/NodeList；GPU 工作还需确认分配 GPU。CCDS 使用 SLURM，不设置或伪造 PBS_JOBID。

Login 只做连接、提交、状态和不超过 64 KiB 的小型控制文件读写。保留 SSH host-key verification；凭据仅本地连接器读取，不输出或上传。一个目标文件只有一个写入者；不清理其他任务、不盲目重试。

详见 `artifacts/resources_64685.json`。文件系统空闲空间不代表个人 quota。
