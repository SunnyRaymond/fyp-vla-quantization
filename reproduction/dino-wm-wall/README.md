# DINO-WM：官方 Wall checkpoint 评估

范围：只做 checkpoint evaluation，不重新训练，不量化。

**已按用户要求在第7轮后停止：47/50=94%，与论文96%相差2个百分点。** 详情见 [RESULTS.md](RESULTS.md)。结果来自完整一轮的50-case日志核对；提前终止未执行完整轨迹和视频导出，不代表原12轮正常完成或完整产物验证通过。

## 来源

- Paper：arXiv:2411.04983v2。
- 官方仓库：https://github.com/gaoyuezhou/dino_wm
- 固定 source commit：`0a9492fa12044b852ae9e001cc74604b79c8bb0c`。
- 固定 DINOv2 source commit：`7764ea0f912e53c92e82eb78a2a1631e92725fc8`（本次上游 main；原始训练使用的 DINOv2 revision 未知）。
- 官方 checkpoint bundle：https://osf.io/download/xvzs4/ （953,204,628 bytes）。只解压 Wall。
- Bundle 中实际模型目录为 `outputs/wall_single`，不是 README 示例的 `wall`。发布 checkpoint 的实际 epoch 为65，大小368,656,057 bytes；配套配置里的training.epochs=1000是配置上限，不等于checkpoint训练了1000 epochs。
- 官方 Wall dataset：https://osf.io/download/49rnx/ （1,668,205,895 bytes）。
- 远端目录：`/scratch/users/ntu/yguo017/dino-wm-wall`。

## 本次 protocol

采用官方 `conf/plan_wall.yaml`：seed 99，50 cases，random_state goals，goal_H=5，CEM candidates=300，topk=30，optimization steps=10，objective alpha=1，每轮执行5个 model actions。

唯一新增的任务预算是 `max_mpc=12`；checkpoint frameskip 若为5，则每个 case 最多执行300个 environment steps。这个上限取自官方 Wall environment registration 的300步，但官方自定义 rollout 并不会可靠应用 Gym TimeLimit，且论文未明确给出主表 Wall 的 MPC 总轮数。因此这是有界的官方 checkpoint 评估，不能无条件声称与主表96%使用完全相同的协议。保存各轮结果供后续比较。

成功阈值保留官方实现：最终位置距目标小于4.5。成功后的终止/动作屏蔽保留官方 MPC 行为；失败仍计入50-case分母，不换 seed，不重抽目标。

## 运行适配

- Wall-only environment registration，避免导入无关的 MuJoCo/d4rl；Wall dynamics 与 success predicate 不改。
- 官方 checkpoint 的 `hydra.yaml` 只改 dataset 路径，保留 `hydra.original.yaml`。
- 使用 checkpoint 配套参数，模型设为 evaluation mode，禁用 gradient；保留 FP32。
- 实际软件：Python3.11、PyTorch2.2.0+cu121、torchvision0.17.0、Hydra1.3.2。因下载链路缓慢，复用已有离线wheelhouse；与官方Torch2.3.0有版本差异，结果须保留此边界。
- 环境包在节点本地展开，避免共享文件系统上安装大量小文件。
- 绕过 SLURM launcher，从 PBS 调用官方 `planning_main`。
- 禁用 visualization-only decoder，避免每次 CEM diagnostic 解码；保存真实环境与目标并排的逐case视频。CEM objective不依赖decoder。
- 不上传 W&B，日志保存在作业 artifact 目录。

## 作业

1. `prepare_offline.pbs`：CPU准备独立环境，成功作业 `16170585.pbs101`。
2. `prepare_assets.pbs`：下载、检查官方资产、缓存DINOv2，成功作业 `16170788.pbs101`。
3. `smoke.pbs`：2 cases、1 MPC round。GPU作业 `16170793.pbs101`完成planning，但最后JSON序列化失败；已修复runner，并通过CPU作业 `16170840.pbs101`从原有输出恢复汇总、完成独立核验，没有重跑GPU。短测1/2成功；CEM约25.98秒，CEM阶段峰值显存3.994 GiB。
4. `full.pbs`：50-case评估，作业 `16170841.pbs101`，单张A100 / 110 GB / 4h上限。第7轮结束仍为94%，按用户指令取消；实际walltime1小时16分16秒，PBS F / Exit_status143。保留日志与配置，未重跑。

早期在线依赖准备作业已停止，其缓存与日志保留；这些准备尝试不计入正式评估的GPU运行时间。

## 产物与核验

计划完整产物包括 `evaluation_config.yaml`、`model_metadata.json`、`plan_targets.pkl`、`logs.json`、`cases.json`、`trajectories.npz`、每个case的MP4、`summary.json`。本次提前停止，后四类最终导出未执行；实际保存清单和日志汇总见RESULTS.md。

完成必须满足：PBS正常退出、50个唯一case、逐case距离与官方success判定一致、summary与cases计数一致、动作预算不超限。只有具备完整逐case结果的作业才报告正式成功率。
