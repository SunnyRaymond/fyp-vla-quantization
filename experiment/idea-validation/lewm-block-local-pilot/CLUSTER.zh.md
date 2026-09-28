# 集群定位与控制

记录日期：2026-09-26。只通过现有 host-key-verified SSH helper 做了轻量 `hostname`、PBS 查询和指定路径的 `test -f/-x`、`stat`；没有读取 checkpoint、HDF5、prepared tensor 或其他大文件内容，没有提交作业。

## 已确认的远端路径

| 用途 | 路径 / 结果 |
|---|---|
| 新实验 root | `/scratch/users/ntu/yguo017/lewm-block-local-pilot`；当前不存在 |
| LeWM source | `/scratch/users/ntu/yguo017/lewm-pusht-iteration/le-wm`；目录存在 |
| StableWM source | `/scratch/users/ntu/yguo017/lewm-pusht-iteration/stable-worldmodel`；目录存在 |
| Python runtime | `/scratch/users/ntu/yguo017/lewm-pusht-iteration/venv/bin/python`；可执行，`Python 3.11.13` |
| `STABLEWM_HOME` | `/scratch/users/ntu/yguo017/lewm-pusht-iteration/stablewm_home`；目录存在 |
| LeWM teacher | `.../stablewm_home/pusht/lewm_object.ckpt`；存在，72,345,781 B |
| PushT HDF5 | `.../stablewm_home/pusht_expert_train.h5`；存在，46,300,921,856 B |
| Phase 2 manifest | `.../state-coverage/artifacts/24926383.pbs101/context_manifest_512.json`；存在，146,156 B |
| Phase 2 prepared rows | `.../state-coverage/artifacts/24926383.pbs101/prepared_512/prepared_rows.pt`；存在，154,012,459 B |
| Reconstructed balanced rows | `.../teacher-screening/artifacts/25213164.pbs101/prepared_balanced_rows_reconstructed.pt`；存在，134,704,389 B |
| Reference checkpoint | `.../teacher-screening/artifacts/25213164.pbs101/balanced_base_step3000_reconstructed.pt`；存在 |
| LeWM interface probe | `.../interface-probe/artifacts/24554356.pbs101/interface_probe.json`；存在 |

上表省略号均展开在 `/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/` 下。三份已用过的 episode/task exclusion manifests 也都存在：seeded32 与 real-observation80 在 `lewm-pusht-iteration/artifacts/student-induced-shadow-pilot/` 下，closed-loop50 在 `lewm-transfer/official-lewm-dataset-teacher-baseline/artifacts/25534994.pbs101/selected_tasks.json`。

现有远端官方资产布局使用 `pusht/lewm_object.ckpt`；`pusht/lewm/config.json` 与 `pusht/lewm/weights.pt` 不存在。因此本实验按已冻结的 `.ckpt` loader 路径工作，不触发下载或资产准备。数据集有 46.3 GB；只允许 PBS compute job 按冻结协议读取。

参考函数文件存在：`lewm-transfer/run_lewm_recurrent_student.py` 与 `lewm-transfer/terminal-response-loss/runner.py`。PBS 会把 `le-wm`、`stable-worldmodel`、整个 `lewm-transfer` 及实验 `source` 放进 `PYTHONPATH`，由 runner 按冻结路径加载旧 evaluator/preprocessing 函数。

按要求从远端只读取回 10,080 B 的 `stable-worldmodel/stable_worldmodel/solver/cem.py` 到本地 `source_reference/cem.py`，未执行该文件。远端实现用 `torch.topk(costs, k=self.topk, dim=1, largest=False)` 更新 elites；未显式设置 `sorted`。方差更新为 `topk_candidates.std(dim=1)`，沿用 PyTorch 默认 `correction=1`。冻结 solver 更新应遵循原生 topk；如果 pairedTrace 另存 argsort 排序，应确认它不替代 solver 的 elite indices。两个 source staging 目录均没有 `.git` 元数据；远端 `git -C ... rev-parse HEAD` 失败，因此无法确认这些目录当前的 commit id。

## PBS 与 guard

首次定位时 `qstat -u yguo017` 为空，新实验 root 也不存在，所以未发现本 pilot 已有作业或重复提交。登录节点仅用于连接、状态查询和小型控制文件 staging。

查询到 `normal` 是路由队列；`g1` 已启用，可申请单 GPU，walltime 范围为 02:00:01–24:00:00。`gdev` 最长 2 小时，`glong` 最短 24 小时，均不适合此处的 4 小时任务。直接指定 `g1` 曾被拒绝（`Access to queue is denied`），所以恢复后 `run_pilot.pbs` 使用已批准的 `normal` 路由、`select=1:ncpus=16:mem=110gb:ngpus=1`、`walltime=04:00:00`，沿用高内存 LeWM evaluator 模板，并把 30 秒 GPU utilization/显存记录追加到 `job.log`。

PBS 在创建输出或读数据前验证 `PBS_JOBID`、真实 `PBS_NODEFILE`、当前 host 非 login/head/submit 且列于 nodefile，并要求 `CUDA_VISIBLE_DEVICES`。每个 job 先独占检查 `runs/$PBS_JOBID`，随后在该 compute allocation 中把小型 `.py/.json/.md/.pbs` source 复制到 `source_snapshot`；runner、freeze 和 `PYTHONPATH` 都指向这份快照，避免运行中源文件变化。机制检查只由 runner 在 PBS allocation 内做，登录节点不执行 Python imports、模型加载或数值检查。PBS trap 会记录 preflight 失败和 runner 的真实 exit code；runner 正常返回 0 后还必须留下非空 `DONE.json` 才记 job 成功。DONE 内容由 runner 完整写入，PBS 不覆盖它。

## staging / 控制

待 `STAGE.json` 的精确源文件清单和所有本地源文件就绪后，可运行：

```powershell
python experiment/idea-validation/lewm-block-local-pilot/cluster_control.py recover-g1-denied
python experiment/idea-validation/lewm-block-local-pilot/cluster_control.py stage
python experiment/idea-validation/lewm-block-local-pilot/cluster_control.py submit
python experiment/idea-validation/lewm-block-local-pilot/cluster_control.py status
python experiment/idea-validation/lewm-block-local-pilot/cluster_control.py progress
python experiment/idea-validation/lewm-block-local-pilot/cluster_control.py fetch
```

控制器复用 `block-local-action-dynamics/cluster_control.py` 的安全连接与已固定 host-key 校验，只传输 manifest 中的 `.py/.json/.md/.pbs`，总量不得超过 200 KB；每个目标先检查不存在，再以 `.uploading` 文件和原子 rename 写入。不会 staging HDF5、checkpoint、prepared tensor 或模型资产。

2026-09-26 已完成一次受限 staging：当时 STAGE.json 列出的 8 个控制/源码文件合计 114,867 B。唯一一次直接 `g1` 的 `qsub run_pilot.pbs` 被调度器拒绝，输出 `Access to queue is denied`；之后 status 查询没有发现 `lewm_block_pilot` 作业，远端 `submit.once` 锁存在。失败 attempt 保存在本地 `JOB.json`；明确确认无作业后，恢复动作会保留 `JOB_g1_rejected.json` 和远端旧 source/lock，再 staging 最终冻结版本并通过 `normal` 路由只重提一次。

`recover-g1-denied` 只接受无 job id 的 g1 拒绝记录；它重新查询 `qstat -u yguo017` 并确认看不到 `lewm_block_pilot` 后，保留旧 tracker 与远端 source/lock，再将本地状态 rearm 一次。新 staging 使用空的 `/source`，随后 `submit` 写新一次性锁并只调用一次 normal-route `qsub run_pilot.pbs`。若新响应不确定，只允许查询 `status`。
