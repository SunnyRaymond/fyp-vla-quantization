# CCDS-TC1 inventory

查询时间：2026-09-13（Asia/Singapore）。本记录只保留 cluster 状态、路径和 job ID；未读取或写出任何 credential 内容。

## 访问与边界

- 已审查 `nscc-access/ccds_control.py`：`inventory` 仅发出轻量 scheduler/目录命令；Paramiko 使用本机 `known_hosts` 和 `RejectPolicy`，不接受未知 host key。
- 直接运行 `python nscc-access/ccds_control.py inventory` 在 `client.connect` 阶段 `TimeoutError: timed out`；未建立 inventory 会话。`10.96.189.11:22` TCP 可达。
- 随后使用已配置的 `CCDS-TC1` SSH alias 完成同范围轻控制查询；alias 保留 `StrictHostKeyChecking=true`、`BatchMode=yes` 和专用 Ed25519 key。没有使用或显示 password。
- head/login 上只执行 `hostname`、`pwd`、`squeue`、`sinfo`、`sacct` 和单层 `ls`。没有在 head/login 上运行 Python、Torch、模型、下载、安装或重 I/O。

## 当前 scheduler 状态

`squeue -u yguo017` 在 probe 前后均为空：无运行中、等待中或其他 active job。

`sinfo -h -o "%P %a %l %D %t %G"`：

```text
UGGPU-TC1* up infinite 2 mix docker:1,gpu:3
UGGPU-TC1* up infinite 5 idle docker:1,gpu:3
```

最近 `sacct -X -u yguo017 -S 2026-09-12T00:00:00` 记录的 job IDs（状态均为已结束，除标注 FAILED 外）：

```text
64662 COMPLETED   64664 COMPLETED   64665 FAILED      64666 COMPLETED
64667 COMPLETED   64668 COMPLETED   64670 COMPLETED   64676 COMPLETED
64685 COMPLETED   64687 COMPLETED   64688 COMPLETED   64689 COMPLETED
64690 FAILED      64691 FAILED      64693 COMPLETED   64694 COMPLETED
64696 COMPLETED   64704 COMPLETED   64705 COMPLETED   64706 COMPLETED
64707 COMPLETED   64708 COMPLETED   64728 COMPLETED   64729 COMPLETED
64730 FAILED      64731 FAILED      64732 COMPLETED
```

## GPU/Torch probe

短 probe 使用 `UGGPU-TC1`、1 GPU、2G、`--time=00:02:00`，并在 compute allocation 内检查非 login hostname、`SLURM_JOB_ID`、partition 和 `CUDA_VISIBLE_DEVICES`。结果：

```text
SLURM_JOB_ID=64732
SLURM_JOB_NODELIST=TC1N03
CUDA_VISIBLE_DEVICES=0
hostname=tc1n03
GPU=Tesla V100-PCIE-32GB, compute capability=7.0,
    driver=580.173.02, memory=32768 MiB, used=0 MiB
torch=2.2.0+cu121
torch.cuda.is_available()=True
torch.cuda.get_device_name(0)=Tesla V100-PCIE-32GB
```

只 import 既有 venv 的 `torch`；没有加载 checkpoint、执行 inference/benchmark 或修改环境。64732 已完成，收尾 `squeue` 为空。

## 远端准确路径（单层 `ls`）

远端 home：`/tc1home/UG/yguo017`。

| 路径 | 结果 | 单层可见内容 |
|---|---|---|
| `/tc1home/UG/yguo017/cem_update_ccds` | 存在 | `artifacts/`, `cache/`, `control/`, `modelroot/`, `run/`, `venv/`, `prepare_summary.json`, `requirements-resolved.txt` |
| `/tc1home/UG/yguo017/cem_update_ccds/venv` | 存在 | `bin/`, `lib/`, `lib64 -> lib`, `share/`, `pyvenv.cfg` |
| `/tc1home/UG/yguo017/cem_update_ccds/modelroot` | 存在 | `checkpoints/`, `data/`, `downloads/`, `source/` |
| `/tc1home/UG/yguo017/cem_update_ccds/run` | 存在 | `pools/`, `targets/` |
| `/tc1home/UG/yguo017/cem_update_ccds/pools` | 不存在 | `ls` 报 `No such file or directory`；已知 pools 路径实际位于 `run/pools/` |
| `/tc1home/UG/yguo017/prr_ccds` | 存在 | `artifacts/`, `control/` |

本次没有进入上述子目录递归扫描，也没有读取 checkpoint、records 或 raw arrays；因此路径存在性与目录名已确认，文件内容/模型完整性仍需在后续获批 compute allocation 中按 idea 的最小 smoke 另行核验。
