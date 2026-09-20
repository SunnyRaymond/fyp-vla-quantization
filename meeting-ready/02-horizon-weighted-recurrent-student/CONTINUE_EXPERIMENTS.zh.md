# 只从本目录继续实验

本文件定义 bundle 的路径边界。`src/`、`config/`、`jobs/`、`artifacts/` 和 `reports/` 已经足够继续当前 predictor-level recipe；`src/cache_core.py` 是 teacher cached-prefix rollout 所需的本地 helper，不再从旧的 `dino-wm-shared-prefix-cache` 目录导入。真正运行 official DINO-WM teacher 仍需要 ASPIRE2A 上的外部 source、checkpoint、data 和 runtime。

## 1. Bundle 内部路径

PBS wrapper 以自己所在的 `jobs/` 的上一级作为默认 bundle root：

```text
BUNDLE_ROOT/
├─ src/run_dino_pusht_*.py
├─ config/*.json
├─ config/*.md
└─ artifacts/<job-id>/...
```

因此不再依赖原始 `experiment/idea-validation/jepa-action-prefix-compiler/` 路径。若 bundle 被放到远端其他目录，设置 `DINO_APC_ROOT` 指向 bundle 根目录即可；也可以分别用 `DINO_APC_SRC_ROOT`、`DINO_APC_CONFIG_ROOT` 和 `DINO_APC_ARTIFACT_ROOT` 覆盖。

## 2. 仍需准备的外部 ASPIRE2A 资产

以下内容有意不复制到 bundle：

```text
DINO_WM_ROOT/
├─ source/                              # official DINO-WM source
├─ checkpoints/outputs/pusht/...        # frozen teacher checkpoint
├─ data/pusht_noise/                    # PushT data
├─ artifacts/pusht-python-overlay-pusht-v5/
├─ PUSHT_DEPS_READY_V5
└─ runtime-complete.tar                 # staged Python/Torch runtime
```

默认 `DINO_WM_ROOT` 是 `/scratch/users/ntu/yguo017/dino-wm-wall`。`--root`/`DINO_WM_ROOT` 必须指向 official DINO-WM 根目录，不是本 bundle。不要把 checkpoint、数据或完整 source repository 复制进这个组会目录。

ASPIRE2A 约束仍然有效：login node 只做连接、提交、qstat 和轻量控制；模型加载、teacher rollout、pre-encoding、training、evaluation、下载/解压/编译和重 I/O 必须在获批 PBS allocation 中完成。wrapper 自带 compute-node guard，并每 30 秒记录 GPU utilization/VRAM。

## 3. 最小运行流程

先将整个 bundle 放到远端可读目录（使用集群允许的传输路径），例如：

```text
/scratch/users/ntu/yguo017/meeting-ready/02-horizon-weighted-recurrent-student
```

在登录节点只做环境变量和提交操作：

```bash
export DINO_WM_ROOT=/scratch/users/ntu/yguo017/dino-wm-wall
export DINO_APC_ROOT=/scratch/users/ntu/yguo017/meeting-ready/02-horizon-weighted-recurrent-student
qsub "$DINO_APC_ROOT/jobs/dino_pusht_recurrent_student.pbs"
```

要复现 horizon-weighted treatment：

```bash
qsub "$DINO_APC_ROOT/jobs/dino_pusht_horizon_weighted.pbs"
```

两个 wrapper 的默认引用如下：

| 输入 | bundle 内默认位置 |
|---|---|
| recurrent/horizon freeze | `config/` |
| 256-context manifest | `artifacts/24466744.pbs101/context_density_manifest.json` |
| legacy 128-context manifest | `artifacts/24373175.pbs101/query_coverage_manifest.json` |
| Dhigh summary | `artifacts/24477396.pbs101/context_density_summary.json` |
| recurrent control summary | `artifacts/24503839.pbs101/recurrent_student_summary.json` |
| horizon treatment 的 control summary | `artifacts/24494756.pbs101/query_slate_rank_summary.json` |

PBS 输出默认写入 external `DINO_WM_ROOT/artifacts/`，使用新 job id 命名；不会覆盖 bundle 内历史 evidence。若希望结果写入另一个位置，可设置 `DINO_PUSHT_*_OUT_ROOT`。

## 4. 修改实验时的规则

- 先复制对应 freeze 到 `config/` 并改成新的版本名；保留旧 freeze 和历史 artifacts 不动。
- 新 treatment 复制或新增 runner 到 `src/`，让 local imports 仍通过同一 `src/` 目录解析；不要把旧 `experiment/idea-validation` 路径重新写回 runner。
- 继续沿用 manifest、held-out block key 和 baseline summary 的 pairing；不要把不同 slate 或不同 seed 的结果直接拼成 paired delta。
- 每次实验使用新的 output/job id；summary JSON、`job.log`、`job_status.txt`、`gpu_usage.csv` 和 telemetry 一并取回即可，不需要取回 checkpoint。
- 先做最小检查，再提交：

```bash
python -m py_compile "$DINO_APC_ROOT"/src/*.py
bash -n "$DINO_APC_ROOT/jobs/dino_pusht_recurrent_student.pbs"
bash -n "$DINO_APC_ROOT/jobs/dino_pusht_horizon_weighted.pbs"
```

不要在 login node 执行 runner 或加载模型；也不要为了“验证完整性”对大目录做递归 hash。

## 5. 证据读取顺序

1. 先看新 job 的 `job_status.txt` 和 `final_exit_status.txt`，确认是运行完成还是 wrapper 失败。
2. 再读 summary JSON 的 `integrity`、`convergence`、`gates`、`decisions` 和 `claim_boundary`。
3. 用 `reports/RESULT_HORIZON_WEIGHTED.zh.md` 的 gate 定义解释结果，不根据单个有利 block 改写结论。
4. 将新结果写成新的 `reports/RESULT_<NEW_ID>.zh.md`，保留本包已有的两份正式结果。

当前包已经停止在 horizon-weighted 结论处；后续 spatial/token mixer 等已知 NO-GO 分支不应作为本 idea 的默认继续方向。
