# LeWorldModel PushT：CEM iteration cache

这是跨模型 iteration-cache bundle 中的 LeWM PushT case。核心 idea 是：在一次
`CEMSolver.solve` 内，固定的 initial observation embedding 和 goal embedding
只编码一次；candidate-dependent 的 action encoder、autoregressive predictor、
criterion 和 CEM 更新全部保持官方实现。

## 先看什么

1. 组会快速说明：`MEETING_CARD.zh.md`
2. 已验证结果：`RESULT.zh.md`
3. 冻结实验协议：`LEWM_PUSHT_ITERATION_CACHE_FREEZE.json`
4. 可继续运行的入口：`lewm_pusht_iteration.pbs`

## 已验证结果

实验 `24382364.pbs101` 完成了 paired GPU benchmark；初次 verifier 只因把
非有限诊断值直接输出为 JSON 而退出。随后 CPU recovery job
`24389764.pbs101` 使用修复后的 verifier 重读同一个 `system_summary.json`，
最终 decision gate 和 system gates 均 PASS。关键小型证据在
`artifacts/24382364.pbs101/`，没有复制 checkpoint、数据集或完整大 summary。

- bitwise exact：30 个 paired units 的 final actions、first actions、costs 和
  30 次 CEM decision trace 完全一致
- full planner latency median reduction：`29.4091%`
- plan-section latency median reduction：`29.4358%`
- inner CEM cost median reduction：`29.7539%`
- peak-memory 最大 ratio：`0.97083`
- frozen gate：最低 full-solve reduction `10%`，peak-memory ratio 不超过 `1.10`

这个证据支持“在该官方 LeWM PushT CEM 边界内，cache 保持行为等价并降低
planner latency”。它不是 closed-loop task success 结论；冻结评估使用两个
固定 observation、固定 CEM 配置，并且不包含环境交互。

## 继续实验入口

在 ASPIRE2A 上，先把整个 bundle 复制到远端，例如：

```text
/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/01-iteration-cache-cross-model-planners/
```

然后确认下列外部资产已经存在。它们故意没有复制进 bundle：

```text
$LEWM_REPO_ROOT/jepa.py
$LEWM_REPO_ROOT/module.py
$LEWM_REPO_ROOT/config/...
$STABLEWM_HOME/pusht/lewm_object.ckpt
$STABLEWM_HOME/pusht_expert_train.h5
$LEWM_PYTHON_BIN
```

当前已使用的远端 staging 约定是：

```text
LEWM_REPO_ROOT=/scratch/users/ntu/yguo017/lewm-pusht-iteration/le-wm
STABLEWM_HOME=/scratch/users/ntu/yguo017/lewm-pusht-iteration/stablewm_home
LEWM_PYTHON_BIN=/scratch/users/ntu/yguo017/lewm-pusht-iteration/venv/bin/python
```

若 checkpoint 尚未生成，可在 compute-node allocation 中先提交
`lewm_asset_prep.pbs`；它负责准备官方 source、依赖和 object
checkpoint。也可以只使用已经准备好的 external assets，不需要再次下载。

主实验提交前可设置：

```bash
export LEWM_REPO_ROOT=/scratch/users/ntu/yguo017/lewm-pusht-iteration/le-wm
export STABLEWM_HOME=/scratch/users/ntu/yguo017/lewm-pusht-iteration/stablewm_home
export LEWM_PYTHON_BIN=/scratch/users/ntu/yguo017/lewm-pusht-iteration/venv/bin/python
qsub cases/lewm-pusht-cem/lewm_pusht_iteration.pbs
```

PBS 脚本默认以本 case 目录为 control root，以 case 的
`artifacts/$PBS_JOBID` 保存结果；也可以用
`LEWM_ITERATION_CONTROL_DIR` 和 `LEWM_ITERATION_OUT` 覆盖。脚本包含
`PBS_JOBID`、compute-host、GPU allocation 和 30 秒 GPU telemetry guard。

若主 job 只在最后 verifier 输出阶段失败，可在 CPU allocation 中运行
`lewm_verifier_recovery.pbs`，设置 `LEWM_RECOVERY_SOURCE_JOB_ID`
和 `LEWM_ITERATION_OUT` 指向同一结果目录；它不会重新运行模型。

## 约束

- login node 只做连接、提交、qstat 和小型控制文件操作。
- 下载、解压、依赖准备、checkpoint 转换、模型加载和 benchmark 必须在 PBS
  compute allocation 内完成。
- 不要把 checkpoint、HDF5 dataset 或大型 `system_summary.json` 纳入 Git/bundle。
- 继续实验时先保留 freeze 中的 solver、seed、paired order 和 exactness gate；
  若要改变它们，应另建 experiment id。

## 目录

```text
cases/lewm-pusht-cem/
├── README.zh.md
├── MEETING_CARD.zh.md
├── RESULT.zh.md
├── iteration_cache.py
├── run_lewm_pusht_iteration.py
├── verify_lewm_pusht_iteration.py
├── prepare_lewm_checkpoint.py
├── LEWM_PUSHT_ITERATION_CACHE_FREEZE.json
├── lewm_pusht_iteration.pbs
├── lewm_verifier_recovery.pbs
└── lewm_asset_prep.pbs
└── artifacts/24382364.pbs101/
    ├── verifier.json
    ├── verifier_recovery_summary.json
    └── qstat_24382364_24389764.txt
```
