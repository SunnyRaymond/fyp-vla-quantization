# H+VQ8 与 R8+VQ8：额外成本记录

- Protocol：`fastwam-vq-learned-r128-c8-v1`；上游 PBS job：`25902163.pbs101`。
- 覆盖完整：`True`；新增 model forwards：`0`；新增 teacher queries：`0`。
- 下表记录持久化的阶段/query 操作秒数。它们是阶段计时，不是 GPU kernel latency，也不代表等算力。

| 阶段 | 观测/预期 | 合计秒数 | 单项范围（秒） | 缺失项 |
|---|---:|---:|---:|---|
| R8 rotation training | 8/8 | 504.816 | 62.255–68.587 | 无 |
| 8 个 C8 moments 收集查询 | 8/8 | 134.640 | 16.797–16.863 | 无 |
| 614 个 Linear 的 VQ fitting/refitting | 614/614 | 509.468 | 0.013–5.396 | 无 |
| VQ8 task training | 8/8 | 264.117 | 32.318–34.168 | 无 |
| Calibration 8 queries | 8/8 | 48.274 | 5.881–6.385 | 无 |
| Heldout 80 queries | 80/80 | 528.759 | 5.863–7.441 | 无 |

已测阶段合计：**1990.074 秒**。该合计只加总表中各阶段的已测操作时间。

Encoded storage：614/614 banks，
总计 3202835566 bytes (3054.462 MiB)；
缺失项：无。

PBS walltime（单次上游作业）：2689 秒；读取状态：`available`。
Walltime 与阶段计时分开报告；其中包含启动、checkpoint、排队外的作业执行开销等，且 state 可能跨 resume 作业累积，因此不从两者相减推断启动/checkpoint 成本。失败尝试的 walltime 由外部 qstat 记录。

## R8 实际变化证据

R8 实际变化核验：**已确认**。rotation.pt 声明 8 次更新；按阈值 `max_abs(U-I) > 1e-07`，发生非平凡变化的 streams：
**action, video**

| Stream | max_abs(U-I) | Frobenius(U-I) | max_abs(raw A) | saved orthogonality max_abs |
|---|---:|---:|---:|---:|
| action | 0.0162067 | 0.892299 | 0.00791679 | 1.90735e-06 |
| video | 0.0151227 | 0.833955 | 0.00736475 | 1.84774e-06 |

R8 确认以保存的残差变换矩阵相对 identity 的数值为准；更新计数本身不作为变化证据。上述配对结果为描述性统计，不声称统计显著、闭环性能或 native-kernel 加速。

## 来源

- `state.pt`：`/scratch/users/ntu/yguo017/fastwam-vq-learned-r128-c8-20261021/results/state.pt`（update history）。
- `rotation.pt`：`/scratch/users/ntu/yguo017/fastwam-vq-learned-r128-c8-20261021/results/rotation.pt`（raw A、residual U、保存的正交误差）。
- Moments：`/scratch/users/ntu/yguo017/fastwam-vq-learned-r128-c8-20261021/results/moments/{case_id}.pt`；VQ fits：`/scratch/users/ntu/yguo017/fastwam-vq-learned-r128-c8-20261021/results/banks/{index:04d}.pt`。
- Calibration queries：`/scratch/users/ntu/yguo017/fastwam-vq-learned-r128-c8-20261021/results/calibration/queries/{case_id}_0.json`；heldout queries：`/scratch/users/ntu/yguo017/fastwam-vq-learned-r128-c8-20261021/results/heldout/queries/{case_id}_{seed}.json`。
