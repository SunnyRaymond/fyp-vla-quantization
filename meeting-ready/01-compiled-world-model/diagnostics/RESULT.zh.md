# Compiled World Model：无重训机制诊断结果

## 结论

本轮 routing 为 **`COMPILER_BOTTLENECK_SIGNAL`**，但不是 deployable GO。

- 现有 B3 的 candidate common component 不是主要可修复瓶颈：换成 teacher mean 后，development ranking 没有恢复。
- 用 teacher action residual 替换后几乎完全恢复，说明现有 `B(c)phi(u)` 动作响应错误。
- 冻结现有 `phi(u)`，每个 context 用一个 candidate bank 局部拟合 affine coefficients 后，在另一 independent bank 上达到 Spearman/top-30 median `0.8846/0.6500`。这支持 `phi` 含有可用动作特征，主要失败点更接近 `c → (b,B)` coefficients 的生成或训练。
- 简单 zero-reference anchored residual 反而使 B3 ranking 下降，未通过 quality gate。因此不应直接发展当前公式的 01-v2，也不进入 official CEM。

原 job `25309513` 的 predictor-level **NO-GO 保持不变**。

## 作业与范围

- PBS job：`25327163.pbs101`，A100-SXM4-40GB。
- runner/PBS `Exit_status=0`；walltime `00:05:24`，cput `00:01:11`，PBS peak memory `1,328,708 kb`。
- PBS `Stageout_status=1`，但 scratch 内 summary、`job_status=EXIT_STATUS=0` 和 `final_exit_status=RUNNER_EXIT_STATUS=0` 完整生成并回传，没有重跑。
- 全程未创建 optimizer、未调用 backward、未训练网络；GPU telemetry 每 5 秒写入 `job.log`。
- `official CEM`、closed-loop、新 final test 均为 `NOT_RUN_BY_SCOPE`。

## Diagnostic 1：common / action contrast

### Unseen development contexts

数据为旧 development `valid[552:560]`，8 episodes × early/middle/late × 2 banks，共 48 blocks。teacher contrast energy 占 total latent energy 的 median 约 `0.0120`；没有 block 触发 low-contrast exclusion。MSE decomposition 最大闭合误差为 `5.59e-8`。

| Model / prediction | Spearman median | top-30 median |
|---|---:|---:|
| B1 raw | `0.0796` | `0.1333` |
| B1：teacher mean + student residual | `0.0107` | `0.1167` |
| B1：student mean + teacher residual | `0.9880` | `0.8667` |
| B1 anchored zero reference | `0.0293` | `0.1333` |
| B3 raw | `0.1307` | `0.1667` |
| B3：teacher mean + student residual | `0.0453` | `0.1333` |
| B3：student mean + teacher residual | `0.9951` | `0.9333` |
| B3 anchored zero reference | `0.0583` | `0.1000` |
| anchor-aligned positive control raw | `0.8904` | `0.7167` |
| positive control anchored | `0.8926` | `0.6667` |

B3 的 median contrast relative error 为 `1.4810`，`gamma=0.8400`。这不是简单的“动作变化幅度被压成零”：预测 contrast 的能量与 teacher 同量级，但方向/几何错误。按 horizon，contrast relative error 从 H1 的 `2.6654` 降到 H5 的 `1.3999`，仍始终大于 1。

相比之下，B1 的 `gamma=0.0737`，更接近 action response collapse。旧 positive control 的 `gamma=0.2691` 虽较小，却有明显更好的 ranking，说明仅匹配 residual energy 并不足够，方向和 cost-sensitive geometry 更重要。

### Training contexts：原候选与新候选

| Model | Candidate bank | raw Spearman / top-k | contrast relative error | gamma |
|---|---|---:|---:|---:|
| B3 | 训练时原 64 candidates | `0.1552 / 0.2000` | `1.5239` | `1.0164` |
| B3 | 同 context 新 64 candidates | `0.0663 / 0.1500` | `1.5256` | `1.0783` |
| positive control | 原候选 | `0.9377 / 0.9000` | `0.8332` | `0.3793` |
| positive control | 新候选 | `0.9247 / 0.8000` | `0.8424` | `0.3597` |

B3 连训练 contexts 的原 candidate banks 都没有形成可靠 ranking，换新 candidates 后更弱；因此不能把失败只解释为 unseen-context generalization。teacher action-residual oracle 在两类 banks 上均恢复到 Spearman `>=0.995`、top-k `>=0.90`，进一步定位到 action response coefficients。

## Diagnostic 2：冻结 phi 的跨-bank局部拟合

对每个 development context 的 terminal `phi(u)`，使用 bank A 的 300 teacher latents 固定 ridge-fit `[1,phi] → z_5`，再测试 bank B；然后反向，共 48 directions。lambda 未调参，两个 banks 没有 exact duplicate candidate。

| Metric | Fit bank median | Cross-bank median | Cross-bank minimum |
|---|---:|---:|---:|
| Spearman | `0.9796` | `0.8846` | `0.3949` |
| top-30 overlap | `0.8667` | `0.6500` | `0.2000` |
| relative terminal MSE | `0.000582` | `0.003897` | — |

- `42/48` directions 的 cross-bank Spearman `>=0.7`；`35/48` 的 top-30 overlap `>=0.5`；全部 48 directions 的 relative terminal MSE `<=0.1`。
- 按 episode 聚合，8/8 episode 的 Spearman median 均 `>0.82`；7/8 的 top-30 median `>=0.5`，剩余 episode 9523 为 `0.4833`。
- ridge system condition number median 为 `1.40e6`，因此该信号并非无条件稳健；但固定 regularization 下的 independent-bank结果显著强于原 compiler output。

这项结果说明：**现有 shared `phi(u)` 的局部表达容量是有用的；当前 B3 的主要失败不是 rank 192 不够，而是 compiler 没有从 context 生成正确 coefficients。** 由于 coefficients 使用了一个 teacher-paid bank 拟合，这仍是 oracle mechanism evidence，不是部署性能。

## Diagnostic 3：zero-reference anchored residual

使用与 labels 无关的全零合法 action sequence，按

`teacher(c,u_ref) + student(c,u) - student(c,u_ref)`

修正预测。B3 development Spearman/top-30 从 `0.1307/0.1667` 降到 `0.0583/0.1000`，仅 `3/8` episodes joint strict improvement，冻结要求为 `>=5/8`。quality gate 明确失败。

30-query timing 完整包含一次 teacher reference：

| Path | p50 | p95 |
|---|---:|---:|
| official teacher 30×300 | `575.978 ms` | `580.206 ms` |
| raw B3 30×300 | `12.757 ms` | `12.835 ms` |
| anchored B3：1×teacher ref + 30×300 | `30.273 ms` | `30.492 ms` |

anchored path 在该 predictor boundary 上相对 teacher 为 `19.03×`，但质量失败，所以不能宣称有效 acceleration。它也比 raw B3 慢，倍率为 `0.421×`。

## Routing decision

| Check | Result |
|---|---|
| teacher common-component oracle recovery | **FAIL** |
| teacher action-residual oracle recovery | **PASS** |
| frozen phi cross-bank local-fit gate | **PASS** |
| deployable zero-reference anchored quality | **FAIL** |
| anchored timing | **PASS** |
| route | **`COMPILER_BOTTLENECK_SIGNAL`** |

下一步若继续 01，只应研究更受约束、数据效率更高的 compiler parameterization或 coefficient supervision，并保持 `phi`、rank 和 candidate banks 不变来隔离变量。不要继续扫 rank，也不要把 simple anchoring 推进到 CEM。任何新的 compiler 都需要重新冻结 predictor-level comparison，并与旧 anchor-aligned positive control 和 matched dense prefix 同台比较。

## Claim boundary

这是旧 development data 上的无重训 mechanism diagnostic。common/action replacements 与 local ridge fit 都包含 oracle information；结果不支持 planner replacement、official CEM、closed-loop PushT 或 task success。
