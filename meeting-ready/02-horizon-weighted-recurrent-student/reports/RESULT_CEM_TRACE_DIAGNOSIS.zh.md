# PushT CEM trace diagnosis

## 结论

Student 最可能做不好的不是单次 latent rollout 的平均误差，而是 **CEM 真正依赖的局部 action-to-cost 排序，尤其是 elite selection boundary**。初始 candidate pool 相同时，student 已经会选出不同的 top-30；随后 teacher 与 student 各自用自己的 elites 更新 proposal，误差被 CEM feedback 逐轮放大，最终优化到一个 teacher 并不认可的 action distribution。

这个判断只覆盖 paired pilot 中 teacher success / student failure 的 6 个 cases，在首个固定 observation 上做 mechanism diagnosis。它不是新的 closed-loop success rate，也没有证明 spatial mean pooling 或 closed-loop OOD 是根因。

## 诊断设计

- parent closed-loop pilot：`24544733.pbs101`
- diagnosis job：`24560503.pbs101`，exit code `0`
- cases：`[0, 1, 2, 4, 5, 7]`，即全部 6 个 teacher-only successes
- 每个 case：`30` CEM iterations、`300` candidates、top-`30`、`H=5`、packed action width `10`
- 保存 iterations：`1 / 5 / 10 / 30`
- teacher/student 在每个 case 和 iteration 共享同一组 frozen standard-normal innovations；candidate 0 固定为各自的 pre-update mean
- official CEM update 语义保持为 plain `torch.argsort`、elite mean、`torch.std` 默认 unbiased estimator
- 两个互补视图：
  - teacher pool：同一批 teacher-proposed actions 同时由 teacher 和 student 评分；
  - student pool shadow：student 自己 proposal 中的 actions 再由 frozen teacher 评分。

独立实验单位是 PushT case；300 candidates 只是 case 内 nested measurements，不能当作 300 个独立样本。

## Case-level median 结果

| CEM iteration | teacher pool Spearman | teacher pool top-30 overlap | student-pool teacher-shadow Spearman | shadow top-30 overlap | teacher cost regret of student elites | first-action RMS drift |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.975736` | `0.766667` | `0.975736` | `0.766667` | `0.042395` | `0.135645` |
| 5 | `0.692362` | `0.366667` | `0.495475` | `0.333333` | `0.182082` | `0.507735` |
| 10 | `0.745187` | `0.400000` | `0.541895` | `0.166667` | `0.153710` | `0.647466` |
| 30 | `0.421793` | `0.083333` | `0.238582` | `0.066667` | `0.096018` | `0.731702` |

关键点：

1. **误差在第一次 elite selection 前已经存在。** Iteration 1 两臂 action pool 完全相同，但 median top-30 overlap 只有 `0.766667`。Spearman 看起来仍高，是因为全局排序相关性会掩盖 top-k cutoff 附近的交换；CEM 只使用这 30 个 elites。
2. **CEM feedback 明显放大分叉。** Median first-action RMS drift 从 iteration 1 的 `0.135645` 增至 iteration 30 的 `0.731702`，约 `5.39×`。
3. **Student 最终在优化自己的 surrogate。** Iteration 30 的 student-pool teacher-shadow Spearman median 为 `0.238582`，top-30 overlap 仅 `0.066667`；student elites 在 teacher shadow 排名中的 mean rank median 为 `117.53 / 300`。
4. **不是单一 outlier。** 六个 cases 的 iteration-30 shadow top-30 overlap 分别为 `0.0333, 0.1000, 0.0333, 0.1667, 0.1333, 0.0000`；first-action RMS drift 都在 `0.6505–1.0145`。

冻结判据因此给出：

- `direct_initial_pool_scoring_mismatch = SUPPORTED`
- `iterative_cem_amplification = SUPPORTED`
- `student_distribution_shadow_misranking_at_iter30 = SUPPORTED`
- `spatial_mean_pooling_causal_attribution = UNTESTED`
- `closed_loop_ood_causal_attribution = UNTESTED`

## 这解释了什么

此前 held-out evaluation 的 absolute Spearman median 约 `0.97`，但它衡量的是较宽的候选排序。真实 CEM 会反复截取 top-10% 并重拟合 proposal；即使整体 Spearman 较高，elite boundary 上的小错也会改变下一轮 candidate distribution。这个 trace 把 closed-loop 的 `2/8` failure signal 与 predictor-level 指标之间的断层具体定位到了：

```text
initial local elite mismatch
  -> different mu/sigma update
  -> different candidate distribution
  -> student surrogate 与 teacher objective 越来越不一致
  -> materially different first action
```

## 尚未证明的内容

- 这是对 exact parent initial targets 的重新采样 diagnosis，不是对原 closed-loop run 中每个随机 candidate 的逐字节 replay。
- 只检查首个 MPC observation；不能排除后续 observation drift 进一步恶化结果。
- 没做 architecture ablation，所以不能把根因归给 spatial mean pooling、hidden size 或 horizon weighting 中的某一个。
- 不能从这 6 个 selected failure cases 推断 population-level PushT failure rate。

## 下一步最小实验

不建议先加训练步数。最有信息量的是一个直接对准 planner objective 的 intervention：训练时加入 teacher-cost local rank / elite-aware loss，并用同一份 trace protocol 检查 iteration-1 top-30 overlap 是否提高、iteration-30 first-action drift 是否下降。只有这两个 mechanism gates 同时改善，才值得再开新的 bounded closed-loop pilot。

## Artifacts

- aggregate summary：`artifacts/24560503.pbs101/cem_trace_summary.json`
- per-case traces：`artifacts/24560503.pbs101/case_00.json` 等 6 个 JSON
- execution record：`artifacts/24560503.pbs101/job.log`
- GPU telemetry：24 samples，median utilization `100%`，peak memory `23655 MiB`

