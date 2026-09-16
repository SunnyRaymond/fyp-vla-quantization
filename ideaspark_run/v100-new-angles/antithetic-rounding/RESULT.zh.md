# Antithetic Rounding：机制初步成立，实际方案 no-go

2026-09-13。最终结论：**mechanism preliminary_go / practical_no_go**。CCDS GPU job **64733**、CPU独立复算 **64734** 均完成、exit 0:0；不继续增加实验，也不修订成新的rounding seed/bitwidth搜索。

## 最小实测

官方 DINO-WM Wall epoch65，实际 `num_hist=1`、`concat_dim=1`、384+10+10输入；V100 TC1N02，GPU job elapsed **3m48s**，PyTorch peak allocated 4,151,328,768 bytes（是本次FP32数值模拟峰值，不是native W4部署显存）。6个新source episodes（96–101）、每组300个H5 candidates、3个重复rounding seeds。只量化24个predictor Linear，encoder FP32；两个W4模型各自完成rollout后平均score。

| 方法 | mean top30 FP-score regret ↓ | H5 elite-mean action MSE ↓ | predictor logical bytes | predictor forwards |
|---|---:|---:|---:|---:|
| W4 RTN | 0.02491611 | 0.01779676 | 10,070,976 | 1 |
| Independent W4 pair | 0.02061188 | 0.01942467 | 20,141,952 | 2 |
| Antithetic W4 pair | **0.01315747** | **0.01429430** | 20,141,952 | 2 |
| W8 RTN | **0.0000069433** | **0.00037992** | **19,999,680** | **1** |

Antithetic相对同成本Independent的regret降低 **36.17%**；6/6 episode与3/3 rounding-seed平均改善，超过冻结的5%、4/6、2/3门槛。相对W4 RTN的regret与action MSE亦通过非劣gate。该结果支持“同边缘rounding分布下的负相关coupling可以改善此单次shortlist”的窄机制。

但两份W4的logical storage略高于单份W8，且需要两次predictor forward；W8在所测两项保真指标均更好。因此按事前practical gate停止。**没有工程缺陷证据支持再调方法挽救该实际方案**。这个no-go是否定目前的精度/成本组合，不抹去已测到的机制增量。

## 证据与边界

- [冻结协议](IDEA_AND_PROTOCOL.zh.md)、[GPU原始小汇总](artifacts/64733/summary.json)、[独立复算](artifacts/64734/verification.json)。Scheduler记录位于campaign `scheduler/64733`、`scheduler/64734`。
- CPU从完整scores/actions独立重算，所有aggregate在容差内一致、mechanism verdict一致，无复算错误。复算直接验证的是数值W8 dominance；logical bytes/forward对比来自GPU保存的量化布局与调用定义，不是native deployment测量。
- 完整raw arrays、quantizer audit、protocol/helper snapshots保留在 `/tc1home/UG/yguo017/v100_newangles_ccds/artifacts/64733`；raw SHA256：`59490c59aff46cff198e07b982ad186258ca01ba8bbaa77fb3fa5a610369fbfc`。未把大数组传到head或本地。
- checkpoint SHA256、helper identity、allocation ownership、actual V100、weights精确restore与encoder不变检查通过。CPU没有独立重跑模型或重算weight covariance；这些工程检查依据GPU记录。
- Subset metadata确认新source IDs为1297、360、1153、1185、1766、1471，与0–95的底层source IDs不交。84–95 reserved TEST state未打开；无完整历史state fingerprint registry，不能主张所有历史state绝对独立。
- 6个episode是比较单位；3个rounding seeds是重复测量。没有统计显著性、任务success、闭环规划、native W4速度或真实显存节省结论。

本候选至此归档，所有正负结果保留。裁剪pipeline没有继续生成卡片或完整验证。
