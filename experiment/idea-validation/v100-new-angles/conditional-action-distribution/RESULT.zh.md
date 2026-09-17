# Conditional action marginal：statistical_inconclusive，停止

冻结的初筛结论是 **statistical_inconclusive**：0/8条件满足bounded dissociation，只有1/8可进入完整negative gate。未看到支持“mapping明显改变而marginal仍在FP参考带内”的样本，但不能越过预先冻结的灵敏度要求，改判整体mechanism_no_go。

六个condition的FP-only平移对照未达到SWD ratio≥1.50；另一个condition不满足large-mapping-drift要求。所有condition的原始Q/FP SWD ratio为2.18–3.79、energy ratio为3.89–11.45，均在reference band外。这些是描述性shift迹象，不能替代完整gate。没有检测到预定义collapse。

| Condition | Dpair/Dnoise | Q/FP SWD ratio | 平移SWD ratio | 冻结标签 |
|---|---:|---:|---:|---|
| task0 / 85 | 1.584 | 3.620 | 1.314 | sensitivity_inconclusive |
| task0 / 88 | 1.046 | 3.794 | 1.446 | sensitivity_inconclusive |
| task1 / 21 | 0.574 | 3.450 | 1.463 | sensitivity_inconclusive |
| task1 / 37 | 0.724 | 2.954 | 1.385 | sensitivity_inconclusive |
| task2 / 59 | 0.935 | 3.757 | 1.494 | sensitivity_inconclusive |
| task2 / 80 | 0.445 | 2.519 | 1.397 | sensitivity_inconclusive |
| task3 / 49 | 0.426 | 3.247 | 1.619 | marginal_shift |
| task3 / 50 | 0.169 | 2.182 | 1.688 | mapping_binding_absent |

实现检查通过，没有代码错误证据。本次停止，不调平移方向、projection数、阈值或noise数量来改善结论；不开展完整distribution或environment验证。

GPU64778在TC1N05的Tesla V100-PCIE-32GB完成3m12s，workload189.440s，峰值allocated3,095,739,392bytes。固定8conditions、FP32/expert-W4两arms、各64noise共1024个action samples。首condition FP/Q的batch4与逐条最大差分别1.848e−6、1.788e−6，低于冻结1e−5 gate。CPU64779完成4s，独立FP64重算SWD/energy、FP-only reference和positive control、noise exact replay、IQR与跨condition判断；runtime/source/checkpoint/sample/weight restore等检查通过。

完整raw保留在 `/tc1home/UG/yguo017/v100_newangles_ccds/artifacts/64778/raw_marginal.npz`，SHA256为 `0d3f89ef5a6ba671251f05c2d7f5061b4356bb1eb34c82cb8edf6bbbb2800c45`。逐condition完整数值与工程证据见 [verification.json](artifacts/64779/verification.json)，运行记录见 [summary.json](artifacts/64778/summary.json)。最初缺素材及CPU补齐经过见 [PREPARATION_STATUS.zh.md](PREPARATION_STATUS.zh.md)，原失败64775保留。

这是单checkpoint、单W4 recipe上的有限7D action screen，不证明distribution等价、统计显著性、任务成功、真实W4速度或新颖性。每个condition内的64noise不作为64个独立episode。
