# Antithetic Rounding 运行状态

2026-09-13：CCDS SLURM **64733** 已完成，TC1N02 单 V100，elapsed 3m48s，exit 0:0。远端证据目录 `/tc1home/UG/yguo017/v100_newangles_ccds/artifacts/64733`。协议和运行脚本由 compute job 快照保存。

全部工程 gate 通过，Antithetic 相对 Independent 的 shortlist regret 降低36.17%，6/6 episodes和3/3 rounding seeds改善，action MSE gate通过，故机制为 preliminary_go；但单份W8 RTN在两项指标与logical bytes/forward预算上均占优，最终为 practical_no_go。CPU64734独立复算通过且结论一致；[结果已定稿](RESULT.zh.md)，不扩大实验。

独立 code review 的阻断修订：allocation hostname 采用 DNS 大小写正常化并继续检查 ownership/partition/NodeList；补齐 episode、seed、RTN gates；W8 dominance 独立输出 practical verdict；记录并验证 num_hist=1、checkpoint SHA256 与 helper SHA256。数据隔离检查 Subset metadata 映射后的 source IDs 与历史/保留 0–95 不重叠，并记录新 state fingerprint；不打开 84–95。没有完整历史 state fingerprint registry，因此不声称跨全部历史任务绝对 state 独立。此局限不通过读取 reserved TEST 消除。

只有完整工程通过才解释 numerical result。后续 CPU allocation 将从原始 scores/actions 独立复算所冻结指标。
