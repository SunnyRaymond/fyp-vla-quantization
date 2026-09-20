# DINO-WM PushT CEM：结果卡

| 项目 | 结果 |
|---|---:|
| authoritative job | `24054689.pbs101` |
| decision equivalence | PASS；30/30 bitwise exact |
| full-plan reduction | `4.6681%` |
| peak-memory ratio | `1.00470`，PASS |
| latency gate | FAIL；要求 `>=10%` |
| overall progression gate | **FAIL** |

结论：DINO-WM PushT 上 cache 没有改变 CEM 行为，但只省下约 4.7% full-plan
latency。早期 `23986275.pbs101` transfer 结果同样是 FAIL；本 case 不支持把
DINO-WM PushT 写成已验证的 system-level speedup。
