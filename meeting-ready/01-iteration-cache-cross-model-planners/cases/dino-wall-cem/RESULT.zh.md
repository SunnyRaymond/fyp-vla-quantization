# DINO-WM Wall CEM：结果卡

| 项目 | 结果 |
|---|---:|
| job | `23435258.pbs101` |
| decision screen | PASS；10/10 observations |
| paired median latency reduction | `14.3485%` |
| latency gate | FAIL；要求 `>=20%` |
| max peak-memory ratio | `1.00007`，PASS |
| overall system gate | **FAIL** |
| closed-loop authorization | NO |

结论：缓存后的 chained-CEM decision trace 在 frozen screen 中保持一致，但收益
没有达到预设 system gate。该结果是 approximate factorized-cache screen，不能与
下面的 DINO-WM Wall GD exact iteration-cache PASS 混写。
