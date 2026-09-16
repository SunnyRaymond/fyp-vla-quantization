# GPU 前输入锁定

CPU prep 64813 在 TC1N07 完成（14s）；CPU 64814 在 TC1N02 生成紧凑 receipt。完整 manifest 177,780 bytes，SHA256 `71243c83702ada092481abb2772787ebfc01774d8b92d63c4a5812e1771a04ef`，保留于 compute asset。因超过小文件取回上限，未将完整 manifest 下载到本机；紧凑 receipt 保留全部 sample 身份及完整文件 hash，不替代完整源清单。

| Task | Episode | Frame | Length |
|---|---:|---:|---:|
| 0 | 105 | 118 | 237 |
| 1 | 52 | 118 | 237 |
| 2 | 84 | 123 | 247 |
| 3 | 66 | 124 | 248 |
| 4 | 7 | 136 | 272 |
| 5 | 8 | 129 | 259 |

这里的 episode 84 属于 LIBERO dataset，与 DINO-WM/Wall 的 test_locked index 84–95 不是同一 namespace。

固定 base identity SHA256 `be4a49ebe588a49a29bd26ed98b8a01a648a247e66d45e12a935ac7d8d0c4e64`；extension identity SHA256 `f1d55051d660d25d7edfb01522870cdef08650be7752bbe7b52942741d1119ee`。CPU 已核对 source identity，未加载模型、推理、下载或运行环境 rollout。

64811 在 wrapper 阶段失败（exit 125），未产生模型输出。发现 timeout duration 写成不接受的组合单位后，仅改成 285s，提交 64813；空日志不能充当具体报错证据。所有记录保留。
