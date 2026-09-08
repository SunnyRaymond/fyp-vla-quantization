# DINO-WM Wall 官方 checkpoint 评估

**结果：第7轮完成后为47/50=94%，按用户要求停止。比论文96%低2个百分点，相当于少成功1个case。** PBS作业16170841.pbs101已进入F状态，Exit_status=143来自主动终止；不是正常完成原12轮预算。

| MPC完成轮次 | 每case最大环境步数 | 成功case | 成功率 |
|---|---:|---:|---:|
| 1 | 25 | 26/50 | 52% |
| 2 | 50 | 44/50 | 88% |
| 3 | 75 | 46/50 | 92% |
| 4 | 100 | 46/50 | 92% |
| 5 | 125 | 47/50 | 94% |
| 6 | 150 | 47/50 | 94% |
| 7 | 175 | 47/50 | 94% |

失败case的0-based ID为21、24、42。第5至7轮成功率未提升。单张A100作业实际walltime为1小时16分16秒；7轮CEM合计4450.53秒，每轮约10.6分钟；CEM阶段PyTorch峰值allocated显存4.061GiB（不是整进程总显存）。

13个原始artifact已下载并核对传输大小；另生成逐case日志记录和停止汇总。文件位于 `artifacts/16170841.pbs101/`，核心结果为 `stopped_summary.json`、`stopped_cases_from_log.json`、`logs.json`、`job.log`。未重跑GPU。

## 官方究竟测试几轮？

- 论文使用 **50 个测试起点/目标对**，Wall 表格成功率 **96%**。不能把 50 cases 理解为 50 次重复完整测试。
- 官方 Wall 配置中的 `opt_steps=10` 是**每轮 MPC 内部的 CEM 优化次数**。
- 外层 MPC 的 `max_iter=null` 在公开代码中转为无穷大，循环在全部成功或达到上限时结束。论文没有明确报告主表实际使用的外层 MPC 总轮数，不能把内层的 10 次当作论文的外层预算。
- 论文的完整评估重复次数/逐次 seed 清单也未在已核查材料中明确给出。公开配置提供 seed=99；这不能证明论文表格恰好使用本次这 50 对目标。

来源：[论文 v2](https://arxiv.org/html/2411.04983v2)、[固定版本 Wall 配置](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/conf/plan_wall.yaml)、[固定版本 MPC 实现](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/planning/mpc.py)。

## 本次范围与停止规则

只使用官方发布的 `wall_single` checkpoint；没有训练、微调或量化。50 cases，seed=99，random_state goals；CEM candidates=300、topk=30、每轮优化10次、每轮执行5个 model actions，frameskip=5。

原计划最多12轮，但用户在第6轮结束后要求：下一轮结束仍未达到96%即停止。因此停止检查点为第7轮，对仍未成功的 case 最多执行175个 environment steps；已经成功的 case 保留官方成功处理逻辑。

## 与官方结果的可比性

| 项目 | 官方材料 | 本次 |
|---|---|---|
| 测试规模 | 50 对起点/目标 | 50 cases |
| Wall 成功率 | 96% | 第7轮47/50=94% |
| 外层 MPC 预算 | 论文未明确；代码默认无上限 | 用户指定第7轮为停止检查点 |
| Checkpoint 身份 | 发布 Wall checkpoint | 使用发布的 epoch65；与表格使用权重是否完全相同未确认 |
| 软件 | 环境配置 PyTorch2.3.0 / torchvision0.18.0 | PyTorch2.2.0+cu121 / torchvision0.17.0，Python3.11.16 |
| DINOv2 revision | 原训练 revision 未明确 | 固定为7764ea0f912e53c92e82eb78a2a1631e92725fc8 |
| Decoder | 提供可视化解码 | 禁用仅用于可视化的 decoder；planning objective 不依赖它 |

发布 checkpoint 配置含 training.epochs=1000，但实际权重 epoch=65；配置上限不代表实际训练轮数。论文训练超参数与发布配置也存在差异，因此不声称重新实现了论文的训练过程。

这些差异不构成单一原因的证据：不能把成功率差距直接归因于 PyTorch、权重或规划轮数。也未通过更换 seed、目标或训练来追逐96%。

## 结果证据边界

按用户要求提前停止时，程序尚未执行仅在正常退出时生成的完整轨迹和50个视频导出。保存各轮 aggregate logs、原始控制台逐case flags/distances、规划时间、模型和评估配置、初始目标数据。

`stopped_summary.json` 和 `stopped_cases_from_log.json` 从停止后的日志生成，核对50个flags的成功数量、50个距离的均值与aggregate一致。这是日志一致性核对，不等价于从独立保存的完整轨迹重算成功率；不生成 `VERIFIED`，不称为完整产物验证通过。短测视频仅属于2-case短测。
