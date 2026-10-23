# Fast-WAM VQ：future latent 与 teacher-future 配对诊断

研究对象是已冻结的 `row_output_vq_a4` 和最新 final-action distilled VQ（epoch 2，原 test first10 motor RMSE 0.058110）。只做诊断，不训练或选择新 checkpoint。

## 完整范围

保留原协议的 8 calibration contexts、8 selection contexts、44 test contexts，共 60 paired contexts。两份 shard 各 30 contexts，分别是 4 calibration、4 selection、22 test；同一 case 的两个 seeds 在同一 shard。最终合并仍是 22 test cases，每 case 先平均 seeds，再对 cases 等权平均。已看过的 test cohort 不能作为新的独立泛化证据。

每个输入比较五条 arms：BF16、parent/self future、parent/teacher future、distilled/self future、distilled/teacher future。两组诊断在同一个逐输入流程推进，共享 BF16 sampled future。标准查询 180 次，reader-only teacher-future 查询 120 次；两份 shard 各加三次 self-future parity reader 查询，共 6 次。不存在新的 training updates 或 environment episodes。

## 指标与公平条件

- Final clean latent endpoint 指整段 future 的去噪终点，不是只取视频最后一帧；指标排除被 observation clamp 的第 0 个 latent frame。
- 记录整体及每个 latent 时间位置的 RMSE/NMSE，并记录相邻 latent 变化量误差。NMSE 的分母使用对应 teacher energy，设 1e-12 下限。
- Action 记录 first10 motor、full32 motor、late22 motor、first10 gripper RMSE。
- 输入、prompt/proprio、video/action seeds、10+10 scheduler、sigma shift 和未量化模块保持相同；每次替换都检查 initial action noise 与 native query 完全一致。
- Teacher-future 保留当前模型的 prefill/action weights，跳过 video DiT 并注入 BF16 的 final future；首帧仍使用当前真实 observation。每次重新做 prefill，不复用另一个 checkpoint 的 K/V。
- 每次 native query 相对原保存 action 做 parity。每个 checkpoint 在每个 shard 先通过 self-future 替换 parity，防止把替换入口的差异当成上游误差。

## 运行和监控

`control.py` 复用既有 trusted SSH helper，上传/取回仅限小文件。GPU PBS 每段两小时、单 GPU、110GB host memory；每 15 秒把利用率和显存写入 job log。结果按 query 原子保存到远端 `results/shard-0` 或 `results/shard-1`。临近 allocation 上限时正常保存 `resumable`，由经过 terminal 验证的 `continue` 接续。每个 shard 只有一个 writer；不因队列注释或查询失败重提。

固定独立监控对话「Fast-WAM 独立实验监控」（`01a13522-a305-77b3-8fe4-1be2e9cc43a5`）此前通过 heartbeat 每 30 分钟检查一次，始终复用同一对话，使用其 `gpt-6-luna / xhigh` 配置；普通 Q/R 不通知。两个 GPU shards 完成后，用户已要求关闭该监控，automation 状态为 `PAUSED`，后续汇总由原对话处理。所有 torch/model load、推理、latent 指标和汇总计算在真实 PBS compute allocation 内；login node 仅轻量控制。

`analysis-submit --shard 0` 支持仅完整 shard 0 的 interim 汇总，写入独立的 `results/analysis-shard-0`，明确标记 30/60 contexts、11/22 test cases 和 `full_cohort_complete=false`。默认 `analysis-submit` 仍要求两 shards 全部通过完成门槛，写入 `results/analysis`，严格保留 60 contexts 和 22 test cases。CPU allocation 内先运行 scope/配对聚合测试，再生成指标、逐 case CSV、中文报告和图；不读取 `.pt` 或补跑模型。

## 证据边界

这是固定输入上的 BF16 fidelity 和模型内部 teacher-future intervention。Latent 更接近 BF16 不等于更接近真实未来；teacher-future 改善 action 支持该输入上的上游贡献，但不能当成可相加的因果百分比。使用 decoded BF16 VQ 与 BF16 K/V，不支持 native W4A4 加速或 closed-loop success 结论。

完整结果已于 2026-10-13 完成：GPU shards `25787685.pbs101`、`25787687.pbs101` 均通过完整完成门槛；CPU 汇总 `25790422.pbs101` 为 F/Exit0、PIPELINE_COMPLETE，6 项 scope/聚合测试通过。完整 cohort 是 8 calibration、8 selection、44 test contexts，共 60 contexts；5 arms 共 300 metric rows，保留全部 22 test cases（Original 10、Plus 12）。两 shards 的三组 self-future parity 均为零误差。

完整输出位于 `retrieved/results/analysis/`：`analysis.json`、`paired_cases.csv`、`REPORT.zh.md` 和 `paired_diagnostic.png`。此前 shard 0 的 interim 输出仍保留在 `retrieved/results/analysis-shard-0/`，不替代完整汇总。CPU job log 的有界尾部保存在 `retrieved/artifacts/25790422.pbs101/job.log.tail.txt`。

完整 test first10 motor RMSE 复现 parent `0.0572557581` 与 distilled `0.0581104419`。Distilled 的 future latent NMSE 由 parent 的 `0.0844354419` 增至 `0.0930109973`；使用 teacher future 后，distilled 的 first10/full32/late22 RMSE 为 `0.0382938322/0.0389597915/0.0391921317`。Parent 的 teacher-future first10/full32/late22 在 22/22 cases 改善；distilled 对应为 21/22、22/22、22/22。Distilled−parent 的 latent RMSE 变化与 first10 action RMSE 变化之 case-level Spearman ρ=`0.1123658950`，仅为探索性描述。

这些结果支持测试“action supervision + future-latent preservation”的组合，尚不证明 latent-only loss 或具体 λ 会改善量化。Calibration 的 action 改善可以同时伴随 latent 误差增大；test 中全局 latent 与 action 变化的弱关联也说明，保留 action 目标和逐 case 检查很重要。任何新增 λ 或 loss 选择仅使用 calibration/selection，现有 test 是已经看过的 recovery-diagnostic cohort。

## 缓存读取修复

首段两个 GPU jobs 在模型加载前因缺少历史 distilled calibration action 退出。原因是最终 allocation 的 `actions.jsonl` 只保存该段新增 evaluation，先前的 calibration/selection 已保存在 resumable checkpoint 的 `run_state.evaluation_rows` 中。新入口改为从完整 checkpoint 读取所选 epoch 的全部配对 action，数值协议和选定权重不变。

恢复前先在 CPU allocation 检查完整 60 contexts 的 BF16、parent、distilled references 和 tensor metrics。只有成功终态、完成 marker、匹配 summary 与三组各 60 条 reference 同时通过，才重新提交修复后的 GPU入口；保留旧 job artifact 和错误记录。
