# R128 C80 QClip：LIBERO-Plus 闭环评估

本实验评估 `25937633.pbs101` 产出的 final80 QClip checkpoint。它从同一个最终 Q0 reader-KL checkpoint 开始，只复用 614 个已学习的 per-Linear clipping ratios；R0、VQ weights 和其余参数保持冻结。本轮不训练，也不补做固定输入 forward。

`qclip` 在原始 280 条 LIBERO-Plus manifest 上运行一次。四个 worker 按 `index % 4 == worker_id` 各跑 70 条，沿用同一组 state、environment seed、sampler seed 和 episode cap。开跑前每个 worker 执行四次 startup query：重放 QClip calibration endpoint 的八项指标并要求最大误差不超过 `1e-6`；重放 Q0 case 2000 / seed 0 的完整 action 与 future anchor并要求精确相等；再检查正式 inference action parity、614 个目标模块和冻结安装 metadata。任一门槛失败都会在 episode 前停止该 worker。

结果与 17 个冻结参考 arm 配对：原有 15 个 reference arm，加上已完成的 Q0 `az` 和 `reader_kl` 各 280 条 episode，总计 4,760 个 reference slots。PBS allocation 内会先只读校验完整 reference cohort，再启动 episode workers；reference JSONL 不会被恢复逻辑改写。只有四个 worker 都以 exit code 0 完成、汇总包含全部 280 条新 episode 和 4,760 个 reference slots，才会写入 `PIPELINE_COMPLETE`。

任务只提交一个 PBS job：`normal` queue，要求 4 GPUs、8 小时，正常路由须落在 `g3`。运行至 7 小时 50 分钟时会写 checkpoint 并以成功状态退出；本 protocol 不提交第二个 job。PBS `job.log` 每 15 秒记录 GPU utilization 与显存占用。四个 worker 分别绑定一张 allocation GPU，任一 worker 失败会终止其余 worker。

推理使用 decoded BF16 dense MXVQ4 weights、dynamic A4 activations 和 BF16 KV cache。此实验测量闭环任务结果与实际 query timing，不构成 native low-bit kernel 或加速结论。

本地只运行标准库测试，不加载 Torch、不载入模型，也不连接集群：

```powershell
C:\Users\Raymond\miniconda3\python.exe -Xutf8 -B -m unittest test_evaluation test_aggregate test_control
bash -n run.pbs
```
