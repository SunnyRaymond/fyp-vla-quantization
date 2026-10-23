# H-W16A4 / C80-R128-W16A4：LIBERO+ 配对闭环评测

对源对话 `01a15ec9-1d87-7f93-a02b-5a99830d086a` 的两个固定配置，各跑原冻结的 280 个 LIBERO+ episodes：4 suites × 7 扰动维度 × 10 variants，总计 560 条新 episodes。两组使用同一 environment/sampler seeds 和 initial states。

| 配置 | 固定来源 | 本次训练 |
|---|---|---|
| H-W16A4 | `25903221.pbs101` 的 H 对照；原始 BF16 weights、固定 Smooth0.5/signs/H128，U=I | 无 |
| C80-R128-W16A4 | `25905203.pbs101` 的 `coverage_c80/arms/C80_once_lr1e4/deployment.pt`；固定 final update8 | 无 |

C80 原训练使用 80 个独立 contexts、8 次每次平均 10-context gradient 的 Adam 更新、LR1e-4，只学习 action/video 各一个共享 128×128 Cayley residual。部署保持原模型 BF16 weights、D=I、clipping ratio=1、BF16 KV；两组都没有 VQ 或 weight quantization。

主要指标是 closed-loop success rate，并报告总体、suite、扰动维度、suite × dimension，以及逐 episode paired wins/losses/ties。固定输入 RMSE/latent NMSE 只用于源输出重放核对，与本轮成功率分别解释。源 C80 的 motor/gripper 表现差异不用于重新选择 checkpoint 或调参。

推理保持 10 video + 10 action steps、32-step action horizon、每次执行前 10 个 actions、30 settling steps；常规 cap400、libero_10 cap700。复用原 13 组对照的 3,640 条既有 reference slots，不重新跑 reference episodes。

## 运行安排

单一 normal PBS allocation，4 GPU、32 CPU、220 GB、24h。worker0/1 为 H，2/3 为 C80；各配置按 manifest index 奇偶分片，每 worker140 episodes，错开20秒启动。每张卡加载一次原模型，共四次模型加载。每15秒在 job.log 写 GPU utilization/VRAM。

每个 worker 启动做3次 query：原源 forward 对照冻结 C8 BF16 teacher 重算7项源 metrics；缓存固定变换后的 BF16 weights 再做完整 action/future query，要求与缓存前逐值差0；随后核对正式 infer action。共12次启动 student queries，无新训练、teacher queries 或 calibration fitting。

源端保存的是 endpoint metrics，未保存完整 action/future；因此不声称能直接对照源完整 tensor。缓存前后的完整 tensor parity 和实际 infer parity 在本轮 allocation 中验证。固定 weights 使用与源相同的 Smooth/sign/H/U FP32 运算后转BF16，缓存一次，正式 query 继续执行原 activation transform 和 dynamic A4。

有效 episode 原子追加，恢复只接受同协议、同来源的连续前缀。85500秒软截止时保存合法 checkpoint，不把未达560条标成完成。完成要求 PBS/artifact exit0、PIPELINE_COMPLETE、四 worker 各140、两组各280、3,640 reference slots 和启动检查均通过。

所有模型加载、tensor I/O、推理和真实 reference 聚合均在已获批 PBS compute allocation 中执行；login 只做轻量控制、状态查询及小文件操作。`control.py upload/submit/status/logs/pull/resume` 提供对应入口，保留 SSH host-key verification。

提交后由 gpt-6-luna / xhigh subagent 每30分钟检查。Q/R 无实质变化时安静；失败、完成或需要处理时通知原任务。此轮测的是 BF16 weights / A4 的闭环结果，不声称 native low-bit kernel 加速。
