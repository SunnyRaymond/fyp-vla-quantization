# Fast-WAM Optional IDM：LIBERO-goal 最小 smoke 已跑通

当前结果：**2026-10-02 新作业 `25658646.pbs101` 已完成单 episode，1/1 成功，PBS `Exit_status=0`**。本次使用 `first_frame` 模式；[结果 JSON](artifacts/25658646.pbs101/results.json) 与本页末尾的复跑记录给出当前证据。以下 2026-09-09 内容保留为首次跑通的历史记录。

2026-09-09，ASPIRE2A PBS 作业 **16180074.pbs101** 正常结束，Exit_status=0，allocated walltime=6分14秒。

## 结果与证据

- Checkpoint：`libero_optional_idm_2cam224.clean.pt`，对应官方发布的 `libero_optional_idm_2cam224.pt`；12,041,735,545 bytes，SHA-256与HF LFS记录一致。
- FastWAM revision：`7faa71108368fbb3b6885649f112af607427a2d4`。
- 配置：`libero_optional_idm_2cam224_1e-4`，`action_infer_mode=first_frame`，`sigma_shift=1.0`，`compile_action_infer=false`。
- 任务：LIBERO-goal task 0，`open the middle drawer of the cabinet`，trial 0，共1 episode。
- JSON结果：**1/1成功**，任务执行55.935760秒；此时间不含完整模型加载，不能作为模型推理延迟。
- GPU：一张A100-SXM4-40GB；PyTorch实际UUID与PBS分配UUID一致。
- 渲染：OSMesa CPU软件渲染，两路256×256图像；模型推理使用CUDA。
- 视频：228,479 bytes，完整FFmpeg解码无错误；最后一帧已目视核查抽屉打开。

产物在本文件同级 `artifacts/16180074.pbs101/`：

- [视频](artifacts/16180074.pbs101/rollout.mp4)
- [结果JSON](artifacts/16180074.pbs101/gpu0_task0_results.json)
- [评测日志](artifacts/16180074.pbs101/eval.log)
- [完整作业日志](artifacts/16180074.pbs101/job.log)
- [实际执行的PBS脚本](artifacts/16180074.pbs101/osmesa_smoke.pbs)
- [最终画面](artifacts/16180074.pbs101/final-frame.png)

## 症结与修复

原问题是robosuite的EGL实现按整数解析CUDA_VISIBLE_DEVICES，但PBS给的是GPU UUID；EGL又暴露不同于CUDA mask的全局设备列表，不能假定index 0就是获批GPU。此前的反复设备映射没有解决这个接口差异。

本次改为复用现有LIBERO SIF里的OSMesa，CPU生成画面、分配的GPU做模型推理。无需改写PBS的GPU mask，也不需要在未分配的GPU上创建GL context。

Host Python与容器的兼容处理仅在作业启动时生效：绑定libffi.so.6、libssl.so.1.1、libcrypto.so.1.1，并优先使用容器的libstdc++，避免host GCC11覆盖LLVM15所需版本。CPU预检覆盖完整eval入口、真实reset、initial state及10步动作后的两路图像；GPU作业在模型加载前再次验证。

## 范围边界

这是**管线 smoke 成功**，且这一次episode完成了任务；不是完整LIBERO-goal评测，也不是IDM模式或GPU EGL渲染速度复现。

主checkpoint来自官方HF发布；VAE/T5使用此前准备的公开HF mirror，tokenizer来自官方Wan-AI repo。文件传输完整性已验证，当前loader实际能加载并完成rollout；mirror自身hash不能单独证明与DiffSynth官方文件逐字节相同。

## 后续复用

当前GPU作业已结束，监测已暂停，无需自动重跑。若用户另行要求重跑，可在login node仅提交小型控制脚本：

```bash
qsub /scratch/users/ntu/yguo017/fastwam-smoke/osmesa_smoke.pbs
```

所有模型加载、渲染、推理及重I/O均在PBS compute allocation内执行。旧 `egl_preflight.pbs` 和 `smoke_libero_goal.pbs` 已禁用，不能继续使用原全局EGL探测路线。

## 2026-10-02 单 episode 复跑

- 新作业：`25658646.pbs101`，最终状态 `F`、PBS `Exit_status=0`；先因 queue `ngpus` 总限额为 `Q`，随后在 compute node `x1000c1s1b0n0` 运行。PBS walltime 为 9 分 03 秒（请求 1 GPU、16 CPUs、110 GB、30 分钟）。OSMesa preflight 通过。
- 入口：`osmesa_smoke.pbs`。每 15 秒将获批 GPU 的 utilization 和 memory 写入 job log，并分别记录 task success/failure 与管线完成状态。
- 本次使用官方 Optional IDM checkpoint 的 `first_frame` / Fast-WAM direct-action mode。`eval_libero_single.py` 调用 `model.infer_action`，再把 action chunk 执行于 LIBERO 环境；job.log 记录 `Loaded checkpoint via model.load_checkpoint`。这不是 IDM 模式或 suite 评估。
- 结果：`libero_goal` task 0（`open the middle drawer of the cabinet`）**1/1 成功**，episode 执行时间 57.45 秒。JSON 本地副本：[results.json](artifacts/25658646.pbs101/results.json)，327 bytes；exit code 本地记录为 0，并有 `PIPELINE_COMPLETE`、`TASK_SUCCESS` 标记。
- GPU：PBS 分配 A100-SXM4-40GB；job.log 15 秒采样记录利用率最高显示 41%，显存占用约 24,769 MiB。
- 视频：非空 MP4 228,479 bytes，保留在远端 `/scratch/users/ntu/yguo017/fastwam-smoke/artifacts/25658646.pbs101/rollout.mp4`；没有保留本地媒体副本。远端同目录保存 `job.log`、`eval.log`、`results.json`、`exit_code.txt` 与任务 outcome marker。
