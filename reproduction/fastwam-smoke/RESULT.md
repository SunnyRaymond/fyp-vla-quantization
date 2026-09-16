# Fast-WAM Optional IDM：LIBERO-goal 最小 smoke 已跑通

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
