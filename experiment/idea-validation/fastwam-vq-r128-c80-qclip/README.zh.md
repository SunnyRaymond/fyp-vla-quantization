# C80 R128 Q-Clip：固定 Q0 的 activation-only 校准

从 `25921795.pbs101` 的最终 `reader_kl/update8` checkpoint 开始。仅训练614个 FP32 clipping ratios；原模型、VQ codebooks/row gains/indices/B32 exponents、Smooth/H/R、bias、BF16 reader均冻结。保持完整变换后的 dynamic per-token A4；ratio跨token及denoising step共享，初始1，范围[0.5,1]。

Loss固定为 first10×6 motor MSE + 0.04245452683169643×future latent NMSE + 0.006083668576461056×frozen reader soft-bin KL。复用原160份BF16 teacher packets和reader statistics，不重新生成teacher或重算beta。新student future的reader forward和反传仍需要执行，并独立计数。

训练沿用旧clip-only recipe：原C80 cases2000–2079/seed0，四suite交错顺序，一遍80次单context更新，Adam LR0.01，gradient norm clip1，首次更新后清Adam state。固定最终update80，不用heldout挑checkpoint。训练使用staged reader VJP，再精确重放student；新clip算子放在完整Smooth/H/learned-R之后。

对Q0和Qclip分别覆盖calibration80和原heldout40×2seed=80，共320个逻辑端点。Q0 case2000/seed0复用原完整tensor anchor，其余缺失的同checkpoint records补算。每条端点原子保存，断点续跑复用已验证结果。四suite按seed→context→suite等权汇总。报告first10 motor RMSE/MSE、future latent NMSE、full32/late22 motor RMSE、gripper RMSE和经过真实controller转换后的gripper disagreement。端点mixed_az_loss仅包含A+Z，训练日志另外报告reader KL和完整loss。

原heldout已用于工程分析，结果属于固定cohort比较。无W16训练臂、无闭环episodes、无native kernel或speed结论。

提交相同工作内容的01:59:00和12:00:00两个normal/1GPU作业。compute node通过非阻塞文件锁选出单一writer，控制端/subagent确认先运行者后取消另一。两者共用可恢复results；exit75才按已提交更新和端点续跑。不能因查询超时或排队解释而重提。每15秒在job log写GPU utilization/VRAM。

仅上传小型源文件。模型、checkpoint和所有heavy I/O均在验证PBS allocation后的compute node执行，禁止login-node计算。控制入口见control.py，提交/续跑handle保存在RUN_STATE.json；monitor subagent仅操作本任务handles。
