# C80 joint final80：fixed-input RMSE

来源训练线程：`codex://threads/01a11a6e-7199-7d72-82ec-ca3b3aaef7e7`。本评估线程：`codex://threads/01a14a81-66a7-7200-a3c2-27a15969f596`。已按用户授权发消息，来源线程确认 final80 训练完成并通过完成门槛后回信，提供实际 final training job ID 和 deployment。

本目录只做 evaluation，不重新训练。读取来源 C80 joint 的最终 codebooks、row gains 和 clipping ratios；保留原 fixed indices、B32 exponents、Smooth0.5/H128、bias、BF16 KV 和采样设置。评估时冻结所有梯度，但保留 clipping forward 分支以实际使用 learned ratios。

固定输入为原 C80 calibration80（case2000–2079，seed_index0），以及原 held-out40 contexts ×2 seeds（case3000–3039，seed_index0/1），四 suites 等权。所有输入、seeds 与 teacher 直接复用；不增加 development set，不按 held-out 选择 checkpoint。复用已完成的 baseline/clip_only 共320条端点记录，仅新增 joint160条端点。主指标为 first10 motor RMSE 与 future latent NMSE；full32、late22、gripper 仅诊断。

提交前须确认来源 final training job 已 terminal/exit0、artifact exit0、TRAINING_COMPLETE、matching complete calibration_summary、receipt=80 updates，且 deployment/reference action 均存在。模型操作放在正式 GPU allocation；先验证原 baseline reference action，再验证 joint loader 与来源保存的 reference action 一致。完整端点逐查询保存，只有验证为合法软截止（terminal exit75 + checkpointed）后允许续跑，保留已完成 queries。

GPU任务为 normal、1 GPU、16 CPU、110GB、walltime01:59:00，软截止6840秒；GPU utilization/VRAM每15秒写job log。CPU汇总不占GPU。gpt-6-luna / xhigh subagent每10分钟监控，普通Q/R静默，仅在完成、失败或需要处理时报告。上游训练、模型操作与重I/O均不能放在login或本机；本机只做源码、轻量JSON和控制操作。

预期来源：`/scratch/users/ntu/yguo017/fastwam-vq-c80-clip-joint-plus-20261017/results/training/joint/deployment.pt`。实际最终job与来源receipt在提交时绑定到 `source_evidence.json`，目前不假定训练已完成。

结果属于固定输入的 decoded-BF16 VQ + fake A4 数值比较。本任务不运行新的 closed-loop、native kernel 或 channel scaling。

## 完成结果

已读取来源 final80 `25858027.pbs101` 完成评估。GPU `25858590.pbs101` 用时00:47:26，CPU aggregate `25859109.pbs101` 用时00:00:16；均通过 terminal/exit0、artifact exit0、PHASE_COMPLETE 与 matching complete summary。原 baseline 和 joint reference action parity均为0。新方法160个端点、复用对照320个端点，合计480；本任务训练更新数0、新 teacher数0。Luna监控已停止。

原 held-out joint first10 motor RMSE `0.041846837196499106`，future latent NMSE `0.048973336233757436`；相对 baseline 为−5.5307%/−19.5541%，相对 C80 clip_only80 为−4.5648%/−10.7110%。主指标两项均改善；late22/full32/gripper仅诊断。

见 [完整结果表](results/RESULTS.zh.md)、[解释](results/ANALYSIS.zh.md) 和 `completion_evidence.json`。本轮未继续提交 channel scaling 或其他训练。

后续闭环结果已只读核对并追加到解释文档：C80 clip_only为207/280，joint为202/280；joint对clip_only paired9 wins/14 losses/257 ties，camera_viewpoints净少5个成功。RMSE均值降低并没有在本轮转化为更高闭环成功率，后续应保留两组对照，不能只按RMSE选择方案。本线程没有追加推理或episode重跑。
