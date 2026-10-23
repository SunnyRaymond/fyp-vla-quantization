# 第一轮总结：Microscaling + action/latent distillation

用户于本轮完成后要求仅总结第一轮、取消第二轮并停止 goal。本文件只纳入第一轮结果，不报告两轮 optimal lambda。

## 实验范围

- 五个 rho：0.1、0.5、0.7、1.0、1.5；每组从同一未训练 Microscaling VQ+A4 初始化，完整完成2 epochs/16 updates，保存epoch0/1/2，共15个selection points。
- 冻结calibration八个contexts；selection为cases1000–1003，每case两个paired sampler seeds。先在case内平均seeds，再对四个cases等权平均。
- train FP32 master codebooks与log row gains；固定VQ indices、B32 Microscaling exponents、Smooth/H128、A4规则和其他model参数。Adam/lr/clip/reset schedule与全部候选一致。
- loss为first10六个motor coordinates的action MSE，加lambda乘两个clean future latent frames的global NMSE，排除clamped frame0。
- lambda = rho × 0.03033027132999191；共同归一化系数只从初始Micro calibration测量一次。
- 主指标为first10 motor RMSE。late22、first10 gripper和case P90不得高于未训练Micro parent的1.2倍；最低eligible first10的1%范围内优先较低rho，再优先较早epoch。

五组均通过完整完成判据：PBS F/Exit0、artifact exit0、PIPELINE_COMPLETE、匹配run/job/protocol、614 targets、2 epochs/16 updates及完整selection history。第一轮统一CPU汇总25813028.pbs101成功并通过canonical verify；所有训练后checkpoints均通过20% guards。

## 每组first10 action RMSE最低的checkpoint

下表每个rho按其first10均值最小的epoch列出；不表示该epoch的全部指标都是本组最优。全部误差相对同条件BF16 teacher，越低越好。

| 版本 | lambda | epoch | first10 motor RMSE | full32 motor RMSE | gripper RMSE | future NMSE | first10 case P90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 未训练Micro parent | — | 0 | 0.047929 | 0.048523 | 0.020474 | 0.050511 | 0.051752 |
| rho0.1 | 0.003033 | 2 | 0.041802 | 0.044273 | 0.020333 | 0.029060 | 0.046056 |
| rho0.5 | 0.015165 | 2 | 0.042850 | 0.045170 | 0.019828 | 0.027881 | 0.047763 |
| rho0.7 | 0.021231 | 2 | 0.042338 | 0.044467 | 0.019760 | 0.022679 | 0.044178 |
| rho1.0 | 0.030330 | 2 | 0.042257 | 0.044926 | 0.022898 | 0.030852 | 0.045985 |
| rho1.5 | 0.045495 | 1 | 0.041793 | 0.043857 | 0.018222 | 0.027984 | 0.045678 |

## 三个清楚区分的结论

1. **按预定协议选中rho0.1 / epoch2，lambda=0.0030330271329991913。** first10 RMSE=0.0418016123585403，相对parent下降12.78%；future NMSE=0.029060483910143375，下降42.47%。该结果来自1% near-tie内优先较低rho的规则。
2. **严格first10最小值是rho1.5 / epoch1，lambda=0.045495406994987865。** first10 RMSE=0.04179286723956466。它与rho0.1/epoch2满足1% near-tie条件；其full32、late22、gripper、P90、future和temporal delta均值也低于rho0.1/epoch2。因此，协议选rho0.1应理解为预定选择偏好的结果，不能描述成它在所有指标全面领先。
3. **future fidelity最佳是rho0.7 / epoch2，lambda=0.021231189930994338。** future NMSE=0.02267876104451716，相对parent下降55.10%；temporal delta NMSE=0.03792815958149731，下降57.40%；first10 case P90=0.04417848065495491，下降14.63%。这三项均为第一轮训练后checkpoints中的最低值。它的first10均值比rho0.1/epoch2高1.28%，超出near-tie范围。

## 对latent-aware idea的判断

五个rho都能在selection上降低action和future均值，但增加latent权重并没有产生单调改善。rho0.7/epoch2相对rho0.5/epoch2在七个aggregate指标全部更低，说明合适的权重可以得到更好的联合恢复结果。

同时，global clean-future fidelity不能充分代表action fidelity：rho0.7/epoch2在四个cases的future NMSE都优于rho0.1/epoch2，action却只赢1/4 cases。rho1.0还出现了case1000 future误差变大、action误差变小的情况。它们支持继续区分latent整体重建、action-sensitive信息和action branch/conditioning的量化误差；当前没有敏感方向、区域/通道或gradient-angle证据，不能确认具体机制因果。

epoch选择也很重要。rho0.7的epoch2改善first10/P90和future均值，但full32、late22、gripper比epoch1退化；rho1.5的first10与future均值都是epoch1更好。单独比较最终epoch会漏掉第一轮严格最低action checkpoint。

## 边界与停止记录

第一轮证据仅为历史selection四个cases的decoded-BF16/fake-A4固定输入teacher fidelity，不是native-kernel、closed-loop或独立泛化结果。rho0 action-only同配置对照已随第二轮取消，故不能单独证明latent loss相对action-only的额外收益，也不能宣称已完成两轮optimal lambda搜索。

第二轮曾提交的三个jobs已按用户要求qdel并确认终态：25813090.pbs101、25813093.pbs101在运行后F/Exit143；25813094.pbs101在排队时取消、F且无exec_host。第二轮不纳入分析，不再续跑或提交后续实验；没有新建定时监控。第一轮数据与checkpoint保留。

原始统一汇总：retrieved/results/choice-r1-25813028.pbs101/choice.json。完成证据：choice_round1_terminal.json及五个run_r1-*_terminal.json。取消证据：round2_user_cancellation.json。
