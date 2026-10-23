# rho=0.7：Microscaling + action/latent distillation 分析

rho=0.7 对应 lambda=0.021231189930994338。续跑 job25811378.pbs101 已完整通过 PBS F/Exit0、artifact exit0、PIPELINE_COMPLETE 与 canonical search_complete 检查：614 targets、16 updates、2 epochs 和完整 epoch0/1/2 selection。初始 segment25808524.pbs101 的13个updates和既有epoch1结果已沿同一run保留。

CPU 汇总25812002.pbs101 已完整验证，纳入当前四个完整候选的12个selection points。全部来自冻结 cases1000–1003、每个case两个paired seeds；先在case内平均seeds，再对cases等权平均。误差相对同条件BF16 teacher，以下结果均只涉及selection。

## 主结果

| 指标 | 未训练 Micro | rho0.7 epoch1 | rho0.7 epoch2 | epoch2相对parent |
|---|---:|---:|---:|---:|
| first10 motor RMSE | 0.047929 | 0.042795 | 0.042338 | -11.67% |
| full32 motor RMSE | 0.048523 | 0.042870 | 0.044467 | -8.36% |
| late22 motor RMSE | 0.048719 | 0.042671 | 0.045333 | -6.95% |
| first10 gripper RMSE | 0.020474 | 0.016690 | 0.019760 | -3.49% |
| first10 case P90 | 0.051752 | 0.049796 | 0.044178 | -14.63% |
| future latent NMSE | 0.050511 | 0.027421 | 0.022679 | -55.10% |
| temporal delta NMSE | 0.089023 | 0.048473 | 0.037928 | -57.40% |

rho0.7 epoch1/2 均通过原定20% guards。epoch2 的first10 action、future NMSE、temporal delta NMSE分别在四个cases全部改善。这是相对未训练Micro parent的联合恢复证据；rho0 action-only同配置对照未完成，尚不能单独归因latent loss。

## 和另外三组比较

| epoch2版本 | first10 motor RMSE | case P90 | gripper RMSE | future NMSE | temporal delta NMSE |
|---|---:|---:|---:|---:|---:|
| rho0.1 | 0.041802 | 0.046056 | 0.020333 | 0.029060 | 0.046638 |
| rho0.5 | 0.042850 | 0.047763 | 0.019828 | 0.027881 | 0.045997 |
| rho0.7 | 0.042338 | 0.044178 | 0.019760 | 0.022679 | 0.037928 |
| rho1.0 | 0.042257 | 0.045985 | 0.022898 | 0.030852 | 0.051265 |

rho0.7/epoch2 是当前四组中 future NMSE、temporal delta NMSE 和first10 case P90最低的候选；gripper均值也是这四个epoch2候选中最低，不过与rho0.5差距很小。它相对rho0.5在七个已汇总指标全部更低：first10 -1.20%、full32 -1.56%、late22 -1.50%、gripper -0.34%、P90 -7.51%、future -18.66%、temporal delta -17.54%。这些是四case的aggregate比较，不意味着每个case在每项上都更好。

相对rho0.1/epoch2，rho0.7的future NMSE低21.96%、temporal delta低18.68%、P90低4.08%、gripper低2.82%；同时first10均值高1.28%，full32高0.44%、late22高0.20%。因此，它在future重建与较差case的first10误差上有优势，rho0.1在action均值上有优势。P90是四个seed-averaged case RMSE的线性插值统计，不代表独立大样本的尾部保证。

## 逐case显示：latent fidelity与action fidelity仍不同步

| case | rho0.1 action | rho0.7 action | rho0.1 future NMSE | rho0.7 future NMSE |
|---|---:|---:|---:|---:|
| 1000 | 0.048116 | 0.044795 | 0.024125 | 0.023513 |
| 1001 | 0.039681 | 0.042739 | 0.032859 | 0.024881 |
| 1002 | 0.041251 | 0.042300 | 0.029449 | 0.020652 |
| 1003 | 0.038158 | 0.039517 | 0.029810 | 0.021669 |

相对rho0.1，rho0.7在全部四个cases的future NMSE和temporal delta NMSE上更低；action则只在case1000更低，其余三个cases更高。这个逐case结果进一步支持：最终global clean-latent reconstruction error并不能充分代表action fidelity。即使每个case的future误差都变小，action的排序仍可相反。

一种可能解释是action对latent中特定方向更敏感，而当前global NMSE对整个future tensor进行重建约束；同时视频conditioning/KV与action branch的量化误差也会影响输出。sampler实际使用video KV prefill和10步action denoising，final future NMSE没有直接约束这些表征及action head输出。现有JSON没有敏感方向/区域/通道误差或gradient-angle证据，不能把这个解释当成已确认机制，也不能宣称future imagery在真实环境中更正确。

## epoch1到epoch2的取舍

rho0.7 从epoch1到epoch2：first10均值再下降1.07%、P90下降11.28%、future NMSE下降17.29%、temporal delta下降21.75%；同时full32上升3.72%、late22上升6.24%、gripper上升18.40%。三项guards仍低于parent，故均通过。

future均值的进一步下降主要来自case1001：该case future NMSE从0.049554变为0.024881；其余三case分别从0.021347→0.023513、0.019142→0.020652、0.019640→0.021669，均略有回升。first10 action则在case1000/1001下降、case1002/1003上升。因此，epoch2的均值与P90变好主要反映不同case间的重新分配，不能解释为所有cases持续同步改善。

epoch1是rho0.7在full32、late22、gripper上的较好checkpoint，epoch2则在first10均值/P90以及future/temporal均值上更好。当前直接action loss只监督first10的六个motor coordinates，late22与gripper没有对应的直接action loss；结果与这种监督范围带来的取舍相容，但不构成机制因果证明。

## 选择结论与证据边界

按预先冻结的first10 action主指标、20% guards和1% near-tie规则，当前四组仍选择rho0.1/epoch2。rho0.7的first10均值比它高1.28%，超出1% near-tie范围；本次不因为它在future和P90上的优势而改动选择规则。

rho0.7提供了更有价值的联合恢复结果：相对未训练parent，action与future在四个cases都改善，而且相对rho0.5可同时改善全部七个aggregate指标。它值得保留在最终比较中；仍须等rho1.5和第二轮（包含rho0 action-only）完成，才冻结所测试候选中的optimal lambda。

本次新增模型forward=0，test forward=0；没有读取或下载.pt，所有新数值汇总在真实PBS CPU allocation内完成。结果限于decoded-BF16/fake-A4、历史selection四个cases的固定输入teacher fidelity，不能当成native-kernel、closed-loop或独立泛化证据。

原始四组CPU汇总：retrieved/results/partial-25812002.pbs101/；验证记录：partial_analysis_four_terminal.json。
