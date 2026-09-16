# Euler Jacobian：一次局部几何诊断（读取新输出前冻结）

2026-09-13。IdeaSpark C02 controlled diagnostic + C04 decomposition；prior gate 仅允许 checkpoint-specific diagnostic，generic method novelty no-go。不训练 quantizer，不做 endpoint/closed-loop/full validation。

## 假设与范围

同一 fixed-prefix、t=.5、FP trajectory x 上，expert W4 是否使 Euler map 的第一 action token 32D Jacobian 出现 robust orientation reversal 或严重局部病态，而 FP 仍 regular。M=I32−0.1 Jv。det<0 仅表示局部 orientation reversal，不证明 global folding；小 singular value 不证明成功率损失。32D 含25个 padding coordinates；7D physical slice 不闭合。仅 velocity error 不支持假设。

## 固定输入与计算

复用 rounding_persistence_ready/manifest.json，SHA256 71243c83702ada092481abb2772787ebfc01774d8b92d63c4a5812e1771a04ef。六个先前冻结的 observation 有意复用：task0..5，episode105/52/84/66/7/8，middle frame118/118/123/124/136/129。本实验不读取旧 scientific outputs。checkpoint、source、processor、raw sample identity 沿用且实测核对。

新 CPU torch.Generator seed2301 生成单个 float32 noise[1,50,32]，六个 observation 共用。FP K10 的前5步(t=1,.9,.8,.7,.6，dt=-.1)生成共同 x(t=.5)，不生成 Q trajectory。固定 prefix KV，eval、FP32、no RTC/compile，所有 params requires_grad=False，只对 x leaf 求导。FP/W4 两 arm，expert112 Linear，W4 row-max RTN symmetric [-7,7]；CUDA scale=maxabs*(1/7)，zero row scale1。保存 FP/code/scale/dequant/实际 readback/restore 全量快照。

v[0,0,j] 对全 x[1,50,32] 做32次串行 autograd.grad，保存完整32×50×32 Jacobian。对 first32×32 block 在独立 CPU float64 计算 SVD/slogdet。未来49 tokens 的 derivative 必须 absmax<=1e-7。固定 tail perturb=.1*CPU torch.randn([49,32],seed2402) 的 first velocity 应 allclose(atol1e-6,rtol0)。FP no-grad vs grad velocity 同一阈值。

固定两个32D unit Rademacher directions（CPU torch.randint seed2401，除sqrt32），central FD epsilon=.002。每 arm/state/direction 保存 vplus/vminus；FD 与 Jd 的 error norm <= 1e-3+0.02*norm(Jd)。该 gate 是实现校验，不按失败调整 epsilon/precision。prefix cache 内容不变、FP恢复、非 expert weights 不变、quant readback、source identity、V100 allocation 必须通过。

## 预注册判据

FP eligible：det(MFP)>0，sigma_min(MFP)>=.05 且 sigma_min/sigma_max>=.01。至少4/6 eligible，否则 inconclusive_binding。

Q severe conditioning：sigma_min(MQ)<=.005 且 <=.1*sigma_min(MFP)，同时 condition ratio<=.1*FP ratio；或 robust orientation reversal：det(MQ)<0 且 sigma_min(MQ)>=.005 且 condition ratio>=1e-4。至少3个 eligible states flag => scope_limited_preliminary_go；否则 mechanism_no_go。先检查全部 engineering/AD/FD/closure gates；任一失败 => implementation_inconclusive，不能报告科学 go/no-go。全部12个 arm/state 完成才分析；超时或缺失 => inconclusive_budget。

GPU 一次10分钟，内部480秒/外部540秒；CPU replay一次5分钟。任何结论均STOP，不加time/noise/task，不深入完整验证。仅有已定位实现错误且切面仍可做时另立repair gate。

## 原始数组合同（raw.npz）

float32: noise[1,50,32], x_fp[6,50,32], v[6,2,50,32], jac[6,2,32,50,32], directions[2,32], fd_values[6,2,2,2,32]（sign order plus,minus）, tail_probe_v[6,2,32], tail_delta[49,32], fp_noop_velocity[6,50,32], fp_path[6,6,50,32], fp_path_v[6,5,50,32]。completed bool[6,2]；episode_ids int64[6]=[105,52,84,66,7,8]；timestep float32 scalar .5, dt float32 scalar -.1, fd_epsilon float32 scalar .002。arm order FP,W4。

CPU replay独立计算科学量、FD与closure/no-op/path重建；GPU engineering.json + runtime.json + quant_snapshot.pt + source/protocol/helper receipts 独立核验。raw与完整快照只留 compute filesystem，本机只取小型 report。所有计算/重I/O只在真实CCDSSLURM allocation；head只轻量控制。
