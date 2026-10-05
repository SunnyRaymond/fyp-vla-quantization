# W4A8 完整输出误差分解：冻结协议摘录

以下保留原 paired-followup 预注册协议中周报提及的 A 实验；原双实验协议未改写，保存于仓库外归档。旧 learned phase 锁定，原 10% 扩展门槛和 NO_GO_FOR_EXPANSION 保持。

沿用 FastWAMOptionalIDM clean checkpoint、固定源码与 stats、BF16 reference、first_frame、20 inference steps、sigma_shift=1、compile=false、32×7 normalized action chunk。使用全部实际执行 denoiser/proprio Linear 的 W4 G128 与 dynamic-row A8；floating GEMM fake-quant。

## A：完整输出误差分解

重放旧作业 `25666715.pbs101/frozen_observations.pt` 的已看过 TEST IDs8..15，每轨迹2 observations。这是探索性诊断，不是新的独立留出验证；不用于新方法拟合/选择。

Sampler seeds仍2026/2027；四个新的 frozen dither seeds2101..2104交替配对，16 observations×4 draws=64条件。对相同 observation/sampler 取得 BF16 reference、W4＋原精度激活、普通RTN W4A8、已锁定learned phase W4A8的完整 action arrays。原精度激活为模型实际BF16，并非另一个均匀 A16 quantizer。

逐条件定义 `dW=aW4_originalActivation-aBF16`，`dA=aW4A8-aW4_originalActivation`，精确检查 `MSEtotal=MSE(dW)+MSE(dA)+2 mean(dW*dA)`。RTN与learned分别报告三项及恒等式残差；保留完整输出以复算。负交互项是误差抵消，不能把各项解读成独立因果贡献或误差占比，W4-only不是全路径不可突破下界。

另按官方controller计算执行prefix10的连续6维 command MSE与gripper命令不一致率：denormalize、gripper*2-1、invert_gripper、按cfg sign；不添加新的clip或controller。先轨迹内平均，再平均8轨迹；bootstrap以轨迹为单位，区间条件于这组冻结draws，不能把observations/draws/actions当独立样本。

工程检查：真实allocation guard、已有source/checkpoint metadata identity、cachedBF16 label重放一致、BF16 repeat/hook-disabled identity、完整site coverage、finite/range、分解恒等式。GPU预算30分钟，无校准搜索。

## 操作边界

所有模型计算、数据读写和重操作在真正 PBS allocation；login node 只做轻量控制。GPU utilization 与显存每 15 秒写入 job.log。保留 SSH host-key verification，不输出 credentials，不动 banked reset。
