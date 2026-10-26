# Fast-WAM AQLM loss-pair

只做 AQLM-adapted 的两臂 fixed-input 消融，完成后停止；不授权、也不包含 280-episode 闭环入口。

共同初始化使用官方 `Vahe1994/AQLM@e79a896ed6656fe4ed06193d42d004e7d0bbdbb2` 的 `AQEngine`：input-aware XTX、residual K-means、continuous codebook/scale Adam 与 discrete beam-search。配置为 group8、4 个 256-entry codebooks，nominal index 4 bpw；codebooks/scales 的 BF16 存储、padding 与共同 Smooth/H/R preprocessing 均计入报告。

原有 Smooth0.5、H128 和 learned R0 作为两臂相同的冻结 preprocessing；不是原论文未经 adaptation 的 LLM recipe。A4 为既有 dynamic per-token absmax/qmax7，权重 decode 为 BF16；不声称 native kernel 加速。

共享拟合使用全部 C80 contexts、seed0、完整 sampler 的所有 Linear calls；每个 call 固定均匀抽取最多16 token rows，仅减少 activation 存储，不减少 calibration contexts。每个 module 的所有 C80 采样行进入官方 XTX。

用户授权的 `minimal_v3` 初始化：K-means 最多5 iterations，官方 `max_points_per_centroid=16`，在256 centroids下每次残差 codebook 聚类最多4,096个 weight-group vectors；所有 weight groups 仍执行 assignment。交替优化最多1 epoch，每 epoch5次 Adam updates、一次 beam-search，relative MSE tolerance=0.005，beam size=1。每层最多5次连续更新，全部 C80 XTX 保留。

此前已完成的 heavy_v1 和 light_v2 banks 均原样复用；剩余 layers 用 minimal_v3。manifest 和 preparation summary 分别记录各 profile 数量，两臂读取同一份混合来源的 shared initialization。此次仅改初始化预算，其他训练和评估设置不变。revision3凭据为 `fit_budget_amendment.json`，先前v2凭据与旧协议保存在 history 与远端 control 目录。后续预算修改先完成代码和检查，再受控切换作业。

Capture 的 I/O 修正不改变采样：GPU 上先收集同样的 BF16 rows，一次批量传到 CPU，每个 context 写一个归档；已有按 module 保存的完成 cases 原样复用。拟合前将完整 C80 激活载入获批的180GB host memory，避免对每个 layer 重读整个归档。记录 forward/copy/save 分项耗时；启动时检查新传输布局的 bitwise parity，恢复后再检查 BF16 teacher replay。

官方 K-means 的 CUDA `index_reduce` 不支持 PyTorch strict deterministic 模式。因此仅在共同的 `AQEngine.quantize()` 范围内关闭该模式，并在 finally 中恢复；两臂训练与 sampler replay 仍开启。两臂共享同一份拟合完成的 banks，不各自重新拟合。跨次重新拟合的 bitwise reproducibility 不作保证，当前配对比较的初始化保持相同。

初始化后冻结 indices、底模与所有 preprocessing，仅训练每个目标 Linear 的 codebooks/scales。两臂同一八批 C80 schedule、8 Adam updates、batch10、LR5e-4、gradient clip1；首个 update 后都清空 optimizer moments，与当前参考 schedule 保持一致。

- `reconstruction`：AQLM 原 Linear-output reconstruction objective 的 decoded-BF16 adaptation。BF16 teacher 输入固定；每个模块在相同 A4 operand 上比较其 AQLM weight 与原 transformed weight 的输出 MSE。先平均该模块所有 calls，再平均614 modules、10 contexts。不是完整 block fine-tuning。
- `ours`：现有 first10 motor6 MSE + frozen lambda × future NMSE（排除 frame0）+ beta × frozen BF16 reader soft-bin KL。beta 仅按初始 C80 规则确定；不额外保留 reconstruction 项。reader 参数冻结，但 future 输入梯度保留。

两臂同步启动，每臂一张 GPU。固定 final8，不根据 held-out 选择 checkpoint。每臂完成 C80 final calibration queries 和完整40 contexts × 2 seeds held-out queries。汇总顺序为 seeds→context→suite→四 suites 等权，另列40 contexts的paired胜负平。

协议字段说明：`evaluation` 保留的是历史 reference protocol 的 teacher-cache 元数据，其中旧 `queries=0` 与旧 checkpoint-selection 文本不控制本次执行。本实验的最终评估以新 `heldout`、`expected` 两节为准，worker 与汇总器均要求每臂实际80个 held-out records；它们只用于 final8 的最终评估，不用于 checkpoint selection。

阶段依次为 CPU bootstrap、单GPU共享准备/拟合、双GPU两臂训练和fixed-input评估。控制器使用 SSH host-key verification 和 durable submission intents；查询超时不构成重提依据。所有下载、解压、模型计算、重I/O只允许在已验证的 compute allocation 中；GPU利用率/显存每15秒写入 job log。

本地控制：`control.py upload`、`control.py submit --stage bootstrap`，之后完成上一阶段再提交 `prepare`、`pair`。`status` 为轻量状态；`logs --stage prepare` 读取日志尾部。checkpoint75或已确认失败后需显式使用新 attempt 恢复，不能在Q/R状态重提。

最终产物：`results/comparison.json`、`results/report.zh.md`、两臂部署参数、完整 fixed-input records。闭环 episode 数必须始终为0。

最终比较里的 `mixed_loss` 定义为共同的 `A + lambda*Z`，不含 reader KL；ours 的 reader KL 单独保存在训练历史中，避免把两臂不同定义的训练总 loss 直接作数值比较。
