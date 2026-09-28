# E2：matched predictor-only action-response straightening

状态：已完成一次 **5 updates／arm 的 runtime 与 loss-scale pilot** (`25537028.pbs101`，Exit status 0)。该结果不构成 E2 matched-training 结论；E1 的直接 Jacobian surrogate 路线已达到成本 NO-GO，故不再延长这个训练 recipe。逐项值见 `artifacts/25537028.pbs101/e2_summary.json`。

这次 pilot 的接口检查通过：HDF5 四帧 `[4,224,224,3]`，三个 packed action token `[3,10]`，recorded prediction／target `[1,3,192]`，teacher imagined branch `[1,5,192]`。默认 `λ=1e-4` 下，末次 curvature 项约 `1.81e6`，加权后约 181，基础 recorded＋imagined loss 约 1.76；梯度裁剪前范数约 11374（`λ=0` 臂约 5.27）。这说明预设权重严重失衡，不能把该臂的 5-update 差异解释为有效正则效果。两臂在 `δ=0.02` 的 held-out Taylor relative RMSE 均约 0.0032–0.0034，`2δ` 均约 0.0064–0.0067；该半径也远小于 E1 中真实 CEM 提案的 1–9 L2 尺度。

## 实验比较

从同一个 pinned LeWM `lewm_object.ckpt` 初始化两臂，使用同一 parent-episode manifest、batch 顺序、记录转移、frozen-teacher imagined targets、optimizer 和 update 数：

\[
L = L_{recorded} + \beta L_{imagined} + \lambda L_{curvature}.
\]

- `lambda_0`：`λ=0`。
- `lambda_positive`：`λ>0`，默认 `1e-4`，可通过 PBS 环境变量覆盖。它与 `λ=0` 臂共享所有其它训练条件。
- `L_recorded`：用 HDF5 中 stride-5 的四帧 `[z_t,z_{t+1},z_{t+2},z_{t+3}]` 和前三个 10-D packed action token，调用官方 `predict`，监督实际记录的下一帧 frozen encoder/projector embedding。每个窗口完全留在一个 episode 内。
- `L_imagined`：从同一记录 context 的末端 embedding 出发，对 logged 5-token action plan 及 `v±δu` 三条动作序列作 frozen official-teacher autoregressive rollout；student 匹配这三个 **imagined latent target**。它们是 teacher imagined branches，不是观测到的真实反事实分支。
- `L_curvature`：terminal latent 的中心三点有限差分，`mean_D((F(v+δu)-2F(v)+F(v-δu))²)/δ⁴`。方向是单一合法 action coordinate，三点来自同一中心，不做分别 clipping。 held-out Taylor 测试再用 `δ` 与 `2δ` 检查相对局部线性误差。

训练 `action_encoder`、`predictor`、`pred_proj`；冻结 `encoder`、`projector`。两臂均用 AdamW，默认 lr `5e-5`、weight decay `1e-3`、gradient clip `1.0`。这沿用 pinned LeWM optimizer 数值；为小型固定 update harness 未配置原训练 epoch scheduler。

## 数据与 runtime gate

当前代码依照 pinned `config/train/data/pusht.yaml` 的 `frameskip: 5`，把每个 episode 内连续 40 个 primitive-action rows 分成 8 个 5-action token；图像取 offset `[0,5,10,15]`，动作 transition 取前 3 个 token，imagined rollout 取随后 5 个 token。runner 在 compute allocation 中检查 `episode_idx`、`step_idx`、HDF5 键、动作范围、编码器输出形状、真实转移预测目标形状和 teacher rollout `[1,5,192]`。任一约定不符即停止，不替换归一化、不静默裁剪。

主线程已核对的 pinned LeWM `train.py` loss hook 为 context 前三项和 action embeddings 预测 `emb[:,1:]`，基础项是 prediction MSE，加 `SIGReg(emb.transpose(0,1))`。没有 target/EMA encoder。此处 encoder/projector 冻结，所以 SIGReg 对本 harness 的 trainable modules 是常数；不把它误写成可优化信号。实现依据：pinned `train.py`、`jepa.py` 与 `config/train/data/pusht.yaml`；本地 checkpoint 对象会在 PBS runtime 做模块/shape gate。

每个 arm 有独立 state-dict checkpoint 和简短 training trace；summary 包含 episode/context manifest、同一 update schedule、运行时接口 probe、recorded-transition MSE、center imagined-teacher MSE、action-response 范数与 relative error、held-out Taylor RMSE/relative RMSE。context split 是开发用 episode-disjoint split，不得替代 final test。

## 文件

- `run_e2_action_response.py`：匹配训练和 predictor-level held-out 评估。
- `run_e2_action_response.pbs`：PBS GPU-only wrapper，含真实 `PBS_JOBID`、hostname、CUDA allocation guard；将 GPU utilization/VRAM 每 30 秒写入该 job log。
- `test_e2_action_response.py`：只测纯 finite-difference 数学、合法 action triplet 和 episode split，不加载 checkpoint/HDF5。

本目录准备完成不代表已经运行或验证了集群 runtime。没有下载、hash、环境安装或作业提交。

## 小型 CPU 测试

在项目根目录运行：

```powershell
python -m unittest discover -s meeting-ready/03-action-response-straightening/e2 -p "test_*.py" -v
```

PBS wrapper 的默认预算为 16 train contexts、8 held-out contexts、4/2 个 parent episodes、batch 4、每臂 100 updates、`δ=0.02`。实际仅以 `E2_UPDATES=5` 完成接口与量级 pilot；默认 100 updates 未运行。

## 已验证边界与剩余风险

1. 试跑抽到的 HDF5 action 均通过 `[-1,1]` 范围检查，`episode_idx`／`step_idx` 连续性也通过。stride-5 图像／动作窗口与官方 dataset loader 是否语义完全一致，尚未逐样本对照，不能据此宣称原始训练 recipe 的严格复现。
2. Object checkpoint 的模块、三步预测和 teacher rollout 输出形状已在 GPU compute allocation 验证，40GB A100 显存内完成了两臂各 5 次更新；100-update 训练尚未运行。
3. `λ=1e-4` 在实际训练 loss 中权重严重失衡。若未来研究 predictor 几何，需要事先按训练集量级重定权重，并把训练尺度对齐实际 CEM proposal；不得根据 held-out pilot 反复 retune。这不改变 E1 的直接 Jacobian 成本 NO-GO。
