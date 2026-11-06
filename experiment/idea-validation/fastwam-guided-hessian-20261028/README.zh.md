# WAM-guided Hessian（插件）：S1 探针 + S2 加权 Hessian + QTIP W2 宿主

## 做什么

把 `L = A + λZ + βK` 的 Gauss-Newton 矩阵变成每个 Linear 的 token/去噪步权重 `u` 和输出行权重 `d`，得到

```
H̃ = Σ_t (u_t/ū + γ) x_t x_tᵀ / Σ_t (u_t/ū + γ)
```

替换宿主量化器里的 `XᵀX`，宿主代码一行不改。文件格式与 A16 baseline 的 `prep/hessians` 相同。

| 头 | 定义 | 探针（Rademacher ε） |
|---|---|---|
| A | 前 10 步 × 6 维 motor MSE（60 个 scalar 的均值） | `ε/√60`，放在 action[:10,:6] |
| Z | future frame 1: 的 NMSE | `ε/‖z_T[:,:,1:]‖`，放在 future[:,:,1:] |
| K | soft-bin KL(teacher ‖ reader 读出的动作) | `ε·√(h/2)`，h 为 teacher 动作处 KL 的对角曲率 |

λ = 0.04245（与旧 loss 相同），β = 0.006084（loss-pair reader_kl 的冻结值），γ = 0.1，C80 × 每个 context 1 个探针。

## 每个 context 的反传（BF16 模型，线性化点即 teacher）

1. query 前向（video 10 步 + prefill + action，带梯度）。所有目标 Linear 的输出经过 `Probe`：值不变，挂在一个零 anchor 上让 BF16 模型建图，反传时记录 `‖g_t‖²` 和 `Σ_t g_tc²‖x_t‖²`。
2. reader 前向：同一组权重，从 query 自己的 future（detach 成叶子）跑 prefill + action 10 步。在 teacher 点，这就是冻结 BF16 reader。
3. 从 reader 的动作反传 `v_A`：记录 prefill、action、proprio 的调用，同时得到 z 处的 `c1`。
4. 从 reader 的动作反传 `v_K`：不记录，只得到 z 处的 `c2`。
5. 从 query 的 future 反传 `c1 + √β·c2 + √λ·v_Z`：记录 video 去噪步的调用。贵的 video 段只反传一次。

reader 的每次调用按输入签名（两个 FP64 和）对回 query 的调用编号，所以权重能对上之后回放中的每一次调用。每个 suite 的前 2 个 context 上，三个 z 方向的余切分开反传（头 Az、K、Z），用来给出 C1 的分解。

## 宿主：QTIP W2（`host.json`）

- 代码：A16 baseline 的 `methods/qtip`，原样复制；每层 seed 与 baseline 相同（20261028 + 层号），符号向量一致，只有 Hessian 不同。QTIP 只吃输入侧 Hessian，行权重 `d` 这里不用。
- 臂：`QTIP_U_W2` = baseline 已经跑好的 QTIP_W2 权重（H_U = mean xxᵀ），直接复用，不重量化；`QTIP_G_W2` = QTIP W2 用 H̃。
- 每层同时算两个臂在 H_U 和 H̃ 下的 proxy。
- `protocol.json` 在集群上已冻结，其中的 `gptq`、`evaluation.arms`、`allocation`（g1）由 `host.json` 取代；loss、探针、Hessian、门限和 held-out 面板仍以 `protocol.json` 为准。

## 作业：一个 g2 双卡作业（`run.pbs`）

| 步骤 | GPU 0 | GPU 1 |
|---|---|---|
| check（约 10 分钟） | QTIP 用 baseline H_U 重跑最小的一层，和 baseline 权重比对；1 个探针 context + 3 层回放（签名核对）；BF16 和 baseline QTIP_W2 各 1 个 held-out endpoint | 用 baseline 的 loop_worker 在 QTIP_W2 权重上跑启动门（T5 缓存 anchor、eager vs cache+Graph 逐位一致）和 1 个 episode |
| probe | 一半 C80 context（分头诊断的 context 计 2 倍成本，按成本均分），等另一半写完，再做一半层的加权 Hessian（按 in² 均分） | 同左，另一半 |
| quant | 一半层的 QTIP W2（按 baseline QTIP_W2 每层实测秒数均分） | 另一半 |
| eval | QTIP_U_W2 held-out 40×2 + C1 swap | QTIP_G_W2 同上 |
| final | 诊断、`quantized/QTIP_G_W2/COMPLETE.json`、`summary.json`、`RESULTS.zh.md` | — |
| loop | LIBERO-Plus 280，每卡 2 个 worker 共享一个 episode 队列；baseline 的 `loop_worker.py`/`analysis.py`/引擎文件，T5 缓存和 BF16 anchor 链到 baseline 的 prep | 同左 |

任何一张卡的进程失败，作业立刻结束。所有步骤跳过盘上已有的结果，重新提交 `--attempt r1` 即从断点继续。check 通过后写 `results/check/PASSED`，重提时跳过。

## 第一阶段结果（26043905）

逐 token 对角的 H̃ 让 held-out mixed loss +14%、future NMSE +17%；闭环 208/280，baseline QTIP_W2 为 204（配对 14 胜 10 负）。GN 对角估计 A 头只有 6.9% 经 future，但换回 BF16 future 的实测去掉了 first10 motor MSE 的 44–50%。详见 `pulled/26043905.pbs101/`。

## 第二阶段：去噪轨迹相干的 Gauss-Newton（`host.json` 的 `coherent`）

同一个 W 在 10 个 video 去噪步、prefill、10 个 action 步里反复使用，量化误差 ΔW 在每次调用里完全相同，它对 loss 的一阶影响是 ⟨ΔW, G⟩，G = Σ_调用 Σ_t g_t x_tᵀ。H̃（和 GuidedQuant）只保留 Σ_t ‖g_t‖² x_t x_tᵀ，丢掉跨 token、跨步的交叉项。

- 草图：Probe 的反传钩子拿到 g 和（save_for_backward 的）x，累加 S = Rᵀ G（R 为 ±1/√32，按 context、探针、层各取新的），E_R[SᵀS] = GᵀG。不加反传次数，只多一次 video 反传把 A 经 future（c1）和 βK+λZ 分开。
- 两个变体：C = A+λZ+βK，CA = 仅 A（直接路径 + 经 future 的路径；相当于 GuidedQuant 式的终端 loss 对照）。
- H = SᵀS/均值对角 + 0.1·H_U/均值对角，S 堆叠 80 个 context × 2 个探针。QTIP 代码不变，种子与 baseline 相同。
- 门槛（预注册）：相干模型下 A 头经 future 的能量占比 1 − Σ‖S_A‖²/Σ‖S_A+S_Az‖² ≥ 0.2，否则不量化。
- 行缩放（0 bit）：QTIP 的 SV 本来就是每输出行一个 fp32，量化 diag(s)W 再把行除回 s 等于改 SV。s_c² = √d_c/‖W_c‖。预注册规则是两半 context 交叉预估 guided loss 降 ≥10% 才采用；`rowcheck.py`（26046194）在第一阶段的逐 token 行权重上得到 0.845（样本内 0.817），两半相关 0.86，采用。两臂用各自变体的相干行权重 E‖G_c‖²（输入侧草图 G Q，Q 为 ±1/√16），逐 token 行权重一并保存作对照。

| 步骤 | GPU 0 | GPU 1 |
|---|---|---|
| check | 1 个分头诊断 context：草图对 autograd 权重梯度（一个 video 层、一个 action 层），草图文件读回成 H_C，一层 QTIP | baseline QTIP_W2 的闭环启动门 + 1 个 episode |
| sketch | 一半 C80（诊断 context 计 1.5 倍） | 另一半 |
| gate | 读 80 个 json | — |
| quant | 一半层：H_C、H_CA 各一次 QTIP W2；四个权重 × 四个 Hessian 的 proxy | 另一半 |
| eval | QTIP_C_W2 held-out 40×2 + swap | QTIP_CA_W2 |
| loop | QTIP_C_W2 然后 QTIP_CA_W2，每卡 2 个 worker | 同左 |
| final | `results/coherent/RESULTS.zh.md`：门槛、分头占比、proxy 矩阵、四臂 held-out、闭环配对 | — |

## 第二阶段结果（26046240）

闭环 QTIP_C_W2 199、QTIP_CA_W2 204（QTIP_W2 204，QTIP_G_W2 208）。原因：GN 质量在 context 间重尾，H_C 的有效样本只有 5.3/80（2020/2031/2034 三个 context 占 61%），QTIP 在 H_C 上连自身 proxy 都没赢过 H̃ 的权重。详见 `pulled/26046240.pbs101/coherent/RESULTS.zh.md`。

## 第三阶段：舍入阶段迁移到 GPTVQ、QuIP#（`host.json` 的 `hosts`）

同一份第一阶段 H̃，不重采探针，不改宿主代码，每层种子与 baseline 相同。

| 步骤 | GPU 0 | GPU 1 |
|---|---|---|
| check | 每个宿主：最小层在 H_U 上复现 baseline 权重（相对差 ≤ 1e-3），再在 H̃ 上跑一次；加载模型，装 baseline GPTVQ_W2，1 个 held-out endpoint | baseline QTIP_W2 的闭环启动门 + 1 个 episode |
| quant | 一半层（按 baseline 两个宿主每层秒数之和均分），GPTVQ W2 和 QuIP# W2 都用 H̃ | 另一半 |
| eval | GPTVQ_G_W2、GPTVQ_W2 的 held-out 40×2 + swap | QUIPSHARP_G_W2、QUIPSHARP_W2 |
| loop | GPTVQ_G_W2 然后 QUIPSHARP_G_W2，每卡 2 个 worker | 同左 |
| final | `results/hosts/RESULTS.zh.md`：三个宿主 baseline vs 插件，配对胜负与 McNemar p | — |

CPU 测试（`test_probe.py`、`test_hosts.py`）用 `cputest.pbs` 在 ASPIRE2A CPU 节点上跑：

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --attempt r4 cputest
```

## 第三阶段结果（26050310）

H̃ 不迁移。闭环 baseline → 插件：QTIP 204 → 208（14 胜 / 10 负，p 0.54），GPTVQ 201 → 197（8 / 12，p 0.50），QuIP# 198 → 196（12 / 14，p 0.85）；三宿主合计 603 → 601，34 胜 / 36 负，p 0.90。三个宿主上 H̃ 都把自身 proxy 减半、proxy(H_U) 抬高 36–57%，future NMSE 都变差。C1 在三个宿主上都成立：换回 BF16 future 后 first10 motor 降 24%（QTIP）、36%（GPTVQ）、36%（QuIP#）。详见 `pulled/26050310.pbs101/hosts/RESULTS.zh.md`。

## 门 1：在宿主自己的 W2 模型上取曲率（`gate.json`，`gate_worker.py`，`run_gate.pbs`）

第一阶段在 BF16 点线性化：A 头只有 6.9% 经过 future，βK 只占 1.2%；而 W2 下换回 BF16 future 能去掉 held-out first10 motor MSE 的 44–61%。门 1 把三个宿主的 baseline W2 权重装进模型再跑同一套 GN 探针：A 走模型自己（W2）的 reader；K 先单独跑一遍，用冻结的 BF16 reader（614 个 BF16 master）读 student 自己的 future，得到 c_K = Jᵀg_K；Z 不变。

| 步骤 | 内容 |
|---|---|
| probe（2 卡，按 context 分） | 每个宿主：σ²（baseline 权重每层的均方舍入误差）、K pass、探针；8 个诊断 context 分头反传 |
| select | 每层损伤 σ²·mass（工作点、所有头），取前 16 层；两个线性化点的分头占比、context ESS |
| hessian（2 卡） | 选中层的 H̃_op（第一阶段同形，γ = 0.1，输入取 W2 轨迹）；顺带重算 baseline 的第一个 held-out endpoint 核对 |
| quant（2 卡） | 宿主在 H̃_op 上按 baseline 种子重量化选中层 |
| eval（2 卡） | `GATE_<宿主>_G`（选中层用第一阶段 H̃ 的权重）和 `GATE_<宿主>_O`（用 H̃_op）的 held-out 40×2 + swap，其余层是 baseline |
| final | `results/gate/{summary.json, RESULTS.zh.md, DECISION.json}` |

预先登记的规则：至少 2 个宿主上，O 臂相对 baseline 的 held-out mixed loss 和"future 在 first10 motor MSE 里的占比"（1 − mse_swap/mse）都下降，则提交全量。第一阶段的 H̃ 在三个宿主上这两项都是上升的。

全量（`gate.json` 的 `full`，`run_full.pbs`，g3 4 卡）：三个宿主 614 层的 H̃_op，宿主全部重量化（门 1 的 16 层逐位复用），O 臂 held-out 40×2 和 LIBERO-Plus 280；每个宿主与 baseline 和第一阶段 H̃ 臂配对比较，并做三宿主合计。脚本开头检查 `DECISION.json` 为 go 才继续。

### 门 1 结果（26054103 跑完 probe/select 后在 hessian 步因 bug 停下，26056114 接着跑完）：不推进全量

- **线性化点不改变分头占比。** 在 W2 上，A 经 future 的比例是 QTIP 0.063、GPTVQ 0.071、QuIP# 0.059；BF16 点是 0.069。按 σ² 加权后是 0.17、0.20、0.09，同一组 σ² 放在 BF16 点上是 0.19、0.20、0.10。而 BF16 reader 读 student 自己的 future，motor MSE 达到 student 自身的 35%、48%、60%。可见 GN 在哪个点都低估 future 通路。
- **工作点的 GN 更不稳。** 损伤的 context ESS：QTIP 25.7、GPTVQ 1.1、QuIP# 12.3。GPTVQ 上 context 2035 一个就占了 91% 的质量（V 头 8.3e4，其余 context 只有几百），但它的 motor MSE 只有 0.00058。
- **16 层 held-out**，baseline 重算差为 0：
  - QTIP：H̃_op 让 mixed loss 从 0.00336 降到 0.00327，future 占比从 0.437 降到 0.434，满足规则。
  - GPTVQ、QuIP#：H̃_op 和 H̃ 都让 mixed loss 和 future 占比上升。
  - 所有宿主上，重加权 Hessian 都让这 16 层的 proxy(H_U) 升高 35–50%。
- 结论：1/3，不推进全量。详见 `pulled/26056114.pbs101/gate/RESULTS.zh.md`。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --phase gate submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --phase full submit
```

## C2 连续阶段初测：宿主 W2 上调两侧对角 scale（`tune.json`，`tune_worker.py`，`run_tune.pbs`）

起点是各宿主 A16 baseline 的 W2 权重（H_U，码字冻结），每个目标 Linear 调 FP32 log-scale s（行）、t（列），
W' = bf16(diag(e^s) W diag(e^t))，loss A + λZ + βK，预算照 Q0（C80 一遍，8 次更新各 10 个 context）。
第一次（26057132）用 Adam：LR 5e-4 时变化被 BF16 舍入吃掉，2e-3 起 QTIP / QuIP# loss 上升 34–69%，三宿主都没过步长检查。
改成按梯度比例的步长 δ = −η g / max|g|。g2 三卡排队太久，拆成按宿主的单卡 g1 作业（`tune.json` 的 `split`，`race.py` 竞速）。

### 结果：QTIP 跑完（26059212），GPTVQ 停在第 1 次更新后（26059859，用户叫停），QuIP# 没跑

| | QTIP baseline U | QTIP 调后 T（η = 0.01） |
|---|---:|---:|
| C80 mixed loss | 0.002539 | 0.002229 |
| C80 future NMSE | 0.03859 | 0.03837 |
| held-out mixed loss | 0.00336 | 0.00310（0.923×） |
| held-out first10 motor RMSE | 0.0324 | 0.0279 |
| held-out future NMSE | 0.05325 | 0.05328 |
| held-out 换上 BF16 future 后 motor RMSE | 0.0246 | 0.0180 |
| future 在 motor MSE 里的占比 | 0.44 | 0.60 |

- 动作变好、future 不动：按预先登记的规则 QTIP 不满足（future NMSE 没降）。增益全在动作一侧，future 占比反而升高，和 C1 要的方向相反。
- 有效参数极少：δ = −η g / max|g| 下越过 BF16 阈值的 scale，QTIP η = 0.01 时 9 个、GPTVQ η = 0.02 时 100 个（共 388 万个）。
  梯度集中在条件层：`video_expert.time_projection.1` 占 |g|² 的 29% / 25%（QTIP / GPTVQ），`action_expert.action_encoder` 19% / 19%，
  其次 `proprio_encoder`、time embedding。实际调的是十几到一百个调制条件的 scale，不是视频专家主体的量化误差。
- proxy(H_U) 不变（0.0058252 → 0.0058253）：对角 scale 从 H_U 下的最优点出发，主体层的舍入噪声它修不动。
- GPTVQ 步长检查：η 0.005 / 0.01 / 0.02 / 0.05 的实际/预测 0.27 / 0.48 / 0.60 / 0.39，选 0.02。

## C2 v2 门：低秩支路上比较联合目标与解耦目标（`lowrank.json`，`lowrank_worker.py`，`run_lowrank.pbs`）

问题：调的参数在每一层都有容量时，C1 推出的目标能不能把 future 修好？把动作项解耦以后，能不能比联合目标修得更多？
参数化与模型、宿主无关：614 个目标 Linear 每个都挂同样的 rank-4 BF16 支路 y = xWᵀ + b + (xVᵀ)Uᵀ（每层 +16·4·(m+n) bit），
宿主 W2 权重不动。Cosmos-Policy 这类共享 DiT 也能照挂。

| 臂 | 目标 | 说明 |
|---|---|---|
| E | 不训练 | EoRA 式闭式初始化 L = [ΔW C]_r C⁻¹（C Cᵀ = 阻尼后的 H_U），作为已有工作的参照 |
| J | A + λZ + βK | 联合目标，从 E 出发 |
| D | A_tf + λZ + βK | 解耦目标，从 E 出发：A_tf 是量化模型读 BF16 future 时的动作误差（C1 的 swap 路径），动作一侧不会因为迁就坏掉的 future 得到奖励 |

A_tf 在 Cosmos-Policy 上的对应做法：把 BF16 的 future slot 放进量化模型的 solver，读出 slot 4 的动作（`cosmos-policy-vqwam-20261024/reader.py` 做的是反方向）。
更新按张量做相对步长 P ← P − η‖P‖ g/‖g‖，每层都按自己的范数走一步，避免对角 scale 那次把整步都花在少数条件层上。
η 在第 1 批上按步长规则从 {0.01, 0.02, 0.05, 0.1} 里选，然后做 8 次更新（C80 跑一遍，和 Q0 的预算一样）。
判定（预先登记）：一臂通过需同时满足 held-out future NMSE ≤ 0.9 × U、mixed loss ≤ 0.97 × U，并且这两项都低于 E。D 通过就上闭环。
“D 的 future 比 J 低、且 mixed 不比 J 高”算作解耦增量，只报告，不作为门槛。
单卡 g1（`fwlr1`，两臂依次跑，6 h）和双卡 g2（`fwlr2`，两臂各占一卡，4 h）两个作业竞速，谁先开跑留谁（`race.py`）：26060556（g1）先开跑。

**事后修改的步长网格（J 的步长是在扩展网格上选的）。** 26060556 在原网格 {0.01, 0.02, 0.05, 0.1} 上做步长检查：
η = 0.01 时，J 的实际下降是一阶预测的 0.45（0.001056 / 0.002352），D 是 0.51（0.000961 / 0.001893），
更大的 η 两臂的实际下降都很小甚至为负。于是 J 按规则被跳过，D 用 0.01 训练，这样就没法比较 J 和 D，而这正是这个门要回答的问题。
经用户同意：D 不停，继续跑完；网格往小处扩展到 {0.0025, 0.005, 0.01, 0.02, 0.05, 0.1}，
J 单独补跑（g1 作业 `fwlj1`），仍按 0.5 规则选 η。扩展后 D 按规则选出的仍是 0.01，因为 0.01 本来就是 D 能通过的最大值。
J 在 26060556 里留下的两个文件改名保留为 `step.r0_26060556.json` 和 `arm.r0_26060556.json`。E、J、D 都完成后，按原判据出结果。

### 结果：E、D 来自 26060556，J 来自 26061075（J 的 η = 0.005 是在扩展网格上选的）：两臂都不通过，不上闭环

| | bpw | C80 mixed | C80 future NMSE | held-out mixed | first10 motor RMSE | future NMSE | future 占比 | gripper RMSE | 换 BF16 future 后 motor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U（QTIP W2） | 2.0159 | 0.00254 | 0.0386 | 0.00336 | 0.0324 | 0.0532 | 0.437 | 0.0860 | 0.0246 |
| E（EoRA，不训练） | 2.0571 | 0.00220 | 0.0387 | 0.00306（0.908×） | 0.0264 | 0.0535（1.005×） | 0.666 | 0.0276 | 0.0157 |
| J（η = 0.005） | 2.0571 | 0.00159 | 0.0294 | 0.00291（0.864×） | 0.0246 | 0.0523（0.982×） | 0.712 | 0.0288 | 0.0135 |
| D（η = 0.01） | 2.0571 | 0.00147 | 0.0267 | 0.00285（0.846×） | 0.0238 | 0.0518（0.972×） | 0.723 | 0.0294 | 0.0128 |

held-out future NMSE / mixed loss 按 suite（各 10 个 context）：

| suite | U | E | J | D |
|---|---:|---:|---:|---:|
| libero_spatial | 0.0505 / 0.00350 | 0.0515 / 0.00325 | 0.0527 / 0.00321 | 0.0518 / 0.00310 |
| libero_object | 0.0597 / 0.00361 | 0.0620 / 0.00326 | 0.0591 / 0.00301 | 0.0599 / 0.00301 |
| libero_goal | 0.0494 / 0.00302 | 0.0497 / 0.00275 | 0.0501 / 0.00273 | 0.0484 / 0.00262 |
| libero_10 | 0.0534 / 0.00333 | 0.0509 / 0.00295 | 0.0473 / 0.00267 | 0.0470 / 0.00266 |

- 判定：J future 0.982×、mixed 0.864×，D future 0.972×、mixed 0.846×，两臂都低于 E，但 future 都远没到 0.9×，都不通过。
  解耦增量按规则算“是”（D 的 future 和 mixed 都低于 J），但差得很小（future 1%，mixed 2%），J 的 η 只有 D 的一半，
  而且按 suite 看不一致（libero_object 上 J 的 future 比 D 低），不能算作解耦有效的证据。
- 容量问题解决了，泛化没有：C80 上 future NMSE 从 E 的 0.0387 降到 J 0.0294（−24%）、D 0.0267（−31%），held-out 上只降 2–3%。
  训练只过一遍 C80（每个 context 只用于一次更新），所以这不是反复拟合同一批样本：在 C80 上修好的 future 迁移不到 held-out 的 context。
  按 suite 看，held-out future 真正下降的只有 libero_10（0.0534 → 0.047，−12%），其余三个 suite 在 ±3% 以内，有的还上升。
- E 本身修的是动作：不训练、只多 0.041 bpw，gripper RMSE 0.086 → 0.028，mixed 0.908×，future 不动（1.005×）。
  训练再把 mixed 从 0.908× 推到 0.846×，future 占比从 0.44（U）升到 0.67（E）、0.72（D）。
- 梯度仍然集中在条件层：`video_expert.time_projection.1` 占第 1 批 |g|² 的 55%（J）/ 60%（D），其次 `action_expert.head`、`proprio_encoder`。
  按张量相对步长让每层都走了一步，但一阶下降主要来自这几层。
- 参照：Q0（25921795）在 VQ 码本 / gain 上用 reader KL 调过，held-out future 降了 13%。低秩支路在同样的预算下只降了 3%。

### 迁移诊断：修复跟着什么走（`lowrank.json` 的 `diag`，作业 `fwld1`）

不需要新作业就能排除的一种解释是“C80 和 held-out 分布不同”。两者是同样的 40 个任务：C80 每个任务取 2 个 demo 快照，held-out 取 1 个，
demo 互不重叠，相位位置（0.2 / 0.5 / 0.8）也相同。训练只过一遍 C80，所以第 k 批在第 k 次更新之前是没见过的数据。
J 在这时的 loss（A + λZ，和 E 的 C80 fit 同一个量）只有 E 在同一批上的 0.93–1.07 倍；训完以后，同一批降到 0.64–0.77 倍；
第 1 批在之后又经过 7 次别的更新，仍然保持在 0.68 倍：

| 批 | E 的 mixed | J 更新前（这批没见过） | 比 | J 训后 fit | 比 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.00262 | 0.00262 | 1.00 | 0.00178 | 0.68 |
| 2 | 0.00203 | 0.00189 | 0.93 | 0.00134 | 0.66 |
| 3 | 0.00212 | 0.00208 | 0.98 | 0.00162 | 0.76 |
| 4 | 0.00192 | 0.00198 | 1.03 | 0.00123 | 0.64 |
| 5 | 0.00311 | 0.00305 | 0.98 | 0.00236 | 0.76 |
| 6 | 0.00224 | 0.00240 | 1.07 | 0.00173 | 0.77 |
| 7 | 0.00210 | 0.00201 | 0.96 | 0.00157 | 0.75 |
| 8 | 0.00149 | 0.00141 | 0.95 | 0.00110 | 0.74 |

每一批的改进只留在这批自己身上，几乎不惠及没见过的 context，同分布也一样。D 在 C80 上逐批的 future 比值是 0.60–0.87，同一个模式。
所以要回答的问题变成：每个 context 记住的是它的场景，还是它那一条 sampler seed 的去噪轨迹？两者需要的修法不一样。

单卡 g1 作业 `fwld1` 跑两件事，不训练：
1. **换 seed**：先用 BF16 生成 C80 在 seed 1 下的 teacher（缓存只有 seed 0）。生成前，用同样的方法重生成前 2 个 context 的 seed 0 teacher，
   和缓存比 future NMSE 要 ≤ 1e-3。然后在同样 80 个 context、没训练过的噪声上测 E、J、D 的 fit。
   保留比例 ρ = (1 − r₁) / (1 − r₀)，r = 臂 / E 的 future NMSE。
2. **逐样本梯度**：在 E 处，对前 4 个训练批次的 40 个 context × seed 0/1 共 80 个样本，求 D 的目标的梯度，按路径拆开：
   F = λZ + βK 经学生 query 回传（future 一侧），A = A_tf 经读 teacher future 的学生回传（动作一侧）。
   每个张量存 160 个向量的 Gram 矩阵。由 Gram 矩阵得出：
   - 按组（视频 / 动作专家的主体和时间条件）、按路径的余弦：同 context 两个 seed、同 suite、其他；
   - 一批样本的 |平均梯度|² / 平均 |梯度|²；
   - 用 D 的逐张量相对步长（η = 0.01）在一批上走一步，各类样本的一阶下降与这批自身之比 T_seed、T_ctx。
   核对：第 1 批的 Σ‖P‖‖ḡ‖ 要和 step.json 的 predicted_per_eta 相对差 ≤ 1%。

预先登记的判读：
- D 的 ρ（future）≥ 0.5：修复跟着 context 走。如果同时 T_ctx < 0.2，说明每个 context 的修复是各自的，
  下一步是加标定 context，或者把参数化限制在共享的方向上。
- ρ ≤ 0.25：修复跟着噪声轨迹走，目标拟合的是单条去噪轨迹。下一步是对多个 sampler seed 取平均，或者改用逐步 teacher forcing 的目标。
- ρ 介于两者之间：两种成分都有。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --phase lowrankd submit
```

### 迁移诊断结果（26063678）：future 一侧的修复 2/3 跟着去噪轨迹、1/3 跟着 context，通用部分约 0.2%

核对全部通过：重生成的 seed 0 teacher 和缓存逐位相同（future NMSE 0，动作最大差 0）；
第 1 批 Σ‖P‖‖ḡ‖ = 0.18943，step.json 是 0.18934（+0.05%）；80 个样本两条路径的回放差都是 0。
报告：`pulled/26063678.pbs101/lowrank/diag/RESULTS.zh.md`。

**换 seed（同样 80 个 C80 context，没训练过的噪声）：**

| 臂 | future NMSE s0 | s1 | s0 / E | s1 / E | ρ future | ρ mixed |
|---|---:|---:|---:|---:|---:|---:|
| E | 0.0387 | 0.0402 | — | — | — | — |
| J | 0.0294 | 0.0367 | 0.760 | 0.913 | 0.36 | 0.58 |
| D | 0.0267 | 0.0360 | 0.690 | 0.896 | 0.34 | 0.67 |

按 suite 的 D future / E，s0 → s1：spatial 0.63 → 0.74，object 0.81 → 1.07，goal 0.63 → 0.90，libero_10 0.70 → 0.85。
libero_object 换了噪声后比 E 还差。

**E 处的逐样本梯度（D 的目标，40 context × 2 seed）：**

| | F（future 一侧） | A（动作一侧） |
|---|---:|---:|
| 余弦：同 context 两个 seed | 0.330 | 0.836 |
| 余弦：同 suite 不同 context | 0.013 | 0.096 |
| 余弦：其他 | 0.009 | 0.052 |
| 一步的一阶下降 / 这批自身：同 context，seed 1（T_seed） | 0.309 | 0.898 |
| 同上：其他 context，seed 0（T_ctx） | 0.100 | 0.417 |
| 同上：其他 context，seed 1 | 0.021 | 0.423 |

- F 的梯度能量 69% 在视频专家的时间条件层，30% 在视频专家主体；但按 D 的逐张量相对步长，一阶下降的 94% 来自视频专家主体。
  A 的能量 88% 在动作专家主体；动作专家的时间条件层跨 context 余弦 0.34，是唯一明显共享的一组。
- 一批 10 个 seed 0 样本的 |平均梯度|² / 平均 |梯度|²：F 0.096–0.134，A 0.105–0.163，总体 0.102–0.132；
  互不相关时这个值是 0.10。

**拆分。** 把每个样本的梯度写成 通用 + 本 context + 本轨迹 三部分，用余弦和 T 反推各部分占的能量
（通用部分 s 由“其他 context、seed 1”的 T 得出：T = s / (s + (1 − s)/10)）：

| | 本轨迹 | 本 context | 通用 |
|---|---:|---:|---:|
| F（future 一侧） | 67% | 33% | 0.2% |
| A（动作一侧） | 16% | 77% | 7% |

**“其他 context”在 seed 0 和 seed 1 上的差别（4 批都是 0.07–0.14 对 −0.01–0.04）是标定集的痕迹。**
seed 由 case 编号决定（base + 2·case + i），不同 context 之间没有共用的噪声。
但 QTIP 宿主所用的 H_U 和 E 的 EoRA 初始化，都来自 C80 在 seed 0 下的 BF16 query（A16 baselines 的 `prep_worker.py`），
所以 seed 0 的 C80 轨迹正是量化器拟合过的样本，它们之间共享约 1.1% 的能量。
held-out 既换 context 又换噪声，对应的是“其他 context、seed 1”。

**和实测对得上。** 每次更新对 held-out 的一阶增益约为这批自身的 0.021，8 次更新累计 8 × 0.021 = 0.17。
训过的批次自身是 1，再加上其余 7 次更新对它的迁移 0–0.10，所以 held-out / C80 应在 0.10–0.17 之间。
实测 D 是 (1 − 0.967) / (1 − 0.690) = 0.106。

**判读（预先登记）。** D 的 ρ（future）= 0.34，落在 0.25 和 0.5 之间，两种成分都有。
T_seed（F）= 0.31 ≥ 0.2，同 context 跨 seed 共享；T_ctx（F）= 0.10 < 0.2，跨 context 不共享。

**迁移不过去的原因。** 以 C80 future NMSE 为目标时，每个样本的 future 梯度里，2/3 是学生沿这一条噪声轨迹偏离 teacher 的部分，
1/3 是这个 context 自己的部分，换 context 又换噪声后仍然成立的只有 0.2%。
一次更新 10 个 context 时，future 一侧的一阶效果 98% 落在这批样本自己身上。
所以只训一遍、每个 context 只见一次，也照样是记忆。
动作一侧不同：通用部分 7%，是 future 的 30 倍，集中在动作专家的时间条件层。
这解释了为什么至今所有在权重侧做的调整，在 held-out 上都只改善了动作：tune.json 的对角缩放、E 的初始化、J 和 D 的训练都是这样。

**对下一步的含义。**
- 只对多个 seed 取平均：能去掉本轨迹的 67%，但剩下的是 本 context 33% 对 通用 0.2%。
  要让一步的一半作用在通用方向上，每次更新需要约 150 个 context。
  整条 rollout 的梯度每个样本约 25 s，150 个就是一小时一步，在 C80 规模上不可行。
- 逐步 teacher forcing：给学生 teacher 在每一步的状态，让它匹配 teacher 在这一步的输出，不再沿学生自己的轨迹累积偏离。
  这样能去掉本轨迹那一部分，每个样本也只需要一次 denoiser 前向和反向，几千个 (context, 步, 噪声) 样本的成本可以接受。
  它的通用部分有多大，要像这次一样在 E 处用 Gram 矩阵量出来：跨 context、跨 seed 的 T ≥ 0.2 才值得在这个目标上搭 C2。

### 逐步 teacher forcing 的 future 目标（`lowrank.json` 的 `tfdiag`，作业 `fwlt1`）

**目标。** Fast-WAM-IDM 的视频阶段是 10 步 Euler，每步 x ← x + Δ·v，之后把第 0 帧重置为观测帧。
在每个样本上先用 BF16 跑一遍视频阶段，记下每一步读入的状态 x_i^T、timestep、速度 v_i^T 和 Δ_i，在动作阶段之前停下。
学生（E）在同样的 x_i^T 上算速度 v_S，用自己的 context 和同样的 timestep。把每步的位移差加到 teacher future 上：

x~ = x_T + Σ_i Δ_i (v_S(x_i^T) − v_i^T)（只加在第 1 帧以后，FP32）

目标是 tf = λZ(x~) + βK(x~)，Z 是相对 x_T 的 future NMSE，K 是冻结的 BF16 reader 在 x~ 上的 KL。
和 rollout 的目标 ro = λZ + βK（diag 的 F 路径）相比，tf 不让学生沿自己的轨迹累积偏离，因此按构造就去掉了“本轨迹”那部分。
任何生成 future 的 WAM 都能这样做：先想后做的 IDM，以及联合去噪的 DiT 都一样。

**样本和量。** 和 diag 相同：E，前 4 个训练批次的 40 个 context × seed 0/1。每个样本算 tf 梯度和 ro 梯度（ro 重算，fwld1 只存了 Gram），
每个张量存 160 个向量 [tf, ro] 的 Gram 矩阵。
- 两个目标在 E 处的值；tf 位移 x~ − x_T 和 rollout 位移的余弦、长度比；逐步的位移能量，以及它和 rollout 位移的余弦。
  这些量说明 rollout 的 future 误差里，有多少是各步局部误差直接相加得到的，有多少来自后续步的传播。
- 两条路径各自的余弦：同 context 两个 seed / 同 suite / 其他；同一样本 tf 和 ro 之间的余弦。
- 一步（D 的逐张量相对步长，η = 0.01，方向 = 一批 seed 0 样本的 tf 或 ro 平均梯度）：
  - T：对自己目标的一阶下降 / 对这批自身的下降；
  - G：对 ro 目标的一阶下降 / ro 那一步对这批自身的下降。

**核对。** 以下任一不满足，作业即报错：
- 记录的视频阶段最后一步要和 teacher future 一致（NMSE ≤ 1e-6），每个记录的状态要等于前一步截断后的结果（差 0）；
- 装 BF16 权重做 teacher forcing，得到的速度要等于记录值（差 0）；
- 重算的 ro 梯度要复现 diag/gram.pt 的 F 块（相对差 ≤ 1e-3）。

第一个样本上的步长核对（η = 0.0025）只报告，不作门槛。

**预先登记的判读。**
- go：T（tf 步，其他 context，seed 1）≥ 0.2，并且 G（tf 步，其他 context，seed 1）≥ 0.06。
  0.06 的依据：fwld1 里 ro 那一步对这类样本的作用是 0.021，D 训练 8 次后 held-out future 降了 3.3%。
  3 倍就是 0.06，按同样的步长训练 8 次正好到 0.9× 的线。
  go 的下一步是在 A_tf + λZ(x~) + βK(x~) 上训练一个臂。
- 共享但 G 不够：局部误差是共享的，但其他 context 的 rollout future 误差不是由它构成的，后续步的传播占主导。只做 teacher forcing 到不了 held-out future。
- 不共享：局部误差同样按 context 各自不同。经权重、靠 C80 规模的标定集修 future，这条路走不通。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --phase lowrankt submit
```

### 逐步 teacher forcing 的结果（26065573，`fwlt1r1`）：通用部分很大，但几乎不作用于 held-out 类样本的 rollout future，no-go

**第一次提交 26065237 失败。**
- 原因：FastWAM 每步 `step` 之后在同一个张量上原地把第 0 帧重置成观测帧，sampler 拿到的 future 因此已经是重置后的。
  我的录制在重置前 clone，第 0 帧不对，x~ 交给冻结 reader 时没通过它的第 0 帧核对。
- 修复：future 改为观测帧加最后一步的第 1 帧以后；状态核对加上第 0 帧；测试里的假模型和假 reader 照真实行为改了。
- 失败那次的目录改名为 `results/lowrank/tfdiag.r0_26065237` 保留。

**核对全部精确为 0。**
- 80 个样本的 teacher 状态：future NMSE 0，第 0 帧差 0，状态衔接差 0。
- 装 BF16 权重做 teacher forcing，速度差 0。
- 重算的 ro 梯度复现 fwld1 的 Gram，相对差 0。
- 回放差 0。
- 步长核对：实际下降 / 一阶预测 = 0.91。

报告：`pulled/26065573.pbs101/lowrank/tfdiag/RESULTS.zh.md`。

**两个目标在 E 处的值（80 个样本平均）：**

| | Z | K | λZ + βK |
|---|---:|---:|---:|
| ro（自己的 rollout） | 0.0368 | 0.0168 | 0.00166 |
| tf（逐步 teacher forcing） | 0.0407 | 0.0093 | 0.00178 |

- tf 位移 x~ − x_T 和 rollout 位移的余弦为 0.65（各样本 0.37–0.86），长度比 1.19。
- 逐步位移的能量从第 1 步的 3.2e-4 升到第 10 步的 1.3e-3。
- 各步方向高度一致：Σ 的能量是各步能量之和的 7.1 倍，达到完全同向时的 77%。
- 可见 W2 的速度误差是一个每步方向几乎相同的系统偏差，rollout 的 future 误差大部分就是它的累加，不是轨迹混沌。

**逐样本梯度：**

| | tf | ro |
|---|---:|---:|
| 余弦：同 context 两个 seed | 0.875 | 0.330 |
| 余弦：同 suite 不同 context | 0.598 | 0.013 |
| 余弦：其他 | 0.590 | 0.009 |
| 一批 10 个样本的 \|ḡ\|² / 平均 \|g\|² | 0.52–0.67 | 0.10–0.13 |

- tf 梯度能量的 86% 在视频专家的时间条件层，只有 6 个张量，跨 context 余弦 0.66；视频专家主体占 13.5%，跨 context 余弦 0.08。
- 同一样本 tf 和 ro 梯度的余弦是 0.24，不同样本之间 0.03。

**走一步（η = 0.01）：**

| 样本 | T（tf 步，tf 目标） | G（tf 步，ro 目标） | G（ro 步，ro 目标） |
|---|---:|---:|---:|
| 这批自身 | 1 | 0.411 | 1 |
| 同 context，seed 1 | 0.871 | 0.196 | 0.311 |
| 其他 context，seed 0 | 0.503 | 0.048 | 0.100 |
| 其他 context，seed 1 | 0.521 | 0.030 | 0.021 |

**判读（预先登记）：**
- T（tf，其他 context，seed 1）= 0.52 ≥ 0.2，共享。
- G（tf，其他 context，seed 1）= 0.030 < 0.06，不够，ro 自己那一步是 0.021。
- no-go：局部误差有很大的共享部分，但其他 context 的 rollout future 误差几乎不随它变。单靠 teacher forcing 到不了 held-out future。

**事后分析（`tfpool.py`，CPU 作业 26065919，读 tfdiag/gram.pt；不在预先登记之内）。** 全张量的 G 复现 summary，差 0。
- 按组限制这一步，其他 context、seed 1 上的 G：

  | 步只在 | tf | ro |
  |---|---:|---:|
  | 全部 1228 个张量 | 0.030 | 0.021 |
  | 视频专家主体（606） | 0.020 | 0.023 |
  | 视频专家时间条件（6） | 0.011 | −0.001 |

  时间条件那 6 个张量承载了 tf 的共享部分，单独走一步能拿到 ro 整步的一半。但在逐张量相对步长下，整步主要由主体的 606 个张量决定。
- 方向由更多样本平均：从其他 1–3 批的两个 seed（20 / 40 / 60 个样本）取平均梯度，测留出那一批，同样按 G 归一：

  | 方向 | 样本数 | 留出批 seed 0 | 留出批 seed 1 |
  |---|---:|---:|---:|
  | ro | 20 / 40 / 60 | 0.077 / 0.095 / 0.106 | −0.011 / −0.016 / −0.019 |
  | tf | 20 / 40 / 60 | 0.045 / 0.058 / 0.065 | 0.024 / 0.026 / 0.028 |

  - ro 方向样本越多，对 seed 0 越好，对新噪声越差。C80 的 seed 0 轨迹是 H_U 和 EoRA 的标定集，样本越多，ro 越是在拟合这个标定集共有的成分，它不迁移到新噪声。
  - tf 方向在新噪声上一直为正，但 60 个样本时停在 0.028。
  - 在 D 的步长下，两个目标都没有一个方向能在“其他 context、新噪声”上到 0.06。

**结论。**
- 在 C80 规模上，用 rollout 或逐步 teacher forcing 的 future 目标去调 W2 权重，对没见过的 context 和噪声只有约 1.4 倍于 D 的一阶作用。
- 已知可迁移的有两处：
  - 动作一侧：fwld1 测得通用部分 7%。
  - 视频时间条件层上那个系统性的局部误差：跨 context 余弦 0.66，但对 rollout 的作用很小。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --phase lowrankt --attempt r2 tfpool
```

### 时间条件层归因（`lowrank.json` 的 `tcond`，作业 `fwtc1`）

**依据。** tfdiag 里 tf 梯度能量的 86% 和它的共享部分，都在视频专家的 3 个时间条件 Linear 上：`time_embedding.0`、`time_embedding.2`、`time_projection.1`。
这些层的输入只是 timestep 的正弦编码：第 1 帧以后的 token 用 t，第 0 帧用 0，和 context 无关。
采样固定为 10 步时，每个专家这些层的输出只有十来种取值，所有 context 都一样。
所以把它们换回 BF16 主权重，输出就和 BF16 模型完全一致。这等价于按 timestep 存一张输出表，存储代价约 +0.0005 bpw。
要回答的是：W2 的 held-out future 误差里，有多少来自这条路径。

**臂（都在 held-out 40×2 + C1 swap 上评测）：**
- `QTIP_TV_W2`：U（QTIP W2），视频专家时间条件的 3 个 Linear 换回 BF16；
- `QTIP_TA_W2`：U，动作专家时间条件的 3 个 Linear 换回 BF16；
- `QTIP_TVA_W2`：U，两者都换；
- `QTIP_LRE_TVA_W2`：E（EoRA），两者都换，这 6 个 Linear 的低秩支路置零。

**核对。** 在 BF16 下取两个不同 context、不同 seed 的样本（2000/s0，2047/s1），6 个时间条件 Linear 的输入要逐次完全相同（差 0），否则作业报错。
不同输入行数就是一张表要存的 timestep 数。

**bpw 两种算法：**
- BF16 权重：视频专家这 3 个 Linear 有 66.8M 参数，约 +0.16 bpw；
- 按 timestep 存表：约 +0.0005 bpw。

**预先登记的判读（r = U_TV 的 held-out future NMSE / U 的）：**
- r ≤ 0.9：时间条件路径承载了至少十分之一的 W2 future 误差，而且修复精确、与 context 无关。C2 v3 以按 timestep 的条件表为基础，下一步跑 U_TV 对 U 的闭环。
- 0.9 < r ≤ 0.97：只占一小部分，和 E_TVA 一起看再定。
- r > 0.97：tfdiag 找到的共享局部误差不是 W2 future 误差的来源。这个宿主上经权重修 future 已经到头，C2 转向动作一侧。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --phase lowrankc submit
```

### 时间条件层归因结果（26066073）：r = 0.999，时间条件路径几乎不承载 W2 的 future 误差

**输入核对通过。** 两个不同 context、不同 seed 的样本，6 个时间条件 Linear 的输入逐次完全相同，差为 0。
视频专家每次 query 调用 11 次（10 步加 prefill 的 t = 0），有 11 种输入；动作专家调用 10 次，有 10 种。
报告：`pulled/26066073.pbs101/lowrank/tcond/RESULTS.zh.md`。

| 臂 | bpw（BF16 权重 / 存表） | future NMSE | / 基臂 | mixed | / 基臂 | first10 motor | gripper |
|---|---:|---:|---:|---:|---:|---:|---:|
| U（QTIP W2） | 2.0159 | 0.0532 | — | 0.00336 | — | 0.0324 | 0.0860 |
| U + 视频时间条件 BF16 | 2.1713 / 1.9941 | 0.0532 | 0.999 | 0.00337 | 1.002 | 0.0325 | 0.0788 |
| U + 动作时间条件 BF16 | 2.0335 / 2.0135 | 0.0532 | 1.000 | 0.00334 | 0.993 | 0.0320 | 0.0881 |
| U + 两者 BF16 | 2.1889 / 1.9917 | 0.0532 | 0.999 | 0.00334 | 0.994 | 0.0321 | 0.0809 |
| E（EoRA） | 2.0571 | 0.0535 | — | 0.00306 | — | 0.0264 | 0.0276 |
| E + 两者 BF16 | 2.2297 / 2.0326 | 0.0528 | 0.986 | 0.00302 | 0.987 | 0.0263 | 0.0275 |

存表的 bpw 低于基臂：按 timestep 存表以后，推理时不再需要这些层的权重，它们原来的 W2 比特被省掉，表本身只有几十万个值。

**判读（预先登记）。** r = 0.999 > 0.97，可忽略。
- 时间条件层把输出完全恢复到 BF16，held-out future NMSE 不变，mixed 和 motor 也几乎不变。
- tfdiag 里 tf 梯度能量集中在这些层，反映的是 loss 对它们的敏感度：调制量影响每个 token。
- 但 W2 在这些层上的量化误差本身对 future 几乎没有贡献。共享的局部误差方向和误差的实际来源是两回事。

**到这里为止的结论。** 在 QTIP W2 宿主上：
- 经权重修 future：对角缩放、低秩支路（联合 / 解耦）、逐步 teacher forcing、时间条件精确化，都没能在 held-out 上把 future 降到 0.9× 以下，最多降 3%（D）。
- 可迁移的改进一直在动作一侧：E 不训练 mixed 就到 0.908×，gripper 0.086 → 0.028；fwld1 测得动作一侧的梯度有 7% 是通用的。
- 按预先登记，C2 转向动作一侧。

### J 的 LIBERO-Plus 280 闭环（`lowrank_loop.py`，`run_lowrank_loop.pbs`，作业 `fwljl` 26066178，g2）

J 没过门，这次闭环是用户点名要跑的，只跑 J 这一臂。

- **运行时**：沿用 A16 baselines 的 loop_worker（BF16 条件缓存 + CUDA Graph，T5 缓存）。宿主装 QTIP_W2 的稠密 W2 权重，614 个 Linear 再挂 J 的 rank-4 BF16 因子，支路算法与 lowrank_worker 一致。
- **安装检查**：4 个 worker 全部通过，相对差都是 0。
  - 只装宿主权重时，case 2000 的 first10 motor RMSE 为 0.0313032，与 QTIP_W2 闭环的锚点逐位相同。
  - 挂上支路后为 0.0208447，与 J 在 C80 上的 fit 行逐位相同。
- **其他检查**：cache + graph 对 eager、T5 缓存对完整 T5 的差都是 0。

| 臂 | bpw | 成功/280 | 对 QTIP_W2 胜/负 | 对 BF16 胜/负 | policy call p50 |
|---|---:|---:|---:|---:|---:|
| **QTIP_LRJ_W2（J）** | 2.057 | **209** | 13 / 8（McNemar p 0.38） | 6 / 13 | 0.745 s |
| QTIP_W2 | 2.016 | 204 | — | 7 / 19 | 0.505 s |
| GPTVQ_W2 | 2.063 | 201 | | 10 / 25 | 0.506 s |
| QuIP#_W2 | 2.010 | 198 | | 4 / 22 | 0.518 s |
| QuIP#_W3（W3 最好） | 3.010 | 205 | | 5 / 16 | 0.504 s |
| BF16 | 16 | 216 | | | |

逐组看 J − QTIP_W2：
- 按 suite：libero_10 +5（48 → 53），spatial +1，goal 0，object −1。
- 按扰动：robot_initial_states +3，background_textures +2，sensor_noise +2，camera_viewpoints −1，objects_layout −1，其余持平。

**判读。**
- J 是目前所有 A16 W2/W3 臂里最高的。与 BF16 的差距从 12 缩到 7。
- 对 QTIP_W2 净胜 5，但 p 0.38，280 个 episode 上不显著。
- 增益主要落在长程的 libero_10。这与 held-out 上的情况一致：改善集中在动作一侧（gripper 0.086 → 0.029，first10 motor 0.0324 → 0.0246），future 只降了 1.8%。
- 锚点 future NMSE 0.0244（QTIP_W2 为 0.0450），但 case 2000 是 C80 的拟合样本，不能当泛化证据。
- policy call 慢了 48%。原因是支路没有融合，每次前向多出 614 对 FP32 小矩阵乘；这是实现问题，不是 rank 4 本身的开销。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --phase lowrankl submit
```

### 支路提速（`lowrank_loop.py --lr-mode`，`lowrank_bench.py`，作业 `fwljb`，gdev 1 卡）

用户要求在后续闭环之前先把支路的开销降下来。

**耗时构成。** 26066178 每个 episode 83.9 s，QTIP_W2 是 78.4 s。每个 episode 约 24 次 policy call：J 合计约 18 s，QTIP_W2 约 12 s，其余约 66 s 是 CPU 仿真。作业日志里 GPU 大部分时间利用率是 0%。所以支路开销占闭环墙钟时间约 7%，但单次 policy call 慢了 48%。

**三种装法**（`lowrank_loop.py --lr-mode`）：
- `branch`：原样，与 lowrank_worker 逐位一致。x 转 FP32，做两次 FP32 小矩阵乘，结果转回 BF16 再加到输出上。每个 Linear 多出约 5 遍激活大小的显存读写。
- `fused`：先算 BF16 的 x Vᵀ，再用一次原地 addmm 把 (x Vᵀ)Uᵀ 加进 BF16 输出（FP32 累加）。只多读一遍 x、多读写一遍输出。
- `merged`：W' = bf16(W + U V)，之后就是普通 BF16 Linear，和每个 A16 臂一样（所有 baseline 在这个运行时里跑的都是反量化后的稠密 BF16 权重）。零额外开销，误差来自 W' 存成 BF16 时的舍入。
- 另有 `host`：只装 QTIP_W2 权重，作为耗时参照。

**检查。** 每个 worker 依次检查：
1. 只装宿主权重时，复现宿主闭环锚点（≤ 1e-4）。
2. 装原样支路时，复现 C80 拟合行（≤ 1e-4）。
3. fused / merged 还要复现原样支路对锚点的改变量：first10 motor RMSE(fast − branch) / RMSE(branch − host) ≤ 0.05，future 的 ‖fast − branch‖ / ‖branch − host‖ ≤ 0.05。

**测速作业。** 同一张卡上四种装法各起一个进程：走闭环 worker 的 build_runtime（含 T5 缓存、安装检查、cache + Graph 逐位检查），然后在 case 2000 上先热身 5 次，再计时 40 次 infer，每次前后都同步。`--compare` 汇总各装法的耗时（相对 host）和锚点偏差。

**预先登记的选择规则。** merged 过检查就用 merged，否则用 fused，都不过就保留 branch。结果写入 `lowrank.json` 的 `loop.mode`。fused / merged 的闭环目录名为 `<arm>_<mode>`，除了与 QTIP_W2 配对，也与原样支路的闭环配对。

**结果（26068632，A100-40GB，单进程独占）：两个快装法都没过，按规则保留 branch。**

| 装法 | policy call p50 | 相对 host | 锚点 first10 motor RMSE | 动作偏差 / 支路改变量 | future 偏差 / 支路改变量 | 检查 |
|---|---:|---:|---:|---:|---:|---|
| host | 0.497 s | 1.00× | 0.03130 | — | — | 过 |
| branch | 0.703 s | 1.41× | 0.02084 | — | — | 过（宿主、拟合行都逐位一致） |
| fused | 没计时 | — | 0.02067 | 0.041 | **0.056** | 不过 |
| merged | 没计时（权重与 host 一样，耗时应等于 host） | — | 0.02065 | 0.050 | **0.073** | 不过 |

- fused 与 branch 只差舍入：x Vᵀ 存成 BF16，输出只舍入一次。可它的偏差也有 4–6%，与 merged 的 5–7% 同一量级。
- 说明单个锚点 query 经 10 步 video + 10 步 action 采样之后，任何 BF16 级别的扰动都会被放大到支路改变量的约 5%。0.05 这个阈值正好落在舍入噪声上，分不出"实现错了"和"舍入不同"。
- 两个快装法的锚点 RMSE 都比 branch 略低（0.0207 对 0.0208），偏差方向没有系统性变差。
- `--compare` 在缺少未计时装法的结果文件时崩溃（作业 exit 64），已修复：未过检查的装法只记偏差、不计时，测试中加了这种情况。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --phase lowrankb submit
```

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --phase lowrank1 submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --phase lowrank2 submit
```

## EoRA-J 迁移到 GPTVQ W2 和 QuIP# W2（`lowrank.json` 的 `xfer`，`run_lowrank_xfer.pbs`）

用户要求把 J 原样搬到另外两个 W2 宿主上，分别跑 280 闭环。

**原样迁移的含义。** 只换宿主权重，其余全部沿用 QTIP 上的设置：
- rank 4。
- EoRA 初始化：用同一份 H_U、同样的阻尼 0.01，dW = W_master − W_host。
- λ、β 不变。
- 同一步长网格 0.0025–0.1 和 0.5 规则，η 由规则在新宿主上重新选出。
- 在 C80 上更新 8 次。

**臂名与基线：**
- 新臂名为 `GPTVQ_LRE_W2` / `GPTVQ_LRJ_W2`、`QUIPSHARP_LRE_W2` / `QUIPSHARP_LRJ_W2`；初始化文件放在 `lowrank/init_<HOST>`。
- D 不跑。
- U 的 held-out 来自 26050310，U 的 C80 拟合来自 tune 的 `fit_before`。

| 作业 | 队列 | 内容 |
|---|---|---|
| `fwlx1` | gdev 2 卡 2 h | 每卡一个宿主：EoRA 因子，E 的比特、C80 拟合、held-out 40×2 + swap |
| `fwlx2` | gdev 2 卡 2 h | 每卡一个宿主：J 的步长检查、8 次更新、比特、拟合、held-out；然后 `xfinal` 汇总三个宿主（QTIP 作参照）到 `lowrank/xfer/` |
| `fwlxl` | g3 4 卡 4 h | 两个 J 臂并行，各占 2 卡 × 2 worker，闭环 worker 用 `lowrank_loop.py --lr-mode branch` |

**闭环。** 每个闭环 worker 先检查：只装宿主权重时复现宿主自己闭环的锚点，挂上支路后复现 J 的 C80 拟合行（都 ≤ 1e-4），然后做 eager 对 cache + Graph 的逐位检查。配对对象是宿主自己的闭环：GPTVQ_W2 201/280，QUIPSHARP_W2 198/280。汇总写在 `lowrank/xfer/loops.json`，附 QTIP 的 J（209 对 204）作参照。

### 离线结果（E：26068766；J：26068868）

括号里是相对该宿主 U 的倍数。

| 宿主 | 臂 | bpw | η | C80 mixed | held-out mixed | future NMSE | first10 motor | gripper |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| QTIP（参照） | U | 2.016 | — | 0.00254 | 0.00336 | 0.0532 | 0.0324 | 0.0860 |
| | E | 2.057 | — | 0.00220 | 0.908 | 1.005 | 0.816 | 0.321 |
| | J | 2.057 | 0.005 | 0.00159 | 0.864 | 0.982 | 0.758 | 0.335 |
| GPTVQ | U | 2.063 | — | 0.00409 | 0.00587 | 0.0956 | 0.0403 | 0.0365 |
| | E | 2.104 | — | 0.00356 | 0.923 | 0.958 | 0.890 | 0.957 |
| | J | 2.104 | 0.005 | 0.00270 | 0.909 | 0.970 | 0.841 | 1.319 |
| QuIP# | U | 2.010 | — | 0.00310 | 0.00413 | 0.0718 | 0.0311 | 0.0748 |
| | E | 2.052 | — | 0.00296 | 0.959 | 0.988 | 0.927 | 0.620 |
| | J | 2.052 | 0.005 | 0.00221 | 0.969 | 0.995 | 0.916 | 0.719 |

**E（EoRA 初始化）。** 在三个宿主上都有改善，主要在动作一侧。

**J 的步长。** 步长规则在三个宿主上都选了 η = 0.005。实测下降与一阶预测之比分别是 0.75（QTIP）、0.57（GPTVQ）、0.65（QuIP#）。

**J 相对 E。** J 让 C80 mixed 再降 24–28%，但到 held-out 上：
- GPTVQ：mixed 只多降 1.4 个百分点（0.923 → 0.909），motor 降了，gripper 变差（0.957 → 1.319）。
- QuIP#：mixed 0.959 → 0.969，比 E 差；gripper 0.620 → 0.719 也变差。
- QTIP：J 比 E 多降 4.4 个百分点。

也就是说，联合目标在 C80 上的额外拟合在新宿主上几乎不迁移到 held-out。这与 fwld1 的迁移诊断一致：rollout 梯度主要是轨迹和 context 特有的。

### 闭环结果（26069268，g3 4 卡）：一升一降，三宿主合计不显著

**检查。** 8 个 worker 全部通过：
- 只装宿主权重时，锚点 first10 motor RMSE 与宿主自己的闭环逐位相同（GPTVQ 0.0333124，QuIP# 0.0213511）。
- 挂上支路后与 J 的 C80 拟合行一致：GPTVQ 相对差 0，QuIP# 6e-8。
- cache + graph 对 eager、T5 缓存对完整 T5 的差都是 0。

| 宿主 | J 成功/280 | 宿主成功/280 | J − 宿主 | 对宿主胜/负 | McNemar p | J 对 BF16 胜/负 | 宿主对 BF16 胜/负 | policy call p50（J / 宿主） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QTIP（26066178） | **209** | 204 | +5 | 13 / 8 | 0.38 | 6 / 13 | 7 / 19 | 0.745 / 0.505 s |
| GPTVQ | 197 | 201 | −4 | 10 / 14 | 0.54 | 6 / 25 | 10 / 25 | 0.742 / 0.506 s |
| QuIP# | **202** | 198 | +4 | 9 / 5 | 0.42 | 7 / 21 | 4 / 22 | 0.742 / 0.518 s |
| 合计 | 608 | 603 | +5 | 32 / 27 | 0.60 | | | |

逐组看 J − 宿主：

| 组 | QTIP | GPTVQ | QuIP# |
|---|---:|---:|---:|
| libero_10 | +5 | −3 | +2 |
| libero_goal | 0 | +4 | +2 |
| libero_object | −1 | −3 | +1 |
| libero_spatial | +1 | −2 | −1 |
| background_textures | +2 | 0 | +1 |
| camera_viewpoints | −1 | **−4** | +2 |
| language_instructions | 0 | +1 | 0 |
| light_conditions | 0 | −1 | +2 |
| objects_layout | −1 | −1 | 0 |
| robot_initial_states | +3 | 0 | 0 |
| sensor_noise | +2 | +1 | −1 |

**判读。**
- 三个宿主上 J 的净效应都在 ±5 以内，单个宿主和合计都不显著（合计 32 胜 27 负，p 0.60）。
- QTIP 上 libero_10 +5 的增益没有在另外两个宿主上重复：GPTVQ −3，QuIP# +2。各组的符号在宿主之间不一致。
- GPTVQ 上的 −4 有一半来自 camera_viewpoints（18 → 14）。
- 离线指标里，只有 gripper 相对 U 的倍数与闭环增量同序：GPTVQ 1.319 对应 −4，QuIP# 0.719 对应 +4，QTIP 0.335 对应 +5。held-out mixed（0.909 / 0.969 / 0.864）和 first10 motor（0.841 / 0.916 / 0.758）都不同序。只有 3 个点，而且增量都在噪声内，只能当作观察。
- GPTVQ_W2 本身的 gripper RMSE 已经很低（0.0365，QTIP_W2 为 0.0860），J 在它上面把 gripper 拉高到 U 的 1.3 倍。
- policy call 在三个宿主上慢 43–48%，与测速作业一致。新的两条闭环 episode 平均 89 s，宿主为 79–81 s。

**结论。** J 原样迁移后，闭环上相对宿主没有稳定增益。QTIP 上的 +5 是三次里最好的一次，不是可复现的效果。作为 C2 候选，EoRA-J 不能写成"跨宿主即插即用地提升闭环"。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --phase lowrankx1 submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --phase lowrankx2 submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --phase lowrankxl submit
```

## 轨迹相干层目标的回溯检验（`traj.json`，`traj_worker.py`，`run_traj.pbs`，作业 `fwtr1`，gdev）

**依据。** tfdiag 发现 W2 的速度误差在 10 个视频去噪步上几乎同向：累加后的能量是各步能量之和的 7.1 倍，80 个样本都在 6.4–7.5 之间。
如果同一份权重误差每步都重复，一层对 future 的贡献约为每个 token 上的 ΔW Σ_i Δ_i x_i，对应的层目标是：

tr(ΔW H_traj ΔWᵀ)，H_traj = Σ_token (Σ_i Δ_i x_i)(Σ_i Δ_i x_i)ᵀ

H_traj 只用激活，和 H_U 一样稳定，不依赖逐 context 的梯度。这里不重新量化，只检验它能不能给已有 W2 臂的 held-out future NMSE 排对序。
之前已经知道 H_U 排不对：EoRA 专门压低 H_U 下的误差，future 却完全没变。

**收集。**
- 在 H_U 用过的同一批数据（C80 的 seed 0，`runtime.infer`）上，对 306 个视频专家 Linear 收集两个 Hessian：
  - H_traj：去噪步上按 token 跨步累加；
  - H_S：同样的调用、同样的权重 Δ_i²，但不含跨步项，用作对照。
- 每层每次 query 调用 11 次的视为分步层：前 10 次是去噪步，第 11 次是动作阶段的 prefill，不计入。调用次数不是 11 的层，所有调用都不累加地计入。
- 调用次数先用一次 dry query 定下来，之后每次 query 都核对。
- 核对：在 4 层上用全部调用收集的 Hessian，要复现基线 H_U，相对差 ≤ 0.01。
- H_traj 保存到 `results/traj/hessians`，go 之后重新量化时直接用。

**proxy。** 每层 p = tr(ΔW H ΔWᵀ) / tr(W H Wᵀ)，ΔW = W_arm − W_master。每个臂对视频层按权重数加权平均。
已有的 19 个臂都参与：
- QTIP 11 个：U、G、C、CA、GR、T、两个 gate 臂、LRE/LRJ/LRD；
- GPTVQ 4 个、QuIP# 4 个。

**预先登记的判读。** 主判据用 QTIP 的 11 个臂：宿主相同、方法不同，宿主之间的差距帮不上忙。指标是 proxy 和 held-out future NMSE 的 Spearman ρ。
- go：ρ_traj ≥ 0.6，并且 ρ_traj − ρ_U ≥ 0.2。下一步用保存的 H_traj 重新量化 QTIP W2，过 held-out future ≤ 0.9× U 再上闭环。
- 要把功劳归给跨步项，还需要 ρ_traj − ρ_S ≥ 0.1；否则增益只是来自"只保留去噪步的调用"。
- 19 个臂合起来的 ρ，以及对 mixed、first10 motor、闭环的 ρ，只报告不判。

这套代码只上传自己的文件，用自己的提交状态，不改动其他 phase 的源文件。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/traj_control.py cputest
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/traj_control.py submit
```

### 结果（26066187，gdev，42 分钟）：三种 Hessian 的排序完全相同，no-go

**核对。**
- 4 层上用全部调用收集的 Hessian 和基线 H_U 逐位相同（相对差 0）。
- Δ_i 全是 −0.1001。
- 305 层每次 query 调用 11 次。只有 `video_expert.head.head` 调用 10 次，它在 prefill 里不跑，按不分步处理。

报告：`pulled/26066187.pbs101/traj/RESULTS.zh.md`。

**排序（QTIP 的 11 个臂，proxy 对 held-out future NMSE 的 Spearman ρ）：**

| | ρ H_U | ρ H_S | ρ H_traj |
|---|---:|---:|---:|
| future NMSE（主判据） | 0.682 | 0.682 | 0.682 |
| mixed | 0.873 | 0.873 | 0.873 |
| first10 motor | 0.709 | 0.709 | 0.709 |
| 闭环（5 个臂） | 0.447 | 0.447 | 0.447 |

- 19 个臂合起来，future 上 ρ 分别是 0.686、0.686、0.677。
- 判读：ρ_traj − ρ_U = 0 < 0.2，no-go。

**为什么完全一样。** 每层、每个臂，三种 proxy 之间几乎是同一个比例：H_S / H_U ≈ 0.96–0.97，H_traj / H_U ≈ 0.85–0.89，19 个臂都如此。
- 主权重 W 自己的 κ 中位数是 9.70（最大 10），90% 的层 ≥ 8.98。也就是说，同一个 token 位置上，视频专家各层的输入在 10 个去噪步之间几乎不变。
- 只有前 3 个 block 的 cross_attn.o 等 4 层 κ 低于 8。
- 量化误差的 κ 中位数是 8.79，各臂都在 8.69–8.95 之间，彼此几乎没有差别。

所以 H_traj 约等于 10 × H_S 约等于常数 × H_U。tfdiag 测到的跨步同向是真的，但它来自"每一步的激活本来就几乎一样"。H_U 已经把它算进去了，换 Hessian 不会给舍入带来新信息。

**附带的发现。** 在 QTIP 内部，H_U proxy 对 held-out future 的排序其实不差（ρ 0.68），主要是把 G/C/CA/GR 这些 proxy 偏高、future 偏差的臂和 U 一类的臂分开。
- 我之前说"proxy 排不对"，依据是 W3 的闭环排序，以及 EoRA 的例子。
- 但在视频层上，EoRA 的 H_U proxy 只比 U 低 1%（0.00667 对 0.00674），算不上反例。

**结论。** 结合同一时间跑完的时间条件层归因（r = 0.999），从激活统计出发修改层目标这条路，在这个模型上没有可用的新信号：
- 跨步结构已经被 H_U 包含；
- 只依赖 timestep 的条件路径几乎不承载 future 误差。

## FLAQ 模块 1：逐组换成 W3，直接测 C1 目标的敏感度（`sens.json`，`sens_worker.py`，`run_sens.pbs`）

**在论文里的位置。** C2 打包成插件框架 FLAQ，分三步：测 → 保 → 修。
- 测（本节）：量出每组层的量化误差对动作（A）、想象的 future（Z）和"future 传到动作上"（K）各伤害多少。
- 保：在给定的额外比特预算下，把高一档的位宽给伤害最大的组。
- 修：在分配后的模型上挂 J 的低秩支路。

**为什么要直接测，而不是用 Gauss-Newton 估。** 门 1 里，Gauss-Newton 把 future→动作这条路只算成 6–7%；可实际让 BF16 reader 读学生自己的 future，动作误差的 35–60% 都来自这条路。线性近似看不到误差在 10 个去噪步里一步步叠加的效果，直接换掉一组层再测，就没有这个问题。

**为什么不需要重新量化。** A16 基线的四个宿主都用同一份 BF16 Hessian，没有逐层顺序重收（`fastwam-a16-baselines-20261028/README.zh.md` 第 11 行）。所以每层的量化互相独立，把某几层从 W2 换成同一宿主的 W3，得到的正是宿主在这种位宽分配下会产出的模型。

**做法。**
- 起点：宿主的 W2 模型（先做 QTIP；GPTVQ、QuIP# 在 `sens.json` 的 `hosts` 里已配好，换 `FW_HOSTS` 即可）。
- 分组：两个专家各 30 个块（每块 10 个 Linear），加视频条件层、视频 head、动作条件层、动作输入输出层，共 64 组。一个视频块占全部权重的 2.77%，一个动作块占 0.56%。+0.06 bpw 的预算大约够升 2 个视频块或 10 个动作块，所以按块分组已经足够细。
- 配置：全 W2、全 W3、每组单独升到 W3，共 66 个。
- 每个配置都在 C80 的 80 个场景（seed 0）上跑一遍，每个场景记录：
  - 学生 query 对 BF16 teacher 的 mixed loss、future NMSE、gripper；
  - 冻结 BF16 reader 读学生 future 得到的 reader KL，以及它的 first10 motor MSE。
- 每组的"收益"= 全 W2 的值 − 这组升级后的值，分别算 A、λZ、βK 和总和 T。"密度"= 平均收益 ÷ 这组升级多花的比特（每 Mbit）。

**检查。**
- 全 W2 配置要逐场景复现 tune 的 C80 拟合（`tune/QTIP/fit_before.json`），相对差 ≤ 1e-6。
- 每个进程在第一个配置前、最后一个配置后各测一次 case 2000，两次必须逐位相同，并等于全 W2 的 case 2000。这样可以确认每组换回 W2 时没有留下改动。

**预先登记的判读。**
- 稳定：把 C80 按 case 号奇偶分成两半，各算一次 T 的密度排序，前 10 组至少重合 8 组，就进入步骤 2。步骤 2 按这个排序分配比特，并在 held-out 上与只看 A、层误差代理、随机三种同预算分配比较。
- 不稳定：重合不到 8 组，说明排序跟着场景变。先报告哪些组不稳定，再决定是加场景还是用更粗的分配。
- 同时报告（不作为门槛）：
  - 每个专家的收益里 A、Z、K 各占多少；
  - 64 组单独升级的收益之和与全部升到 W3 的收益之比（判断各组效果能不能直接相加）；
  - 收益超过 2 倍标准误的组有多少；
  - 200 次按 suite 随机对半的重合度分布。
- 产出：`results/sens/<host>/{summary.json, selection.json, RESULTS.zh.md}`。`selection.json` 列出 +0.03 / 0.06 / 0.12 / 0.25 bpw 下按 T 和按 A 选出的组，供步骤 2 使用。

**作业与竞速。** tcond（26066073）里每个"学生 query + reader"约 3.9 s，66 × 80 = 5280 次，约 5.7 GPU·h，拆成 4 份。
- `fwse1`：gdev，4 卡，一个作业跑完 4 份，< 2 h。
- `fwse1g0`–`fwse1g3`：g1，各 1 卡，各跑 1 份，3 h。
- 竞速规则（用户 2026-10-31 要求两种同时提交）：
  - 先开跑的赢；
  - gdev 先开跑，就撤掉 4 个 g1；
  - 任意一个 g1 先开跑，就撤掉 gdev，其余 g1 保留；
  - 最后一个跑完的分片负责写汇总。
- 已跑完的配置在重提（rN）时会跳过。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/sens_control.py diff
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/sens_control.py cputest
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/sens_control.py --phase sens submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/sens_control.py --phase sens0 submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/sens_control.py race
```

**实际经过（QTIP）。**
- g1 的 `fwse1g0`（26069856）先开跑，gdev 的 `fwse1`（26069855）在排队时被撤掉。
- 用户看到 g3 很空，让把仍在排队的 `fwse1g1`–`g3` 合成一个 g3 四卡作业：三个 g1 在排队时撤掉，`fwse1x`（26070509，g3，4 卡，3 h）接管第 1–3 份，共 49 个配置，每卡 12–13 个。
- 接管模式下，worker 把这三份的配置合在一起，再平均分给本作业的 4 个进程（`FW_ARMS=1:2:3`）。不接管时的分法和原来一样，所以不会和第 0 份重叠。
- 第 0 份 64 分钟跑完：1362 次"学生 query + reader"，每次约 2.8 s。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/sens_control.py --phase sensx submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/sens_control.py --interval 120 watch --phases sens0 sensx
```

### 结果（26069856 第 0 份 + 26070509 第 1–3 份）：预先登记的稳定性不通过；视频块之间测不出真实差异

报告：`pulled/26070509.pbs101/sens/QTIP/RESULTS.zh.md`。

**检查全部通过。** 全 W2 配置与 tune 的 C80 拟合逐场景完全相同（相对差 0）。每个进程首尾的 case 2000 逐位相同，换回 W2 没有残留。动作专家各组的 Z、K 收益正好是 0，符合模型结构（视频阶段先跑完，动作专家碰不到 future），说明测量链路是对的。

**整体升级很稳。** 全部从 W2 升到 W3：
- T 从 0.00266 降到 0.000775（−71%）；
- future NMSE 从 0.0386 降到 0.0133；
- reader KL 从 0.0199 降到 0.0060。

按奇偶分半看，Z 的下降几乎一样（0.00111 对 0.00104），T 也一样（0.00187 对 0.0019）。

**预先登记的判据不通过。** T 排序的前 10 组，奇偶两半只重合 4 组（要求 8 组），Spearman ρ 0.26；200 次随机对半的重合均值是 0.47。

**为什么不稳（事后分析）。**
- 30 个视频块的平均收益，离散程度（方差 1.21e-09）和抽样噪声（1.22e-09）一样大，块间真实差异的估计值是 0，Z 也是一样。也就是说，在 C80 的精度下这些块同样重要，多测场景也排不出有意义的顺序。
- 只升一个块时，少数场景的 future 会大幅摆动，主要是 2035 和 2049，而且都往变坏的方向。这两个场景就抵掉了奇数半边全部 30 个视频块的 Z 收益（奇数半边 2e-5，偶数半边 0.00102）。2035 在门 1 里也占了 GPTVQ Gauss-Newton 质量的 91%。
- 换成中位数或截尾均值后有所改善，但仍不达标：全部 64 组的 Z 中位数 ρ 0.69、重合 0.6；只看 30 个视频块，ρ ≤ 0.6。

**稳定的发现。**
- `action.io`（action_encoder、动作 head、proprio_encoder）和 `video.head`（写出想象 latent 的那一层）在 200 次随机对半里 100% 排前两名。
  - 两者都非常显著：T 收益分别是 4.1e-4 ± 3.6e-5 和 1.1e-4 ± 1.4e-5。
  - 加上 `action.cond`，三组拿回了全部 W2→W3 收益的 28%，只用了全部升级比特的 0.22%。
  - `video.head` 的收益主要来自 Z，`action.io` 的收益来自 A。
- 动作专家的 30 个块升级后合计收益约为 0（−1.3e-5），这些块留在 W2 几乎没有代价。
- 单组收益之和是全部升级收益的 0.74，说明组间有轻度的超可加性。

**对 C2 的含义。**
- 按块分配比特，在视频专家内部不会比随机分配好，因为各块之间没有可测的差别。
- 能稳定区分的只有"接口层"和"专家级别"。在专家级别上，只看 A 和看 C1 选出的方向一致，所以靠分配做不出能体现 C1 的消融。
- 接口层（输入输出层和 head）在常见做法里本来就保留高精度，A16 基线把它们也量化成了 W2。为了公平，这一点应当同时用到所有基线上，所以不能算作 C2 的增益。

## RAVQ 未校准版：QTIP 舍入 + C1 模块级比特预算（`ravq.json`，`ravq_worker.py`，`ravq_qtip.py`，`run_ravq.pbs`）

**在论文里的位置。** C2 不再是挂在别人量化器上的插件，而是我们自己的 WAM VQ 框架 RAVQ。这一节是它的未校准版（不训练任何参数），
校准版（只调共享参数）以后在它之上做，G4（共享参数上的梯度是否跨场景共享）也在它上面测。

**结构和每一块的来由。**
- 舍入：QTIP，A16 基线的同一份移植、同一份 H_U、同样的逐层种子。W2 上它是最强的宿主（204/280，误差代理最低）。
- 比特往想象网络挪（C1）：换回 BF16 future 能去掉三个宿主 24–36% 的动作误差；视频专家 W2→W3 拿走 A + λZ + βK 收益的 71%，
  主要经 Z 和 K；动作专家 30 个块全部 W2→W3 的收益是 0（FLAQ 模块 1）。所以在总比特不变时，把动作专家主体的比特给视频专家主体。
- 视频主体内部按率失真分配：块之间端到端测不出差别（块收益的离散度等于抽样噪声），所以用标准做法，
  按"每多存 1 bit 能降多少按权重数加权的 H_U 代理"贪心挑层。
- 两项公平口径，参照臂也给：接口层（action_encoder、动作 head、proprio_encoder、视频 head）存 BF16，占权重 0.01%；
  时间条件层只看 timestep，按 timestep 存输出表，输出与 BF16 完全一致，比特还比 W2 少（tcond 26066073）。
- QTIP 的格栅码对 K 是通用的，移植只因为官方 CUDA kernel 只有 K = 2–4 才限制了 K。`ravq_qtip.py` 放开到 K = 1–4，
  并把打包后的码（trellis、SU、SV、Wscale、LUT）存下来给校准版用；K = 2 时与移植逐位一致（CPU 测试）。

| 臂 | 视频主体 | 动作主体 | 作用 |
|---|---:|---:|---|
| QTIP_W2I | 2 | 2 | 2-bit 公平参照（接口 BF16 + 时间表） |
| QTIP_A1W2I | 2 | 1 | 动作主体降到 1 bit 的代价（G5） |
| RAVQU_W2 | 2 + 部分 3 | 1 | 我们的 2-bit：动作省下的比特给视频，总比特不超过 QTIP_W2I |
| QTIP_W3I | 3 | 3 | 3-bit 公平参照 |
| QTIP_A2W3I | 3 | 2 | 3-bit 下动作主体降一档的代价 |
| RAVQU_W3 | 3 + 部分 4 | 2 | 我们的 3-bit |
| QTIP_W3 | 3 | 3 | A16 基线原样，补测 held-out |

**作业。** `fwrv1`（gdev 4 卡）：quant（K = 1 的动作主体 300 层、K = 4 的选中视频层）→ assemble（每个臂 614 个指向来源文件的
符号链接、sources.json、COMPLETE.json）→ held-out 40×2 + swap（7 个臂）→ final（`results/ravq/`）。
闭环 `fwrvl`（g3 4 卡，`run_ravq_loop.pbs`）跑 `results/ravq/DECISION.json` 列出的臂，每臂与公平参照和 A16 原基线配对。

**预先登记的判读。**
- 公平口径单独能收回多少：QTIP_W2I / QTIP_U_W2、QTIP_W3I / QTIP_W3。主表必须给每个宿主都加这两项。
- 动作主体降位的代价：换 BF16 future 后的 motor（= 动作专家自身的误差）、motor、gripper；future NMSE 必须不变（动作专家碰不到 future）。
- RAVQU_Wb 上闭环的条件：held-out future NMSE ≤ 0.9 × QTIP_WbI 且 mixed loss ≤ QTIP_WbI。QTIP_W2I 的闭环无论如何都跑，它重置基线表。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/ravq_control.py cputest
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/ravq_control.py submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/ravq_control.py --phase ravql submit
```

**结果（held-out，`26072226.pbs101`；首次 26071726 的量化完成，组装因父运行链的 `self.source` 属性遮蔽同名方法而失败，改名
`layer_source` 后 r1 续跑）。** 完整表见 `pulled/26072226.pbs101/ravq/RESULTS.zh.md`。

| 臂 | bpw | mixed | motor | future NMSE | gripper | swap motor |
|---|---:|---:|---:|---:|---:|---:|
| QTIP_W2（A16） | 2.016 | 0.00336 | 0.0324 | 0.0532 | 0.0860 | 0.0246 |
| QTIP_W2I | 1.993 | 0.00283 | 0.0247 | 0.0498 | 0.0336 | 0.0130 |
| QTIP_A1W2I | 1.826 | 0.00293 | 0.0267 | 0.0498 | 0.0397 | 0.0162 |
| RAVQU_W2 | 1.993 | 0.00203 | 0.0206 | 0.0365 | 0.0397 | 0.0127 |
| QTIP_W3（A16） | 3.016 | 0.00095 | 0.0136 | 0.0164 | 0.0377 | 0.0092 |
| QTIP_W3I | 2.981 | 0.00082 | 0.0107 | 0.0149 | 0.0250 | 0.0050 |
| QTIP_A2W3I | 2.813 | 0.00083 | 0.0111 | 0.0149 | 0.0257 | 0.0055 |
| RAVQU_W3 | 2.981 | 0.00060 | 0.0097 | 0.0111 | 0.0337 | 0.0046 |

- 公平口径（接口 BF16 + 时间表）单独：W2 mixed 0.840×、gripper 0.390×；W3 mixed 0.863×、gripper 0.663×。主要来自接口层。
- 动作主体降一档：future 完全不变（相对差 0），动作专家自身误差（swap motor）+24%（W2→1）/ +12%（W3→2），motor 只 +8% / +3%。
- RAVQU 对公平参照（同比特）：W2 future 0.732×、mixed 0.717×、motor 0.835×；W3 future 0.743×、mixed 0.733×、motor 0.908×。两档都上闭环。
- 代价在 gripper：W2 1.183×（与 A1W2I 相同，来自动作主体 K = 1），W3 1.348×（A2W3I 只 1.028×，80 个端点上 gripper 噪声大）。
- 预算买到 48 个视频层（权重 16.7%），两档选中同一批：self_attn.o 17、self_attn.v 14、ffn.2 12、ffn.0 4、cross_attn.q 1，
  集中在 block 15–29（视频专家后半段的输出侧投影）。K = 4 代理选层估 0.00098，实测 0.00118。
- 闭环：26072518（四臂）跑了约 10 分钟后按用户要求取消；`fwrvlr1` 26072555 只跑 RAVQU_W2、RAVQU_W3 各 280（`ravq.json` loop_job.run），
  先与 A16 原基线闭环配对（QTIP_W2 204、QTIP_W3 202），公平参照 QTIP_W2I / QTIP_W3I 的闭环之后再跑。
- RAVQU_W2 闭环（26072555，280 回合跑完，汇总时 A16 的 analysis.py 缺 COMPLETE.json `bits` 报错；`bits` 已补，r2 26073056
  补汇总并跑 RAVQU_W3）：**205/280，对 QTIP_W2 204 配对 12 胜 / 11 负**，四个套件、七个扰动维度都在 ±2 内。
  held-out 上它对 A16 QTIP_W2 是 mixed 0.60×、future 0.69×、motor 0.64×、gripper 0.46×，闭环没有兑现。
  取消的 QTIP_W2I 留下 49 回合：38，同一批回合上 QTIP_W2 37、RAVQU_W2 35。
- RAVQU_W3 闭环（r2 26073056）：**202/280，对 QTIP_W3 202 配对 8 胜 / 8 负**，对 BF16 4 胜 / 18 负。
  held-out 上它对 A16 QTIP_W3 是 mixed 0.63×、future 0.68×、motor 0.71×、gripper 0.89×，闭环同样没有兑现。
- 闭环平台：A16 的九个非崩溃臂（GPTQ_W3，GPTVQ / QuIP# / QTIP 的 W2、W3，RAVQU_W2 / W3）H_U 代理跨 8 倍（0.0012–0.0096），
  成功 195–205、均值 202，与误差大小无关；三个宿主 W2 → W3 合计 603 → 610，多一位只换来 +7/840。BF16 216。
  历史上 W4A4 / W16A4 的各臂也都落在 195–208。
- 尺度对照：BF16 自己换扩散采样种子，first10 motor 的变化（fastwam-teacher-multiseed：同状态 seed 间标准差 0.0236，
  两个 seed 之间 RMSE 约 0.033）与 QTIP_W2 的量化误差（0.0324）同量级，是 QTIP_W3（0.0136）的 2.4 倍。

## 命令（项目根目录）

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py upload
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --attempt r3 submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/poll.py --until-done --interval 300 --attempt r3
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/control.py --attempt r3 rowcheck
```

## G4：GPTVQ W2 上共享参数的梯度能否跨场景迁移（`vqg4.json`，`vqg4_worker.py`，`run_vqg4.pbs`）

**问题。** Q0 调码本和行增益后，held-out 上的 future 降了 13–16%。G4 测这种共享参数上的 C1 梯度能不能带到别的场景、别的噪声上。
如果能，RAVQ 的校准版可以用共享参数修 future。测法与 fwld1 的 Gram 迁移测试相同。

**设置。** 宿主是 GPTVQ_W2。码本从稠密权重中恢复：607 层，45352 本码本。只跳过 action_encoder（7 列，有 341 个不同向量），占权重 1.2e-6。
参数族共六个：
- 码本条目偏移，分三种共享粒度：GPTVQ 原生每本、每层 256 个元条目、每模块 256 个元条目；
- 行 log-scale；
- 列 log-scale；
- 输出偏置。

三条路径分别测：λZ、βK、A_tf。样本 80 个（40 context × seed 0/1）。回放差为 0，梯度归约一致性为 2.3e-8。
作业 fwg4g `26071994.pbs101`（g1）。

**结果（T = 其他 context、新噪声上的一阶下降 / 本批自身）：**

| 参数族 | F→F | A→A | F→F 同 context 换 seed |
|---|---:|---:|---:|
| 码本（原生） | 0.002 | 0.179 | 0.260 |
| 每层共享 | 0.006 | 0.323 | 0.263 |
| 每模块共享 | 0.017 | 0.539 | 0.303 |
| 行 scale | 0.005 | 0.387 | 0.261 |
| 列 scale | 0.009 | 0.350 | 0.254 |
| 输出偏置 | 0.008 | 0.420 | 0.246 |

**判读：no_go（门槛 0.2，weak 0.06）。**
- future 一侧（Z、K）的梯度在共享参数上同样只属于具体场景。同一 context 换噪声有 0.25–0.30，换 context 就掉到 ≤ 0.017；跨 context 余弦为 0.00–0.03。这与 fwld1 在 rank-4 支路上测到的 0.021 一致：换成共享参数并没有让 future 变得可迁移。
- 动作一侧可以迁移，而且共享越粗迁移越好：码本 0.18，每层 0.32，每模块 0.54。行 scale 的跨 context 余弦是 0.49。
- 对 RAVQ 的影响：在 W2 上，future 只能靠比特（视频主体升位）来修。共享参数校准（④）只针对读取侧（A_tf）做，载体选模块级码本和行 scale。Q0 在 held-out 上的 future 收益不能用"可迁移的 future 梯度"来解释。

## RAVQ 校准版：① 读出度量当 future 目标，④ 只调 QTIP 自己的共享参数（`ravqc.json`，`ravqc_worker.py`，`run_ravqc.pbs`）

**在论文里的位置。** RAVQ 的四个部件里，② 模块级比特预算和 ③ QTIP 舍入是未校准版（上一节），这一节做 ① 和 ④，起点是 RAVQU_W2 / RAVQU_W3。

**① 读出度量 M。** 动作专家通过冻结 reader 读想象的 future；future 误差里真正伤动作的，是落在 reader 敏感方向上的那部分。
- 定义：在 BF16 模型上、teacher future 处，K（reader soft-bin KL）的 Gauss-Newton 矩阵 M_c = J_Rᵀ diag(h/2) J_R，h 是 teacher 动作处 KL 的曲率（first10 × 6 motor）。
- 估计：C80 的 80 个场景，每个场景 4 个 Rademacher 探针 c = J_Rᵀ(ε√(h/2))，只取 future 第 1 帧以后；每个场景先归一到单位迹再平均（前几轮的 GN 质量在场景间重尾，ESS 只有 1–26/80）。
- 形式：前 32 个特征方向 + 剩余对角，再加 10% 单位阵：M = 0.9 (D / tr M_R) M_R + 0.1 I，tr M = D；M = I 时 Z_M 退回普通 NMSE。
- 目标：Z_M = δzᵀ M δz / ‖z_T‖²。λ_M 按规则定：λ_M = λ × mean(Z) / mean(Z_M)（各臂起点的 C80），让 λ_M Z_M 的初始质量与 Q0 里 λZ 相同。
- M 只依赖 BF16 模型，一次估计，所有比特档、所有宿主共用。奇偶两半的对角相关和子空间重合作为稳定性报告。

**④ 只调共享参数。** QTIP 把旋转域里的每一对权重解码成 512×2 表里的一个条目（符号由 trellis 状态的哈希位决定）乘以该层的 Wscale；W = diag(SV) H_mᵀ (Wscale D) H_n diag(SU)。调的是：
- 解码表：按（角色，比特）分组，每组一张 512×2 偏移表，组内所有层共用（模块级码本：G4 里共享越粗、动作侧迁移越好，模块级 0.54）；
- 每行 / 每列的 log 尺度：就是 QTIP 已经存着的 SV（FP32）/ SU（BF16），0 额外比特；
- 有 bias 的层的输出偏置，0 额外比特。
- W' = bf16(e^s ⊙ (W_q + ΔW_表) ⊙ e^t)，b' = bf16(b + δb)，参数为 0 时与 RAVQU 逐位相同。接口层（BF16）和时间表（精确）不调。
- 目标 A + λ_M Z_M + βK（β 冻结值不变，K 读学生 future 时 bias 换回原值）；预算同 Q0（C80 一遍，8 次更新各 10 个场景）。
- 步长：每个张量（一层的行 / 列 / 偏置向量，一组的表）在“欧氏范数 = 权重变化 Frobenius 范数”的坐标里走 η × 参照范数；η 在第 1 批上按 0.5 规则从 {0.000625 … 0.02} 选，选中的那一步就是第 1 次更新。
- 梯度：QTIPGrad 的前向就是普通 F.linear；反向里 dD = Wscale · hadUt(g ⊙ SV e^s)ᵀ hadUt(x ⊙ SU e^t)，再经每对的码 scatter 到表上；行 / 列 / 偏置梯度由 g·y、x·(gW)、Σg 直接得到，不需要稠密的权重梯度。

**码从哪来。** K = 1 / 4 的层由 RAVQ 作业存下了 trellis，直接解出每对的（条目，符号）。K = 2 / 3 的层来自 A16 基线，只存了稠密权重：从 BF16 权重反解再找最近的候选，会把约 0.2% 的码对读错（表里 t0 接近 0 的条目和它的符号孪生几乎重合，落在 BF16 噪声里；CPU 测试 26073382），所以这些层用 ravq_qtip（同一移植、同种子、同 H_U）重新量化取 trellis，要求逐位复现基线权重；若不能逐位复现，就用这次新的 QTIP 舍入作该层起点，并计数报告。

| 作业 | 队列 | 内容 |
|---|---|---|
| `fwrc1` | gdev 4 卡 | codes（1208 个（层，K））→ metric（80 场景 × 4 探针）→ metric_final |
| `fwrc2` | g2 2 卡 | calib（每卡一个臂：C80 起点 → 步长检查 → 8 次更新 → 部署 → C80 终点 → held-out 40×2 + swap）→ zm（6 个参照臂的 held-out Z_M）→ final |
| 闭环 | g2 2 卡 | 不再用 `fwrcl`（280）：按用户规则，held-out mixed 不高于 RAVQU 的臂跑 840（面板 A+B+C），见 `fastwam-ravqc-plus840-20261031` |

**实际作业链（PBS afterok 依赖，后一段只在前一段 exit 0 后启动）。** `fwrc1` 26073411 取码完成、度量阶段失败（见下）→ `fwrc1r1` 26073871 → `fwrc2r1` 26073904 → `fwr840r1` 26073907。第一次的 `fwrc2` 26073421 和 `fwr840` 26073561 提前提交、在作业里等文件，空占了约 4 GPU·h，已取消。

**fwrc1 的失败。** 冻结 reader 的检查点重算上下文是一次性的生成器；非重入检查点每次反向都要进入一次，度量阶段在同一张 reader 图上做 4 次探针反向，第 2 次反向就报 `AttributeError: '_GeneratorContextManager' object has no attribute 'args'`。修法：`RavqcWork.bf16_reader` 在 reader 打开期间换成每次进入都新建一个的重算上下文（`Reentrant`），重算里照样走 BF16 主权重。CPU 测试 26073868 的 `test_reader_backward_passes` 在同一 torch 上先复现原错误，再核对修好后每次反向都重算、重算期间走 BF16 路由、各探针梯度等于直接 autograd。

### ① 的结果（26073871）：reader 只看很窄的一块 future，对角稳定、方向不稳定

| 量 | 值 |
|---|---:|
| future 维数 D（第 1、2 帧） | 37632 |
| 有效秩（tr² / tr(M²)） | 41.5 |
| 前 32 个特征方向占的迹 | 0.520 |
| 对角质量前 10% 元素的占比 | 0.617 |
| 奇偶两半对角的相关 | 0.863 |
| 奇偶两半前 32 维子空间的重合（平均 cos²） | 0.190 |
| 各场景原始迹的有效样本数 | 62.2 / 80 |
| future 第 1 帧 / 第 2 帧的质量 | 0.980 / 0.020 |

- **C1 的直接证据。** 动作专家对想象 future 的敏感度集中在约 40 个有效方向上（D 的千分之一），而且几乎全在第 1 帧。普通 NMSE 把 37632 维一视同仁，绝大部分误差落在 reader 不看的地方。
- **哪部分能泛化。** 对角（哪些通道、哪些位置重要）在两半场景间相关 0.86，是稳定的；前 32 个方向的子空间重合只有 0.19，是场景特有的。这和“future 难泛化”的教训一致：M 里能跨场景用的是对角那部分。
- **对校准的含义。** M 的低秩部分占 52% 的迹，可能把校准拉向 C80 特有的方向；是否有害看 held-out（fwrc2r1）。消融时可以对比只用对角 + ρI 的 M。

**码。** 1208 项：348 项直接用存下的 trellis，859 项重新量化后逐位复现基线权重，1 项（分片 1）不能逐位复现、用新的 QTIP 舍入作起点（与基线权重的相对差 2.6%）。码存在 `results/ravqc/codes/`，之后不必再取。

### 第一次校准（fwrc2r1 26073904）：两臂都没训成，更新规则改回 tune 的做法

| η（逐张量归一） | W2 预测下降 | W2 实际下降 | W3 预测下降 | W3 实际下降 |
|---:|---:|---:|---:|---:|
| 0.000625 | 0.0123 | −0.0007 | 0.0105 | −0.0029 |
| 0.0025 | 0.0493 | −0.0169 | 0.0421 | −0.0333 |
| 0.02 | 0.394 | −0.438 | 0.337 | −0.569 |

起点的总损失 W2 0.00169、W3 0.00098。六档步长全部让损失上升，没有一档满足 0.5 规则，两臂都记为未训练，门控也就没有放行闭环（fwr840r1 26073907 只跑了门控）。

- **原因。** 原规则让每个张量（约 1800 个：每层的行、列、偏置，每组的表）都在度量坐标里走 η × 参照范数，与梯度大小无关。这等于一个“人人都动”的符号步：每个张量把预算集中到自己梯度最大的几行几列上（text_embedding.0 的列尺度在最小一档就改了约 4%）。最小一档的一阶预测下降已经是损失本身的 7 倍，早已越过线性区。tune.json 的 26057132 见过同样的事：Adam 第一步在 300 万个尺度上走 sign(g)，损失涨 34–69%。
- **新规则（`update_rule_id` max-relative-gradient-v2）。** 和 tune 一样沿梯度走，让相对梯度最大的那个元素恰好移动 η，其余按比例：行 / 列对数尺度按原样，偏置元素相对该行的输出尺度 |W_r| rms(x)，表项相对表值的 rms。网格 {0.0025, 0.005, 0.01, 0.02, 0.05}（tune 的 QTIP 在 0.01 实现了预测的 57%）。0.5 规则、8 次更新、目标不变。
- **复用与留档。** fit_before 和参照臂的 zm 与规则无关，直接复用；旧的 arm / step 文件改名为 `*.r0_26073904.json`。CPU 测试核对：每个元素的相对位移等于 −η g_u / max|g_u|、最大位移恰为 η、一阶预测与实际参数位移一致。
- **作业链。** CPU 测试 26074308 → `fwrc2r2` 26074313 → `fwr840r2` 26074318（afterok）。840 的门控输出解析也修了：26073907 在没有臂通过时把 “GATE {…}” 那一行当成了臂名（命令替换吃掉了末尾的空行），现在只读 `RUN` 开头的那一行（CPU 测试 26074311 覆盖两种情况）。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/ravqc_control.py cputest
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/ravqc_control.py --phase ravqc submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/ravqc_control.py --phase ravqc2 --attempt r1 --after <fwrc1 作业号> submit
```

### 第二次校准（fwrc2r2 26074313）：W3 在训练场景上好了 42%，held-out 反而坏了 22%；改为 v3（FP32 折算 + 只调共享解码表）

步长检查（第 1 批，目标 A + λ_M Z_M + βK）：

| 臂 | 起点 | η | 预测下降 | 实际下降 | 实际 / 预测 |
|---|---:|---:|---:|---:|---:|
| W2 | 0.00169 | 0.0025 | 0.00046 | 0.00013 | 0.28 |
| | | 0.005 | 0.00093 | 0.00014 | 0.16 |
| | | 0.01 | 0.00185 | 0.00012 | 0.065 |
| | | 0.02 | 0.0037 | −0.00010 | — |
| W3 | 0.00098 | 0.0025 | 0.00030 | 0.000076 | 0.25 |
| | | **0.005（选中）** | 0.00060 | 0.00039 | **0.65** |
| | | 0.01 | 0.0012 | 0.00051 | 0.43 |
| | | 0.02 | 0.0024 | 0.00018 | 0.08 |

W2 没有一档到 0.5，未训练；W3 用 η = 0.005 训了 8 次。W3 的结果（比值 = 校准后 / 未校准 RAVQU_W3）：

| 比较范围 | mixed | future NMSE | Z_M | motor MSE |
|---|---:|---:|---:|---:|
| C80 全部 80 个场景（都训过） | 0.575 | 0.564 | 0.567 | 0.643 |
| C80 未见批次（第 2–8 批在自己那次更新之前） | — | 0.784 | 0.915 | 0.940 |
| held-out 40×2 | **1.218** | 1.149 | 1.144 | 1.466（RMSE 1.061） |

held-out mixed 分套件：spatial 1.05、object 1.45、goal 1.06、libero_10 1.05；换成 BF16 future 后的 motor 0.986。门控不放行，fwr840r2 26074318 只跑了门控（4 秒，退出码 0，修过的解析按预期给出空臂名单）。

- **这是背题，不是没训好。** 训过的场景上降了 42%；未见批次的 Z 0.78 主要来自第 6 批（0.35），其余六批在 0.95–1.10；held-out 四个套件全部变差。动作专家本身几乎没动（换 BF16 future 后 motor 0.986），变差的是学生的 future，再经 reader 传到动作上。
- **实际在动的参数。** 8 次更新里有 6 次由 video_expert.text_embedding.0 的某个偏置元素定步长，这个偏置最后相对改了 26%；视频块权重只改了 0.19%。原因在折算方式：W' = bf16(e^s (W0 + ΔW) e^t) 里的 W0 已经在 BF16 网格上，单个元素的改动小于半个间隔（0.2–0.39%）就舍回 W0，实际生效的是"截掉小改动后的梯度步"。W2 三档步长的实际下降都卡在 0.00012–0.00014，W3 在 0.0025 只兑现 0.25、到 0.005 反而兑现 0.65，都是这个截断造成的。
- **和 C1 的设计原则对上了。** 行 / 列尺度和偏置是每层每通道各自的参数，fwld1 和低秩门控已经说明这类自由参数会背 C80。④ 的本意是"只调共享参数"，跨层共用的只有解码表。

**v3 规则（`update_rule_id` lut-only-fp32-fold-v3，ravqc.json `training.update_rule_changes[1]`）：**

- **折算从码的 FP32 解码出发。** W' = bf16(e^s (W_hi + ΔW) e^t)，W_hi 是码在原表下的 FP32 解码，它的 BF16 舍入就是 W0（码阶段逐位核对过），所以零参数时仍逐位等于 W0。小于半个间隔的改动会按比例让一部分元素跳一个间隔，期望上等于设定的改动。CPU 测试 26075164（128×256 层）：表偏移约 0.024%|W| 时，实际兑现比例 1.06（旧折算 0.145）；行尺度 +0.1% 时 0.97（旧折算 0）。这条是实现上的修正，以后所有校准都要用。
- **只调解码表。** 每个（角色，比特）组一张 512×2 的偏移，同组所有层共用，每臂 4 组、4096 个数。行 / 列 / 偏置的定义留在 families 里，供以后消融。
- **其他。** 步长网格 {0.00125, 0.0025, 0.005, 0.01, 0.02, 0.05}；0.5 规则、8 次更新、目标和 λ_M 都不变。RESULTS 增加"未见批次"一栏，作为 C80 内部的泛化检查。r2 的规则相关文件改名为 `*.r0_26074313*`，fit_before 和 zm 直接复用。
- **作业链。** CPU 测试 26075164 → `fwrc2r3` 26075166 → `fwr840r3` 26075168（都是 afterok）。

对论文的意义：① 的结论不受影响（读出度量窄、对角稳定）。这次失败出在 ④ 的参数集：每通道参数在 80 个场景上背了题，正好是"共享参数才能泛化"这条设计原则的反例，可以写进消融。v3 是 ④ 的严格版本，故事收成"在读出度量下校准跨层共享的码本"。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/ravqc_control.py --phase ravqc2 --attempt r3 cputest
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/ravqc_control.py --phase ravqc2 --attempt r3 --after <CPU 测试作业号> submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-ravqc-plus840-20261031/control.py submit --attempt r3 --after <fwrc2r3 作业号>
```

### 第三次校准（fwrc2r3 26075166，v3）：W3 的 C80 收益来自 4 个离群场景，没见过的场景得不到好处；held-out 变差，门控不放行

步长检查（第 1 批）：

| 臂 | η | 预测下降 | 实际下降 | 实际 / 预测 |
|---|---:|---:|---:|---:|
| W2 | 0.00125 | 0.000089 | 0.000064 | 0.72 |
| | **0.0025（选中）** | 0.00018 | 0.00011 | **0.64** |
| | 0.005 | 0.00036 | 0.00010 | 0.29 |
| W3 | 0.0025 | 0.00014 | 0.00020 | 1.43 |
| | 0.005 | 0.00029 | 0.00048 | 1.66 |
| | **0.01（选中）** | 0.00057 | 0.00046 | **0.81** |
| | 0.02 | 0.0011 | 0.00033 | 0.29 |

FP32 折算生效了：小步长下实际下降跟得上预测，W3 甚至超过预测（见下面的离群场景）。

| 臂 | C80 mixed 前 → 后 | C80 future 前 → 后 | 未见批次 A / Z / Z_M | held-out mixed / future / motor / gripper（÷ RAVQU） |
|---|---|---|---|---|
| W2 | 0.00146 → 0.00142 | 0.0261 → 0.0255 | 1.018 / 0.992 / 1.000 | **1.006** / 1.009 / 0.999 / 0.999 |
| W3 | 0.00064 → 0.00035 | 0.0129 → 0.0069 | 0.976 / 0.914 / 0.968 | **1.260** / 1.226 / 1.026 / 0.765 |

W3 held-out 分套件（mixed）：spatial 1.15、object 1.47、goal 1.15、libero_10 1.00。两臂都没过"held-out mixed 不高于未校准"，`fwr840r3` 26075168 只跑了门控。

**W3 的 C80 收益从哪来。** 逐场景看 future NMSE：80 个场景的中位数约 0.006，但 2019、2037、2031、2034 四个场景在 0.07–0.14，占 C80 future 总量的 40%。校准把它们降到约 0.008（12 倍），其余 72 个场景只从 0.367 降到 0.325（0.89×，而且都是训练过的）。未见批次逐批看：不含大离群场景的第 4、7、8 批在自己那次更新之前是起点的 0.998、0.982、1.046，也就是别的批次的更新对它们完全没用；唯一明显的"迁移"是第 6 批 0.78，来自它里面的 2019（0.14）被前面的更新顺带降了。第 6 批那次更新的最大相对梯度是其他批次的 10 倍。

**含义。**
- 离群场景是量化后 future 已经走到另一个模式上的 rollout（与 BF16 差 0.1 量级），它们的梯度最大，max-normalized 的步长方向基本由它们决定；一张共享表上很小的改动就能把它们"拨回去"，这是对这几个场景的拟合，换一批场景就是噪声，所以 held-out 变差。
- 这与 G4（GPTVQ W2）一致：future 一侧的梯度即使在模块级共享参数上，跨 context 的迁移 T 也只有 ≤ 0.017，动作一侧 0.54。
- W2 基本没动：held-out 1.006 里还有一部分不是校准造成的。W2 起点有 1 层（`video_expert.blocks.7.cross_attn.k`，K = 2）重新量化时不能逐位复现 RAVQU 的权重，用了新的 QTIP 舍入（相对差 2.55%）；case 2000 的 first10 motor RMSE 因此从 RAVQU 的 0.02160 变成 0.02198（+1.7%）。所以 RAVQC_W2 的零点不是 RAVQU_W2。下一版校准要把这类层排除在外（保留 RAVQU 存的权重、不参与解码表），并重算 fit_before。

### 下一步（2026-11-01 05:50 SGT 起，夜间自主决定）

**1. RAVQU 先跑 840（`fastwam-ravqc-plus840-20261031`，协议升为 v2，作业 `fwr840u` 26075585，g2）。** RAVQU（② 模块比特预算 + ③ QTIP 舍入，不校准）在 held-out 上已经是公平参照的 0.72×（W2）/ 0.73×（W3），但闭环只有面板 A 的 280（W2 205 对 QTIP 204），280 排不出量化器的先后。这次补跑 B、C 两个面板（每臂 560，约 6 小时），面板 A 用 guided-hessian 里那次闭环（同一 A16 运行时；安装检查要求 case 2000 的 anchor 与那次闭环 startup 的 0.0216015 / 0.0085155 相对差 ≤ 1e-4）。以后任何 RAVQC 的 840 都要拿它作配对对照。
- 协议改动：arms 加入 RAVQU_W2 / RAVQU_W3（`uncalibrated`，面板 B、C，不经 held-out 门控），排在 RAVQC 前面；v1 的远端副本改名 `protocol.v1.json`（v1 没有跑过任何 RAVQ 臂的回合，`results/prep` 的 T5 合并与协议无关，直接复用）。
- CPU 测试 26075582 全过：门控（RAVQU 总是放行、RAVQC 按规则）、安装检查的参照值（RAVQC 用 C80 fit 行，RAVQU 用面板 A 的 startup，没通过或不是本臂的 startup 拒收）、调度（RAVQU 560 / RAVQC 840）、汇总（RAVQU 的 A 取自原闭环，B、C 取自本作业；RAVQC 对 RAVQU 在 839 个槽位上配对）。

**2. 解码表梯度的跨场景迁移诊断（`ravqc.json` 的 `gradtx`，作业 `fwrcg2` / `fwrcg3`，各 gdev 4 卡）。** 在每次校准的起点（表偏移为 0，与 fwrc2r3 的起点相同），C80 的 80 个场景各自算一次解码表梯度，动作部分（A）和 future 部分（λ_M Z_M + βK）各回放一次，然后统计：
- 通用份额：一个场景的梯度能量里，所有场景共有的部分占多少（以前的门槛 0.06）；
- 对半迁移 T：随机分两半，沿前一半的平均梯度走一步，后一半得到的一阶下降占前一半自身的比例；
- 批间迁移 T：训练用的 10 场景一批之间的同一个量（应与 r3 的未见批次一致）；
- 加权变体：每个场景按自身损失归一、去掉 Z 超过中位数 5 倍的场景；按组分别算；
- 自检：第 1 批的平均梯度必须复现 fwrc2r3 步长检查的梯度范数（比值 1）。

它回答的是：用户定下的 A + λ_M Z_M + βK 里，到底哪一项的梯度在共享解码表上能带到没见过的场景。G4 和 r3 都提示答案是"只有动作那一项"；如果诊断确认，④ 的校准就只能沿可迁移的方向走（例如只用动作部分，或只用在场景之间方向一致的分量），① 的读出度量继续作为评价和比特分配里的 future 目标。这一步由用户拍板，诊断只提供数字。

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-ravqc-plus840-20261031/control.py --attempt u submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/ravqc_control.py --phase ravqcg2 submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-guided-hessian-20261028/ravqc_control.py --phase ravqcg3 submit
```

### 解码表梯度迁移诊断（fwrcg2r1 26075732、fwrcg3r1 26075733，按组拆分由 CPU 作业 26075990 重算）：训练方向在两臂上都不迁移；唯一能迁移的是动作项落在 action / text 表上的那一小部分

第一次提交（fwrcg2 26075618、fwrcg3 26075620）所有 shard 都报 `KeyError: 'FW_ARM'`：容器用 `--cleanenv`，只放行白名单里的环境变量。改为 worker 按 `FW_JOB` 从 `ravqc.json` 的 `gradtx_jobs` 查臂（`gradtx_arm`），CPU 测试 26075731 通过后以 r1 重交（afterok）。之后在汇总里补了"每张表 × 动作项 / future 项"的拆分和每张表在平均梯度里的能量占比（`per_group_parts`）；`gradtx_resummary.py`（CPU，26075990）从已存的 80 个逐场景梯度文件重算，原有各项逐位复现（相对 1e-9）后才写入。CPU 测试 26075989 覆盖了新字段和重算脚本。

自检：两臂第 1 批的平均梯度与 fwrc2r3 步长检查的梯度范数之比 0.9988–1.0029（BF16 层输出处回传梯度的舍入量级），回放最大差 0。

| 臂 | 梯度 | 通用份额 | 对半迁移 T | 批间迁移 T | 前 5 个场景占平均梯度 |
|---|---|---:|---:|---:|---:|
| W2 | 动作部分（A） | 0.0015 | 0.076 | 0.044 | 0.71 |
| | future 部分（λ_M Z_M + βK） | 0.0007 | 0.032 | 0.017 | 0.71 |
| | **两者之和（训练走的方向）** | **0.0007** | **0.033** | **0.020** | 0.71 |
| | 之和，按场景自身损失归一 | 0.0020 | 0.081 | 0.026 | 0.45 |
| | 之和，去掉 Z > 5× 中位数（1 个） | 0.0007 | 0.033 | 0.019 | 0.71 |
| W3 | 动作部分（A） | 0.0000 | 0.004 | 0.003 | 0.88 |
| | future 部分 | −0.0001 | −0.004 | 0.011 | 0.67 |
| | **两者之和** | **−0.0000** | **−0.001** | **0.013** | 0.70 |
| | 之和，按场景自身损失归一 | 0.0008 | 0.029 | 0.011 | 0.23 |
| | 之和，去掉 Z > 5× 中位数（11 个） | 0.0010 | 0.038 | 0.022 | 0.54 |

按表（"占比"是这张表在两者之和的平均梯度 |mean|² 里的份额，即一步更新主要改哪张表）：

| 臂 | 表 | 占比 | 之和 对半 T | 动作项 对半 T | future 项 对半 T |
|---|---|---:|---:|---:|---:|
| W2 | action_body@1 | 0.002 | 0.647 | **0.647** | 0（无梯度） |
| | text@2 | 0.009 | 0.101 | 0.337 | 0.045 |
| | video_body@2 | 0.623 | 0.020 | 0.045 | 0.019 |
| | video_body@3 | 0.367 | 0.053 | 0.056 | 0.055 |
| W3 | action_body@2 | 0.000 | 0.162 | 0.162 | 0（无梯度） |
| | text@3 | 0.005 | −0.049 | −0.051 | −0.027 |
| | video_body@3 | 0.618 | −0.015 | 0.010 | −0.019 |
| | video_body@4 | 0.377 | 0.023 | −0.006 | 0.023 |

**含义。**
- r3 的失败在起点的梯度几何里就已经决定了，不是步长或实现的问题：W3 的批间迁移 T 0.013，正对应 r3 里不含离群场景的未见批次在自己那次更新前是起点的 0.998 / 0.982 / 1.046。
- 每一步更新 99% 改的是两张视频表，而视频表上的梯度没有跨场景共有的方向（通用份额 ≤ 0.0012）。这一点对动作项和 future 项都成立（视频表上动作项 T ≤ 0.056，future 项 ≤ 0.055），所以"只用动作项"救不了视频表。
- 离群场景（2019、2031、2034、2037、2047、2051、2055，和 r3 是同一批）占了平均梯度的 70% 左右，但不是唯一原因：去掉它们或按场景归一后 T 仍 ≤ 0.04，剩下的场景之间同样各走各的。
- 唯一能迁移的是动作项落在非视频表上的部分：W2 的 action_body@1 通用份额 0.041（低于 0.06 的门槛，但 40 个场景平均后迁移 65%，前 5 个场景只占 18%，不是离群拉出来的；理论值 40×0.041/(1+39×0.041) = 0.64 与实测一致），text@2 的动作项 0.34。future 项在 action_body 表上的梯度恒为 0（视频 token 不读动作专家）。但这部分只占一步更新的 0.2%（W2）/ 0.03%（W3），W3 上也只剩 0.16。这和 G4 在 GPTVQ 上"动作侧迁移、future 侧不迁移"是同一个现象：能迁移的只有动作专家自己那一侧。
- 用户决定（2026-11-01）：④ 留在 RAVQ 和论文里（① 读出度量作 Z 目标 → ② 模块比特预算 → ③ QTIP 舍入 → ④ 在 A + λZ + βK 上校准共享参数）。这一节的迁移诊断和 held-out MSE 只作内部调参参考，不进论文，也不代表闭环表现；消融由用户做，闭环说了算。

### RAVQU 840 闭环结果（`fwr840u` 26075585，2026-11-01 11:58 SGT 跑完）

RAVQU_W2 / RAVQU_W3 各补跑面板 B、C 共 560 个 episode，面板 A 用 26072555 / 26073056 那次闭环。安装检查通过，840 个槽位里没有无效槽位。RAVQC 两臂按规则没有进闭环，因为 r3 的 held-out mixed 是 RAVQU 的 1.006 倍（W2）和 1.26 倍（W3），都高于 RAVQU。完整表格在 `fastwam-ravqc-plus840-20261031/pulled/26075585.pbs101/results/RESULTS.zh.md`。

| 臂 | A | B | C | 合计 / 840 |
|---|---:|---:|---:|---:|
| BF16 | 216 | 206 | 230 | 652（77.6%） |
| **RAVQU_W2** | 205 | 199 | 223 | **627**（74.6%） |
| QTIP_W2 | 204 | 198 | 214 | 616 |
| QuIP#_W2 | 198 | 198 | 213 | 609 |
| GPTVQ_W2 | 201 | 191 | 210 | 602 |
| **RAVQU_W3** | 202 | 204 | 223 | **629**（74.9%） |
| QTIP_W3 | 202 | 210 | 219 | 631 |
| GPTVQ_W3 | 203 | 206 | 221 | 630 |
| QuIP#_W3 | 205 | 202 | 219 | 626 |

配对检验用精确 McNemar，同一槽位配对：

| 配对 | 胜 / 负 | p | A | B | C |
|---|---:|---:|---:|---:|---:|
| RAVQU_W2 对 QTIP_W2 | 46 / 35 | 0.266 | 12 / 11 | 18 / 17 | 16 / 7 |
| RAVQU_W2 对 QuIP#_W2 | 51 / 33 | 0.063 | 18 / 11 | 15 / 14 | 18 / 8 |
| RAVQU_W2 对 GPTVQ_W2 | 62 / 37 | **0.015** | 17 / 13 | 23 / 15 | 22 / 9 |
| RAVQU_W2 对 BF16 | 30 / 55 | 0.009 | 6 / 17 | 16 / 23 | 8 / 15 |
| RAVQU_W3 对 QTIP_W3 | 31 / 33 | 0.901 | 8 / 8 | 12 / 18 | 11 / 7 |
| RAVQU_W3 对 GPTVQ_W3 | 42 / 43 | 1 | 12 / 13 | 16 / 18 | 14 / 12 |
| RAVQU_W3 对 QuIP#_W3 | 43 / 40 | 0.826 | 10 / 13 | 16 / 14 | 17 / 13 |
| RAVQU_W3 对 BF16 | 24 / 47 | 0.009 | 4 / 18 | 11 / 13 | 9 / 16 |

**2 bpw 的结果。**
- RAVQU_W2 是 2-bit 里最好的一臂。它比 GPTVQ 多 25 个（显著），比 QuIP# 多 18 个（p 0.063），比 QTIP 多 11 个（不显著）。
- BF16 与 QTIP_W2 之间差 36 个，RAVQU_W2 收回了其中 11 个，约 31%。
- RAVQU_W2 的 627 和三个 3-bit 宿主（626–631）在同一水平。这只是合计数的比较：汇总只给同比特的臂配对，跨比特配对要用逐槽位记录重算。
- 增益集中在面板 C：223 对 214，16 胜 7 负；面板 A、B 基本持平。
- 按 suite 看，增益最大的是长时程的 libero_10：155 对 QTIP_W2 的 146，同时 GPTVQ 145、QuIP# 148、BF16 165。其次是 libero_object（174 对 169）。goal 和 spatial 各差 1–2 个。
- 按扰动看，7 个维度里 6 个不低于 QTIP_W2：sensor_noise +4、camera +3、language +3，light −2。
- 要在 840 上对 QTIP_W2 达到 p < 0.05，需要净胜大约 18–22 个，现在是 +11。

**3 bpw 的结果。** RAVQU_W3 是 629，QTIP 631、GPTVQ 630、QuIP# 626，四者持平（p ≥ 0.83）。3-bit 下所有量化器离 BF16 都只差 21–26 个，宿主之间分不出先后。

**后面怎么用。** 这次 840 是以后任何 RAVQC（含 ④）840 的配对对照：同一槽位、同一 A16 运行时。下一版 ④ 校准之前，要先改 W2 的起点：排除 requantized_new 层，保留 RAVQU 存的权重，并重算 fit_before（见上文 r3 一节）。
