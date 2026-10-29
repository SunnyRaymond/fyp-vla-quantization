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
