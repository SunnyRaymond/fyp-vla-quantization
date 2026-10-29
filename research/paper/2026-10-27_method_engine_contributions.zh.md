# 量化框架、Inference Engine 与贡献规划（v2）

2026-10-27。v2 相对 v1 的变化：

- 发现 **A4 是当前 Q0 的主要误差来源，而现有 engine 拿不到 A4 的任何运行时收益**，“为什么要 A4”无法自圆其说（§4）。
- 方向调整为 **路线 3：W4A8 级 VQ**——VQ 负责存储（约 4 bpw 及更低），计算走 INT8/FP8 tensor core（§5）。
- **主评测定为 LIBERO-Plus**，并整理现有量化方法在 LIBERO-Plus 上的差距（§6）。
- 贡献措辞按路线 3 重写（§8），实施顺序重排（§10）。

依据：当前代码（链接见各节）、`experiment/HANDOFF.zh.md` 与 handoff 01–06、各实验 RESULTS/summary、三篇同期论文 v1（QuantWAMs 2607.28405、Q-WAM 2609.33269、SteerQuant 2609.39056）与 Fast-WAM v2。

---

## 0. 当前结论

1. **当前 Q0（MXVQ 权重 + A4）的问题。** 同一份 MXVQ 权重关掉 A4，held-out 动作误差减半；闭环上只开 A4 就从 216 掉到 204。而 engine 的 GEMM 是 BF16，A4 只有精度代价，没有速度或显存收益。
2. **新方向：W4A8 级 VQ。** 融合 kernel 读 packed index，在片上查表得到 INT8 权重，走 INT8 tensor core。memory-bound 的 GEMM 拿带宽收益，compute-bound 的 GEMM 拿算力收益，不需要区分 video/action token，所以对 Cosmos-Policy、LingBot-VA 这类单一 DiT 同样成立。权重解码成 8-bit 后 tensor core 按 8-bit 速率跑，A4 不会更快，所以 A8 是 VQ 权重唯一说得通的搭档。
3. **主评测用 LIBERO-Plus。** 标准 LIBERO 上同期方法都已接近无损（−0.2 到 −1.0 pp），区分不出好坏；Fast-WAM-IDM 在我们的 280 面板上，W4A4 baselines 掉 4.6–7.5 pp，camera viewpoint 一项最多掉 9/21。
4. **最大风险。** W4A8 档的标量方法在标准 LIBERO 已近无损（SteerQuant Cosmos-Policy W4A8 98.6 vs FP 98.5）。我们的优势要靠 **LIBERO-Plus 的难扰动** 和 **更低 bpw（3、2 bpw 档）** 体现。
5. **阶段 0 测速已完成（§11）。** 同等 cache + CUDA Graph 优化的 BF16 是 **541 ms/call**，当前 Q0 engine 是 896 ms，**比它慢 1.66×**；“相对 released BF16 1.64×”完全来自 cache 和 Graph，与量化无关。A100 上 INT8 dense GEMM 对每次 call 的 GEMM 总时间是 1.56×（video 1.73×、action 0.92×），但 GEMM 只占去噪时间的 46%，所以路线 3 在 A100 上端到端上限约 1.17×。精度检查推迟到阶段 1。

---

## 1. 模型背景：为什么 future 在 Fast-WAM-IDM 里是关键中间量

我们用的是 Fast-WAM 的 **Optional-IDM** checkpoint，`idm` 模式，即 Fast-WAM 论文里的 *Fast-WAM-IDM*（imagine-then-execute）。一次 policy call：

```
observation(2 cams 224) + proprio + prompt
  │
  ├─ T5 text encoder、VAE 编码当前帧（BF16，不量化）
  │
  ├─ Video expert 去噪 10 步 ──► clean future latent z ∈ [1,48,3,14,28]
  │      (frame0 = 观测 clamp；frame1/2 = 想象的未来)
  │
  ├─ 用 [当前帧, 想象未来] prefill video K/V
  │
  └─ Action expert 去噪 10 步（cross-attend 到 video K/V）──► 32×7 action chunk
         闭环执行前 10 步后 replan
```

**action expert 读的是 video expert 想象出来的 future。** video expert 的量化误差先在 10 步去噪里累积成 future 误差，再被 action expert 读进动作。这和 Fast-WAM 默认模式（test-time 不生成 future，video 只在训练时做监督）完全不同。

| 论文 | Fast-WAM 变体 | test-time 生成 future？ |
|---|---|---|
| QuantWAMs | Fast-WAM（BF16 LIBERO 97.6%，只保护 action 去噪步） | 否 |
| Q-WAM | Fast-WAM（原文：“keep video prediction only as training supervision”） | 否 |
| SteerQuant | FastWAMJoint（future 与 action 联合去噪）+ Cosmos-Policy | 是，但只把 future stream 当作误差传到 action 的通道 |
| **我们** | **Fast-WAM-IDM** + Cosmos-Policy（+ 计划 LingBot-VA） | **是，并把 future 本身当保真目标** |

Q-WAM 在无 imagination 的 Fast-WAM 上测得 action expert 只占 17% 参数却承载 89.3% 的 action mass。IDM 里 video expert 的输出直接被 action 读取，这个比例必然不同，这是 C1 的切入点。

量化范围：**614 个 Linear、6,016,317,440 个 weights**（video 306、action 307、proprio encoder 1）。T5（10.58 GiB）、VAE（1.31 GiB）、norm、bias、KV 不量化。

---

## 2. 当前量化框架（Q0）实现

> 这一节描述现有 Q0（MXVQ 权重 + A4）。路线 3 会保留其中的校准目标与大部分流程，改动见 §5。

每个目标 Linear 计算 `Y = X Wᵀ + b`。Q0 对它做四件事：换坐标、量化 activation、用 MXVQ 存 weight、用任务末端目标校准 codebooks。

### 2.1 换坐标：Smooth + Hadamard + 学到的 128×128 正交块

```
X_t = X · S⁻¹ · R,   W_t = W · S · R    ⇒   X_t · W_tᵀ = X · Wᵀ
```

- `S`：SmoothQuant 式对角缩放，α=0.5，冻结。
- `R` = 保存的 ±1 signs → 128 宽块 Hadamard（FWHT）→ 每个 stream（video / action）共享的 128×128 正交矩阵 `U`。
- `U` 用 Cayley 参数化：`K = A − Aᵀ`，`U = (I − K/2)⁻¹(I + K/2)`。代码：[rotation_bank.py:36](../../experiment/idea-validation/fastwam-vq-learned-r128-c8/rotation_bank.py)。
- `U` 先在 W16A4 下用 C80 学 8 步得到 R0，然后冻结。

**U 不是贡献。** H-W16A4 是 204/280，R0-W16A4 是 203/280，单独没有闭环收益。

### 2.2 Activation：dynamic per-token 标量 A4

```python
scale = max(|z_row|) / 7
q     = clamp(round(z / scale), -7, 7)      # round half to even
x̂    = q * scale                           # BF16
```

每个 token 一个 scale，在线计算。训练 backward 用 identity STE。代码：[rotation_bank.py:23](../../experiment/idea-validation/fastwam-vq-learned-r128-c8/rotation_bank.py)。

### 2.3 Weight：Microscaling additive VQ（MXVQ）

每个 Linear 一对 codebook `C0, C1 ∈ BF16[256×4]`。每行 weight 沿 input 维每 4 个切成一个向量，存两个 uint8 index `(i0, i1)`；每 32 个 input 权重共享一个 uint8 exponent `e`；每行一个 BF16 `row_gain`。

```
S_block = row_gain · 2^(e − 127)
Ŵ[r, 4v : 4v+4] = BF16( (C0[i0] + C1[i1]) · S_block[r, v // 8] )
```

**小例子。** `row_gain = 0.02`，`e = 126` → `S = 0.01`。`i0 = 17`，`i1 = 203`：

```
C0[17]  = ( 1.00, -0.50,  0.25,  2.00)
C1[203] = ( 0.10,  0.00, -0.05, -0.20)
和       = ( 1.10, -0.50,  0.20,  1.80)
× 0.01  → (0.011, -0.005, 0.002, 0.018)
```

两本 256 项的加法 codebook 相当于每个 4 维向量有 65,536 个候选码字，只需两次查表、存 2 字节。

**bpw：** indices 4.00 + exponent 0.25 + codebooks/row gains/Smooth/signs/U ≈ **4.271052 bpw**；target 编码 2.991 GiB（BF16 约 11.2 GiB，3.76×）。

**microscale 的收益：** 同 d4/双 codebook 容量下，固定输入 first10 motor RMSE **0.05705 → 0.05240（−8.15%）**，280 闭环 202 → 204。代码：[micro_codebook.py:126](../../experiment/idea-validation/fastwam-vq-microscale/micro_codebook.py)。

### 2.4 初始化拟合

[micro_codebook.py:63](../../experiment/idea-validation/fastwam-vq-microscale/micro_codebook.py) `fit_vq`：

1. 每行 RMS 作 row gain；每 32-block 的 exponent = `round(log2(block RMS / row RMS))`。
2. 除以 `S_block` 得到归一化向量。
3. 加权 k-means 拟合 `C0`，对残差拟合 `C1`，交替 8 轮；最后在全部向量上重新分配 index。
4. 距离度量 = **A4 后输入的逐通道二阶矩 × S²**，近似最小化 Linear 输出误差。

之后 indices、exponents 固定。

### 2.5 任务末端校准：A + λZ + βK

**训练什么。** 只训练 614 组 codebooks（FP32 master，前向用 BF16 存储值）和 log row gains；codebook 值共 1.26M 个，另加每个输出行一个 gain。底模、indices、exponents、S、R 全部冻结。

**梯度怎么到 codebook。** `Ŵ = (C0[i0] + C1[i1]) · S`，所以 `∂L/∂C_b[k]` 是所有用到码字 k 的位置上 `∂L/∂Ŵ · S` 之和；`∂L/∂log(row_gain)` 是该行 `Σ ∂L/∂Ŵ ⊙ Ŵ`。一个码字被成千上万个向量共享，参数少，80 个 context 不容易过拟合。

| 项 | 定义 | 作用 |
|---|---|---|
| **A** | `mean((a_Q[:10,:6] − a_T[:10,:6])²)` | 执行的动作像 teacher |
| **Z** | `‖z_Q[:,:,1:] − z_T[:,:,1:]‖² / ‖z_T[:,:,1:]‖²`，排除 frame0 | 想象的未来像 teacher |
| **K** | reader-consistency KL | 未来里对动作重要的误差加重惩罚 |

`L = A + λZ + βK`，`λ = 0.04245`，`β = 0.1 × λ × mean(Z₀) / mean(K₀) = 0.006084`（初始 C80 上标定一次后冻结）。代码：[reader_loss.py](../../experiment/idea-validation/fastwam-vq-c80-frozen-reader-kl/reader_loss.py)。

**Reader KL 直觉。** Z 对 future 的所有误差一视同仁，但 action expert 只“读” future 的一部分信息。K 把 student 的 future `z_Q` 交给**原始 BF16 模型的 action 通路**（video prepare/prefill + 10 步 action 去噪，frame0 用真实观测），看读出的动作是否还和 teacher 一样：

1. 读出 `R_T(o, z_Q)`；喂 teacher 自己的 future 时 `R_T(o, z_T) = a_T`，逐位验证为 0 差。
2. 前 10 步 × 6 维每个 scalar：C80 统计标准化 → σ=0.375 高斯平滑 → 16 bin（内边界 [−3.5, 3.5]，两端无穷）→ `KL(p_teacher ‖ p_student)` → 60 个 scalar 平均。

代码：[frozen_reader.py:121](../../experiment/idea-validation/fastwam-vq-c80-frozen-reader-kl/frozen_reader.py)、[softbin_loss.py:82](../../experiment/idea-validation/fastwam-vq-dual-kl/softbin_loss.py)。

**数学上。** 在 teacher 附近 `K ≈ ½ δzᵀ Jᵀ F J δz`，`J` 是 reader 对 future 的 Jacobian，`F` 是 soft-bin 分布的 Fisher：用动作通路的敏感度给 future 误差加权的 Mahalanobis 距离。它和 Q-WAM 的 AOG（`JᵀJ`）同类；Q-WAM 在 activation 空间找敏感方向开 16-bit 分支，我们在 future latent 输出上把它当训练损失作用于 codebooks。

**为什么用冻结 BF16 reader。** 用 student 自己的 reader，future 误差可以和 reader 误差互相凑；BF16 reader 是固定、无偏的裁判。部署时量化 reader 自身的误差由 A 项约束。

**空白：** 没有 reader-MSE 消融，不声称“KL 形式优于 MSE”。

### 2.6 训练流程：分阶段 VJP

student 完整采样图和 BF16 reader 图同时保留反向激活放不进显存，所以每个 10-context batch 分三段（[worker.py:267](../../experiment/idea-validation/fastwam-vq-r128-c80-loss-pair/worker.py)）：

1. student no-grad 前向，缓存 action/future，在 FP32 叶子上求 A、Z 的 cotangent。
2. reader session：student dense weight 停到 CPU，换上原 BF16 weight，求 `β·K` 对 future 的 cotangent；frame0 梯度严格为 0。
3. 恢复 student 开梯度重放，要求逐位一致；两组 future cotangent 在 FP32 相加，一次 backward 到 codebooks。

10 个 context 梯度平均 → clip 1 → Adam(lr 5e-4)；第一步后清空 moments。**C80（40 tasks × 2 条 demo 快照）分 8 个 batch，8 次更新、80 次曝光。**

### 2.7 Cosmos-Policy 上的差异

Shared DiT，action、proprio、future 都是 latent slot。280 个 attention/MLP Linear（1.76B weights）；R8 → VQ8；action 损失用完整 16×6；future 是 slot 6/7；reader 在第 3 次 NFE 前切开，用 student 的 noisy future slots 替换，由冻结 DiT 跑原 solver 剩余步。`λ=0.0646`、`β=1.73`。代码：[reader.py](../../experiment/idea-validation/cosmos-policy-vqwam-20261024/reader.py)。

---

## 3. 当前 Inference Engine 实现

> 路线 3 会替换其中“解码到 BF16 scratch + cuBLAS BF16”这一段，condition cache 和 CUDA Graph 保留。

### 3.1 要解决的问题

MXVQ 把 target weight 压到 3.76×，但推理时把 614 层都解码成 dense BF16 常驻，显存一点不省（Q0 eager dense：25.14 GB，6.53 s）。试过通用 Triton packed VQ-GEMM：p50 **7.09 s**，比 eager 还慢；输出相对 eager 偏差 first10 RMSE **0.045**。放弃，改为每层临时解码成 BF16 交给 cuBLAS。

### 3.2 单个 Linear 的数据流（C+decoder+buffer）

```
x (BF16, ≤294×14336)
 ├─[Triton] Smooth 除法 + signs + FWHT-128 (FP32) ─► H buffer ┐  H/U buffer 借用 weight scratch
 ├─[torch.matmul] × U (FP32 128×128 块) ───────────► U buffer ┘
 └─[Triton] per-row A4 (round-half-even) ──────────► A4 buffer（独立）

packed codes (常驻 GPU)
 └─[Triton] decode：FP32 (C0[i0]+C1[i1]) × ldexp(row_gain, e−127) → BF16
                   ─► 共享 weight scratch（113,246,208 B，= 最大一层）

cuBLAS F.linear(A4 operand, scratch weight, bias) ─► y
```

代码：[abc_runtime.py:120](../../experiment/idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/abc_runtime.py)、[abc_runtime.py:167](../../experiment/idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/abc_runtime.py)、[fused_vq.py:104](../../experiment/idea-validation/fastwam-vq-r128-c80-fused-tuning/fused_vq.py)。

**逐位一致：** decode 保持训练时完全相同的运算顺序（FP32 码字相加 → ldexp → 乘 → 一次 BF16 cast，`enable_fp_fusion=False`）；A4 kernel 手写 round-half-even 对齐 `torch.round`。

### 3.3 Query 级条件缓存

一次 query 内 20 次去噪，text context 和 proprio 不变，原代码每步重算各 block 的 cross-attn K/V 投影。缓存后目标 Linear 调用 **6436 → 5258 次/query**，3.69 → 3.05 s。代码：[condition_cache.py:9](../../experiment/idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/condition_cache.py)。

### 3.4 CUDA Graph

每个 Linear 是 4–5 个小 kernel，一次 query 两万多次 launch，batch 1 时 GPU 主要在等 host。对 `video_step`、`video_prefill`、`action_step` 各录一个 Graph（[graph_runtime.py:70](../../experiment/idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/graph_runtime.py)），replay 前把新输入 copy 进固定地址。**3.05 → 0.957 s**，计算量没变，省的全是 host 开销。

### 3.5 阶段结果（同 Q0、同 80 个固定 query、A100）

| Runtime | p50 (s) | Peak allocated (GB) | 数值 |
|---|---:|---:|---|
| 原始 BF16（released pipeline） | 1.450 | 25.13 | 参考 |
| Q0 eager dense | 6.528 | 25.14 | 量化参考 |
| Layer decode（常驻 packed） | 3.533 | 16.55 | 全等 |
| B：workspace + cache | 3.047 | 16.69 | 全等 |
| C：B + CUDA Graph | 0.957 | 16.95 | 全等 |
| **C + decoder + buffer** | **0.887** | **16.91** | **全等；action/future anchor 0** |

标准 LIBERO 400 episodes 全程在此 engine 上运行，6415 次 policy call，p50 **0.900 s**。

### 3.6 显存账

| 项 | 大小 |
|---|---:|
| Target 编码（含 S/signs/U） | 2.99 GiB |
| 非量化参数 | 11.90 GiB（T5 10.58、VAE 1.31） |
| Query peak allocated | 16.91 GB |

peak 的大头是 T5。同一 episode 内 instruction 不变，T5 输出可按 episode 缓存、本体放 CPU；16.91 GB 减去 T5 的 11.36 GB，peak 预计约 6 GB（需实测）。这与路线 3 无关，任何版本都可以做。

### 3.7 局限

0.887 s vs 1.45 s 的 1.64× 来自 cache 和 Graph，不是低比特计算。阶段 0 实测（§11）：BF16 加上同样的 cache 和 Graph 是 **541 ms**，输出与 BF16 eager 逐位一致；当前 Q0 engine 896 ms，**比同等优化的 BF16 慢 355 ms（1.66×）**。这 355 ms 是逐层解码（每 query 写出 107 GiB BF16）加上 Smooth/H、FP32 U 和 A4 这些额外 kernel 的开销。

---

## 4. A4 问题

### 4.1 数据：A4 是主要误差来源

同一份 MXVQ 权重（R0 + VQ，A+Z 训练 8 步），held-out 40×2：

| 配置 | first10 动作 RMSE | future NMSE | 280 闭环 |
|---|---:|---:|---:|
| BF16 | — | — | 216 |
| BF16 权重 + A4（H-W16A4） | — | — | 204 |
| MXVQ 权重 + A4（Seq8；Q0 同设置） | 0.0442 | 0.0514 | 208（Q0） |
| **同一份 MXVQ 权重，关掉 A4** | **0.0231** | **0.0227** | 未测 |

来源：[transport RESULTS](../../experiment/idea-validation/fastwam-vq-r128-c80-transport/results/25921642.pbs101/RESULTS.zh.md)、[loss-pair-plus RESULTS](../../experiment/idea-validation/fastwam-vq-r128-c80-loss-pair-plus/RESULTS.zh.md)。

- 关掉 A4，动作误差减半，future 误差降 56%。4.27 bpw 的 VQ 权重本身代价不大。
- 校准的主要工作是用 codebook 补偿 A4 误差：同 A4 下 BF16 权重 203/204，Q0 208。
- 当前 engine 的 GEMM 是 BF16，A4 在部署中只有精度代价。

### 4.2 量化的三种收益

weight-only 方法（AWQ/GPTQ 的 Marlin 类 kernel、AQLM、QuIP#）推理时也用 FP16/BF16 做乘法，加速来自**少读字节**：在寄存器或 shared memory 里把 4-bit 权重解码成 FP16 直接喂 tensor core，dense 权重从不写回显存。这只在 memory-bound 时有效。

一个 GEMM 的耗时约为 max(计算时间, 搬运时间)。权重主导搬运量时，算术强度约等于 **M FLOP/Byte**（BF16 权重，M 是 token 行数）。A100 拐点约 150–200 FLOP/Byte。

| | video expert | action expert |
|---|---|---|
| M | 294 个 video token | 约 32 个 action token |
| 算术强度 vs 拐点 | ≈294，偏 compute-bound | ≈32，memory-bound |
| BF16 profile 的 GPU kernel 时间（10 步） | 413 ms（约 72%） | 158 ms |
| weight-only 融合 kernel 能否加速 | 基本不能 | 理论上能；但实测单次 GEMM 只有 10–20 µs，是 latency-bound，INT8 dense 也只有 0.93×（§11） |

| 收益 | 作用在哪 | 需要什么 | Q0 + 当前 engine |
|---|---|---|---|
| 容量 | 整个模型 | 压缩存储 | 有，25.1 → 16.9 GB |
| 带宽 | memory-bound GEMM | 片上解码的融合 kernel | 没有，流量更多 |
| 算力 | compute-bound GEMM | 两个 operand 都是低比特，走低比特 tensor core | 没有 |

A4 唯一站得住的理由是第三种。但它要求权重也落在 tensor core 能直接用的格点上；VQ 解码出的 codebook 之和是任意 BF16 值，喂不进 INT4/FP4 tensor core。所以 Q0 付了 A4 的精度代价，格式上拿不到 A4 的算力收益。

### 4.3 三条路线

1. **放弃算力 claim**：部署用 weight-only，A4 只为同口径比较。“为什么要 A4”仍然只能答“为了可比”。
2. **只补带宽**：写片上解码的 VQ×BF16 kernel，只对 memory-bound 的 action expert 有效。**不可行**：三个目标模型里 Cosmos-Policy 和 LingBot-VA 都是单一 DiT，video 和 action token 走同一组 GEMM，没法分开处理。
3. **拿到算力收益（选定）**：VQ 解码成 8-bit，走 INT8/FP8 tensor core，配 A8。见 §5。

---

## 5. 路线 3：W4A8 级 VQ（VQ 存储 + 8-bit 计算）

### 5.1 硬件依据

| GPU | BF16 | INT8 | FP8 | INT4 / FP4 | 硬件 block-scaled MMA |
|---|---|---|---|---|---|
| A100（我们的集群） | 1× | 2× | 无 | INT4 4×，两侧都必须是 INT4 | 无 |
| H100 | 1× | 2× | 2× | 无 INT4 | 无 |
| Blackwell（B200 / RTX 50） | 1× | — | 2× | FP4 4×，两侧都必须是 FP4 | 有：MXFP8/6/4、NVFP4 |

- 权重解码成 8-bit，MMA 就按 8-bit 速率跑，activation 再降到 4-bit 不会更快；Blackwell 的 FP8×FP4 混合也按 FP8 速率。**对 VQ 权重，A8 是唯一合理搭档。**
- SteerQuant 的数据：W4A8 拿到 W4A4 大部分加速（Cosmos-Policy 1.96× vs 2.23×，Cosmos-Edge 1.56× vs 1.69×），FastWAMJoint 差得多些（1.38× vs 1.86×）。
- 命名：文献中的 W4A8（QServe、QQQ）也是 4-bit 存储、寄存器里反量化成 INT8、INT8 MMA。我们只是把“反量化”换成“查表相加”，属于 **W4A8 级**，不叫 W8A8。

**一个融合 kernel 同时拿两种收益：** memory-bound 的 GEMM 读 4.27 bit 而不是 16 bit（约 3.7× 少流量）；compute-bound 的 GEMM 走 INT8（A100 上 2× BF16 峰值）。kernel 不关心 token 来自 video 还是 action，单一 DiT 同样适用。

### 5.2 格式：整数 codebook

```
现在：  Ŵ = BF16( (C0[i0] + C1[i1]) · row_gain · 2^(e−127) )

改成：
  w_int8 = C0[i0] + C1[i1]                 # C0、C1 元素为整数，和落在 INT8 范围
  Y = (A_int8 · W_int8ᵀ)_INT32 · a_scale[token] · row_scale[row] + bias   # scale 全在 epilogue
```

**为什么必须是整数 codebook。** 解码在 kernel 内循环里做。A100 上 INT8 tensor core 吞吐约为 CUDA core 的 32 倍。video GEMM 中每个权重在一次 GEMM 里只做 294 次乘加，折合约 9 条 CUDA core 指令的时间；按 128 行切 tile 要解码 3 次。浮点解码（查表、相加、ldexp、取整、类型转换）每次约 10 条指令，3 次 30 条，解码成为瓶颈；整数解码每次 1–2 条，能藏在 tensor core 时间里。

**d4 × INT8 = 32 bit。** 一次 shared memory 读取拿到一个码字，两次读取加一条 packed add（4 路 INT8 SIMD 加法）得到 4 个权重。两本 codebook 共 2 KB。

**精度上为什么整数格点够用。** 4 bpw 的 VQ 本身误差约为 0.07σ（σ 为权重标准差）。若 INT8 每行格点用满 ±127，格点步长约 0.03σ，取整误差约 0.009σ，可以忽略。

**整数累加的附带好处。** INT32 累加与顺序无关、逐位确定，旧 fused kernel 的数值漂移（Triton BF16 GEMM 累加顺序）在这条路径上不存在；训练侧用 `torch._int_mm` 做逐位参考。

**难点：microscale 指数。** INT8 每行一套固定格点，B32 的 2 的幂指数无法无损塞进去：一行里 block 幅度相差 2^D 倍，整数 codebook 的取值范围就要缩小 2^D 倍，小幅度 block 分辨率变粗。选项：

| 选项 | bpw | 解码开销 | 精度风险 |
|---|---:|---|---|
| a. 去掉指数，只留 row scale | ≈4.02 | 最低 | microscale 在 A4 下带来过 −8%，A8 下需重测；Hadamard 后行内幅度已较均匀 |
| b. 小范围移位（每 block ×1/×2） | ≈4.27 | +1 条移位 | codebook 范围减半 |
| c. K 循环内每 32 元素 FP32 重缩放 | ≈4.27 | 在 A100 上接近 GEMM 本身 | 不建议 |

阶段 0 先测 a 的精度代价再定。Blackwell 上这个矛盾消失（§5.5）。若最终选 a，方法名里的 “Microscaling” 需要重新考虑。

### 5.3 Activation：A8

- per-token INT8（absmax/127），Smooth + Hadamard 融合进量化 kernel，输出 INT8 + per-token scale。
- 考虑去掉 U：U 在 A4 下就没有闭环收益（204 vs 203），且跑在 FP32 SIMT 路径上，A8 下更不值。
- 权重在新坐标下重新拟合，去掉 U 不影响整体流程。

### 5.4 Kernel 设计（A100 / H100 / Ada 通用的 INT8 路径）

1. **Activation 量化 kernel：** x → Smooth 除法 + signs + FWHT → per-token absmax → INT8 + FP32 scale。
2. **融合 VQ-INT8 GEMM（Triton）：**
   - 每个 CTA 把本层两本 codebook（2 KB）载入 shared memory；
   - K 循环内：读 uint8 index tile → gather 码字 → packed add（必要时移位）→ 得到 INT8 权重 tile → `tl.dot(int8, int8) → int32`；
   - epilogue：`× a_scale[token] × row_scale[row] + bias` → BF16。
3. **参考与验证：** 用 `torch._int_mm` 在 PyTorch 侧构造逐位参考；全部 shape 做逐位比对。
4. **接回 engine：** condition cache 和 CUDA Graph 照用；BF16 weight scratch 不再需要，每 query 107 GiB 的写流量也消失。
5. **先例：** QServe、QQQ 已证明“寄存器里 W4 → INT8 反量化 + INT8 MMA”能做得快；我们是同一模式，反量化换成查表。

### 5.5 双侧 microscaling 与 Blackwell（MX 版本）

双侧同长度 microscaling 就是 OCP MX 格式：权重和 activation 沿 K 方向每 32 个元素各有一个 2 的幂（E8M0）scale，tensor core 累加时自动乘上。

- **Blackwell：** 硬件原生，缩放零开销。现有 B32 指数直接对应 MX 的权重 scale，可做 MXFP8×MXFP8 版本；activation 的块级 scale 比 per-token 细得多，A8 精度几乎无损。microscale 在这里保留得最完整。
- **A100/H100：** 没有硬件支持，两侧块级 scale 意味着每 32 个元素把 INT32 部分和转 FP32 再乘 scale，开销接近 GEMM 本身。DeepSeek-V3 在 H100 上做 FP8 细粒度缩放时把块大小定为 128，正是因为这个开销。
- **加速来源：** 缩放本身不加速，它让更低位宽保住精度。FP4 速率要求两侧都是 FP4，VQ 解码不到 FP4 格点上，所以 Blackwell 上我们拿到的是 MXFP8 速率（2× BF16），与 A100 INT8 同档。

结论：A100 主实现用 per-row（W）× per-token（A）的 epilogue scale；Blackwell MX 版本作为加分项。autodl 若能租到 RTX 5090，可用于 Blackwell 部署演示。

### 5.6 bpw 旋钮

bpw ≈ 各 codebook index 位数之和 ÷ 向量维度（+ B32 指数 0.25，若保留）：

| 配置 | bpw（不含指数） | 定位 |
|---|---:|---|
| d4，2×256（当前） | 4.02 | 对齐 W4（W4A4 方法：Q-WAM 4.62，QuantWAMs 名义 4.8） |
| d4，1×4096 或 2×64 | 3.02 | W3 档 |
| d8，2×256 | 2.02 | W2 档 |

4 bpw 已与 W4A4 方法存储对齐，不需要为对齐调 codebook。旋钮应往低调：低 bpw 是 VQ 相对标量量化的主场。

### 5.7 对比组

| 组 | 存储 | 计算 | 方法 | 要证明什么 |
|---|---|---|---|---|
| **主对比：W4A8** | ~4 | INT8/FP8 | SmoothQuant W4A8、QServe/QQQ 式 W4A8、ViDiT-Q W4A8、PTQ4DiT/Q-DiT W4A8、SteerQuant W4A8 报告值 | 存储和算力都相同时精度更高，尤其在 LIBERO-Plus 难扰动和 future 保真上 |
| 存储相同、计算位宽更低：W4A4 | 4–4.8 | INT4/FP4 | 已有 QuaRot/SpinQuant/SVDQuant/ViDiT-Q 适配版；三篇同期 W4A4 报告值 | 精度明显更高，代价是计算 8 对 4 |
| 计算相同、存储翻倍：W8A8 | 8 | INT8 | SmoothQuant W8A8、ViDiT-Q W8A8 | 显存减半，精度持平 |
| 存储相同、无计算加速：weight-only | ~4 | BF16 | GPTQ/AWQ W4A16、AQLM、QuIP# | 精度持平，额外拿到 INT8 计算加速 |
| 低 bpw | 2–3 | INT8 | 标量 W3A8/W2A8、AQLM/QuIP# 2–3 bit 配 A8 | VQ 的主场，标量方法在这一档会崩 |

论文主图：Pareto 图，横轴 bpw，纵轴 LIBERO-Plus 成功率，每个点标注计算位宽。三篇同期都没公开代码，只能比报告值且协议不同；LingBot-VA 让我们能和 QuantWAMs（LIBERO-Long 98.5→98.0）、Q-WAM（RoboTwin）在同一模型上对照报告值。

### 5.8 风险

- **C1 可能变弱。** A8 让整体误差变小，future 保真在 4 bpw 下效应可能不明显，很可能要到 3/2 bpw 才重新显现。C1 的固定输入诊断要在新设置下重做（便宜）。
- **整数 codebook + 去掉 microscale 的精度代价未知。** 阶段 0 先测 INT8 格点部分。
- **A100 上 INT8 dense 对我们的 shape 未必真有 2×。** 阶段 0 先测速度上限。
- **kernel 是最大工程量。** 旧 fused kernel 又慢又漂移，但那是 BF16 MMA + 浮点解码；整数路径更简单且有先例。
- **4 bpw 下可能没有可见优势。** W4A8 标量方法在标准 LIBERO 已近无损；必须依赖 LIBERO-Plus 和低 bpw 拉开差距。

---

## 6. 评测设定：以 LIBERO-Plus 为主

### 6.1 为什么用 LIBERO-Plus

标准 LIBERO 上同期方法相对 FP 只差 −0.2 到 −1.0 pp（QuantWAMs Fast-WAM 97.6→97.4，SteerQuant FastWAMJoint 98.9→98.4、Cosmos-Policy 98.5→98.3），我们自己 Q0 是 391/400（97.75%，Fast-WAM 论文 BF16 IDM 报告 98.0）。天花板效应下区分不出方法好坏。

LIBERO-Plus 的扰动能把差距拉开。现有 280 面板（4 suites × 7 扰动维度 × 10 variants，每 variant state0 一条配对 episode）：

**Fast-WAM-IDM（每维度 40 条）：**

| 扰动维度 | BF16 | QuaRot | SpinQuant | Smooth/H W4A4 | ViDiT-Q | H-W16A4 | Ours A+Z | **Ours A+Z+K** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Camera viewpoints | 21 | 12 | 16 | 14 | 14 | 14 | 14 | **19** |
| Sensor noise | 27 | 21 | 25 | 24 | 23 | 25 | 24 | 24 |
| Robot initial states | 34 | 30 | 31 | 30 | 32 | 30 | 33 | 31 |
| Background textures | 26 | 24 | 23 | 26 | 26 | 25 | 24 | 26 |
| Language instructions | 38 | 36 | 36 | 37 | 37 | 38 | 38 | 39 |
| Light conditions | 35 | 36 | 39 | 35 | 37 | 37 | 36 | 35 |
| Objects layout | 35 | 36 | 33 | 34 | 34 | 35 | 35 | 34 |
| **合计 /280** | **216** | 195 | 203 | 200 | 203 | 204 | 204 | **208** |

SVDQuant 适配版 108/280。来源：[loss-pair-plus summary.json](../../experiment/idea-validation/fastwam-vq-r128-c80-loss-pair-plus/summary.json)。

- W4A4 baselines 掉 13–21 条（4.6–7.5 pp），远大于标准 LIBERO 上的差距。
- 损伤集中在 **camera viewpoints**（−5 到 −9/21）、**sensor noise**（−2 到 −6/27）、**robot initial states**（−2 到 −4/34）。
- Camera viewpoints 是各臂波动最大的维度（12–19/40）。A+Z → A+Z+K 在这里 14 → 19（5 胜 0 负），但 R0-W16A4（不量化权重、无 K）和未训练的 Micro VQ 也到 19，所以不能把这 5 条归功于 K。这一维度需要更大的评测量才能下结论。

**Cosmos-Policy（每维度 40 条）：**

| 扰动维度 | BF16 | SpinQuant | QuaRot | ViDiT-Q | Ours |
|---|---:|---:|---:|---:|---:|
| Camera viewpoints | 29 | 26 | 26 | 29 | 28 |
| Robot initial states | 22 | 19 | 21 | 21 | 21 |
| Objects layout | 36 | 34 | 35 | 34 | 35 |
| Background textures | 31 | 29 | 31 | 32 | 32 |
| Language instructions | 38 | 36 | 35 | 37 | 37 |
| Light conditions | 40 | 40 | 40 | 40 | 39 |
| Sensor noise | 37 | 37 | 39 | 39 | 38 |
| **合计 /280** | **233** | 221 | 227 | 232 | 230 |

SVDQuant 适配版 78/280。来源：[Cosmos INTERIM_RESULTS](../../experiment/idea-validation/cosmos-policy-vqwam-20261024/INTERIM_RESULTS.zh.md)。Cosmos-Policy 对量化更鲁棒，W4A4 下 ViDiT-Q 已只差 1 条，优化空间主要在 Fast-WAM-IDM（及待测的 LingBot-VA）。换成 A8 后各方法差距还会缩小，所以低 bpw 档很关键。

### 6.2 Calibration 数据来源

- C80 全部来自**原始 LIBERO** demonstrations：40 tasks × 2 条不同 demo × 每条 1 个快照（early/middle/late 分层），不含任何 LIBERO-Plus 扰动。
- 评测在 LIBERO-Plus 扰动变体上进行，task 与 C80 相同（同 40 tasks），扰动从未在校准中出现。论文中可表述为：**只用干净 demonstration 校准，在未见过的扰动下评测鲁棒性**。不是 unseen-task 泛化。
- 已观察到：用原始 episode 校准优于加入扰动的数据。暂不深究，留作后续 ablation。

### 6.3 统计功效与规模

280 条配对 episode 下，单个成功率的标准误约 2.5 pp；Q0 vs A+Z 的 14 胜 10 负，McNemar p≈0.54。论文主表建议扩大 LIBERO-Plus 评测（例如每 variant 多个 initial state 或更多 variants，到 1000–1400 条配对 episode），并报告配对检验。engine 让大规模评测变得负担得起：标准 LIBERO 400 条在 4 卡上只用了 1 h 23 m。

### 6.4 三个目标模型

| 模型 | 结构 | test-time 生成 future | 状态 |
|---|---|---|---|
| Fast-WAM-IDM | video expert + action expert（MoT） | 是，先 video 10 步再 action 10 步 | 主平台，全部现有结果 |
| Cosmos-Policy | 单一 DiT，action/future 为 latent slot | 是，联合去噪 | 有 reader、engine 适配，六臂闭环完成 |
| LingBot-VA | 单一 DiT | 是（video 20 步、action 50 步） | 待接入：runtime、reader 切点、engine 都要新写 |

---

## 7. 与三篇同期工作对照

| | QuantWAMs | Q-WAM | SteerQuant | **Ours（当前 Q0 → 路线 3）** |
|---|---|---|---|---|
| Fast-WAM 变体 | 无 test-time future | 无 test-time future | FastWAMJoint（联合去噪） | **Fast-WAM-IDM（先想象再执行）** |
| 其他模型 | LingBot-VA | ImageWAM, LingBot-VA | Cosmos-Policy, Cosmos-Edge | Cosmos-Policy，计划 LingBot-VA |
| Weight 格式 | INT/NVFP4 标量，20% 层升 W8A8 | INT4 g32 / NVFP4 + rank-32 16-bit 分支 | per-channel INT4 | VQ（整数 codebook），无高精度例外 |
| Activation | A4，2% 通道 BF16，1/10 action 步 A8 | A4 | A8 / A4 | Q0：A4 → 路线 3：A8 |
| 有效 bpw | 名义 4.8（按层数平均） | 4.62（Fast-WAM） | ~4 | 4.27 → 4.02 / 3.02 / 2.02 |
| 校准目标 | 联合 video–action loss 的 Fisher；闭环 replay 选保护步 | 最终 action 的 Jacobian Gramian | 一阶 action-impact 加权的局部重建 | teacher action + 想象的 future + reader KL，端到端训练 codebooks |
| Future 的角色 | 训练 loss 的一部分 | 不生成 | 误差传到 action 的通道 | **独立保真目标** |
| 校准数据 | 32 轨迹 + 32 FP16 rollout + 验证 rollout | 50 episodes，≈1.3×10⁵ 次 backward | 100–320 observations | 80 个原始 LIBERO demo 快照，8 次更新 |
| Kernel | NVFP4/FP8，Blackwell | INT4 / NVFP4 融合 | Rudder INT 融合，RTX 4090 | Q0：decode→BF16 cuBLAS → 路线 3：VQ→INT8 融合 GEMM |
| 显存口径 | 目标 blocks：14.4→4.2 GB | 目标 blocks：11.85→3.44 GB | 整 DiT peak：12.1→3.4 GB | 整进程 peak：25.1→16.9 GB（Q0） |
| 速度口径 | 目标 blocks 1.4–1.6× | 指定层 1.06×（Fast-WAM） | DiT replay 1.86–2.23× | 整次 policy call（Q0 对 released BF16 1.64×） |
| 主评测 | RoboTwin、LIBERO | RoboTwin | LIBERO、RoboLab | **LIBERO-Plus**（+ 标准 LIBERO） |
| 真机 | AgiBot G2 | G1 + UR3 | AgileX（W4A8） | 无 |

我们的差异点：① 唯一把 test-time 想象的 future 当保真目标；② 唯一用非均匀 VQ，bpw 计入全部开销；相对 QuantWAMs（W8 层 + BF16 通道）和 Q-WAM（16-bit 分支）没有高精度例外；③ 显存和延迟报整进程 / 整次 call；④ 主评测在更有区分度的 LIBERO-Plus。

劣势：没有真机；路线 3 的 kernel 尚未实现，目前没有低比特计算加速。

---

## 8. 三条贡献（按路线 3 起草）

方括号是待实验填写的数字或待定的名字。“现有证据版”是按目前数据能写到的强度。

### C1：Future-latent fidelity

> We identify the imagined future as a main carrier of quantization error in imagine-then-execute WAMs. Calibrating on actions alone fits calibration actions better but degrades the imagined future and generalizes worse, and restoring only the full-precision future removes [34–42]% of the action error. We therefore make fidelity of the imagined future an explicit quantization target alongside action fidelity.

- 定位限定在 imagine-then-execute WAM；三个目标模型都在 test-time 生成 future。
- 现有证据（旧 row-scale VQ、A4 下）：

| 证据 | 数字 | 来源 |
|---|---|---|
| 只用 action 蒸馏：calibration 变好，held-out 和 future 变差 | calib first10 0.0467→0.0443；test first10 +1.5%、late22 +5.8%；future NMSE 0.084→0.093 | action-distill + latent-diagnostic |
| 同预算加 Z | test first10 0.0581→**0.0530（−8.9%）**；future NMSE 0.093→**0.072** | action-latent |
| 只把 future 换成 BF16 版本 | 动作 RMSE first10 **−34%**、full32 **−41%**、late22 **−42%** | latent-diagnostic |
| 冻结 BF16 reader 读不同 student 的 future | A-only 0.0412 / A+Z 0.0351 | future-reader |
| 只缩放 future 误差幅度 | 复现 A→A+Z 收益的 **84%** | future-nmse-matched |
| 误差分解（held-out） | action MSE 的 **27%** 经由 future 进入 | c80-frozen-reader-kl |

- 路线 3 下要重做：固定输入诊断（A8、各 bpw 档）+ matched A-only 臂的闭环。等闭环出现差距再把 “an explicit target” 升级为 “a key target”。

### C2：VQ 存储 + 8-bit 计算的校准框架

> We propose [Name], a W4A8-class post-training quantization framework for WAMs that stores weights as additive vector quantization with integer codebooks and computes in 8-bit. All transformer Linears are encoded at [4.0 / 3.0 / 2.0] bits per weight with no high-precision exceptions, and activations use 8-bit per-token quantization. Codebooks and row scales are the only trainable parameters; they are calibrated end-to-end through the quantized sampler against the teacher's executed action, its imagined future, and a reader-consistency KL that scores the student's future through the frozen teacher's future-to-action pathway. Calibration uses only 80 clean demonstration snapshots.

- 统一叫 **reader-consistency KL**，不用 “relational”。
- “joint” 的准确含义：codebook 是唯一可训练参数，同时吸收 weight 和 activation 的量化误差。
- 方法名待定；若 microscale 指数在 INT8 路径上被去掉，名字里不再用 “Microscaling”。
- 现有证据（Q0，A4 下）：matched A+Z 204 → A+Z+K 208（14 胜 10 负）；280 面板上量化臂最高；Cosmos 230 vs BF16 233。路线 3 下需全部重测。
- 不声称校准目标与量化器无关：同样的 loss 套在 AQLM 初始化上训练发散（闭环 125/245 vs reconstruction 179/245）。

### C3：VQ-INT8 融合 kernel + WAM 推理 runtime

> We co-design the codebook format with a fused kernel that decodes codewords on-chip directly into INT8 tensor-core operands, so one kernel cuts weight traffic on memory-bound layers and doubles peak throughput on compute-bound layers, without separating video and action tokens. Together with query-level condition caching and CUDA-graph replay of the fixed-shape denoising loop, the runtime [reduces peak memory from X to Y GB and policy-call latency from X to Y s against an equally optimized BF16 runtime] on Fast-WAM-IDM, Cosmos-Policy and LingBot-VA.

- 适用范围（写在方法章节）：任意 WAM 的 Linear；格式上只要求码字能解码成 INT8/FP8；面向 batch-1 部署；Blackwell 上有 MX 版本。
- 对比基线必须是**同等 cache/Graph 优化的 BF16**（实测 541 ms），不再用 released BF16 pipeline 当分母。
- cache + Graph 本身是对任何 WAM 都成立的无损优化：Fast-WAM-IDM 上 1475 → 541 ms（2.73×），输出逐位一致。可以作为 runtime 部分的一项，但它不是量化贡献，也不新颖（CUDA Graph 与条件缓存都是常规手段）。
- 现有证据（Q0 engine）：显存 25.13 → 16.91 GB、逐位一致、400 条闭环全程在 engine 上运行；但延迟比同等优化 BF16 慢 1.66×，C3 的速度部分只能等路线 3 的融合 kernel。

---

## 9. 证据地图

| Claim | 现在能说 | 还不能说 | 补什么 |
|---|---|---|---|
| C1 future 承载量化误差 | A4 下固定输入：换回 BF16 future 去掉 34–42% 动作误差；只调 future 幅度复现 84% 收益 | A8/低 bpw 下是否仍成立；闭环上 Z 必不可少 | 新设置下诊断；matched A-only 闭环 |
| C1 与无 imagination WAM 对比 | 定性 | “IDM 里 video expert 承担 X% 误差，vs Q-WAM 的 10.7%” | video-only / action-only 量化诊断 |
| C2 精度优势 | A4 下 LIBERO-Plus 280 面板量化臂最高（208） | A8 下、同 W4A8 档的优势 | 路线 3 全套 + W4A8 baselines |
| C2 低 bpw 优势 | — | 全部 | 3/2 bpw 档 vs 标量 W3A8/W2A8 |
| C2 reader K 有用 | matched 204→208（14 胜 10 负） | 统计显著；归因到具体扰动维度 | 扩大 Plus 评测；A8 下重测 |
| C3 显存 | 整进程 −32.7%，逐位一致（Q0） | 路线 3 kernel 的显存 | kernel 实现后测；T5 缓存 |
| C3 速度 | 对 released BF16 1.64×（全部来自 cache/Graph）；对同等优化 BF16 慢 1.66× | 相对同等优化 BF16 更快 | 融合 kernel 实测；消费级 GPU 测速 |
| 跨模型 | Cosmos 230 vs 233 | LingBot-VA 任何结果 | LingBot-VA 接入 |
| 真机 | — | 任何真机结论 | AgileX |

---

## 10. 实施顺序

**阶段 0：go/no-go 检查，都不需要训练。**

- **速度上限：** A100 上 cuBLASLt INT8 dense GEMM vs cuBLAS BF16，用 Fast-WAM 17 种真实 shape（M=294 与 M≈32）；同时测 BF16 + 同样 cache/Graph 的整次 policy call。若 INT8 dense 在这些 shape 上没比 BF16 快多少，路线 3 的速度前提不成立。
- **精度：** Fast-WAM held-out 40×2 固定输入，四个臂：BF16 权重 + A8；Q0 权重 + A8；Q0 权重取整到每行 INT8 格点 + A8；同上去掉 U。看 A4 → A8 回收多少，INT8 格点和去 U 各损失多少。

**阶段 1：格式与校准（Fast-WAM）。**
- 整数 codebook 重新拟合；A+Z+K 损失不变，前向换成 INT8 STE。
- 4.02 bpw 起步，再做 3.02、2.02。
- 消融梯子：A / A+Z / A+Z+K（matched 初始化与预算）。
- 评测：固定输入诊断（含 C1 的 future 诊断）+ LIBERO-Plus 闭环。

**阶段 2：kernel 与 runtime。**
- Triton 融合 VQ-INT8 GEMM + activation 量化 kernel；`torch._int_mm` 逐位参考。
- 17 种 shape 对比 cuBLAS BF16 与 cuBLASLt INT8 dense。
- 接回 engine（cache + Graph）；整次 policy call 对比同等优化 BF16。
- 顺带做 T5 按 episode 缓存。

**阶段 3：移植。** Cosmos-Policy（已有 reader/engine 适配）、LingBot-VA（新写 runtime、reader 切点、engine）。

**阶段 4：baselines 与评测规模。** W4A8 / W4A4 / W8A8 / weight-only 对比组；低 bpw 档；扩大 LIBERO-Plus 到 1000+ 条配对 episode；calibration 来源 ablation（原始 vs 扰动 episode）。

**之后：** AgileX 真机（SteerQuant 也用 AgileX）。

**阶段 0 状态（2026-10-27）：** 测速 `26028322.pbs101` 完成，结果见 §11。精度检查 `26028311.pbs101` 按用户要求在启动 1.5 分钟后取消（PBS F/141，无结果），并入阶段 1。

---

## 11. 阶段 0 测速结果

作业 `26028322.pbs101`，A100-SXM4-40GB，torch 2.7.1+cu128，walltime 16 分 38 秒，五个模式全部完成。目录：[fastwam-int8-speed-20261027](../../experiment/idea-validation/fastwam-int8-speed-20261027/pulled/26028322.pbs101/results/RESULTS.zh.md)。

### 11.1 整次 policy call（同 80 个固定 query、3 次 warmup、同一 allocation）

| Runtime | p50 ms | 去噪 p50 ms | peak allocated GB | 相对 BF16 eager | 相对 BF16+cache+Graph |
|---|---:|---:|---:|---:|---:|
| BF16 eager（released） | 1475.3 | 1407.1 | 25.13 | 1.00× | 0.37× |
| **BF16 + cache + Graph** | **540.9** | 476.2 | 25.49 | 2.73× | 1.00× |
| Q0 c_buffer engine | 896.1 | 828.0 | 16.91 | 1.65× | **0.60×** |

- 三者 80 个 action 与各自参照逐位一致（BF16+cache+Graph vs BF16 eager 最大差 0；两个历史复现最大差 0）。历史 p50 复现误差 1–2%。
- Q0 engine 比同等优化 BF16 多 355 ms。按 FLOP 和流量粗估：FP32 U 矩阵乘约 50–90 ms，逐层解码约 100–150 ms，Smooth/H 与 A4 的 kernel 约 50–80 ms（本作业没有逐项 profile）。

### 11.2 GEMM：INT8 dense（cuBLASLt）vs BF16（cuBLAS）

按每 query 的实际调用次数（cache 后 5258 次）加权：

| 分组 | BF16 ms/call | INT8 ms/call | INT8 加速 | 未融合 INT8（PyTorch 逐算子量化+缩放） |
|---|---:|---:|---:|---:|
| 全部 | 219.5 | 140.9 | **1.56×** | 506.1 ms（0.43×） |
| video（M=294） | 191.1 | 110.4 | 1.73× | 0.52× |
| action（M=32） | 28.4 | 30.4 | 0.93× | 0.20× |

主要形状：video 的 294×3072×3072（每 query 1991 次）1.77×，FFN 294×3072×14336 1.97×，294×14336×3072 1.48×。action 的 32×3072/4096×1024 只有 0.63–0.69×。

### 11.3 含义

1. **当前 engine 的速度说法必须改。** 它比同等优化的 BF16 慢 1.66×，“1.64×”只能归于 cache + Graph。
2. **video GEMM 是 compute-bound，INT8 有实收益**（1.73×），与 §4.2 的 roofline 判断一致。
3. **action GEMM 是 latency-bound，不是 bandwidth-bound。** 单次 10–20 µs，权重 6 MB 级、按带宽只需约 4 µs；INT8 和压缩权重都帮不上，能帮的是减少 kernel 数（融合）。§4.2 里“weight-only 对 action expert 接近 4×”的推断不成立。
4. **融合是必须的。** 用 PyTorch 逐算子做量化和缩放，每个 Linear 多出约 40 µs，整体反而慢 2.3×。
5. **路线 3 在 A100 上的端到端上限约 1.17×。** GEMM 只占 BF16+Graph 去噪时间的 46%（219/476 ms）；即使 VQ 解码、activation 量化、Hadamard 全部免费融合进 INT8 GEMM，也只能省约 79 ms（541 → 约 462 ms）。剩下的时间在 attention、norm/modulation、T5/VAE 编码等非 GEMM 部分。
6. **部署级 GPU 上 INT8 的相对收益更大。** A100 上 INT8:BF16 峰值是 2:1；GeForce 卡（RTX 4090）BF16 用 FP32 累加时只有半速，INT8:BF16 约 4:1。SteerQuant 在 RTX 4090 上测到 1.9–2.2×，部分原因在此。机器人部署常用消费级/边缘 GPU，值得在 RTX 4090/5090 上复测 GEMM 微基准。
7. **显存仍是路线 3 最稳的收益**：25.5 GB → 约 17 GB，T5 按 episode 缓存后约 6 GB 量级（待测）。

### 11.4 由此得出的 A8 论证

A4 比 A8 多出的端到端收益有上限：A100 上 A8 端到端约 1.17×（GEMM 实测 1.56×），A4 按 INT4 GEMM 约 3× 推算约 1.35×，差约 15%；而 A4 要求权重是标量 4-bit（VQ 不可用，H100 没有 INT4），并且是最大的精度损失来源（只开 A4 就 216 → 204/280）。所以 activation 用 8-bit，比特预算花在权重上。A8 的精度代价由阶段 1a 测定。

---

## 12. 阶段 1 实施计划（2026-10-27 定稿）

C3 倾向走系统方向（融合 kernel + runtime），1a、1b 结果出来后再定。

### 12.1 队列策略

- walltime ≤ 2h、1–4 卡的作业提交到 `normal` 会自动路由到 **gdev**，最多 10 个并行；短任务优先放 gdev。
- \> 2h 的作业走 g1–g4，排队是串行的；1b 准备/校准需要时可以用。
- 监控：haiku 子代理，按登记表轮询，约 20 分钟查一次，没有变化不输出，作业结束时拉回结果并唤醒主任务。

### 12.2 1a：A8 精度（不训练）

**固定输入**（`fastwam-a8-accuracy-20261027` v2，远端 root `...-20261027-v2`，job `26029093.pbs101`，gdev 单卡）：

| 臂 | 处理 |
|---|---|
| Q0_A4 | 复用 Q-Clip `25937633` 的 80 条 endpoint |
| Q0_A16 | 新跑：weight-only 下界 |
| Q0_A8 | 新跑：核心问题 |
| Q0_H_INT8_A8 | 新跑：1b 格式（只用 H、每行 INT8、A8）的免训练版本 |

另有权重级诊断（INT8 行取整误差 vs VQ 误差，不跑 query），以及每臂 case 2000 的 eager anchor，供 engine 评测器启动检查。

判定：Q0_A8 回收 A4→A16 差距 ≥80% → A8 成立；Q0_H_INT8_A8 的动作 RMSE 比 Q0_A8 差不超过约 10% → 只用 H 可行，否则 1b 再考虑可学变换。

**engine 版 LIBERO-Plus 评测器**（`fastwam-plus-engine-eval-20261027`）：只写不跑，供后续使用。A4/A8 可配置、U 可选、bundle 可配置；首次使用时自动做启动 anchor 逐位检查和 8 个 slot 的历史结果复现检查。

**1a 结果（`26029171.pbs101`，2026-10-27）：** 两条判定都通过。

| 臂 | first10 motor RMSE | future NMSE | 相对 Q0_A4 |
|---|---:|---:|---:|
| Q0_A4 | 0.04561 | 0.06174 | — |
| Q0_A16 | 0.02179 | 0.02039 | motor −52%，future −67% |
| Q0_A8 | 0.02160 | 0.02120 | motor −53%，future −66% |
| Q0_H_INT8_A8 | 0.02173 | 0.02083 | motor −52%，future −66% |

- A4→A8 回收 A4→A16 差距的 **100%**（motor），future 98%；A8 与 A16 无差别，40/40 context 都优于 A4。
- 只用 H + 每行 INT8 相对 Q0_A8：motor +0.6%，future −1.8%，不重训就几乎无损；U 可以去掉。
- INT8 行取整的权重误差（NMSE 7.9e-5）只有 VQ 误差（1.4e-2）的 0.5%。
- 结论：Q0 的误差有一半以上来自 A4；在 A8 下，4.27 bpw VQ 权重本身的误差约为 A4 下的一半。§11.4 的 A8 论证成立。下一步重点转为低 bpw 档与 future 保真在 A8 下是否仍然重要。

### 12.3 1b：只跑主臂 A+Z+K

- 格式：固定 Smooth（α=0.5）+ signs + H128，不用 U；per-token A8；d4、2×256 整数 codebook + 每行 scale，不用 B32 指数（≈4.02 bpw）。
- 准备：H 坐标、BF16 权重、A8 下在 C80 收集二阶矩；拟合 614 层整数 VQ0′。
- 校准：A+Z+K，C80，8 次更新 × 10 contexts，Adam lr 5e-4，clip 1，首步后清 moments；λ = 0.04245 不变，β 按冻结规则重算；STE 覆盖 A8 与整数 codebook；复用全部 teacher 缓存和冻结 BF16 reader。
- 评测：held-out 固定输入 80 条；导出 engine bundle 与 eager anchor，闭环用 engine 评测器（后续）。
- 作业：< 2h 的阶段放 gdev；校准若 > 2h 整段放 g1–g4。

**结果**（prep `26029721`，gdev 40 min；train `26030496`，g1 3 h 26 min）。held-out 40 contexts × 2 seeds，四 suite 等权：

| 配置 | bpw | first10 motor | future NMSE | full32 motor | first10 gripper |
|---|---:|---:|---:|---:|---:|
| Q0（R0+U+MXVQ, A4） | 4.27 | 0.0456 | 0.0617 | 0.0511 | 0.0573 |
| Q0_A8（同权重，A8） | 4.27 | 0.0216 | 0.0212 | — | — |
| VQ0′（H+整数 VQ, A8，未校准） | 4.02 | 0.0209 | 0.0291 | 0.0242 | 0.0396 |
| **1b（VQ0′ + A+Z+K 8 步）** | 4.02 | **0.0183** | 0.0304 | **0.0219** | **0.0295** |

- 拟合：614 层，加权 weight NMSE 0.0147（Q0 的 MXVQ 为 0.0142），码字对和最大用到 127。
- 校准：β = 0.00623；整数 codebook 改动 365638 / 1257472 条（29%）；staged replay 与 formal parity 误差均为 0。
- 动作：1b 比 VQ0′ 好 12.6%（motor）、25%（gripper），比 Q0_A8 好 15%，bpw 低 0.25。
- Future：C80 上每步更新前的 future NMSE 从 0.055 降到 0.016–0.020，但 held-out 基本不动（0.0291→0.0304），比 Q0_A8 差 43%。A+Z+K 在 C80 上压住了 future，泛化到 held-out 的部分只体现在动作上。
- 与 Q0 的主要差别是没有 U：Q0 的 U 在相同 8 步校准下把 held-out future 拉到 0.021。可学变换（U 或可学 H）是 future 差距的首要候选，消融里优先。
- 导出：`results/export/inference_bundle.pt`（16.0 GB 容器，format = export_runtime.FORMAT，use_u = false，A8）与 eager anchor（case 2000, seed 0），可直接接 engine 评测器。
- 闭环（LIBERO-Plus 280，engine，2×A100，作业 26033945）：成功 **210/280**。BF16 为 216，Q0（A4）208，A+Z 204，最强 W4A4 基线 203。首用 Q0_A4 复现检查 8/8 通过，说明 engine 与历史 eager 评测逐条一致。配对：对 BF16 11 胜 17 负，对 Q0_A4 18 胜 16 负，对 A+Z 18 胜 12 负。分扰动看，camera_viewpoints 18/40（BF16 21，Q0 19，A+Z 14），light_conditions 38/40 高于 BF16。配对差异都不显著（sign test：对 BF16 p≈0.35，对 Q0 p≈0.86，对 A+Z p≈0.36）。
- 闭环不能用来判断 future 是否重要。闭环里实际对照的 Q0 是 A4，held-out future 0.0617；1b 是 0.0304，action 和 future 都更好，成功率 210 vs 208 和这一点一致。Q0_A8（future 0.0212）从没跑过闭环，所以 43% 的差距闭环根本测不到。280 条的分辨率也不够：±6 条在噪声以内。future 的作用以 C1 的受控证据为准：冻结 BF16 action 通路去读不同 student 的 future，A-only 0.0412 vs A+Z 0.0351；只把 future 换成 BF16，first10 −34%。
- 部署形态：packed index 常驻 GPU，target 编码 4.02 bpw × 6.016B ≈ **2.82 GiB**（Q0 2.99 GiB，BF16 11.2 GiB，3.98×）；614 层 operand 与 eager 逐位一致，anchor action/future 误差为 0。导出容器 16.0 GB，其中 11.9 GiB 是不量化的 T5/VAE。
- 显存：每卡两个 worker，nvidia-smi 稳定在 34.3 GiB，约 **16.8 GiB/进程**（含 CUDA context 与渲染），与 Q0 bench 的 16.9 GB peak 一致；BF16 cache+Graph 是 25.5 GB，两个进程放不进 A100-40GB。
- 延迟：policy call p50 0.836 s，p10 0.788 s。p90 1.57 s 约为两倍，是同卡两个 worker 撞车造成的，所以这些数不是干净的单进程延迟。同作业里的 Q0_A4 复现臂（除首调用外）p50 0.95 s，去掉 U 后大约快 10%。和 BF16 cache+Graph 的 541 ms 比仍慢约 1.5×，因为 engine 还是逐层解码成 BF16 再调 cuBLAS（native_integer_gemm = false），A8 只做了数值模拟。policy 占 episode 时间的 28.7%，其余是仿真。
- 结论：1b 证明了“H-only + 整数码本 + A8”这一 INT8-kernel-ready 格式在部署形态下不掉闭环精度，显存收益已实现；速度收益要等 VQ→INT8 融合 kernel，单进程延迟需要另跑 bench。

**可学 H 决策实验（C8，作业 `26035622`，g2 两卡并行，1 h 17 min）**

- 第一版（只在激活侧学旋转，index 固定）改变了网络函数，C80 曲线从第 2 步起发散，已停掉。
- 第二版两阶段，保函数：激活和权重两侧同乘 G（x·S·H·G、W·S·H·G）。每层先用 ITQ 式交替（1024 行样本，8 轮 × 50 步，lr 0.02）学 butterfly G，再在 W·S·H·G 上重拟合整数 VQ；只有加权误差下降的层才保留 G。之后冻结 G，与 H 臂用同一 C8 配方校准：8 步，每步 1 个 context，β 和 teacher 都沿用 C80 的。

| 配置 | first10 motor | future NMSE | full32 motor | first10 gripper |
|---|---:|---:|---:|---:|
| VQ0′（H，未校准） | 0.0209 | **0.0291** | 0.0242 | 0.0396 |
| VQ0_G（G 重拟合，未校准） | **0.0200** | 0.0303 | **0.0240** | 0.0389 |
| C8_H | 0.0245 | 0.0299 | 0.0286 | **0.0227** |
| C8_G | 0.0248 | 0.0368 | 0.0275 | 0.0444 |
| 1b（H，C80） | 0.0183 | 0.0304 | 0.0219 | 0.0295 |

- G 几乎没动：加权 VQ 误差 0.013366→0.013314（−0.4%），逐层下降的中位数 0.08%，超过 1% 的只有 48/614 层，平均旋转距离 0.010。保留 G 的有 342 层，但收益都在噪声量级。H128 坐标对这个整数 d4 码本已经接近 ITQ 的不动点。
- 未校准时 G 对 motor 略好（−4%），future 略差（+4%）。校准后 C8_G 的 future 比 C8_H 差 23%，gripper 差将近一倍。预设判据是“C8_G 的 future 比 C8_H 低 ≥15% 且动作不变差”，没达到。
- **判定：可学 H 这条路不再做。** future 差距不来自坐标系选择：VQ0′ 未校准时已是 0.0291，C8 和 C80 校准都没有把它拉下来。差距来自权重 VQ 本身，加上 A+Z+K 校准对 future 的泛化弱（C80 上压到 0.016–0.020，held-out 不动）。前面“U 是首要候选”的假设也随之降级：U 的作用若存在，是通过校准目标起效，而不是降低 VQ 拟合误差。
- 附带观察：C8 每步只用 1 个 context，motor 比未校准还差 17%，gripper 却好 43%。8 个 context 不够做动作校准；正式配方保持 C80。

### 12.4 留到消融

A-only、A+Z、可学变换（butterfly G 已在 C8 上否定，见 12.3；可学 Smooth 未测）、microscale 移位、3/2 bpw、BF16W_A8、calibration 来源（原始 vs 扰动 episode）、weight-only 部署、RTX 4090/5090 测速、Cosmos-Policy 的 GEMM 占比。

---

## 13. 论文 Experiments 节方案（2026-10-28）

### 13.1 定位与"相同预算"口径

方法归为 **W4A8 类**：权重 4 bpw 存储（VQ），片上解码成 INT8，激活 per-token A8，计算走 INT8 tensor core。"相同预算"同时对齐三项：

| 维度 | 口径 |
|---|---|
| 权重存储 | 总 bpw，scale、codebook、exponent、Smooth/signs 全部计入。标量 INT4 g128 + BF16 scale = 4.125；我们 4.02 |
| 激活 | A8 per-token dynamic absmax |
| 计算路径 | INT8×INT8→INT32；标量 W4A8 也是 INT4→INT8 再算（QServe/QoQ 同类 kernel） |
| 校准数据 | 同一份 C80（80 个原始 LIBERO 快照） |
| 量化范围 | 614 个 Linear 全部量化，无高精度例外；T5、VAE、norm、bias、KV 不量化 |

最关键的对手是**标量 W4A8**：存储、带宽、kernel 类别都相同，差别只在"标量格点 + 重建校准"对"VQ + 端到端校准"。

### 13.2 主表（Fast-WAM-IDM，LIBERO-Plus 280 + 原始 LIBERO）

| 组 | 行 | 作用 |
|---|---|---|
| 上界 | BF16；W8A8（SmoothQuant） | 天花板 |
| W4A16 weight-only | GPTQ、AWQ；AQLM、QuIP#、GPTVQ（~4 bpw） | 显存同，计算仍是 BF16 |
| **W4A8（主对比）** | RTN、GPTQ、QuaRot+GPTQ、SpinQuant+GPTQ、ViDiT-Q W4A8、SmoothQuant+GPTQ | 同预算 |
| W4A4 | QuaRot、SpinQuant、ViDiT-Q、SVDQuant（已有 195–203） | A4 掉精度且 engine 不更快 |
| Ours | 4.02 bpw / A8 | — |

列：W bpw、A bits、计算路径、Plus 成功率（附 7 维 macro 平均）、原始 LIBERO 成功率、held-out first10 motor RMSE / future NMSE、peak 显存、policy call 延迟。成功率附与 BF16 的配对胜负和 sign test。280 条只能分辨 ±7 条以上的差距，主表闭环的作用是"不输 BF16"，不靠它证明"更好"。

### 13.3 证明"同预算更好"的三类证据

1. **bpw 扫描（主图）**：4 / 3 / 2 bpw × A8，对比标量 W4A8 / W3A8 / W2A8（GPTQ、QuaRot+GPTQ、SpinQuant+GPTQ）。纵轴 Plus 成功率与 held-out action 误差。主张写成 Pareto 前沿更优。
2. **离线指标**：同预算下 held-out 40 context × 2 seed 的 first10 motor RMSE、future NMSE、gripper；分辨率远高于闭环，可做配对检验。闭环只确认离线差距是否转化为成功率。
3. **格式 × 目标 2×2**：{标量 INT4 g128, VQ} × {重建（GPTQ/k-means）, A+Z+K}。回答"是 VQ 的功劳还是 loss 的功劳"。标量格式上的 A+Z+K 训练对象是 group scale。

### 13.4 Experiments 节结构

1. Setup：三个模型；LIBERO-Plus 280（主）+ 原始 LIBERO；C80 校准（干净 demo 校准、扰动下评测）；指标与配对检验。
2. 主表（13.2）。
3. bpw–性能曲线（13.3-1）。
4. 跨模型：Cosmos-Policy、LingBot-VA 各跑 BF16、最强 W4A8 基线、Ours（4 bpw）。
5. 系统效率：BF16+cache+Graph（541 ms）、标量 W4A8 kernel、VQ→INT8 融合 kernel；逐形状 GEMM、policy call 延迟、peak 显存。与标量 W4A8 同速即可，卖点是同速下低 bpw 更准。
6. 消融：A / A+Z / A+Z+K；H / 可学 H / U；A8 vs A4；整数 vs FP codebook；校准来源。
7. C1 分析：冻结 action 通路读不同 student 的 future、只换 BF16 future、future 误差幅度匹配。future 的作用由这一节的受控实验论证，不依赖闭环成功率。

### 13.5 缺口与顺序

| 缺口 | 处理 |
|---|---|
| 标量 W4A8 基线全空 | 先跑 GPTQ / QuaRot+GPTQ / SpinQuant+GPTQ W4A8 g128 的 held-out（13.6），最强的一两个再上闭环 |
| 3 / 2 bpw 两档（我们与对手） | 主图需要；标量基线的 W3/W2 可复用 13.6 的 Hessian 管线 |
| 2×2 中"标量 + A+Z+K" | 13.6 之后 |
| SmoothQuant+GPTQ、ViDiT-Q W4A8 | 补主表 |

### 13.6 标量 W4A8 基线（离线 held-out）

实验目录：[fastwam-w4a8-scalar-baselines-20261028](../../experiment/idea-validation/fastwam-w4a8-scalar-baselines-20261028/README.zh.md)。

- 坐标：plain（不旋转）；QuaRot = 与 W4A4 基线同一个 randomized signed block-Hadamard R128（seed 20261008，每 stream 一个，局部右乘 `xR`、`WR`）；SpinQuant = 同一初始化下已学好的 Cayley R128（job 25726472，step 200，原样复用）。这样 W4A8 行与已有 W4A4 闭环行的旋转完全一致，差别只有 A8 与 GPTQ。
- 权重：对称 INT4（±7）、group 128、BF16 scale，4.125 bpw；GPTQ 用 C80 上的 Hessian `XᵀX`（原坐标收集一次，旋转臂用 `RᵀHR`），damp 1%，block 128，不开 act-order，per-group MSE clip 搜索（GPTQ/QuaRot 的 w_clip 做法）。RTN 行用 absmax，与 W4A4 基线的量化器相同。
- 激活：per-token A8（`absmax/127`，与 1b 相同）。
- 臂：GPTQ / QuaRot+GPTQ / SpinQuant+GPTQ 三个主臂，加对应 RTN 三臂与 W16A8（plain）参照。
- 评测：与 1a/1b 相同的 held-out 40 × 2，直接对比 1b（first10 0.0183、future 0.0304）与 Q0_A8。

**结果**（作业 26036415，g1，1 h 32 min；[RESULTS](../../experiment/idea-validation/fastwam-w4a8-scalar-baselines-20261028/pulled/26036415.pbs101/RESULTS.zh.md)）：

| 臂 | bpw | first10 motor | future NMSE | first10 gripper |
|---|---:|---:|---:|---:|
| W16A8（plain） | 16 | 0.0053 | 0.0011 | 0.0119 |
| SpinQuant+GPTQ W4A8 | 4.125 | **0.0095** | **0.0088** | 0.0115 |
| QuaRot+GPTQ W4A8 | 4.125 | 0.0098 | 0.0089 | 0.0034 |
| GPTQ W4A8（plain） | 4.125 | 0.0118 | 0.0141 | 0.0244 |
| QuaRot RTN W4A8 | 4.125 | 0.0144 | 0.0192 | 0.0218 |
| SpinQuant RTN W4A8 | 4.125 | 0.0162 | 0.0195 | 0.0286 |
| 1b（VQ，A+Z+K） | 4.02 | 0.0183 | 0.0304 | 0.0295 |
| RTN W4A8（plain） | 4.125 | 0.0190 | 0.0277 | 0.0454 |
| Q0_A8（MXVQ + U，A+Z+K） | 4.27 | 0.0216 | 0.0212 | — |

- **4 bpw 下标量 W4A8 + GPTQ 全面胜过我们的 VQ**：QuaRot+GPTQ 的 action 误差约为 1b 的一半，future NMSE 约为 1b 的 1/3.4；连不旋转的 plain GPTQ 也胜过 1b。1b 只和 plain RTN 相当。四个 suite 方向一致。
- A8 本身几乎无损（W16A8 future 0.0011），误差全部来自 4-bit 权重。
- GPTQ 把输出能量误差代理压到 RTN 的 1/5（0.00085 vs 0.0046）。差距来源是 Hessian 误差反馈：我们的 VQ0′ 只用对角二阶矩做加权 k-means，没有误差补偿；A+Z+K 的 8 步端到端校准（0.0209→0.0183）补不回来。
- SpinQuant（在 W4A4 下学的旋转）在 A8 下与 QuaRot 持平，学旋转对 A8 没有额外收益。
- **含义**：C2 里"VQ 存储在同预算下更准"在 4 bpw 不成立。VQ 的优势只可能出现在 3/2 bpw，或在 Hessian 感知的 VQ 拟合（GPTVQ/AQLM 式误差反馈）之后。A+Z+K 能否在 GPTQ 初始化之上继续降误差，是 2×2 消融中最关键的一格。
