# SmolVLA timestep half constant-folding：prior gate

## 结论

这是一个真实存在的独立模块切面，但“缓存 timestep-only feature 以保护 diffusion/flow 的时间信息”作为通用 quantization 方法已有直接 prior，因此 **generic method novelty no-go**。在不宣称新 quantizer、mixed-bit deployment 或 latency gain 的前提下，仍有一个窄的 **SmolVLA-specific branch-attribution diagnostic**：固定 K=10、在相同 full-row RTN grid 下，只把 `action_time_mlp_in` 的 time half 从 W4 替换为 FP32 timestep table，检验 quantization error 是否主要经该分支进入 action expert。这个问题未被上述论文直接在 SmolVLA action expert 上回答，可 conditional-go 一次小 screen；若不执行 matched-grid control，则为 identifiability no-go。

## v0.4.4 接口和尺寸

官方 pinned [LeRobot v0.4.4 `modeling_smolvla.py`](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py#L653-L677) 的 `embed_suffix` 顺序是：

```text
action_emb = action_in_proj(noisy_actions)
time_emb   = sinusoidal(timestep); expand_as(action_emb)
cat([action_emb, time_emb], dim=last)
action_time_mlp_in -> SiLU -> action_time_mlp_out
```

`time_emb` 对一个 denoising step 的整个 50-token chunk 相同，`sample_actions` 以 `time=1+step*(-1/num_steps)` 循环；[config.py](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/configuration_smolvla.py#L28-L41) 固定 `chunk_size=50`、`max_action_dim=32`，并在 [L61-L63](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/configuration_smolvla.py#L61-L63) 将 `num_steps` 默认设为 10。官方 expert 构造把 backbone `hidden_size` 乘 `expert_width_multiplier` 得到 expert width（[source](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/smolvlm_with_expert.py#L89-L125)）；本 checkpoint 的 960×0.75 因而是 `h=720`，所以实际模块为：

```text
action_in_proj:          32 -> 720
action_time_mlp_in:   1440 -> 720       # W shape [720,1440]
action_time_mlp_out:    720 -> 720
```

写 `W=[W_a,W_t]` 后，第一层 preactivation 对每个 token 严格满足 `W_a a + W_t e(t) + b`。因此 K=10 时可以预计算 `T[t]=W_t e(t)` 的 `[10,720]` FP32 table；它只能替代 affine input 中的 time matmul，不能跨过后面的 SiLU，也不能声称整个 action expert 是 time-separable。该 module 是当前 expert-112 allowlist 之外的明确新 locus，不能把旧 expert-W4 结果冒充此切面的证据。

## 最近 primary prior 与重叠

1. [TFMQ-DM / CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_TFMQ-DM_Temporal_Feature_Maintenance_Quantization_for_Diffusion_Models_CVPR_2024_paper.html) 明确把只依赖 timestep、与 sampling data 无关的部分定义为 Temporal Information Block，并用 temporal-feature maintenance、finite-set calibration 和 cache-based maintenance 保存/缓存其 quantized counterpart。它直接覆盖“time-only feature 可以单独保护或缓存以减小低比特 denoising drift”的通用主张。
2. [Temporal Feature Matters](https://arxiv.org/abs/2407.19547) 延伸同一问题，明确提出 pre-compute/cache temporal features 作为 diffusion PTQ maintenance strategy。它没有验证 SmolVLA 的 `[action_emb,time_emb]` concat、也没有给出 action-dependent half 的 matched-grid attribution；因此它阻断 generic novelty，却没有完全回答本模型的窄诊断。
3. [TeaCache](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Timestep_Embedding_Tells_Its_Time_to_Cache_for_Video_Diffusion_CVPR_2025_paper.html) 进一步把 timestep embedding 用于 training-free temporal cache decision。它是 caching/acceleration prior 而非本 W4 branch experiment，不能当作 SmolVLA quantization 结果，但会阻断“首次利用 timestep-only cache”之类的宽 claim。

所以可保留的 claim 只能是：**在 pinned SmolVLA v0.4.4 action-time fusion 中，time-half 的 quantization perturbation 是否对固定 K=10 flow action 输出产生可分离的贡献。** 不能写成新的 constant-folding 算法、一般 diffusion PTQ recipe 或已证实的 storage/latency improvement。

## 公平的最小诊断边界

full-row RTN control 是必要的。先对完整 `[720,1440]` row 计算一次 baseline `s_j=max(|W_j|)/7` 和 signed W4 code；protected arm 必须复用这同一组 `s_j/code(W_a)`，只把 `W_t e(t)` 换成同一 FP32 checkpoint 的 precomputed table，bias 在两臂都保持 FP32。不能对 action half 重新求 absmax/scale，否则增益同时来自新的 scale allocation，不能归因于保护 time half。FP、full-row W4、matched-grid time-protected W4 三者共享完全相同的 image/state/language/noise/timestep 输入、K=10 和 forward budget。

table 只保存 `T[t]=W_t e(t)`，不把 bias 混入，以免改变 bias rounding。table 生成和 lookup 必须保持 runtime FP32/device dtype；直接 `W_t @ e(t)` 与 table-add 的 pre-SiLU 输出应有预注册的 implementation no-op 检查。若 table 在 FP64 或 CPU 另算，或重新量化 `W_t`，那是不同 arithmetic/quantizer oracle。

存储账本也必须诚实：`W_t` 是 `720×720`，FP16 保留权重约 `1,036,800` bytes；固定十步的 FP32 table 只需 `10×720×4=28,800` bytes，另加 action-half W4 code `720×720×0.5=259,200` bytes和 scales/metadata。这个节省依赖 K=10 且允许丢弃 `W_t`；若仍保存 FP32/FP16 `W_t`，就不能声称 table 带来 weight-storage benefit。屏幕本身不测 kernel latency，不能从字节账本推出 speedup。

## Conditional gate / stop 条件

若 root 选择一次 bounded diagnostic，最低可证伪门是：table substitution 的 pre-SiLU no-op 通过；protected arm 使用 baseline full-row `s_j/code(W_a)`；并在固定输入上同时报告 `W_a`、`W_t` 的 grid error、pre-SiLU error 和最终 normalized action/velocity error。若 protected 与 full-row W4 的差异不能在预注册的数值/效应边界内稳定归因到 `W_t`，或必须改变 scale、K、steps、calibration 或输入以获得结果，则记 `identifiability_no_go`，不调参挽救。

即使该窄 attribution 通过，也只能是 checkpoint-specific mechanism signal；TFMQ-DM/Temporal Feature Matters 已使 generic temporal-cache novelty no-go，且本切面不支持 task success、完整 validation、native low-bit performance 或新算法认证。若研究目标必须是可部署新 recipe，则直接停止，不值得为 novelty 追加 GPU；若目标只是核对该独立 module 的 branch sensitivity，保留一次 conditional screen 即足够。
