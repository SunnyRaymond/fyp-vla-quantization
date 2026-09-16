# Euler Jacobian：AD 与 closure source audit

日期：2026-09-13。只读核对现有 `flow_screen.py`、
`rounding-persistence/persistence_batch2_screen.py` 与 pinned LeRobot v0.4.4；
未运行模型、数据或 cluster。

## 结论

源码层面没有 structural blocker。32 次 `autograd.grad` 可以得到 first-token
输出对完整 `x_t[0,0,:]` 的 32×32 Jacobian，并可同时保留每个输出对完整
`x_t[0,:,:]` 的 `[32,50,32]` gradient 以检查 future-token closure。该实现必须
走 direct `model.denoise_step`；现有 inference recorder 不能直接复用。

## 关键证据

1. 官方 `predict_action_chunk` 带 `@torch.no_grad()`，但 `denoise_step` 没有
   no-grad decorator。官方 [modeling_smolvla.py v0.4.4](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py)
   的 `denoise_step` 只执行 `embed_suffix(x_t,timestep)`、官方 cache forward
   和 `action_out_proj`（约 L789–819），因此在 `torch.enable_grad()` 中对 x leaf
   求导是可行的。
2. `embed_suffix` 为 50 个 action positions 设置 `att_masks=[1]*50`，
   `make_att_2d_masks` 形成 causal mask；first query 只接收 prefix 与自身。
   因此每个 `v[0,0,j]` 的 gradient 的 `x[:,1:,:]` 部分应为零（按预注册小
   tolerance 检查），first-token 的坐标 block 才是闭合的 32D map。7D
   physical block 不能替代这个 closure。
3. 官方 `sample_actions` 先 prefill prefix KV，再逐步把同一 cache 传给
   `denoise_step`；官方 Euler 为 `dt=-1/10`、`x_t=x_t+dt*v_t`。在
   `smolvlm_with_expert.py` 的 attention 实现中，`fill_kv_cache=False` 只把
   suffix K/V 与 past cache 拼到局部变量，未写回 past cache；现有 persistence
   wrapper 也在 `persistence_batch2_screen.py:521–565` 检查 cache object/digest
   不变。因此固定 prefix cache 可安全用于一次局部 Jacobian，但仍应保留
   before/after digest receipt。

## 现有 helper 的不可直接复用处

- `flow_screen.py:836–863` 的 `_DenoiseRecorder` 将 velocity `detach()`，并通过
  `predict_action_chunk` 运行；它适合记录，不保留 AD graph。
- `persistence_batch2_screen.py:591–633` 的 `_manual_fp_action` 明确包在
  `torch.no_grad()`；`_run_schedule` 的 `x_inputs/velocities` 也在
  `:545–546` detach。可复用其 prefix 构造与 signature 检查，不能复用其
  no-grad/recording wrapper 作为 Jacobian路径。
- direct path 应仿照该文件 `:591–624`：`prepare_images/state`、`embed_prefix`、
  `make_att_2d_masks`、prefix `forward(fill_kv_cache=True)`，然后显式调用
  `model.denoise_step(prefix_pad_masks=..., past_key_values=..., x_t=leaf,
  timestep=tensor([.5]))`。先 assert `use_cache=True`、`rtc_config=None`、
  `compile_model=False`、eval、参数 `requires_grad=False`。

## AD 实现合同

`leaf = x_source.detach().clone().requires_grad_(True)`，形状 `[1,50,32]`；
对 `j=0..31` 依次调用
`autograd.grad(v[0,0,j], leaf, retain_graph=(j<31), create_graph=False)`。
堆叠后轴定义为 `[output_coord, input_position, input_coord]`，应为
`[32,50,32]`；`J=grad[:,0,:]`，`M=I32-0.1*J`。不要在外层保留
`inference_mode`，也不要把 `v.detach()` 放在 grad 之前。若 future block
超过冻结 tolerance，立即记 implementation/closure failure。

结论边界仍是固定 FP-path 的单点 local normalized 32D Euler map；
`det(M)<0` 只能叫 local orientation reversal，不能叫 global fold、non-invertible
或 physical/task failure。prefix cache、权重 restore、FP/W4 same-x 与源码
identity 均应成为 engineering gate。
