# Euler Jacobian GPU implementation audit

审查范围：当前 `euler_screen.py`、`euler_gpu.sh`、冻结 `PROTOCOL.zh.md`，以及已固定的 `persistence_batch2_screen.py`/`flow_screen.py` helper；仅做静态源码和 AST 检查。

## 结论

当前版本没有发现会阻断本次 bounded screen、改变科学解释或破坏 allocation 安全边界的实现问题，具备提交条件。`euler_screen.py` 的 AST 检查通过。

## 合同核对

- helper signature 对齐：`_load_runtime(flow,args,torch,identity)`、`_load_preprocessor(model_path,vlm_path)`、`_load_raw_sample(path,entry,torch)`、`_prepare_batch(raw,preprocessor,torch)`、`_module_bindings`、`_apply_map`/`_restore_fp` 和 digest helper 均按当前固定 helper 的签名调用。
- gradient 路径绕过了 `predict_action_chunk` 的 `no_grad`，仅在冻结参数的前提下对 `x` leaf 使用 `torch.enable_grad()` 和 32 次串行 `autograd.grad`；prefix 建 cache 与 FP K10 path 使用 `no_grad`，不会抑制后续 input Jacobian。`denoise_step` 的官方直接调用参数和 timestep `[1]` 形状匹配。
- prefix cache 在每个 sample 建立一次，FP/W4 arm 间用 digest 检查未改变；W4 仅作用于 112 个 expert `Linear`，每个 arm 前恢复 FP，最终恢复和 non-expert digest 均有硬检查。缓存不包含 action-dependent suffix，符合 fixed-prefix 协议。
- raw 轴和冻结协议一致：`noise[1,50,32]`、`x_fp[6,50,32]`、`v[6,2,50,32]`、`jac[6,2,32,50,32]`、`fd_values[6,2,2,2,32]`（plus/minus）、`tail_probe_v[6,2,32]`、两组 path 及 `completed[6,2]`。共享 noise、`.5` 状态、`.002` central-FD 和 `.1` tail 均固定。
- source/helper/protocol/freeze/input manifest/checkpoint 都做 SHA pin；V100 由 helper 检查并另核对 compute capability `[7,0]`。新增 `use_cache is True` 检查避免 cache 语义漂移。
- wrapper 首先执行 allocation guard，使用 `UGGPU-TC1` 单 V100、4 CPU、24G、10 分钟；runner 内部 480 秒 deadline，外层 540 秒 timeout，留有收尾余量。无本地模型或下载入口。

## 仅需记录的边界（不阻断）

`raw.npz` 是逐 arm 写入但不是临时文件原子替换；中断时只能按 `engineering.status`/`completed`/receipt 将结果判为 partial/inconclusive，不能读取 partial raw 作科学结论。`quant_readback_restore` 的命名 check 本身为常量，但其前置 `_apply_map` 和 `_restore_fp` 已逐层 `torch.equal` 硬失败；CPU verifier 仍应复核 snapshot/readback。以上不构成本次已冻结 screen 的执行阻断。
