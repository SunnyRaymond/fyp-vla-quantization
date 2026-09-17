# Broadcast coupling GPU audit

范围：当前 `broadcast_screen.py`、`broadcast_gpu.sh`、冻结 `PROTOCOL.zh.md`/freeze，以及复用的 `teacher_bias_screen.py` helper；仅静态源码、AST 和 shell syntax 检查，未运行模型或集群。

## 结论

GPU runner 本身未发现会阻断本次最小 screen 的问题，AST 与 `bash -n` 均通过。此前记录的 freeze pin 差异已确认只是 controller 的 CRLF→LF 规范化，不构成实际 blocker（见更正记录）。

## 合同核对

- `TEACHER_SHA` 与当前 `teacher_bias_screen.py` 一致；helper 的 `_manifest_samples`、`_load_helpers`、`_bind_prepared_runtime`、`_runtime_structure`、`_source_identity`、`_make_observations` 调用签名均匹配。复用的 `smoke_runner.py`/`screen_runner.py` SHA 也与 helper 固定值一致。
- manifest SHA、protocol SHA、checkpoint/source identity 在 helper 和 runner 中均做 hard pin；真实 V100 与 compute capability `[7,0]`、eval、全参数 `requires_grad=False` 有硬检查。
- `model.encode(obs, actions[:1])` 的预期输出 `[1,1,196,404]`，broadcast tail 的 exact equality 检查正确。predictor pre-hook 比较实际 `inputs[0]` `[1,196,404]` 与 `value[0]`，保存的 tail 为 `[196,20]`；总计 FP/FP-copy、RTN-before/after、Shared3、Spatial3 共 60 次 predictor call。
- A4 quantization 在 allocation 内用 CPU `torch.float32` RNG seeds 2501/2502/2503，scale 为每 state 20D vector 的 max-abs/7；offset 使用 seed2601 的 `randperm(196)` 后 `%3`。Shared 与 Spatial 的 draw/multiset 轴顺序和协议一致，RTN input exact equality 也已检查。
- 所有预测在 `torch.no_grad()` 下进行；每 state 保存 initial latent、codes/quantized vectors、hook seen tail、FP/RTN/两 arm visual outputs。每次仅单步 predictor，state digest 在前后比较，未引入训练或环境 rollout。
- shell 先 source allocation guard，再复制小 helper，`UGGPU-TC1` 单 V100、4 CPU、16G、5 分钟；runner 240 秒 deadline，外层 270 秒 timeout，预算边界一致。

## Pin 观察的更正

此前在 Windows working tree 观察到 `broadcast_input_freeze.json` 的本地 SHA 为 `3270b080...`，而 verifier pin 是 `233bd772...`，因此曾将其报告为 CPU verification blocker。Root 确认 controller upload 会把 CRLF 规范化为 LF；远端提交的 freeze 字节 SHA 为 `233bd772...`，且 GPU64838/CPU64839 的 `pin_freeze` 与全部 source/receipt checks 均已通过。因此这是本地换行格式差异，不是实际阻断；不应修改 pins，也无需重跑 GPU。

另有一个非阻断 provenance 注意：GPU runner 直接 hard-code manifest/seed/offset，未逐项比较 freeze 内的同名字段；当前值与协议一致，CPU replay 仍会重建 RNG，因此不改变本次 runner 的科学语义。
