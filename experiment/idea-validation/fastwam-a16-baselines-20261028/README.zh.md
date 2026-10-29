# A16 weight-only 基线：GPTQ / QuIP# / QTIP / GPTVQ，3-bit 与 2-bit，LIBERO-Plus 280 闭环

结果：[RESULTS.zh.md](RESULTS.zh.md)（8 个臂全部完成，2026-10-29）。

## 口径

- 激活全部 BF16（A16），只量化权重；614 个目标 Linear 全部量化，无高精度例外。
- 预算：总 bpw ≈ bits，scale、zero、码本、符号向量、padding 全部计入 `bits_total`。各方法都不做微调。
- GPTQ 用官方默认：按行非对称 min-max、无分组、blocksize 128、damp 1%，加官方的 act-order 开关（[methods/gptq/quant.py](methods/gptq/quant.py)）。
- QuIP#、QTIP、GPTVQ 从官方仓库移植成逐层函数，提交 commit 与偏差记录在各自的 `methods/<method>/UPSTREAM.md`。
- Hessian：每层 BF16 输入的 `mean x xᵀ`，C80（case 2000–2079，seed index 0）80 次 query 的全部调用，原坐标 FP32。所有方法用同一份 BF16 模型的 Hessian，不做顺序重收。

## 闭环

- 运行时：反量化成稠密 BF16 权重，走 BF16 + condition cache + CUDA Graph（`26028322` 的 bf16_cache_graph，541 ms/call）。
- T5：worker 加载模型时不加载 T5（`model.load_text_encoder=false`），`encode_prompt` 从预计算缓存取。每进程约 15 GB，每卡 2 个 worker。
- 启动检查（每个 worker）：① 用 T5 缓存跑 BF16 的 case 2000，action/future 必须与完整 T5 的结果逐位一致；② 装上该臂权重后，eager 与 cache+Graph 的 case 2000 输出逐位一致。没有 held-out 评测。
- 面板、seed、settling、步数上限、replan 与 `fastwam-plus-engine-eval-20261027` 相同。

## 作业结构

`run.pbs` 按 `jobs.json` 中该作业的 plan 依次执行：

| 步骤 | 内容 |
|---|---|
| `prep` | 各 GPU 分摊 Hessian 收集并保存 BF16 master；shard 0 另做 T5 缓存和完整 T5 的 BF16 anchor |
| `wait:prep` | 等别的作业完成 prep |
| `quant:M:B` | 方法 M、B bit，614 层按 GPU 分片量化，再汇总 bpw 与误差代理 |
| `loops` | 按 `arm_priority`（3-bit 在前）认领已量化、未完成的臂跑 280 闭环；一个臂同一时间只归一个作业，作业死掉后可被接管 |

所有步骤都跳过已在盘上的结果，重新提交即续跑。g1、g2、g3 每个用户各能同时跑 1 个作业，所以量化（g1/g2）与闭环（g3/g2）可以并行。

## 命令（项目根目录）

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-a16-baselines-20261028/control.py upload
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-a16-baselines-20261028/control.py submit --job a16g3
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-a16-baselines-20261028/control.py status --job a16g3
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-a16-baselines-20261028/control.py pull
```

## 输出（远端 `results/`）

- `prep/`：`hessians/`、`masters/`、`t5_cache.pt`、`bf16_anchor.pt`、`modules.json`、`COMPLETE.json`
- `quantized/<arm>/`：`weights/`（BF16）、`stats/`（每层 bits_total、误差代理、配置）、`COMPLETE.json`
- `loops/<arm>/`：`episodes/`、`worker_*/startup.json`、`summary.json`、`RESULTS.zh.md`、`DONE`
