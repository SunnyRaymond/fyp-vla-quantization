# LeWM PushT student transfer：CPU preflight

CPU-only PBS preflight：`24546788.pbs101`，Exit status `0`。模型加载、HDF5
metadata 读取和少量 schema 探测都在 compute allocation 内完成；没有训练、forward
或修改 checkpoint。

权威小型结果：[preflight_summary.json](artifacts/24546788.pbs101/preflight_summary.json)

## Checkpoint interface

| object | type | parameter count | 关键形状 |
|---|---|---:|---|
| root model | `jepa.JEPA` | 18,034,478 | latent width `192` |
| `encoder` | `transformers.models.vit.modeling_vit.ViTModel` | 5,501,376 | CLS `[1,1,192]`；position `[1,257,192]`；patch projection `[192,3,14,14]` |
| `predictor` | `module.ARPredictor` | 10,791,360 | `pos_embedding [1,3,192]`；6 个 Transformer layer；attention qkv `[3072,192]`；FFN `192→2048→192` |
| `action_encoder` | `module.Embedder` | 156,206 | action input `10`；`10→768→192` |
| `projector` | `module.MLP` | 792,768 | `192→2048→192` |
| `pred_proj` | `module.MLP` | 792,768 | `192→2048→192` |

因此 LeWM transfer 的 student interface 可以先固定为：

```text
z_t: 192-D compact latent
a_t: 10-D action block
context: predictor positional history length 3
teacher target: pred_proj(predictor(...))，输出 192-D latent
```

本次 preflight 没有把 `latent_dim/action_dim` 暴露为 root attribute；上面的
维度由权威 parameter shapes 和 predictor/action encoder interface 确定。

## PushT HDF5 schema

文件：`stablewm_home/pusht_expert_train.h5`；总 transition 数 `2,336,736`，
episode 数 `18,685`。

| dataset | shape | dtype | 用途/可用长度 |
|---|---|---|---|
| `pixels` | `[2336736,224,224,3]` | `uint8` | observation image |
| `proprio` | `[2336736,4]` | `float32` | proprio input |
| `state` | `[2336736,7]` | `float32` | raw state |
| `action` | `[2336736,2]` | `float32` | per-step action |
| `episode_idx` | `[2336736]` | `int64` | transition→episode index |
| `step_idx` | `[2336736]` | `int64` | episode-local step index |
| `ep_len` | `[18685]` | `int32` | episode lengths；min `49`，max `246`，前 8 个均 `109` |
| `ep_offset` | `[18685]` | `int64` | episode start offsets；前 8 个 `0,109,218,327,436,545,654,763` |

官方 source 中没有找到可直接 import 的 `PushTDataset` 类；因此 HDF5 schema
是当前可靠的数据入口。下一步可按 `ep_offset[e] : ep_offset[e]+ep_len[e]`
切片取得完整 episode，再按 LeWM 的 observation/action preprocessing 生成
teacher targets。

## Reproduce

```bash
qsub lewm_student_preflight_cpu.pbs
```

默认 external assets：

```text
/scratch/users/ntu/yguo017/lewm-pusht-iteration/le-wm
/scratch/users/ntu/yguo017/lewm-pusht-iteration/stablewm_home
/scratch/users/ntu/yguo017/lewm-pusht-iteration/venv/bin/python
```
