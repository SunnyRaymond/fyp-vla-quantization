# C80 R128 remaining model-state inventory

- Phase: `pre`; bundle source: `25921795.pbs101`.
- State tensor keys: 3,547; logical tensor bytes across key references: 12,797,540,578.
- Unique resident backing storage: 12,779,984,326 bytes (11.902288 GiB) across 1,475 storages.
- Repeated key references beyond one per storage: 2,072; aliases/views are not extra resident storage.
- Expected unique bytes `12779984326`: match = `True`.

## Largest remaining components

| Component | Unique storage bytes | Tensor keys | Modules | Quantized target modules | Purpose evidence |
|---|---:|---:|---:|---:|---|
| text_encoder | 11,361,820,672 | 242 | 242 | 0 | Text or token conditioning path; inferred from state key names. |
| latent_autoencoder | 1,409,377,336 | 196 | 124 | 0 | Image or video latent encoding/decoding path; inferred from state key names. |
| video_world_model | 5,990,784 | 1,557 | 1,464 | 306 | Video, future-latent, or dynamics path; inferred from state key names. |
| action_policy | 2,787,342 | 1,551 | 1,461 | 307 | Action or control prediction path; inferred from state key names. |
| unclassified | 8,192 | 1 | 1 | 1 | No architecture-specific marker in the state key; purpose is unclassified. |

## Quantization and execution boundary

- `614` target modules are represented in the VQ payload; `1840` dense state keys are excluded from `model_state`.
- This inventory describes the remaining `model_state` storage. Component purposes are conservative name-path classifications; `unclassified` entries stay unassigned rather than being called text-encoder weights.
- Static inventory does not establish module execution. The optional runtime probe records calls only when attached to an already-authorized forward; unobserved modules are not labeled unused.
- Full module and alias groups: `storage_inventory_modules.json`.

## Largest backing storages

| Storage bytes | Owner state key | Alias refs | View present |
|---:|---|---:|---|
| 2,100,297,728 | `text_encoder.token_embedding.weight` | 1 | False |
| 83,886,080 | `text_encoder.blocks.0.ffn.fc1.weight` | 1 | False |
| 83,886,080 | `text_encoder.blocks.0.ffn.fc2.weight` | 1 | False |
| 83,886,080 | `text_encoder.blocks.0.ffn.gate.0.weight` | 1 | False |
| 83,886,080 | `text_encoder.blocks.1.ffn.fc1.weight` | 1 | False |
| 83,886,080 | `text_encoder.blocks.1.ffn.fc2.weight` | 1 | False |
| 83,886,080 | `text_encoder.blocks.1.ffn.gate.0.weight` | 1 | False |
| 83,886,080 | `text_encoder.blocks.10.ffn.fc1.weight` | 1 | False |
| 83,886,080 | `text_encoder.blocks.10.ffn.fc2.weight` | 1 | False |
| 83,886,080 | `text_encoder.blocks.10.ffn.gate.0.weight` | 1 | False |
| 83,886,080 | `text_encoder.blocks.11.ffn.fc1.weight` | 1 | False |
| 83,886,080 | `text_encoder.blocks.11.ffn.fc2.weight` | 1 | False |

## Largest component candidates for memory work

These are resident storage opportunities by size. Any lower precision or offload change still needs parity and latency checks.

- `text_encoder`: 11,361,820,672 unique bytes; Text or token conditioning path; inferred from state key names.
- `latent_autoencoder`: 1,409,377,336 unique bytes; Image or video latent encoding/decoding path; inferred from state key names.
- `video_world_model`: 5,990,784 unique bytes; Video, future-latent, or dynamics path; inferred from state key names.
- `action_policy`: 2,787,342 unique bytes; Action or control prediction path; inferred from state key names.
- `unclassified`: 8,192 unique bytes; No architecture-specific marker in the state key; purpose is unclassified.
