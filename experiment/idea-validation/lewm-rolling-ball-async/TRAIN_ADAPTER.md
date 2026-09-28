# Rolling Ball vanilla LeWM training adapter

This adds a task-data entry point around the pinned vanilla LeWM implementation. It trains only the visual predictor; it does not run ReflexBench, select a task policy, or establish task success.

## Frozen model and objective

`train_baseline.py` imports the pinned upstream `train.py`, `utils.py`, `jepa.py`, and `module.py` after checking that it is running on a host listed in a real PBS allocation. It instantiates the upstream Hydra `cfg.model`, calls the upstream `lejepa_forward`, and uses upstream `SIGReg`. The objective remains prediction MSE plus `0.09 × SIGReg`; AdamW uses learning rate `5e-5`, weight decay `1e-3`, and the original epoch-level `LinearWarmupCosineAnnealingLR`. The model remains a non-pretrained ViT-Tiny, 14-pixel patches, 224-pixel input, 192-D latent, six-layer AR predictor, and 3-frame history.

For each sample, the data adapter supplies four consecutive RGB frames and three actions: action `k` is the transition from frame `k` to frame `k+1`. This is the alignment used by vanilla `train.py`: it predicts `emb[:, 1:]` from the first three image/action embeddings. The task uses `frameskip=1` at 25 Hz, matching the 40 ms control step; the official PushT config uses `frameskip=5`, so this is a documented task adaptation. Episode IDs, rather than overlapping windows, define the frozen 180/20 train/validation split.

The pinned upstream training defaults retained here are batch size 128, 100 epochs, bf16, and gradient clipping 1.0. The task seed is explicitly set to 0, and data workers are set to 2 for this PBS job. Image preprocessing is the upstream `get_img_preprocessor` (ImageNet `ToImage` normalization and 224 resize). Action mean/std are computed from train episodes only and reused unchanged for validation; the standard deviation uses sample correction to match upstream `torch.std` and is floored at `1e-6` by the data adapter.

## PBS invocation and outputs

The current cluster wrapper invokes the script as:

```bash
python train_baseline.py --root "$TASK_ROOT" --output-dir "$OUT" --max-epochs 100
```

The one-epoch smoke invocation may add `--limit-train-batches 2 --limit-val-batches 2`. Any non-100-epoch or batch-limited run is marked `SMOKE_ONLY` and writes metrics and summary without a raw baseline checkpoint. The first smoke job revealed an automatic internal Manager checkpoint; the adapter now disables requeue checkpoints and places the Manager cache under the PBS run directory. A formal run writes both raw-model-state checkpoints: `last.ckpt` is the frozen epoch-100 baseline, and `best_val_pred.ckpt` is retained only as a validation-MSE diagnostic. The diagnostic does not select the baseline because validation MSE alone can favor latent scale/collapse changes. `metrics.jsonl`, `summary.json`, resolved `config.yaml`, and `train_goal_bank_indices.json` are also written to the output directory. The terminal-frame index list contains training episodes only.

Each checkpoint contains the Hydra `model_config`, CPU `model_state_dict`, full resolved training config, action normalization (`mean`/`std` lists of length 8), demonstrated raw action bounds (`min`/`max` lists of length 8), episode split, frozen source/package identities, and train-only terminal-frame indices. To load it later, instantiate the stored model config with the same pinned source imports, call `load_state_dict`, then use the native LeWM `model.get_cost(info_dict, action_candidates)` interface. The CEM adapter must normalize candidate absolute joint targets with the stored action statistics and translate them to the environment's relative joint command. The saved min/max describe observed demonstrations only; they are not verified safe planner bounds.

At startup inside PBS, `validate_windows()` checks train and validation window indices without loading image arrays, and one transformed training sample checks the four-frame/three-action shapes, contiguous episode-local indices, matching global image/action rows, train-split membership, and action normalization against the raw training rows. `--alignment-check-only` runs those checks and exits before model construction. No local model import or training is permitted.

## Task-level limits

The data release has 200 Rolling Ball episodes but no explicit per-episode success label; collection defaults make success plausible, not verified. The trained predictor's validation MSE is not an interception success measure. Absolute joint-target actions in the demonstrations differ from the environment's scaled relative joint commands, so the planner must apply the explicit coordinate conversion and respect the environment's valid action interface.

Training terminal frames are retained as possible inputs to a later goal adapter, not as a validated goal. Ball-lane and reset variation mean a single fixed demonstration terminal image may be inappropriate across episodes. Choosing among a train-only goal bank based on the current legal camera image is a separate, unvalidated design step. Demonstration action coverage is narrow; a CEM search can leave that support even when its candidates are clipped to the empirical min/max. Neither the action bounds nor the best validation checkpoint establish a useful CEM objective.

The script does not evaluate a checkpoint in Isaac Sim. The available ASPIRE2A A100 lacks RT Cores, while the native Isaac RGB path requires a supported rendering GPU, so visual closed-loop results remain unrun.

## Pinned source references

- [Vanilla LeWM `train.py`](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py)
- [Vanilla LeWM JEPA model](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/jepa.py)
- [Vanilla LeWM modules and SIGReg](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py)
- [Vanilla LeWM training config](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/train/lewm.yaml)
- [Vanilla LeWM model config](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/train/model/lewm.yaml)
- [Official image preprocessing and checkpoint helper](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/utils.py)
- [ReflexBench pinned source](https://github.com/LxRoboticsLab/ReflexBench/tree/8bb931485093c6d98f8729774ad01bf824964e16)
- [ReflexBench dataset pinned revision](https://huggingface.co/datasets/cyx337/ReflexBench_dataset/tree/9295b6e9878609a992047f0b8b65421a493299e7)
