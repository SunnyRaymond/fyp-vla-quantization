"""pi0.5 on the senior's three AgileX Piper tasks: registers the `pi05_agilex3` openpi TrainConfig and runs openpi's own
entry points with it.

  agilex_openpi.py norm-stats                 scripts/compute_norm_stats.py for pi05_agilex3
  agilex_openpi.py train [openpi overrides]   scripts/train.py pi05_agilex3 ... (e.g. --exp-name bs32_6k --resume)
  agilex_openpi.py serve [serve_policy args]  scripts/serve_policy.py, e.g.
      serve policy:checkpoint --policy.config=pi05_agilex3 --policy.dir=<checkpoints/pi05_agilex3/bs32_6k/5999>

Data: LeRobot v2.1 dataset `local/agilex3_10hz` under $HF_LEROBOT_HOME (convert_openpi.py), 10 Hz, keys image /
wrist_image (240x320 RGB from the fixed and wrist RealSense), state (7: six joints in rad + gripper opening in m),
actions (7: the next 10 Hz state), task = the instruction (becomes the prompt).

Transforms are openpi's LIBERO ones: fixed camera -> base_0_rgb, wrist -> left_wrist_0_rgb, right-wrist slot zero and
masked; with extra_delta_transform the six joints of each action chunk become deltas from the current state (gripper
stays absolute) and are turned back into absolute joint targets at inference. Robot-side request keys:
observation/image, observation/wrist_image, observation/state, prompt; the reply holds `actions` (10 x 7).

Training follows openpi's real-robot configs (pi05_droid_finetune: batch 32, default AdamW / EMA 0.99 / warmup 1k,
peak 2.5e-5) with 6k steps and the cosine ending at step 6000. openpi keeps the trainable weights, AdamW moments and EMA
in fp32 and computes in bf16; FSDP shards them over the 4 A100-40GB of a g3 node.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ENV_ROOT = Path(os.environ.get("ENV_ROOT", "/scratch/users/ntu/yguo017/openpi-pi05"))
RUN = Path(os.environ.get("RUN", "/scratch/users/ntu/yguo017/pi05-agilex-3task-20261114"))
REPO = Path(os.environ.get("OPENPI_REPO", str(ENV_ROOT / "openpi")))
sys.path.insert(0, str(REPO))  # openpi's scripts/ is a package next to src/

import openpi.models.pi0_config as pi0_config  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.optimizer as _optimizer  # noqa: E402
import openpi.training.weight_loaders as weight_loaders  # noqa: E402

NAME = "pi05_agilex3"
REPO_ID = "local/agilex3_10hz"
STEPS = 6000

CONFIG = _config.TrainConfig(
    name=NAME,
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10),
    data=_config.LeRobotLiberoDataConfig(
        repo_id=REPO_ID,
        base_config=_config.DataConfig(prompt_from_task=True),
        extra_delta_transform=True,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        str(ENV_ROOT / "cache/openpi/openpi-assets/checkpoints/pi05_base/params")),
    lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=2.5e-5, decay_steps=STEPS, decay_lr=2.5e-6),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.99,
    batch_size=32,
    num_train_steps=STEPS,
    log_interval=20,
    save_interval=2000,
    keep_period=2000,
    fsdp_devices=4,
    num_workers=12,
    assets_base_dir=str(RUN / "assets"),
    checkpoint_base_dir=str(RUN / "checkpoints"),
    wandb_enabled=False,
)
_config._CONFIGS.append(CONFIG)
_config._CONFIGS_DICT[NAME] = CONFIG


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("norm-stats", "train", "serve"):
        raise SystemExit(__doc__)
    action, rest = sys.argv[1], sys.argv[2:]
    if action == "norm-stats":
        import scripts.compute_norm_stats as compute_norm_stats
        compute_norm_stats.main(NAME)
    elif action == "train":
        import scripts.train as train
        sys.argv = [sys.argv[0], NAME, *rest]
        train.main(_config.cli())
    else:
        import tyro
        import scripts.serve_policy as serve_policy
        sys.argv = [sys.argv[0], *rest]
        serve_policy.main(tyro.cli(serve_policy.Args))


if __name__ == "__main__":
    main()
