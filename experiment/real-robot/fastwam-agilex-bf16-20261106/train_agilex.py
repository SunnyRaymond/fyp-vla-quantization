"""FastWAM training entry for the AgileX run: scripts/train.py plus the fw_agilex hooks.

Lives in <artifacts>/scripts/ next to a copy of FastWAM's configs/ (with configs/{data,task} for AgileX added), so the
hydra config path is ../configs exactly as in the upstream script. deepspeed is imported before GroupedAdamW is
installed so its ZeRO optimizer whitelist keeps the real torch classes (ds config also sets
zero_allow_untested_optimizer). With FW_DUMP_CONFIG=<path> it only writes the composed config there (prep job).
"""
import os
from pathlib import Path

DUMP = os.environ.get("FW_DUMP_CONFIG")
if not DUMP:
    import deepspeed  # noqa: F401  (import order, see above)

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import fw_agilex  # noqa: E402

if not DUMP:
    fw_agilex.install()

from fastwam.runtime import run_training  # noqa: E402
from fastwam.utils.config_resolvers import register_default_resolvers  # noqa: E402

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    if DUMP:
        Path(DUMP).write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
        return
    run_training(cfg)


if __name__ == "__main__":
    main()
