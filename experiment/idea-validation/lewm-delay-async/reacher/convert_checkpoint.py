"""Convert the pinned official Reacher HF weights to the LeWM object checkpoint."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path


def main() -> None:
    nodefile = Path(os.environ.get("PBS_NODEFILE", "/no-pbs-nodefile"))
    host = platform.node().split(".", 1)[0].lower()
    if not os.environ.get("PBS_JOBID") or not nodefile.is_file() or "login" in host:
        raise RuntimeError("checkpoint conversion requires a PBS compute allocation")
    allocated = {line.split(".", 1)[0].lower() for line in nodefile.read_text().splitlines() if line.strip()}
    if host not in allocated:
        raise RuntimeError("host not in PBS_NODEFILE")

    import torch
    import stable_pretraining as spt
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    stage = Path("/scratch/users/ntu/yguo017/lewm-pusht-iteration")
    source = stage / "hf_reacher"
    target = stage / "stablewm_home/reacher/lewm_object.ckpt"
    cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))

    def params(key: str) -> dict:
        return {k: v for k, v in cfg[key].items() if k != "_target_"}

    encoder = spt.backbone.utils.vit_hf(
        cfg["encoder"]["size"],
        patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"],
        pretrained=False,
        use_mask_token=False,
    )

    def mlp(key: str):
        return MLP(
            input_dim=cfg[key]["input_dim"],
            output_dim=cfg[key]["output_dim"],
            hidden_dim=cfg[key]["hidden_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**params("predictor")),
        action_encoder=Embedder(**params("action_encoder")),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )
    model.load_state_dict(torch.load(source / "weights.pt", map_location="cpu", weights_only=False), strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, target)
    print(f"converted_checkpoint={target} size={target.stat().st_size}")


if __name__ == "__main__":
    main()
