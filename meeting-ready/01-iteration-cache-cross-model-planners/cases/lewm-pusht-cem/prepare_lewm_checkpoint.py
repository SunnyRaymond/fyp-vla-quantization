#!/usr/bin/env python3
"""Convert the official LeWM HF state dict to the object checkpoint API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import stable_pretraining as spt
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    source = args.stage_root / "hf_pusht"
    target = args.stablewm_home / "pusht" / "lewm_object.ckpt"
    cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))

    def module_kwargs(value: dict) -> dict:
        # The official Hydra config carries construction metadata that the
        # pinned Python constructors do not accept.
        return {key: item for key, item in value.items() if key != "_target_"}

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
        predictor=ARPredictor(**module_kwargs(cfg["predictor"])),
        action_encoder=Embedder(**module_kwargs(cfg["action_encoder"])),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )
    state = torch.load(source / "weights.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, target)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
