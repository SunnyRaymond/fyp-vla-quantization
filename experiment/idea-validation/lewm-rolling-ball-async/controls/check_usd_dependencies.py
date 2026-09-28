import json
from pathlib import Path

from pxr import UsdUtils


asset_root = Path("/root/autodl-tmp/rolling-ball-lewm/assets/local/Assets/Isaac/5.1")
roots = (
    "Isaac/Props/Mugs/SM_Mug_A2.usd",
    "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd",
    "Isaac/Environments/Grid/default_environment.usd",
)

for relative_path in roots:
    source = asset_root / relative_path
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(source))
    layer_paths = [Path(layer.realPath) for layer in layers]
    asset_paths = [Path(path) for path in assets]
    file_paths = sorted(set(layer_paths + asset_paths))
    missing = [str(path) for path in file_paths if not path.is_file()]
    report = {
        "root": relative_path,
        "usd_layers": len(layer_paths),
        "file_assets": len(asset_paths),
        "files": [str(path.relative_to(asset_root)) for path in file_paths],
        "bytes": sum(path.stat().st_size for path in file_paths if path.is_file()),
        "missing_files": missing,
        "unresolved_nonfile_assets": sorted(set(unresolved)),
    }
    print(json.dumps(report, sort_keys=True))
    if missing or set(unresolved) != {"OmniPBR.mdl"}:
        raise SystemExit(f"dependency gate failed for {relative_path}")
