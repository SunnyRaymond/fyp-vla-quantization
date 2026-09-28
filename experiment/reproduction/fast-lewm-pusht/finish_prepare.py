"""Finish the already staged source/runtime in a CPU PBS allocation."""
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import traceback
import urllib.request
from pathlib import Path

ROOT = Path("/scratch/users/ntu/yguo017/fast-lewm-pusht-reproduction")
STAGE = Path("/scratch/users/ntu/yguo017/lewm-pusht-iteration")


def save(path, value):
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def main():
    job = os.environ.get("PBS_JOBID")
    host = platform.node().split(".")[0].lower()
    nodefile = Path(os.environ.get("PBS_NODEFILE", "/missing"))
    if not job or not nodefile.is_file() or any(word in host for word in ("login", "head", "submit")):
        raise RuntimeError("CPU allocation is required")
    if host not in {x.strip().split(".")[0].lower() for x in nodefile.read_text().splitlines()}:
        raise RuntimeError("Host is outside allocation")
    out = ROOT / "artifacts" / job
    out.mkdir(exist_ok=True, parents=True)
    previous = json.loads((ROOT / "prepared.json").read_text())
    save(out / "previous_preparation.json", previous)
    report = {"status": "RUNNING", "job_id": job, "compute_host": host,
              "source_sha": json.loads((ROOT / "source_identity.json").read_text())["source_sha"],
              "source_path": str(ROOT / "upstream"), "stablewm_root": str(ROOT / "runtime_overlay"),
              "runtime_python": sys.executable, "resumes_job": previous["job_id"]}
    try:
        sys.path[:0] = [str(ROOT / "upstream"), str(ROOT / "runtime_overlay")]
        metadata_url = "https://huggingface.co/api/models/naiverer/fast-leworldmodel?blobs=true"
        with urllib.request.urlopen(metadata_url, timeout=45) as response:
            metadata = json.load(response)
        revision = metadata["sha"]
        name = "Fast-lewm_pusht_object.ckpt"
        entry = next(x for x in metadata["siblings"] if x["rfilename"] == name)
        size = entry.get("size", entry.get("lfs", {}).get("size"))
        checkpoint = ROOT / "checkpoints" / name
        checkpoint.parent.mkdir(exist_ok=True)
        if checkpoint.exists():
            raise FileExistsError("A writer already completed the checkpoint")
        url = f"https://huggingface.co/naiverer/fast-leworldmodel/resolve/{revision}/{name}"
        partial = checkpoint.with_suffix(".partial")
        if partial.exists():
            raise FileExistsError("A checkpoint partial already exists")
        with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if size is not None and partial.stat().st_size != size:
            raise RuntimeError("Downloaded size differs from the fixed release metadata")
        partial.replace(checkpoint)
        report.update(hf_revision=revision, checkpoint_url=url, checkpoint_path=str(checkpoint),
                      checkpoint_size_bytes=checkpoint.stat().st_size)
        import hdf5plugin
        import hydra
        import stable_worldmodel as swm
        import torch
        from hydra.core.hydra_config import HydraConfig
        from omegaconf import OmegaConf
        from jepa import JEPA
        model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(model, JEPA):
            raise TypeError(f"Unexpected released checkpoint type {type(model)}")
        with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "upstream" / "config" / "eval")):
            cfg = hydra.compose(config_name="pusht", return_hydra_config=True)
        cfg.hydra.runtime.output_dir = str(out / "official")
        HydraConfig.instance().set_config(cfg)
        frozen = json.loads((ROOT / "source" / "FREEZE.json").read_text())
        for key, value in frozen["settings"].items():
            assert OmegaConf.select(cfg, key) == value, key
        versions = {}
        for package in ("torch", "torchvision", "stable-worldmodel", "stable-pretraining", "transformers",
                        "numpy", "pymunk", "gymnasium", "mujoco", "h5py", "hdf5plugin", "hydra-core"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        report.update(status="PASS", checkpoint_load="PASS", config_compose="PASS", versions=versions,
                      model_class=str(type(model)), model_parameters=sum(p.numel() for p in model.parameters()),
                      action_encoder_class=str(type(model.action_encoder)), predictor_class=str(type(model.predictor)),
                      stablewm_file=swm.__file__, has_auto_cost_model=hasattr(swm.policy, "AutoCostModel"))
        report["world_methods"] = {}
        for name in ("evaluate", "evaluate_from_dataset", "step"):
            method = getattr(swm.World, name, None)
            report["world_methods"][name] = None if method is None else {
                "signature": str(inspect.signature(method)), "source_tail": inspect.getsource(method).splitlines()[-65:]}
        report["policy_constructor"] = inspect.getsource(swm.policy.WorldModelPolicy.__init__).splitlines()
    except Exception:
        report.update(status="FAIL", error=traceback.format_exc())
    save(out / "prepared.json", report)
    save(ROOT / "prepared.json", report)
    print(json.dumps({k: report.get(k) for k in ("status", "job_id", "checkpoint_load", "config_compose", "error")}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
