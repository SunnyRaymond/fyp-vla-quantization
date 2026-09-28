#!/usr/bin/env python3
"""Guarded CPU-only PLDM runtime/configuration/data-loader resume probe."""
from __future__ import annotations

import importlib.metadata
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("PLDM_ROOT", "/scratch/users/ntu/yguo017/pldm-reproduction"))
SOURCE_SHA = "1bd7e564ecd961205bc18b23067b19e9ca24ac90"
PREVIOUS_ATTEMPT = "25570653.pbs101"
GYM_OVERLAY = Path("/scratch/users/ntu/yguo017/experiment/reproduction/lpwm/runtime-overlay-minimal")
ARM_OVERLAY = ROOT / "runtime_overlay" / "arm-pytorch-utilities-0.4.3"
PYTORCH_SEED_OVERLAY = ROOT / "runtime_overlay" / "pytorch-seed-0.2.0"
STATSMODELS_OVERLAY = ROOT / "runtime_overlay" / "statsmodels-patsy-pinned"
SCIENTIFIC_OVERLAY = ROOT / "runtime_overlay" / "numpy-1.26.4-scipy-1.10.0-pinned"
PANDAS_OVERLAY = ROOT / "runtime_overlay" / "pandas-2.0.1-pinned"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{job_id}")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def allocation() -> tuple[str, str]:
    job_id = os.environ.get("PBS_JOBID", "")
    nodefile = os.environ.get("PBS_NODEFILE", "")
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("Refusing CPU data/config work without PBS_JOBID and PBS_NODEFILE")
    host = socket.gethostname().split(".")[0].lower()
    if any(part in host for part in ("login", "head", "submit")):
        raise RuntimeError(f"Refusing probable login node: {host}")
    nodes = {
        line.split()[0].split(".")[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in nodes:
        raise RuntimeError(f"Current host {host} is not in PBS_NODEFILE")
    return job_id, host


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    job_id, host = allocation()
    report_dir = ROOT / "reports" / job_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "PREPARATION.json"
    observed_dependency_paths = sorted(report_dir.glob("observed-dependency-*.json"))
    observed_dependency_records = [json.loads(path.read_text(encoding="utf-8")) for path in observed_dependency_paths]
    report = {
        "status": "RUNNING",
        "cpu_job_id": job_id,
        "previous_attempt": str(ROOT / "reports" / PREVIOUS_ATTEMPT / "PREPARATION.json"),
        "pbs_host": host,
        "started_utc": utc_now(),
        "source_sha": SOURCE_SHA,
        "remote_root": str(ROOT),
        "scope": "Reuse completed official render; validate CPU imports, official config composition, and one real training-loader batch only",
        "observed_dependency_installs": observed_dependency_records,
        "resource_amendments": [
            {"allocation_walltime_minutes": 45, "trigger": "Observed statsmodels.api import failure: cannot import name _lazywhere from inherited SciPy 1.17.1", "isolated_install": "official binary wheels only, --no-deps: numpy==1.26.4 and scipy==1.10.0", "shared_runtime_modified": False},
            {"allocation_walltime_minutes": 15, "trigger": "Observed statsmodels.api import failure: installed pandas 3.0.6 deprecate_kwarg API is incompatible with official statsmodels 0.14.4; upstream requirements.txt pins pandas==2.0.1 and its exact pinned pytz==2022.7.1 is absent from the inherited runtime", "isolated_install": "official pandas==2.0.1 and pytz==2022.7.1 binary wheels only, --no-deps", "shared_runtime_modified": False},
        ],
    }
    atomic_json(report_path, report, job_id)

    try:
        previous_path = ROOT / "reports" / PREVIOUS_ATTEMPT / "PREPARATION.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous.get("source_sha") != SOURCE_SHA:
            raise RuntimeError("Previous render attempt source SHA differs from the frozen commit")
        if previous.get("stages", {}).get("render", {}).get("status") != "PASS":
            raise RuntimeError("Previous attempt did not complete the official dataset render")

        upstream = ROOT / "upstream"
        staged = ROOT / "staged"
        dataset = ROOT / "datasets" / "good_quality_data_memmap"
        config_rel = Path("configs/wall/icml/seqlen90_3M.yaml")
        config_path = staged / "pldm" / config_rel
        source_identity = json.loads((upstream / "source_identity.json").read_text(encoding="utf-8"))
        staged_identity = json.loads((ROOT / "staged_source_identity.json").read_text(encoding="utf-8"))
        metadata = json.loads((dataset / "render_metadata.json").read_text(encoding="utf-8"))
        if source_identity.get("source_sha") != SOURCE_SHA or staged_identity.get("source_sha") != SOURCE_SHA:
            raise RuntimeError("Pinned source identity gate failed")
        if staged_identity.get("algorithm_or_config_changes") != []:
            raise RuntimeError("Staged source reports algorithm/config changes")
        if metadata.get("source_commit") != SOURCE_SHA or metadata.get("states_dtype") != "uint8" or metadata.get("frame_count") != 3_686_400:
            raise RuntimeError("Previously rendered official dataset identity gate failed")
        if not config_path.is_file():
            raise RuntimeError(f"Official frozen training config is missing: {config_path}")
        if not GYM_OVERLAY.is_dir() or not ARM_OVERLAY.is_dir() or not PYTORCH_SEED_OVERLAY.is_dir() or not STATSMODELS_OVERLAY.is_dir() or not SCIENTIFIC_OVERLAY.is_dir() or not PANDAS_OVERLAY.is_dir():
            raise RuntimeError(f"Required isolated dependency overlay is missing: gym={GYM_OVERLAY.is_dir()} arm={ARM_OVERLAY.is_dir()} pytorch_seed={PYTORCH_SEED_OVERLAY.is_dir()} statsmodels={STATSMODELS_OVERLAY.is_dir()} numpy_scipy={SCIENTIFIC_OVERLAY.is_dir()} pandas={PANDAS_OVERLAY.is_dir()}")

        overlays = [
            SCIENTIFIC_OVERLAY,
            PANDAS_OVERLAY,
            *[Path(record["overlay"]) for record in observed_dependency_records],
            ROOT / "runtime_overlay" / "zarr-2.14.2-pinned",
            ROOT / "runtime_overlay" / "gdown",
            ROOT / "runtime_overlay" / "html-parser",
            GYM_OVERLAY,
            ARM_OVERLAY,
            PYTORCH_SEED_OVERLAY,
            STATSMODELS_OVERLAY,
        ]
        overlays = [path.resolve() for path in overlays if path.is_dir()]
        sys.path[:0] = [str(staged.resolve()), *map(str, overlays)]

        import numpy as np
        import scipy
        import pandas
        import pytz
        import torch
        import zarr
        import numcodecs
        import gym
        import gym_notices
        import arm_pytorch_utilities
        import pytorch_seed
        import statsmodels
        import statsmodels.api
        import patsy
        import pldm.train as train_module
        import pldm.evaluation.evaluator as evaluator_module
        import pldm_envs.wall.data.offline_wall as offline_wall_module
        from pldm.data.utils import make_dataloader

        torch.set_num_threads(1)
        source_modules = (train_module, evaluator_module, offline_wall_module)
        module_origins = {}
        for module in source_modules:
            origin = Path(module.__file__).resolve()
            if not is_under(origin, staged):
                raise RuntimeError(f"PLDM import escaped staged source: {origin}")
            module_origins[module.__name__] = str(origin)
        for module in (zarr, numcodecs):
            origin = Path(module.__file__).resolve()
            if not is_under(origin, ROOT / "runtime_overlay" / "zarr-2.14.2-pinned"):
                raise RuntimeError(f"Zarr dependency import escaped its task overlay: {origin}")
            module_origins[module.__name__] = str(origin)
        for module in (gym, gym_notices):
            origin = Path(module.__file__).resolve()
            if not is_under(origin, GYM_OVERLAY):
                raise RuntimeError(f"Gym dependency import escaped the validated read-only overlay: {origin}")
            module_origins[module.__name__] = str(origin)
        arm_origin = Path(arm_pytorch_utilities.__file__).resolve()
        if not is_under(arm_origin, ARM_OVERLAY):
            raise RuntimeError(f"arm_pytorch_utilities import escaped its pinned task overlay: {arm_origin}")
        module_origins["arm_pytorch_utilities"] = str(arm_origin)
        seed_origin = Path(pytorch_seed.__file__).resolve()
        if not is_under(seed_origin, PYTORCH_SEED_OVERLAY):
            raise RuntimeError(f"pytorch_seed import escaped its pinned task overlay: {seed_origin}")
        module_origins["pytorch_seed"] = str(seed_origin)
        for module in (statsmodels, statsmodels.api, patsy):
            origin = Path(module.__file__).resolve()
            if not is_under(origin, STATSMODELS_OVERLAY):
                raise RuntimeError(f"Statsmodels dependency import escaped its isolated overlay: {origin}")
            module_origins[module.__name__] = str(origin)
        for module, version in ((np, "1.26.4"), (scipy, "1.10.0")):
            origin = Path(module.__file__).resolve()
            if not is_under(origin, SCIENTIFIC_OVERLAY):
                raise RuntimeError(f"Scientific stack import escaped its isolated pinned overlay: {module.__name__}={origin}")
            if module.__version__ != version:
                raise RuntimeError(f"Scientific stack version mismatch: {module.__name__}={module.__version__}, expected {version}")
            module_origins[module.__name__] = str(origin)
        if pandas.__version__ != "2.0.1" or not is_under(Path(pandas.__file__).resolve(), PANDAS_OVERLAY):
            raise RuntimeError(f"Pandas import/version escaped its official pinned task overlay: version={pandas.__version__} origin={pandas.__file__}")
        module_origins["pandas"] = str(Path(pandas.__file__).resolve())
        if importlib.metadata.version("pytz") != "2022.7.1" or not is_under(Path(pytz.__file__).resolve(), PANDAS_OVERLAY):
            raise RuntimeError(f"Pytz import/version escaped its official pinned task overlay: version={importlib.metadata.version('pytz')} origin={pytz.__file__}")
        module_origins["pytz"] = str(Path(pytz.__file__).resolve())
        observed_dependency_origins = {}
        for record in observed_dependency_records:
            module = __import__(record["trigger_module_not_found"])
            version = importlib.metadata.version(record["distribution"])
            origin = Path(module.__file__).resolve()
            overlay = Path(record["overlay"])
            if version != record["version"] or not is_under(origin, overlay):
                raise RuntimeError(f"Observed dependency overlay gate failed: {record['distribution']} version={version} origin={origin} expected={record['version']} under {overlay}")
            module_origins[record["trigger_module_not_found"]] = str(origin)
            observed_dependency_origins[record["distribution"]] = {"version": version, "origin": str(origin), "overlay": str(overlay)}

        config_values = [
            f"output_root={ROOT / 'checkpoints'}",
            "output_dir=tworooms-seqlen90-3M-seed101",
            "run_name=tworooms-seqlen90-3M-seed101",
            f"data.offline_wall_config.offline_data_path={dataset}",
            "data.offline_wall_config.lazy_load=true",
            "data.offline_wall_config.device=cpu",
            "seed=101",
            "wandb=false",
        ]

        def compose(flag: str) -> dict:
            sys.argv = ["train.py", flag, str(config_rel), "--values", *config_values]
            cfg = train_module.TrainConfig.parse_from_command_line()
            planning = cfg.eval_cfg.wall_planning
            return {
                "env_name": cfg.env_name,
                "seed": cfg.seed,
                "epochs": cfg.epochs,
                "quick_debug": cfg.quick_debug,
                "n_steps": cfg.n_steps,
                "objectives": [getattr(value, "name", str(value)) for value in cfg.objectives_l1.objectives],
                "optimizer_type": str(cfg.optimizer_type),
                "base_lr": cfg.base_lr,
                "compile_model": cfg.compile_model,
                "wandb": cfg.wandb,
                "lazy_load": cfg.data.offline_wall_config.lazy_load,
                "offline_device": str(cfg.data.offline_wall_config.device),
                "offline_path": str(cfg.data.offline_wall_config.offline_data_path),
                "eval_levels": planning.levels,
                "eval_envs": planning.n_envs,
                "eval_steps": planning.n_steps,
                "eval_batch": planning.n_envs_batch_size,
                "replan_every": planning.replan_every,
                "mppi_samples": planning.level1.mppi.num_samples,
                "eval_mpcs": cfg.eval_mpcs,
                "probing_pred_epochs": cfg.eval_cfg.probing.epochs,
                "probing_encoder_epochs": cfg.eval_cfg.probing.epochs_enc,
                "probing_l1_depth": cfg.eval_cfg.probing.l1_depth,
                "probing_preds_enabled": cfg.eval_cfg.probing.probe_preds,
                "probing_encoder_enabled": cfg.eval_cfg.probing.probe_encoder,
            }

        plural_config = compose("--configs")
        singular_config = compose("--config")
        expected_config = {
            "env_name": "wall",
            "seed": 101,
            "epochs": 2,
            "quick_debug": False,
            "n_steps": 16,
            "objectives": ["VICReg", "IDM"],
            "wandb": False,
            "lazy_load": True,
            "offline_device": "cpu",
            "offline_path": str(dataset),
            "eval_levels": "medium",
            "eval_envs": 100,
            "eval_steps": 200,
            "eval_batch": 20,
            "replan_every": 1,
            "mppi_samples": 2000,
            "eval_mpcs": 20,
            "probing_pred_epochs": 20,
            "probing_encoder_epochs": 30,
            "probing_l1_depth": 16,
            "probing_preds_enabled": True,
            "probing_encoder_enabled": True,
        }
        config_pass = plural_config == singular_config and all(plural_config.get(key) == value for key, value in expected_config.items())
        if not config_pass:
            raise RuntimeError(f"Official config composition gate failed: {plural_config}")

        # The loader/normalizer reads the actual complete memmap dataset. This is
        # deliberately CPU-only; the frozen GPU device remains CUDA in formal runs.
        sys.argv = ["train.py", "--configs", str(config_rel), "--values", *config_values]
        cfg = train_module.TrainConfig.parse_from_command_line()
        load_started = time.monotonic()
        dataset_obj = offline_wall_module.OfflineWallDataset(cfg.data.offline_wall_config)
        loader = make_dataloader(dataset_obj, loader_config=cfg.data)
        loader_build_seconds = time.monotonic() - load_started
        batch_started = time.monotonic()
        first_batch = next(iter(loader))
        first_batch_seconds = time.monotonic() - batch_started
        epoch_indices = list(range(0, int(cfg.epochs) + 1))
        loader_report = {
            "status": "PASS" if len(epoch_indices) == 3 and len(loader) > 0 else "FAIL",
            "dataset_samples": len(dataset_obj),
            "batch_size": loader.dataloader.batch_size,
            "drop_last": loader.dataloader.drop_last,
            "optimizer_updates_per_epoch": len(loader),
            "configured_epochs": cfg.epochs,
            "actual_loop_epoch_indices": epoch_indices,
            "actual_passes": len(epoch_indices),
            "expected_optimizer_updates": len(loader) * len(epoch_indices),
            "training_n_steps": cfg.data.offline_wall_config.n_steps,
            "loader_build_and_normalizer_seconds": loader_build_seconds,
            "first_batch_seconds": first_batch_seconds,
            "first_batch_state_shape": list(first_batch.states.shape),
            "first_batch_state_dtype": str(first_batch.states.dtype),
            "loader_type": type(loader.dataloader).__name__,
            "lazy_load": cfg.data.offline_wall_config.lazy_load,
            "loader_config_device_override_for_cpu_preflight": str(cfg.data.offline_wall_config.device),
            "dataset_runtime_device": str(dataset_obj.device),
        }
        if loader_report["status"] != "PASS":
            raise RuntimeError(f"Actual official training-loader gate failed: {loader_report}")

        # Construct the official online probing loaders only to read their
        # metadata lengths. Reuse the real offline normalizer and never iterate
        # or generate a probing batch during CPU preparation.
        from pldm.data.dataset_factory import DatasetFactory

        original_probe_device = str(cfg.data.wall_config.device)
        cfg.data.wall_config.device = "cpu"
        probe_factory = DatasetFactory(cfg.data, probing_cfg=cfg.eval_cfg.probing)
        probing_datasets = probe_factory._create_wall_probing_datasets(dataset_obj.normalizer)
        probe_train_batches = len(probing_datasets.ds)
        probe_val_batches = len(probing_datasets.val_ds)
        probe_config = cfg.eval_cfg.probing
        probe_training_passes = int(probe_config.epochs) + int(probe_config.epochs_enc)
        probe_validation_passes = int(bool(probe_config.probe_preds)) + int(bool(probe_config.probe_encoder))
        probing_report = {
            "status": "PASS" if probe_train_batches > 0 and probe_val_batches > 0 and probe_training_passes == 50 else "FAIL",
            "source": "official DatasetFactory._create_wall_probing_datasets",
            "train_loader_batches_per_pass": probe_train_batches,
            "validation_loader_batches_per_evaluator_pass": probe_val_batches,
            "configured_batch_size": cfg.data.wall_config.batch_size,
            "configured_sequence_depth": probe_config.l1_depth,
            "pred_prober_training_epochs": probe_config.epochs,
            "encoder_prober_training_epochs": probe_config.epochs_enc,
            "total_training_passes": probe_training_passes,
            "training_batches_across_50_passes": probe_training_passes * probe_train_batches,
            "validation_passes_from_official_eval_flags": probe_validation_passes,
            "validation_batches_across_enabled_eval_passes": probe_validation_passes * probe_val_batches,
            "shape_probe_batches_in_official_methods": 2,
            "total_batch_work_metadata_estimate": probe_training_passes * probe_train_batches + probe_validation_passes * probe_val_batches + 2,
            "probe_batch_iteration_during_cpu_preparation": False,
            "probe_data_device_override_for_length_only": {"original": original_probe_device, "preflight": "cpu"},
            "offline_normalizer_reused": True,
        }
        if probing_report["status"] != "PASS":
            raise RuntimeError(f"Official probing-loader metadata gate failed: {probing_report}")

        versions = {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pandas.__version__,
            "pytz": importlib.metadata.version("pytz"),
            "zarr": zarr.__version__,
            "numcodecs": numcodecs.__version__,
            "gym": importlib.metadata.version("gym"),
            "gym-notices": importlib.metadata.version("gym-notices"),
            "arm_pytorch_utilities": importlib.metadata.version("arm_pytorch_utilities"),
            "pytorch-seed": importlib.metadata.version("pytorch-seed"),
            "statsmodels": importlib.metadata.version("statsmodels"),
            "patsy": importlib.metadata.version("patsy"),
        }
        stages = {
            "source": previous.get("stages", {}).get("source"),
            "render": previous.get("stages", {}).get("render"),
            "zarr_overlay_install": {"status": "REUSED", "path": str(ROOT / "runtime_overlay" / "zarr-2.14.2-pinned")},
            "gym_overlay_reuse": {"status": "PASS", "path": str(GYM_OVERLAY), "versions": {"gym": versions["gym"], "gym-notices": versions["gym-notices"]}},
            "arm_pytorch_utilities_overlay": {"status": "PASS", "path": str(ARM_OVERLAY), "version": versions["arm_pytorch_utilities"]},
            "pytorch_seed_overlay": {"status": "PASS", "path": str(PYTORCH_SEED_OVERLAY), "version": versions["pytorch-seed"]},
            "statsmodels_overlay": {"status": "PASS", "path": str(STATSMODELS_OVERLAY), "versions": {"statsmodels": versions["statsmodels"], "patsy": versions["patsy"]}},
            "numpy_scipy_resource_amendment": {"status": "PASS", "path": str(SCIENTIFIC_OVERLAY), "versions": {"numpy": versions["numpy"], "scipy": versions["scipy"]}, "allocation_walltime_minutes": 45, "trigger": "Observed statsmodels.api import failure: cannot import name _lazywhere from inherited SciPy 1.17.1; isolated official pinned binary wheels only, no-deps; shared runtime unchanged"},
            "pandas_api_compatibility_amendment": {"status": "PASS", "path": str(PANDAS_OVERLAY), "version": versions["pandas"], "allocation_walltime_minutes": 15, "trigger": "Observed statsmodels 0.14.4 import failure with inherited pandas 3.0.6; upstream requirements.txt pins pandas==2.0.1"},
            "observed_dependency_installs": {"status": "PASS", "records": observed_dependency_records, "runtime_origins": observed_dependency_origins},
            "staged_source_compatibility": {"status": "PASS", "patches": staged_identity.get("patches", [])},
            "runtime_probe_before_data": {"status": "PASS", "versions": versions, "module_origins": module_origins},
            "config_composition": {"status": "PASS", "plural_and_singular_flags_match": True, "config": plural_config},
            "probing_loader_metadata": probing_report,
        }
        final = {
            "status": "PASS",
            "cpu_job_id": job_id,
            "render_job_id": previous.get("cpu_job_id"),
            "previous_attempt": str(previous_path),
            "pbs_host": host,
            "started_utc": report["started_utc"],
            "finished_utc": utc_now(),
            "source_sha": SOURCE_SHA,
            "remote_root": str(ROOT),
            "runtime_python": sys.executable,
            "runtime_versions": versions,
            "module_origins": module_origins,
            "staged_source_identity": staged_identity,
            "observed_dependency_installs": observed_dependency_records,
            "dataset_path": str(dataset),
            "dataset_metadata": metadata,
            "cpu_config_device_override": "cpu",
            "formal_config_device": "cuda",
            "stages": stages,
            "actual_training_loader": loader_report,
            "actual_probing_loader": probing_report,
        }
        atomic_json(report_path, final, job_id)
        atomic_json(ROOT / "PREPARATION.json", final, job_id)
        print(json.dumps({
            "status": final["status"],
            "cpu_job_id": job_id,
            "source_sha": SOURCE_SHA,
            "loader": loader_report,
            "probing_loader": probing_report,
            "runtime_versions": versions,
            "module_origins": module_origins,
        }, indent=2), flush=True)
        return 0
    except Exception as exc:
        report.update({
            "status": "FAIL",
            "ended_utc": utc_now(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc()[-7000:],
            },
        })
        atomic_json(report_path, report, job_id)
        atomic_json(ROOT / "PREPARATION.json", report, job_id)
        print(f"CPU_RESUME_FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
