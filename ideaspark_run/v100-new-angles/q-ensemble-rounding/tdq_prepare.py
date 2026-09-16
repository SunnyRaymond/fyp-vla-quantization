"""Stage the pinned TD-MPC2 asset and prepare reset-only Cartpole inputs.

All downloads, hashing, archive extraction, dependency imports, and DMControl
resets are performed by the bounded CPU allocation.  This file never loads a
checkpoint with torch and never calls env.step or env.render.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import socket
import sys
import tarfile
import time
from typing import Any
import urllib.error
import urllib.request


ROOT_DEFAULT = Path("/tc1home/UG/yguo017/v100_newangles_ccds")
SOURCE_COMMIT = "e9f59321933cbc8e11a002b842adc7d4ffae8ff1"
HF_REVISION = "73a50e2719ed8258c72c7d1fefd23b781d66e35e"
CHECKPOINT_SHA256 = "4919e562d7f22f41a11118d1db1a0ebcb0e2b4681fc7fc594d772f0d5940869b"
CHECKPOINT_SIZE = 31_344_610
MODEL_REPO = "nicklashansen/tdmpc2"
TASK = "cartpole-balance"
SEEDS = (5201, 5202, 5203, 5204, 5205, 5206, 5207, 5208)
SOURCE_URL = f"https://github.com/nicklashansen/tdmpc2/archive/{SOURCE_COMMIT}.tar.gz"
CHECKPOINT_URL = (
    f"https://huggingface.co/{MODEL_REPO}/resolve/{HF_REVISION}/"
    "dmcontrol/cartpole-balance-1.pt?download=true"
)
DOWNLOAD_LIMIT = 500 * 1024 * 1024
SELECTED_SOURCE = (
    "tdmpc2/common/__init__.py",
    "tdmpc2/common/parser.py",
    "tdmpc2/common/world_model.py",
    "tdmpc2/common/layers.py",
    "tdmpc2/common/math.py",
    "tdmpc2/tdmpc2.py",
    "tdmpc2/config.yaml",
    "tdmpc2/envs/dmcontrol.py",
    "tdmpc2/envs/wrappers/timeout.py",
    "docker/environment.yaml",
)
EXPECTED_DEPENDENCIES = {
    "dm-control": "1.0.16",
    "mujoco": "3.1.2",
    "tensordict": "0.7.2",
    "omegaconf": "2.3.0",
}
EXPECTED_ENVIRONMENT_DEPENDENCIES = {"gymnasium": "0.29.1"}
RAW_SCHEMA = "tdmpc2-cartpole-reset-input-v1"
MANIFEST_SCHEMA = "tdmpc2-q-coupling-preparation-v1"


class PreparationError(RuntimeError):
    pass


class ResourceBlocked(PreparationError):
    pass


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_status(out: Path, value: dict[str, Any], assetdir: Path | None = None) -> None:
    payload = {
        "schema": "tdmpc2-q-coupling-preparation-status-v1",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname().split(".", 1)[0],
        "updated_epoch": int(time.time()),
        **value,
    }
    _json_write(out / "status.json", payload)
    if assetdir is not None and assetdir.exists():
        _json_write(assetdir / "status.json", payload)


def _allocation() -> dict[str, Any]:
    from allocation_guard import require_allocation

    return require_allocation()


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _all_dist_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if isinstance(name, str) and isinstance(version, str):
            canonical = re.sub(r"[-_.]+", "-", name).lower()
            versions.setdefault(canonical, version)
    return dict(sorted(versions.items()))


def _runtime_identity() -> dict[str, Any]:
    installed_distributions = _all_dist_versions()
    versions = {
        name: installed_distributions.get(re.sub(r"[-_.]+", "-", name).lower())
        for name in ("numpy", "torch", "dm-control", "mujoco", "gymnasium", "tensordict", "omegaconf")
    }
    missing = [name for name, version in versions.items() if version is None]
    mismatch = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in EXPECTED_DEPENDENCIES.items()
        if versions.get(name) != expected
    }
    non_exact_environment = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in EXPECTED_ENVIRONMENT_DEPENDENCIES.items()
        if versions.get(name) != expected
    }
    if missing:
        raise ResourceBlocked(f"required runtime distributions are missing: {missing}")
    if mismatch:
        raise ResourceBlocked(f"DMControl dependency identity mismatch: {mismatch}")
    if sys.version_info < (3, 10):
        raise ResourceBlocked(f"Python 3.10+ required, got {sys.version}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in {"", "-1"}:
        raise PreparationError("CPU job has a visible CUDA device")
    try:
        import tensordict
        from tensordict.nn import TensorDictParams
        import omegaconf
        from omegaconf import OmegaConf
    except Exception as exc:
        raise ResourceBlocked(f"required runtime imports are unavailable: {exc}") from exc
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "installed_distributions": installed_distributions,
        "exact_dependency_mismatches": mismatch,
        "non_exact_environment_dependencies": non_exact_environment,
        "imports": {
            "tensordict": tensordict.__name__,
            "TensorDictParams": TensorDictParams.__name__,
            "omegaconf": omegaconf.__name__,
            "OmegaConf": OmegaConf.__name__,
        },
        "expected_official_environment": {
            "python": "3.11",
            "torch": "2.7.1",
            **EXPECTED_DEPENDENCIES,
        },
        "exact_official_environment": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def _download(url: str, destination: Path, budget: int) -> dict[str, Any]:
    if destination.exists():
        raise PreparationError(f"refusing to overwrite staged file: {destination}")
    temporary = destination.with_name(destination.name + ".downloading")
    if temporary.exists():
        raise PreparationError(f"single-writer temporary file already exists: {temporary}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    request = urllib.request.Request(url, headers={"User-Agent": "tdmpc2-q-coupling-prep/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > budget:
                raise ResourceBlocked(f"download exceeds bounded budget: {url}")
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > budget:
                    raise ResourceBlocked(f"download exceeds bounded budget: {url}")
                stream.write(block)
                digest.update(block)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if temporary.exists():
            temporary.unlink()
        raise ResourceBlocked(f"download unavailable: {url}: {exc}") from exc
    os.replace(temporary, destination)
    return {"url": url, "path": str(destination), "size": total, "sha256": digest.hexdigest()}


def _extract_archive(archive: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise PreparationError(f"refusing to overwrite source directory: {destination}")
    destination.mkdir(parents=True)
    members_count = 0
    root_name: str | None = None
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            names = [member.name for member in bundle.getmembers() if member.name]
            if not names:
                raise PreparationError("source archive is empty")
            root_name = PurePosixPath(names[0]).parts[0]
            if not root_name.endswith(SOURCE_COMMIT):
                raise PreparationError(f"source archive root is not pinned commit: {root_name}")
            for member in bundle.getmembers():
                parts = PurePosixPath(member.name).parts
                if len(parts) == 1 and parts[0] == root_name and member.isdir():
                    continue
                if len(parts) < 2 or parts[0] != root_name or ".." in parts:
                    raise PreparationError(f"unsafe source archive member: {member.name}")
                if member.issym() or member.islnk():
                    raise PreparationError(f"linked source archive member is not accepted: {member.name}")
                relative = Path(*parts[1:])
                target = destination / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise PreparationError(f"unsupported source archive member: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise PreparationError(f"duplicate source archive member: {member.name}")
                source = bundle.extractfile(member)
                if source is None:
                    raise PreparationError(f"cannot read source archive member: {member.name}")
                with source, target.open("wb") as stream:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                members_count += 1
    except (tarfile.TarError, OSError) as exc:
        raise ResourceBlocked(f"source archive extraction failed: {exc}") from exc
    return {"root": root_name, "file_count": members_count}


def _source_identity(source: Path) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for relative in SELECTED_SOURCE:
        path = source / relative
        if not path.is_file():
            raise PreparationError(f"pinned source file is missing: {relative}")
        selected[relative] = {"size": path.stat().st_size, "sha256": _sha256(path)}
    texts = {relative: (source / relative).read_text(encoding="utf-8") for relative in SELECTED_SOURCE if relative.endswith((".py", ".yaml"))}
    marker_checks = {
        "world_model_has_q_modes": all(token in texts["tdmpc2/common/world_model.py"] for token in ("return_type", "torch.randperm", "two_hot_inv")),
        "planner_uses_avg": "_estimate_value" in texts["tdmpc2/tdmpc2.py"] and any(token in texts["tdmpc2/tdmpc2.py"] for token in ("return_type='avg'", 'return_type="avg"')),
        "dmcontrol_cartpole_wrapper": all(token in texts["tdmpc2/envs/dmcontrol.py"] for token in ("def make_env", "suite.load", "task_kwargs={'random': cfg.seed}", "_obs_to_array", "range(2)", "max_episode_steps=500")),
        "config_has_num_q": "num_q: 5" in texts["tdmpc2/config.yaml"],
    }
    if not all(marker_checks.values()):
        raise PreparationError(f"pinned source marker check failed: {marker_checks}")
    return {
        "repository": "https://github.com/nicklashansen/tdmpc2",
        "commit": SOURCE_COMMIT,
        "archive_url": SOURCE_URL,
        "selected_files": selected,
        "marker_checks": marker_checks,
    }


def _reset_observations(seeds: tuple[int, ...]) -> tuple[Any, list[dict[str, Any]]]:
    import numpy as np
    from dm_control import suite

    observations = []
    positions = []
    velocities = []
    records = []
    expected_keys = ["position", "velocity"]
    for seed in seeds:
        environment = suite.load(
            "cartpole",
            "balance",
            task_kwargs={"random": int(seed)},
            visualize_reward=False,
        )
        try:
            timestep = environment.reset()
            observation = timestep.observation
            keys = list(observation.keys())
            if keys != expected_keys:
                raise PreparationError(f"unexpected observation keys for seed {seed}: {keys}")
            position = np.asarray(observation["position"], dtype=np.float32).reshape(-1)
            velocity = np.asarray(observation["velocity"], dtype=np.float32).reshape(-1)
            if position.shape != (3,) or velocity.shape != (2,):
                raise PreparationError(f"unexpected Cartpole component shapes: {position.shape}, {velocity.shape}")
            flat = np.concatenate((position, velocity)).astype(np.float32, copy=False)
            if flat.shape != (5,) or not np.isfinite(flat).all():
                raise PreparationError(f"invalid flattened observation for seed {seed}: {flat.shape}")
            positions.append(position)
            velocities.append(velocity)
            observations.append(flat)
            records.append({
                "seed": int(seed),
                "observation_keys": expected_keys,
                "position_shape": [3],
                "velocity_shape": [2],
                "flatten_order": ["cart_position", "pole_zz", "pole_xz", "cart_velocity", "pole_angular_velocity"],
                "observation_shape": [5],
                "env_steps": 0,
                "render_calls": 0,
            })
        finally:
            environment.close()
    return (np.asarray(observations, dtype=np.float32), np.asarray(positions, dtype=np.float32), np.asarray(velocities, dtype=np.float32)), records


def _save_npz(path: Path, arrays: dict[str, Any]) -> None:
    import numpy as np

    if path.exists():
        raise PreparationError(f"refusing to overwrite raw observation file: {path}")
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _main(args: argparse.Namespace, allocation: dict[str, Any]) -> None:
    import numpy as np

    root = Path(args.root).expanduser().resolve()
    assetdir = Path(args.assetdir).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_status(out, {"state": "running", "phase": "guard_verified"})
    runtime = _runtime_identity()
    _write_status(out, {"state": "running", "phase": "runtime_verified", "runtime": runtime})
    if assetdir.exists():
        raise PreparationError(f"isolated asset directory already exists; refusing overwrite: {assetdir}")
    assetdir.mkdir(parents=True)
    lock = assetdir / "PREPARATION.lock"
    lock.write_text(json.dumps({
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "owner": allocation.get("user"),
        "hostname": socket.gethostname().split(".", 1)[0],
        "partition": allocation.get("partition"),
        "created_epoch": int(time.time()),
    }, sort_keys=True) + "\n", encoding="utf-8")
    _write_status(out, {"state": "running", "phase": "asset_lock_created"}, assetdir)

    downloads: dict[str, Any] = {}
    source_archive = assetdir / f"tdmpc2-{SOURCE_COMMIT}.tar.gz"
    downloads["source_archive"] = _download(SOURCE_URL, source_archive, DOWNLOAD_LIMIT)
    source_dir = assetdir / "source"
    extraction = _extract_archive(source_archive, source_dir)
    source_identity = _source_identity(source_dir)
    source_identity.update({"archive": downloads["source_archive"], "extraction": extraction})
    _json_write(assetdir / "source_identity.json", source_identity)
    _write_status(out, {"state": "running", "phase": "source_staged", "source_sha256": downloads["source_archive"]["sha256"]}, assetdir)

    checkpoint = assetdir / "checkpoints" / "cartpole-balance-1.pt"
    downloads["checkpoint"] = _download(CHECKPOINT_URL, checkpoint, DOWNLOAD_LIMIT)
    if downloads["checkpoint"]["size"] != CHECKPOINT_SIZE or downloads["checkpoint"]["sha256"].lower() != CHECKPOINT_SHA256:
        raise PreparationError(
            "checkpoint integrity mismatch: "
            f"size={downloads['checkpoint']['size']} sha256={downloads['checkpoint']['sha256']}"
        )
    checkpoint_identity = {
        "repo": MODEL_REPO,
        "revision": HF_REVISION,
        "path": "dmcontrol/cartpole-balance-1.pt",
        "url": CHECKPOINT_URL,
        "size": downloads["checkpoint"]["size"],
        "sha256": downloads["checkpoint"]["sha256"],
        "expected_lfs_oid": CHECKPOINT_SHA256,
        "expected_size": CHECKPOINT_SIZE,
        "torch_loaded": False,
    }
    _json_write(assetdir / "checkpoint_identity.json", checkpoint_identity)
    _write_status(out, {"state": "running", "phase": "checkpoint_staged", "checkpoint_sha256": CHECKPOINT_SHA256}, assetdir)

    _write_status(out, {"state": "running", "phase": "reset_started"}, assetdir)
    (observations, positions, velocities), records = _reset_observations(SEEDS)
    npz_path = assetdir / "observations.npz"
    metadata = {
        "schema": RAW_SCHEMA,
        "task": TASK,
        "seeds": list(SEEDS),
        "observation_keys": ["position", "velocity"],
        "flatten_order": ["cart_position", "pole_zz", "pole_xz", "cart_velocity", "pole_angular_velocity"],
        "observation_shape": [len(SEEDS), 5],
        "source_commit": SOURCE_COMMIT,
        "checkpoint_revision": HF_REVISION,
        "env_steps": 0,
        "render_calls": 0,
        "model_loaded": False,
    }
    _save_npz(npz_path, {
        "observations": observations,
        "position": positions,
        "velocity": velocities,
        "seeds": np.asarray(SEEDS, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    })
    observation_record = {
        "path": npz_path.name,
        "size": npz_path.stat().st_size,
        "sha256": _sha256(npz_path),
        "shape": list(observations.shape),
        "dtype": str(observations.dtype),
        "records": records,
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "raw_schema": RAW_SCHEMA,
        "asset_root": str(assetdir),
        "task": TASK,
        "seed_order": list(SEEDS),
        "source": source_identity,
        "checkpoint": checkpoint_identity,
        "runtime": runtime,
        "config": {
            "resolved_task": TASK,
            "obs": "state",
            "num_q": 5,
            "model_size": 5,
            "compile": False,
            "planner_primary_return_type": "avg",
            "target_return_type": "min",
        },
        "dmcontrol": {
            "domain": "cartpole",
            "task": "balance",
            "observation_keys": ["position", "velocity"],
            "flatten_order": ["cart_position", "pole_zz", "pole_xz", "cart_velocity", "pole_angular_velocity"],
            "observation_shape": [5],
            "action_dim": 1,
            "action_repeat": 2,
            "outer_max_episode_steps": 500,
            "reset_calls": len(SEEDS),
            "env_steps": 0,
            "render_calls": 0,
        },
        "observations": observation_record,
        "download_budget_bytes": DOWNLOAD_LIMIT,
        "downloaded_bytes": sum(item["size"] for item in downloads.values()),
        "inference_ran": False,
        "full_rollout": False,
    }
    _json_write(assetdir / "runtime_identity.json", runtime)
    _json_write(assetdir / "manifest.json", manifest)
    _json_write(out / "manifest.json", manifest)
    _json_write(out / "source_identity.json", source_identity)
    _json_write(out / "checkpoint_identity.json", checkpoint_identity)
    _json_write(out / "runtime_identity.json", runtime)
    _write_status(out, {
        "state": "complete",
        "phase": "complete",
        "samples": len(SEEDS),
        "manifest": str(assetdir / "manifest.json"),
        "observation_sha256": observation_record["sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_commit": SOURCE_COMMIT,
        "model_loaded": False,
        "env_steps": 0,
        "render_calls": 0,
    }, assetdir)
    print(json.dumps({"state": "complete", "manifest": str(assetdir / "manifest.json"), "samples": len(SEEDS)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT_DEFAULT))
    parser.add_argument("--assetdir", default=str(ROOT_DEFAULT / "tdmpc2_q_coupling"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    allocation = _allocation()
    try:
        _main(args, allocation)
    except ResourceBlocked as exc:
        assetdir = Path(args.assetdir).expanduser().resolve()
        _write_status(Path(args.output).expanduser().resolve(), {"state": "resource_blocked", "phase": "resource_blocked", "error": str(exc)}, assetdir if assetdir.exists() else None)
        raise SystemExit(4)
    except Exception as exc:
        assetdir = Path(args.assetdir).expanduser().resolve()
        _write_status(Path(args.output).expanduser().resolve(), {"state": "failed", "phase": "failed", "error": f"{type(exc).__name__}: {exc}"}, assetdir if assetdir.exists() else None)
        raise


if __name__ == "__main__":
    main()
