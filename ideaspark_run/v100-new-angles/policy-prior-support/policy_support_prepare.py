"""Prepare reset-only Cartpole state observations for the policy-prior support.

The parent TD-MPC2 manifest is an explicit input.  This preparation records its
small-manifest hash and identity references, but never hashes or loads the
checkpoint/source payload.  It only calls DMControl ``reset``; no model,
checkpoint, ``step`` or ``render`` operation is allowed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any


TOP_DEFAULT = Path("/tc1home/UG/yguo017/v100_newangles_ccds")
SEEDS = tuple(range(5217, 5225))
TASK = "cartpole-balance"
RAW_SCHEMA = "tdmpc2-cartpole-reset-input-v1"
MANIFEST_SCHEMA = "policy-prior-support-preparation-v1"
EXPECTED_PARENT_MANIFEST_SHA256 = (
    "9240979105b919f6051f27261d00ff33652105f962d039728638caabb042d369"
)
CONTROL_MAX_BYTES = 65536
EXPECTED_PACKAGES = {
    "numpy": "1.26.4",
    "dm-control": "1.0.16",
    "mujoco": "3.1.2",
}
FLATTEN_ORDER = [
    "cart_position",
    "pole_zz",
    "pole_xz",
    "cart_velocity",
    "pole_angular_velocity",
]


class PreparationError(RuntimeError):
    """A fail-closed preparation or identity error."""


class ResourceBlocked(PreparationError):
    """A required allocation, runtime, or parent asset is unavailable."""


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


def _ensure_small(path: Path) -> None:
    if path.stat().st_size > CONTROL_MAX_BYTES:
        raise PreparationError(f"control artifact exceeds 64 KiB: {path}")


def _status(out: Path, value: dict[str, Any], assetdir: Path | None = None) -> None:
    payload = {
        "schema": "policy-prior-support-preparation-status-v1",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname().split(".", 1)[0],
        "updated_epoch": int(time.time()),
        **value,
    }
    _json_write(out / "status.json", payload)
    _ensure_small(out / "status.json")
    if assetdir is not None and assetdir.exists():
        _json_write(assetdir / "status.json", payload)
        _ensure_small(assetdir / "status.json")


def _allocation() -> dict[str, Any]:
    # The shell calls this guard before any heavy operation; repeat it here so
    # direct execution also fails closed instead of creating an asset locally.
    from allocation_guard import require_allocation

    return require_allocation()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_identity() -> dict[str, Any]:
    import numpy as np

    try:
        import dm_control
        import mujoco
    except Exception as exc:
        raise ResourceBlocked(f"DMControl runtime imports unavailable: {exc}") from exc
    packages = {name: _package_version(name) for name in EXPECTED_PACKAGES}
    mismatches = {
        name: {"expected": expected, "actual": packages.get(name)}
        for name, expected in EXPECTED_PACKAGES.items()
        if packages.get(name) != expected
    }
    if mismatches or np.__version__ != EXPECTED_PACKAGES["numpy"]:
        raise ResourceBlocked(f"CPU preparation runtime mismatch: {mismatches}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in {"", "-1"}:
        raise PreparationError("CPU preparation has a visible CUDA device")
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "imports": {
            "numpy": np.__name__,
            "dm_control": dm_control.__name__,
            "mujoco": mujoco.__name__,
        },
        "python_path_head": sys.path[:5],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "model_loaded": False,
        "checkpoint_loaded": False,
    }


def _parent_reference(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResourceBlocked(f"explicit TDQ parent manifest is missing: {path}")
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ResourceBlocked("parent manifest exceeds the small-manifest bound")
    digest = _sha256(path)
    try:
        parent = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceBlocked(f"cannot read parent manifest: {exc}") from exc
    if not isinstance(parent, dict):
        raise ResourceBlocked("parent manifest must be a JSON object")
    checkpoint = parent.get("checkpoint")
    source = parent.get("source")
    if not isinstance(checkpoint, dict) or not isinstance(source, dict):
        raise ResourceBlocked("parent manifest lacks source/checkpoint identity")
    checkpoint_path = str(checkpoint.get("path", ""))
    if "cartpole-balance-1.pt" in checkpoint_path:
        raise ResourceBlocked("explicit parent points to rejected checkpoint seed 1")
    if "cartpole-balance-" not in checkpoint_path or not checkpoint_path.endswith(".pt"):
        raise ResourceBlocked(f"parent checkpoint is not an explicit Cartpole candidate: {checkpoint_path}")
    source_commit = source.get("commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise ResourceBlocked("parent source commit is missing")
    parent_sha_expected = os.environ.get("TDQ_PARENT_MANIFEST_SHA256", "").strip().lower()
    if parent_sha_expected and parent_sha_expected != digest:
        raise ResourceBlocked(
            f"TDQ parent manifest hash mismatch: expected {parent_sha_expected}, actual {digest}"
        )
    if digest != EXPECTED_PARENT_MANIFEST_SHA256:
        raise ResourceBlocked(
            "policy-prior-support requires the frozen TDQ seed3 parent manifest: "
            f"expected {EXPECTED_PARENT_MANIFEST_SHA256}, actual {digest}"
        )
    reference = {
        "path": str(path),
        "sha256": digest,
        "schema": parent.get("schema"),
        "asset_root": parent.get("asset_root"),
        "checkpoint": {
            "path": checkpoint.get("path"),
            "revision": checkpoint.get("revision"),
            "sha256": checkpoint.get("sha256"),
            "size": checkpoint.get("size"),
            "compatibility_cpu_job": checkpoint.get("compatibility_cpu_job"),
        },
        "source": {
            "repository": source.get("repository"),
            "commit": source_commit,
            "archive_sha256": (source.get("archive") or {}).get("sha256")
            if isinstance(source.get("archive"), dict)
            else None,
        },
    }
    return reference


def _reset_observations(seeds: tuple[int, ...]) -> tuple[Any, Any, Any, list[dict[str, Any]]]:
    import numpy as np
    from dm_control import suite

    observations: list[Any] = []
    positions: list[Any] = []
    velocities: list[Any] = []
    records: list[dict[str, Any]] = []
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
                raise PreparationError(
                    f"unexpected Cartpole component shapes: {position.shape}, {velocity.shape}"
                )
            flat = np.concatenate((position, velocity)).astype(np.float32, copy=False)
            if flat.shape != (5,) or not np.isfinite(flat).all():
                raise PreparationError(f"invalid flattened observation for seed {seed}: {flat.shape}")
            positions.append(position)
            velocities.append(velocity)
            observations.append(flat)
            records.append(
                {
                    "seed": int(seed),
                    "observation_keys": expected_keys,
                    "position_shape": [3],
                    "velocity_shape": [2],
                    "flatten_order": FLATTEN_ORDER,
                    "observation_shape": [5],
                    "env_steps": 0,
                    "render_calls": 0,
                }
            )
        finally:
            environment.close()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(positions, dtype=np.float32),
        np.asarray(velocities, dtype=np.float32),
        records,
    )


def _save_npz(path: Path, arrays: dict[str, Any]) -> None:
    import numpy as np

    if path.exists():
        raise PreparationError(f"refusing to overwrite observations: {path}")
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _protocol_reference(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResourceBlocked(f"frozen policy-support protocol alias is missing: {path}")
    size = path.stat().st_size
    if size > 65536:
        raise ResourceBlocked(f"protocol alias exceeds controller bound: {path}")
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": size}


def _run(args: argparse.Namespace, allocation: dict[str, Any]) -> None:
    import numpy as np

    root = Path(args.root).expanduser().resolve()
    assetdir = Path(args.assetdir).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    parent_path = Path(args.parent_manifest).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    if assetdir != root and root not in assetdir.parents:
        raise PreparationError(f"policy-support asset must remain under preparation root: {assetdir}")
    if assetdir.exists():
        raise PreparationError(f"isolated policy-support asset directory already exists: {assetdir}")
    out.mkdir(parents=True, exist_ok=True)
    _status(out, {"state": "running", "phase": "guard_verified"})
    parent_ref = _parent_reference(parent_path)
    protocol_ref = _protocol_reference(protocol_path)
    runtime = _runtime_identity()
    _status(
        out,
        {
            "state": "running",
            "phase": "runtime_verified",
            "runtime": runtime,
            "parent_manifest_sha256": parent_ref["sha256"],
            "protocol_sha256": protocol_ref["sha256"],
        },
    )
    assetdir.mkdir(parents=True)
    _json_write(
        assetdir / "PREPARATION.lock",
        {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "owner": allocation.get("user"),
            "hostname": socket.gethostname().split(".", 1)[0],
            "partition": allocation.get("partition"),
            "created_epoch": int(time.time()),
        },
    )
    _status(out, {"state": "running", "phase": "asset_lock_created"}, assetdir)
    _status(out, {"state": "running", "phase": "reset_started"}, assetdir)
    observations, positions, velocities, records = _reset_observations(SEEDS)
    metadata = {
        "schema": RAW_SCHEMA,
        "schema_version": 1,
        "task": TASK,
        "seeds": list(SEEDS),
        "observation_keys": ["position", "velocity"],
        "flatten_order": FLATTEN_ORDER,
        "observation_shape": [len(SEEDS), 5],
        "parent_manifest_sha256": parent_ref["sha256"],
        "protocol_sha256": protocol_ref["sha256"],
        "source_commit": parent_ref["source"]["commit"],
        "env_steps": 0,
        "render_calls": 0,
        "model_loaded": False,
        "checkpoint_loaded": False,
    }
    npz_path = assetdir / "observations.npz"
    _save_npz(
        npz_path,
        {
            "observations": observations,
            "position": positions,
            "velocity": velocities,
            "seeds": np.asarray(SEEDS, dtype=np.int64),
            "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        },
    )
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
        "schema_version": 1,
        "raw_schema": RAW_SCHEMA,
        "asset_root": str(assetdir),
        "task": TASK,
        "seed_order": list(SEEDS),
        "parent_manifest": parent_ref,
        "protocol": protocol_ref,
        "source": parent_ref["source"],
        "runtime": runtime,
        "allocation": allocation,
        "config": {
            "resolved_task": TASK,
            "obs": "state",
            "model_loaded": False,
            "checkpoint_loaded": False,
        },
        "dmcontrol": {
            "domain": "cartpole",
            "task": "balance",
            "observation_keys": ["position", "velocity"],
            "flatten_order": FLATTEN_ORDER,
            "observation_shape": [5],
            "reset_calls": len(SEEDS),
            "env_steps": 0,
            "render_calls": 0,
        },
        "observations": observation_record,
        "inference_ran": False,
        "full_rollout": False,
    }
    _json_write(assetdir / "runtime_identity.json", runtime)
    _json_write(assetdir / "manifest.json", manifest)
    _json_write(out / "manifest.json", manifest)
    _ensure_small(assetdir / "manifest.json")
    _ensure_small(out / "manifest.json")
    _status(
        out,
        {
            "state": "complete",
            "phase": "complete",
            "samples": len(SEEDS),
            "manifest": str(assetdir / "manifest.json"),
            "observation_sha256": observation_record["sha256"],
            "parent_manifest_sha256": parent_ref["sha256"],
            "protocol_sha256": protocol_ref["sha256"],
            "source_commit": parent_ref["source"]["commit"],
            "model_loaded": False,
            "checkpoint_loaded": False,
            "env_steps": 0,
            "render_calls": 0,
        },
        assetdir,
    )
    print(json.dumps({"state": "complete", "manifest": str(assetdir / "manifest.json"), "samples": len(SEEDS)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(TOP_DEFAULT))
    parser.add_argument("--assetdir", default=str(TOP_DEFAULT / "policy_prior_support_ready"))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--parent-manifest",
        default=os.environ.get("TDQ_PARENT_MANIFEST", ""),
        help="Explicit frozen compatible TDQ parent manifest.",
    )
    parser.add_argument(
        "--protocol",
        default=os.environ.get("POLICY_SUPPORT_PROTOCOL", ""),
        help="Copied policy_support_protocol.zh.md alias to hash and record.",
    )
    args = parser.parse_args()
    if not args.parent_manifest:
        raise SystemExit("TDQ_PARENT_MANIFEST or --parent-manifest is required")
    if not args.protocol:
        raise SystemExit("POLICY_SUPPORT_PROTOCOL or --protocol is required")
    allocation = _allocation()
    try:
        _run(args, allocation)
    except ResourceBlocked as exc:
        out = Path(args.output).expanduser().resolve()
        assetdir = Path(args.assetdir).expanduser().resolve()
        _status(out, {"state": "resource_blocked", "phase": "resource_blocked", "error": str(exc)}, assetdir if assetdir.exists() else None)
        raise SystemExit(4)
    except Exception as exc:
        out = Path(args.output).expanduser().resolve()
        assetdir = Path(args.assetdir).expanduser().resolve()
        _status(out, {"state": "failed", "phase": "failed", "error": f"{type(exc).__name__}: {exc}"}, assetdir if assetdir.exists() else None)
        raise


if __name__ == "__main__":
    main()

