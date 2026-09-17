"""Verify one isolated CPU runtime repair without loading a model or env."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any


EXPECTED = {"dm-control": "1.0.16", "mujoco": "3.1.2"}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from allocation_guard import require_allocation

    allocation = require_allocation()
    output = Path(args.output).expanduser().resolve()
    vendor = Path(args.vendor).expanduser().resolve()
    result: dict[str, Any] = {
        "schema": "tdmpc2-q-coupling-runtime-repair-v1",
        "state": "running",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname().split(".", 1)[0],
        "allocation": {key: value for key, value in allocation.items() if key != "user"},
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "vendor": str(vendor),
        "expected": EXPECTED,
        "updated_epoch": int(time.time()),
    }
    try:
        if not vendor.is_dir():
            raise RuntimeError(f"isolated vendor directory is missing: {vendor}")
        versions = {name: _version(name) for name in EXPECTED}
        result["versions"] = versions
        if any(versions[name] != expected for name, expected in EXPECTED.items()):
            raise RuntimeError(f"repair package identity mismatch: {versions}")
        from dm_control import suite
        import mujoco

        result["imports"] = {"dm_control_suite": suite.__name__, "mujoco": mujoco.__name__}
        result["model_loaded"] = False
        result["env_created"] = False
        result["state"] = "complete"
        _write(output / "runtime_repair.json", result)
        _write(vendor / "runtime_repair.json", result)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        result["state"] = "resource_blocked"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _write(output / "runtime_repair.json", result)
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(4)


if __name__ == "__main__":
    main()
