"""Verify the completed extra2 overlay in-process before fresh preparation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
from typing import Any


EXPECTED = {"dm-control": "1.0.16", "mujoco": "3.1.2", "tensordict": "0.7.2", "omegaconf": "2.3.0"}


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name, version = distribution.metadata.get("Name"), distribution.version
        if isinstance(name, str) and isinstance(version, str):
            result.setdefault(_canon(name), version)
    return dict(sorted(result.items()))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra", required=True)
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from allocation_guard import require_allocation

    allocation = require_allocation()
    extra, vendor, control, output = (Path(value).expanduser().resolve() for value in (args.extra, args.vendor, args.control, args.output))
    result: dict[str, Any] = {
        "schema": "tdmpc2-q-coupling-runtime-extra2-resume-v1",
        "state": "running",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname().split(".", 1)[0],
        "allocation": {key: value for key, value in allocation.items() if key != "user"},
        "python": sys.version,
        "python_executable": sys.executable,
        "extra": str(extra),
        "vendor": str(vendor),
        "control": str(control),
        "updated_epoch": int(time.time()),
    }
    try:
        if not extra.is_dir() or not vendor.is_dir():
            raise RuntimeError(f"required existing overlay/vendor is missing: {extra} {vendor}")
        sys.path[:0] = [str(extra), str(vendor), str(control)]
        from dm_control import suite
        import mujoco
        import tensordict
        from tensordict.nn import TensorDictParams
        import omegaconf
        from omegaconf import OmegaConf

        versions = _versions()
        if any(versions.get(name) != expected for name, expected in EXPECTED.items()):
            raise RuntimeError(f"overlay import versions do not match: {versions}")
        result.update({
            "state": "complete",
            "versions": {name: versions.get(name) for name in ("dm-control", "mujoco", "tensordict", "omegaconf", "torch", "gymnasium")},
            "installed_distributions": versions,
            "imports": {
                "dm_control_suite": suite.__name__, "mujoco": mujoco.__name__, "tensordict": tensordict.__name__,
                "TensorDictParams": TensorDictParams.__name__, "omegaconf": omegaconf.__name__, "OmegaConf": OmegaConf.__name__,
            },
            "model_loaded": False,
            "env_created": False,
        })
        _write(output / "runtime_extra2_resume.json", result)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        result["state"] = "resource_blocked"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _write(output / "runtime_extra2_resume.json", result)
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(4)


if __name__ == "__main__":
    main()
