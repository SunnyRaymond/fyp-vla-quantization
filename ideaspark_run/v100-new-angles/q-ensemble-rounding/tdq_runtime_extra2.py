"""Resolve a reuse-aware dependency overlay and verify its imports."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


TOP_LEVEL = (
    "dm-control==1.0.16",
    "mujoco==3.1.2",
    "tensordict==0.7.2",
    "omegaconf==2.3.0",
)
CONSTRAINT_KEYS = (
    "torch", "torchvision", "torchaudio", "triton", "numpy", "scipy", "protobuf",
    "packaging", "pyparsing", "setuptools", "gymnasium", "tensordict", "omegaconf",
)
FORBIDDEN_PREFIXES = ("torch", "torchvision", "torchaudio", "triton", "nvidia-", "cuda-", "cudnn", "cublas")
DOWNLOAD_LIMIT = 500 * 1024 * 1024


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _versions() -> dict[str, str]:
    """Return all visible distributions, preserving the first sys.path copy."""
    result: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if isinstance(name, str) and isinstance(version, str):
            result.setdefault(_canon(name), version)
    return dict(sorted(result.items()))


def _constraints(path: Path, installed: dict[str, str]) -> list[str]:
    lines = [f"{name}=={installed[name]}" for name in CONSTRAINT_KEYS if name in installed]
    path.write_text("# Reuse constraints generated from visible venv/vendor metadata.\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return lines


def _head_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "tdmpc2-q-coupling-runtime/2"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = response.headers.get("Content-Length")
            if value is None:
                raise RuntimeError(f"no Content-Length for planned download: {url}")
            return int(value)
    except urllib.error.HTTPError as exc:
        if exc.code not in {405, 501}:
            raise RuntimeError(f"cannot inspect planned download size: {url}: {exc}") from exc
        request = urllib.request.Request(url, method="GET", headers={"Range": "bytes=0-0", "User-Agent": "tdmpc2-q-coupling-runtime/2"})
        with urllib.request.urlopen(request, timeout=30) as response:
            content_range = response.headers.get("Content-Range", "")
            match = re.search(r"/([0-9]+)$", content_range)
            if match:
                return int(match.group(1))
            value = response.headers.get("Content-Length")
            if value is None:
                raise RuntimeError(f"no bounded size metadata for planned download: {url}")
            return int(value)


def _dry_run(python: str, constraints: Path, report: Path, output: Path) -> dict[str, Any]:
    command = [
        python, "-m", "pip", "install", "--dry-run", "--report", str(report),
        "--constraint", str(constraints), "--index-url", "https://pypi.org/simple",
        "--disable-pip-version-check", *TOP_LEVEL,
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (output / "pip_dry_run.log").write_text(completed.stdout[-60000:], encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"pip dry-run failed with exit {completed.returncode}")
    try:
        return json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"pip dry-run report is invalid: {report}") from exc


def _make_plan(report: dict[str, Any], installed: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    entries = report.get("install")
    if not isinstance(entries, list):
        raise RuntimeError("pip report has no install list")
    planned: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    total = 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("metadata"), dict):
            raise RuntimeError("pip report contains malformed install item")
        metadata = entry["metadata"]
        name, version = metadata.get("name"), metadata.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise RuntimeError("pip report item lacks package name/version")
        canonical = _canon(name)
        if installed.get(canonical) == version:
            reused.append({"name": name, "version": version, "canonical": canonical})
            continue
        if any(canonical == prefix or canonical.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            raise RuntimeError(f"resolver requires a new forbidden package: {name}=={version}")
        download = entry.get("download_info")
        if not isinstance(download, dict) or not isinstance(download.get("url"), str):
            raise RuntimeError(f"new package lacks a downloadable URL: {name}=={version}")
        size = _head_size(download["url"])
        total += size
        planned.append({"name": name, "version": version, "canonical": canonical, "url": download["url"], "size": size})
    if total > DOWNLOAD_LIMIT:
        raise RuntimeError(f"planned new downloads exceed 500 MiB: {total}")
    return planned, reused, total


def _install(python: str, extra: Path, planned: list[dict[str, Any]], output: Path) -> None:
    if not planned:
        return
    requirements = [f"{item['name']}=={item['version']}" for item in planned]
    command = [
        python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--no-cache-dir",
        "--disable-pip-version-check", "--target", str(extra), *requirements,
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (output / "pip_install.log").write_text(completed.stdout[-60000:], encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"pip install failed with exit {completed.returncode}")


def _verify(extra: Path) -> dict[str, Any]:
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
    from dm_control import suite
    import mujoco
    import tensordict
    from tensordict.nn import TensorDictParams
    import omegaconf
    from omegaconf import OmegaConf

    versions = _versions()
    expected = {"dm-control": "1.0.16", "mujoco": "3.1.2", "tensordict": "0.7.2", "omegaconf": "2.3.0"}
    if any(versions.get(name) != version for name, version in expected.items()):
        raise RuntimeError(f"post-repair versions do not match: {versions}")
    return {
        "state": "complete",
        "versions": {name: versions.get(name) for name in ("dm-control", "mujoco", "tensordict", "omegaconf", "torch", "gymnasium")},
        "imports": {
            "dm_control_suite": suite.__name__, "mujoco": mujoco.__name__, "tensordict": tensordict.__name__,
            "TensorDictParams": TensorDictParams.__name__, "omegaconf": omegaconf.__name__, "OmegaConf": OmegaConf.__name__,
        },
        "model_loaded": False,
        "env_created": False,
        "extra": str(extra),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--extra", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from allocation_guard import require_allocation

    allocation = require_allocation()
    output, extra, vendor = (Path(value).expanduser().resolve() for value in (args.output, args.extra, args.vendor))
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": "tdmpc2-q-coupling-runtime-extra2-v1",
        "state": "running",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname().split(".", 1)[0],
        "allocation": {key: value for key, value in allocation.items() if key != "user"},
        "python": sys.version,
        "python_executable": args.python,
        "vendor": str(vendor),
        "extra": str(extra),
        "top_level_requirements": list(TOP_LEVEL),
        "download_limit_bytes": DOWNLOAD_LIMIT,
        "updated_epoch": int(time.time()),
    }
    try:
        installed_before = _versions()
        constraints_path = output / "reuse_constraints.txt"
        result["constraint_lines"] = _constraints(constraints_path, installed_before)
        report_path = output / "pip_dry_run_report.json"
        report = _dry_run(args.python, constraints_path, report_path, output)
        planned, reused, total = _make_plan(report, installed_before)
        result["installed_distributions_before"] = installed_before
        result["reused_packages"] = reused
        result["planned_packages"] = planned
        result["planned_download_bytes"] = total
        result["forbidden_new_packages"] = []
        _write(output / "dependency_plan.json", result)
        _install(args.python, extra, planned, output)
        result.update(_verify(extra))
        result["installed_distributions_after"] = _versions()
        _write(output / "runtime_extra2.json", result)
        _write(extra / "runtime_extra2.json", result)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        result["state"] = "resource_blocked"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _write(output / "runtime_extra2.json", result)
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(4)


if __name__ == "__main__":
    main()
