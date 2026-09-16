"""Resolve and install one bounded dependency overlay, then run preparation."""

from __future__ import annotations

import argparse
import hashlib
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
FORBIDDEN_PREFIXES = ("torch", "torchvision", "torchaudio", "triton", "nvidia-", "cuda-", "cudnn", "cublas")
DOWNLOAD_LIMIT = 500 * 1024 * 1024


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _versions() -> dict[str, str | None]:
    names = ("dm-control", "mujoco", "tensordict", "omegaconf", "torch", "gymnasium")
    return {name: next((dist.version for dist in importlib.metadata.distributions() if _canon(dist.metadata.get("Name", "")) == _canon(name)), None) for name in names}


def _head_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "tdmpc2-q-coupling-runtime/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = response.headers.get("Content-Length")
            if value is None:
                raise RuntimeError(f"no Content-Length for planned download: {url}")
            return int(value)
    except urllib.error.HTTPError as exc:
        if exc.code not in {405, 501}:
            raise RuntimeError(f"cannot inspect planned download size: {url}: {exc}") from exc
        request = urllib.request.Request(url, method="GET", headers={"Range": "bytes=0-0", "User-Agent": "tdmpc2-q-coupling-runtime/1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            content_range = response.headers.get("Content-Range", "")
            match = re.search(r"/([0-9]+)$", content_range)
            if match:
                return int(match.group(1))
            value = response.headers.get("Content-Length")
            if value is None:
                raise RuntimeError(f"no bounded size metadata for planned download: {url}")
            return int(value)


def _run_report(python: str, extra: Path, report: Path) -> dict[str, Any]:
    command = [
        python, "-m", "pip", "install", "--dry-run", "--report", str(report),
        "--target", str(extra), "--only-binary=:all:", "--disable-pip-version-check",
        *TOP_LEVEL,
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (report.parent / "pip_dry_run.log").write_text(completed.stdout[-60000:], encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"pip dry-run failed with exit {completed.returncode}")
    try:
        return json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"pip dry-run did not produce a valid report: {report}") from exc


def _plan(report: dict[str, Any], existing: dict[str, str | None]) -> tuple[list[dict[str, Any]], int]:
    items = report.get("install")
    if not isinstance(items, list):
        raise RuntimeError("pip report has no install list")
    planned: list[dict[str, Any]] = []
    total = 0
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
            raise RuntimeError("pip report contains malformed install item")
        metadata = item["metadata"]
        name, version = metadata.get("name"), metadata.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise RuntimeError("pip report item lacks package name/version")
        canonical = _canon(name)
        if any(canonical == prefix or canonical.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            raise RuntimeError(f"resolver plan includes forbidden heavyweight package: {name}=={version}")
        if existing.get(canonical) == version:
            continue
        download = item.get("download_info")
        if not isinstance(download, dict) or not isinstance(download.get("url"), str):
            raise RuntimeError(f"new package lacks a downloadable wheel URL: {name}=={version}")
        size = _head_size(download["url"])
        total += size
        planned.append({"name": name, "version": version, "canonical": canonical, "url": download["url"], "size": size})
    if total > DOWNLOAD_LIMIT:
        raise RuntimeError(f"planned download exceeds 500 MiB: {total}")
    return planned, total


def _install(python: str, extra: Path, planned: list[dict[str, Any]]) -> None:
    if not planned:
        return
    requirements = [f"{item['name']}=={item['version']}" for item in planned]
    command = [python, "-m", "pip", "install", "--no-deps", "--no-cache-dir", "--only-binary=:all:", "--disable-pip-version-check", "--target", str(extra), *requirements]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (extra.parent / "pip_install.log").write_text(completed.stdout[-60000:], encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"pip install failed with exit {completed.returncode}")


def _verify(extra: Path) -> dict[str, Any]:
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
        "versions": versions,
        "imports": {
            "dm_control_suite": suite.__name__,
            "mujoco": mujoco.__name__,
            "tensordict": tensordict.__name__,
            "TensorDictParams": TensorDictParams.__name__,
            "omegaconf": omegaconf.__name__,
            "OmegaConf": OmegaConf.__name__,
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
    parser.add_argument("--control", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    from allocation_guard import require_allocation

    allocation = require_allocation()
    output, extra, vendor = (Path(value).expanduser().resolve() for value in (args.output, args.extra, args.vendor))
    output.mkdir(parents=True, exist_ok=True)
    if extra.exists():
        raise RuntimeError(f"refusing to overwrite existing extra overlay: {extra}")
    extra.mkdir(parents=True)
    result: dict[str, Any] = {
        "schema": "tdmpc2-q-coupling-runtime-extra-v1",
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
    report_path = output / "pip_dry_run_report.json"
    try:
        existing = {_canon(key): value for key, value in _versions().items()}
        report = _run_report(args.python, extra, report_path)
        planned, total = _plan(report, existing)
        result["existing_versions_before"] = existing
        result["planned_packages"] = planned
        result["planned_download_bytes"] = total
        result["forbidden_packages"] = []
        _write(output / "dependency_plan.json", result)
        _install(args.python, extra, planned)
        result.update(_verify(extra))
        result["existing_versions_after"] = {_canon(key): value for key, value in _versions().items()}
        _write(output / "runtime_extra.json", result)
        _write(extra / "runtime_extra.json", result)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        result["state"] = "resource_blocked"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _write(output / "runtime_extra.json", result)
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(4)


if __name__ == "__main__":
    main()
