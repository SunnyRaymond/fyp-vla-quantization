"""Install one compatibility NumPy overlay after a bounded plan check."""

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
import urllib.request


REQUIREMENT = "numpy==1.26.4"
LIMIT = 500 * 1024 * 1024


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name, version = distribution.metadata.get("Name"), distribution.version
        if isinstance(name, str) and isinstance(version, str):
            result.setdefault(_canon(name), version)
    return dict(sorted(result.items()))


def _size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "tdmpc2-q-coupling-numpy-fix/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        value = response.headers.get("Content-Length")
        if value is None:
            raise RuntimeError(f"planned NumPy wheel has no Content-Length: {url}")
        return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--extra", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from allocation_guard import require_allocation

    allocation = require_allocation()
    extra, output = (Path(value).expanduser().resolve() for value in (args.extra, args.output))
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": "tdmpc2-q-coupling-numpy-fix-v1",
        "state": "running",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname().split(".", 1)[0],
        "allocation": {key: value for key, value in allocation.items() if key != "user"},
        "python": sys.version,
        "python_executable": args.python,
        "extra": str(extra),
        "requirement": REQUIREMENT,
        "download_limit_bytes": LIMIT,
        "versions_before": _versions(),
        "updated_epoch": int(time.time()),
    }
    report = output / "numpy_dry_run_report.json"
    try:
        if extra.exists():
            raise RuntimeError(f"refusing to overwrite existing extra3 directory: {extra}")
        extra.mkdir(parents=True)
        command = [args.python, "-m", "pip", "install", "--dry-run", "--report", str(report), "--no-deps", "--only-binary=:all:", "--disable-pip-version-check", "--index-url", "https://pypi.org/simple", REQUIREMENT]
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (output / "numpy_dry_run.log").write_text(completed.stdout[-60000:], encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"NumPy dry-run failed with exit {completed.returncode}")
        payload = json.loads(report.read_text(encoding="utf-8"))
        entries = payload.get("install")
        if not isinstance(entries, list) or len(entries) != 1:
            raise RuntimeError(f"NumPy dry-run returned unexpected install plan: {entries}")
        entry = entries[0]
        metadata = entry.get("metadata", {})
        if _canon(str(metadata.get("name"))) != "numpy" or metadata.get("version") != "1.26.4":
            raise RuntimeError(f"NumPy plan is not exact: {metadata}")
        download = entry.get("download_info", {})
        url = download.get("url")
        if not isinstance(url, str):
            raise RuntimeError("NumPy plan has no wheel URL")
        wheel_size = _size(url)
        if wheel_size > LIMIT:
            raise RuntimeError(f"NumPy wheel exceeds bounded download limit: {wheel_size}")
        result["plan"] = {"name": "numpy", "version": "1.26.4", "url": url, "size": wheel_size}
        result["planned_download_bytes"] = wheel_size
        _write(output / "numpy_plan.json", result)
        install = [args.python, "-m", "pip", "install", "--no-deps", "--no-cache-dir", "--disable-pip-version-check", "--target", str(extra), REQUIREMENT]
        installed = subprocess.run(install, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (output / "numpy_install.log").write_text(installed.stdout[-60000:], encoding="utf-8")
        if installed.returncode != 0:
            raise RuntimeError(f"NumPy install failed with exit {installed.returncode}")
        sys.path.insert(0, str(extra))
        import numpy
        if numpy.__version__ != "1.26.4":
            raise RuntimeError(f"NumPy overlay import version mismatch: {numpy.__version__}")
        result.update({"state": "complete", "versions_after": _versions(), "numpy_import": numpy.__version__})
        _write(output / "numpy_fix.json", result)
        _write(extra / "numpy_fix.json", result)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        result["state"] = "resource_blocked"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _write(output / "numpy_fix.json", result)
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(4)


if __name__ == "__main__":
    main()
