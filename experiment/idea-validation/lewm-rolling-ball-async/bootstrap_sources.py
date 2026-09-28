"""Download pinned training source and task-local video decoder in a CPU allocation."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

from prepare_cpu import allocation_guard

ROOT = Path("/scratch/users/ntu/yguo017/lewm-rolling-ball-async")
SOURCES = {
    "upstream_lewm": ("lucas-maes/le-wm", "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"),
    "upstream_stablewm": ("galilai-group/stable-worldmodel", "10c26dbd5677083fa31dba69eb738b973845e9a4"),
}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "rolling-lewm-source-prep"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    job, host = allocation_guard()
    result = {"job_id": job, "host": host, "status": "RUNNING", "sources": {}}
    out = ROOT / "runs" / job
    try:
        for directory, (repo, revision) in SOURCES.items():
            tree = json.loads(fetch("https://api.github.com/repos/" + repo + "/git/trees/"
                                    + revision + "?recursive=1"))
            if tree.get("truncated"):
                raise RuntimeError("Incomplete source tree for " + repo)
            paths = [x["path"] for x in tree["tree"] if x["type"] == "blob" and (
                (x["path"].endswith((".py", ".yaml", ".yml")) and
                 (directory == "upstream_lewm" or x["path"].startswith("stable_worldmodel/"))) or
                x["path"] in ("pyproject.toml", "README.md", "LICENSE"))]
            if not 1 <= len(paths) <= 160:
                raise RuntimeError("Unexpected source selection size for " + repo)
            for relative in paths:
                target = ROOT / directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".partial")
                partial.write_bytes(fetch("https://raw.githubusercontent.com/" + repo + "/"
                                          + revision + "/" + relative))
                partial.replace(target)
            identity = {"repository": repo, "commit": revision, "files": paths}
            (ROOT / directory / "PINNED.json").write_text(json.dumps(identity, indent=2) + "\n")
            result["sources"][directory] = identity
        deps = ROOT / "deps"
        deps.mkdir(exist_ok=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "--only-binary=:all:", "--no-deps", "--target", str(deps), "av==16.0.1"],
                       check=True, timeout=300)
        sys.path.insert(0, str(deps))
        import av
        result["video_decoder"] = {"av": av.__version__,
                                   "av1_decoder": av.codec.Codec("av1", "r").name}
        result["packages"] = {}
        for name in ("torch", "stable-pretraining", "stable-worldmodel", "lightning", "hydra-core", "transformers"):
            try:
                result["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result["packages"][name] = None
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        (out / "source_preparation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
