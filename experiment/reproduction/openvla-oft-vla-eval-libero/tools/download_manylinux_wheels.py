from pathlib import Path
from subprocess import run
from urllib.parse import urlparse
import tomllib

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "source/vla-evaluation-harness/src/vla_eval/model_servers/oft.py.lock"
OUT = ROOT / "source/wheelhouse-manylinux_2_28"
ARIA2 = ROOT / "tools/aria2/aria2-1.37.0-win-64bit-build1/aria2c.exe"

platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)] + [
    "manylinux2014_x86_64",
    "manylinux2010_x86_64",
    "manylinux1_x86_64",
    "linux_x86_64",
    "any",
]
supported = list(cpython_tags((3, 11), platforms=platforms)) + list(
    compatible_tags((3, 11), interpreter="cp311", platforms=platforms)
)
rank = {tag: index for index, tag in enumerate(supported)}

selected = {}
for package in tomllib.loads(LOCK.read_text(encoding="utf-8"))["package"]:
    if "registry" not in package.get("source", {}):
        continue
    candidates = []
    for wheel in package.get("wheels", []):
        filename = Path(urlparse(wheel["url"]).path).name
        tags = parse_wheel_filename(filename)[3]
        best = min((rank[tag] for tag in tags if tag in rank), default=None)
        if best is not None:
            candidates.append((best, wheel["size"], filename, wheel["url"]))
    if candidates:
        selected[(package["name"], package["version"])] = min(candidates)

OUT.mkdir(parents=True, exist_ok=True)
plan = OUT / "selected-wheels.tsv"
plan.write_text(
    "name\tversion\tbytes\tfilename\turl\n"
    + "".join(
        f"{name}\t{version}\t{size}\t{filename}\t{url}\n"
        for (name, version), (_, size, filename, url) in sorted(selected.items())
    ),
    encoding="utf-8",
)
aria2_input = OUT / "aria2-input.txt"
aria2_input.write_text(
    "".join(
        f"{url}\n  dir={OUT.as_posix()}\n  out={filename}\n"
        for _, _, filename, url in selected.values()
    ),
    encoding="utf-8",
)

run(
    [
        str(ARIA2),
        "-c",
        "-x8",
        "-s8",
        "-j8",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        f"--input-file={aria2_input}",
    ],
    check=True,
)
bad = [filename for _, size, filename, _ in selected.values() if (OUT / filename).stat().st_size != size]
assert not bad, f"size mismatch: {bad}"
print(f"READY wheels={len(selected)} bytes={sum(item[1] for item in selected.values())} path={OUT}")
