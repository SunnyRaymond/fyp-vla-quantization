"""Start-of-job check of the closed-loop glue, on the A16 baselines' QTIP_W2 weights: the same copied loop_worker,
engine files, environment and prep link the QTIP_G_W2 loop uses hours later. build_runtime runs its own gates
(T5-cache anchor == full-T5 anchor bitwise; eager == cache+Graph bitwise), then one short episode runs.

argv: loop protocol, check root (holds results/prep and results/quantized/QTIP_W2 links), baseline root.
"""
from __future__ import annotations

from pathlib import Path
import sys

from common import guard, read, save
import loop_worker

ARM = "QTIP_W2"


def main():
    guard()
    root, baseline = Path(sys.argv[2]), Path(sys.argv[3])
    for name in ("prep", f"quantized/{ARM}"):
        link = root / "results" / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(baseline / "results" / name)
    p = dict(read(sys.argv[1]), remote_root=str(root), arms={ARM: dict(method="qtip", bits=2)})
    runtime = loop_worker.build_runtime(p, ARM, root / "results" / "startup")
    rows = read(p["panel"]["manifest"])["variants"]
    index = next(i for i, r in enumerate(rows) if r["suite"] != "libero_10")
    try:
        record = loop_worker.episode(runtime, rows[index], index, ARM)
    finally:
        if runtime.env is not None:
            runtime.env.close()
    record.pop("policy_call_seconds")
    save(root / "results" / "episode.json", record)
    print(f"LOOP_CHECK_PASSED index={index} success={record['success']} steps={record['steps']} "
          f"seconds={record['episode_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
