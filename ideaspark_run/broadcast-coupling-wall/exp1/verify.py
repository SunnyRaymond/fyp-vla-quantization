"""Independent CPU-side contract and pre-gate verification for Wall Exp1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from core import ARMS, DEFAULT_EPISODES, HORIZONS, PATCHES, TAIL_DIM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(job_dir: Path) -> dict:
    summary = load_json(job_dir / "summary.json")
    records = [json.loads(line) for line in (job_dir / "per_state_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    checks = {
        "summary_complete": summary.get("status") == "complete",
        "episodes_exact": summary.get("episode_indices") == list(DEFAULT_EPISODES),
        "horizons_exact": summary.get("horizons") == list(HORIZONS),
        "arms_exact": summary.get("arms") == list(ARMS),
        "record_count": len(records) == len(DEFAULT_EPISODES) * len(HORIZONS),
        "model_unchanged": summary.get("model_state_unchanged") is True,
        "a100": "A100" in json.dumps(summary.get("gpu", {})),
    }
    by_key = {(int(row["episode_index"]), int(row["horizon"])): row for row in records}
    checks["paired_keys"] = set(by_key) == {(e, h) for e in DEFAULT_EPISODES for h in HORIZONS}
    raw_checks = []
    raw_seen = set()
    for episode in DEFAULT_EPISODES:
        row = by_key.get((episode, HORIZONS[0]))
        if row is None:
            raw_checks.append(False)
            continue
        raw_path = Path(row["raw_treatment_path"])
        if raw_path in raw_seen or not raw_path.is_file() or sha256(raw_path) != row["raw_treatment_sha256"]:
            raw_checks.append(False)
            continue
        raw_seen.add(raw_path)
        with np.load(raw_path, allow_pickle=False) as data:
            required = {"episode_index", "underlying_index", "initial_tail", "fp32", "rtn", "shared_stochastic", "iid_patch_stochastic", "balanced_broadcast"}
            good = required.issubset(data.files)
            if good:
                initial = data["initial_tail"]
                shared = data["shared_stochastic"]
                balanced = data["balanced_broadcast"]
                good = (
                    initial.shape == (PATCHES, TAIL_DIM)
                    and shared.shape == (3, PATCHES, TAIL_DIM)
                    and balanced.shape == shared.shape
                    and np.isfinite(initial).all()
                    and np.isfinite(shared).all()
                    and np.isfinite(balanced).all()
                    and np.array_equal(np.sort(shared, axis=0), np.sort(balanced, axis=0))
                )
                if good:
                    shared_mse = float(np.mean((shared.astype(np.float64) - initial.astype(np.float64)) ** 2))
                    balanced_mse = float(np.mean((balanced.astype(np.float64) - initial.astype(np.float64)) ** 2))
                    good = math.isclose(shared_mse, balanced_mse, rel_tol=1e-12, abs_tol=1e-12)
            raw_checks.append(bool(good))
    checks["raw_treatments"] = len(raw_checks) == len(DEFAULT_EPISODES) and all(raw_checks)

    gains: list[float | None] = []
    positive = 0
    for episode in DEFAULT_EPISODES:
        row = by_key.get((episode, 5))
        if row is None:
            gains.append(None)
            continue
        shared = float(row["arms"]["shared_stochastic"]["real_future_feature_mse"])
        balanced = float(row["arms"]["balanced_broadcast"]["real_future_feature_mse"])
        gain = (shared - balanced) / shared if shared > 1e-12 else None
        gains.append(gain)
        positive += int(gain is not None and math.isfinite(gain) and gain >= 0.10)
    engineering_pass = all(checks.values())
    if not engineering_pass:
        decision = "inconclusive_binding"
    elif positive >= 5:
        decision = "pre_gate_go"
    else:
        decision = "mechanism_no_go_stop"
    result = {
        "schema": "broadcast-coupling-wall-exp1-verification-v1",
        "engineering_pass": engineering_pass,
        "checks": checks,
        "primary_gate": {
            "horizon": 5,
            "comparison": "balanced_broadcast_vs_shared_stochastic",
            "per_episode_relative_gain": gains,
            "threshold": "at least 5/6 episodes with gain >= 0.10",
            "positive_count": positive,
        },
        "decision": decision,
        "expansion_authorized": decision == "pre_gate_go",
        "boundary": "initial exact-broadcast A4 fake-quantization only; later autoregressive calls FP32; no native low-bit or STaMP claim",
    }
    target = job_dir / "verification.json"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path)
    args = parser.parse_args()
    result = verify(args.job_dir.resolve())
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["engineering_pass"] else 65


if __name__ == "__main__":
    raise SystemExit(main())
