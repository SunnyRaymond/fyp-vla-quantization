import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core import ARMS, DEFAULT_EPISODES, HORIZONS, PATCHES, TAIL_DIM
from verify import verify


class VerifyContractTest(unittest.TestCase):
    def test_positive_gate_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = {
                "status": "complete",
                "episode_indices": list(DEFAULT_EPISODES),
                "horizons": list(HORIZONS),
                "arms": list(ARMS),
                "model_state_unchanged": True,
                "gpu": {"visible_gpus": [{"name": "NVIDIA A100-SXM4-40GB"}]},
            }
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            records = []
            initial = np.zeros((PATCHES, TAIL_DIM), dtype=np.float32)
            shared = np.stack([np.full_like(initial, value) for value in (0.0, 1.0, 2.0)])
            balanced = np.roll(shared, shift=1, axis=0)
            for episode in DEFAULT_EPISODES:
                raw = root / f"raw_episode_{episode:03d}.npz"
                np.savez_compressed(
                    raw,
                    episode_index=np.asarray(episode, dtype=np.int64),
                    underlying_index=np.asarray(episode + 1000, dtype=np.int64),
                    initial_tail=initial,
                    fp32=initial[None],
                    rtn=initial[None],
                    shared_stochastic=shared,
                    iid_patch_stochastic=shared,
                    balanced_broadcast=balanced,
                )
                raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
                for horizon in HORIZONS:
                    arms = {
                        arm: {"real_future_feature_mse": 0.8 if arm == "balanced_broadcast" else 1.0}
                        for arm in ARMS
                    }
                    records.append({
                        "episode_index": episode,
                        "horizon": horizon,
                        "arms": arms,
                        "raw_treatment_path": str(raw),
                        "raw_treatment_sha256": raw_hash,
                    })
            (root / "per_state_records.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            result = verify(root)
            self.assertTrue(result["engineering_pass"])
            self.assertEqual(result["decision"], "pre_gate_go")
            self.assertEqual(result["primary_gate"]["positive_count"], len(DEFAULT_EPISODES))


if __name__ == "__main__":
    unittest.main()
