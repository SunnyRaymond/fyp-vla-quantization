import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import queue_race


class QueueRaceTests(unittest.TestCase):
    def _config(self, root, **updates):
        config = {"enabled": True, "race_id": "pbs-race-2026-10-25",
                  "scope_stages": list(queue_race.SHORT_STAGES),
                  "wide_stage": queue_race.WIDE_STAGE,
                  "wide_stages": list(queue_race.WIDE_STAGES)}
        config.update(updates)
        (root / "queue_race.json").write_text(json.dumps(config), encoding="utf-8")
        return root / "queue_race.json"

    def test_first_short_winner_allows_same_branch_shards_and_rejects_wide(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._config(root)
            guarded = []

            def fake_guard():
                if not guarded:
                    self.assertFalse((root / "control").exists())
                guarded.append(True)

            with patch.object(queue_race, "guard", side_effect=fake_guard), \
                    patch.dict(os.environ, {"PBS_JOBID": "short-1.pbs101"}):
                first = queue_race.run(root, "gate_spin")
                shard = queue_race.run(root, "gate_qua")
                wide = queue_race.run(root, "gate_wide")
            self.assertEqual(first["status"], "claimed")
            self.assertEqual(shard["status"], "same_branch")
            self.assertEqual(shard["exit_code"], 0)
            self.assertEqual(wide["status"], "branch_conflict")
            self.assertEqual(wide["exit_code"], 75)
            self.assertEqual(wide["owner"]["branch"], "short")
            self.assertEqual(len(guarded), 3)

    def test_atomic_race_has_one_branch_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._config(root)
            barrier = threading.Barrier(2)
            results = []

            def attempt(stage):
                barrier.wait()
                results.append(queue_race.run(root, stage))

            with patch.object(queue_race, "guard", return_value=None), \
                    patch.dict(os.environ, {"PBS_JOBID": "race-job.pbs101"}):
                threads = [threading.Thread(target=attempt, args=(stage,))
                           for stage in ("gate_svd", "gate_wide")]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            self.assertEqual(sum(result["status"] == "claimed" for result in results), 1)
            self.assertEqual(sum(result["status"] == "branch_conflict" for result in results), 1)
            self.assertEqual({result["exit_code"] for result in results}, {0, 75})
            owner_path = root / "control" / "races" / "pbs-race-2026-10-25" / "claim" / "owner.json"
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            self.assertIn(owner["branch"], ("short", "wide"))
            self.assertTrue(owner["pbs_job_id"])
            self.assertTrue(owner["hostname"])
            self.assertTrue(owner["created_utc"])

    def test_short_and_each_wide_variant_race_mutually_exclusively(self):
        for wide_stage in queue_race.WIDE_STAGES:
            with self.subTest(wide_stage=wide_stage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._config(root)
                barrier = threading.Barrier(2)
                results = []

                def attempt(stage):
                    barrier.wait()
                    results.append(queue_race.run(root, stage))

                with patch.object(queue_race, "guard", return_value=None), \
                        patch.dict(os.environ, {"PBS_JOBID": "race-job.pbs101"}):
                    threads = [threading.Thread(target=attempt, args=(stage,))
                               for stage in ("ours_gate", wide_stage)]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                self.assertEqual(sum(row["status"] == "claimed" for row in results), 1)
                self.assertEqual(sum(row["exit_code"] == 75 for row in results), 1)

    def test_wide_reentry_requires_the_same_pbs_job_across_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._config(root)
            with patch.object(queue_race, "guard", return_value=None), \
                    patch.dict(os.environ, {"PBS_JOBID": "wide-winner.pbs101"}):
                first = queue_race.run(root, "gate_wide_g1")
                same_job_other_variant = queue_race.run(root, "gate_wide_g3")
            self.assertEqual(first["status"], "claimed")
            self.assertEqual(same_job_other_variant["status"], "same_branch")
            self.assertEqual(same_job_other_variant["exit_code"], 0)

            with patch.object(queue_race, "guard", return_value=None), \
                    patch.dict(os.environ, {"PBS_JOBID": "other-wide-job.pbs101"}):
                other_job = queue_race.run(root, "gate_wide_g2")
            self.assertEqual(other_job["status"], "wide_job_conflict")
            self.assertEqual(other_job["exit_code"], 75)

    def test_disabled_or_unconfigured_stages_skip_without_guard_or_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(queue_race, "guard") as guard:
                self.assertEqual(queue_race.run(root, "gate_spin")["exit_code"], 0)
                self._config(root, enabled=False)
                self.assertEqual(queue_race.run(root, "gate_spin")["status"], "skipped")
                self._config(root)
                outside = queue_race.run(root, "unrelated_stage")
                self.assertEqual(outside["status"], "skipped")
                self.assertEqual(outside["exit_code"], 0)
                guard.assert_not_called()
            self.assertFalse((root / "control").exists())

    def test_cli_emits_small_json_skip_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._config(root)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = queue_race.main(["--root", str(root), "--stage", "outside"])
            self.assertEqual(code, 0)
            decision = json.loads(output.getvalue())
            self.assertEqual(decision["status"], "skipped")
            self.assertLess(len(output.getvalue()), 512)

    def test_race_id_path_traversal_is_rejected_before_lock_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._config(root, race_id="../escape")
            with patch.object(queue_race, "guard", return_value=None), \
                    patch.dict(os.environ, {"PBS_JOBID": "job.pbs101"}):
                with self.assertRaisesRegex(ValueError, "safe basename"):
                    queue_race.run(root, "ours_gate")
            self.assertFalse((root / "control").exists())

    def test_legacy_single_wide_stage_config_remains_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._config(root)
            path = root / "queue_race.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config.pop("wide_stages")
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(queue_race, "guard", return_value=None), \
                    patch.dict(os.environ, {"PBS_JOBID": "wide-job.pbs101"}):
                first = queue_race.run(root, "gate_wide")
                unsupported_variant = queue_race.run(root, "gate_wide_g1")
            self.assertEqual(first["status"], "claimed")
            self.assertEqual(unsupported_variant["status"], "skipped")
            self.assertEqual(unsupported_variant["exit_code"], 0)

    def test_unknown_wide_stage_and_unknown_config_variant_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._config(root)
            with patch.object(queue_race, "guard") as guard:
                with self.assertRaisesRegex(ValueError, "Unknown wide stage"):
                    queue_race.run(root, "gate_wide_g4")
                self._config(root, wide_stages=["gate_wide", "gate_wide_g4"])
                with self.assertRaisesRegex(RuntimeError, "fixed wide stages"):
                    queue_race.run(root, "gate_wide")
                guard.assert_not_called()

    def test_owner_validation_rejects_wrong_race_and_unknown_branch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._config(root)
            with patch.object(queue_race, "guard", return_value=None), \
                    patch.dict(os.environ, {"PBS_JOBID": "job.pbs101"}):
                queue_race.run(root, "ours_gate")
                owner_path = (root / "control" / "races" / "pbs-race-2026-10-25"
                              / "claim" / "owner.json")
                owner = json.loads(owner_path.read_text(encoding="utf-8"))
                owner["race_id"] = "another-race"
                owner_path.write_text(json.dumps(owner), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "different race"):
                    queue_race.run(root, "perf_bf16")
                owner["race_id"] = "pbs-race-2026-10-25"
                owner["branch"] = "unknown"
                owner_path.write_text(json.dumps(owner), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "unknown branch"):
                    queue_race.run(root, "perf_bf16")

    def test_missing_owner_times_out_without_stealing_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._config(root)
            claim = root / "control" / "races" / "pbs-race-2026-10-25" / "claim"
            claim.mkdir(parents=True)
            with patch.object(queue_race, "guard", return_value=None), \
                    patch.object(queue_race, "OWNER_WAIT_SECONDS", 0.01), \
                    patch.dict(os.environ, {"PBS_JOBID": "job.pbs101"}):
                with self.assertRaisesRegex(TimeoutError, "lock was retained"):
                    queue_race.run(root, "gate_wide")
            self.assertTrue(claim.is_dir())
            self.assertFalse((claim / "owner.json").exists())


if __name__ == "__main__":
    unittest.main()
