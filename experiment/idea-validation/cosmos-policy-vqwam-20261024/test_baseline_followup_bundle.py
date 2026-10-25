"""Standard-library receipts for the two-GPU baseline follow-up runner."""
import json
from pathlib import Path
import tempfile
import unittest

import baseline_followup_bundle as bundle


class BaselineFollowupBundleTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {
            "protocol": "protocol-c2",
            "performance": {"case_ids": list(range(40)), "seed_indices": [0, 1],
                            "blocks": 3, "repetitions": 3},
            "calibration": {"case_ids": list(range(80))},
        }

    def test_vidcal_requires_complete_current_job_and_all_280_modules(self):
        summary = {"status": "complete", "protocol": "protocol-c2", "pbs_job_id": "44.pbs",
                   "arm": "viditq_w4a4", "completed_modules": 280, "expected_modules": 280,
                   "completed_weight_numel": 1_761_607_680,
                   "target_weight_numel": 1_761_607_680,
                   "bpw_within_target_tolerance": True}
        self.assertTrue(bundle._vidcal_complete(summary, self.protocol, "44.pbs"))
        self.assertFalse(bundle._vidcal_complete(summary, self.protocol, "45.pbs"))
        summary["completed_modules"] = 279
        self.assertFalse(bundle._vidcal_complete(summary, self.protocol, "44.pbs"))
        summary["completed_modules"] = 280
        summary["bpw_within_target_tolerance"] = False
        self.assertFalse(bundle._vidcal_complete(summary, self.protocol, "44.pbs"))

    def test_real_protocol_perf_summaries_resolve_and_record_both_720_panels(self):
        protocol = json.loads((Path(__file__).with_name("protocol.json")).read_text(encoding="utf-8"))
        perf = protocol["performance"]
        repetitions = perf["blocks"] * perf["repetitions"]
        by_query = [{"case_id": case, "seed_index": seed,
                     "completed_repetitions": repetitions,
                     "expected_repetitions": repetitions}
                    for case in perf["case_ids"] for seed in perf["seed_indices"]]
        self.assertEqual(len(by_query), 80)
        for stage in ("perf_svdquant", "perf_viditq"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root, job_id = Path(directory), "44.pbs"
                arm = protocol["stage_arms"][stage]
                summary_path = bundle._summary_path(root, stage, protocol)
                self.assertEqual(summary_path, root / "results/performance" / arm / "summary.json")
                summary_path.parent.mkdir(parents=True)
                summary_path.write_text(json.dumps({
                    "status": "complete", "protocol": protocol["protocol"],
                    "pbs_job_id": job_id, "arm": arm, "complete": True,
                    "coverage": {"queries": 80, "completed_queries": 80,
                                 "samples": 720, "expected_samples": 720,
                                 "by_query": by_query, "missing": []},
                }), encoding="utf-8")
                marker = root / "artifacts" / job_id / "stages" / stage / "PIPELINE_COMPLETE"
                marker.parent.mkdir(parents=True)
                marker.touch()
                row = {"pbs_job_id": job_id}
                bundle._record_stage(root, stage, 0, protocol, job_id, row)
                self.assertEqual(row["status"], "complete")
                self.assertEqual(row["summary_pbs_job_id"], job_id)
                self.assertTrue(row["complete_marker_exists"])

    def test_parent_complete_requires_all_three_current_job_summaries(self):
        stages = {stage: {"status": "complete", "exit_code": 0,
                          "pbs_job_id": "44.pbs", "summary_pbs_job_id": "44.pbs",
                          "complete_marker_exists": True}
                  for stage in bundle.STAGES}
        self.assertEqual(bundle._finish_status(stages), ("complete", 0, True))
        stages["vidcal"]["summary_pbs_job_id"] = "old.pbs"
        self.assertEqual(bundle._finish_status(stages), ("checkpointed", 2, False))

    def test_failed_calibration_or_perf_is_not_reported_complete(self):
        stages = {stage: {"status": "complete", "exit_code": 0,
                          "pbs_job_id": "44.pbs", "summary_pbs_job_id": "44.pbs",
                          "complete_marker_exists": True}
                  for stage in bundle.STAGES}
        stages["vidcal"].update(status="failed", exit_code=1)
        stages["perf_viditq"].update(status="not_started_dependency", exit_code=None)
        self.assertEqual(bundle._finish_status(stages), ("failed", 1, False))

    def test_two_device_profile_does_not_change_original_three_device_default(self):
        self.assertEqual(bundle.shared.parse_visible_devices("0,1", expected_count=2), ["0", "1"])
        with self.assertRaises(bundle.shared.BundleError):
            bundle.shared.parse_visible_devices("0,1", expected_count=3)


if __name__ == "__main__":
    unittest.main()
