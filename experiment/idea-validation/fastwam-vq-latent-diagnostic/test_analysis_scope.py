"""Small stdlib-only tests for shard selection and completion gates."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_results as analysis
import metrics


PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
ALL_CONTEXTS = metrics.frozen_contexts(PROTOCOL)
CONTEXTS_BY_SHARD = {
    shard: [row for row in ALL_CONTEXTS if row["shard"] == shard]
    for shard in (0, 1)
}


def synthetic_reader(calls, missing_shard=None, incomplete_shard=None):
    def read(path):
        path = Path(path)
        calls.append(path)
        shard = int(path.parent.name.split("-")[1])
        contexts = CONTEXTS_BY_SHARD[shard]
        if path.name == "summary.json":
            if shard == missing_shard:
                raise RuntimeError(f"missing shard {shard} summary")
            context = contexts[0]
            return {
                "status": "running" if shard == incomplete_shard else "complete",
                "protocol": PROTOCOL["protocol"],
                "shard": shard,
                "job_id": f"synthetic-{shard}",
                "context_count": 30,
                "expected_context_count": 30,
                "native_context_queries": 90,
                "teacher_future_queries": 60,
                "selected_epoch": 2,
                "target_linears": PROTOCOL["target_linears"],
                "episodes": 0,
                "self_future_parity": {
                    arm: {"case_id": context["case_id"], "seed_index": context["seed_index"],
                          "max_abs": 0.0, "rmse": 0.0}
                    for arm in ("bf16", "parent", "distilled")
                },
            }

        context = next(row for row in contexts
                       if path.stem == f"case_{row['case_id']}_seed_{row['seed_index']}")
        domain = "original" if context["case_id"] < 1014 else "plus"
        return {
            "protocol": PROTOCOL["protocol"],
            "context": context,
            "native_queries": 3,
            "teacher_future_queries": 2,
            "arms": list(analysis.EXPECTED_ARMS),
            "rows": [{"case_id": context["case_id"], "seed_index": context["seed_index"],
                      "split": context["split"], "arm": arm, "domain": domain,
                      "dimension": "synthetic"}
                     for arm in analysis.EXPECTED_ARMS],
        }

    return read


class AnalysisScopeTests(unittest.TestCase):
    def test_full_and_partial_scope_labels_keep_expected_denominators(self):
        full = analysis.build_scope(PROTOCOL, CONTEXTS_BY_SHARD, (0, 1))
        partial = analysis.build_scope(PROTOCOL, CONTEXTS_BY_SHARD, (0,))
        self.assertEqual(full["context_count"], 60)
        self.assertEqual(full["test_case_count"], 22)
        self.assertEqual(analysis.scope_label(full),
                         "full frozen cohort，60/60 contexts，22/22 testcases，full_cohort_complete=true")
        self.assertEqual(partial["context_count"], 30)
        self.assertEqual(partial["test_case_count"], 11)
        self.assertEqual(analysis.scope_label(partial),
                         "shard0 interim，30/60 contexts，11/22 testcases，full_cohort_complete=false")

    def test_partial_reads_only_selected_shard_and_keeps_receipt_gate(self):
        calls = []
        with patch.object(analysis, "read_json", side_effect=synthetic_reader(calls)):
            rows, _, gates = analysis.load_receipts(
                PROTOCOL, HERE / "synthetic-results", CONTEXTS_BY_SHARD, (0,))
        self.assertEqual(len(rows), 150)
        self.assertEqual(list(gates), ["0"])
        self.assertTrue(all("shard-0" in str(path) for path in calls))

    def test_full_scope_fails_when_a_shard_summary_is_missing(self):
        calls = []
        with patch.object(analysis, "read_json",
                          side_effect=synthetic_reader(calls, missing_shard=1)):
            with self.assertRaisesRegex(RuntimeError, "missing shard 1 summary"):
                analysis.load_receipts(
                    PROTOCOL, HERE / "synthetic-results", CONTEXTS_BY_SHARD, (0, 1))
        self.assertTrue(any(path.name == "summary.json" and "shard-1" in str(path)
                            for path in calls))

    def test_full_scope_fails_when_a_shard_is_not_complete(self):
        calls = []
        with patch.object(analysis, "read_json",
                          side_effect=synthetic_reader(calls, incomplete_shard=1)):
            with self.assertRaisesRegex(RuntimeError, "shard 1 summary is not complete"):
                analysis.load_receipts(
                    PROTOCOL, HERE / "synthetic-results", CONTEXTS_BY_SHARD, (0, 1))

    def test_latent_profiles_average_seeds_then_cases_and_check_vector_length(self):
        rows = []
        for case_id, values in ((10, (1.0, 3.0)), (11, (5.0, 7.0))):
            for seed_index, value in enumerate(values):
                rows.append({
                    "split": "test", "arm": "parent_self", "case_id": case_id,
                    "seed_index": seed_index, "latent_shape": [1, 2, 3, 1, 1],
                    "future_rmse": [value, value + 10],
                    "future_nmse": [value + 20, value + 30],
                    "temporal_delta_rmse": [value + 40, value + 50],
                    "temporal_delta_nmse": [value + 60, value + 70],
                })
        profiles = analysis.aggregate_latent_time_profiles(rows, PROTOCOL)
        profile = profiles["test"]["parent_self"]
        self.assertEqual(profile["latent_shape"], [1, 2, 3, 1, 1])
        self.assertEqual(profile["future_latent_frame_indices"], [1, 2])
        self.assertEqual(profile["future_rmse"], [4.0, 14.0])
        self.assertEqual(profile["temporal_delta_rmse"], [44.0, 54.0])
        self.assertEqual(profile["temporal_delta_transitions"][0],
                         {"from_latent_frame": 0, "to_latent_frame": 1})
        self.assertEqual(profile["context_count"], 4)
        bad_rows = [dict(row) for row in rows]
        bad_rows[0]["future_rmse"] = [1.0]
        with self.assertRaisesRegex(RuntimeError, "future_rmse length"):
            analysis.aggregate_latent_time_profiles(bad_rows, PROTOCOL)

    def test_case_teacher_future_rescue_counts_positive_as_improvement(self):
        rows = []
        for case_id, value in enumerate((1.0, -1.0, 0.0)):
            row = {"case_id": case_id}
            for policy in ("parent", "distilled"):
                for metric in analysis.ACTION_METRICS:
                    row[f"{policy}_teacher_future_{metric}_rescue"] = value
            rows.append(row)
        summary = analysis.summarize_case_teacher_future_rescue(rows)
        first10 = summary["policies"]["parent"]["first10motor_rmse"]
        self.assertTrue(summary["positive_is_improvement"])
        self.assertEqual((first10["improved_count"], first10["worsened_count"],
                          first10["unchanged_count"], first10["median_rescue"]),
                         (1, 1, 1, 0.0))

if __name__ == "__main__":
    unittest.main()
