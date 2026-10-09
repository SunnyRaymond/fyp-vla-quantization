from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import cohort


def synthetic_source() -> dict:
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    dimensions = (
        "objects_layout", "camera_viewpoints", "robot_initial_states",
        "language_instructions", "light_conditions", "background_textures", "sensor_noise",
    )
    variants = []
    for suite_index, suite in enumerate(suites):
        for dimension_index, dimension in enumerate(dimensions):
            for candidate in range(50):
                variants.append({
                    "suite": suite,
                    "task_id": candidate,
                    "task_name": f"{suite}_task_{candidate}",
                    "dimension": dimension,
                    "subtype": ("long" if candidate % 2 else "short") if dimension == "language_instructions"
                               else ("noise" if candidate % 2 else "clean") if dimension == "sensor_noise"
                               else f"subtype_{candidate % 3}",
                    "original_task": f"task_{candidate % 5}",
                    "difficulty": candidate % 3,
                    "variant_id": f"{suite}:{dimension}:{candidate:03d}",
                    "classification_id": suite_index * 10000 + dimension_index * 100 + candidate,
                    "state_id": 0,
                    "score": candidate,
                })
    return {
        "schema_version": 1,
        "protocol": cohort.SOURCE_PROTOCOL,
        "seed": 20261005,
        "source": {
            "repository": cohort.OFFICIAL_REPOSITORY,
            "commit": "official-source-commit",
            "classification_metadata": "synthetic test fixture",
        },
        "variants": variants,
    }


class CohortTests(unittest.TestCase):
    def test_builds_balanced_cohort_and_four_disjoint_shards(self):
        source = synthetic_source()
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "pilot.json"
            output_path = Path(directory) / "out" / "manifest.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            document = cohort.build_cohort(source_path, output_path)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        cohort.validate_cohort(document)
        self.assertEqual(saved, document)
        self.assertEqual(document["arms"], list(cohort.ARMS))
        self.assertEqual(document["source"]["commit"], source["source"]["commit"])
        self.assertEqual(document["source_manifest"]["seed"], source["seed"])
        self.assertEqual([row["index"] for row in document["variants"]], list(range(280)))
        self.assertEqual(len({row["variant_id"] for row in document["variants"]}), 280)

        cells = Counter((row["suite"], row["dimension"]) for row in document["variants"])
        self.assertEqual(len(cells), 28)
        self.assertEqual(set(cells.values()), {10})
        selected = {row["variant_id"]: row for row in document["variants"]}
        for row in document["variants"]:
            source_row = next(item for item in source["variants"] if item["variant_id"] == row["variant_id"])
            for key in ("suite", "task_id", "task_name", "dimension", "subtype", "original_task",
                        "difficulty", "classification_id", "state_id", "score"):
                self.assertEqual(row[key], source_row[key])
        self.assertEqual(set(selected), {row["variant_id"] for row in document["variants"]})

        workers_by_dimension = Counter(
            (row["dimension"], worker_id)
            for shard in document["shards"]
            for index in shard["indices"]
            for row in [document["variants"][index]]
            for worker_id in [shard["worker_id"]]
        )
        self.assertEqual(set(workers_by_dimension.values()), {10})

    def test_seed_reproduces_selection(self):
        source = synthetic_source()
        changed_scores = synthetic_source()
        for row in changed_scores["variants"]:
            row["score"] = 50 - row["score"]
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "pilot.json"
            changed_path = Path(directory) / "pilot_changed_scores.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            changed_path.write_text(json.dumps(changed_scores), encoding="utf-8")
            first = cohort.build_cohort(source_path, Path(directory) / "first.json")
            second = cohort.build_cohort(changed_path, Path(directory) / "second.json")
        self.assertEqual([row["variant_id"] for row in first["variants"]],
                         [row["variant_id"] for row in second["variants"]])
        self.assertEqual(first["shards"], second["shards"])

    def test_rejects_wrong_source_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "pilot.json"
            source = synthetic_source()
            source["protocol"] = "unexpected"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source protocol"):
                cohort.build_cohort(source_path, Path(directory) / "out.json")


if __name__ == "__main__":
    unittest.main()
