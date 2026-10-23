import unittest

from prepare_data import SUITES, SUITE_PHASES, freeze_development, validate_development


def fake_inputs():
    rows = []
    for suite_index, suite in enumerate(SUITES):
        for task_id in range(10):
            task_name = f"{suite}-task-{task_id}"
            for demo_id in (0, 1):
                case_id = 2000 + suite_index * 20 + task_id * 2 + demo_id
                rows.append({
                    "case_id": case_id, "suite": suite, "task_id": task_id,
                    "task_name": task_name, "domain": "original", "phase": "early",
                    "demo_id": demo_id, "frame_index": 2, "sampler_seeds": [case_id, case_id + 1],
                    "observation_file": f"inputs/case_{case_id}.npz",
                    "metadata_file": f"inputs/case_{case_id}.json",
                    "dataset_file": f"/data/{suite}/{task_id}.hdf5", "dataset_source": "local",
                    "source_repo_id": None, "source_revision": None, "source_repo_path": None,
                    "dataset_size_bytes": 1000, "demo_length": 20, "dimension": "original",
                    "variant": "original", "variant_id": None, "original_task": task_name,
                    "source_index": None, "environment_seed": 830_000_000 + case_id * 17,
                })
            case_id = 3000 + suite_index * 10 + task_id
            rows.append({
                "case_id": case_id, "suite": suite, "task_id": task_id,
                "task_name": task_name, "domain": "original", "phase": "early",
                "demo_id": 2, "frame_index": 2, "sampler_seeds": [case_id, case_id + 1],
                "observation_file": f"inputs/case_{case_id}.npz",
                "metadata_file": f"inputs/case_{case_id}.json",
                "dataset_file": f"/data/{suite}/{task_id}.hdf5", "dataset_source": "local",
                "source_repo_id": None, "source_revision": None, "source_repo_path": None,
                "dataset_size_bytes": 1000, "demo_length": 20, "dimension": "original",
                "variant": "original", "variant_id": None, "original_task": task_name,
                "source_index": None, "environment_seed": 830_000_000 + case_id * 17,
            })
    return {"phase_positions": {"early": 0.2, "middle": 0.5, "late": 0.8}, "inputs": rows}


def fake_protocol():
    return {
        "protocol": "fastwam-vq-bilevel-rho-v1",
        "development": {
            "selection_seed": 20261019,
            "outer_case_ids": list(range(4000, 4008)),
            "confirmation_case_ids": list(range(5000, 5008)),
            "phase_pairs": [list(SUITE_PHASES[suite]) for suite in SUITES],
        },
    }


def fake_catalogs():
    return {(suite, task_id): {demo_id: 20 + demo_id for demo_id in range(10)}
            for suite in SUITES for task_id in range(10)}


class DevelopmentSamplingTests(unittest.TestCase):
    def setUp(self):
        self.source = fake_inputs()
        self.protocol = fake_protocol()
        self.catalogs = fake_catalogs()

    def test_freezes_two_disjoint_demo_splits(self):
        outer, confirmation = freeze_development(self.source, self.protocol, self.catalogs)
        self.assertEqual([row["case_id"] for row in outer], list(range(4000, 4008)))
        self.assertEqual([row["case_id"] for row in confirmation], list(range(5000, 5008)))
        validate_development(outer + confirmation, self.source, self.protocol)

        source_demos = {(row["suite"], row["task_id"], row["demo_id"])
                        for row in self.source["inputs"]}
        for suite_index, suite in enumerate(SUITES):
            selected_tasks = []
            for slot, phase in enumerate(SUITE_PHASES[suite]):
                index = suite_index * 2 + slot
                outer_row, confirmation_row = outer[index], confirmation[index]
                self.assertEqual(outer_row["suite"], suite)
                self.assertEqual(confirmation_row["suite"], suite)
                self.assertEqual(outer_row["task_id"], confirmation_row["task_id"])
                self.assertEqual(outer_row["phase"], phase)
                self.assertEqual(confirmation_row["phase"], phase)
                self.assertNotEqual(outer_row["demo_id"], confirmation_row["demo_id"])
                self.assertNotIn((suite, outer_row["task_id"], outer_row["demo_id"]), source_demos)
                self.assertNotIn((suite, confirmation_row["task_id"], confirmation_row["demo_id"]), source_demos)
                selected_tasks.append(outer_row["task_id"])
            self.assertEqual(len(set(selected_tasks)), 2)

    def test_sampling_is_repeatable_and_validation_rejects_shared_demo(self):
        first = freeze_development(self.source, self.protocol, self.catalogs)
        self.assertEqual(first, freeze_development(self.source, self.protocol, self.catalogs))
        outer, confirmation = first
        confirmation[0] = {**confirmation[0], "demo_id": outer[0]["demo_id"]}
        with self.assertRaises(RuntimeError):
            validate_development(outer + confirmation, self.source, self.protocol)


if __name__ == "__main__":
    unittest.main()
