"""Guarded CPU fake-environment checks for the original paired280 loop."""
import copy
from pathlib import Path
import tempfile
import unittest

import training_closedloop as closedloop


def toy_parent():
    manifest = "/frozen/original/manifest.json"
    return {
        "protocol": "control-kl-v1",
        "frozen_evaluation": {"closedloop_manifest": manifest, "closedloop_episodes": 280},
        "source_closed_protocol": {
            "cohort": {
                "manifest_protocol": closedloop.PROTOCOL,
                "reference_manifest": manifest,
                "variants_per_arm": 280,
                "env_seed_base": 800_000_000,
                "sampler_seed_base": 900_000_000,
                "seed_stride": 1000,
                "state_id": 0,
                "paired_seeds_across_arms": True,
            },
            "inference": {
                "execute_per_replan": 10, "settling_steps": 30,
                "max_steps_normal": 400, "max_steps_long": 700,
            },
        },
    }


def toy_manifest():
    rows = []
    for suite in range(4):
        for dimension in range(7):
            for _ in range(10):
                index = len(rows)
                rows.append({
                    "index": index,
                    "variant_id": 17 if index == 0 else f"suite{suite}:{index}",
                    "suite": "libero_10" if suite == 3 else f"suite{suite}",
                    "dimension": f"dimension{dimension}", "task_id": index,
                })
    return {"protocol": closedloop.PROTOCOL, "variants": rows}


class FakeEnv:
    def __init__(self, torch, stop_at):
        self.torch, self.stop_at, self.steps = torch, stop_at, 0
        self.first_command_value = None

    def step(self, command):
        if self.first_command_value is None:
            self.first_command_value = command[0]
        self.steps += 1
        return {"step": self.steps}, 0.0, self.steps == self.stop_at, {}


class FakeDeadline:
    def __init__(self, values):
        self.values, self.reserves = list(values), []

    def near(self, reserve=180):
        self.reserves.append(reserve)
        return self.values.pop(0)


class FakeSwitching:
    def __init__(self, torch, stop_at):
        self.torch = torch
        self.banks = type("Banks", (), {"seen": set()})()
        self.runtime = type("Runtime", (), {})()
        self.runtime.q = type("Q", (), {
            "_native_int4_gemm_calls": 0, "_kv_active": False,
        })()
        self.runtime.env = FakeEnv(torch, stop_at)
        self.runtime.infer_seeds = []
        self.runtime.description = "toy task"

        def task(row, index):
            self.runtime.task_identity = (row["task_id"], index)
            return {"observation": "toy"}

        def infer(datum, seed, measure=True):
            if measure is not False:
                raise AssertionError("Closed-loop calls must not measure query latency")
            self.runtime.infer_seeds.append(seed)
            self.banks.seen = set(range(closedloop.EXPECTED_MODULES))
            return torch.full((32, 7), float(len(self.runtime.infer_seeds))), {}

        self.runtime.task = task
        self.runtime.infer = infer
        def command(action):
            # This value is not exactly representable as float32.
            return torch.full((32, 7), 0.1000000000001, dtype=torch.float64)

        self.runtime.command = command
        self.datum = lambda observation: observation


class ClosedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from control_common import guard
        guard()
        import torch
        cls.torch = torch

    def setUp(self):
        self.parent = toy_parent()
        self.row = toy_manifest()["variants"][0]
        self.binding = {"unit": "train_candidate", "producer_job": "train-1.pbs"}
        self.state = {
            "current_index": None, "current_replan": None,
            "episode_query_attempted": 0, "episode_query_completed": 0,
        }

    def test_two_shards_cover_original_280_without_overlap(self):
        left, right = closedloop.shard_indices(0), closedloop.shard_indices(1)
        self.assertEqual((len(left), len(right)), (140, 140))
        self.assertFalse(set(left) & set(right))
        self.assertEqual(sorted(left + right), list(range(280)))
        with self.assertRaises(ValueError):
            closedloop.shard_indices(True)
        rows = closedloop.validate_manifest_doc(toy_manifest(), self.parent)
        self.assertEqual(len(rows), 280)
        closedloop.validate_closedloop_protocol(self.parent)
        foreign = copy.deepcopy(self.parent)
        foreign["source_closed_protocol"]["cohort"]["reference_manifest"] = "/other.json"
        with self.assertRaises(ValueError):
            closedloop.validate_closedloop_protocol(foreign)
        self.assertEqual(closedloop.episode_cap({"suite": "libero_10"}), 700)
        self.assertEqual(closedloop.episode_cap({"suite": "libero_spatial"}), 400)

    def test_seed_chunk_and_done_are_recorded_and_validated(self):
        switching = FakeSwitching(self.torch, stop_at=12)
        deadline = FakeDeadline([False, False])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0.json"
            record = closedloop.run_and_save_episode(
                switching, self.parent, self.row, 0, "candidate", self.binding,
                "closedloop_candidate_0", "job-2.pbs", deadline, self.state, path)
            self.assertTrue(path.is_file())
        self.assertEqual(switching.runtime.task_identity, (0, 0))
        self.assertEqual(switching.runtime.infer_seeds, [900_000_000, 900_000_001])
        self.assertEqual(record["sampler_seeds"], switching.runtime.infer_seeds)
        self.assertEqual((record["steps"], record["replans"]), (12, 2))
        self.assertTrue(record["success"])
        self.assertEqual(record["termination"], "success")
        self.assertEqual(record["variant_id"], "17")
        self.assertEqual(switching.runtime.env.first_command_value, 0.1000000000001)
        self.assertEqual(record["environment_seed"], 800_000_000)
        self.assertEqual(len(record["first_action_chunk"]), 32)
        self.assertTrue(all(len(row) == 7 for row in record["first_action_chunk"]))
        self.assertEqual(deadline.reserves, [180, 180])
        self.assertIs(closedloop.validate_record(
            record, self.parent, self.row, 0, "candidate", self.binding), record)
        foreign = copy.deepcopy(record)
        foreign["deployment_binding"] = {"producer_job": "other.pbs"}
        with self.assertRaises(ValueError):
            closedloop.validate_record(foreign, self.parent, self.row, 0,
                                      "candidate", self.binding)

    def test_softstop_discards_partial_episode_without_json(self):
        switching = FakeSwitching(self.torch, stop_at=200)
        deadline = FakeDeadline([False, True])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0.json"
            with self.assertRaises(closedloop.SoftStop):
                closedloop.run_and_save_episode(
                    switching, self.parent, self.row, 0, "candidate", self.binding,
                    "closedloop_candidate_0", "job-3.pbs", deadline, self.state, path)
            self.assertFalse(path.exists())
        self.assertEqual(self.state["episode_query_attempted"], 1)
        self.assertEqual(self.state["episode_query_completed"], 1)

    def test_timeout_cap_and_completed_record_reuse(self):
        for index in (0, 210):
            row = toy_manifest()["variants"][index]
            cap = closedloop.episode_cap(row)
            switching = FakeSwitching(self.torch, stop_at=None)
            state = {"episode_query_attempted": 0, "episode_query_completed": 0}
            deadline = FakeDeadline([False] * (cap // 10))
            with tempfile.TemporaryDirectory() as directory:
                episodes = Path(directory)
                record = closedloop.run_and_save_episode(
                    switching, self.parent, row, index, "candidate", self.binding,
                    "closedloop_candidate_0", "job-4.pbs", deadline, state,
                    episodes / f"{index}.json")
                rows = toy_manifest()["variants"]
                reused = closedloop._existing_records(
                    episodes, rows, closedloop.shard_indices(0), self.parent,
                    "candidate", self.binding)
                self.assertEqual(list(reused), [index])
            self.assertFalse(record["success"])
            self.assertEqual(record["termination"], "control_step_timeout")
            self.assertEqual(record["steps"], cap)
            self.assertEqual(record["replans"], cap // 10)
            self.assertEqual(switching.runtime.env.steps, cap)


if __name__ == "__main__":
    from control_common import guard
    guard()
    unittest.main()
