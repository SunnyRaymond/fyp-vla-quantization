"""Allocation-guarded CPU checks for complete-pair raw-loss evaluation."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from control_common import guard


ROOT = Path(__file__).resolve().parent
DUAL_KL = ROOT.parent / "fastwam-vq-dual-kl"
if (DUAL_KL / "softbin_loss.py").is_file() and str(DUAL_KL) not in sys.path:
    sys.path.insert(0, str(DUAL_KL))

import final_c80_eval as evaluation


_GUARDED = False


def _ensure_guarded():
    global _GUARDED
    if not _GUARDED:
        guard()
        _GUARDED = True


def _toy_config():
    # Unit-test coefficients only; these do not select any experiment setting.
    return {
        "action_absolute_weight": 1.0,
        "response_weight": 0.4,
        "latent_absolute_weight": 0.7,
        "control_weight": 0.5,
        "lambda_latent": 0.8,
    }


def _toy_stats():
    return {
        "action": {"mean": [0.0] * 6, "scale": [1.0] * 6},
        "latent": {"mean": [0.0] * 48, "scale": [1.0] * 48},
        "config": {
            "bins": 16, "interior_min": -3.5, "interior_max": 3.5,
            "sigma": 0.375, "std_floor": 0.02,
        },
    }


def _fixture(torch):
    action = torch.zeros(evaluation.ACTION_SHAPE)
    future = torch.zeros(evaluation.FUTURE_SHAPE)
    noise = torch.zeros(evaluation.NOISE_SHAPE)
    identities = {"clean": "input-clean", "noisy": "input-noisy"}
    bindings = {}
    for variant in evaluation.VARIANTS:
        bindings[variant] = {
            "binding": {
                "protocol": "parent-v1", "unit": "training_teacher", "index": 2000,
                "arm": "bf16", "variant": variant, "seed": 910000,
                "description": "toy task", "input_identity": identities[variant],
            },
            "producer_job": "teacher-12.pbs", "source_file": None,
            "source_calibration_job": "calibration-4.pbs",
            "source_kind": "new_query", "source_eligibility": {},
        }
    expected = {
        "protocol": "parent-v1", "unit": "final_c80_candidate_0",
        "arm": "candidate", "index": 2000, "case_id": 2000, "suite": "libero_spatial",
        "deployment_binding": {"arm": "candidate", "producer_job": "train-1.pbs"},
        "teacher_bindings": bindings, "input_identities": identities,
        "sampler_seed": 910000, "description": "toy task",
    }
    record = {
        **copy.deepcopy(expected), "producer_job": "eval-8.pbs",
        "complete": True, "status": "complete",
        "endpoints": {
            variant: {
                "action": action.clone(), "future": future.clone(),
                "action_initial_noise": noise.clone(),
                "counts": dict(evaluation.COUNTS),
                "source": {"kind": "new_endpoint_query", "producer_job": "eval-8.pbs"},
            } for variant in evaluation.VARIANTS
        },
        "reader_actions": {variant: action.clone() for variant in evaluation.VARIANTS},
        "reader_counts": {
            variant: {
                "video_prefill_calls": 1, "action_scheduler_steps": 10,
                "video_scheduler_steps": 10, "teacher_future_injection_steps": 10,
                "teacher_future_injection_verified": True, "future_tail_matches": True,
                "first_frame_clamp_verified": True,
            } for variant in evaluation.VARIANTS
        },
        "raw_terms": {key: 0.0 for key in evaluation.RAW_KEYS},
    }
    return record, expected


class FinalC80EvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_guarded()
        import torch
        cls.torch = torch

    def test_shards_keep_all_80_original_training_cases(self):
        cases = {case_id: {"case_id": case_id, "shard": case_id % 2}
                 for case_id in evaluation.CASE_IDS}
        even = evaluation.case_ids_for_shard(cases, 0)
        odd = evaluation.case_ids_for_shard(cases, 1)
        self.assertEqual((len(even), len(odd)), (40, 40))
        self.assertFalse(set(even) & set(odd))
        self.assertEqual(sorted(even + odd), list(evaluation.CASE_IDS))
        cases.pop(2000)
        with self.assertRaises(ValueError):
            evaluation.case_ids_for_shard(cases, 0)

    def test_record_validates_pair_teacher_identity_shapes_and_raw_terms(self):
        torch = self.torch
        record, expected = _fixture(torch)
        self.assertIs(evaluation.validate_record(torch, record, expected), record)
        core_binding = {key: expected[key] for key in (
            "protocol", "unit", "arm", "index", "case_id", "suite",
            "deployment_binding", "teacher_bindings", "input_identities")}
        self.assertIs(evaluation.validate_record(torch, record, core_binding), record)
        foreign = copy.deepcopy(record)
        foreign["teacher_bindings"]["clean"]["binding"]["input_identity"] = "foreign"
        with self.assertRaises(ValueError):
            evaluation.validate_record(torch, foreign, expected)
        partial = copy.deepcopy(record)
        partial["complete"] = False
        with self.assertRaises(ValueError):
            evaluation.validate_record(torch, partial, expected)
        malformed = copy.deepcopy(record)
        malformed["endpoints"]["noisy"]["future"] = torch.zeros(1)
        with self.assertRaises(ValueError):
            evaluation.validate_record(torch, malformed, expected)
        malformed = copy.deepcopy(record)
        malformed["raw_terms"]["control_kl_clean"] = float("nan")
        with self.assertRaises(ValueError):
            evaluation.validate_record(torch, malformed, expected)
        malformed = copy.deepcopy(record)
        malformed["endpoints"]["clean"]["action_initial_noise"].fill_(1.0)
        with self.assertRaises(ValueError):
            evaluation.validate_record(torch, malformed, expected)

    def test_cached_case_loading_refreshes_description_for_each_prompt(self):
        class FakeRuntime:
            description = "stale"

        class FakeSwitching:
            def __init__(self):
                self.runtime = FakeRuntime()

            def datum(self, observation):
                return observation

        class FakeTeacher:
            @staticmethod
            def _resolve(root, path):
                return path

            @staticmethod
            def _load_npz(np, path):
                return {"identity": path}

        switching = FakeSwitching()
        cases = (
            {"case_id": 2000, "clean_observation": "clean-2000.npz"},
            {"case_id": 2001, "clean_observation": "clean-2001.npz"},
        )
        descriptions = ("first cached prompt", "second cached prompt")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "pairs"
            for case, description in zip(cases, descriptions):
                case_dir = dataset / str(case["case_id"])
                case_dir.mkdir(parents=True)
                (case_dir / "metadata.json").write_text(json.dumps({
                    "description": description,
                    "noisy_observation": f"noisy-{case['case_id']}.npz",
                }), encoding="utf-8")
            identity_module = mock.Mock(datum_identity=lambda torch, datum: datum["identity"])
            with mock.patch.dict(sys.modules, {"diagnostic_runtime": identity_module}):
                for case, description in zip(cases, descriptions):
                    evaluation._load_case(
                        self.torch, object(), switching, FakeTeacher, root,
                        {"dataset_subdir": "pairs"}, case)
                    self.assertEqual(switching.runtime.description, description)

    def test_compute_raw_terms_uses_paired_and_reader_kl_terms(self):
        torch = self.torch
        base_action = torch.zeros(evaluation.ACTION_SHAPE)
        teacher_future = torch.zeros(evaluation.FUTURE_SHAPE)
        targets, endpoints, readers = {}, {}, {}
        for variant, action_value, future_value, reader_value in (
                ("clean", 0.10, 0.02, 0.18), ("noisy", 0.31, 0.04, 0.24)):
            target_action = base_action.clone()
            if variant == "noisy":
                target_action[:10, :6] = 0.04
            target_future = teacher_future.clone()
            student_action = base_action.clone()
            student_action[:10, :6] = action_value
            student_future = teacher_future + future_value
            targets[variant] = {
                "action": target_action, "future": target_future,
                "counts": dict(evaluation.COUNTS),
                "action_initial_noise": torch.zeros(evaluation.NOISE_SHAPE),
                "producer_job": "teacher-12.pbs",
            }
            endpoints[variant] = {"action": student_action, "future": student_future}
            reader = base_action.clone()
            reader[:10, :6] = reader_value
            readers[variant] = reader
        raw = evaluation.compute_raw_terms(
            torch, endpoints, readers, targets, _toy_stats(), _toy_config())
        self.assertEqual(set(raw), evaluation.RAW_KEYS)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in raw.values()))
        self.assertGreater(raw["action_absolute_kl"], 0.0)
        self.assertGreater(raw["action_response_kl"], 0.0)
        self.assertGreater(raw["latent_absolute_kl"], 0.0)
        self.assertGreater(raw["control_kl_clean"], 0.0)
        self.assertGreater(raw["control_kl_noisy"], 0.0)


if __name__ == "__main__":
    _ensure_guarded()
    unittest.main()
