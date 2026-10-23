"""CPU-only toy checks for pre-transform capture and frozen query bindings."""
import types
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from diagnostic_runtime import (NoiseTap, _source_protocol_and_deployment,
                                install_compensated_from_base,
                                max_abs_difference, require_exact_first_chunk,
                                validate_binding)
from sensor_worker import _assert_frozen_seeds, _frozen_noise_level, _finish_episode, _sensor_pair_run, _rng_changes, _capture_run


class _Image:
    def __init__(self, values):
        self.values = list(values)

    def copy(self):
        return _Image(self.values)


class _Array:
    @staticmethod
    def array(value, copy=False):
        values = value.values if isinstance(value, _Image) else value
        return list(values) if copy or isinstance(values, list) else values


class _Scalar:
    def __init__(self, value, device="cpu"):
        self.value, self.device = value, device

    def to(self, device=None):
        return _Scalar(self.value, device or self.device)


class _Module:
    def __init__(self):
        self.weight = types.SimpleNamespace(data=_Scalar(0.0))


class SensorToyTests(unittest.TestCase):
    def test_failed_capture_keeps_attempted_queries_and_closes_runtime(self):
        from unittest.mock import Mock
        case = {'index': 61, 'ordinal': 0, 'capture_shard': 0,
                'original_success': {'per_channel_clip': True, 'relational': False}}
        protocol = {'protocol': 'toy', 'source_closed_job': 'source.pbs',
                    'resource': {'gpu_soft_deadline_seconds': 6600}}
        runtime, tap = Mock(), Mock()
        tap.install.return_value = tap
        def fail_after_query(*args):
            args[-1][0] += 1
            raise RuntimeError('metadata failed after native query')
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict('sys.modules', {'numpy': types.SimpleNamespace(), 'torch': types.SimpleNamespace()}), \
             patch('sensor_worker.SwitchingRuntime', return_value=runtime), \
             patch('sensor_worker.NoiseTap', return_value=tap), \
             patch('sensor_worker._noise_module'), patch('sensor_worker._sha256_file', return_value='toy'), \
             patch('sensor_worker.Deadline') as deadline, \
             patch('sensor_worker.read', return_value={'per_channel_clip': {'steps': 10}}), \
             patch('sensor_worker._capture_episode', side_effect=fail_after_query), \
             patch('sensor_worker.save') as save, patch.dict('os.environ', {'PBS_JOBID': 'toy.pbs'}):
            deadline.return_value.near.return_value = False
            with self.assertRaisesRegex(RuntimeError, 'metadata failed'):
                _capture_run(protocol, 'capture_mse_0', {'discordant_cases': [case]}, Path(directory))
        summary = save.call_args.args[1]
        self.assertEqual(summary['new_queries'], 1)
        self.assertEqual(summary['failed_index'], 61)
        self.assertFalse(summary['complete'])
        self.assertFalse(summary['source_replay_passed_all'])
        runtime.close.assert_called_once()
        tap.uninstall.assert_called_once()

    def test_serialized_rng_changes_are_reported_without_mutating_states(self):
        from copy import deepcopy
        before = {'numpy': {'kind': 'MT19937', 'keys': [1, 2, 3], 'position': 2,
                            'has_gauss': 0, 'cached_gaussian': 0.0},
                  'python': (3, (1, 2), None), 'torch_cpu': [4, 5],
                  'torch_cuda': [[6, 7]]}
        snapshot = deepcopy(before)
        torch = types.SimpleNamespace(equal=lambda left, right: left == right)
        unchanged = {k: False for k in ('python_changed', 'numpy_changed',
                                      'torch_cpu_changed', 'torch_cuda_changed')}
        self.assertEqual(_rng_changes(torch, before, deepcopy(before)), unchanged)
        for field in ('numpy', 'python', 'torch_cpu', 'torch_cuda'):
            after = deepcopy(before)
            if field == 'numpy':
                after[field]['keys'][0] += 1
            elif field == 'python':
                after[field] = (3, (1, 3), None)
            elif field == 'torch_cpu':
                after[field][0] += 1
            else:
                after[field][0][0] += 1
            self.assertEqual(_rng_changes(torch, before, after),
                             {**unchanged, field + '_changed': True})
        none_state = {**before, 'numpy': None}
        self.assertEqual(_rng_changes(torch, none_state, deepcopy(none_state)), unchanged)
        self.assertEqual(before, snapshot)

    def test_batch_reuses_runtime_and_stops_at_incomplete_gate_or_deadline(self):
        from unittest.mock import Mock
        p = {'protocol': 'toy', 'resource': {'gpu_soft_deadline_seconds': 6600}}
        for complete, near, expected in ((True, False, 2), (False, False, 1), (True, True, 1)):
            runtime = Mock()
            with patch('diagnostic_runtime.SwitchingRuntime', return_value=runtime) as factory, \
                 patch('sensor_worker.Deadline') as deadline, \
                 patch('sensor_worker._sensor_run', return_value={'complete': complete, 'completed': 20 if complete else 3}) as worker, \
                 patch('sensor_worker.save'), patch.dict('os.environ', {'PBS_JOBID': 'test.pbs'}):
                deadline.return_value.near.return_value = near
                result = _sensor_pair_run(p, {}, Path('toy'))
            factory.assert_called_once()
            self.assertEqual(worker.call_count, expected)
            self.assertTrue(all(call.kwargs['runtime'] is runtime for call in worker.call_args_list))
            runtime.close.assert_called_once()
            self.assertEqual(result['complete'], expected == 2)

    def test_episode_completion_is_preserved_in_summary_row(self):
        case = {'original_success': {'relational': False}}
        for complete in (False, True):
            with patch('sensor_worker._save_torch'), patch('sensor_worker.save'):
                row = _finish_episode(None, Path('episode.pt'), Path('episode.json'), {},
                    61, 'complete' if complete else 'first_chunk_parity_failed',
                    complete, 400, False, 0., [], {}, 0, 400, case, 'relational')
            self.assertIs(row['episode_complete'], complete)

    def test_noise_hook_calls_original_once_and_returns_exact_object(self):
        calls = []
        output = object()

        def transform(image, severity=1):
            calls.append((image, severity))
            return output

        module = types.SimpleNamespace(**{
            name: transform for name in NoiseTap.FUNCTIONS
        })
        tap = NoiseTap().install(module)
        source = _Image([1, 2, 3])
        returned = module.motion_blur(source, severity=4)
        self.assertIs(returned, output)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], source)
        self.assertEqual(calls[0][1], 4)
        self.assertEqual(tap.calls, 1)
        self.assertEqual(tap.last_clean.values, [1, 2, 3])
        self.assertIsNot(tap.last_clean, source)
        tap.uninstall()

    def test_clean_observation_replaces_only_agentview(self):
        tap = NoiseTap()
        tap.last_clean = _Image([1, 2])
        wrist = [7, 8]
        proprio = [9]
        noisy = {"agentview_image": [20, 21],
                 "robot0_eye_in_hand_image": wrist,
                 "robot0_proprio": proprio}
        clean = tap.clean_observation(_Array, noisy)
        self.assertIsNot(clean, noisy)
        self.assertEqual(clean["agentview_image"], [1, 2])
        self.assertIs(clean["robot0_eye_in_hand_image"], wrist)
        self.assertIs(clean["robot0_proprio"], proprio)
        self.assertEqual(noisy["agentview_image"], [20, 21])

    def test_cached_compensation_always_starts_from_base(self):
        module = _Module()
        base = {"toy": _Scalar(3.0)}
        scales = {"toy": _Scalar(1.0)}

        def compensate(weight, log2_scale):
            return _Scalar(weight.value * (2 ** log2_scale.value))

        for _ in range(2):
            install_compensated_from_base({"toy": module}, base, scales, compensate)
        self.assertEqual(module.weight.data.value, 6.0)
        self.assertEqual(base["toy"].value, 3.0)

    def test_query_binding_includes_seed_and_variant(self):
        binding = {"protocol": "toy", "unit": "sensor_0", "index": 3,
                   "arm": "bf16", "variant": "clean", "seed": 9,
                   "description": "task", "input_identity": {"image": "x"}}
        self.assertTrue(validate_binding(dict(binding), binding))
        changed = dict(binding, seed=10)
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_binding(changed, binding)

    def test_first_chunk_mismatch_is_rejected(self):
        self.assertEqual(require_exact_first_chunk([[1, 2], [3, 4]], [[1, 2], [3, 4]]), 0.0)
        self.assertEqual(max_abs_difference([[1, 2]], [[1, 2.25]]), 0.25)
        with self.assertRaisesRegex(RuntimeError, "max_abs=0.25"):
            require_exact_first_chunk([[1, 2]], [[1, 2.25]])

    def test_frozen_task_noise_suffix_and_seed_binding(self):
        case = {"index": 61, "environment_seed": 800061000,
                "sampler_seed": 900061000,
                "row": {"task_name": "pick_place_initstate_0_noise_15"}}
        self.assertEqual(_frozen_noise_level(case), 15)
        _assert_frozen_seeds(case)
        for suffix in ("noise_0", "noise_51", "noise_x"):
            case["row"]["task_name"] = f"task_{suffix}"
            with self.assertRaises(RuntimeError):
                _frozen_noise_level(case)

    def test_per_channel_uses_training_protocol_and_unitless_deployment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "evaluation" / "results"
            training = root / "training"
            results.mkdir(parents=True)
            training.mkdir()
            training_protocol = {"protocol": "training-v1", "marker": "source"}
            (training / "protocol.json").write_text(json.dumps(training_protocol), encoding="utf-8")
            evaluation_protocol = {
                "protocol": "evaluation-v1",
                "sources": {
                    "training_root": str(training),
                    "training_protocol": "training-v1",
                    "deployment": str(training / "deployment.pt"),
                },
            }
            (results.parent / "protocol.json").write_text(
                json.dumps(evaluation_protocol), encoding="utf-8")
            deployment = {
                "protocol": training_protocol,
                "updates": 80,
                "baseline_checkpoint": "baseline.pt",
            }
            fake_torch = types.SimpleNamespace(load=lambda *args, **kwargs: deployment)
            protocol = {
                "source_closed_protocol": {
                    "references": {"per_channel_results": str(results)},
                    "sources": {"baseline_deployment": "baseline.pt"},
                },
            }
            with patch.dict("sys.modules", {"torch": fake_torch}):
                loaded, source, path = _source_protocol_and_deployment(
                    protocol, "per_channel_clip")
            self.assertEqual(loaded, deployment)
            self.assertEqual(source, training_protocol)
            self.assertEqual(path, str(training / "deployment.pt"))


if __name__ == "__main__":
    unittest.main(buffer=True)
