import unittest
from unittest.mock import patch

import numpy as np

import runner


class _FakeTensor:
    """Small tensor-shaped mock sufficient for the rollout boundary."""

    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    def clone(self):
        return _FakeTensor(self.value.copy())

    def __getitem__(self, key):
        return _FakeTensor(self.value[key])

    def to(self, **_kwargs):
        return self

    def unsqueeze(self, dimension):
        return _FakeTensor(np.expand_dims(self.value, axis=dimension))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _FakeTorch:
    def no_grad(self):
        return _FakeNoGrad()


class _RecordingActionBlocks:
    """Five explicit blocks, with access recording for propagation slices."""

    def __init__(self):
        self.rows = np.arange(5, dtype=np.float32).reshape(5, 1)
        self.accesses = []

    def __getitem__(self, key):
        self.accesses.append(key)
        return _FakeTensor(self.rows[key])


class _FakeModel:
    def __init__(self):
        self.replace_calls = []

    def replace_actions_from_z(self, next_z, next_action):
        self.replace_calls.append(next_action.value.copy())
        return next_z


class RolloutArmContractTest(unittest.TestCase):
    def test_requested_horizons_draws_and_action_blocks(self):
        fake_torch = _FakeTorch()
        z0 = _FakeTensor(np.zeros((1, 1, runner.PATCHES, runner.VISUAL_DIM + runner.TAIL_DIM), dtype=np.float32))
        action_blocks = _RecordingActionBlocks()
        model = _FakeModel()
        call_labels = []

        def fake_predict_once(_model, _current_z, _quantized_tail, label, _torch, _device):
            call_labels.append(label)
            prediction = np.full(
                (1, 1, runner.PATCHES, runner.VISUAL_DIM + runner.TAIL_DIM),
                len(call_labels),
                dtype=np.float32,
            )
            return _FakeTensor(prediction), {
                "hook_readback": True,
                "input_unchanged": True,
                "output_shape": list(prediction.shape),
            }

        expected_draws = {
            "FP32": 1,
            "RTN": 1,
            "shared_stochastic": 3,
            "iid_patch_stochastic": 3,
            "balanced_broadcast": 3,
        }
        for arm, n_draws in expected_draws.items():
            treatment = {
                "values": np.zeros(
                    (n_draws, runner.PATCHES, runner.TAIL_DIM), dtype=np.float32
                ),
                "audit": {"arm": arm, "n_draws": n_draws},
            }
            with patch.object(runner, "_predict_once", side_effect=fake_predict_once):
                outputs, audit = runner._rollout_arm(
                    model,
                    z0,
                    action_blocks,
                    arm,
                    treatment,
                    max_horizon=5,
                    torch=fake_torch,
                    device="cpu",
                )

            # HORIZONS is frozen to (1, 5): steps 2/3/4 are traversed but
            # must never become output dictionary keys.
            self.assertEqual(tuple(runner.HORIZONS), (1, 5))
            self.assertEqual(set(outputs), {1, 5})
            self.assertEqual([len(outputs[h]) for h in runner.HORIZONS], [n_draws, n_draws])
            self.assertEqual(audit["quantized_calls"], n_draws)
            self.assertEqual(audit["propagation_calls"], n_draws * 4)
            self.assertEqual(audit["calls"], n_draws * 5)

            expected_labels = []
            for draw in range(n_draws):
                expected_labels.extend(
                    [f"{arm}:initial_draw_{draw}"]
                    + [f"{arm}:propagation_step_{step}_draw_{draw}" for step in range(1, 5)]
                )
            self.assertEqual(call_labels[-n_draws * 5 :], expected_labels)

        # Block 0 is the initial-action slot consumed before _rollout_arm;
        # propagation must use the remaining explicit blocks 1, 2, 3, 4.
        np.testing.assert_array_equal(action_blocks.rows[:, 0], np.arange(5, dtype=np.float32))
        self.assertEqual(
            action_blocks.accesses,
            [slice(index, index + 1) for index in (1, 2, 3, 4)] * 11,
        )
        flattened_actions = [int(call.reshape(-1)[0]) for call in model.replace_calls]
        self.assertEqual(flattened_actions, [1, 2, 3, 4] * 11)


if __name__ == "__main__":
    unittest.main()
