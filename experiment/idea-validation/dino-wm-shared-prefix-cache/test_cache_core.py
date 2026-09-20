"""Lightweight CPU tests for the shared-prefix rollout helpers."""

from __future__ import annotations

import unittest

import torch

from cache_core import (
    encode_and_repeat,
    rollout_from_encoded_obs,
    rollout_with_cache,
)


class DummyModel:
    """Deterministic concat_dim=0 model with the VWorldModel rollout contract."""

    concat_dim = 0
    num_hist = 2
    num_proprio_repeat = 1
    num_action_repeat = 1

    def __init__(self) -> None:
        self.encode_obs_calls = 0
        self.encode_act_calls = 0

    def encode_obs(self, obs):
        self.encode_obs_calls += 1
        # Keep visual/proprio shapes compatible with VWorldModel.encode_obs.
        return {
            "visual": obs["visual"] + 0.25,
            "proprio": obs["proprio"] * 2.0,
        }

    def encode_act(self, act):
        self.encode_act_calls += 1
        return act + 0.5

    def encode(self, obs, act):
        encoded = self.encode_obs(obs)
        return torch.cat(
            [
                encoded["visual"],
                encoded["proprio"].unsqueeze(2),
                self.encode_act(act).unsqueeze(2),
            ],
            dim=2,
        )

    def predict(self, z):
        # A deterministic suffix that preserves all dimensions.
        return z + 0.125

    def replace_actions_from_z(self, z, act):
        z[..., -1:, :] = self.encode_act(act).unsqueeze(2)
        return z

    def separate_emb(self, z):
        return {
            "visual": z[:, :, :-2, :],
            "proprio": z[:, :, -2, :],
        }, z[:, :, -1:, :]

    def rollout(self, obs_0, act):
        num_obs_init = obs_0["visual"].shape[1]
        action = act[:, num_obs_init:]
        z = self.encode(obs_0, act[:, :num_obs_init])
        for step in range(action.shape[1]):
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = self.replace_actions_from_z(
                z_pred[:, -1:, ...], action[:, step : step + 1]
            )
            z = torch.cat([z, z_new], dim=1)
        z = torch.cat([z, self.predict(z[:, -self.num_hist :])[:, -1:]], dim=1)
        z_obses, _ = self.separate_emb(z)
        return z_obses, z


class CacheCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.obs = {
            "visual": torch.randn(1, 2, 3, 4),
            "proprio": torch.randn(1, 2, 4),
        }
        self.act = torch.randn(1, 5, 4)

    def test_cached_rollout_matches_baseline(self) -> None:
        baseline_model = DummyModel()
        baseline_obses, baseline_z = baseline_model.rollout(self.obs, self.act)

        cached_model = DummyModel()
        encoded = encode_and_repeat(
            cached_model, self.obs, mode="iteration_cache"
        )
        cached_obses, cached_z = rollout_from_encoded_obs(
            cached_model, encoded, self.act
        )

        self.assertTrue(torch.equal(baseline_z, cached_z))
        self.assertTrue(torch.equal(baseline_obses["visual"], cached_obses["visual"]))
        self.assertTrue(
            torch.equal(baseline_obses["proprio"], cached_obses["proprio"])
        )

    def test_factorized_shape_and_no_prefix_mutation(self) -> None:
        repeats = 4
        model = DummyModel()
        original = {
            key: value.clone() for key, value in self.obs.items()
        }
        prefix = encode_and_repeat(
            model, self.obs, mode="factorized_cache", repeats=repeats
        )
        self.assertEqual(prefix["visual"].shape[0], repeats)
        self.assertEqual(prefix["proprio"].shape[0], repeats)
        self.assertTrue(torch.equal(self.obs["visual"], original["visual"]))
        self.assertTrue(torch.equal(self.obs["proprio"], original["proprio"]))

        # The repeated view is not altered by the action-conditioned rollout.
        prefix_before = {key: value.clone() for key, value in prefix.items()}
        actions = self.act.expand(repeats, -1, -1).clone()
        cached_obses, cached_z = rollout_from_encoded_obs(model, prefix, actions)
        for key in prefix:
            self.assertTrue(torch.equal(prefix[key], prefix_before[key]))

        baseline_model = DummyModel()
        baseline_obs = {
            key: value.expand((repeats,) + tuple(value.shape[1:]))
            for key, value in self.obs.items()
        }
        baseline_obses, baseline_z = baseline_model.rollout(baseline_obs, actions)
        self.assertTrue(torch.equal(cached_z, baseline_z))
        self.assertTrue(
            torch.equal(cached_obses["visual"], baseline_obses["visual"])
        )
        self.assertTrue(
            torch.equal(cached_obses["proprio"], baseline_obses["proprio"])
        )

    def test_mode_interfaces_have_expected_encode_reuse(self) -> None:
        repeats = 3
        repeated_obs = {
            key: value.expand((repeats,) + tuple(value.shape[1:]))
            for key, value in self.obs.items()
        }
        actions = self.act.expand(repeats, -1, -1).clone()

        model = DummyModel()
        cached_prefix = encode_and_repeat(
            model, repeated_obs, mode="iteration_cache"
        )
        self.assertEqual(model.encode_obs_calls, 1)
        rollout_with_cache(
            model,
            repeated_obs,
            actions,
            mode="iteration_cache",
            cached_prefix=cached_prefix,
        )
        self.assertEqual(model.encode_obs_calls, 1)

        factorized_model = DummyModel()
        rollout_with_cache(
            factorized_model,
            repeated_obs,
            actions,
            mode="factorized_cache",
        )
        self.assertEqual(factorized_model.encode_obs_calls, 1)


if __name__ == "__main__":
    unittest.main()
