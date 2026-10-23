import unittest

from calibrate import _parity_cache_is_current


class CalibrationParityCacheTest(unittest.TestCase):
    @staticmethod
    def _passed_cache(version=4, differentiable_status="passed"):
        return {
            "status": "passed", "endpoint_version": version,
            "checks": {
                "inference": {"status": "passed"},
                "differentiable": {
                    "status": differentiable_status,
                    "gradient": {
                        "generated_latent_requires_grad": True,
                        "generated_latent_grad_fn": "AddBackward0",
                        "action_requires_grad": True,
                        "action_grad_fn": "MeanBackward1",
                    },
                },
            },
        }

    def test_failed_parity_is_rechecked(self):
        self.assertFalse(_parity_cache_is_current(
            {"status": "failed", "endpoint_version": 4,
             "checks": {"inference": {"status": "passed"},
                        "differentiable": {"status": "passed"}}}, 4
        ))

    def test_passed_old_version_is_rechecked(self):
        self.assertFalse(_parity_cache_is_current(
            {"status": "passed", "endpoint_version": 3}, 4
        ))

    def test_current_passed_parity_is_reused(self):
        self.assertTrue(_parity_cache_is_current(self._passed_cache(), 4))

    def test_missing_differentiable_check_is_rechecked(self):
        parity = self._passed_cache()
        del parity["checks"]["differentiable"]
        self.assertFalse(_parity_cache_is_current(parity, 4))

    def test_failed_differentiable_check_is_rechecked(self):
        self.assertFalse(_parity_cache_is_current(
            self._passed_cache(differentiable_status="failed"), 4
        ))

    def test_differentiable_check_without_gradient_flow_is_rechecked(self):
        parity = self._passed_cache()
        parity["checks"]["differentiable"]["gradient"]["action_grad_fn"] = None
        self.assertFalse(_parity_cache_is_current(parity, 4))


if __name__ == "__main__":
    unittest.main()
