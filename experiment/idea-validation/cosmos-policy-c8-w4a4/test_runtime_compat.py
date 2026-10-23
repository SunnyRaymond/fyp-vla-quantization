"""CPU-only regression for the frozen LIBERO-Plus noise_35 environment path."""
from __future__ import annotations

import json
import os
import random
import unittest
from pathlib import Path

import cosmos_runtime
from evaluate import environment_seed, load_manifest


class LiberoNumpyCompatibilityTests(unittest.TestCase):
    def test_noise_35_reset_and_step_with_numpy2_alias(self):
        root = Path(os.environ["ROOT"])
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES", ""), "")
        protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        _, variants = load_manifest(protocol["cohort"]["manifest"], protocol)
        index = 270
        row = variants[index]
        self.assertEqual(
            (row["suite"], int(row["task_id"]), row["dimension"], row["subtype"]),
            ("libero_10", 1693, "sensor_noise", "noise"),
        )

        import numpy as np

        if "float_" in np.__dict__:
            delattr(np, "float_")
        compatibility = cosmos_runtime._ensure_libero_numpy_compat()
        self.assertIs(np.float_, np.float64)

        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        import libero.libero.envs.env_wrapper as noise_module

        plus_source = (
            root.parent / "fastwam-libero-plus-pilot-20261005" / "LIBERO-plus"
            / "libero" / "libero"
        ).resolve()
        for source in (benchmark.__file__, noise_module.__file__):
            Path(source).resolve().relative_to(plus_source)

        suite = benchmark.get_benchmark_dict()[row["suite"]]()
        task = suite.get_task(int(row["task_id"]))
        self.assertEqual(task.name, row["task_name"])
        env = OffScreenRenderEnv(
            bddl_file_name=str(suite.get_task_bddl_file_path(int(row["task_id"]))),
            camera_heights=256,
            camera_widths=256,
        )
        original_fog = noise_module.fog
        original_plasma_fractal = noise_module.plasma_fractal
        observed = {"fog_severities": [], "plasma_fractal_calls": 0}

        def observed_fog(image, severity=1):
            observed["fog_severities"].append(int(severity))
            return original_fog(image, severity=severity)

        def observed_plasma_fractal(*args, **kwargs):
            observed["plasma_fractal_calls"] += 1
            return original_plasma_fractal(*args, **kwargs)

        noise_module.fog = observed_fog
        noise_module.plasma_fractal = observed_plasma_fractal
        try:
            self.assertEqual(int(env.noise), 35)
            seed = int(environment_seed(index, protocol))
            env.seed(seed)
            np.random.seed(seed % (2**32))
            random.seed(seed)
            env.reset()
            reset_fog_calls = len(observed["fog_severities"])
            self.assertGreater(reset_fog_calls, 0)
            state_runtime = cosmos_runtime.CosmosRuntime.__new__(cosmos_runtime.CosmosRuntime)
            state_runtime.root = root
            state_runtime.environment_ready = json.loads(
                (root / "environment_ready.json").read_text(encoding="utf-8")
            )
            states, state_path = state_runtime._initial_states(suite, int(row["task_id"]), False)
            self.assertGreater(len(states), 0)
            env.set_init_state(states[0])
            obs, _, _, _ = cosmos_runtime.CosmosRuntime._step(
                env, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            )
            self.assertEqual(tuple(obs["agentview_image"].shape), (256, 256, 3))
            self.assertIn(5, observed["fog_severities"])
            self.assertGreater(len(observed["fog_severities"]), reset_fog_calls)
            self.assertGreater(observed["plasma_fractal_calls"], 0)
        finally:
            noise_module.fog = original_fog
            noise_module.plasma_fractal = original_plasma_fractal
            env.close()

        receipt = {
            "status": "passed",
            "pbs_job_id": os.environ["PBS_JOBID"],
            "index": index,
            "variant_id": row["variant_id"],
            "task_id": int(row["task_id"]),
            "noise_level": 35,
            "fog_severity": 5,
            "fog_reset_and_step": True,
            "plasma_fractal_calls": observed["plasma_fractal_calls"],
            "initial_state_path": state_path,
            "numpy_compatibility": compatibility,
            "model_loads": 0,
            "gpu_forwards": 0,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
        receipt_path = root / "results" / "checks" / "libero_noise_compatibility.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
