"""Check actual LIBERO Goal observations before loading model weights."""
import json
import os
import time
import sys
import importlib.util
from pathlib import Path
import ctypes
import ctypes.util

print("OSMesa library:", ctypes.util.find_library("OSMesa"), flush=True)
ctypes.CDLL("libOSMesa.so.8")
print("OSMesa library loaded", flush=True)

import numpy as np
import torch
from PIL import Image
from libero.libero import benchmark, get_libero_path
from experiments.libero.libero_utils import get_libero_env, get_libero_dummy_action

root = Path(__file__).resolve().parent
entry = root / "FastWAM/experiments/libero/eval_libero_single.py"
sys.path.insert(0, str(entry.parent))
spec = importlib.util.spec_from_file_location("fastwam_eval_preflight", entry)
entry_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry_module)
print("full_eval_entry_import=ok", flush=True)

out = Path(os.environ["ARTIFACTS"])
suite = benchmark.get_benchmark_dict()["libero_goal"]()
task = suite.get_task(0)
start = time.monotonic()
env, description = get_libero_env(task, 256, 0)
try:
    env.reset()
    # Match pinned FastWAM's explicit loading of the trusted LIBERO initial states.
    initial_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    initial_states = torch.load(initial_path, weights_only=False)
    obs = env.set_init_state(initial_states[0])
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action())
    cameras = {}
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        frame = np.asarray(obs[key])
        assert frame.shape == (256, 256, 3), (key, frame.shape)
        assert np.isfinite(frame).all() and frame.std() > 1, key
        Image.fromarray(frame[::-1, ::-1]).save(out / f"{key}.png")
        cameras[key] = {"shape": list(frame.shape), "std": float(frame.std())}
    result = {"render_backend": os.environ.get("MUJOCO_GL"),
              "task": description, "cameras": cameras,
              "elapsed_seconds": time.monotonic() - start}
    (out / "render-preflight.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
finally:
    env.close()
