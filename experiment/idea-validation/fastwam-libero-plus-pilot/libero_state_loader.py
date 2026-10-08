"""Load trusted official Plus initial states with the upstream path/reshape logic."""
import os
from pathlib import Path


def ensure_numpy_compat():
    import numpy as np

    applied = 'float_' not in np.__dict__
    if applied:
        np.float_ = np.float64
    if np.float_ is not np.float64:
        raise RuntimeError('Plus np.float_ must retain its original float64 dtype')
    return {'numpy': np.__version__, 'float_alias': 'float64', 'alias_added': applied}


def load_task_init_states(suite, task_id):
    ensure_numpy_compat()
    import torch
    from libero.libero import get_libero_path

    root = Path(os.environ["ROOT"]) / "LIBERO-plus/libero/libero/init_files"
    if Path(get_libero_path("init_states")).resolve() != root.resolve():
        raise RuntimeError("Initial states must come from the frozen official Plus source")
    original_load = torch.load
    loaded_paths = []

    def load_official_state(file, *args, **kwargs):
        path = Path(file).resolve()
        path.relative_to(root.resolve())
        if not path.is_file():
            raise FileNotFoundError(path)
        loaded_paths.append(str(path))
        kwargs["weights_only"] = False
        return original_load(file, *args, **kwargs)

    # The evaluation is single-threaded; restore immediately after this one call.
    torch.load = load_official_state
    try:
        states = suite.get_task_init_states(int(task_id))
    finally:
        torch.load = original_load
    if len(loaded_paths) != 1:
        raise RuntimeError(f"Expected one official initial-state load, got {loaded_paths}")
    return states, loaded_paths[0]
