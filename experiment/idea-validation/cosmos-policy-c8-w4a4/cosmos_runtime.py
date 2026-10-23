"""Small adapter around the pinned NVIDIA Cosmos Policy LIBERO evaluator."""
from __future__ import annotations

import importlib
import json
import os
import random
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any


ARMS = ("bf16", "spinquant_w4a4", "action_latent_micro_vq_a4")
COSMOS_REVISION = "18a2accadf4e7a3531e56754102af5a24d2316da"
_REGISTRATION_DEFAULT_CHECKPOINT_URIS = frozenset({
    "hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt",
    "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt",
})


def _ensure_libero_numpy_compat():
    """Restore the float64 alias used by the frozen LIBERO-Plus noise wrapper."""
    import numpy as np

    alias_added = "float_" not in np.__dict__
    if alias_added:
        np.float_ = np.float64
    if np.float_ is not np.float64:
        raise RuntimeError("LIBERO-Plus np.float_ must retain its float64 meaning")
    return {"numpy": np.__version__, "float_alias": "float64", "alias_added": alias_added}


def guard(root: str | Path | None = None) -> None:
    """Require a real PBS allocation whose nodefile contains this host."""
    job_id = os.environ.get("PBS_JOBID")
    nodefile = os.environ.get("PBS_NODEFILE")
    host = socket.gethostname().split(".")[0]
    if not job_id or not nodefile or "login" in host.lower():
        raise RuntimeError("Cosmos model/environment work requires an allocated PBS compute node")
    nodes_path = Path(nodefile)
    if not nodes_path.is_file():
        raise RuntimeError(f"PBS_NODEFILE is missing: {nodes_path}")
    nodes = {node.split(".")[0] for node in nodes_path.read_text(encoding="utf-8").split()}
    if host not in nodes:
        raise RuntimeError(f"Current host {host} is not listed in PBS_NODEFILE")
    if root is not None and os.environ.get("ROOT"):
        if Path(os.environ["ROOT"]).resolve() != Path(root).resolve():
            raise RuntimeError("Runtime root differs from the PBS ROOT environment")


def _cpu_tree(value: Any, torch_module):
    """Copy tensor-like leaves to independent CPU tensors for calibration capture."""
    if isinstance(value, torch_module.Tensor):
        # Official video is channels_last_3d after its BTHWC -> BCTHW transpose.
        # Preserve that layout through capture so VAE replay uses the same path.
        return value.detach().to("cpu").clone(memory_format=torch_module.preserve_format)
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return torch_module.from_numpy(value.copy())
    except ImportError:
        pass
    if isinstance(value, dict):
        return {key: _cpu_tree(item, torch_module) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item, torch_module) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item, torch_module) for item in value)
    return value


def _jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class _BatchCaptured(Exception):
    """Internal sentinel used to stop official get_action before diffusion."""


def _finalize_termination(success: bool, termination: str, steps: int, cap: int) -> str:
    """Preserve explicit environment/deadline stops; otherwise classify loop exit."""
    if not success and termination not in ("soft_deadline", "environment_terminated"):
        return "control_step_timeout" if steps >= cap else "stopped"
    return termination


def _episode_is_complete(termination: str, steps: int, next_observations: int, query_count: int) -> bool:
    """An environment-ended episode is complete even when the task was unsuccessful."""
    return bool(
        termination in ("success", "control_step_timeout", "environment_terminated")
        and steps > 0 and next_observations >= steps and query_count > 0
    )


def _config_file_relative_to_repo(repo: str | Path) -> str:
    """Return the official config path in the relative module form expected by Cosmos."""
    repo = Path(repo)
    config_file = repo / "cosmos_policy" / "config" / "config.py"
    if not config_file.is_file():
        raise FileNotFoundError(f"Pinned Cosmos config is missing: {config_file}")
    return config_file.relative_to(repo).as_posix()


@contextmanager
def _defer_registration_checkpoint_downloads(checkpoint_db):
    """Defer only unused checkpoint URIs evaluated while Cosmos registers configs.

    The registration module imports ``get_checkpoint_path`` by value, so the
    temporary wrapper remains reachable after this context exits. Its active
    flag makes later calls delegate to the original resolver normally.
    """
    original = checkpoint_db.get_checkpoint_path
    active = True

    def resolve(checkpoint_uri, *args, **kwargs):
        if active and str(checkpoint_uri) in _REGISTRATION_DEFAULT_CHECKPOINT_URIS:
            return checkpoint_uri
        return original(checkpoint_uri, *args, **kwargs)

    checkpoint_db.get_checkpoint_path = resolve
    try:
        yield
    finally:
        active = False
        checkpoint_db.get_checkpoint_path = original


def _get_model_with_registration_downloads_deferred(cosmos_utils, cfg):
    """Call the official loader, deferring only its registration-time defaults."""
    loader = importlib.import_module("cosmos_policy._src.predict2.utils.model_loader")
    config_module_name = loader.get_config_module(cfg.config_file)
    config_module = importlib.import_module(config_module_name)
    original_make_config = config_module.make_config
    checkpoint_db = importlib.import_module(
        "cosmos_policy._src.imaginaire.utils.checkpoint_db"
    )

    def make_config_with_deferred_registration_defaults(*args, **kwargs):
        with _defer_registration_checkpoint_downloads(checkpoint_db):
            return original_make_config(*args, **kwargs)

    config_module.make_config = make_config_with_deferred_registration_defaults
    try:
        return cosmos_utils.get_model(cfg)
    finally:
        config_module.make_config = original_make_config


class CosmosRuntime:
    """Load one official Cosmos model and reuse it across episodes and arms."""

    def __init__(self, protocol, root, out, arm="bf16", *, text_embeddings_path=None):
        guard(root)
        self.protocol = json.loads(Path(protocol).read_text(encoding="utf-8")) if isinstance(
            protocol, (str, Path)
        ) else protocol
        self.root = Path(root).resolve()
        self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        ready_path = self.root / "environment_ready.json"
        assets_path = self.root / "asset_paths.json"
        if not ready_path.is_file() or not assets_path.is_file():
            raise RuntimeError("Cosmos CPU preparation is incomplete; environment_ready.json and asset_paths.json are required")
        self.environment_ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if self.environment_ready.get("status") != "ready_for_gpu_smoke":
            raise RuntimeError(f"Cosmos environment is not GPU-ready: {self.environment_ready.get('status')!r}")
        if self.environment_ready.get("source_revision") != COSMOS_REVISION:
            raise RuntimeError("Cosmos source revision differs from the frozen protocol revision")
        self.assets = json.loads(assets_path.read_text(encoding="utf-8"))
        self.repo = self.root / "vendor" / "cosmos-policy"
        if not self.repo.is_dir():
            raise RuntimeError(f"Pinned Cosmos source checkout is missing: {self.repo}")
        sys.path.insert(0, str(self.repo))

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("A GPU allocation is required for Cosmos model loading")
        self.torch = torch
        os.environ["DETERMINISTIC"] = "True"

        self.run_libero_eval = importlib.import_module(
            "cosmos_policy.experiments.robot.libero.run_libero_eval"
        )
        self.cosmos_utils = importlib.import_module("cosmos_policy.experiments.robot.cosmos_utils")
        self.libero_utils = importlib.import_module(
            "cosmos_policy.experiments.robot.libero.libero_utils"
        )
        self.benchmark = importlib.import_module("libero.libero.benchmark")

        inference = self.protocol["inference"]
        snapshot = Path(self.assets["policy_snapshot"]).resolve()
        stats_path = snapshot / "libero_dataset_statistics.json"
        if not stats_path.is_file() and (snapshot.parent / "libero_dataset_statistics.json").is_file():
            stats_path = snapshot.parent / "libero_dataset_statistics.json"
        if snapshot.is_file():
            checkpoint_path = snapshot
        elif (snapshot / "model").is_dir():
            checkpoint_path = snapshot / "model"
        else:
            pt_files = sorted(snapshot.glob("*.pt")) if snapshot.is_dir() else []
            if len(pt_files) > 1:
                raise RuntimeError(f"Policy snapshot has ambiguous .pt checkpoints: {[p.name for p in pt_files]}")
            checkpoint_path = pt_files[0] if pt_files else snapshot
        # C8 uses the official read-only cache; closed-loop evaluation uses the merged Plus cache.
        embeddings_path = (Path(text_embeddings_path) if text_embeddings_path is not None
                           else self.root / "results" / "text_embeddings.pkl").resolve()
        self.text_embeddings_path = embeddings_path
        if not stats_path.is_file() or not embeddings_path.is_file():
            raise RuntimeError("Frozen dataset statistics or precomputed language embeddings are missing")

        cfg_type = self.run_libero_eval.PolicyEvalConfig
        self.cfg = cfg_type(
            config=inference["config"],
            ckpt_path=str(checkpoint_path),
            # Pinned config_helper maps this path directly to an importable module name.
            config_file=_config_file_relative_to_repo(self.repo),
            suite="libero",
            use_third_person_image=True,
            num_third_person_images=1,
            use_wrist_image=bool(inference["use_wrist_image"]),
            num_wrist_images=1,
            use_proprio=bool(inference["use_proprio"]),
            flip_images=bool(inference["flip_images"]),
            use_jpeg_compression=bool(inference["use_jpeg_compression"]),
            num_denoising_steps_action=int(inference["denoising_steps"]),
            unnormalize_actions=bool(inference["unnormalize_actions"]),
            normalize_proprio=bool(inference["normalize_proprio"]),
            dataset_stats_path=str(stats_path),
            t5_text_embeddings_path=str(embeddings_path),
            trained_with_image_aug=bool(inference["trained_with_image_aug"]),
            chunk_size=int(inference["chunk_size"]),
            num_open_loop_steps=int(inference["execute_per_replan"]),
            deterministic=bool(inference["deterministic"]),
            deterministic_reset=False,
            randomize_seed=False,
            use_variance_scale=bool(inference["use_variance_scale"]),
            task_suite_name="libero_goal",
            env_img_res=256,
            seed=7,
            data_collection=False,
        )
        self.cosmos_utils.init_t5_text_embeddings_cache(str(embeddings_path))
        self.dataset_stats = self.cosmos_utils.load_dataset_stats(str(stats_path))
        self.model, cosmos_config = _get_model_with_registration_downloads_deferred(
            self.cosmos_utils, self.cfg
        )
        selected_checkpoint_path = Path(str(cosmos_config.checkpoint.load_path)).resolve()
        expected_checkpoint_path = Path(self.cfg.ckpt_path).resolve()
        if selected_checkpoint_path != expected_checkpoint_path:
            raise RuntimeError(
                "Official Cosmos loader selected a checkpoint different from the frozen "
                f"policy snapshot: {selected_checkpoint_path} != {expected_checkpoint_path}"
            )
        self.selected_checkpoint_load_path = str(selected_checkpoint_path)
        train_chunk = int(cosmos_config.dataloader_train.dataset.chunk_size)
        if train_chunk != int(inference["chunk_size"]):
            raise RuntimeError(f"Checkpoint chunk size {train_chunk} differs from protocol")
        self.resize_size = 224
        self.suites = {}
        self.env = None
        self.query_count_by_arm = {name: 0 for name in ARMS}
        self.query_forward_count = 0
        self.controller = None
        self.target_linears = None
        self.quantization_evidence = None
        try:
            quantization = importlib.import_module("quantization")
            self.target_linears = quantization.select_target_linears(self.model, self.protocol)
            self.target_manifest = quantization.target_manifest(self.target_linears)
            self.controller = quantization.QuantizationController(
                self.model, self.target_linears, root=self.root, protocol=self.protocol
            )
            self._write_build_manifest()
        except ModuleNotFoundError as exc:
            if exc.name != "quantization" and not str(exc.name).startswith("quantization."):
                raise
            self.target_manifest = []
        self.current_arm = None
        self.set_arm(arm)

    def set_arm(self, arm: str):
        if arm not in ARMS:
            raise ValueError(f"Unknown Cosmos arm: {arm}")
        if arm != "bf16" and self.controller is None:
            raise RuntimeError("quantization.py is required before enabling a low-bit arm")
        if self.controller is not None:
            self.quantization_evidence = self.controller.set_arm(
                arm, artifact_dir=None
            )
        self.current_arm = arm
        return self.quantization_evidence

    def _policy_observation(self, raw_obs):
        return self.run_libero_eval.prepare_observation(
            raw_obs, self.resize_size, flip_images=bool(self.cfg.flip_images)
        )

    def _write_build_manifest(self):
        if not self.target_manifest:
            return
        path = self.out / "build_manifest.json"
        temporary = path.with_name(path.name + ".tmp")
        payload = {
            "protocol": self.protocol.get("protocol"),
            "source_revision": COSMOS_REVISION,
            "source_roots": {
                "cosmos_policy": str(self.repo.resolve()),
                "base_libero": str(self._source_root("base_libero_root", "fastwam-smoke/libero/libero")),
                "libero_plus": str(self._source_root(
                    "libero_plus_root", "fastwam-libero-plus-pilot-20261005/LIBERO-plus/libero/libero"
                )),
            },
            "checkpoint_path": str(Path(self.cfg.ckpt_path).resolve()),
            "selected_checkpoint_load_path": self.selected_checkpoint_load_path,
            "dataset_stats_path": str(Path(self.cfg.dataset_stats_path).resolve()),
            "text_embeddings_path": str(self.text_embeddings_path),
            "target_linears": self.target_manifest,
        }
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def query(self, raw_obs, language: str, seed: int, *, suite="libero_goal", capture=False,
              decode_future=False):
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Cosmos requires the original non-empty task language")
        cache = self.cosmos_utils.t5_text_embeddings_cache
        if language not in cache:
            raise RuntimeError(f"T5 cache miss for task language; refusing on-demand encoder load: {language!r}")
        self.cfg.task_suite_name = suite
        self.cfg.seed = int(seed)
        self.run_libero_eval.set_seed_everywhere(int(seed))
        observation = self._policy_observation(raw_obs)
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        parallel_future = bool(self.protocol["inference"]["parallel_action_future_value"])
        original_future_decoder = self.cosmos_utils.get_future_images_from_generated_samples
        if parallel_future and not decode_future:
            # get_action's parallel flag controls post-sampler future-image decoding,
            # not whether future slots are present in its joint diffusion batch.
            # Keep the protocol's joint action/future path while skipping unused RGB decode.
            self.cosmos_utils.get_future_images_from_generated_samples = lambda *args, **kwargs: {}
        try:
            with self.torch.inference_mode():
                result = self.cosmos_utils.get_action(
                    self.cfg,
                    self.model,
                    self.dataset_stats,
                    observation,
                    language,
                    seed=int(seed),
                    randomize_seed=False,
                    num_denoising_steps_action=int(self.protocol["inference"]["denoising_steps"]),
                    generate_future_state_and_value_in_parallel=parallel_future,
                )
        finally:
            self.cosmos_utils.get_future_images_from_generated_samples = original_future_decoder
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        actions = result.get("actions")
        if hasattr(actions, "detach"):
            actions = actions.detach().float().cpu().numpy()
        else:
            import numpy as np
            actions = np.asarray(actions)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        expected = (int(self.protocol["inference"]["chunk_size"]), 7)
        if actions.shape != expected or not bool(self.torch.isfinite(self.torch.as_tensor(actions)).all()):
            raise RuntimeError(f"Cosmos returned invalid action chunk: shape={actions.shape}, expected={expected}")

        first_query = self.query_count_by_arm[self.current_arm] == 0
        self.query_count_by_arm[self.current_arm] += 1
        self.query_forward_count += 1
        metric = {
            "seed": int(seed),
            "query_seconds": float(elapsed),
            # get_action is the official API boundary; it does not expose a stable denoiser-only timer.
            "denoising_seconds": None,
            "peak_allocated_gb": self.torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": self.torch.cuda.max_memory_reserved() / 1e9,
            "first_query": bool(first_query),
            "timing_scope": (
                "official_get_action_joint_action_future_with_rgb_decode"
                if parallel_future and decode_future
                else "official_get_action_joint_action_future_without_rgb_decode"
                if parallel_future else "official_get_action_action_chunk_only"
            ),
        }
        output = {"actions": actions.copy(), **metric}
        if capture:
            captured = _cpu_tree(result, self.torch)
            output["captured"] = captured
            for key in (
                "data_batch", "generated_latent", "orig_clean_latent_frames",
                "latent_indices", "future_image_predictions",
            ):
                if key in captured:
                    output[key] = captured[key]
            if "data_batch" not in output:
                raise RuntimeError("Pinned get_action output did not expose data_batch")
        return output

    def _suite(self, suite_name):
        if suite_name not in self.suites:
            self.suites[suite_name] = self.benchmark.get_benchmark_dict()[suite_name]()
        return self.suites[suite_name]

    def _initial_states(self, suite, task_id, smoke_original):
        if smoke_original:
            base_root = self._source_root("base_libero_root", "fastwam-smoke/libero/libero")
            init_path = base_root / "init_files" / "libero_goal" / "open_the_middle_drawer_of_the_cabinet.pruned_init"
            if not init_path.is_file():
                raise FileNotFoundError(f"Frozen original LIBERO smoke initial states are missing: {init_path}")
            states = self.torch.load(init_path, map_location="cpu", weights_only=False)
            if len(states) == 0:
                raise RuntimeError("Original LIBERO task has no initial states")
            return states, str(init_path)

        # This is the strict loader already used by the frozen Plus evaluator; it does not import Fast-WAM.
        import torch
        from libero.libero import get_libero_path

        plus_root = self._source_root(
            "libero_plus_root", "fastwam-libero-plus-pilot-20261005/LIBERO-plus/libero/libero"
        )
        expected_root = (plus_root / "init_files").resolve()
        if Path(get_libero_path("init_states")).resolve() != expected_root:
            raise RuntimeError("LIBERO-Plus initial states are not from the frozen official Plus checkout")
        original_load = torch.load
        loaded_paths = []

        def load_official_state(file, *args, **kwargs):
            path = Path(file).resolve()
            path.relative_to(expected_root)
            if not path.is_file():
                raise FileNotFoundError(path)
            loaded_paths.append(str(path))
            kwargs["weights_only"] = False
            return original_load(file, *args, **kwargs)

        torch.load = load_official_state
        try:
            states = suite.get_task_init_states(int(task_id))
        finally:
            torch.load = original_load
        if len(loaded_paths) != 1 or len(states) == 0:
            raise RuntimeError(f"Expected exactly one frozen Plus state file, got {loaded_paths}")
        return states, loaded_paths[0]

    def _source_root(self, ready_key, relative_fallback):
        configured = self.environment_ready.get(ready_key)
        if configured:
            path = Path(configured)
            return path if path.is_absolute() else (self.root / path).resolve()
        return (self.root.parent / relative_fallback).resolve()

    @staticmethod
    def _step(env, action):
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, bool(terminated or truncated), info
        obs, reward, done, info = result
        return obs, reward, bool(done), info

    def _save_video(self, frames, path):
        if path is None or not frames:
            return None
        import imageio.v2 as imageio
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(path, frames, fps=30, codec="libx264")
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Rollout video was not written: {path}")
        return str(path)

    def episode(self, row, index, video_path=None):
        started = time.monotonic()
        suite_name = str(row["suite"])
        task_id = int(row["task_id"])
        env_seed = int(row.get("env_seed", 800_000_000 + int(index) * 1000))
        sampler_seed = int(row.get("sampler_seed", 900_000_000 + int(index) * 1000))
        cap = int(self.protocol["inference"]["max_steps_long"] if suite_name == "libero_10"
                  else self.protocol["inference"]["max_steps_normal"])
        steps = replans = next_observation_count = 0
        episode_query_start = self.query_forward_count
        success = False
        complete = False
        termination = "error"
        error = None
        queries = []
        frames = []
        action_min = [float("inf")] * 7
        action_max = [float("-inf")] * 7
        gripper_values = []
        first_action_chunk = None
        first_next_proprio = None
        task_name = row.get("task_name")
        language = None
        init_state_source = None
        libero_numpy_compatibility = None
        try:
            # The Plus fog perturbation calls plasma_fractal(dtype=np.float_).
            # Install the NumPy 2 compatibility alias before importing its env modules.
            libero_numpy_compatibility = _ensure_libero_numpy_compat()
            suite = self._suite(suite_name)
            smoke_original = bool(row.get("smoke_original", False))
            if smoke_original:
                if suite_name != "libero_goal" or task_id != 0:
                    raise RuntimeError("Original-task smoke is frozen to LIBERO-Goal task 0")
                task_name = str(row["original_task"])
                if task_name != "open_the_middle_drawer_of_the_cabinet":
                    raise RuntimeError(f"Unexpected original LIBERO smoke task: {task_name!r}")
                language = row.get("language")
                if not isinstance(language, str) or not language.strip():
                    raise RuntimeError("Original LIBERO instruction must be supplied by the frozen smoke row")
                base_root = self._source_root("base_libero_root", "fastwam-smoke/libero/libero")
                bddl_path = base_root / "bddl_files" / suite_name / f"{task_name}.bddl"
                if not bddl_path.is_file():
                    raise FileNotFoundError(f"Original LIBERO BDDL is missing: {bddl_path}")
            else:
                task = suite.get_task(task_id)
                if task_name and task.name != task_name:
                    raise RuntimeError(f"Manifest task identity changed: expected {task_name!r}, got {task.name!r}")
                task_name = task.name
                language = task.language
                bddl_path = suite.get_task_bddl_file_path(task_id)
            from libero.libero.envs import OffScreenRenderEnv
            self.env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_path),
                camera_heights=256,
                camera_widths=256,
            )
            self.env.seed(env_seed)
            import numpy as np
            np.random.seed(env_seed % (2**32))
            random.seed(env_seed)
            self.torch.manual_seed(env_seed)
            self.torch.cuda.manual_seed_all(env_seed)
            self.env.reset()
            states, init_state_source = self._initial_states(
                suite, task_id, bool(row.get("smoke_original", False))
            )
            obs = self.env.set_init_state(states[int(row.get("state_id", 0))])
            if video_path is not None:
                frames.append(self.libero_utils.get_libero_image(obs, flip_images=True).copy())
            dummy = self.libero_utils.get_libero_dummy_action("cosmos")
            settling = int(self.protocol["inference"]["settling_steps"])
            for _ in range(settling):
                obs, _, done, _ = self._step(self.env, dummy)
                next_observation_count += 1
                if done:
                    raise RuntimeError("LIBERO task completed during fixed settling steps")
            if video_path is not None:
                frames.append(self.libero_utils.get_libero_image(obs, flip_images=True).copy())

            while steps < cap and not success:
                deadline = getattr(self, "soft_deadline_epoch", None)
                if deadline is not None and time.time() >= deadline:
                    termination = "soft_deadline"
                    break
                seed = sampler_seed + replans
                parity_record = None
                full_query_metric = None
                if (getattr(self, "verify_action_only_parity", False)
                        and not getattr(self, "action_only_parity", None)):
                    full = self.query(obs, language, seed, suite=suite_name, decode_future=True)
                    query = self.query(obs, language, seed, suite=suite_name, decode_future=False)
                    import numpy as np
                    difference = float(np.max(np.abs(full["actions"] - query["actions"])))
                    parity_record = {
                        "full_path_query_seconds": full["query_seconds"],
                        "action_only_query_seconds": query["query_seconds"],
                        "action_max_abs_difference": difference,
                        "parity": bool(np.allclose(full["actions"], query["actions"], rtol=0.0, atol=1e-6)),
                    }
                    full_query_metric = {
                        key: full[key] for key in (
                            "query_seconds", "denoising_seconds", "peak_allocated_gb",
                            "peak_reserved_gb", "seed", "first_query", "timing_scope",
                        )
                    }
                    full_query_metric.update({
                        "replan": replans, "step_start": steps,
                        "parity_reference": True, "used_for_action": False,
                    })
                    self.action_only_parity = parity_record
                    if not parity_record["parity"]:
                        raise RuntimeError(f"Official action-only path differs from full get_action: {parity_record}")
                else:
                    query = self.query(obs, language, seed, suite=suite_name, decode_future=False)
                actions = query["actions"]
                metric = {key: query[key] for key in (
                    "query_seconds", "denoising_seconds", "peak_allocated_gb",
                    "peak_reserved_gb", "seed", "first_query", "timing_scope",
                )}
                metric.update({"replan": replans, "step_start": steps, "used_for_action": True})
                if full_query_metric is not None:
                    queries.append(full_query_metric)
                if parity_record is not None:
                    metric["action_only_parity"] = parity_record
                queries.append(metric)
                if first_action_chunk is None:
                    first_action_chunk = actions.tolist()
                for action in actions[:min(int(self.protocol["inference"]["execute_per_replan"]), cap - steps)]:
                    if deadline is not None and time.time() >= deadline:
                        termination = "soft_deadline"
                        break
                    command = [float(x) for x in action]
                    for col, value in enumerate(command):
                        action_min[col] = min(action_min[col], value)
                        action_max[col] = max(action_max[col], value)
                    gripper_values.append(command[-1])
                    obs, _, done, _ = self._step(self.env, command)
                    steps += 1
                    next_observation_count += 1
                    task_success = bool(self.env.check_success()) if hasattr(self.env, "check_success") else done
                    if first_next_proprio is None:
                        first_next_proprio = [float(x) for x in self._policy_observation(obs)["proprio"]]
                    if video_path is not None:
                        frames.append(self.libero_utils.get_libero_image(obs, flip_images=True).copy())
                    if task_success:
                        success = True
                        termination = "success"
                        break
                    if done:
                        termination = "environment_terminated"
                        break
                replans += 1
                if termination in ("soft_deadline", "environment_terminated"):
                    break
            termination = _finalize_termination(success, termination, steps, cap)
            complete = _episode_is_complete(
                termination, steps, next_observation_count, len(queries)
            )
        except Exception as exc:
            traceback.print_exc()
            error = f"{type(exc).__name__}: {exc}"
            if termination != "soft_deadline":
                termination = "error"
        finally:
            if self.env is not None:
                try:
                    self.env.close()
                finally:
                    self.env = None
        saved_video = None
        if video_path is not None:
            try:
                saved_video = self._save_video(frames, video_path)
            except Exception as exc:
                error = error or f"video: {type(exc).__name__}: {exc}"
                if termination not in ("success", "control_step_timeout", "environment_terminated", "soft_deadline"):
                    termination = "error"
        actual_actions = steps > 0
        closed_loop_complete = bool(complete and actual_actions and next_observation_count >= steps)
        gripper_min = min(gripper_values) if gripper_values else None
        gripper_max = max(gripper_values) if gripper_values else None
        return {
            "protocol": self.protocol.get("protocol"),
            "arm": self.current_arm,
            "index": int(index),
            "variant_id": str(row.get("variant_id", f"smoke-{suite_name}-{task_id}")),
            "suite": suite_name,
            "dimension": row.get("dimension", "smoke"),
            "task_id": task_id,
            "task_name": task_name,
            "language": language,
            "state_id": int(row.get("state_id", 0)),
            "env_seed": env_seed,
            "sampler_seed": sampler_seed,
            "initial_state_source": init_state_source,
            "complete": complete,
            "closed_loop_complete": closed_loop_complete,
            "success": bool(success),
            "termination": termination,
            "steps": steps,
            "replans": replans,
            "modelqueries": self.query_forward_count - episode_query_start,
            "next_observation_count": next_observation_count,
            "episode_seconds": time.monotonic() - started,
            "queries": queries,
            "first_action_chunk": first_action_chunk,
            "first_next_observation_proprio": first_next_proprio,
            "action_min": [None if value == float("inf") else value for value in action_min],
            "action_max": [None if value == float("-inf") else value for value in action_max],
            "gripper_action_min": gripper_min,
            "gripper_action_max": gripper_max,
            "rollout_path": saved_video,
            "quantization_evidence": self.quantization_evidence,
            "libero_numpy_compatibility": libero_numpy_compatibility,
            "action_only_parity": getattr(self, "action_only_parity", None),
            "error": error,
        }

    def _c8_paths(self):
        calibration = self.protocol["calibration"]
        source_root = Path(calibration["source_root"])
        plan_path = Path(calibration.get("plan_file", source_root / "data" / "plan.complete.json"))
        if not plan_path.is_absolute():
            plan_path = source_root / plan_path
        if not plan_path.is_file():
            fallback = source_root / "data" / "plan.json"
            if fallback.is_file():
                plan_path = fallback
        if not plan_path.is_file():
            raise FileNotFoundError(f"Frozen complete C8 plan is missing: {plan_path}")
        data_root = source_root / "data"
        return calibration, plan_path, data_root

    def c8_raw_observation(self, case_id: int):
        """Load one frozen C8 raw observation without sampling or model inference."""
        calibration, plan_path, data_root = self._c8_paths()
        case_id = int(case_id)
        if case_id not in {int(x) for x in calibration["case_ids"]}:
            raise ValueError(f"Case {case_id} is outside the frozen C8 cohort")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if (plan.get("status") != "complete"
                or plan.get("protocol") != "fastwam-vq-calibration-suites-v1"):
            raise RuntimeError("C8 input plan is not complete")
        if set(int(x) for x in plan.get("calibration_cohorts", {}).get("c8", [])) != {
            int(x) for x in calibration["case_ids"]
        }:
            raise RuntimeError("C8 membership differs from the frozen protocol")
        row = next((item for item in plan["inputs"] if int(item["case_id"]) == case_id), None)
        if row is None:
            raise KeyError(f"C8 case {case_id} is absent from the frozen plan")
        for key in ("observation_file", "metadata_file"):
            rel = Path(row[key])
            if rel.is_absolute():
                raise ValueError(f"C8 {key} must be relative to the frozen data root")
            resolved = (data_root / rel).resolve()
            resolved.relative_to(data_root.resolve())
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            row[key] = str(resolved)
        import numpy as np
        with np.load(row["observation_file"], allow_pickle=False) as archive:
            raw_obs = {key: archive[key].copy() for key in archive.files}
        metadata = json.loads(Path(row["metadata_file"]).read_text(encoding="utf-8"))
        if (int(metadata.get("case_id", -1)) != case_id or metadata.get("domain") != "original"
                or metadata.get("suite") != row.get("suite")
                or int(metadata.get("task_id", -1)) != int(row["task_id"])
                or not isinstance(metadata.get("description"), str)
                or not metadata.get("description", "").strip()):
            raise RuntimeError("C8 metadata does not match the frozen original-domain case")
        required = {
            "agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
            "robot0_eef_quat", "robot0_gripper_qpos",
        }
        if not required.issubset(raw_obs):
            raise RuntimeError(f"C8 case {case_id} is missing raw LIBERO fields: {sorted(required - raw_obs.keys())}")
        return raw_obs, metadata, row

    def c8_query(self, case_id: int, seed=None, capture=True, decode_future=False):
        """Run official get_action on one C8 snapshot and optionally retain its tensor batch/targets."""
        raw_obs, metadata, row = self.c8_raw_observation(case_id)
        if seed is None:
            seed = int(row["sampler_seeds"][0])
        result = self.query(
            raw_obs, metadata["description"], int(seed),
            suite=str(metadata["suite"]), capture=bool(capture),
            decode_future=bool(decode_future),
        )
        result["case_id"] = int(case_id)
        result["metadata"] = metadata
        result["plan_row"] = row
        return result

    def c8_data_batch(self, case_id: int):
        """Return ``(cpu_data_batch, metadata, plan_row)`` without diffusion sampling.

        Pinned ``get_action`` performs official image/proprio/text preprocessing and
        assembles the exact model input immediately before calling
        ``model.generate_samples_from_batch``. This adapter temporarily intercepts
        that single boundary, copies the batch to CPU, and restores the model method.
        No denoiser, VAE tokenizer or model forward runs; shared C8 files stay read-only.
        """
        raw_obs, metadata, row = self.c8_raw_observation(case_id)
        language = metadata["description"]
        seed = int(row["sampler_seeds"][int(self.protocol["calibration"]["seed_index"])])
        self.cfg.seed = seed
        self.run_libero_eval.set_seed_everywhere(seed)
        if language not in self.cosmos_utils.t5_text_embeddings_cache:
            raise RuntimeError(f"T5 cache miss for C8 language; refusing encoder load: {language!r}")
        self.cfg.task_suite_name = str(metadata["suite"])
        observation = self._policy_observation(raw_obs)
        captured = {}
        original_generate = self.model.generate_samples_from_batch

        def stop_before_sampler(data_batch, *args, **kwargs):
            captured["data_batch"] = data_batch
            raise _BatchCaptured()

        self.model.generate_samples_from_batch = stop_before_sampler
        try:
            try:
                with self.torch.inference_mode():
                    self.cosmos_utils.get_action(
                        self.cfg, self.model, self.dataset_stats, observation, language,
                        seed=seed, randomize_seed=False,
                        num_denoising_steps_action=int(self.protocol["inference"]["denoising_steps"]),
                        # Stop at the sampler boundary, before post-sampling image decode.
                        generate_future_state_and_value_in_parallel=bool(
                            self.protocol["inference"]["parallel_action_future_value"]
                        ),
                    )
            except _BatchCaptured:
                pass
        finally:
            self.model.generate_samples_from_batch = original_generate
        if "data_batch" not in captured:
            raise RuntimeError("Pinned get_action did not reach its sampler boundary")
        batch = _cpu_tree(captured["data_batch"], self.torch)
        return batch, metadata, row

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        if hasattr(self.model, "close"):
            self.model.close()

