"""Profile one or two complete DINO-WM Wall tasks with bounded MPC execution.

The official source checkout is left untouched.  This runner replaces the planner,
evaluator, and world-model rollout methods at runtime so the measured execution
keeps the same checkpoint, task, objective, and control flow.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--n-evals", type=int, default=2, choices=(1, 2))
parser.add_argument("--max-mpc", type=int, default=12)
args = parser.parse_args()

root = args.root.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
os.environ["DATASET_DIR"] = str(root / "data")
os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, str(root / "source"))

import gym
import hydra
import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf

import plan
from env.serial_vector_env import SerialVectorEnv
from planning.cem import CEMPlanner
from planning.evaluator import PlanEvaluator
from planning.mpc import MPCPlanner
from utils import move_to_device, seed, slice_trajdict_with_t


torch.set_num_threads(8)
torch.set_grad_enabled(False)
if not torch.cuda.is_available():
    raise RuntimeError("GPU required; refusing accidental CPU profiling")


def json_default(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class LatencyRecorder:
    """Hierarchical synchronized wall-clock records with incremental JSONL output."""

    def __init__(self, path: Path):
        self.path = path
        self.pending: list[dict] = []
        self.stack: list[int] = []
        self.context: dict = {}
        self.next_id = 1
        self.write_seconds = 0.0
        self.path.write_text("", encoding="utf-8")

    @contextlib.contextmanager
    def metadata(self, **values):
        old = self.context.copy()
        self.context.update({k: v for k, v in values.items() if v is not None})
        try:
            yield
        finally:
            self.context = old

    @contextlib.contextmanager
    def section(self, label: str, *, cuda: bool = False, **metadata):
        event_id = self.next_id
        self.next_id += 1
        parent_id = self.stack[-1] if self.stack else None
        if cuda:
            torch.cuda.synchronize()
        started_ns = time.perf_counter_ns()
        self.stack.append(event_id)
        error = None
        try:
            yield
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if cuda:
                torch.cuda.synchronize()
            ended_ns = time.perf_counter_ns()
            popped = self.stack.pop()
            assert popped == event_id
            record = {
                "event_id": event_id,
                "parent_id": parent_id,
                "label": label,
                "duration_ms": (ended_ns - started_ns) / 1e6,
                "cuda_synchronized": cuda,
                "context": {**self.context, **metadata},
            }
            if error is not None:
                record["error"] = error
            self.pending.append(record)

    def flush(self):
        if not self.pending:
            return
        started = time.perf_counter()
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            for record in self.pending:
                stream.write(json.dumps(record, default=json_default, sort_keys=True) + "\n")
        self.pending.clear()
        self.write_seconds += time.perf_counter() - started


recorder = LatencyRecorder(output / "latency_events.jsonl")


def install_profiled_rollout(model):
    """Replace only this model instance's rollout with an equivalent timed version."""

    def profiled_rollout(obs_0, act):
        num_obs_init = obs_0["visual"].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:]
        with recorder.section("wm_rollout.encode_initial", cuda=True):
            z = model.encode(obs_0, act_0)
        step = 0
        while step < action.shape[1]:
            with recorder.metadata(rollout_step=int(step)):
                with recorder.section("wm_rollout.predict_autoregressive", cuda=True):
                    z_pred = model.predict(z[:, -model.num_hist :])
                with recorder.section("wm_rollout.replace_action", cuda=True):
                    z_new = z_pred[:, -1:, ...]
                    z_new = model.replace_actions_from_z(
                        z_new, action[:, step : step + 1, :]
                    )
                with recorder.section("wm_rollout.append_state", cuda=True):
                    z = torch.cat([z, z_new], dim=1)
            step += 1
        with recorder.metadata(rollout_step="terminal"):
            with recorder.section("wm_rollout.predict_terminal", cuda=True):
                z_pred = model.predict(z[:, -model.num_hist :])
                z_new = z_pred[:, -1:, ...]
            with recorder.section("wm_rollout.append_terminal", cuda=True):
                z = torch.cat([z, z_new], dim=1)
            with recorder.section("wm_rollout.separate_embeddings", cuda=True):
                z_obses, _ = model.separate_emb(z)
        return z_obses, z

    model.rollout = profiled_rollout


def profiled_eval_actions(self, actions, action_len=None, filename="output", save_video=False):
    if filename == "output_final":
        evaluator_kind = "final"
    elif filename.startswith("plan_") and "_output_" in filename:
        evaluator_kind = "cem_diagnostic"
    elif filename.startswith("plan"):
        evaluator_kind = "mpc_execution"
    else:
        evaluator_kind = "other"

    with recorder.metadata(evaluator_kind=evaluator_kind, evaluator_filename=filename):
        with recorder.section("evaluator.total", cuda=True):
            n_evals = actions.shape[0]
            if action_len is None:
                action_len = np.full(n_evals, np.inf)

            with recorder.section("evaluator.preprocess_initial", cuda=False):
                initial_cpu = self.preprocessor.transform_obs(self.obs_0)
            with recorder.section("evaluator.initial_to_gpu", cuda=True):
                trans_obs_0 = move_to_device(initial_cpu, self.device)
            # Preserve the official evaluator's goal preprocessing even though its
            # transformed value is not consumed later in eval_actions.
            with recorder.section("evaluator.preprocess_goal", cuda=False):
                goal_cpu = self.preprocessor.transform_obs(self.obs_g)
            with recorder.section("evaluator.goal_to_gpu", cuda=True):
                move_to_device(goal_cpu, self.device)
            with torch.no_grad():
                with recorder.section("evaluator.imagined_rollout", cuda=True):
                    i_z_obses, _ = self.wm.rollout(obs_0=trans_obs_0, act=actions)
            with recorder.section("evaluator.select_imagined_final", cuda=True):
                i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)

            with recorder.section("evaluator.actions_to_cpu_and_denormalize", cuda=True):
                exec_actions = rearrange(
                    actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
                )
                exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
            with recorder.section("evaluator.environment_rollout", cuda=False):
                e_obses, e_states = self.env.rollout(self.seed, self.state_0, exec_actions)
            with recorder.section("evaluator.select_environment_final", cuda=False):
                e_final_obs = self._get_trajdict_last(
                    e_obses, action_len * self.frameskip + 1
                )
                e_final_state = self._get_traj_last(
                    e_states, action_len * self.frameskip + 1
                )[:, 0]

            with recorder.section("evaluator.task_metrics", cuda=False):
                eval_results = self.env.eval_state(self.state_g, e_final_state)
                successes = eval_results["success"]
                logs = {
                    "success_rate" if key == "success" else f"mean_{key}":
                    np.mean(value.astype(float)) if key == "success" else np.mean(value)
                    for key, value in eval_results.items()
                }
                visual_dists = np.linalg.norm(
                    e_final_obs["visual"] - self.obs_g["visual"], axis=1
                )
                proprio_dists = np.linalg.norm(
                    e_final_obs["proprio"] - self.obs_g["proprio"], axis=1
                )
                logs["mean_visual_dist"] = np.mean(visual_dists)
                logs["mean_proprio_dist"] = np.mean(proprio_dists)
            with recorder.section("evaluator.preprocess_achieved", cuda=False):
                achieved_cpu = self.preprocessor.transform_obs(e_final_obs)
            with recorder.section("evaluator.achieved_to_gpu", cuda=True):
                achieved_gpu = move_to_device(achieved_cpu, self.device)
            with recorder.section("evaluator.encode_achieved", cuda=True):
                e_z_obs = self.wm.encode_obs(achieved_gpu)
            with recorder.section("evaluator.embedding_divergence", cuda=True):
                logs["mean_div_visual_emb"] = torch.norm(
                    e_z_obs["visual"] - i_final_z_obs["visual"]
                ).item()
                logs["mean_div_proprio_emb"] = torch.norm(
                    e_z_obs["proprio"] - i_final_z_obs["proprio"]
                ).item()
            print("Success rate:", logs["success_rate"], flush=True)

        if filename == "output_final":
            with recorder.section("final_export.videos", cuda=False):
                lengths = np.full(len(successes), np.inf) if action_len is None else action_len
                case_records = []
                for case_id, success in enumerate(successes):
                    frames = e_obses["visual"][case_id]
                    steps = (
                        actions.shape[1] * self.frameskip
                        if not np.isfinite(lengths[case_id])
                        else int(lengths[case_id]) * self.frameskip
                    )
                    frames = frames[: steps + 1]
                    goal = self.obs_g["visual"][case_id, 0]
                    with imageio.get_writer(
                        str(output / f"case_{case_id:02d}.mp4"), fps=12
                    ) as writer:
                        for frame in frames:
                            side_by_side = np.concatenate([frame, goal], axis=1)
                            writer.append_data(np.clip(side_by_side, 0, 255).astype(np.uint8))
                    final_state = e_states[case_id, steps]
                    case_records.append(
                        {
                            "case_id": case_id,
                            "env_seed": int(self.seed[case_id]),
                            "success": bool(success),
                            "executed_env_steps": int(steps),
                            "goal_distance": float(
                                np.linalg.norm(self.state_g[case_id][:2] - final_state[:2])
                            ),
                            "initial_state": np.asarray(self.state_0[case_id]).tolist(),
                            "goal_state": np.asarray(self.state_g[case_id]).tolist(),
                            "final_state": np.asarray(final_state).tolist(),
                        }
                    )
            with recorder.section("final_export.trajectory_and_cases", cuda=True):
                np.savez_compressed(
                    output / "trajectories.npz",
                    states=e_states,
                    goals=self.state_g,
                    normalized_actions=actions.detach().cpu().numpy(),
                    action_lengths=lengths,
                )
                (output / "cases.json").write_text(
                    json.dumps(case_records, indent=2), encoding="utf-8"
                )
        recorder.flush()
        return logs, successes, e_obses, e_states


def profiled_cem_plan(self, obs_0, obs_g, actions=None):
    with recorder.section("planner.cem.total", cuda=True):
        with recorder.section("cem.preprocess_initial", cuda=False):
            initial_cpu = self.preprocessor.transform_obs(obs_0)
        with recorder.section("cem.initial_to_gpu", cuda=True):
            trans_obs_0 = move_to_device(initial_cpu, self.device)
        with recorder.section("cem.preprocess_goal", cuda=False):
            goal_cpu = self.preprocessor.transform_obs(obs_g)
        with recorder.section("cem.goal_to_gpu", cuda=True):
            trans_obs_g = move_to_device(goal_cpu, self.device)
        with recorder.section("cem.encode_goal", cuda=True):
            z_obs_g = self.wm.encode_obs(trans_obs_g)
        with recorder.section("cem.initialize_distribution", cuda=True):
            mu, sigma = self.init_mu_sigma(obs_0, actions)
            mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for opt_step in range(self.opt_steps):
            with recorder.metadata(cem_opt_step=int(opt_step)):
                with recorder.section("cem.optimization_step.total", cuda=True):
                    losses = []
                    for traj_index in range(n_evals):
                        with recorder.metadata(traj_index=int(traj_index)):
                            with recorder.section("cem.repeat_conditions", cuda=True):
                                cur_trans_obs_0 = {
                                    key: repeat(
                                        arr[traj_index].unsqueeze(0),
                                        "1 ... -> n ...",
                                        n=self.num_samples,
                                    )
                                    for key, arr in trans_obs_0.items()
                                }
                                cur_z_obs_g = {
                                    key: repeat(
                                        arr[traj_index].unsqueeze(0),
                                        "1 ... -> n ...",
                                        n=self.num_samples,
                                    )
                                    for key, arr in z_obs_g.items()
                                }
                            with recorder.section("cem.sample_candidates", cuda=True):
                                action = (
                                    torch.randn(
                                        self.num_samples, self.horizon, self.action_dim,
                                        device=self.device,
                                    )
                                    * sigma[traj_index]
                                    + mu[traj_index]
                                )
                                action[0] = mu[traj_index]
                            with torch.no_grad():
                                with recorder.section("cem.candidate_world_model_rollout", cuda=True):
                                    i_z_obses, _ = self.wm.rollout(
                                        obs_0=cur_trans_obs_0, act=action
                                    )
                            with recorder.section("cem.objective", cuda=True):
                                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                            with recorder.section("cem.topk_and_distribution_update", cuda=True):
                                topk_idx = torch.argsort(loss)[: self.topk]
                                topk_action = action[topk_idx]
                                losses.append(loss[topk_idx[0]].item())
                                mu[traj_index] = topk_action.mean(dim=0)
                                sigma[traj_index] = topk_action.std(dim=0)

                    with recorder.section("cem.scalar_logging", cuda=False):
                        self.wandb_run.log(
                            {
                                f"{self.logging_prefix}/loss": np.mean(losses),
                                "step": opt_step + 1,
                            }
                        )
                    if self.evaluator is not None and opt_step % self.eval_every == 0:
                        logs, successes, _, _ = self.evaluator.eval_actions(
                            mu, filename=f"{self.logging_prefix}_output_{opt_step + 1}"
                        )
                        with recorder.section("cem.diagnostic_logging", cuda=False):
                            logs = {
                                f"{self.logging_prefix}/{key}": value
                                for key, value in logs.items()
                            }
                            logs.update({"step": opt_step + 1})
                            self.wandb_run.log(logs)
                            self.dump_logs(logs)
                        if np.all(successes):
                            break
                recorder.flush()
        return mu, np.full(n_evals, np.inf)


def profiled_mpc_plan(self, obs_0, obs_g, actions=None):
    del actions
    with recorder.section("planner.mpc.total", cuda=True):
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()
        cur_obs_0 = obs_0
        memo_actions = None

        while not np.all(self.is_success) and self.iter < self.max_iter:
            with recorder.metadata(mpc_iter=int(self.iter)):
                with recorder.section("mpc.round.total", cuda=True):
                    self.sub_planner.logging_prefix = f"plan_{self.iter}"
                    actions, _ = self.sub_planner.plan(
                        obs_0=cur_obs_0, obs_g=obs_g, actions=memo_actions
                    )
                    with recorder.section("mpc.select_actions_and_warm_start", cuda=True):
                        taken_actions = actions.detach()[:, : self.n_taken_actions]
                        self._apply_success_mask(taken_actions)
                        memo_actions = actions.detach()[:, self.n_taken_actions :]
                        self.planned_actions.append(taken_actions)
                        action_so_far = torch.cat(self.planned_actions, dim=1)

                    with recorder.section("mpc.reset_execution_origin", cuda=False):
                        self.evaluator.assign_init_cond(obs_0=init_obs_0, state_0=init_state_0)
                    logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                        action_so_far,
                        self.action_len,
                        filename=f"plan{self.iter}",
                        save_video=True,
                    )
                    with recorder.section("mpc.success_and_feedback_update", cuda=False):
                        new_successes = successes & ~self.is_success
                        self.is_success = self.is_success | successes
                        self.action_len[new_successes] = (
                            (self.iter + 1) * self.n_taken_actions
                        )
                        logs = {f"{self.logging_prefix}/{key}": value for key, value in logs.items()}
                        logs.update({"step": self.iter + 1})
                        self.wandb_run.log(logs)
                        self.dump_logs(logs)
                        e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
                        cur_obs_0 = e_final_obs
                        e_final_state = e_states[:, -1]
                        self.evaluator.assign_init_cond(
                            obs_0=e_final_obs, state_0=e_final_state
                        )
                        self.iter += 1
                        self.sub_planner.logging_prefix = f"plan_{self.iter}"
                recorder.flush()

        with recorder.section("mpc.concatenate_planned_actions", cuda=True):
            planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(obs_0=init_obs_0, state_0=init_state_0)
        return planned_actions, self.action_len


PlanEvaluator.eval_actions = profiled_eval_actions
CEMPlanner.plan = profiled_cem_plan
MPCPlanner.plan = profiled_mpc_plan


original_load_model = plan.load_model


def profiled_load_model(*load_args, **load_kwargs):
    with recorder.section("setup.load_model_and_checkpoint", cuda=True):
        model = original_load_model(*load_args, **load_kwargs)
        model.eval()
        model.decoder = None
        install_profiled_rollout(model)
    with recorder.section("artifact.model_metadata", cuda=True):
        metadata = {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "dtype": str(next(model.parameters()).dtype),
            "parameters_without_decoder": sum(t.numel() for t in model.parameters()),
        }
        (output / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    return model


plan.load_model = profiled_load_model


def run_pipeline(cfg_dict):
    with recorder.section("pipeline.total", cuda=True):
        device = torch.device("cuda:0")
        with recorder.section("setup.read_model_config", cuda=False):
            model_path = Path(cfg_dict["ckpt_base_path"]) / "outputs" / cfg_dict["model_name"]
            model_cfg = OmegaConf.load(model_path / "hydra.yaml")
        seed(cfg_dict["seed"])
        with recorder.section("setup.load_dataset", cuda=False):
            _, datasets = hydra.utils.call(
                model_cfg.env.dataset,
                num_hist=model_cfg.num_hist,
                num_pred=model_cfg.num_pred,
                frameskip=model_cfg.frameskip,
            )
            dataset = datasets["valid"]
        model_ckpt = model_path / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
        model = plan.load_model(
            model_ckpt,
            model_cfg,
            model_cfg.num_action_repeat,
            device=device,
        )
        with recorder.section("setup.create_environments", cuda=False):
            environment = SerialVectorEnv(
                [
                    gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
                    for _ in range(cfg_dict["n_evals"])
                ]
            )
        with recorder.section("setup.workspace_and_targets", cuda=False):
            workspace = plan.PlanWorkspace(
                cfg_dict=cfg_dict,
                wm=model,
                dset=dataset,
                env=environment,
                env_name=model_cfg.env.name,
                frameskip=model_cfg.frameskip,
                wandb_run=None,
            )
        with recorder.section("planning.perform_total", cuda=True):
            logs = workspace.perform_planning()
        recorder.flush()
        return logs


config = OmegaConf.load(root / "source" / "conf" / "plan_wall.yaml")
del config["hydra"]
del config["defaults"]
config.ckpt_base_path = str(root / "checkpoints")
config.model_name = "wall_single"
config.n_evals = args.n_evals
config.planner.max_iter = args.max_mpc
config.n_plot_samples = 0
config.saved_folder = str(output)
config.wandb_logging = False
OmegaConf.save(config, output / "evaluation_config.yaml")

torch.cuda.reset_peak_memory_stats()
run_started = time.monotonic()
os.chdir(output)
official_logs = run_pipeline(OmegaConf.to_container(config, resolve=True))
recorder.flush()

cases = json.loads((output / "cases.json").read_text(encoding="utf-8"))
run_summary = {
    "complete": len(cases) == args.n_evals,
    "completion_definition": "Each case succeeds or reaches the fixed max_mpc budget; final replay/export completes.",
    "n_evals": len(cases),
    "successes": sum(case["success"] for case in cases),
    "success_rate": sum(case["success"] for case in cases) / len(cases),
    "elapsed_seconds": time.monotonic() - run_started,
    "profiler_jsonl_write_seconds": recorder.write_seconds,
    "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    "max_mpc_rounds": args.max_mpc,
    "max_env_steps": args.max_mpc * 25,
    "source_commit": "0a9492fa12044b852ae9e001cc74604b79c8bb0c",
    "warmup_policy": "No extra task was consumed for warmup; first calls are retained as cold-start observations.",
    "profiling_boundary": "Fine-grained GPU sections synchronize CUDA for attributable wall time; totals include profiling perturbation and are not production-throughput claims.",
    "official_metrics": official_logs,
}
(output / "run_summary.json").write_text(
    json.dumps(run_summary, indent=2, default=json_default), encoding="utf-8"
)
(output / "PROFILE_COMPLETE").touch()
print(json.dumps(run_summary, indent=2, default=json_default), flush=True)
