# Rolling Ball LeWM+CEM policy adapter

This task adapter exposes the pinned vanilla LeWM predictor through the pinned StableWorldModel `CEMSolver` and the ReflexBench HTTP policy protocol. The random-lane task uses a task-specific goal cost: minimum terminal latent MSE to a fixed bank of 180 final available training observations. This is an adaptation, not an unmodified vanilla LeWM task baseline. The released data has no per-episode success labels, so goal-bank frames are not individually verified catches.

## Frozen planning path

- The server loads only the formal epoch-100 `last.ckpt` (`vanilla_lewm_raw_model_state_v1`). The validation-MSE checkpoint is diagnostic and is not used for planning.
- The CEM optimizer is `stable_worldmodel.solver.cem.CEMSolver` from StableWorldModel `10c26dbd5677083fa31dba69eb738b973845e9a4`, configured with keyword-only `action_space=`, `n_envs=1`, and a native `PlanConfig`.
- Each plan uses 300 candidates, 30 iterations, top-30 elites, variance scale 1, horizon 5, action block 1, seed 1234, and FP32. Sampling, candidate zero, `torch.topk`, and unbiased standard deviation remain in the pinned solver. The previous raw normalized plan is shifted by one step for native warm-start. On episode reset `[0]`, both the previous plan and the native solver's `torch_gen` seed are reset; an empty reset leaves state unchanged. This prevents different prior episode lengths from changing the first CEM innovations in the paired conditions. The adapter returns only the first action and replans after one action.
- Each CEM solve receives only the newest `fixed_cam` RGB image, so vanilla JEPA rollout has initial-frame count `H=1` and all five candidate actions are future actions. The predictor keeps its training autoregressive history of 3. This deliberately does not turn a three-image HTTP history into the CEM context.
- The task-cost wrapper calls the checkpoint's pinned vanilla `upstream_lewm/jepa.py::JEPA.rollout(..., history_size=3)` directly. It does not wrap the model in StableWorldModel's alternate latent-caching LeWM class. The only cached embeddings are the frozen training-goal bank encoded once at startup; each candidate rollout encodes its current image through vanilla JEPA.
- Candidate normalized actions are denormalized with the checkpoint's train-only mean/std and projected to the demonstrated raw target min/max, then normalized again before vanilla rollout. The returned first absolute joint target uses the same projection. Constant dimensions remain fixed at their checkpoint training value. These demonstration bounds are not a hardware safety certification; the solver's native Gaussian sampler itself is unchanged and does not enforce `Box` bounds.
- The cost is `min_j mean_d((z_terminal - z_goal[j])^2)` over the 180 train-only terminal embeddings. It reads no ball state, test episode, future validation frame, or privileged task state. The goal frames represent the last available observations in training episodes, not verified success states.
- HTTP proprioception follows the official VLA shape (`joint_positions: [1,7]`, `gripper_state: [1,1]`). The model and cost ignore proprioception. The offline probe gets it only from the same validation row's `state.npy`; the gripper indicator is mapped to 0/1 at `>0.5`. If state metadata or shape does not validate, the probe sends images only and records that fact.

## HTTP contract

The stdlib `ThreadingHTTPServer` serves `GET /info`, `POST /predict`, and `POST /reset`; it adds no web framework dependency. It supports one environment and `joint_pos` / `abs_joint` only. `/info` reports `action_dim=8` and `action_horizon=1`. `/predict` accepts ReflexBench VLA `images.fixed_cam` in `[N]` or `[N,T]` form and consumes the newest frame. The response is `[1,1,8]` absolute targets; ReflexBench performs the environment-specific absolute-to-relative controller conversion.

`latency_s` covers server request parsing, waiting for the serialized planner, image decoding and preprocessing, all CEM iterations, and output projection. `cem_latency_s` times the complete solver call with CUDA synchronization immediately before and after. The response includes an opaque `request_id`, server receive/complete timestamps, and the client's `step_ids` when present. These timestamps do not establish observation age or first-action application time. No latency, RTF, or task-success result is claimed until measured by the native evaluator.

## Guarded offline smoke

Run only in a genuine GPU PBS allocation after the formal checkpoint is available:

```bash
python policy_server.py --root "$TASK_ROOT" --checkpoint "$OUT/last.ckpt" --smoke --output-dir "$OUT"
```

Before importing numerical/model packages, the program requires `PBS_JOBID`, a readable `PBS_NODEFILE` containing the current non-login hostname, and a visible allocated CUDA device. Before loading the checkpoint, it requires the checkpoint's sibling `summary.json` to report `PASS_epoch100`, 100 completed epochs, `smoke_only=false`, and `last.ckpt (epoch 100)`. It then verifies CUDA availability, pinned source identities, checkpoint epoch, the train/validation split, terminal-goal row membership, and data array metadata. The smoke builds a native-shaped VLA JSON request from one validation frame and its aligned state row when valid, checks `/info` and `/reset`, runs one full 300/30/30/H=5 solve, and writes `planner_smoke.json` in the PBS output directory. The summary records timing, finite final-cost range, and protocol/action shape, with `closed_loop_evaluated=false`, `task_success=null`, `observation_age_s=null`, and `rtf=null`.

To start the endpoint rather than run the one-request smoke, use the same guard and checkpoint in the GPU PBS allocation, omitting `--smoke` and `--output-dir` and adding `--host`/`--port` as needed. `--self-check` exercises JSON shape helpers using the Python standard library only; it does not import Torch or load a model.

## Evidence still pending

The smoke records the final CEM cost tensor's shape and finite min/max as a planner-shape diagnostic. The policy server and offline CEM smoke do not establish Rolling Ball interception success, simulator closed-loop behavior, observation age, or real-time speed. The native RGB Isaac evaluation remains separate. Training, fixed-observation planner, and closed-loop results must be reported separately; the goal-bank task-cost adaptation must remain explicit in any baseline description.

## Pinned interface references

- [ReflexBench `POLICY_SERVER.md`](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/scripts/evaluation/POLICY_SERVER.md)
- [ReflexBench `eval.py`](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/scripts/evaluation/eval.py)
- [Vanilla LeWM `jepa.py`](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/jepa.py)
- [Vanilla LeWM image preprocessing](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/utils.py)
- [StableWorldModel `policy.py`](https://github.com/galilai-group/stable-worldmodel/blob/10c26dbd5677083fa31dba69eb738b973845e9a4/stable_worldmodel/policy.py)
- [StableWorldModel `cem.py`](https://github.com/galilai-group/stable-worldmodel/blob/10c26dbd5677083fa31dba69eb738b973845e9a4/stable_worldmodel/solver/cem.py)
- [Franka Panda joint limits](https://github.com/frankarobotics/franka_ros/blob/develop/franka_description/robots/panda/joint_limits.yaml)
