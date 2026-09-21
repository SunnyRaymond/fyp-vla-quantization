# LeWM recurrent student runner：implementation note

端到端 runner 已由 PBS job `24564619.pbs101` 完成执行，`Exit_status=0`。入口是
`run_lewm_recurrent_student.py`，PBS wrapper 是
`run_lewm_recurrent_student.pbs`。

## 已固定的实现

- `LeWMCompactRecurrentTransitionStudent` 使用 `192-D` latent、`10-D`
  packed action、`history_adapter Linear(576,256)`、`latent/action projection`、
  一个共享 `LN -> Linear(256,1024) -> GELU -> Linear(1024,256)` transition，
  以及共享 `LN + Linear(256,192)` residual output head。
- 五个 prediction steps 使用 student 自己的 predicted latent feedback；loss
  weights 固定为 `[1/3, 2/3, 1, 4/3, 5/3]`，训练预算固定为 `1500` updates，
  snapshots 为 `500/1000/1500`。
- official teacher helper 保留 predictor 可接受的真实 history length `1/2/3`；
  只有 student 的固定 `3*192` history adapter 使用左侧首 latent padding。
- `HDF5EpisodeSliceReader` 只从 manifest 指定的 episode slice 读取
  `pixels/action/episode_idx/step_idx/ep_len/ep_offset`，并验证 train/held-out
  episode disjoint。它不会在 runner 中下载、复制或 hash 数据。
- 同一个 GPU job 在 HDF5 上生成 256 train + 8 held-out 的 H=1 manifest，编码
  current/goal，生成 4-slot train slate、16 个 held-out blocks 的 300 candidates，
  并生成 detached official-teacher targets/objectives；之后执行 1500 updates、
  snapshots、ranking/top-30/latent/causality/predictor-latency 与 predictor gate。
- 当前 runner 是 predictor-level only，**不会执行 official CEM，也不会提交 CEM
  PBS**。probe 的 CEM count/shape 只作为冻结接口记录，runner 已不再包含 CEM wrapper。

## 已解析的 interface contract

入口在任何 checkpoint/HDF5/model work 之前读取
`interface-probe/interface_probe.json`。文件不存在时只写
`run_status.json`，状态为 `WAITING_FOR_INTERFACE_PROBE`，并以退出码 `3` 结束。
因此不会把 frozen student horizon `5` 擅自当作 official CEM rollout prediction
count，也不会产生 predictor 或 planner 的 GO/NO-GO 结论。

probe 产生后，需要据以下字段更新/核对 contract（当前已读取
`24554356.pbs101` 的 PASS JSON）：

1. `official_jepa_rollout.H=1`：policy 初始 observation history；student 固定
   latest-3 adapter 在起点 repeat-left-pad 当前 latent；
2. `official_jepa_rollout.future_predictions_beyond_initial_H` 与
   `official_jepa_rollout.predicted_emb_shape`：Stage B wrapper 的真实输出边界；
3. `cem_candidate` 和 `solver.raw_action_dim`：candidate layout 与 primitive
   action packing；
4. probe 中的 `prepare_policy_info` shape/preprocessing：HDF5 observation/action
   准备层在 compute allocation 中据此实现，不在 login node 猜测。

action range 已在首次训练前冻结为 official PushT `[-1,1]^2`；Gaussian slate std
固定为 `sqrt(variance_floor=0.05)=0.22360679775`，runner 会校验这些值。正式
predictor-level 输出只
包括 Stage A 的 ranking/top-30/latent/causality/predictor-latency 汇总与 GO/NO-GO；
Stage B official CEM 在本 runner 中保持 `NOT_RUN_BY_SCOPE`。

## 运行入口与最终状态

`run_lewm_recurrent_student.pbs` 在 compute allocation 中读取固定 probe、freeze、
checkpoint 和 dataset。同一个 job 生成
`context_manifest.json`、`prepared_rows.pt`、student snapshots 与最终
`lewm_recurrent_student_summary.json`；login node 不执行这些步骤。

最终作业输出 `PREDICTOR_LEVEL_COMPLETE`，predictor gate 为 `NO-GO`；按用户要求
没有运行 Stage B official CEM。结果见
[RESULT_LEWM_RECURRENT_STUDENT.zh.md](RESULT_LEWM_RECURRENT_STUDENT.zh.md)。

## 验证

已完成 Python compile、PBS syntax、probe contract、student forward/padding shape 和
最终 GPU job 验证。最终作业保留 30 秒 GPU telemetry；本地只取回 summary、manifest、
job log/status，没有取回 checkpoints 或 `prepared_rows.pt`。
