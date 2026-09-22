# Compiled World Model（LeWM + PushT）实验

本目录验证 [`../01_Compiled_World_Model_LeWM_PushT.md`](../01_Compiled_World_Model_LeWM_PushT.md) 的最小 predictor-level 命题。

## 文件

- `COMPILED_WORLD_MODEL_FREEZE.json`：结果前冻结的 split、models、training、E0 与 GO/NO-GO gates。
- `PROTOCOL.zh.md`：数据边界、对照和 claim boundary。
- `run_compiled_world_model.py`：E0、B1/B2/B3 training、development rank selection、fresh final test、B4 exact contraction 与 timing。
- `run_compiled_world_model.pbs`：带 compute-node guard、timeout 和 5 秒 GPU telemetry 的唯一 GPU job。

## 当前状态

PBS job `25309513.pbs101` 已完成。预注册 verdict 为 **predictor-level NO-GO**：B4 代数等价通过，但 B3 quality、context-dependence、matched B1 comparison 和 efficiency gates 均失败。详见 [`RESULT.zh.md`](RESULT.zh.md) 与 [`MEETING_CARD.zh.md`](MEETING_CARD.zh.md)。

`official CEM`、`closed-loop` 与 joint encoder training 均为 `NOT_RUN_BY_SCOPE`。
