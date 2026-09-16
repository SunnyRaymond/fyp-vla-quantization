# RSLH：现有 checkpoint 结构不支持该切面

2026-09-13。结论：**model_structure_no_go / 未执行数值干预**。这不是已经证明recency cache quantization在其他模型上无效。

候选原理是保留最新history slot FP16、把较旧两个slots量化到W4，并与保留最旧slot的同bytes负对照比较。它要求checkpoint具有至少两个实际history slots。独立预研误用了`num_hist=3`，但当前Wall epoch65已执行的runtime证据是`num_hist=1`：

- [64706运行manifest](../../closed-loop-quantization/experiments/prr-ccds/artifacts/64706/manifest.json)：`num_hist:1`、`runtime_num_hist:1`。
- [64705工程预检](../../closed-loop-quantization/experiments/prr-ccds/artifacts/64705/engineering_summary.json)：已加载model的`num_hist:1`。
- 源码`reproduction/dino-wm-wall/source/models/visual_world_model.py`使用真实`self.num_hist`限制predictor history。

只有一个slot时，“最新”和“最旧”是同一个位置，两种干预及负对照退化为相同操作；无法在现有模型上检验该机制。不得通过把num_hist配置改成3冒充同一pretrained checkpoint验证，也不为此训练或下载另一个world model。

决策：保留候选与结构性排除原因，停止本候选，GPU新增用量为0。新的Antithetic job会顺带记录已加载model结构，但不专门为此再申请GPU。未来如已有合适multi-history checkpoint，应作为新条件的新screen，而非把本次记录改成成功。

最初预研见`tmp/idea-screening-20260912/alternative_angles.md`（项目根起算，随后修正错误前提）；来源/pattern为KIVI、KVQuant的cache粒度思路与IdeaSpark C04/C02。它们不能替代当前模型的结构证据。
