# Gripper decision margin：本轮不进入GPU

**结论：`insufficient_mechanism_no_go`，GPU 0。** 保留原prior gate；没有得到任何经验上的flip-rate。

数值扰动跨过gripper threshold会改变离散命令，这值得在机器人评估中记录，但`|delta|>=margin`只是threshold crossing的必要代数条件。当前proposal没有提出能与已有action-sensitive PTQ区别开的机制、可解释对照或方法，仅统计是否发生crossing，不足以作为本轮独立新idea继续占用GPU。

两个源码入口为[OpenVLA-OFT adapter](https://github.com/moojink/openvla-oft/blob/main/experiments/robot/robot_utils.py)与[LeRobot VLA-JEPA processor](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/vla_jepa/processor_vla_jepa.py)。后者不适用于SmolVLA，不能把它的规则概括为所有LeRobot策略的规则。本轮也没有核验SmolVLA实际执行adapter，不能用其他模型的threshold代替。

这不是“量化不会导致gripper翻转”的科学结论；停止的是当前缺少独立机制问题的proposal，不继续更改threshold或采样去制造signal。
