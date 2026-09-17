# OpenVLA-OFT continuous / discrete interface — objective mismatch no-go

2026-09-13。结论：**structural/objective-mismatch no-go for the proposed comparison**，未运行GPU。这不是证明discrete VLA或其quantization不可做。

候选曾考虑在同一OpenVLA-OFT checkpoint上比较continuous action head与discrete action-token logits的量化敏感性。source确实保留两条代码路径，但所审查公开OFT配置使用L1 continuous head训练；保留discrete输出代码不能证明该head在此checkpoint上仍是可靠、可比较的行为策略。把两条接口的差距归因于quantization会混入训练目标不匹配。

因此不把source-available当成可用基线，不为这个对比下载另一份大checkpoint或切换训练目标。相关代码位置、公开配置与资产边界保留于 [资产审查](../../../../idea/idea-screening-20260912/vla_wam_asset_map.md)；原始source在 `experiment/reproduction/openvla-oft-vla-eval-libero/source/openvla-oft`。该候选停在pipeline结构审查，不做完整实验。
