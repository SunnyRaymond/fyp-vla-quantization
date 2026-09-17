# TD-MPC2 ensemble uncertainty — structural no-go

2026-09-13。结论：**structural_no_go**，GPU 0。原候选假设TD-MPC2 planner把Q-ensemble disagreement当epistemic penalty，因此PTQ可能污染该控制信号。但独立源码审查发现：该实现的target使用随机two-Q min，planner terminal value使用随机two-Q avg，没有原假设需要的variance/disagreement控制路径。

因此不人为给planner增加一个uncertainty penalty再展示量化影响，也不把原假设退化为缺少应用连接的Q-spread离线观察以继续消耗实验资源。审查中提出的这种离线替代未被执行，不作为原idea的验证或修复。若将来针对真正使用disagreement的模型另立候选，需要重新建立用途与先例边界。

这不是证明quantization不会影响Q-ensemble或TD-MPC2行为，而是当前切面依赖的真实控制路径不存在。所有来源与可用小checkpoint信息保留于 [独立brief与gate](../../../../idea/v100-new-angles/ensemble-uncertainty/IDEA_BRIEF_AND_GATE.zh.md)。
