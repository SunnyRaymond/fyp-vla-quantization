# 初始 common-U 案：prior-method_no_go，保留未运行状态

原案把主要收益放在live two-Q min的member flip减少，而TD-MPC2 planner实际使用live two-Q avg。Min属于TD target训练路径；本campaign不训练，仅观察live-min不能充分支持实际planner方法。Common-U若制造正的跨member误差关系，也可能恶化avg误差。

因此不对原common-U/min主张运行GPU；这是一项方法与使用路径的prior no-go，不是经验结果。原PRIOR_GATE中的source和checkpoint身份仍有效，缺少旧offline cache的资源问题已通过独立真实env reset输入方案解决。

另立的stratified-SR方案从开始即以已有five-critic、实际random-two avg为primary，无额外member成本；其来源、机制假设与实验在SUPPLEMENT_STRATIFIED和PROTOCOL中分别保留。这不是给旧min实验改主指标，因为旧案从未运行。
