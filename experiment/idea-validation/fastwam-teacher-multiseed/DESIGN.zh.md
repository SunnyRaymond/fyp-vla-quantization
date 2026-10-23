# Fast-WAM 原 C80 多 seed teacher 诊断

原 80 calibration cases2000–2079、四个 suite 各20；原观测、示范来源、预处理、BF16 Optional-IDM teacher、10+10步 sampler 不变。仅研究 first10 的6维 normalized motor；不训练，不用 held-out，不执行环境动作。

每个状态固定16个不同 seeds。indices0/1沿用原 plan，indices2–15为1900000000+case_id*32+seed_index。原 seed0的80个完整teacher缓存复用，其余最多1200次新teacher生成。额外8个原seed0全action/future精确重放和8个seed15重复，用于分离采样差异与实现不一致。

K4使用四个不重叠组0:4、4:8、8:12、12:16；K8使用两个不重叠组0:8与8:16。报告独立组的均值、方差/协方差与固定投影概率分布的差异。Full16只是有限样本参照，不当作真分布；嵌套K到full16的差异仅为补充。

主指标：同状态motor pairwise RMSE、RMS std；seed中心化误差的时间相关/有效协方差rank；chunk均值的变化；组间均值绝对RMS和相对within-state std的变化；协方差差异；投影分布JS/KL及其平滑尺度敏感性。每case是汇总单位，16seeds不作为16个独立状态；报告所有80cases和suite分解，不挑有利子集。16个样本不能证明多模态或完整高维密度。

投影包含60个位置坐标轴、6个跨10步共同方向、6个前后半段差异方向、16个固定随机60D方向。仅用calibration的共享统计尺，平滑尺度0.1/0.25/0.5 standardized units并列；这是假设的测量分辨率，不是teacher不确定性。共享full16标尺条件下的稳定性与重新拟合标尺的稳定性不同，报告明确这一边界。没有预设“通过即闭环有效”的阈值。

同时报告每case的full16局部标尺，以及全局平滑宽度/case内标准差之比，防止把过度平滑误判为稳定。结构对照使用每case32次固定随机的seed-index独立置换：60个scalar位置分别重新排列16个seeds，精确保留各位置边际样本、破坏时间/坐标间的seed配对；比较实际与置换后的chunk coherence。该对照只识别配对结构，不代表动作模式的任务价值或成功概率。

执行：CPU allocation读取/转换原缓存；单GPU allocation最长01:59:00（软退出6840秒），每15秒记录GPU利用率及显存；gpt-6-luna/xhigh每600秒监控。分析在CPU allocation运行。Login node只作小型控制文件、提交与状态查询。最终完成要求三项作业terminal/exit、完成marker、原80×16覆盖及重复门槛、同作业报告全部匹配。
