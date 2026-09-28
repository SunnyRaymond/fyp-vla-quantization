# Fast-LeWM 迭代内精确 encoder 复用：有限工程验证

2026-09-27，在主实验 `25569974.pbs101` 的12 arms完成后冻结。主实验中Fast-LeWM/30的成功率点估计最高、规划时间约为LeWM/30的五分之一，但质量交互区间跨零。这里保持Fast/30搜索质量，验证当前/goal观测在30轮中重复编码的成本能否去除；历史LeWM已有 exact iteration cache，不声明研究新颖性。

只使用原manifest前8项（含原baseline失败项5），每项固定初始observation，实际B=1。所有官方300 candidates/top30/30 rounds、H1/block25/history1/beta0、GRU执行路径和native CEM不变。只包装action-free `model.encode`，保存同一solve内goal/current两次encode所得emb，随后复用；每solve重新构建缓存。不能跨environment、replan或observation使用缓存，不改变action encoder、predictor、rollout、criterion或get_cost。

每context两条路径各warmup2次、计时5次，交替顺序，复制输入和pristine solver在计时外。每paired solve复位相同native generator/global RNG；缓存首次构建在计时内。保留native solver logging同步，不加每次cost同步；完整solve前后CUDA同步。另做一对未计时trace，逐轮比较所有candidate action vectors、cost vectors以及最终actions/costs，要求bitwise相同，不能放宽为容差。

每solve应30次cost/9000 candidate scores；最终输出finite。性能门为8 contexts各自paired中位数耗时降幅的中位数至少10%，peak allocated memory比不超过1.10。失败保留NO_GO；不调budget、seed、task、rank或误差容差。仅报告fixed-observation planner fidelity/latency，不能称部署收益或闭环成功率无损，也不把主实验B50时间套到本实验B1。

GPU allocation20分钟、内层1050秒；真实PBS和nodefile host guard在模型/数据读取前。利用率/显存每5秒写入job.log。复用已有source、checkpoint、scaler，不下载、不hash、不改共享环境。余额低于40时不提交新pilot；不使用banked reset。
