# CCDS V100 上继续 CEM-Update PTQ

2026-09-12。用户授权在 NSCC access 暂时被 block 时使用 CCDS，NSCC 恢复前不尝试登录该系统。原 `cem-update-ptq/` 的 ASPIRE2A 控制、manifest、已取回的结果与未确认 job 记录均保留。

## 研究边界

沿用 `../cem-update-ptq/PROTOCOL.zh.md` 的固定方法、CAL/DEV targets/seeds、两轮完整配置搜索和机制门槛。官方 Wall epoch65、weight-only RTN、FP32 operators、CEM5、300 candidates、30 elites、W8 quota 3 encoder / 2 predictor 不变。V100 实际计时后单独冻结 walltime，不沿用 A100 吞吐估计。

这是同一机制实验的硬件迁移重跑。ASPIRE2A 的新 candidate arrays 未回传，故在 V100 重采 reference 并评价所有对照；不能混合跨 GPU 的候选池与分数，不能把同一批 CAL/DEV 说成额外独立样本。旧 test 不重新用于选择方法；通过 B 机制门槛后才进入新闭环 C。

## 已核实的集群与执行路径

依据 CCDSGPU-TC1-UG-UserGuide.pdf 第 4、14、16–18 页：CCDS-TC1，SLURM partition UGGPU-TC1。2026-09-12 实际账号 QoS normal：cpu=20、gpu=1、mem=64G，最多两个 job，MaxWall=6h。真实 home 为 /tc1home/UG/yguo017。

使用真实 SLURM_JOB_ID、hostname TC1Nxx、scontrol 的 RUNNING / UserId / NodeList 验证 allocation。PBS-only 入口不直接复用，也不伪造 PBS_JOBID。Login 只连接、提交、查询与小型控制文件读写；所有模型、计算、下载、解压、依赖安装、完整数组验证均在真实 SLURM compute allocation。数据准备优先 CPU-only。

集中 credentials 位于项目根 credentials.env，仅本地连接器读取，不上传到集群、不输出 secret、不写入 Git。CCDS 使用已记录 SSH host keys 与 RejectPolicy；ASPIRE2A 读取路径同步更新，但不进行登录测试。

## 作业记录

- 64664：CPU connectivity/environment probe，2 CPU、8G、无 GPU；TC1N05，COMPLETED 0:0，2秒。
- 64665：CPU runtime 准备，4 CPU、20G、无 GPU；共享 NFS 不支持 uv cache lock，0秒即 FAILED 2:0。未开始模型或数据下载。
- 64666：CPU 准备成功，TC1N05，4 CPU、20G、无 GPU，21分42秒。torch 2.2.0+cu121、checkpoint epoch65/368656057 bytes、source commit 与原实验一致，192 valid trajectories。官方压缩包经提取/CRC 验证。
- 64667：CPU archive metadata 检查，COMPLETED 0:0，记录数据包解压规模约59GB；无GPU。
- 64668：V100 采集与计时预检，TC1N05，COMPLETED 0:0，4分54秒，1 GPU（0.08167 GPU hours）。CAL/DEV 各4 episodes、8 pools，旧目标重叠为零；GPU 为 Tesla V100-PCIE-32GB，数值复现预检通过。单池约1.80秒。
- 64670：完整两轮四方法搜索，1 V100、4 CPU、24G，已提交。依据64668计时，搜索前冻结 workflow deadline 4500秒、SLURM walltime 4590秒（1.275 GPU hours 上限），预计实际约50–75分钟；算法与研究门槛不变，超时只能判未完成。

最终状态：已得出当前冻结配方Stage B no-go，详见RESULTS.zh.md。CPU核验64676 COMPLETED 0:0，7秒、无GPU；工程通过、机制门槛失败，不进入Stage C。

2026-09-12 定时检查：64670 COMPLETED 0:0，22分39秒（1 GPU），完整两轮、每方法71 evaluations；共享缓存实际83个CAL配置。小summary/dev_metrics/selected_maps已保存。补齐verifier的workload/reference.npz和跨method score2_ref一致性检查后，CPU独立核验64676已提交（2CPU、无GPU、10分钟上限）。未更改搜索方法或研究门槛。

正式作业64670已确认在TC1N05运行；启动trace已保存配置0至7的实际有限评分，约14.6秒/8 pools。用户要求此时结束前台等待，已创建每15分钟检查的thread heartbeat `ccds-cem-update`，派Luna xhigh进行有界检查，完成搜索后提交一次CPU独立核验。恢复入口见MONITOR_HANDOFF.zh.md；保持无变化时安静。
