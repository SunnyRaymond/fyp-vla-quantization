# Real-observation rollout probe

该 probe 只使用 `SEEDED_PILOT_FREEZE.json` 中前两个 collection tasks（episode 9136、12704），复用已验证的单 episode runner 和 seed 42 reset。它记录 t0 与实际发生的 t25 solve；每个 solve 只在完整执行5个 10-D action tokens 后采样真实 pixels。对实际 env actions 重新应用官方 `action` StandardScaler 并按5步打包，必须与该 solve 的原生 CEM plan 在 `1e-5` 内匹配，之后才比较 official H=1 encoder observation latent、teacher forecast 与 student forecast。

PushT 的一个10-D token对应连续5个2-D environment actions；`horizon=5`, `receding_horizon=5`, `action_block=5` 因而对应25次 `env.step`。t0的观测点是全局步5/10/15/20/25；若发生t25 solve，第二组是30/35/40/45/50。终止时只比较已有完整token和真实pixels，不补齐缺帧。输出只含任务身份、步数、tensor shape、alignment/finite标记和逐solve/逐horizon MSE；不保存image、action trace或CEM tensor。

不训练、不访问 reserved holdout、不做 teacher shadow 或新的 CEM treatment，也不提交作业。本地仅做语法和 freeze 静态检查。PBS wrapper 将所有模型和 HDF5 读取限制在经 nodefile 验证的单GPU compute allocation 内，最长30分钟，并每5秒把 GPU 利用率/显存写入 `job.log`。
