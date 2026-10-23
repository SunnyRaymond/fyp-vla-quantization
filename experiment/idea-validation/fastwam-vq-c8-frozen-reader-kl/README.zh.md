# C8 frozen-reader KL：单轮验证

沿用原 C8 A+Z、rho1.5、Adam lr5e-4、batch1、固定顺序8 updates，update1后重置optimizer state。只增加 frozen BF16 reader 的 first10 motor6 soft-bin KL。原 action MSE 和 future global NMSE（排除 observed frame0）不变；无 activation clip/per-channel；不运行280闭环。

从原 baseline 使用的相同未训练 C8 banks 开始。复用原8个 teacher futures/actions、原 heldout40×2 teacher 和原 baseline88条记录。辅助项只有 student future 可微，reader weights 固定，输入、prompt/proprio 和 sampler noise 配对。

只用 C8 teacher 的 first10 motor6 拟合固定 action mean/std，std floor0.02；16个 Gaussian soft-bins、sigma0.375、内边界[-3.5,3.5]沿用现有定义。beta只标定一次，使初始C8平均附加loss为原加权NMSE的10%。这是一项冻结的配方，不是已验证的最优权重。若进入MSE ablation，采用相同标准化输出，按初始loss贡献匹配系数，保持相同初始化和训练预算。

训练使用分阶段 chain-rule VJP：先取得当前student输出，再单独计算reader的future cotangent，最后完全相同参数/输入/噪声重放student反传。每次检查重放逐值一致；CPU toy核对staged VJP与joint autograd。原teacher-reader重放、首步部署变化、最终native inference parity为必要检查。所有tensor/model计算位于真实PBS allocation，GPU利用率和显存每15秒写job.log。4h单GPU任务经normal路由，启动时强制验证execution queue=g1。

完整主评测为实际quantized student的40cases×2seeds。先平均case内seed，再suite内case，最后四suite等权；同时报告case胜负、中位数、suite和seed分层。固定gate：first10 motor RMSE均值下降、case配对中位数下降、至少21/40改善、future NMSE均值不退步。仅全部通过后运行一次reader MSE ablation；否则停止这版，不扫参数。Gate只是工程筛查，不是显著性或闭环保证。

使用 `nscc-access/.venv/Scripts/python.exe` 运行本目录 `control.py upload/submit/status/pull/logs`。顺序prepare(CPU)→reader_kl(GPU g1)→aggregate(CPU)。只有aggregate gate通过才可提交reader_mse→aggregate_mse。控制器保存唯一handles和提交intent，未知/排队/运行状态不允许重提。已提交作业在artifact目录内持有源代码快照；失败后仅做必要实现修复，不改变冻结配方。

原heldout已经多轮查看，属于固定开发/回归集合，不能称为全新独立泛化。辅助KL是平滑读出分布KL，不是实际future density或mutual information。Reader自身得分下降不作为成功证据。Baseline没有新增完整future缓存，不能重算其reader KL并声称相对改善。

## 完成结果

完整验证已完成，gate未通过，停止本配方。Heldout motor RMSE +1.93%，12/40改善；NMSE -5.78%。未启动reader MSE ablation、参数搜索或闭环。详见[结论](results/CONCLUSIONS.zh.md)和[完成证据](completion_evidence.json)。10分钟监控已停止。
