# Per-channel只加first10 gripper loss

已完成（2026-10-19）：GPU25869007.pbs101与CPU25869987.pbs101均F/0、完整评估通过；GPU用时01:32:30。结论不支持当前checkpoint进入闭环：visited base错误开合6→9/400，漏open仍2次，motor/latent有代价。未向暂停中的闭环对话发送恢复消息。完整结果与下一步建议见[分析报告](results/ANALYSIS.zh.md)，原始汇总见results/comparison.json。

用户授权的单因素实验：原C80、80 updates、update order、seeds与共享BF16 teacher保持不变，从原C8 rho1.5 baseline的D=1/r=1初始化。只增加first10第7维的MSE，lambda_g=1/6；原motor6的mean系数、rho=1.5、冻结normalization_c、optimizer及首次update后reset规则均保留。只训练614个ratios与1,806,863个channel log2D，VQ books/indices/row gains/原B32 exponents及模型参数冻结。固定final80，不按held-out选checkpoint。

沿用原corrected dynamic-absmax coupled surrogate与BF16 compensation。GPU allocation先执行已有channel surrogate检查和新的独立解析loss-gradient检查，确认motor系数不变、gripper前10步有梯度、late22无新增梯度。模型加载后做原visited BF16 action replay及D1/r1原baseline parity。训练后做uncached/cached完整action/future、实际infer与visited同输入repeat检查。

一份GPU作业walltime01:59、软截止6840秒；不自动续提。每15秒将GPU utilization与VRAM写入job log。gpt-6-luna/xhigh每10分钟只读监控；Q/R无实质变化时静默，失败或终态通知父任务。所有模型、数据与数值工作只在真实PBS allocation；login只作轻量控制和小文件读写，保留host-key verification，禁止banked reset。

完整预算445次WAM queries：80次训练、159次新original evaluation、200次visited evaluation、6次parity。Original evaluation保留calibration80和原held-out40×2=80；calibration首端点复用folding query，故160条记录对应159次新eval queries。所有teacher targets均复用，新增BF16 query仅作parity。

原80 held-out使用既有共享action/future teacher，报告原有motor/gripper/latent指标，并记录新模型的actual commands、开合错误、chunk bias与方向事件。旧original80 controls仅保存标量指标，不能假装已经具有其actual command对照；跨模型的actual command与漏释放比较使用已有visited面板。

Visited沿用40个旧C8-student states（奇数case2001–2079）、seed_index0与base/image±/proprio±五输入。严格复用原600个BF16/clip/joint端点和原per-channel200端点，新增gripper模型200端点，合计1000；四个students产生320项case×modality响应。传感器处理与原200份输入元数据逐条一致。旧image tensor未完整持久化，不声称逐像素重验全部旧tensor。不会把这40 states回灌到本轮训练。

Gripper使用真实denormalize→invert→sign，-1 open/+1 close。只在first10内识别事件；只有有序事件方向相同才算延迟，漏open单独报告，step0不推断跨chunk事件。Motor同时报告逐维RMSE、signed bias与先对chunk取均值再取绝对值的偏置。Latent使用原80 held-out的共享future target；旧visited面板没有future targets，不新增teacher去制造该项。

GPU结束后CPU allocation汇总全部原始记录，并按suite分成小JSON拉取。独立报告original80与visited40，不将seeds、steps、输入变体冒充独立cases；不新建development set。闭环推进依据是actual开合/漏释放是否改善，以及motor收益、偏置和latent的代价。固定final80，不为通过门槛反复调lambda。

结果支持进一步验证时，按用户明确授权，向codex://threads/01a11a6e-7199-7d72-82ec-ca3b3aaef7e7发送结果、checkpoint与冻结协议，让该对话恢复其被暂停的闭环goal；当前goal在确认消息送达后结束。若结果不支持则报告证据，不自动启动闭环。当前实验本身没有新rollout，也不提供native-kernel或闭环成功证据。
