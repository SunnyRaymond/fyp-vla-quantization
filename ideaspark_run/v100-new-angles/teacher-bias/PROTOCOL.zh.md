# Recorded-future error cancellation：最小协议

2026-09-13，模型输出前冻结。适配 IdeaSpark 的 C02 paired diagnostic / C04 error decomposition；独立审查见 `../new-cut-shortlist/TEACHER_BIAS_PRIOR_GATE.zh.md`。本案接受 diagnostic 可识别性，不声称新 quantizer。QuantWM 已有 planning/fidelity 失配观察；本案只问固定 RTN perturbation 能否抵消 predictor 对 recorded future feature 的原有误差。

固定 DINO-WM Wall epoch65，checkpoint SHA256 为 `8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b`。source commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c`，DINOv2 commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。复用已批准 smoke/screen helpers，不改旧目录。

六个样本固定为 unsliced validation-local trajectory indices **124–129**。每条取 raw frame **t=0**，H1 target frame5，H5 target frame25；actions为同条记录的0..24，按5个连续primitive actions concat为10D model action，五组输入得到五次transition。样本不足或源身份不符则停止，不改选。Wall locked84–95不用于本案；CPU仅提取固定六条输入，保存valid-local→underlying trajectory映射与对应文件hash。数据集已有normalization和transform照官方配置执行一次，不能再用旧random-goal Preprocessor对已处理actions/proprio二次归一化。

FP encoder 固定，recorded frames0/5/25走同一正式encode_obs路径，目标仅为 **recorded-future encoder feature**，不是物理state真值。完整visual patch×384 feature MSE是primary；proprio若保留只作单独描述，不并入primary。H5 free rollout包含五次官方predictor transition；H1为其同一trajectory的第一步，不能重排/删掉初始z。

三个arm：FP32、predictor-only RTN-W4、predictor-only RTN-W8。只量化既有24个predictor Linear weight，全部其余weights/activations FP32；采用既有helper的per-row signed absmax RTN规则，各arm从同一pristine snapshot开始并exact restore。无calibration、qparam搜索、训练或CEM。FP共享initial encoded input和真实action blocks；Q和FP均从真实initial observation出发自由预测，未来target不得进入predictor。

保留每条trajectory H1/H5的z_target/z_FP/z_W4/z_W8。独立CPU float64重放 `F=mean((z_FP-z_target)^2)`、`Q=mean((z_W4-z_target)^2)`、`P=mean((z_W4-z_FP)^2)`、`C=2*mean((z_FP-z_target)*(z_W4-z_FP))`，核对 `Q=F+P+C`。记录W8同样统计，但不替代W4 gate。

工程gate：全部finite；source/checkpoint/input/frame/action身份与冻结manifest一致；H5起点与FP encoder current feature一致；官方rollout与等价缓存initial-embedding路径在第一条FP trajectory上 allclose(atol=1e-6,rtol=0)；量化writeback/restore exact；target与actualprediction长度/轴匹配；误差恒等式 atol1e-10+rtol1e-9。工程不通过为implementation_inconclusive。所有六条H5的F及target跨patch/feature variance均须>1e-12，否则inconclusive_degenerate，不删除样本。

Primary gate：至少 **4/6** trajectory 的H5 `1-Q/F>=.05` 才为scope_limited_preliminary_go，否则mechanism_no_go。H1、最差sample、负cross-term数量、aggregate error与W8均如实报告，不加事后gate。先例审查中“明显恶化”的未定义附加条件不采用；固定4/6条件在看任何本案输出前统一。

CPU prep最多5min，GPU single V100最多5min（内部240s），CPU replay最多5min。无论结果如何均到此停止，不追加seed、dataset、quantizer或完整task validation；不从所测feature误差推断规划成功、真实控制改进、泛化或native低比特性能。
