# 固定样本准备记录

CPU64775在选样后退出resource_blocked：task0固定episode85/88需要两个未缓存视频文件file-002.mp4。原job、selection、lock和失败记录保留，没有使用已看过的episode替代。

CPU64776只查询了同一dataset revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` 的元数据，核验两段视频共64,412,505 bytes，有明确LFS SHA256，低于500MB补齐上限。它不执行模型、不下载数据。相应工程资源修订在本候选任何模型输出前写入PROTOCOL，科学门槛不变。

CPU64777在TC1N05 allocation完成15s，下载、hash与decode均在compute执行。新目录 `conditional_marginal_ready` 包含独立source_extension与扩展identity，记录parent identity hash；旧SmolVLA identity、缓存和失败准备目录保持原样。8个新样本全部准备完成：task0 85/88，task1 21/37，task2 59/80，task3 49/50；frame按原定规则。它们属于LIBERO，与WallDataset的84–95 test_locked无关。

GPU64778使用新manifest开始最小screen，仍为FP32/expert-W4两个arms、64noise、8conditions及固定微批次4。完整1024sample之外只有首condition FP/Q各4条batch对照。准备成功不构成科学go。
