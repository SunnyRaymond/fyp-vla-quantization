# PRR CAL 工程预检

2026-09-12，CCDS 作业 64705 在 TC1N03 V100 上 COMPLETED，exit 0:0，allocated GPU 时间 35 秒。CPU 资源复用核验 64704 用时 10 秒。

CAL 72–77、每 episode 两个起点；未读取 DEV/TEST。四组各用 seed1201 做 8 步短拟合，工程 gate 全部通过：

- 6 个 CAL episode 的 H=2 adapter 与原模型 rollout 对齐；同 batch FP null 为零，action 槽与 history 更新不变量通过。
- 四组 quantizer 梯度有限且有实际参数更新；两个 LoRA 组的 A/B 均有非零梯度和更新。
- 四组 hard W4 materialization 与整数/scale 序列化重载后的预测完全一致。
- 每组 8 步含硬化约 1.0–1.4 秒，峰值 allocated VRAM 约 1.80 GiB，reserved 约 1.89 GiB。
- local/recovery weighted target norm 约 2763.09/2762.85；短测有限、可训练。8 步压缩了完整 annealing schedule，不能用其 loss 趋势判断正式收敛。

Q0 与旧 clean_seed_1201 均可作为全组共享 frozen donor。后者 SHA 与旧 stage B 记录一致，来源为预先指定的 CAL-only clean:1201，并非 DEV 选胜者。

新 CAL 底层 episode 与已登记的历史 0–71 映射不重叠，新 initial-condition fingerprints 跨 episode 不重复。历史全量 initial-state fingerprint registry 不完整，因此不宣称已排除所有未登记历史数据的状态重合。

决定：保留 H2、batch2、1000 updates、seeds1201/1202/1203、gamma1、quantizer lr=.01、LoRA-r4 lr=.001，并在打开 DEV 前冻结 manifest_r1.json。四组全部完成后，独立 CPU verifier 根据 hard W4 自身 free rollout 判断；预检通过只表示实现具备正式测试条件，不是 idea 的正面实验结论。
