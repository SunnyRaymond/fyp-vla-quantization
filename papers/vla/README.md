# VLA papers

本目录下每个直属子目录对应一篇 paper，沿用原阅读路线编号；paper 的 PDF、README、supplemental 和必要的 validation note 保留在各自目录内。

- `001–007, 009–034, 037–038`：VLA、action modeling、flow matching 与 quantization foundations
- `039–044`：通用 quantization foundations
- `045–056`：VLA simulation / evaluation benchmarks
- `060–082`：action tokenization 与 active control
- `085–087`：VLA / world-model planning 相关方向
- `088–106`：VLA quantization literature、efficient-VLA survey 与 cross-frame token caching
- `107`：RGSQ VLM quantization
- `108`：KIVI KV-cache quantization foundation

详细阅读入口：[101. VQVLA](101-vqvla/README.md)（weight Vector Quantization 与 Centroid Reuse；固定 v1 PDF、中文公式例子与分时阅读路线）。

2026-10-08 更新的详细阅读包：

- [022. SpinQuant](022-spinquant/README.md)：固定 arXiv v4；正交旋转、R1–R4、Cayley optimization、GPTQ pipeline 与质量/速度实验解读。
- [108. KIVI](108-kivi/README.md)：固定 arXiv v2；K per-channel / V per-token、attention error、grouped/residual cache 与显存/吞吐解读。

两包均含本地 PDF、中文教学 README、物理页阅读路线、未作答的 Reading Questions 与空白 Meeting Card；论文报告结果与本地复现状态分开记录。

重复的 OpenVLA、BitVLA、HBVLA、OPTQ/GPTQ、QuaRot、FAST 和 TD-MPC2 已合并到一个 canonical copy；相近但不是同一篇的 paper（例如 FlashVLA 与 Think Twice, Act Once）仍分别保留。
