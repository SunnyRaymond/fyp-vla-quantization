# VLA papers

本目录下每个直属子目录对应一篇 paper，沿用原阅读路线编号；paper 的 PDF、README、supplemental 和必要的 validation note 保留在各自目录内。

- `001–007, 009, 012, 032–034`：VLA、robot control 与 representation alignment
- `011, 037–038, 107`：VLM 与 VLM quantization
- `010, 024, 027–031, 039`：与 VLA 阅读路线配套的跨领域基础，包括 distillation、mixed-precision quantization、ANN/vector quantization、LoRA、flow matching 与通用二阶压缩
- `045–056`：VLA simulation / evaluation benchmarks
- `060–082`：action tokenization 与 active control
- `085–087`：VLA / world-model planning 相关方向
- `088–106`：VLA quantization literature、efficient-VLA survey 与 cross-frame token caching

详细阅读入口：[101. VQVLA](101-vqvla/README.md)（weight Vector Quantization 与 Centroid Reuse；固定 v1 PDF、中文公式例子与分时阅读路线）。

2026-10-08 分类调整：以 LLM 量化为主要研究对象的阅读包已迁至 [LLM papers](../llm/README.md)，保留原编号。其中 [022. SpinQuant](../llm/022-spinquant/README.md) 与 [108. KIVI](../llm/108-kivi/README.md) 含固定版本 PDF、详细中文教学 README、物理页阅读路线、未作答的 Reading Questions 与空白 Meeting Card。

论文库中重复的 OpenVLA、BitVLA、HBVLA、OPTQ/GPTQ、QuaRot、FAST 和 TD-MPC2 已合并到一个 canonical copy；相近但不是同一篇的 paper（例如 FlashVLA 与 Think Twice, Act Once）仍分别保留。
