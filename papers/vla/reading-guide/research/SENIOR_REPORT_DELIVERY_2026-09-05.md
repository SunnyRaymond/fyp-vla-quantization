# 师兄报告阅读材料交付检查

完成日期：2026-09-05。

## 本次交付

- 主 Reading List：独立新增 [#23 WaterSIC](../papers/23-watersic/README.md) 与 [#24 GRACE](../papers/24-grace/README.md)。主库由 22 个 core identities 增至 24 个；现有 companions 分类保持原有口径。主库 PDF 共 56 份。
- [VLA 三方向独立阅读库](../../vla-ideas-reading-guide/README.md)：30 个 numbered entries，分为 10 / 8 / 12 项；31 份 PDF，含 OAT 同研究线扩展稿。
- 本次总计：32 个 reading entries，33 份 PDF。
- 每篇有中文说明、English terminology、原文定位、分时阅读路线、未作答 Reading Questions 与 Meeting Card。
- 同步更新主库 [README](../README.md)、[PDF inventory](../PDF_INVENTORY.md) 与 [weekly plan](../WEEKLY_PLAN.md)。原始 ReadingList 入口已并入本 guide。

## 核验

本次 33 份 PDF 均通过 header、pypdf reopen、未加密、page count、首页标题匹配与可提取文本检查。新增和更新 Markdown 中 421 个本地链接全部有效；没有缺失 PDF 或 README。两份输入报告均已逐页提取并检查图像；WaterSIC theorem / Algorithm 2 与 GRACE 主结果表另作视觉核对。未计算 SHA-256，未训练模型，未占用 GPU，未声称 benchmark 已复现。

机器可复查记录：[validation](../../vla-ideas-reading-guide/research/validation.json) · [实际 PDF sources](../../vla-ideas-reading-guide/research/sources.json)。

## 重要范围与纠正

- OAT 已支持 prefix-based anytime decoding，“所有 tokenizer 固定码率”不能作为既定前提。
- CoMe-VLA 是原总结之外的重要 VLA active-perception prior art；物理 probing 的目标与 sensing assumptions 仍需具体区分。
- DPC 的直接 policy optimization 与 online CBF safety intervention 是不同部分；不能直接宣称关闭所有保护层后仍继承原 theorem。
- 原 BarrierNet arXiv ID 指向 Coq 论文。已给正确 T-RO DOI，并提供作者主页链接的 2021 前身稿；本地文件不冒充 journal final。
- ABNet 用 ICML 2025 proceedings final，和较早 Attention BarrierNet preprint 分开；BEAST 用 NeurIPS 2025 final。
- WaterSIC / GRACE 的 ICML 收录与 paper identity 已核验；WaterSIC Spotlight 等级及师兄本人作者身份未独立确认。
- 自动 paper-search 的 Semantic Scholar / DBLP / OpenAlex / OpenReview 有 coverage limits，未用数据库无命中推断不存在 prior art。

完整原引用对应表与扩展入口见 [REFERENCE_AUDIT](../../vla-ideas-reading-guide/REFERENCE_AUDIT.md)，搜索限制见 [SEARCH_REPORT](../../vla-ideas-reading-guide/SEARCH_REPORT.md)。
