# 搜索与来源记录

日期：2026-09-05 · [返回总览](README.md)

## 范围与方法

输入是两份本地报告。第一个仅定位 WaterSIC 与 GRACE；第二个先逐项核对 31 个 reference identities，再围绕三个方向补找基础和直接相邻方法。主要依据 arXiv 原始 abstract / version history、下载的 primary full text、ICML/NeurIPS/PMLR 官方条目、publisher 页面和作者项目页。未使用 citation count 对近期 preprints 作质量排序。

使用的检索包括：`WaterSIC`、`GRACE quantization`、`action tokenization`、`active perception vision language action`、`differentiable predictive control`、`MPC guided policy search`、`control-aware action tokenization`、`active probing physical uncertainty information gain`。CLI 搜索窗口为 2018–2026；引用追溯补入 2015/ICRA 2016 的 MPC-Guided Policy Search。

筛选结果为 30 个 numbered entries，见 [文献对应表](REFERENCE_AUDIT.md) 和三份方向 README。另有 OAT 同研究线扩展稿。近期主题主要分为 tokenizer interface、interactive physical estimation、VLA active perception、control-to-policy transfer、latent predictive modeling 与 online safety layer；它们的 benchmark 和 metrics 不直接横比。

## 自动搜索的实际限制

安装的 paper-search CLI 不支持 skill 文件所写的 `--json` 参数，已先用 `--help` 确认，因此保存原始 CLI 输出，并另外从官方 arXiv 页面保存结构化 metadata。

CLI 返回 `arxiv=35, dblp=0, open_alex=16, openreview=0, semantic_scholar=0, crossref=32`，合并后 82 unique records，存在大量泛词误匹配。**82 不是本库相关论文数量。** 某些 connector 未返回 abstract，不能把这些自动结果当作完成了全文相关性审查。实际入库依据指定引用和 primary-source verification，未按 lexical ranking 机械推荐。

关键原始错误示例（完整记录见下面链接）：

```text
[openreview] Error on query 'WaterSIC': openreview not installed. pip install openreview-py
[semantic_scholar] Error on query 'WaterSIC': 429 Client Error:  for url: https://api.semanticscholar.org/graph/v1/paper/search?query=WaterSIC&offset=0&limit=8&fields=title%2Cauthors%2Cyear%2Cabstract%2CcitationCount%2Curl%2Cvenue%2CpublicationDate%2CexternalIds&year=2018-2026
[dblp] Error on query 'WaterSIC': 503 Server Error: Service Unavailable for url: https://dblp.org/search/publ/api?q=WaterSIC&format=json&h=8&f=0
[open_alex] Error on query 'GRACE quantization': 504 Server Error: Gateway Timeout for url: https://api.openalex.org/works?search.semantic=GRACE+quantization&filter=publication_year%3A2018-2026&sort=relevance_score%3Adesc&page=1&per-page=8
```

部分 OpenReview 页面要求 browser verification，MDPI HTML 请求返回 403；PI-VLA 已从 publisher 的公开 PDF host 获取，ABNet / BEAST 从正式 proceedings 获取。WaterSIC 和 GRACE 的 ICML 页面可经普通 HTTP 请求核验，但未独立确认 WaterSIC Spotlight 等级。没有绕过登录或访问控制。

Citation / first-author citation 排名在这些 connector failure 与不同年代混合的情况下不可靠，因此未生成。推荐路线按具体研究贡献和依赖关系组织，见总览。

## 可复查材料

- [完整未筛选 CLI 输出，含所有返回条目与错误](research/paper-search-cli.txt)
- [指定引用及补充论文的官方 metadata](research/reference-metadata.json)
- [实际入库 PDF 的版本、作者、来源与 download URL](research/sources.json)
- [逐项引用对应表](REFERENCE_AUDIT.md)
- [本地 PDF inventory](PDF_INVENTORY.md)

保留原始输出是为了说明 coverage limits，不表示其中所有自动命中都值得阅读。未覆盖来源也不能被解释为不存在 prior art。
