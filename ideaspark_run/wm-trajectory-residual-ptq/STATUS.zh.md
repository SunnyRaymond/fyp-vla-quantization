# 仅保留检索尝试，不推进 residual 候选

2026-09-13。本目录启动时拟探索 temporal residual correction；随后对最新本地结果审查发现，FRT与PRR已在CCDS完成no-go筛选。为避免重演相同切面，本目录不生成或验证该候选。

Phase0曾启动真实connector检索；arXiv缺feedparser、OpenReview缺依赖，Semantic Scholar重复429返回0；OpenAlex获得42条记录，部分semantic补查504。会话恢复时执行handle24917已不存在，且未产生canonical lit_results/fulltext/phase1，因此**不是完整pipeline，也不是经验no-go**。现有connector输出保留，不伪造完成marker，不再为相同方向重新检索。

后续工作见 `../v100-new-angles/CAMPAIGN.zh.md`。用户明确授权裁剪本次用不到的pipeline部分；复用已有fulltexts并对新机制做原始来源补查。
