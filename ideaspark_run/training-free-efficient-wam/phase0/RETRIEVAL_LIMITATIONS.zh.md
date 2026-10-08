# 本次 connector 检索范围和执行调整

2026-10-02：原 `phase0` 命令实际完成 named-paper connector resolution（3 个用户锚点）、arXiv 0–6mo 检索（40 条）和 OpenAlex 6–24mo 检索加 semantic booster（41 条）。使用原 skill 的 `scripts.dedup_merge` 将两个真实输出合为 81 条。保留 source/source_id/paper_url/abstract，并按原 Phase0 行为回填 retrieved_via。

Semantic Scholar 近期窗口在 HTTP429 后先后达到 300s 和 450s 的超时。随后进入旧窗口同一服务时，host 结束了本任务的该子进程和 phase0 父进程，以避免继续重复已确认失败的限流请求；未将它计为成功来源，未重新启动完整检索。独立执行原 OpenReview connector，四条 queries 对运行时 venue 均收到 challenge-required HTTP403，输出0条，不绕过 challenge，不计为成功来源。

因此本轮是 **真实 connector-grounded、覆盖降级** 的 corpus：新近 preprint 来自 arXiv，较老 published/semantic 来自 OpenAlex；缺少 Semantic Scholar 与 OpenReview 的检索贡献，未做原 orchestration 的失败 cap 再分配。不得声称 exhaustiveness。Named anchors 和后续 host coverage 的记录仍需原 resolver 验证；relevance partition、tagging、coverage check、mandatory fulltext、coherence 和 collision gates 继续执行。

这是对重复失败 connector 的执行调整，不是 webfallback，也没有通过 native web 手造 literature records。`.lit_grounding_mode=real` 仅表示上述两种 connector 返回了真实结果，不表示检索全面或候选通过 novelty audit。

后续调用 `context/run_available_connectors.py` 这一 run-local adapter 执行原 skill CLI，仅让这两个已确认不可用的 module import 失败；原 resolver / collision orchestration 继续使用 arXiv 和 OpenAlex 的真实检索与标题匹配。没有修改已安装 skill，没有降低 fulltext、citation、coherence 或 novelty 的判据；最终 prior-art 结论仍需声明缺失这两种检索来源。

Host coverage 提名6篇，原 resolver 验证收入5篇；GeoBoN 的完整标题未由该 resolver 匹配，保留在 `host_refs_unresolved.md`，没有手造或强行收入 corpus。此处是检索匹配失败，**不是论文不存在或机制未被研究的证据**。独立核查已读其 primary arXiv HTML（2607.17454），边界记录在 run 根目录的 `recent_prior_boundaries.zh.md`；后续判重必须考虑这份近邻证据，即便它没有 corpus record。

## Collision 入口补充（2026-10-02）

quality_collision 的原 wrapper 会再启动 run.py 子进程，最初的 run-local module skip 未传入该子进程。首个 collision attempt 的 arXiv/OpenAlex signature retrieval 已完成，随后 Semantic Scholar 再次 HTTP429 backoff；主线程仅终止已确认属于本轮的三个 Python 进程。run-local adapter 现也将该 phase3_collision 子进程重定向回自身，继续使用原 connector、merge 与 receipt 代码；未修改 installed skill。重跑时原 retrieval cache 复用了上述两项成功结果。SS/OpenReview 在 collision 仍不可用；不得据未命中宣称无 prior。

同一 adapter 的第二次尝试仍触发 --help availability probe 的独立解释器；主线程终止了该轮三个已确认进程。最终 adapter 对这两个 exact connector module 的 subprocess probe 返回明确的 run-local unavailable 状态，第三次尝试的原 CLI 已输出 skipped SS/OpenReview、2/4 degraded。其余连接器执行、真实缓存读取和 merge/receipt 均保持原代码。

## Generation 后的精确 prior 补充

主线程定向 primary 阅读发现 ToPi/CAPA 未在原226条collision pool命中。用原 add_host_refs resolver 核验并加入 phase0/lit_results.json：arxiv:2602.01609v1 与 arxiv:2602.00247v1，retrieved_via=host_web；独立摘录在 phase3_collision/extra_verified_priors.json。当前 lit_results 为60条、原 pattern-tagged lit_table仍58条；保持正在执行的version-bound request输入不变，没有伪造collision命中。audit的source枚举不含lit_results-supplement，补充文献以n/a及显式supplemental_evidence注明来源；threat_grounding仍按真实lit_results核对ID。
