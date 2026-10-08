# 候选生成委派约定

本轮为多个 idea，共用 Phase0/1；每个 candidate-N 是一个独立的 one-candidate branch。作者独立执行 Phase2.1+2.2，之后由其他 fresh context 审阅。Phase4 不在本轮范围。

1. 先完整读取本 branch 的 `.quality/generation/request.json`、技能的 `ideate_select.txt` / `ideate_generate.txt`，以及委派里给出的共享约束路径。
2. 使用委派指定的 Phase1 gap 作为 anchor，其他 gap 可 deferred。gap 文本保持原样；mechanism 不预指定。一个真正完整的组件即可，不堆 pattern。若没法获得可信新意，明确标记 application-grade / incremental，不编造 novelty。
3. 读 pattern overview、sub-pattern overview 与实际选中的 card；引用必须来自真实 card。closest_abstracts 是正文对照的索引，原文 delta 必须读 root `phase0/fulltext/` 中相关全文，未获取全文的 collateral 明确标注 abstract-only。
4. W4A4 或 W4A8 是目标 Linear 的实际位宽组合。列出量化的模块、所有 BF16/A8 例外、scale/zero-point 状态、额外分支/算子和是否能够融合。保留 FP norm/softmax 等常规例外不等于全图都是4 bit。主张 mixed 时统计升精度对象，不偷换成纯W4A4。
5. 默认 frozen-backbone PTQ；校准可优化 quantizer，若训练模型参数则明确称 QAT。不得运行模型/GPU/仿真实验；只产出方案及最小 falsification。
6. 核心须有 observation-model premise、精确定义的 estimand（若估计）、可运行的 naive baseline、额外结构的作用。终点 PTQ / 当前 action-aware PTQ / 已有 timestep compensation 作为相应简单对照，不以 baseline 过弱制造效果。
7. 资源未知；compute_budget 写建议 pilot 估计及计数范围，不能继承 factory 150 GPU-days / 8 GPUs。没有 measured speedup，没有先验成功率保证。
8. 字段结构用当前 prompt，包含 intervention_object 和 hook_shape_rationale。alias_terms 覆盖与本机制相关的 collateral；对不相关 collateral 在 composition_note 写具体排除理由，并让2.1/2.2的 composition_note 完全一致。

## v2 输出方式

实际 CLI 的 v2 request 优先于旧 prompt 的直接写出说明。不要直接覆盖 request 的输出文件，否则 `quality_record` 会拒绝并发变化。

在本 branch `.quality/generation/result.json` 写 envelope：

`{"contract_version":2,"request_id":"从 request 复制","artifacts":{"phase2_select/phase2_select_output.json":完整selection对象,"phase2_generate/phase2_generate_output.json":完整candidate对象}}`

随后运行 `run.py quality_record --dir 本branch --stage generation`，它将正式输出两份 JSON。最后完整读取 `run.py next --dir 本branch`；若 deterministic gate 指出真实合约问题，只修指定问题，最多一次 regeneration，不展开后续 coherence/critique。

验收：两个正式 JSON、generation receipt、next 已进入 coherence；返回路径、title、anchor gap index、几句话说明核心区别与低比特范围，最多250 words。失败或弱新意也保留并照实报告。
