# Root静态审查与CPU复算边界

2026-09-13，GPU提交前。科学阈值以PROTOCOL为准；不因实现细节或预想结果改阈值。

已定位并要求producer修正：live/detach/target按相同suffix绑定；仅tensor检查storage alias，TensorDict metadata明确处理；metadata snapshot与digest保持一致；CUDA move必须先于strict checkpoint load；直接WorldModel避免训练agent/optimizer依赖；official Q(avg)的末尾singleton维度显式处理；official pair RNG必须保存/恢复同一CUDA device；只放行3组live Linear weight及其detach aliases，bias/LayerNorm/target全部受bypass digest保护；完整quantizer records须落盘。静态修正不等于runtime通过。

独立verify_tdq.py不加载模型。它在真实CPU allocation中读取raw，校验8×3×4×64×5 member输出及相同CUDA RNG/pair schedule，以FP64枚举10个pairs并复算A/M/C与冻结gate。FP和RTN三seed位置必须完全相同。8state不得因退化被删除；joint fairness/positive/attribution必须同一集合。RTN不作为分母，所以RTN零误差不构成自动inconclusive。

工程复算包括：checkpoint/source/runtime loaded-file identity；manifest与prepared observation hash及raw精确匹配；用CPU torch.Generator6201独立重建candidate actions；FP terminal cache bytes与producer hash对应；三组实际Qweight shape、10个真实parameter entries的storage receipt；八次transaction的recipe/seed/grid、original-weight/scale hash一致性及SR随机draw证据；strict加载、最终restore和target/bypass检查。

这些检查验证记录和实现路径，不独立重跑CUDA kernel，不证明hash内随机变量满足统计定理，也不代替源代码对stratified law的审查。工程字段若缺失会降为implementation_inconclusive；不事后放宽科学阈值。所有raw与失败job保持，单次screen结束后停止本切面。
