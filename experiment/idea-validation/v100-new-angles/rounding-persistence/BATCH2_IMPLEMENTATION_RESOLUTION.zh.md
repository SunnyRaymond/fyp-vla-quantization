# B2 提交前修正记录

原64815 runner不修改。新runner只在同state内批处理两个原定noise，独立verifier保留原raw轴及科学gate。

静态复核发现并修正两处B1→B2迁移遗漏，均发生于新job提交和模型输出之前：

1. 官方LeRobot v0.4.4 `sample_actions` 把时间expand到batch size；wrapper的timestep shape gate由残留 `(1,)` 改为 `(BATCH_SIZE,)=(2,)`。所有元素仍需满足原时间grid及atol1e-6、rtol0。
2. `x_states`新增batch轴后，endpoint重构需要 `x_states[-1,:,:, :7]`。原残留三索引写法会切position轴；现明确保留全部50位置，仅截physical7坐标。

当前新runner LF SHA256：`01059efb28661486931b935ad4a3a565456c043bc76985873a571bbf8c09c700`。独立B2 CPU verifier硬绑定此SHA及冻结protocol SHA `ae3ffb51225848ee951b11b933a6d7a2d642467824378fd5ddd8b2caea079ece`。

原runner LF SHA仍为 `361761c49c2babd4d7cfdfdb620f645da4f132f072d9a938b66246da44d92e07`。AST与shell语法检查通过，但这不等于实际B2 runtime兼容或科学可行。网络超时期间没有上传、提交或运行B2模型。
