# 一次执行效率修复：冻结 gate

64815 在 TC1N07 运行9m03，外部540s timeout结束（exit124），已保存11/12个condition，最后完成state5/noise0。原始文件与结论保留为 **inconclusive_budget**，没有读取或计算partial科学指标。这不是科学no-go，也不能称已证明切面不可做。

采用适配pipeline的C02 matched-control审查：两noise同state共享相同condition与固定weight schedule，将它们组成B2可减少重复prefix、snapshot装载与单条小batch开销。独立agent审查认为可进行一次纯执行修复。Root接受，理由是完成原定screen，未增加统计样本或探索已见科学结果。

唯一允许变化为batch mode：每个state的noise2201/2202一起forward。所有FP/RTN/Frozen/Cyclic/common-FP arms均使用同B2，保持跨样本attention隔离，不跨state组batch，不平均action输出。Raw仍逐noise保存原先相同轴，common input hash分别对batch样本计算。B2浮点计算可能与B1有细微不同，因此不拼接原11个conditions；新run全部重做并独立判定。

继续使用原PROTOCOL.zh.md全部科学定义、样本、初始noise、SR seeds、qparams、K10、normalized first8×7 metric、6state binding与5/6 joint gate；重新执行B2 FP/manual no-op与RTNreuse/reload工程对照。原10分钟allocation、540s内部上限不扩大。初始snapshot与input identities仍绑定，新增明确batch_mode receipt与新runner SHA。

这是唯一获准的执行修复。若B2仍预算不足或工程gate未过，则停止为对应inconclusive，不继续调batch、样本、solver、容差或时间预算。无论B2科学go/no-go均不扩验。最终总结同时列64815的失败和新run结果，不将重跑成本隐藏为单次运行。
