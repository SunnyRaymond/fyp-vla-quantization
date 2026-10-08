# 主候选的计算余量：先看对象，再看模型大小

这是一项算子计数推导，没有运行 WAM、GPU 或真机。参数规模大，不直接意味着本候选删除的路径占比大。

令某层某次 solver evaluation 的 action-query 数为 N_a，observation-key 数为 N_o，共享 attention 内积维度为 d_att。按一次乘加计两 FLOPs，action→observation 的 QK 与 AV 合计约为：

`F_edge = 4 N_a N_o d_att`。

本候选保留全部 video-query 计算、QKV projection 和 FFN。若该层 video FFN 的线性映射集合为 P、active video token 数为 N_v，仅这些线性映射的计算量就是：

`F_video_FFN = 2 N_v sum_{p in P}(d_in,p d_out,p)`。

因此可删除 FLOP 的份额至多为所有可跳过 steps/layers 的 F_edge 总和除以完整 baseline FLOPs；用全部 steps/layers 的 F_video_FFN 总和作分母，可以得到一个更宽松的上界。首个 dense evaluation 和评分/packing 成本还会缩小净收益。N_v 必须取实际冻结配置在各 evaluation 的 active token 数，不能把原 dense future token 数与已经稀疏后的配置混用。

这些维度与 token 数应来自 matching checkpoint、原 mask/packer 和实际层配置。现有资料没有提供可直接复用的完整逐算子 profile，所以本页不给虚构的占比或速度。

**FLOP 占比不是 GPU 时延占比。** kernel fusion、访存与启动开销可能改变二者的关系。候选定义的 phi_obs 是完整原推理中该 QK/AV 路径的实测耗时份额；只有在其余路径不变、忽略额外开销的假设下，完全删除该路径的理想时延上界才是 `1/(1-phi_obs)`。实际测量还须计入贡献评分、压紧、kernel launch 与其余路径。

若 native attention 把 observation/future/action 的计算融合在同一 kernel 内，必须说明如何归因这一子路径；不能把人为拆开的 instrumentation 路径耗时直接当作原实现的独占耗时。无法可靠拆分时，先用算子计数和同后端的完整路径对照报告实际增减，保持 phi_obs 的归因限制。

若实际余量不足，结束此候选。进一步删除 observation tokens 的 FFN 或改变 video-query 图会改变干预对象，应另立假设与实验，不能包装成当前 mask 的补丁。

配置与入口来源见 [baseline_feasibility.zh.md](<D:/Downloads/Final Year Project/ideaspark_run/training-free-efficient-wam/baseline_feasibility.zh.md>)；本候选的固定规则和独立审查均保存在同一 run 的 `phase2_generate`、`phase3_critique` 目录。
