# 冻结联合 WAM 的动作上下文来源掩码

**方法名称：** Action-Context Source Masking (ACSM)

## 研究动机
在冻结的联合 World Action Models（WAMs）中，减少 future-side computation 并不能说明每次 action update 是否可以不再读取完整 observation。WAMs 会沿耦合的 transformer 路径反复更新 action chunks，同时使用 observation 和保留的 future tokens。每个 active native action solver update 仍可能读取全部 observation tokens。现有证据既没有测出 action quality 对这些 source interactions 的依赖，也没有测出它们占完整推理耗时的比例；只看保留了多少 tokens 或端到端 latency，无法判断这条路径是否值得优化。

近来的 WAM 研究把问题收窄为一个推理时测试：Sparse-WAM（2026-09）减少 future-side computation，WAMachine（2026-09）研究状态与 residual reuse，Efficient-WAM（2026-06）则区分 video 和 action 的更新计划。仍未回答的问题是 action-query 对 observation 的敏感性及其成本。本方案会 profile 这条路径，并在冻结模型上比较保留 QK/AV（query-key/attention-value）边数相同的 masks。这个计划并未确认所需 checkpoint、source-provenance 接口或 compact kernels 已经可用。

已有方法停在相邻机制上。Sparse-WAM 选择 future visual tokens，并复用已选区域或被省略区域的预测；它仍保留全部 observation/action tokens，并重新计算 actions。WAMachine 会重映射 trajectories、在执行期间重新绑定 observations，并在 probe 后复用中间 residuals；但它没有单独检验每次 action update 会读取哪些当前 observation-source edges。Efficient-WAM 减少 video updates 并缓存 video keys 和 values，同时继续更新 actions；但它没有测量这些 action updates 对 observation 的敏感性和相关成本。C3ache 跨 chunks 复用 action-expert denoising residuals；FBFM 通过 feedback 注入新状态约束和已提交 action 的重叠部分；GlanceWAM 则把稀疏 future imagination 与执行并行，并针对 staleness 训练。这些工作都没有在 frozen inference 中给 observation sources 排序并移除对应 edges。

如果结果为正，它会分别说明：对每个受测 frozen checkpoint，能否在满足原 task-success gate 的同时减少 observation-source edges，以及实测 headroom 是否足以降低完整推理 latency。这会补上一项针对 WAM action conditioning 的测试，并与已有 scoring recipes 做保留边数相同的比较。若结果为负，结论只限定于本次测试的机会，不代表普遍不可能。

## 方法
### M0_background
*分别固定各 baseline，并先测量目标路径是否有足够 headroom。*

1. 分别冻结 FastWAM-Joint/LIBERO 与 Cosmos3-Edge-Policy-DROID/RoboLab 的指定 checkpoint、native solver schedule、task configuration 和 backend。注明匹配的 FastWAM-Joint checkpoint 尚未确认。Motus/RoboTwin 2.0 只能作为有条件的仿真 fallback，不能记作匹配的主 baseline。
   - _为什么：_ 比较对象是一个确定的 frozen checkpoint 和 native evaluation path；换 weights、schedule、task 或 backend，测量对象也会改变。
2. 按计划检查 native paths：FastWAM 的 `fastwam_joint.py`、`fastwam.py`、`mot.py`，以及 Cosmos 的 `omni_mot_model.py` 和 RoboLab policy-server entry point。分别测 action-query 到 observation 的 query-key/attention-value（QK/AV）、其他 attention、QKV、feed-forward network（FFN）和 mask/packing 的耗时与 FLOPs。计算 $`\phi _{obs}`$（目标 QK/AV 占完整推理耗时的实测比例）及理想 Amdahl 上界。如果可达到的节省无法满足原 latency gate，就停止。

*$\phi _{obs}$ 是 action-query-to-observation QK/AV 在完整推理中的实测时间占比；该理想 Amdahl 上界尚未扣入评分、压紧、kernel 和其他模型路径的代价。*
$$ \mathrm{speedup}_{\max}=\frac{1}{1-\phi_{\mathrm{obs}}} \tag{1} $$

   - _为什么：_ action update 的次数不能说明目标路径占完整推理耗时多少；这项 profile 用来判断删掉这些 edges 是否可能影响完整路径 latency。

### M1_source_scoring
*按 source 对 action 的 output-projected contribution 排序，再用单独留出的 episodes 固定 mask。*

3. 只使用 token packer 已有的 camera/view 或其他 source provenance；如果没有提供，就停止，不自行推断分组。在每个 chunk 首次进行 dense native action evaluation 时，即 $`k_{ref}`$，读取 native attention weights $\alpha $、values V 和 output-projection 的各 head slices $`W_{O}`$。对每层每个 source group，把该组对 action queries 的 contribution 经 native output projection 后取范数并求和，得到 $`r_{lg}`$；再在层内归一化为 $`\rho _{lg}`$。若归一化分母为零，就保留全部 groups。

*对每层和 observation-source group，将该组经 native output projection 后对 action queries 的 attention contribution 取范数并求和，得到 $r_{lg}$。*
$$ r_{lg}=\sum_{q\in Q_l}\left\|\sum_{h\in H_l}W^{l}_{O,h}\left(\sum_{j\in g}\alpha^{l}_{h}(q,j)V^{l}_{h}(j)\right)\right\|_2 \tag{2} $$


*$\rho _{lg}$ 在层内按所有 observation groups 的总贡献归一化；分母为零时按规则保留全部 groups。*
$$ \rho_{lg}=\frac{r_{lg}}{\sum_{g'\in G_o}r_{lg'}} \tag{3} $$

   - _为什么：_ 这个分数衡量每个 observation source 经 output projection 后对 action queries 的贡献；dense reference 的输入、weights、image、instruction、future/action queries 和 training objective 都保持固定。
4. 在留出的选择 episodes $`D_{select}`$ 上，按 $`r_{lg}`$ 降序排列 groups，并用 native source order 打破并列。选择 $\epsilon \in [0,1)$，使 $`S_{l}(\epsilon )`$ 成为累计 $`\rho _{lg}`$ 达到 $1- \epsilon $ 的最短前缀。在通过原 frozen task-success gate 的 masks 中，选实测 full-path latency 最低者。如果没有预先设定的 gate，就报告 success-latency 曲线，不宣称通过。

*$g_{i}$ 按 $r_{lg}$ 降序排列，并以 native source order 稳定打破并列；$S_{l}$ 取累计贡献达到目标的最短前缀，$\epsilon $ 的可选范围为 [0,1)。*
$$ S_l(\epsilon)=\{g_1,\ldots,g_m\},\quad m=\min\left\{t:\sum_{i=1}^{t}\rho_{l g_i}\geq 1-\epsilon\right\} \tag{4} $$

   - _为什么：_ 原 success gate 限定可以删掉多少 context；$`D_{select}`$ 与后续 paired evaluation 分开，避免用判定候选效果的 outcomes 来选择 $\epsilon $。

### M2_sparse_update
*用真正跳过被删 action-to-observation 运算的 kernel 应用 frozen mask。*

5. 在同一 chunk 后续的 native action-solver evaluations 中，把 action queries 和保留的 observation keys/values 送入真正的 variable-length 或 block-sparse kernel，并验证被省略的 QK/AV 运算确实没有执行。只对保留的 keys 重新计算 softmax，再把 attention output 送入原 residual/transformer blocks；由变化后的 action hidden state 和 velocity 更新原 flow state。video-query paths、future/action interactions、FFN 和 solver schedule 都保持 native。
   - _为什么：_ 这一步在 action branch 中真正跳过 edges，可以区分实际省算和仅加 dense mask。重新计算 attention 可能改变 action trajectory；该方法不保证 fidelity 或 task success。

### M3_validation
*将 mask 与 dense inference 和同成本 controls 比较；真机阶段须满足前提后再考虑。*

6. 在没有用于选择 $\epsilon $ 的配对评估 episodes $`D_{eval}`$ 上，对比 dense inference、按 $`\rho _{lg}`$ 排名的 masks，以及保留边成本相同的 source-group-permutation masks。还要比较 native attention-mass、ToPi/VATP value-norm 和 CAPA output-projected-score controls。每组比较使用相同 frozen checkpoint、$`k_{ref}`$ dense forward、精确保留边预算 $`B_{l}`$ 和 compact kernel。对评分 controls，先将 token scores 在已有 groups 内求和，再按 group score 排序并以 native source order 打破并列；先保留完整的 ranked groups，再按相同 token score 排序 boundary group 的 tokens 并取前缀，直到恰好保留 $`B_{l}`$ 条 edges；token 并列按 native token order。只有存在不同且合法的等成本 mask 时才运行 permutation，否则报告 unavailable。记录 task success、逐路径 FLOPs、包含 scoring、packing、kernel launch 和其余 model paths 的完整 latency、action-chunk deviation、first action divergence，再应用原 frozen gates。

*$B_{l}$ 表示保留的 action-query-to-observation edge 数，用于匹配排名 mask 与各评分对照。*
$$ B_l=|Q_l||H_l|\sum_{g\in S_l(\epsilon)}|g| \tag{5} $$

   - _为什么：_ Permutation 检验固定成本后 source assignment 是否重要；评分对照检验 $`\rho _{lg}`$ 是否比已有排序 recipes 带来额外 task-action value。质量与成本是独立证据；只有 $`\rho _{lg}`$ 在原 gates 下胜过所有同成本评分对照，才支持 WAM-specific score advantage。
7. 只有仿真通过 frozen gate，而且匹配的 AgileX Cobot Magic policy/checkpoint、demonstrations/data、source implementation 和同类型 hardware 均已确认后，才考虑真机阶段。baseline 原有 post-training 单独核算；mask 本身仍是 training-free，也不更新 weights。
   - _为什么：_ 真机前提尚未确认，因此仿真或 recorded-data 结果不能代替 closed-loop robot 结果。

