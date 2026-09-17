# 量化 World Action Models 的因果后果校准

**方法名称：** One-Transition Consequence PTQ (OTC-PTQ)

## 研究动机
对于一个耦合 future-video/action 的 World Action Model（WAM），post-training quantization（PTQ）校准不只是复现局部 tensor 数值。一次 numerical intervention 可能改变模型预测的完整 action chunk；只执行第一个 action 后，simulator 的 next observation 也可能改变。实际瓶颈是选择哪些 weight/activation（W/A）site 保持 high precision，因为 local PTQ discrepancy 并不能稳定识别 contact-transition states 上的 consequence。因此，OTC-PTQ 把完整 action chunk 当作中间对象，把随后产生的一次 control transition 作为可测量边界。

当前 Phase 0 frontier 使这个边界可以被测试。QuantWAMs（arxiv:2607.28405v1）明确处理 joint video/action sensitivity 和 closed-loop protection，QuantWM（arxiv:2602.02110v1）提供不同的 latent rollout/planning sensitivity 分支，Fast-WAM 则提供名为 idm 的 feasibility path，但没有提供 numerical PTQ consequence signal。结合已经说明的 RoboTwin 2.0/LIBERO manipulation split 和 Fast-WAM-only staged validation budget，现在可以进行一个小规模 one-transition diagnostic；前提是 pre-run manifest 中固定 exact Optional IDM checkpoint identity。

几个相邻方向已经有工作覆盖。QuantWAMs 测量 joint video/action sensitivity，并修复 closed-loop protection schedule；QuantWM 研究 weight/activation PTQ sensitivity 和 long-horizon latent planning；Fast-WAM 没有提供用于 one-site W/A protection allocation 的 numerical PTQ consequence signal。更早的 task-consequence 工作——VAML、Model Advantage / Value-Aware Model Learning 和 Deep Task-Based Quantization——已经说明 downstream consequence 可以指导 targeting；SQIL 则通过 saliency-weighted QAT 和 action distillation 覆盖 critical-state protection。因此，更窄的 gap 是：当其他 target site 全部保持 FP16 时，一次 numerical W/A intervention 能否根据 paired full-action-chunk 和 next-observation discrepancy，提供可复现的 one-control-transition proxy。OTC-PTQ 不把这些更广泛的 downstream-consequence 或 critical-state-targeting 原则宣称为新贡献。

如果这个 gap 被填补，领域将得到一个有界 diagnostic，用来识别 local PTQ discrepancy 何时不再预测 contact states 上的一次 control transition consequence。它补充 QuantWAMs 的 closed-loop protection 分支和 QuantWM 的 latent-planning sensitivity 分支。它还可以检验 $C_{r}-ranked$ schedule 是否能在相同 precision budget 下迁移到 held-out contact-sensitive failures 和 next-observation deviation；permutation 可以把 consequence association 与 quantizer 和 site-count controls 区分开来。这不会证明 task-loss identity、full-horizon/replan marginal、joint-low-bit optimality 或 bytes/latency objective。

## 方法
### M1_background
*提供可 replay 的 calibration state，并固定 paired measurement 所需的 exact version 与 configuration boundary。*

1. 先冻结 manifest。 在任何 paired measurement 之前，为每个 state 建立不可变 calibration record，记录 state、checkpoint/configuration identities、controller/RNG/history/cache restoration contract 与 environment identity。 manifest 中必须提供 Fast-WAM 与 Optional IDM 的 exact checkpoint identities 和 exact-match fields；缺失或不匹配时 halt。【作者需决定：补齐不可变 checkpoint identities、revision/digest 与 exact-match fields；不得 fallback 到其他 artifact。】 environment record 还必须固定 CPU thread/process limits、accelerator model、visible-memory class 与 memory-allocation policy。【作者需决定：在 manifest 中补齐这些 execution-environment fields，并明确 A100-40GB 与 80GB-class 边界。】 任何必需 identity 或 restoration check 缺失或不匹配，都必须在建立 paired branches 之前 halt。
   - _为什么：_ 只有两个 branch 都从同一个可恢复 state 开始，paired consequence 才能解释。这样可以保留 version boundary，不会把未经验证的 checkpoint 当作可互换对象。

### M2_one_transition_proxy
*建立有界的 single-site W/A intervention，并计算 one-control-transition $C_{r}$ signal。*

2. 建立一个 versioned site registry，给 weight/activation (W/A) site 配置稳定的 $site_{id}$、module path 或 graph edge、tensor shape、dtype、capture point 与 quantization granularity。 对每个 candidate，先恢复 第1步（先冻结 manifest） baseline，再应用只包含一个 $site_{id}$ 的 mask；在注册的 graph edge 上量化选定的 weight tensor 或截取选定的 activation，其余 target sites 全部保持 16-bit floating-point (FP16)。 记录 operator parameters，并验证 mask 只改变了被选中的 site。 必须把 low-bit operator 声明为 fake quantization（图级仿真）或真实的 low-bit execution kernel，并在 calibration 与 held-out evaluation 中保持同一条已声明路径。 【作者需决定：选择并登记 $b_{lo}$、signedness、rounding、clipping、scale/zero-point、granularity、activation calibration data，以及 fake-quantization 与 real-kernel mode；若声称使用真实 kernel，还要记录 backend/version 并做 runtime verification。】 这只是 single-site、FP16-background 的 intervention，不是 joint-low-bit marginal 或 joint-optimal schedule；QuantWM 与 QuantWAMs 的更广 quantization scope 仍属于 prior-art boundary。

*只在 numerical site r 应用 low-bit operator，其余目标 site 保持 FP16。*
$$ \mathcal{Z}^{(r)}_j=\begin{cases}Q_{b_{\mathrm{lo}}}(\mathcal{Z}_r),&j=r\\\mathcal{Z}^{\mathrm{FP16}}_j,&j\ne r\end{cases} \tag{1} $$

   - _为什么：_ 这会隔离有界的 single-site intervention，也避免把 joint coupling、joint-low-bit marginal effects 或 joint optimality 当作新的 claim。
3. 从匹配的 restored states 分别运行 reference branch 与 single-site branch。执行任何 action 之前保存完整 action chunks，包括 time length、action ordering、valid-step mask 与 dtype；每个 branch 分别恢复同一 state，只执行第一个 action，并保存 immediate next observation 以及 terminal 或 simulator-error flags。定义 $d_{a}(j,r)$：完整 action chunk 的 MSE 除以 $max(sigma_{a}^{2}, epsilon_{a}^{2})$，其中 $sigma_{a}^{2}$ 是 $D_{cal}$ 中 FP16 full action chunks 所有 scalar coordinates 的 population variance；定义 $d_{o}(j,r)$：next observation 的 mean per-pixel absolute difference 除以 $max(sigma_{o}, epsilon_{o})$，其中 $sigma_{o}$ 是 calibration 中 per-pixel standard deviations 的 mean。使用 normalized units 中的 $epsilon_{a}=epsilon_{o}=1e-3$，默认 $lambda_{a}=lambda_{o}=1$，预注册 $lambda_{a}/lambda_{o}$ sweep ${0.5,1,2}$，并将 $C_{r}$ 定义为 $lambda_{a} d_{a}+lambda_{o} d_{o}$ 的 episode mean，在完整 episode 上计算 bootstrap intervals。定义 open-loop local surrogate $L_{jr}$ 为 paired hook outputs 所有 elements 的 MSE，$L_{r}$ 为其 episode mean。仍需明确 valid-element mask、observation preprocessing，以及对 terminal、invalid、simulator-error、missing-observation 与 variable-length rows 一致适用的 inclusion policy。$C_{r}$ 仍是 one-control-transition proxy，不是 task loss、后续 rollout/replan marginal 或 jointly-low-bit marginal。 pixel term 的 valid-element mask 与 observation preprocessing 仍需补齐。【作者需决定：在 held-out evaluation 前记录两者，并对两个 branch 一致执行。】terminal、invalid、simulator-error、missing-observation 与 variable-length rows 也需要同一套 inclusion policy。【作者需决定：说明每类 row 是 retained、masked 还是 excluded，并记录每个 exclusion reason。】

*$d_{a}$ 是完整 action chunk 的 MSE 除以 $max(sigma_{a}^{2}, epsilon_{a}^{2})$，$d_{o}$ 是 next observation 的 per-pixel 绝对差除以 $max(sigma_{o}, epsilon_{o})$。在 normalized units 中使用 $epsilon_{a}=epsilon_{o}=1e-3$，默认 $lambda_{a}=lambda_{o}=1$ 并执行预注册 sensitivity sweeps；$C_{r}$ 按 episode mean 聚合，并在完整 episode 上计算 bootstrap intervals。*
$$ d_a(j,r)=\frac{\operatorname{MSE}(\mathbf a_{jr},\mathbf a_{j0})}{\max(\sigma_a^2,\epsilon_a^2)},\quad d_o(j,r)=\frac{\operatorname{mean}_{\mathrm{pixel}}\left|\mathbf o_{(j+1)r}-\mathbf o_{(j+1)0}\right|}{\max(\sigma_o,\epsilon_o)},\quad C_r=\operatorname{mean}_{j\in D_{\mathrm{cal}}}\left(\lambda_a d_a(j,r)+\lambda_o d_o(j,r)\right) \tag{2} $$

   - _为什么：_ 这定义了一个表示 action-mediated consequence 的 one-control-transition proxy。$C_{r}$ 不是 task loss、full-horizon/replan marginal，也不是 jointly-low-bit marginal。

### M3_matched_protection
*使用 $C_{r}$ 和 local discrepancy 形成 matched protection schedule，同时分开表达不同 budget 的含义。*

4. 使用同一个 site registry 建立两份 deterministic protection schedule：一份按 $C_{r}$ 排序，另一份按 local discrepancy 排序。将 $B_{site}$ 定义为 matched local-fidelity baseline 在 preregistered run manifest 中记录的 nonnegative integer high-precision-site count，并在本方法中固定 B 等价于 $B_{site}$。按 score 降序排列，ties 按 registry order 处理；把恰好 $top-B_{site}$ sites 保持为 FP16，其余 target sites 保持为 $b_{lo}$。$B_{site}$ 的数值和 manifest entry 仍是 open inputs；如果 entry 缺失或 $B_{site}$ 超过 registry 中的 site 数量，就 halt。persistent weight bytes（包括 copies 与 scales）、real kernel 下的 peak live activation workspace、scale/copy overhead 与 measured latency 必须另外记录，不能替代 site-count budget。 numeric $B_{site}$ manifest entry 仍是 open。【作者需决定：提供 nonnegative integer baseline site count；缺失或超过 registry size 时 halt。】deployment bytes、activation workspace、scale/copy overhead 与 latency 仍需独立 measurement protocol。【作者需决定：提供 packing/timing protocol，或明确把这些 outcome 标为 out of scope。】

*按 $C_{r}$ 选择前 $B_{site}$ 个 numerical site；site count 不等同于 bytes 或 latency 目标。*
$$ \mathcal{P}_{C_r}(B_{\mathrm{site}})=\operatorname{TopB}_{\mathrm{site}}(\{(r,C_r):r\in\mathcal{R}\}) \tag{3} $$

   - _为什么：_ 这个比较检验 proxy 是否会在匹配的 precision budget 下改变 protection choice，而不是引入 storage 或 runtime objective。

### M4_validation
*通过 permutation 和 oracle controls 检验 contact-transition divergence 与有界 transfer。*

5. 使用不可变的 第1步（先冻结 manifest） manifest，在命名的 held-out RoboTwin 2.0/LIBERO manipulation split 上运行带有 sigma_shift_FW=1.0 的 idm entrypoint 与 infer_action。 对两份 schedule 使用相同的声明性 $b_{lo}$、B、$B_{site}$、preprocessing、reset procedure、controller restoration 与 branch order。 在查看结果之前，列出 held-out task identifiers、episode 与 initial-state identifiers、seeds、split membership、来自 simulator events 的 contact predicate，以及 task success/failure rule。 【作者需决定：提供 exact split 与 contact/failure definitions，包括 contact 是否必须在第一个 action 之前发生，还是允许在之后发生。】 对每个 paired episode，为两份 schedule 恢复同一个 initial state，运行声明的 evaluation horizon，并记录 task outcome 与带有 aggregation window 的 next-observation deviation。 【作者需决定：固定 idm 与 infer_action entrypoint revisions 和 arguments、action horizon、termination rule，并决定 next-observation deviation 只看 first transition，还是覆盖 later transition/full episode；后者必须与 $C_{r}$ 分开汇总。】 建立 permutation control：在相同 site identifiers 之间重新分配已记录的 $C_{r}$ values，同时保持 score multiset、B、$B_{site}$、tie-breaking 与其他 schedule rules 不变。 指定的 oracle simulator-state branch 只能作为标注清楚的 positive-control branch 运行。 【作者需决定：指定 permutation seed 与 replicate/list、oracle state source、branch insertion point 以及它所控制的 comparison；不能用 oracle 替代真实 association。】 保持解释边界：$C_{r}$ 仍是 one-control-transition ranking proxy；task failures 与 full-horizon summaries 是独立的 downstream outcomes，不是 task loss，也不是 full-horizon/replan marginal。 QuantWM 与 QuantWAMs 继续保留其更广的 world action model (WAM) quantization boundaries；VAML、Model Advantage / Value-Aware Model Learning、Deep Task-Based Quantization 与 SQIL 继续代表 broad consequence、value-aware、task-based 或 critical-state-protection principles 的 prior-art boundary。

*按 state subset 比较 local score 与 $C_{r}$ 的 rank correlation，预期差异集中在 contact-transition states。*
$$ \Delta\rho=\rho_{\mathrm{noncontact}}-\rho_{\mathrm{contact}},\quad \rho_s=\operatorname{Spearman}(\{L_r\}_s,\{C_r\}_s) \tag{4} $$

   - _为什么：_ contact-transition subset 和 permutation 是 mechanism tests：candidate 预测真实 $C_{r}$ association 会带来 ranking divergence 和 transfer，而 permutation 后这个 advantage 会消失。oracle branch 用来检查 calibration noise。

