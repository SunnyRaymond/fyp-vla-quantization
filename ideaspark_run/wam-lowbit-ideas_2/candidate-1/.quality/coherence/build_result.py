from pathlib import Path
import json

base = Path(r"D:\Downloads\Final Year Project\ideaspark_run\wam-lowbit-ideas_2\candidate-1\.quality\coherence")
script = (base / "t2_trace.py").read_text(encoding="utf-8-sig")
stdout = (base / "t2_trace.stdout.txt").read_text(encoding="utf-8-sig")

steps = [
    ("S1", "1. 载入 Fast-WAM 官方 Optional IDM checkpoint、first_frame 路径、LIBERO-Goal evaluator 和其固定 controller 配置；冻结所有模型权重、动作归一化、chunk/replan 设置与 generation seed 规则。", ["官方 checkpoint F_BF16: 参数张量", "evaluator E: observation->chunk 的调用路径", "controller 配置 C: 状态到分支/动作映射", "seed 规则"], ["冻结的 BF16 reference F_BF16", "固定 E、P_h、C_c、动作归一化和 seed 规则"]),
    ("S2", "2. 用 BF16 reference rollouts 记录现有 evaluator 会读取的 (observation, controller state, executed prefix length, seed)，按 trajectory 划分 PTQ calibration 与 held-out fixed-input validation，避免相邻帧泄漏。", ["冻结 E/F_BF16", "BF16 evaluator 轨迹"], ["两组按 trajectory 不相交的记录 d_i=(o_i tensor, c_i state, h_i integer, xi_i seed)", "h_i 对应当条样本实际执行的 prefix"]),
    ("S3", "3. 缓存 BF16 reference 的完整输出及精确 controller command；从 evaluator 的分支标记构造 branch signature，并用 calibration commands 计算每个连续执行维度的标准差。", ["calibration/held-out d_i", "F_BF16 outputs: [L,d]", "固定 P_h 与 C_c"], ["reference full chunks a_i*: [L,d]", "reference commands u_i*: [h_i,d]", "reference branch signatures b_i*: branch tuples", "连续维 reference commands v_i*: [h_i,d_cont]", "每维 s_j: 标量标准差；零标准差用 evaluator 原生单位尺度"]),
    ("S4", "4. 将所有执行图 Linear 转成 W4A8：weights 用 signed per-output-channel INT4，activations 用 per-token INT8；按后端规则保留 non-Linear 的 BF16/FP32 算子，列出任何因 kernel 不支持而未覆盖的 Linear。", ["BF16 执行图及每个 Linear 的权重/激活张量", "后端 kernel 支持规则"], ["量化执行图：各 Linear 为 per-output-channel signed W4 / per-token A8", "BF16/FP32 non-Linear 运算", "unsupported Linear 列表；若非空则不能称完整覆盖"]),
    ("S5", "5. 用标准局部重建 PTQ 参数初始化每个 Linear 的 scales、clipping bounds 和 activation zero-point；只允许这些量变化，不训练模型参数，也不增加保护 branch。", ["量化执行图", "calibration 权重/激活范围", "标准局部重建 PTQ 规则"], ["初始量化参数 theta_0: 每 Linear 的 weight scale/clipping 与 activation scale/clipping/zero-point", "模型参数保持冻结"]),
    ("S6", "6. 对每层依次枚举从其校准权重/激活范围得到的合法量化候选；每个候选都跑完整量化 WAM，再用同一记录的 prefix 和 controller state 得到 command，计算 branch mismatch rate 与连续 command normalized squared error。", ["当前 theta", "每层有限候选集 Theta_l", "calibration d_i", "reference b_i*, v_i*, s_j", "完整 W4A8 前向输出"], ["每候选的 u_i(theta), b_i(theta), v_i(theta)", "J_cal(theta)=(branch mismatch rate, normalized continuous command squared error): lexicographic pair"]),
    ("S7", "7. 仅在全 calibration objective 词典序严格下降时接受该层更新；重复 layer sweep，直到一轮无更新，并冻结所得 W4A8 配置。", ["当前 theta", "各层有限 Theta_l", "固定 J_cal 评分器"], ["更新后的 theta 或原 theta", "无更新 sweep 时得到 theta_final"]),
    ("S8", "8. 在 held-out BF16 reference records 上用原 controller 重放最终 W4A8 模型，报告 full-chunk deviation、branch mismatch、executed-command error、有效 Linear 位宽及精度例外占用；不从这些固定输入指标推出闭环 success 或速度收益。", ["theta_final", "held-out d_i 与 BF16 reference outputs/commands", "原 P_h/C_c"], ["held-out full-chunk deviation、branch mismatch、executed-command error", "有效 Linear 覆盖/精度例外统计"]),
    ("S9", "9. 部署路径删除 teacher、calibration search 与 BF16 correction；若 backend 没有 Fast-WAM-compatible packed W4A8 kernel，则只报告 quantizer proposal，不声称真实 native low-bit latency。", ["theta_final", "适配的部署 backend/kernel"], ["运行期仅 W4A8 模型及原 controller", "无 teacher/search/correction branch", "若无兼容 kernel，则仅有 quantizer proposal"]),
]
formal = []
for sid, quote, consumes, produces in steps:
    formal.append({"step": sid, "consumes": consumes, "produces": produces,
                   "note": "逐字引用：" + quote, "status": "instantiated"})

trace = {
  "likely_fatal_step": {
    "step": "S6",
    "why_selected_before_execution": "此步把变量长度 prefix、controller branch signature 与连续误差压成同一 lexicographic score，是从‘实际已执行 command’到可优化目标的唯一连接；若 branch signature/reference 或跨不同 h_i 的聚合不可执行，核心机制就无法评分。",
    "attack_and_result": "用纯合成全模型输出实际执行 S6–S7 的单坐标有限候选搜索，并以两种可辩护聚合阅读重跑；两者均可计算且都选 B。实际 evaluator 与采样 replay 是否精确匹配仍是 untested premise，不由该合成结果验证。"
  },
  "formalized_procedure": formal,
  "parameter_audit": {
    "result": "method fields 中未发现未绑定的行为性数字参数。INT4/INT8 是量化定义；h_i 是记录的输入；候选数/样本数没有被作为算法参数硬编码，校准规模属 evaluation scale。每层候选集由 backend 根据 calibration ranges 生成的有限合法集合。",
    "unbound_parameters": []
  },
  "dual_readings": [
    {
      "term": "branch mismatch rate",
      "reading_a": "逐记录比较完整 ordered branch tuple，rate=mean_i 1[b_i(theta) != b_i*]。",
      "reading_b": "将所有已执行 branch bits 池化，rate=不匹配 bit 数/已执行 branch bit 总数。",
      "execution": "T2、T3、T5 均在两种 reading 下执行；本合成 T2 的主候选均选 B，负对照后均选 C，T5 full-chunk naive 均选 A。数值 rate 可不同（例如 C 的主集为 0.5 vs 0.3333），本实例排序相同。",
      "finding_tag": "reading_robust: 文本未确定 rate 的分母/汇总单位这一歧义在两种 reading 下都存在；本实例没有证明两种 reading 对所有输入排序相同。"
    },
    {
      "term": "normalized continuous command squared error",
      "reading_a": "每记录先对实际执行维度求和，再对记录取均值；符合 R_cmd 中的 sum_j。",
      "reading_b": "对所有已执行标量维度池化后取均值。",
      "execution": "T2、T3、T5 均在两种 reading 下执行；主校准都选 B，打乱配对后都选 C，full-chunk baseline 都选 A。A 的主集误差为 0.019025 vs 0.012683。",
      "finding_tag": "reading_robust: 变量 h_i 下文本未指定跨记录/维度的有限样本聚合。"
    }
  ],
  "estimand_match": {
    "declared_estimand": "R_cmd(theta)=E_{(o,c,h,xi)~D_ref}[lex(branch mismatch, sum_j normalized continuous command squared error)]。",
    "constructed_quantity": "S6 在 paired BF16-reference calibration records 上计算同一 controller-prefix branch/continuous command distortion；S8 在 trajectory-disjoint held-out BF16 reference records 上报告同一固定输入 quantity。",
    "verdict": "quantity/process/conditioning match at record level; held-out empirical mean is not by itself a population-generalization guarantee, which the candidate explicitly disclaims by not assuming IID/full coverage."
  },
  "premises_left_untested": [
    "官方 evaluator/controller 的 P_h、C_c 与记录的 c_i/h_i 可精确重放；若状态历史未记录则退回 stateless map。",
    "同一 observation/seed 下 BF16 与 W4A8 replay 提供可配对 generation noise。",
    "LIBERO-Goal reference trajectory 上实际存在足够 clip/gripper branch events，且 branch-aware 目标会降低未见 reference-record distortion。",
    "目标 backend/kernel 能覆盖所有执行图 Linear 并具备所称 W4A8 部署路径；否则候选自身限定为 proposal。"
  ],
  "dry_run": {
    "instance": "2 条 calibration + 2 条 trajectory-disjoint held-out synthetic records；reference chunks a=[0,0,0], b=[2.9,0,0]；controller c1/c2 对称 clip limit 分别 1/3；prefix h_a=1,h_b=2；三组完整 synthetic WAM 输出 A: a=[0.2,0,0], b=[2.7,0,0]; B: a=[0,4,0], b=[2.9,0,0]; C: a=[0.01,0,0], b=[3.2,0,0]。所有输出仅为量化候选规则的合成输入，非模型输出生成实验。reference s_0=1.45，s_1=0 使用 native scale 1。",
    "execution": {"mode": "executed", "script": script, "output": stdout},
    "computed_quantities": [
      {"quantity": "reference scales", "value": "s_0=1.45; s_1=1.0 fallback", "arithmetic": "population SD([0,2.9])=1.45; dimension 1 has only reference 0, so the declared zero-SD native-scale fallback is 1."},
      {"quantity": "main command objective (record-mean reading)", "value": "A=(0,0.01902497); B=(0,0); C=(0.5,0.00240190)", "arithmetic": "A: mean[(0.2/1.45)^2,(−0.2/1.45)^2]=0.01902497; B has zero executed errors; C has one clipped branch among two records and mean normalized squared error 0.00240190."},
      {"quantity": "main objective (pooled reading)", "value": "A=(0,0.01268331); B=(0,0); C=(0.333333,0.00160127)", "arithmetic": "Pool executed branch bits/scalar command errors; both readings select B."},
      {"quantity": "full-chunk naive scores", "value": "A=0.0133333; B=2.6666667; C=0.0150167; naive selects A", "arithmetic": "mean over two records and 3 dimensions; B's unexecuted a suffix error contributes 4^2 while h_a=1."},
      {"quantity": "strict coordinate search", "value": "A→B then no update; final B", "arithmetic": "candidate order A,B,C; J(B)<lex J(A), while C has nonzero first component; next sweep accepts none."},
      {"quantity": "paired negative control", "value": "swap calibration (c,h): A→C; original selected B, shuffled selected C", "arithmetic": "shuffled calibration s_0=0.5; held-out raw row-mean squared command error 0→0.00505, branch mismatch 0→0.5; shuffled-scale normalized row-mean error 0→0.0202."}
    ],
    "anomalies": []
  },
  "negative_control": {
    "executed": True,
    "moved_outcome": True,
    "evidence": "按 falsification_prediction 将两个 calibration observation 的 controller state/prefix 作一次 swap，仍在原配对 held-out records 评估。校准选择从 B 切到 C；held-out raw mean squared executed-command error 从 0 变为 0.00505，record branch mismatch 从 0 变为 0.5；用各自 calibration s_j 归一化时 row-mean error 从 0 变为 0.0202。此结果是本合成实例的 instance-contingent 结果，且归一化尺度随置换从 1.45 变为 0.5；raw error 同向上升，不能据此宣称真实 WAM 控制效果。"
  },
  "degenerate_probes": [
    {"probe": "empty calibration set", "behavior": "score undefined: record rate divides by zero and every s_j is absent; script returns an explicit undefined result. No empty-set handling is specified.", "finding": "只在零条校准记录这一参数取值下不可评分；合法实验需有 records，prompt 明令不能因缺少样本数而 fault；parameter note, not blocking."},
    {"probe": "k=0", "behavior": "candidate declares no k/count parameter or top-k operation; not applicable.", "finding": None},
    {"probe": "all-identical reference commands", "behavior": "s_j=0 invokes the explicitly stated native-unit fallback; objective remains finite (A error 0.04 on this two-identical-a probe).", "finding": None},
    {"probe": "ties exactly at controller threshold", "behavior": "illustrative controller uses strict outside-limit branch; x=1 at limit emits command 1 and branch=false; score remains defined. The real evaluator's tie convention is part of the untested exact-C premise.", "finding": None},
    {"probe": "single calibration record", "behavior": "zero SD uses native scale; score halts with finite result. An independent trajectory-held-out split cannot be demonstrated from a single trajectory, which is an evaluation-size choice, not an algorithm impossibility.", "finding": None},
    {"probe": "maximum allowed candidate count", "behavior": "spec states no numeric maximum, only a finite backend-generated set. The one-coordinate search terminates for any finite set N; no largest N exists in the written spec to instantiate.", "finding": None}
  ],
  "claim_step_map": [
    {"claim": "将 branch mismatch 与连续 command error 词典序优化，因此不需任意权重折算。", "established_by": "S6", "strength_grade": "established", "assumptions_missing": [], "arbitration": "argument", "measured": None},
    {"claim": "有限候选集 + 严格词典序下降保证停止，但不保证 global optimum。", "established_by": "S6-S7", "strength_grade": "conditional", "assumptions_missing": ["候选配置空间固定且有限；同一固定记录/seed 对每个配置给出可重复的 objective。候选集有限和 seed 固定已写，后端分数可重复仍需兑现。"], "arbitration": "proof-obligations", "measured": None},
    {"claim": "所有执行图 Linear 用 W4A8 且不保留 BF16 Linear side branch。", "established_by": "S4", "strength_grade": "conditional", "assumptions_missing": ["目标 backend 对所有执行 Linear 均提供合法 W4A8 kernel；候选已声明若未覆盖则不得声称完整覆盖。"], "arbitration": "argument", "measured": None},
    {"claim": "校准/held-out command 是固定 evaluator 的精确 controller-prefix 输出。", "established_by": "S2-S3,S8", "strength_grade": "empirical", "assumptions_missing": [], "arbitration": "argument", "measured": None},
    {"claim": "held-out executed-command error 低于强 W4A8 baseline，且 full-chunk 排序会在 suffix/branch 条件反转。", "established_by": "S6,S8", "strength_grade": "empirical", "assumptions_missing": [], "arbitration": "argument", "measured": None},
    {"claim": "校准 estimand 仅为 BF16 reference-trajectory fixed-input distortion，不推出闭环 success、跨 rollout fidelity 或 native latency。", "established_by": "S2,S8-S9", "strength_grade": "established", "assumptions_missing": [], "arbitration": "argument", "measured": None},
    {"claim": "部署时移除 teacher、search、BF16 correction branch。", "established_by": "S9", "strength_grade": "established", "assumptions_missing": [], "arbitration": "argument", "measured": None}
  ],
  "naive_comparison": {
    "declared_branch": "(i) false-premise (implicit: Block 2 frames full-chunk score as failing to distinguish executed prefix/controller branch from unused suffix; no literal (i) label is present)",
    "naive_version": "同一 W4A8 backend、冻结权重、相同 reference records 和相同有限候选/前向预算，只按完整 chunk 的 BF16 reconstruction MSE 选候选；不计算 P_h/C_c command 或 branch signature。",
    "naive_fairness": "Respect W4A8/frozen-backbone deployment, same calibration information, candidate set and full-model candidate evaluations; simpler scoring omits controller mapping and branch-signature composition, so it does not relax budget/information/deployment constraints.",
    "declared_naive": {"version": "Block 2 的 standard per-layer weight reconstruction / activation-range calibration，以 full-chunk MSE 选择 clipping scales。", "instance_behavior": "在此有限合成候选上，full-chunk objective 选 A (0.0133333)，其 executed objective=(0,0.01902497)。", "agrees_with_constructed": "yes: 两者都按 full-chunk MSE 选 A；该同向结论只对应此 synthetic instance。"},
    "instance_behavior": {"naive": "A, full-chunk MSE=0.0133333; executed normalized error=0.01902497。", "mechanism": "B, branch rate=0; executed normalized error=0。", "divergence": "A vs B; candidate objective reduces normalized command error 0.01902497→0 on this constructed instance.", "kind": "instance_contingent"},
    "naive_kind": "alternative_design",
    "identity": "differs",
    "verdict": "confronts_obstacle",
    "reasoning": "On this instance, the same-budget full-chunk baseline spends its score on an unexecuted suffix and selects A, while prefix/controller scoring selects B with zero executed error. This is a constructed ranking reversal, not evidence of real WAM benefit; alternative-design verdicts are instance-contingent."
  }
}

report = {
  "trace_report": trace,
  "contract_version": 2,
  "base_candidate_sha256": "1888353c44fef3eae8763a30b89732211b479d117e37f6340af5fadd9fffe3d6",
  "verdict": "findings",
  "unrepaired": [],
  "suggested_repairs": [
    {
      "kind": "wording",
      "op": "append_sentence",
      "field": "core_mechanism",
      "value": "For each record, define the reference signature b_i* as the ordered tuple of controller branch flags from C_c_i(P_h_i F_BF16(o_i;xi_i)); define branch mismatch rate as mean_i 1[b_i(theta) != b_i*]. Define the finite-record continuous term as mean_i sum over executed continuous dimensions j of ((v_i(theta),j-v_i*,j)/s_j)^2, with s_j computed over calibration commands that execute dimension j and the stated native-scale fallback when s_j=0.",
      "delta_summary": "T1 found that branch-mismatch denominator and ragged-prefix continuous-error aggregation admit record-level and pooled readings; the synthetic T2/T5 choices agreed, but the written finite-sample objective is not uniquely specified."
    }
  ],
  "applied_revisions": []
}

result = {
  "contract_version": 2,
  "request_id": "bb05ffcb-862a-4b6a-8223-9e81e1fb723b",
  "artifacts": {"phase2_coherence/phase2_coherence_output.json": report}
}
(base / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("wrote result.json; report chars", len(json.dumps(report, ensure_ascii=False)))
