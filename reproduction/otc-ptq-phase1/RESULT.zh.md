# OTC-PTQ 第一阶段实测结果（2026-09-09）

**结论：测量流程通过；暂不建议按当前 OTC 配方扩大实验。** 在本次固定的 8 个候选位置、保护 2 个位置的筛选中，OTC 和 Local MSE 选出了同一个集合。OTC 的分数几乎完全由 observation 项决定，且独立 CHECK 上的排名稳定性较弱。现有证据不足以支持投入完整 suite 或四个 suites。

这不是“所有 OTC 思路均无效”的结论，也没有证明 action-only 能提升最终成功率。它是一次有边界的可行性筛选，已经提供了停止当前配方扩跑的理由。

## 实际完成的实验

- 基于已有 FastWAM Optional IDM checkpoint，运行真正的 `idm` mode；源码 revision `7faa71108368fbb3b6885649f112af607427a2d4`，`sigma_shift=1`、`compile=false`。参考模型实际 dtype 为 **BF16**。
- 只使用 LIBERO-Goal 的两个任务：task 0 “open the middle drawer of the cabinet”；task 1 “put the bowl on the stove”。没有运行完整 Goal suite，更没有运行四个 suites。
- 每个任务 4 个初始状态/episodes，episode 0、1 为 CAL，2、3 为 CHECK；每个 episode 取 4 个状态，共 32 个状态。CAL 和 CHECK 按 episode 分离。
- 在 video/action 两个分支的 block 0、10、20、29 的 `self_attn.q` Linear weight 上分别施加 single-site W4 fake RTN，其余权重保持 BF16。共 **32 × 8 = 256 条有效评分**。
- W4 为 symmetric per-output-channel，范围 [-7,7]、scale=maxabs/7、ties-to-even；每次干预后恢复原始权重。没有训练、没有真实低比特 kernel，也没有同时量化全部候选位置。
- 比较 Local MSE、normalized Local MSE、action-only、observation-only、默认 OTC；另做 first-action-only、固定权重比例 0.5/1/2、CAL bootstrap、leave-one-episode-out 和 permutation diagnostics。

## 关键证据

以下 V0/V10/V20/V29 指 video 分支相应 block 的 q projection。CAL 决定候选保护集合，CHECK 不参与选择。Spearman 衡量 8 个位置在 CAL 与 CHECK 之间的排名一致性。

| 评分 | CAL top-2 集合 | CAL–CHECK Spearman | 所选位置在 CHECK 上的平均最大物体位置变化 |
|---|---|---:|---:|
| Local MSE | V29、V20 | 1.000 | 1.946 µm |
| normalized Local MSE | V29、V20 | 1.000 | 1.946 µm |
| action-only | V10、V0 | 1.000 | 3.164 µm |
| first-action-only（辅助） | V10、V0 | 0.952 | 3.164 µm |
| observation-only | V20、V29 | 0.619 | 1.946 µm |
| 默认 OTC | V20、V29 | 0.619 | 1.946 µm |

最后一列是“单独量化这些位置有多扰动物体”的 sensitivity diagnostic，数值较高表示更敏感；**不是保留这些位置后的政策表现，不是成功率，也不是已验证的保护收益**。这里只按位置数量 B=2 比较，不等价于跨分支的固定 bytes 预算。

1. **没有额外的 top-2 分配。** 默认 OTC 与 Local MSE 的顺序不同，但集合相同；在相同规则下据此构造 B=2 保护配置，会得到同一配置。本阶段未实际执行联合低比特配置。
2. **默认分数由 observation 主导。** CAL 平均 d_a=3.480e-6、d_o=1.502e-3，observation 占平均 C 的 99.769%；CHECK 为 99.785%。权重比例 0.5/1/2 的 top-2 仍相同。
3. **存在任务间不稳定性。** OTC 在 task 0 的 CAL–CHECK Spearman 为 0.524，在 task 1 仅 0.071；action-only 分别为 0.976 和 0.905。合并任务得到的 0.619 不能掩盖 task 1 的变化。
4. **物理诊断未提供 OTC 的额外优势。** action-only 选出的集合在 CHECK 上对应更大的物体位置扰动；OTC 与 Local 相同。全部 256 条只有 4 条量化后的 contact 集合不同，CHECK 只有 1 条，因此 OTC 所选位置的 1/32 contact difference 不能作为稳健优势；Local 同样捕捉该事件。
5. **小样本敏感性仍存在。** CAL 的 4 个 leave-one-episode-out 折中，OTC 有 1 折将 V29 换为 V10。分 task 对完整 episode 做 2000 次 bootstrap，V20 入选率 100%，V29 约 76.1%。这些是描述性诊断，不支持广泛泛化声明。

CHECK 上对评分与位置的对应关系做 2000 次 permutation control，OTC 所选集合的物体扰动位于约第 17.5 百分位；action-only 位于最高端。这里报告的是描述性位置，不是预注册显著性检验或 p-value。Contact 分层也没有显示“OTC 相对 Local 的差异专门在接触状态更强”的一致模式。

## 测量是否可信

256 条评分全部通过原始 reference、重复 reference 和 identity 干预检查；记录的 action、下一图像、qpos/qvel、contact 对照误差均为 0（各项按自身单位分别检查）。额外 CPU-only replay 对全部 32 个状态的原始下一步图像、qpos/qvel、contact 和 terminal 进行了验证。

恢复过程中修复了 observable cache、controller/gripper warmup、历史 buffer dtype 和 object visual flags。3/32 个状态在恢复并调用 forward 后，派生的原始 contact cache 不完全相同；原始下一步的 functional replay 仍严格一致。这一区别符合 MuJoCo 在 integration 后保留上一计算阶段派生数据的行为，且所用 robosuite 每次控制前会重新 forward。没有以放宽下一步误差阈值来绕过失败。参见 [MuJoCo 的 mjData consistency 说明](https://mujoco.readthedocs.io/en/stable/computation/index.html#consistency-in-mjdata)。

Contact 指过滤 support/robot-self 后的候选 geom-pair contacts，包含 object-object contacts，不是人工标注的完美“任务关键接触”。qpos norm 包含平移、角度和 quaternion 等混合坐标，**不能写成米**；表中的 µm 来自明确的 object body position，二者已分开记录。

## 假设解释与待定项

**已有证据：** 当前归一化后两项仍相差很大；OTC top-2 没有区别于 Local，CHECK 稳定性不足。

**可能解释，尚未验证：** action MSE 与 image L1 对微小误差的缩放不同，仅用 CAL 数据方差归一化不一定能平衡二者。下一图像也可能包含对任务不重要的变化。当前 single-site 扰动较小、仅两个任务，可能不足以代表联合量化后的长期失败。

**待定实现，未运行：** 若继续，应先明确两项的尺度及为什么某种组合能预测任务相关变化，冻结新配方后再做独立验证；可优先复用已保留的数据进行离线诊断。若使用 CHECK 来修改配方，该 CHECK 之后只能视为开发数据，必须另留新 episode 作确认。更强扰动、其他 sites、联合量化、长时成功率和真实低比特吞吐都需要另立实验，不能由当前结果推得。

因此不建议现在直接扩到完整 LIBERO-Goal 或四个 suites，也不建议仅为寻找正结果反复调权重。先解决分数定义和独立诊断的对应关系，再决定是否投入下一阶段。

## 资源与完成状态

| 项目 | 实测 allocation 用量 |
|---|---:|
| 全部 GPU 作业，含失败调试和 probes | **1.165 A100-hours**（69.9 GPU 分钟） |
| 成功的正式 collection + scoring | 0.6697 A100-hours（40.18 GPU 分钟） |
| CPU-only 验证作业 | 25.52 allocation 分钟 |

从首次 GPU 提交到最后评分结束约 **1 小时 48 分钟**，包含中间调试、CPU 检查与等待，不含此前资源准备。多个作业并行时，allocation-hours 相加，不能直接当作总日历时间。用量按 PBS 请求的 GPU 数 × 实际 walltime 计算；不是对最终 billing credits 的核账。全部 19 个记录作业已结束，无本阶段继续运行的作业。

成功 scoring 作业为 `16216156.pbs101`（task 0）与 `16222339.pbs101`（task 1）。大张量留在 compute 侧；本地保留小型评分 JSON、日志、运行配置及源码快照。全过程未重新运行 IdeaSpark pipeline。

完整数值见 [summary.json](summary.json)，逐作业用量见 [jobs.json](jobs.json)，设计及恢复修订记录见 [PLAN.zh.md](PLAN.zh.md)，最后一致性检查见 [completion_audit.json](completion_audit.json)。
