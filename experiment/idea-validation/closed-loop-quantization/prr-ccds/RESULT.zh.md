# PRR 最低完整实验结果

**结论：当前 recipe 为 mechanism_no_go。平均误差有改善，但跨 seed、跨 episode 的稳定性未达到预先冻结的门槛。工程核验通过，没有把执行故障当作科学失败。**

## 实验完成与证据

四格 q_local / q_recovery / l_local / l_recovery，各 seeds1201/1202/1203，全部完成1000 updates、batch2、H2；共12次拟合。CAL72–77、DEV78–83各6个episode、每个episode两个起点。Q0与已审计的历史clean_seed_1201提供全组共享frozen donor。最终评测使用整数/scale重载后的hard W4自身free H2 rollout，FP reference使用共同完整DEV batch。

- GPU正式作业64706：COMPLETED、exit0:0、TC1N03 V100，19分38秒。
- 独立CPU核验64707：COMPLETED、exit0:0，10秒；engineering_pass=true、errors=[]、12/12完整。
- CPU明细汇总64708：COMPLETED、exit0:0，1秒；重用已有DEV raw，主指标与64707核验结果对齐，没有重新训练、调参或修改gate。
- 连同CAL预检64705的35秒，本轮总GPU消耗1213秒，即20分13秒、0.33694 GPU-hours。CPU资源准备64704另用10秒。

所有重操作均在真实CCDS SLURM compute allocation，未使用ASPIRE2A。本轮没有进入R2/R3或TEST评测。

## 四格结果

以下为相同权重Wz下的MSE，越低越好；是latent预测误差，不是任务成功率。

| 参数分支与目标 | 自身free H2终点误差（primary） | clean单步误差 | frozen-donor recovery误差 |
|---|---:|---:|---:|
| q_local：仅量化参数、local target | 0.218144 | 0.127414 | 0.152304 |
| q_recovery：仅量化参数、paired-clean target | 0.193840 | 0.102747 | 0.136155 |
| l_local：量化参数+LoRA-r4、local target | 0.219623 | 0.110005 | 0.145311 |
| l_recovery：量化参数+LoRA-r4、paired-clean target | 0.185101 | 0.107103 | 0.129986 |

Recovery相对各自local基线，q分支平均H2误差下降11.14%，l分支下降15.72%；两个分支的平均clean误差和donor recovery误差也都降低。因此不能把结果描述成“完全没有作用”或“只改善了训练loss”。正面变化确实延伸到新DEV上的自身H2 rollout，但不稳定。

## 为什么仍然no-go

冻结规则要求：family平均H2误差改善至少5%；至少2/3个seeds同时满足平均改善至少5%、至少4/6个episodes方向改善、clean退化不超过10%；family整体clean也不能退化超过10%。任一family通过即可给conditional_signal。实际两个family均未通过。

下表H2改善为正表示变好；clean变化为正表示退化。

| 分支 | seed | 平均H2改善 | 变好的episodes | clean误差变化 | seed完整通过 |
|---|---:|---:|---:|---:|---|
| quantizer-only | 1201 | +0.94% | 2/6 | -32.65% | 否：幅度与覆盖不足 |
| quantizer-only | 1202 | +7.40% | 4/6 | +9.21% | 是 |
| quantizer-only | 1203 | +22.87% | 3/6 | -27.93% | 否：覆盖不足 |
| quantizer+LoRA | 1201 | +35.94% | 3/6 | -17.80% | 否：覆盖不足 |
| quantizer+LoRA | 1202 | -2.15% | 4/6 | +1.90% | 否：平均误差反而上升 |
| quantizer+LoRA | 1203 | +6.65% | 2/6 | +13.32% | 否：覆盖与clean保护均失败 |

最终q分支只有1/3个seeds通过，LoRA分支为0/3，都低于2/3。全局平均H2与clean gate均通过，拒绝原因集中在seed内的episode覆盖和跨seed一致性。没有门限附近的ambiguous情况。

## 收益与损害发生在哪里

下表是每episode先平均两个窗口后的H2误差改善百分比；负值表示变差。相同DEV episode在不同seed中是重复评测，不是新增独立环境样本。

| 分支 / seed | DEV000 | DEV001 | DEV002 | DEV003 | DEV004 | DEV005 |
|---|---:|---:|---:|---:|---:|---:|
| q / 1201 | +4.59 | -69.99 | -28.96 | -12.42 | -0.99 | +23.65 |
| q / 1202 | +12.95 | -103.77 | +49.08 | -100.32 | +7.52 | +32.96 |
| q / 1203 | -9.39 | +4.58 | +27.09 | -28.58 | -20.66 | +42.61 |
| LoRA / 1201 | -65.19 | +4.92 | +20.07 | -68.21 | -3.61 | +61.01 |
| LoRA / 1202 | +43.59 | -0.45 | +30.29 | -159.41 | +20.10 | +1.09 |
| LoRA / 1203 | -4.55 | -149.63 | -59.97 | -26.27 | +15.28 | +40.51 |

DEV005在全部六次配对中改善，DEV003在全部六次配对中恶化。特别是LoRA seed1201，DEV005从1.04746降到0.40844，改善61.01%；但DEV000/003/004变差。因此平均改善35.94%并不等于多数episode可靠改善。小误差episode的相对百分比也可能较大，应同时参照原始MSE；完整数值保存在report_metrics.json。

这是对观察结果的描述，尚未诊断这些episode对应什么场景或为什么产生差异；不将其直接归因为特定物理机制或distribution mismatch。

## 对改进idea的判断

做得好的部分：paired-clean recovery target在平均意义上优于同参数自由度的local teacher target，且没有造成整体clean能力退化。LoRA的A/B均有有效梯度和参数更新，merge后hard量化与重载验证通过，排除了本轮“LoRA根本没有学起来”这一工程解释。

做得不够的部分：收益没有稳定覆盖不同episode，换seed后也不能重复满足完整门槛；LoRA增加参数自由度后仍未解决。不能由此断言quantizer-only在原则上不可能成功，也不能说LoRA必然优于quantizer-only。描述性交互量(D-C)-(B-A)在三个seed分别为-0.09230、+0.01972、+0.04193，符号不一致，不支持稳定的“LoRA增强recovery收益”结论。

决策：按冻结stop rule停止当前recipe，不自动进入R2/R3/TEST，不根据这些DEV结果重调学习率、steps或放宽gate。若未来修改idea，应提出能解释并改善失败模式的具体新假设，在新的验证数据上另行测试。此次no-go仅针对当前设置；6个DEV episodes与3个fit seeds的pilot不足以否定所有recovery方法，也没有验证真实闭环控制成功率。

## 实现与证据边界

目标是24个predictor Linear的signed W4[-7,7]、per-output-channel scale。LoRA先合并为W+BA再硬量化，最终没有LoRA bypass或alpha参数。四组seed1201 checkpoint的tensor payload相同：未量化参数88,654,160 bytes；量化整数19,857,408 bytes；scales142,272 bytes。整数使用int8容器装逻辑W4数值，没有native packed-W4 kernel或端到端速度/显存收益声明；这些tensor字节不含序列化metadata/padding。

历史0–71与新CAL/DEV的底层episode映射不重叠，新initial-condition fingerprints跨episode不重复。历史全量initial-state fingerprint registry覆盖不完整，不能声称已排除所有未登记历史轨迹的状态重复。两个窗口仍嵌套于同一个episode。

格式勘误：冻结manifest的artifact_contract.raw_shapes.wz遗留[1,P,D]，实际与weighting.shape一致为[P,D]=[196,404]；verifier按实际二维广播重算，未修改训练或冻结manifest。raw metadata里的row IDs通过canonical records连接initial fingerprints。64707的donor_recovery_global把所有arms合并，不适合做方法比较；本报告的分组secondary明细来自64708，不影响原primary gate或no-go判定。

## 可复核产物

- [独立核验](artifacts/64707/verification.json)
- [四格完成摘要与checkpoint SHA](artifacts/64706/raw_final_summary.json)
- [每episode与donor明细](artifacts/64708/report_metrics.json)
- [工程预检](ENGINEERING_RESULT.zh.md)
- [冻结正式manifest](manifest_r1.json)；实际采集manifest见artifacts/64706/manifest.json。
- 原始数组保留远端：`/tc1home/UG/yguo017/prr_ccds/artifacts/64706/raw_final.npz`。
- raw SHA256：`5316368523b54dd5b9383f005b5fd9c5eb39c780bc51aa8bdc31b680dfa7e93e`。

旧FRT结果保持原结论，本轮PRR独立记录。定时监控在本轮结束后暂停。
