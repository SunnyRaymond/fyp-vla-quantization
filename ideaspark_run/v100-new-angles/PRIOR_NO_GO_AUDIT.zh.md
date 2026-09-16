# 本轮必须避开的已有失败机制

2026-09-13，直接读取最新本地 RESULT 文件；不以旧 STATUS 中“尚未运行”覆盖后续结果。

| 既有 idea | 当前最强证据与结论 | 本轮排除约束 |
|---|---|---|
| RankCal | 24 held-out goals：19/24，与ScoreError相同；global ranking改善未同时领先elite/action fidelity；两个random maps为20/24、17/24。资源决策no-go。 | 不再只换排序评分、增加mixed-bit搜索或宣称global ranking等于控制收益。 |
| OTC-PTQ | FastWAM IDM 2任务、8episodes、32states、256 site interventions；observation占C_r约99.77%，top2与LocalMSE同集合；functional replay通过。 | 不重调observation/action权重或把single-site sensitivity冒充联合配置收益。 |
| CEM-Update | CCDS64670/64676，V100，DEV4episodes；Update与MeanOnly同图，two-step final-mu MSE .060577，ScoreError/Rank .045147；engineering_pass=true、机制gate失败。 | 不继续改sigma权重、起点或搜索轮数；没有明确实现故障可解释失败。 |
| FRT | CCDS64694/64696，3methods×3seeds；Q0 transport相对Random改善约26%，fresh_union仅0.30%；工程与hard grid验证通过。 | 不重演冻结residual transport，也不靠lambda/steps把旧DEV调成go。 |
| PRR | CCDS64706/64707/64708，2×2×3seeds；新DEV6episodes；平均free-H2改善11.14%/15.72%，但q仅1/3 seeds通过，LoRA0/3；工程核验通过。 | 不把平均改善当可靠go；不扩大LoRA rank、更新数或放宽episode覆盖gate。 |
| TR-PVQ | 当前STATUS为方案与conditional pilot readiness，canonical pipeline不完整；未在所读STATUS中证明经验结果。 | 不是已验证no-go；避免重复提出trajectory relation/Gram loss或codebook coupling。 |

## 是否有理由修复旧 no-go

本轮所读最终记录没有显示仍未解决、足以使当前科学结论失效的实现缺陷。FRT早期allocation identity与FP batch mismatch已修复并重新通过工程核验；PRR LoRA梯度、merge、hard reload均有效。因此本轮不以“也许实现不够好”为由继续修补这些路线，转向不同干预对象。

这些都是特定recipe的早停证据，不是对所有相关方法的不可能性证明。PRR确有平均改善，FRT确有固定donor改善，必须连同失败门槛一起保留。

## 精确来源（历史文件不修改）

- `../world-model-quantization/experiments/dino-wm-wall/SCREEN_RESULTS.zh.md`
- `../../reproduction/otc-ptq-phase1/RESULT.zh.md`
- `../world-model-quantization/experiments/cem-update-ptq-ccds/RESULTS.zh.md`
- `../closed-loop-quantization/experiments/frt-ccds/STAGE_B_RESULT.zh.md`
- `../closed-loop-quantization/experiments/prr-ccds/RESULT.zh.md`
- `../vq-action-geometry/STATUS.zh.md`

CCDS可复用官方DINO-WM source/checkpoint与screen runner；既有实验的大数组仅用作诊断，不冒充新独立确认数据。历史dataset0–83已用；进一步读取PRR `manifest_r1.json`发现84–95属于test_locked保留区。本轮从96开始，0–95不用于新评估，仍须核实底层episode映射与fingerprints。
