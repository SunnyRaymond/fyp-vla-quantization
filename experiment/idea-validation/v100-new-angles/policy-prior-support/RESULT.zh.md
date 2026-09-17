# Policy-prior proposal support：mechanism_no_go，STOP

2026-09-13。GPU64833与独立CPU64834完成。**8/8 state满足可识别条件，0/8满足预注册joint damage门槛，因此本切面为 mechanism_no_go。** 没有发现足以解释该负结果的实现问题，不调整位宽、样本、门槛或planner迭代数，不重跑、不进入完整验证。

假设是：仅量化actor生成的24个候选，会削弱这些候选对FP planner第一次update的贡献。实验以FP-policy24、W4-policy24、random24三臂替换相同slots，保留相同488个random候选和完整FP scorer；复评一次update mean的FP模型J，并追踪前24slot的elite inclusion和weight mass。它不是对真实return或完整6次MPPI iteration的验证。

## 原始结果

全部8个state中，FP与W4的24个policy候选都进入top64；random24对照只有1–7个进入。FP与W4的policy mass接近，约0.46–0.53。共同488个候选的评分最大差为 **0**，排除了score RNG或残留量化权重污染共同候选的情况。

| reset seed | G_FP | G_W4 | FP/Q/random elite count | joint damage |
|---|---:|---:|---|---|
| 5217 | 0.004776001 | 0.004699707 | 24 / 24 / 1 | 否 |
| 5218 | 0.005050659 | 0.004638672 | 24 / 24 / 7 | 否 |
| 5219 | 0.002929688 | 0.002914429 | 24 / 24 / 3 | 否 |
| 5220 | 0.000885010 | 0.000823975 | 24 / 24 / 2 | 否 |
| 5221 | 0.004440308 | 0.004119873 | 24 / 24 / 2 | 否 |
| 5222 | 0.012329102 | 0.012313843 | 24 / 24 / 1 | 否 |
| 5223 | 0.020706177 | 0.019943237 | 24 / 24 / 3 | 否 |
| 5224 | 0.000885010 | 0.001281738 | 24 / 24 / 4 | 否 |

G为对应update mean相对random24 update mean的FP模型J差。绑定要求G_FP>1e-4、FP mass比random24高1e-4且cutoff无精确tie；8条均满足。联合损害要求G_W4<=0.9G_FP且Q mass<=0.9FP mass，至少4条才能初步go；实际为0条。个别J轻微变化不是所假设的support损失。

## 证据与边界

- CPU准备64832：TC1N04，4s，8个fresh reset；不step/render/model inference。
- GPU64833：TC1N02，V100 32GB，wall21s / workload11.614s，完整8states，全部12个producer guards通过。
- CPU64834：TC1N02，3s，身份/输入/源文件/量化snapshot/RNG/FP恢复检查通过；独立重算top64、weights、mu/std、mass/count和冻结结论。CPU未重做模型推理。
- [独立verification](artifacts/64834/verification.json)、[GPU summary](artifacts/64833/summary.json)、[GPU engineering](artifacts/64833/engineering.json)、[冻结协议](PROTOCOL.zh.md)、[input freeze](../../../../idea/v100-new-angles/policy-prior-support/INPUT_FREEZE.zh.md)。
- raw SHA256：`f29e5bf62107d561901080032f317db5256f399a9664856dfda110ea689c7096`；原始NPZ、quant snapshot与RNG文件保留在compute存储 `/tc1home/UG/yguo017/v100_newangles_ccds/artifacts/64833`。

结论仅针对固定Cartpole-balance seed3 checkpoint、actor三层W4 RTN、8个reset状态及第一次内部update。它没有证明任意actor PTQ都安全，也没有证明native INT4速度、内存收益或真实任务成功率。限定证据内，proposal贡献未出现预设程度的损害；不将此负结果包装为新方法。

静态审查曾误读generator局部变量并引用旧verifier草案合同，已在GPU_AUDIT中撤销；实际GPU完整输出及CPU复算通过。未据此修改模型、阈值或重跑GPU。
