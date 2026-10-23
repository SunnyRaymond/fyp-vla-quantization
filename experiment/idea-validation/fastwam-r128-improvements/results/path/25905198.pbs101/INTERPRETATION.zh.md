# 固定旋转幅度实验：完整结果解读

成功作业25905198.pbs101：scheduler F/Exit_status=0、artifact exit0、PHASE_COMPLETE及完整summary均已核对。4臂各C8 8/8与heldout80/80，614 Linears，全部参数冻结，formal inference parity为0。H/R8原端点重放的七项metric差为0；各非identity U的orthogonality max-abs不超过2.15e-6。没有新增teacher。

Heldout按seed→context→suite平均，四suite等权；40个context为配对单位，80条queries不当作80个独立样本。以下为完整冻结队列上的描述性结果，不作显著性或跨训练随机性的推断。

## 相对H的误差变化

| t | C8 motor | C8 latent | C8 mixed | Heldout motor | Heldout latent | Heldout mixed |
|---|---:|---:|---:|---:|---:|---:|
| 0，H | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.25 | -2.24% | -19.09% | -13.01% | +1.27% | +9.22% | +5.96% |
| 0.5 | -0.97% | +79.22% | +32.57% | +0.18% | +11.09% | +5.75% |
| 1，原R8 | -8.26% | -12.09% | -17.98% | +4.65% | +5.12% | +7.67% |

H heldout motor RMSE=0.0426045556、latent NMSE=0.0383806173；t0.25为0.0431452184/0.0419193139；t0.5为0.0426801653/0.0426372828；原R8为0.0445851211/0.0403438791。

## 支持与限制

1. 减小沿原R8方向的最终幅度可以减轻motor退化，但没有同时改善motor与latent；四个固定点中H的三项heldout总体指标均最低。
2. 不能将原问题只归为旋转幅度过大。t0.25/0.5的latent都比原R8更差，C8 latent/mixed沿路径也不单调。量化模型不能按连续、单调的loss路径理解。
3. t0.25在C8上有6/8个context的mixed loss改善，但case2019仍贡献约78.3%的净改善；原R8为4/8个改善、2019贡献139.4%。收缩使C8改善更分散，仍没有带来heldout总体收益。
4. t0.25和t0.5分别有21/40、22/40个context的motor改善，但总体均值仍退化，表明不能只看win count。
5. 这里是对同一个最终A8做Cayley(tA8)，不等同于更小LR重新训练。后者可能经过不同的梯度和量化分箱轨迹；顺序/LR实验与C80仍需完整执行。

本轮没有按heldout选择更多t、调整后续训练或宣布某个插值点可替换H。完整数值与case/suite数据见comparison.json。三个lane现已全部完成，整体判断见[最终分析](../../final/ANALYSIS.zh.md)。
