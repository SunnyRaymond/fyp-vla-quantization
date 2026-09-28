# 030. Dynamic mode decomposition of numerical and experimental data

> 本地原文：[paper-hal-deposit.pdf](paper-hal-deposit.pdf)。25 PDF pages：第一页为 HAL cover，随后为期刊印刷页 5–28（24 篇文章页）。
>
> 来源：[CORE stable request](https://core.ac.uk/download/52899601.pdf)；cover 标明 HAL Id hal-01020654、Peter Schmid、JFM 2010 vol. 656 pp. 5–28，及 2014-07-09 deposit 日期。这是 HAL 存档副本，不标作 Cambridge publisher-final PDF。期刊记录：[Cambridge article page](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/dynamic-mode-decomposition-of-numerical-and-experimental-data/AA4C763B525515AD4521A6CC5E10DBD4)。
>
> 阅读状态：unread。已核对 PDF signature、页数、HAL cover 与论文首页身份；未运行论文实验。

## Identity

Author: Peter J. Schmid. Title: Dynamic mode decomposition of numerical and experimental data. Journal of Fluid Mechanics 656 (2010), pp. 5–28. DOI: 10.1017/S0022112010001217. 本地文件 paper-hal-deposit.pdf，25 PDF pages（含 cover）。source_url 是无签名的 CORE 稳定请求地址；访问日期 2026-09-26。

## 直觉与方法

Flow snapshot \(v_i\) 是高维 field vector。DMD 先假设相邻快照满足近似固定线性映射 \(v_{i+1}\approx Av_i\)（§2.1, Eq. 2.2）。用快照 span / SVD 得到低维 reduced operator \(\tilde S\)，再求 \(\tilde S y_j=\mu_j y_j\)；原快照空间里的 DMD mode 是 \(\Phi_j=Uy_j\)（§2.2, Eq. 2.10）。

若 reduced operator 可对角化，取 eigenvector 矩阵 \(Y\)、eigenvalue 对角阵 \(\Lambda\)，modal coefficient \(c=Y^{-1}\tilde v\) 则满足 \(c_{i+1}=\Lambda c_i\)，并由 modes 合成近似快照。这是从原文 eigendecomposition 推出的坐标更新：eigenvalue 是一步增长 / 衰减 / 相位倍率；eigenvector 是 reduced 坐标中的方向；DMD mode 是投回 flow field 的 pattern；modal coefficient 是当前权重。DMD modes 不应默认互相正交；复共轭 mode pair 可合看作二维振荡块。

POD/SVD 提供低维 basis；DMD 进一步拟合时间演进。POD 本身按空间能量构造 basis，不自动给出逐坐标时间更新。DMD 的逐模态简单演进要求 reduced map 在数据覆盖范围内可由所选 eigenmodes 代表。切块训练独立 nonlinear predictor 不是这篇论文的方法。

## 论文证据与边界

论文以 numerical plane Poiseuille flow 检查谱随 snapshot 数的收敛，并展示 cavity flow；还分析 flexible-membrane wake 与 jet-between-cylinders 等实验 PIV flow data。Fig. 3 比较 DMD / Arnoldi 谱收敛；Figs. 5–6 展示 cavity spectrum 与 modes；Fig. 9 展示 flexible membrane wake。§2.3 说明可处理 subdomains / lower-dimensional slices。

作者明确说 nonlinear process 上固定 \(A\) 是采样区间内的 linear tangent approximation（§2.1），不是全局精确 nonlinear law；识别频率还受 snapshot interval 约束（§2.6）。没有 action-conditioned world model、LeWM、CEM 或 robot-control evidence。

对 frozen LeWM + PushT，DMD 可作为诊断：冻结同一 encoder 后，问 latent 是否有可重构且近似线性演进的 modal coordinates。它不等于预先按 latent 维切块。PushT 的 action、history 和 contact dynamics 可能让 autonomous fixed \(A\) 失效；把不同 action 混成一套 map 会改变问题。若测试 split prediction，固定 observation/encoder、trajectory、action/history 与 predictor budget，先测 rollout / reconstruction，再分别检查 candidate ranking、CEM trace 与 closed-loop success。任何一个 approximation metric 都不能替代规划成功或 latency 证据。

## 阅读路线

**20 分钟：** Abstract；§2.1 Eq. (2.2)、§2.2 Eqs. (2.7)、(2.10)；Fig. 3。PDF pp. 2、4–6、11；印刷页 pp. 5–9、14。  
**90 分钟：** 加读 §2.3 subdomain、§2.5 convergence、§2.6 sampling interval；看 Figs. 4–6。PDF pp. 8–14；印刷页 pp. 11–17。画出 POD/SVD basis 与 DMD temporal eigenvalues 两步。  
**180 分钟：** 读 §3 数值与实验案例、结论；看 Fig. 9。记录每个 snapshot sequence 的来源、所作的线性假设和支持的结论。PDF p.1 是 HAL cover；文章印刷 p.5 从 PDF p.2 开始。

## Reading Questions

1. Eq. 2.2 中固定 \(A\) 假设了什么？
2. nonlinear dynamics 下作者如何解释这个近似？
3. SVD basis \(U\) 在 reduced DMD 中做什么？
4. Eq. 2.10 的 \(y_j,\mu_j,\Phi_j\) 分别处于什么空间？
5. 从 \(\tilde S=Y\Lambda Y^{-1}\) 推出 coefficient 的单步更新。
6. DMD modes 和 POD modes 在构造与正交性上有何差别？
7. Fig. 3 说明了何种 convergence，不能说明什么？
8. §2.3 subdomain DMD 与切 latent block 的差别是什么？
9. snapshot interval 如何限制频率识别？
10. truncation / noise / non-normality 怎样影响 modal reconstruction？
11. 多个 PushT actions 为什么未必共用一个 \(A\)？
12. action-conditioned DMD 属于原论文还是后续扩展？
13. 怎样的 fixed-observation rollout 结果值得进入 CEM fidelity 检查？
14. 哪些额外证据才能支持 control-success 或 latency claim？

## Meeting Card（留空）

- 我的 1 句理解：
- 数据与 reduced map：
- 最强验证：
- nonlinear-flow 近似边界：
- 与 LeWM + PushT split prediction 的联系：
- 还需确认：
- 下次讨论的问题：

## Primary sources

- Schmid (2010), [Cambridge article record](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/dynamic-mode-decomposition-of-numerical-and-experimental-data/AA4C763B525515AD4521A6CC5E10DBD4), DOI 10.1017/S0022112010001217.
- 本地版本：[CORE stable request](https://core.ac.uk/download/52899601.pdf)；PDF cover 指向 HAL Id hal-01020654。
