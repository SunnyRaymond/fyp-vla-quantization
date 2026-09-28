# 029. A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition

> 本地原文：[Springer journal-typeset PDF copy](paper-publisher-final.pdf)，40 PDF pages。首页核对标题、作者、J Nonlinear Sci (2015), 25:1307–1346、DOI。
>
> 来源：[DOI](https://doi.org/10.1007/s00332-015-9258-5) · [PDF source](https://robotics.caltech.edu/wiki/images/f/fa/DataDrivenApproximation.pdf) · [固定版本备份 arXiv v1](https://arxiv.org/abs/1408.4408v1)。arXiv history 仅列 v1。
>
> 阅读状态：unread。已作 PDF signature、pypdf 页数和首页身份核对；未运行论文实验。

## Identity

Authors: Matthew O. Williams, Ioannis G. Kevrekidis, Clarence W. Rowley.  
Journal final, 2015, Journal of Nonlinear Science 25:1307–1346. DOI: 10.1007/s00332-015-9258-5.  
本地文件：paper-publisher-final.pdf（40 PDF pages）。来源是 Caltech-hosted Springer 排版版；访问日期 2026-09-26。

## 直觉与方法

原系统为离散映射 \(x_{n+1}=F(x_n)\)。Koopman operator 作用于 observable：\((\mathcal K\psi)(x)=\psi(F(x))\)。若 \(\phi_j\) 是 eigenfunction，则 \(\phi_j(F(x))=\mu_j\phi_j(x)\)：这个 scalar 坐标一步只乘 eigenvalue。若 full-state observable 可由 eigenfunctions 表示，则可按 \(x\approx\sum_jv_j\phi_j(x)\) 重构；一步之后 \(F(x)\approx\sum_j\mu_jv_j\phi_j(x)\)。

EDMD 用 snapshot pairs \((x_m,y_m)\), \(y_m=F(x_m)\) 和 observable dictionary \(\Psi(x)\) 拟合有限维算子。论文写作 \(\mathbf K=G^+A\)（Eq. 12–13）；\(\mathbf K\) 的 eigenvector \(\xi_j\) 给出近似 eigenfunction \(\phi_j(x)=\Psi(x)\xi_j\)（Eq. 14）。Koopman mode \(v_j\) 是映回 full state 的向量权重（Eq. 20）。

不要混淆：eigenvalue \(\mu_j\) 是更新倍率；eigenfunction \(\phi_j(x)\) 是 state-to-scalar function；\(\xi_j\) 是有限 EDMD 矩阵的 eigenvector；mode \(v_j\) 是重构方向；当前 modal coefficient 是 \(\phi_j(x)\)。这些精确更新只对真实 eigenfunction 成立。有限 dictionary 通常不对 Koopman operator 不变，Eq. 10 因而有 residual；数据、dictionary 和截断让 EDMD 仅近似解耦。把多个坐标放成 vector block 并独立更新，还需该 block 近似不变子空间；固定维切块本身不够。

## 论文证据与边界

四类例子覆盖 deterministic 与 stochastic 数据：二维 linear system 的定量核对；Duffing oscillator 的 attraction-basin parameterization；double-well SDE 的 stochastic Koopman eigenfunctions；Swiss-roll manifold 的 parameterization / slow dynamics。定位见 §4–5、Fig. 1、4、6、12。论文说 EDMD 在合适数据和 dictionary 下近似 leading Koopman tuples，并在大数据极限对应 dictionary 子空间上的 Galerkin approximation（§2.3）。

这不是全状态精确重构或 universal independent blocks 的保证。原文主体是 autonomous map，没有 action-conditioned PushT、visual encoder、CEM 或 robot control。eigenfunctions 是完整 state 的 observable，不是原 latent 的任意分片。若用于 frozen LeWM + PushT，须另测带 action/history 的 split predictor；latent loss、候选排序、CEM fidelity、闭环成功与部署 latency 是不同证据层。本文未支持其中任何 planner 或 speedup claim。

## 阅读路线

**20 分钟：** Abstract；§2.1 Eqs. (1)–(3)；Fig. 1（PDF p.7 / journal p.1313）。解释 observable、eigenfunction、mode、eigenvalue。  
**90 分钟：** 加读 §2.2.1 Eqs. (10)–(14)（PDF pp.8–9 / journal pp.1314–1315）、§2.2.2 Eq. (20)（PDF p.10 / journal p.1316）、§2.3；看 Fig. 4（PDF p.20）的 missing / erroneous eigenfunctions。  
**180 分钟：** 读 §4.1–4.2、Fig. 6（PDF p.25 / journal p.1331）和 §5 stochastic examples；记录每个例子的 dynamics、dictionary、重构对象和 approximation 限制。PDF page 与印刷页码已分别注明。

## Reading Questions

1. 为什么 Koopman operator 对 observable 线性，却可描述 nonlinear \(F\)？
2. 从 Eq. 1 推出 eigenfunction 的一步更新。
3. Eq. 2 中 \(v_j\) 与 \(\phi_j(x)\) 分别是什么？
4. Eq. 14 如何把 EDMD eigenvector 变成 observable 坐标？
5. \(G^+A\) 如何由 snapshot pairs 得到？
6. Eq. 10 的 residual 从何而来，如何限制独立更新？
7. dictionary 不够时论文观察到哪些 missing / erroneous eigenfunctions？
8. 多个 scalar eigenfunctions 组成 vector block，须满足什么条件？
9. Duffing eigenfunction 怎样表示 attraction basin？
10. stochastic Koopman 的 observable 更新和 deterministic next-state prediction 有何区别？
11. 哪些 full-state reconstruction 假设限制了 Eq. 2？
12. 若用于 PushT，action 与 observation history 必须如何进入 dynamics model？
13. 怎样发现各 block 单独预测准确、合并后却错？
14. 需要哪些 CEM 与闭环测试才能支持规划收益？

## Meeting Card（留空）

- 我的 1 句理解：
- 最强证据：
- 最大假设 / 限制：
- 与 LeWM + PushT 的联系：
- 还需确认：
- 下次讨论的问题：

## Primary sources

- Williams, Kevrekidis & Rowley (2015), [journal DOI](https://doi.org/10.1007/s00332-015-9258-5)。
- 本地版本：[Springer-typeset PDF copy](https://robotics.caltech.edu/wiki/images/f/fa/DataDrivenApproximation.pdf)。
- 固定备份：[arXiv:1408.4408v1](https://arxiv.org/abs/1408.4408v1)。
