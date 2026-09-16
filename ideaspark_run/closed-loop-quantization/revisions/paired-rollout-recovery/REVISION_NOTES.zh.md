# 修订依据、近邻与流程记录

2026-09-12。用户要求在 FRT 初步失败后更新 idea，保留方向但允许改目标。当前交付为 PRR 的有界研究假设和实验草案，没有代码实现或集群执行。

## 不改变的实验结论

[旧 Stage B](../../experiments/frt-ccds/STAGE_B_RESULT.zh.md) 的 fresh-union 相对 Random 改善0.30%、仅1/3 seeds过门，当前 recipe仍是 mechanism_no_go。A/B工程通过、完整12 records/6DEV、9fits完成的记录保留。

旧 cosine 与 fresh-union 并非单独隔离方向漂移：`frt_runner.py::_build_bank` 的 Q0 残差来自原 history；`frt_stage_b.py::_fresh_bank` 在拟合模型 rollout 后的 history 上再取下一步残差。共同 evaluator×bank 评分公平地比较了每一 bank，却不让跨 bank 的差异变成“只改方向”。这一点收窄失败解释，不推翻原预注册 no-go。

没有证据证明：FRT 只因 λ 太小失败；最终误差必然位于稳定子空间；在线刷新 residual 必然有用；或4张A100能让旧配方成立。新版不采用这些前提。

## 本次实际检索

复用前一轮 GAD/QDrop/PD-Quant/QuantWAMs 记录，新增定向查询：

- `post training quantization diffusion denoising quantization error correction trajectory distillation teacher clean input student quantized input`
- `world model quantization error accumulation recovery calibration rollout teacher forcing on policy quantization`
- `world model training hallucinated replay self correcting model Talvitie 2017`

主要依据均为作者/会议 primary sources，访问日期2026-09-12。不是穷尽检索，也没有重新运行整个 connector pool。

| 直接近邻 | 实际阅读位置 | 对候选的约束 |
|---|---|---|
| [Self-Correcting Models for Model-Based RL](https://arxiv.org/html/1612.06018) | §1，§2.3–2.4，§3 | sampled/model state配对参考target、同action约束、hallucinated replay均已有；确定性/策略条件不能无条件搬到DINO/CEM |
| [CTEC](https://ojs.aaai.org/index.php/AAAI/article/view/34039/36194) | Eq.14、Algorithm1、Table3及讨论 | recovery与quantization-aware LoRA均有直接先例；其calibration-only反例仅是其设置，不能断言本项目per-weight rounding必然失败 |
| [AccuQuant](https://arxiv.org/html/2510.20348v1) | 作者项目页、摘要及方法入口 | 多步PTQ、轨迹对齐、低内存实现不能作为本方案独有贡献；full-unroll是机制control，不能替代对该方法的后续对照 |

初步“只把 residual bank 换成最新模型生成”的修补未作为候选，因为它没有改变原 response-matching 目标，还引入样本分布混杂。初步“recovery是新原理”也已撤回：Hallucinated Replay和CTEC直接覆盖一般原则。最终保留的是 **目标×参数自由度的 WM 适配问题**。

## 假设、证据、待定项

- **已知证据：** 当前 FRT recipe 未通过；量化/残差/state接口已有局部工程验证；相关 recovery principles 已存在。
- **研究假设：** 某些 fixed-W4 action-conditioned WM 的短程损失可通过 paired-clean target 减少；其收益是否依赖权重适配可由四格对照区分。
- **待定实现：** 新episode fingerprints、LoRA有效梯度、统一surrogate、merge后整数权重核验、H2自由rollout和actionprefix测试、整体实测资源账本。
- **禁止推论：** donor recovery下降等于closed-loop成功；LoRA胜出等于quantizer表达能力不可能；V100旧计时等于新A100耗时；前作未出现DINO就等于novel。

## 精简 ResearchStudio-Idea 修订

1. 复核现有结果与source，不重跑实验。
2. 有界primary-source补检，发现直接collision后收窄贡献。
3. 使用 `controlled_diagnostic_design` 为主 pattern：固定donor/input，交叉target与参数自由度。`targeted_self_supervised_objective`只辅助表达候选；已有objective不得包装为新原理。
4. `gpt-5.6-luna`、`xhigh`独立审查，先批评机制与context混杂，再读最终草案；这是派发配置，未声称额外runtime introspection。最终意见见 [REVIEW.zh.md](REVIEW.zh.md)。
5. 省略整池重复检索、原生navigator、三语出卡和PDF。当前不是canonical DONE，也不声称Oral级/已novel。

实验设计的配对/阻断和统计单位使用 experimental-design skill；工作流引用已列在实验草案中。旧实验目录、原始结论、weights和raw arrays均未修改。
