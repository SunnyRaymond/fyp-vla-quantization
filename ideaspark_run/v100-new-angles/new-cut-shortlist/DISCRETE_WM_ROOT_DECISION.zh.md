# Discrete tokenizer PTQ：root gate correction

2026-09-13。仅source/design审查，GPU=0，没有经验结论。

IRIS/iVideoGPT保留为 `design_not_frozen / assets_not_bound`，不因当前没有下载就判定V100不可行。一个IRIS checkpoint约127MB的官方metadata是可行性线索；iVideoGPT的action-free WM也属于用户范围。新下载须在真实CCDS CPU allocation内完成，但用户授权的研究工作并不禁止合理staging。DIAMOND只在本次discrete-tokenizer定义下不适用。

独立草案中的FP-token replay已正确降为工程identity check。然而新草案仍不能作为冻结protocol：

- 在不同frame间配对RTN/SR的post-treatment latent MSE，不能同时保证输入难度、codebook margin与世界模型局部敏感性一致。MSE相近不足以识别boundary-specific effect。
- “matched flip-count/codebook-jump replacement”尚未指定合法token生成规则、配对容差、位置选择、不可匹配时处理；因此不是可执行的负对照。
- 32 frames不等于32独立episodes；8配对中4方向一致也不是有判别力的正向门槛。配对不足应为binding/statistical inconclusive，不应写成机制no-go。
- 现有草案把“更大downstream error”当支持，又用“不优于替代token”作否证，方向表述冲突；不能在看数据后决定目标方向。

因此先保留候选与source入口，不为不完整设计staging模型或提交GPU，也不把未运行记作经验no-go。原草案保留以显示审查过程，以上决定覆盖其未经冻结的sample count和gate。

定向prior检索还遇到[TokenBridge, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Bridging_Continuous_and_Discrete_Tokens_for_Autoregressive_Visual_Generation_ICCV_2025_paper.pdf)：其“post-training quantization”是把连续VAE特征离散化，不是本案对已训练tokenizer的数值权重PTQ；不能把术语相同当作exact prior。本次未找到exact overlap不等于novelty认证。

资源与source入口见[独立审查](DISCRETE_WM_RESOURCE_GATE.zh.md)。后续需要的是先完成可识别设计，再绑定小资产；没有完整WM训练、ROM下载或完整benchmark计划。
