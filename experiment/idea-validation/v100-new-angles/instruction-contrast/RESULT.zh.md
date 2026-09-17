# Instruction contrast：本轮停止于可识别性审查

**当前结论：原模块机制主张 `identifiability_no_go`，GPU 0；不是经验上证明语言干预无效。** 原brief与否证文档全部保留。

“language-side W4比expert-side W4更损害grounding”无法由提出的三臂比较识别：两臂量化参数量、扰动大小与传播路径不同；单prompt action MSE也不控制两prompt共同漂移。没有同场景合法替代指令的外部依据时，数值响应变化不能当作grounding变化。

否证阶段提出的双prompt误差分解有解释价值，但把问题降为两个固定配置的输入响应描述，尚未形成值得另跑GPU的独立机制主张。为避免仅靠改名挽救切面，本轮不继续实现或验证。

需纠正早期措辞：SmolVLA两个模块的源码边界已找到，runtime验证仍待flow实验；不应将“尚未runtime核验”当作模型没有此结构。FP contrast很小仅说明这个输入干预的响应不可识别，并不证明FP模型不grounded。上述限制不依赖任何SmolVLA输出，未来不同资产/问题可重新立项，不能将当前记录冒充empirical no-go。
