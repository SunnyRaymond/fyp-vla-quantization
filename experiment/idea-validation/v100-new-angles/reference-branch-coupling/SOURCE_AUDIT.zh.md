# Reference-branch：执行前工程审查与实际producer

2026-09-13。独立design gate通过后，root在任何模型输出产生前修正以下实现项：

- raw IDs统一为整数underlying trajectory IDs与valid_indices；arm/schema写入并由独立replay硬检查。
- 通过实际predictor pre-hook检查 `[1,196,404]` 输入；不把待传入变量冒充实际hook记录。
- FP-copy以clone后的输入重新前向；restore no-op在所有量化arm结束后重新前向。两者不复用原FP数组当检查。
- 各map只应用一次，随后跑六个case；只在最终保存完整quant snapshot，避免多次重写大文件。所有arm继续使用同一冻结输入与同一CPU SR规则。
- 把producer engineering receipts与独立CPU verifier接口明确对齐；CPU不能用臆造字段或默认正确值代替source证据。

这些是执行前实现纠正，没有修改科学阈值、seed、bitwidth、sample、指标或horizon。最终producer LF SHA256 `84959e7e9262c0839faae1e589b7ef60e88c03fd943a06848b4e015f1668f900`，独立replay LF `258c41d8af9e991130db3df0f09935d560019b104e2cbb0d8ea2b9809bc2ee9c`；两者AST和launcher语法检查通过。

## 实际GPU receipt

GPU64840，TC1N04，SLURM COMPLETED 0:0，Elapsed26s；producer work22.795222s，peak allocated372194304 bytes。状态为complete，全部模型参数恢复前后digest一致：`a9209f00512265940624a9493a44cec13bdce60dd9b5fa373a37fc3d0abf6350`。

Raw SHA256 `e537775db57761f860759baf0725ba9f3a73ea23f96e03c53aae68025fc69cd8`，位于`/tc1home/UG/yguo017/v100_newangles_ccds/artifacts/64840/raw.npz`。本地保留[engineering](job-64840/engineering.json)、[runtime](job-64840/runtime.json)、[log](job-64840/run.log)。这些只证明producer已完成；科学结论须等独立CPU重建与replay。
