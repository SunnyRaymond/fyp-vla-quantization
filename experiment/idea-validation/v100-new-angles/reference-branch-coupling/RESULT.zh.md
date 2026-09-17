# Reference-branch coupling：mechanism no-go，停止

2026-09-13。GPU64840与独立CPU64841均完成。最终为 **mechanism_no_go / STOP**：6/6 states可识别，0/6达到冻结的A/B同时改善至少10%门槛。没有发现解释该no-go的实现缺陷，不调整bitwidth、seed、goal帧或阈值。

| Wall trajectory | residual误差A改善 | scalar objective误差B改善 | 联合gate |
|---:|---:|---:|---|
| 1035 | 1.29% | 2.30% | 未过 |
| 1534 | 2.66% | 5.37% | 未过 |
| 1158 | 2.38% | 4.23% | 未过 |
| 203 | 2.79% | 5.52% | 未过 |
| 1837 | 1.90% | 3.75% | 未过 |
| 1095 | 1.51% | 3.02% | 未过 |

A是quantized current-minus-goal residual相对FP residual的MSE；B是scalar visual MSE objective相对FP objective的平方误差。不是直接让quantized objective变小。完整A/B/L的3×3矩阵保存在CPU verification中。

三张W4 encoder maps在current与goal分支共享，较off-diagonal不同map配对，在六个state均有小幅方向性改善，但幅度不足。本结果没有证明共享rounding完全无效；它只否定当前H1、六个复用Wall pair、三个draw达到事前screen门槛，不支持在本轮继续推进。

GPU64840在TC1N04 V100完成，Elapsed26s、work22.795222s；48 encoder Linear、42次单步predictor调用。CPU64841在TC1N04 CPU allocation完成，Elapsed8s；所有engineering/source/input/quantizer/RNG/code/readback/restore检查及独立A/B复算通过。FP-copy与量化后restore均实际重新前向且相等；非目标state保持不变。执行前工程修正见[SOURCE_AUDIT](SOURCE_AUDIT.zh.md)。

证据：[冻结protocol](PROTOCOL.zh.md)、[独立design gate](../../../../idea/v100-new-angles/reference-branch-coupling/CONDITIONAL_GATE.zh.md)、[CPU结论](job-64841/verification.json)、[GPU收据](job-64840/engineering.json)。完整raw/quant snapshot留在`/tc1home/UG/yguo017/v100_newangles_ccds/artifacts/64840`。

- Raw SHA256：`e537775db57761f860759baf0725ba9f3a73ea23f96e03c53aae68025fc69cd8`。
- Producer LF：`84959e7e9262c0839faae1e589b7ef60e88c03fd943a06848b4e015f1668f900`。
- CPU verifier LF：`fe6da97601bbb1fd5ee417b4a50d2540b1fd3ea479fae23854e6717b7b67a579`。
- Replay LF：`258c41d8af9e991130db3df0f09935d560019b104e2cbb0d8ea2b9809bc2ee9c`。

Generic correlated rounding已有prior，`generic_method_novelty_no_go / narrow_application_unverified`保持不变。全部运算是FP32 fake quantization，没有native低比特性能、storage收益、task outcome或统计泛化主张。至此14个经验screen全部停止，账户额度已到remaining1%。
