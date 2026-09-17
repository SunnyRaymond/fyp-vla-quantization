# Root verifier复核处置

科学门槛不变。CPU64808已按原协议重算，FP no-op失败，最终仍是implementation_inconclusive。修订verifier仅加强来源核验并修正CPU模拟CUDA scalar division的算术实现；不重跑GPU、不放宽no-op、不新增样本。

已增加批准的freeze/draft/helper固定SHA、checkpoint实际size/hash、全部selected source实际hash与import精确路径、official common/init.py固定SHA、具体final-head alias身份、raw dtype与已记录V100/device/no-grad字段复核。官方init SHA通过固定commit的 [primary source](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/init.py) 独立取得。CPU核对producer证据与文件，不宣称独立重现GPU kernel或形成防恶意伪造证明。

CPU64808对row-scale做NumPy直接division，和原GPU `maximum / float(7)` 不bit-identical。PyTorch2.6 [CUDA BinaryDivTrueKernel](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu) 明确对CPU scalar分母采用reciprocal multiplication；因此verifier改为FP32 `maximum * float32(1/7)`，保持原GPU quantizer recipe。原64808 exact-grid失败记录不覆盖。CPU再次读取同一raw，只修该工程复算差异和来源检查。

FP-centering no-op失败与此CPU scale差异是两件事：即使quantizer复算通过，no-op失败仍禁止preliminary_go。CPU64809最终全部工程/来源/grid检查通过，科学结论不变。独立ARITHMETIC_REPAIR_GATE完成后root接受STOP；没有证据支持通过改容差/FP reference/数据来挽救这一冻结screen。
