# Fast-WAM Prism 风格 activation / weight 可视化

本目录提供冻结的 `protocol.json`、采集器和 renderer，用来观察选定 Fast-WAM Linear 的 activation 与 weight 数值形状。它是描述性机制图，不是 accuracy benchmark，也不能单独证明 quantization 带来 accuracy、latency 或 closed-loop control 改善。

## 主图如何比较

主 activation slice 固定为 **case 1004、step 4**。`video_expert` 对应 `video` stage，`action_expert` 对应 `action` stage；每个 stream 画 `self_attn.q` 与 `ffn.2`，深度列为 block 0、10、20、29。重复的同一目标记录按 raw manifest 声明的最低 `call_index` 选作主记录；其它调用仍保留在 metrics 和可视化 manifest，不会覆盖或隐藏。

- **Paired-local：3 行 × 4 个深度。** 三行取自同一 BF16 Linear 输入，沿完整 channel 维分别应用 raw identity、learned SpinQuant-adapted `R128`、Smooth0.5 + `H128`，再裁出相同 token 窗口。它展示同一个局部输入在三种基底下的几何形状；不是这些变换在各自完整模型运行中的横向结果。每种变换的候选 A4 QDQ 保存在 NPZ，并计算配对窗口内的 NMSE。
- **实际 arm：5 行 × 4 个深度，PDF 两页。** 五行依次为 BF16、native SpinQuant-adapted INT4、native SmoothScalar INT4、row-scale VQ decoded BF16、micro-scale VQ decoded BF16。第一页画 `pre_quant`：已应用该 arm 的输入变换、但还未进入 A4；BF16 保持 raw identity。第二页画 `post_quant`：native INT4 路径的反量化值、VQ 路径的 decoded BF16 值，BF16 则是 identity copy。VQ decoded BF16 与 native INT4 是不同的执行路径，不能把 VQ 图解释成 native kernel 图。

Activation surface 的 x 轴是 channel，y 轴是 token row，高度和颜色都表示 `|activation|`。每页只画 batch 0 的第一个 CFG branch，从 token row 0 起最多 64 行、从 channel 0 起最多 512 个 channel；shape 较小时按真实 shape 截短。没有 pooling、striding 或通道重排。每个深度列在该页各 arm/transform 间共用线性高度范围和颜色范围，不会逐方法单独归一化。`pre_quant` 所在 basis 随 arm 而异，因此跨 arm 的数值形状也包含 basis 变化。

另有四个 weight bundle，对应两个 stream 与两个 module。每个 PDF 至少含 magnitude surface 和 error heatmap；如果原始 NPZ 或关联的 transform NPZ 保存了 scale，还会增加 scale 页面。Scale 页面每条记录显示按名称排序的第一个 scale 数组；所有 scale 的统计摘要仍进入 metrics。Weight surface 的坐标是 **x=input channel、y=output channel**，高度和颜色为 `|weight|`；窗口从 output 0 起最多 128 行、从 input 0 起最多 512 列，按实际矩阵 shape 截短，不补零。每个 arm 的 `reference` 是同一个未量化 BF16 weight 按该 arm 的 basis 变换得到的等 shape 窗口（例如 BF16 raw、learned SpinQuant-adapted、或 Smooth0.5/Hadamard）。因此跨 arm surface 差异可能同时来自 basis 与量化；同一 arm 内 `|weight-reference|` 才表示相对该 arm 同基底未量化 reference 的差异。

所有 surface 都显示幅值，无法从颜色读回正负号。signed 原值仍在独立 NPZ 中。图只覆盖协议指定的 module、block、case、step 和窗口，不代表整个模型或所有 token 的统计结论。

## 原始数据和 metrics

每条记录一个 NPZ，记录索引位于 `raw/manifest.json`。manifest 的 `records[]` 给出记录类型、case/seed、arm 或 transform、module/stream/block、stage/step/call、真实 shape、数组 keys、source bank 和文件路径；NPZ 内的 `metadata_json` 是 NumPy Unicode scalar，保存相应记录元数据。

- Activation NPZ 保存 signed float32 `pre_quant`、`post_quant`、`output` 和 `token_row_ids`。activation 捕获上限为 batch 0 第一个 CFG branch 的前 128 个 token row，并保留完整 channel 维；绘图和窗口 metrics 使用其前 64 行 × 512 channels（不足时依真实 shape 截短）。
- Paired-local NPZ 保存 `paired_raw`、`paired_learned_spinquant`、`paired_smooth05_hadamard`，对应的 `_qdq` 候选数组，以及 `token_row_ids`。变换在完整 channel 维上完成后才裁取 token 窗口。
- Weight NPZ 保存 signed `weight` 和等 shape 的 `reference`，轴顺序是 `[output_channel, input_channel]`。关联的 transform NPZ 可保存 rotation、scale 或 signs 等辅助数组；renderer 只摘要其中已有的 scale，不拟合或重算变换。

`plots/metrics.json` 为 manifest 中每条 activation、paired-local 和 weight 记录保存数组 shape、绘图窗口 shape、最大绝对值、RMS、逐 token/channel RMS（weight 对应逐 output/input channel RMS），以及可用的 QDQ NMSE。数值统计在同一显示窗口内完成，signed NPZ 不会被改写。重复 call 的所有记录均进入 metrics。

Renderer 会核对 protocol identity、manifest 完成状态、各 case/step/block/stream/module/arm（或 transform）的覆盖、目标记录 call 规则、weight 覆盖以及每个 raw 文件和 transform 文件是否存在；主图缺项、weight 重复或实际 shape 不可绘制时会报错退出，不会静默省略 panel。Renderer 只把主图固定为 case 1004、step 4；其它 case、step 和重复调用仍保留为 raw records 与 metrics，不会批量重画全部 surface。

## 复现

在获批的 **CPU-only PBS compute allocation** 中运行 renderer。它会读取并解压 manifest 中列出的 NPZ，再生成完整 PNG/PDF；完整 raw 与图文件应留在 compute-accessible 的持久结果目录。Login node 和本机不运行 renderer。脚本在读入实验记录前要求有效的 `PBS_JOBID`、有效的 `PBS_NODEFILE`，并拒绝 hostname 含 `login` 的节点；job wrapper 还须设置 allocation 开始时刻 `FW_JOB_START_EPOCH`，renderer 在其后 6600 秒停止启动新 bundle，以给 2 小时 allocation 留出收尾时间。它使用 PBS 环境已有的 Python、NumPy 和 Matplotlib，不安装依赖。

~~~bash
python experiment/idea-validation/fastwam-prism-visualization/plot.py \
  --protocol experiment/idea-validation/fastwam-prism-visualization/protocol.json \
  --out /path/to/fastwam-prism-visualization-results
~~~

`--out` 是采集结果根目录，必须包含 `raw/manifest.json` 及其引用的 NPZ。Renderer 在其下建立 `plots/`：

- `index.html` 是静态 gallery，链接 raw manifest、protocol 副本、metrics、相关 source NPZ、完整 PNG/PDF 和 preview。
- `manifest.json` 逐 bundle/page 列出完整 PNG、PDF 页码、source NPZ 和 preview 路径；每个 paired-local、实际 arm 的 A4 前/后页，以及每个 weight bundle 的 magnitude 主页各有一张 preview，`previews/` 内每张 JPEG 均不超过 220 KiB。Weight error/scale 补充页仍有完整 PNG/PDF。Preview 只降低最终图片分辨率，不对输入矩阵 pooling、striding 或抽样。
- activation bundle 为每个 stream/module 各一份 paired-local 图和实际 arm 图；实际 arm PDF 含 A4 前、后两页。Weight PDF 保存 magnitude 与 error 页面，若 scale 存在则附 scale 页面。
- 完整 PNG 与 PDF 保留完整绘图窗口的每个数据点；PDF surface 可 rasterize 以限制矢量对象数量，但 `plot_surface` 显式使用真实裁剪行列数，不接受 Matplotlib 默认隐藏下采样。
- Renderer 在每个 bundle 的全部页面保存成功后，才原子更新 `plots/manifest.json` receipt。接近 soft deadline 时会以 `status=checkpointed` 退出；使用相同结果目录重跑时，只有 protocol、source identities、记录 ID 与已保存输出都匹配的完整 bundle 会跳过重画，其它 bundle 继续生成。Raw manifest 和 NPZ 不会被 renderer 改写。

`--selfcheck` 只运行窗口边界 helper 的纯本地测试，不读取实验数据、不导入 NumPy/Matplotlib，也不生成图。
