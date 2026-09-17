# 来源、版本与开放程度核验

核验：2026-09-08。[返回入口](README.md)。本次完成论文定位、PDF 保存、关键段落/图表阅读、官方 repository/documentation 与部分 checkpoint 文件列表核验。**未运行模型、不宣称复现论文数值、不下载大权重。**

## 证据层级

1. 论文 claims：固定版本的 arXiv PDF，身份和路径见 [清单](PDF_INVENTORY.md) 与 [manifest](PAPER_MANIFEST.json)。
2. 当前公开资料：官方 README、license、项目页与 Hugging Face metadata。摘要写“will release”可能已过时，repository 有代码也不代表所有实验可重现。
3. 本次验证：PDF 可解析、抽取、首页与选定关键页可渲染；本地导读链接有效。
4. 未验证：训练/推理结果、kernel 正确性、GPU 性能、完整 env 安装、所有权重的下载权限及论文缺失 supplementary。

`sources/` 保存只读的原文提取和官方资料快照，供核对，不是可运行代码库。上游 README 的相对链接不应当作本地 bundle 的导航使用。

## 主要 code / weight 状态

| 工作 | 本次确认 | 未确认 / 条款边界 |
|---|---|---|
| LeWM | 官方 [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) 有 train/eval；HF [lewm-pusht](https://huggingface.co/quentinll/lewm-pusht) 有 weights.pt，MIT | 未验证在本机/集群运行；Google Drive baseline archive 没有下载 |
| DINO-WM | 官方 [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) 有 train/plan 与 PointMaze/PushT/Wall checkpoints 说明，代码 MIT | 其余 paper 环境不推断为都有即用权重 |
| V-JEPA 2 / 2.1 / 2-AC | 官方 [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) 共用 repo，有对应模型入口；根代码 license 为 MIT，并有 Apache license 文件 | 对具体 pretrained artifact 使用范围应看相应权重条款；2.1 encoder 不自动等同 2.1 完整 robot release |
| JEPA-WMs | 官方 [code](https://github.com/facebookresearch/jepa-wms) 与 HF [model repository](https://huggingface.co/facebook/jepa-wms) 可访问，README 给各环境 checkpoint | CC-BY-NC 4.0，有 non-commercial 限制；公开研究资源与无条件 open-source 分开 |
| QuantWM | [官方目录](https://github.com/huawei-noah/noah-research/tree/master/QuantWM) 包含量化和 planning 脚本、readme.md | README 文件名是小写，初次大写路径 404 不代表代码未发布；未审计所有 quantizer 和实际 low-bit backend |
| DreamZero | [官方 code](https://github.com/dreamzero0/dreamzero) 有训练与 server/eval、模型下载说明；代码 Apache-2.0 | 当前 server 路线要求至少两 GPU，GB200/H100 测试；不承诺单 A100 直接复现 |
| LingBot-VA | [code](https://github.com/Robbyant/lingbot-va) 与 [LIBERO-Long weights](https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long) 可访问；weights metadata Apache-2.0 | 此 checkpoint 是 Long，不是全四套 LIBERO |
| LingBot-VA 2.0 | [项目页](https://technology.robbyant.com/lingbot-va-v2) 与独立技术报告公开 | 项目 GitHub/HF 按钮只指组织；本次未确认独立 2.0 code+weights+eval release，故观察项 |
| Fast-WAM | [MIT code](https://github.com/yuantianyuan01/FastWAM)；[HF](https://huggingface.co/yuanty/fastwam) 有 original 与 Optional IDM weights | HF metadata 未列 license，不能自动把 code license 套用全部 weights；当前优化不混入旧论文实验 |
| Faster-WAM / 2608.04404 | [code](https://github.com/hustvl/FasterWAM) 与 [HF](https://huggingface.co/hustvl/FasterWAM) 有 LIBERO/RoboTwin checkpoint | code API 标 Apache-2.0，HF card 标 MIT；按资产各自条款记录，不抹平差异 |
| Cosmos Policy | [code](https://github.com/NVlabs/cosmos-policy) 与 [LIBERO weights](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B) 存在 | code Apache-2.0；HF metadata 无简单 license 标签，不代表权重没有专门条款 |
| Cosmos 3 | [官方 code](https://github.com/NVIDIA/cosmos) 与 [Nano Policy DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) 有实际权重文件 | OpenMDW-1.1；policy DROID 不自动适配 LIBERO；部分低精度支持仍标 coming soon |
| DreamerV3 / TD-MPC2 | 官方训练代码公开，MIT；TD-MPC2 有 pretrained models/data 路线 | 不把第三方 Dreamer4 implementation 当 official continuation |
| SANA-WM | [官方 docs](https://github.com/NVlabs/Sana/blob/main/docs/sana_wm.md) 有 models、inference、FP8/NVFP4 设置 | 官方实现存在不代表本次核实过所有 release 与硬件效果；camera motion 与机器人动作分开 |
| Matrix-Game 3.0 | [官方子目录](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-3) 有 base/distilled 与 INT8 入口 | 根 repo 与子目录 README 的 license 描述不完全同一，使用前以对应资产实际 license 为准 |
| QuantWAMs | paper 与 [project page](https://quantwams.github.io/) 可访问 | Code 显示为不可用按钮，未找到外部 repository href；不能列作已核实开源 baseline |
| Where Bits Matter | [作者 code/run artifacts](https://github.com/suraj-ranganath/DINO-MBQuant) 可访问 | 仅 source-level 可用性；paired statistics、kernel、独立复现未验证 |
| LeWM 独立复现 | [tinylab](https://github.com/joyjeet-singh/tinylab) 与 paper | 单环境、单 seed 的独立报告，不能替代本地复验 |

HF metadata 查询详情保存于 [checkpoint-availability.json](sources/checkpoint-availability.json)，包括 revision、gated flag 与返回的文件名。文件名过滤不表示仓库无其他格式的权重。

## 重要版本边界与资料缺口

- **LeWM**：本地 `2603.19312v3`；当前 official README 的 Paper link 仍指 v1。本库用 v3，不能称为“与 README PDF 完全相同”。
- **JEPA-WMs**：本地 `2512.24497v4`，2026-09-02 更新；首页标 TMLR 05/2026。新版 funding 更新与方法变更要区分。
- **Fast-WAM**：本地 v2；当前 repo 新增 Optional IDM/进一步推理优化。代码结果与旧版本实验并列记录。
- **DreamerV3**：本地 arXiv `2301.04104v2`，标题 Mastering Diverse Domains through World Models；Nature 2025 的 Mastering diverse control tasks through world models 是出版版本的另一标题，见 [Nature](https://doi.org/10.1038/s41586-025-08744-2)。本库不把前者冒充后者的 PDF。
- **QuantWAMs**：v1 为 13 页，最后两页为 references，正文多处指向 Appendix A–E，但文件内没有对应附录。已视觉核对末页；未将缺失内容补写或猜测。Project page 未提供独立 supplementary。这个缺口限制协议复核，不等于实验无效的证据。
- **Faster-WAM**：`2608.04404`（future conditioning）和 `2608.02365`（DoT/action depth）同名不同作；本地核心收前者，后者列观察。
- **RoboCasa**：当前官方有 RoboCasa365 和 horizon 更新。旧 paper subset、最新 365 benchmark 和不同 horizon 的结果不可混报。
- **WorldArena 2.0**：官方网站对 motion metrics 与 track scoring 有更新；读 paper 与跑 leaderboard 时都要记录 scoring revision。

## 检索范围

本次使用公开 web search 定位，最终主要依赖 arXiv、作者项目页、官方 GitHub、官方 Hugging Face 和会议/期刊原文。搜索中出现的第三方综述、论坛和镜像用于发现线索，不作为本地结论的主依据。

代表查询：`Yann LeCun V-JEPA 2 world model open source`、`LeWorldModel github arxiv`、`DINO-WM official`、`world model quantization planning encoder predictor`、`world action quantization`、`QuantWAMs`、`JEPA-WMs`、`Fast-WAM Faster-WAM`、`LingBot-VA 2.0`、`Cosmos Policy Cosmos 3`、`WorldScore Physics-IQ WorldArena`。覆盖 latent WM、robot WAM、model-based RL 基础和 video/interactive WM 的代表路线。

这是面向 FYP 选择的定向研究，未完成所有数据库的系统综述或 citation graph 全量检索；不能证明“除了列出的论文就不存在其他研究”。新兴论文以 preprint/technical report 看待，未使用引用量或 GitHub stars 给 SOTA 排名。
