# 可用实现锚点（2026-10-03 一手页面核验）

- [FastWAM 官方仓库](https://github.com/yuantianyuan01/FastWAM/blob/main/README.md) 有 LIBERO / RoboTwin 训练、评估与 released checkpoints；Optional IDM checkpoint 支持 `idm` / `first_frame` 两种不同 inference modes，必须冻结 mode、checkpoint、scheduler、controller 设置。其官方 dataset 为 `yuanty/LIBERO-fastwam`。新候选先限定一个 mode，不把旧 idm 失败解释成 joint 或 first_frame 证据。
- [Nunchaku 官方仓库](https://github.com/nunchux-ai/nunchaku) / [安装文档](https://nunchaku.tech/docs/nunchaku/installation/installation.html) 是 diffusion W4A4 native kernel 工程参考。当前支持的模型路径不能直接视作 FastWAM drop-in；MoT shape、joint attention、AdaLN、batch 和 GPU 均需适配与实测。
- [QServe / OmniServe 官方仓库](https://github.com/mit-han-lab/omniserve) 提供 W4A8/W8A8 GEMM、DeepCompressor QoQ 校准与 packed conversion 工程参考；已展示的 model zoo 是 LLM，并不建立 WAM task fidelity 或 WAM latency。原 QServe URL 已重定向至 OmniServe。

上述链接只确认现有资源与工程参考，不属于本轮 collision pool；不把 LLM/image diffusion 的实测 speedup 搬到 WAM。GPU 与 kernel 适配尚未执行，预算只能给建议 pilot 范围，不能给 measured cost。
