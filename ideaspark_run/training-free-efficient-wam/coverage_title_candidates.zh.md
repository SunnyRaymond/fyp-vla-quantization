# Host coverage 候选标题（待 connector 核实和去重）

以下为 2026-10-02 primary arXiv 页面核实的标题，用于 Phase 0.5 检查缺失锚点。不是手工创建的 lit_results 记录，也不代替 connector retrieval。

1. Efficient World Action Model Inference with Adaptive Intermediate States — https://arxiv.org/abs/2609.34608 （WAMachine；Sep 28，cross-replan remapping / observation rebinding / layer residual rescaling）
2. DriveCache: Action-Aware Caching for Driving World Model Inference — https://arxiv.org/abs/2608.16354 （action-aware calibrated cache budget，和已有 CREC 提案接近）
3. Efficient-WAM: A 1B-Parameter World-Action Model with Low-Cost Future Imagination — https://arxiv.org/abs/2606.10040v3 （最新 Sep 26；非对称 video/action denoise 的近邻）
4. FBFM: A Training-Free Asynchronous Feedback Mechanism for Flow-Matching in World-Action Models Execution — https://arxiv.org/abs/2607.29235 （在线 masked pseudoinverse feedback）
5. Test-Time Scaling for World Action Models via Zero-Shot Geometric Evaluation — https://arxiv.org/abs/2607.17454 （action-future consistency gate + geometric Best-of-N）
6. Motus: A Unified Latent Action World Model — https://arxiv.org/abs/2512.13030 （可公开复现的 joint-WAM 后备 baseline；需另查 matching checkpoint）

初始 arXiv pool 已包含 Sparse-WAM、C³ache、GlanceWAM、RIFT、Staircase Policy、DVAC 和 X-Cache；merge 后再判是否缺失，不重复补入。Host recall 应只补缺失且 load-bearing 的论文，总数不超过 8。

生成阶段必须检查这些近邻，不允许将 warm-start、跨 chunk residual cache、asynchronous lookahead、观测重绑定、非对称 denoise 或普通 variance-based adaptive chunking 直接换名成新机制。
