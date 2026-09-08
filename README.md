# FYP: VLA Quantization

此 repo 用于备份本地 FYP 工作：reading guides、论文 PDF、research ideas、code audits、实验脚本、配置和评估结果。

## 内容入口

- [WM / WAM 量化专题阅读库](literature%20review/wm-wam-quantization-reading-guide/README.md)：LeCun 相关 baseline、代表开源工作、量化 prior art 与 benchmark。

- `reading-guide/`：主阅读路线。
- `vla-quantization-literature-review-alternative/`：VLA quantization 文献整理。
- `dan-alistarh-quantization-reading-guide/`：quantization 专题。
- `sim-eval-vla-reading-guide/`：simulation evaluation 阅读材料。
- `vla-ideas-reading-guide/`：研究方向与参考材料。
- `bitvla-code-audit-2026-09-05/`、`qvla-code-audit-2026-09-05/`：paper/code audits。
- `openvla-oft-vla-eval-libero/`：OpenVLA-OFT / LIBERO evaluation 工作。
- `ASPIRE2A_README.md`、`nscc-access/`：集群使用说明与脚本。

## 备份范围

`.gitignore` 排除 checkpoints、模型权重、Python environments、缓存、wheelhouse、container images、压缩包、临时文件和凭据。本地文件不会被删除。论文 PDF 保留。

第三方 repo 以实际源码快照保存，上游地址和 commit 见 `UPSTREAM_SOURCES.json`；它们的独立 Git history 不在此备份内。保留各源码中的 license。

部分第三方文件沿用上游 `.gitattributes` 的 Git LFS 配置。恢复时先安装 Git LFS，再运行 `git lfs install`、`git clone`；若已 clone，可在 repo 中运行 `git lfs pull` 获取实际文件。

第三方 Transformers 测试工具中的硬编码 token 已替换为 `HF_TEST_TOKEN` 环境变量；需要相关远程测试时自行在本地设置。

这不是完整磁盘备份。恢复实验时，需按照对应目录文档重新下载模型、依赖和 images，并在本地重新配置 credentials。

## 后续更新

在此目录运行：

```powershell
git add --all
python scripts/check_archive_size.py
git commit -m "Update FYP backup"
git push
```

大小检查会拒绝暂存区内超过 50 MiB 的单个文件。`.gitignore` 本身不能按大小筛选；若发现新类型的大文件，先把相应路径加入 `.gitignore` 并取消该文件的 staging。大小检查不扫描 secrets，提交前仍需避免将 credentials 写入源码或文档。
