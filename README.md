# FYP: VLA Quantization

此 repo 用于备份本地 FYP 工作：research ideas、论文 PDF、code audits、实验脚本、配置和评估结果。

## 内容入口

- [VLA papers](papers/vla/README.md)、[WAM papers](papers/wam/README.md)、[WM papers](papers/wm/README.md)：每个分类下直接按编号进入单篇 paper。

- `idea/`：现有 research ideas、IdeaSpark pipeline、prior-art review 和 idea briefs；明确的验证运行放在 `experiment/idea-validation/`。
- `experiment/idea-validation/`：对现有 ideas 的 screen、pilot、verification 和运行产物。
- `experiment/reproduction/`：paper/code audits、quantization reproduction、VLA evaluation 和 benchmark 运行。
- `papers/vla/`：VLA、VLA quantization、simulation evaluation 及其单篇 paper 文件夹。
- `papers/wam/`：World Action Model papers 的单篇文件夹。
- `papers/wm/`：World Model papers 的单篇文件夹。
- `ASPIRE2A_README.md`、`nscc-access/`：集群使用说明与脚本。

## 备份范围

`.gitignore` 排除 checkpoints、模型权重、Python environments、缓存、wheelhouse、container images、压缩包、临时文件和凭据。本次整理只删除明确的冗余总结、旧入口和生成缓存；论文 PDF 保留。

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
