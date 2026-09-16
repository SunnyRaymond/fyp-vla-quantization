# 提交前实现审查与修正

本次修正在任何 teacher-bias 样本输出或作业提交之前完成，不是根据实验结果改方法。主协议仍为六条Wall124–129、H1/H5、FP/W4/W8和4/6 H5改善5%的冻结gate。

- CPU配置字段修正到 `env.dataset.split_mode/split_ratio`。Source检查保留过去已批准的Wall-only注册与DINOv2 commit pin两处adapter，拒绝其他tracked修改；逐文件保存实际hash。不能用whole-tree clean否决已批准的runtime。
- 第二次独立审查混淆了本地multiline adapter与历史CCDS安装器写入的compact adapter；root依据 `prepare_assets_ccds.py:31` 撤销其“确定blocker”判断，保留实际CCDS期待文本。删除没有indices时的identity mapping fallback，要求明确TrajSubset映射并记录split seed42。
- GPU加载前核对实际Git HEAD、CPU冻结source/config/checkpoint及其hash。输入manifest需在CPU准备完成后从小型回执独立冻结SHA256；目前不存在该freeze，wrapper因此不能直接启动模型。
- 独立审查曾误认为 `pred_proprio` 应为raw4，现已更正：learned proprio encoder产生10D embedding。GPU保存encoded target/prediction10，并另存raw normalized target4，均不并入visual primary。
- 初始feature一致性使用与协议一致的allclose(atol1e-6,rtol0)，记录实际max差值；保存两种量化参数各一次，避免按sample重复写大文件。
- 使用独立 `teacher_bias_protocol.zh.md` alias，保存runner/protocol/input/snapshot identities；不复用可能被别的candidate覆盖的generic protocol文件。

本地只执行源码与shell语法审查，未运行数据准备、模型或数值验证。所有重操作仍要求真实CCDS SLURM allocation；当前网络超时，无teacher作业、无科学结论。

最终CPU verifier与launcher已交付：verify_teacher_bias.py、teacher_bias_cpu_verify.sh。Source键统一相对source root，并覆盖已批准的dino/env adapter；runner pin为3c1166398258c191ed144e8c5e146105b73152c6aff7dc0dac3783feabf25176。Verifier在producer未完成时不加载partial科学数组，独立检查qparams finite/shape/integer/range/positive scale/dequant一致性，但不宣称完整重现CUDA RTN。CPU输出使用controller可读取的numeric job目录。AST和shell syntax通过；所有运行检查仍待真实allocation。
