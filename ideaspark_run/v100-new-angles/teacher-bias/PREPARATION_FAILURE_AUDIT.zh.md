# CPU准备失败审查

64824（TC1N05，4s，exit4）在source检查阶段停止，无样本或模型结果；status仅报Git检查失败。原目录和artifact保留。

64826短CPU诊断明确：git2.52.0可用，HEAD等于冻结commit，仅env/dino两个批准修改；`git show HEAD:models/dino.py`却触发 `failed to stat ... Input/output error`。不把这一错误解释成Git source损坏或模型不可用。最小接口修复是用 `git cat-file blob HEAD:models/dino.py` 直接读取同一Git object，并保留stderr帮助后续诊断；未修改repository内容或放宽source gate。

64827在新asset目录完成相同准备，证明该命令替换解决了当前阻断；仍使用相同六条indices、source/checkpoint和数据处理。实际raw proprio维数检查与修正见INPUT_FREEZE，不是根据模型表现挑选实现。

网络恢复由用户确认GlobalProtect VPN已恢复，并由SSH连接成功复核。首次同时提交第三个job被QOSMaxSubmitJobPerUserLimit拒绝，没有生成GPU job ID；等前两个CPU退出后再提交64825，未换QoS或绕过限制。
