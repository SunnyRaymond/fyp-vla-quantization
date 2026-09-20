# 外部资产说明

本目录只保存 predictor idea 的代码、实验契约、历史小型结果和报告。运行 official DINO-WM teacher 所需的 source、checkpoint、PushT data、Python overlay 和 runtime archive 保留在 ASPIRE2A 原始工作根目录中，不复制进本 bundle。

默认外部根目录：

```text
/scratch/users/ntu/yguo017/dino-wm-wall
```

PBS wrapper 通过 `DINO_WM_ROOT`、`DINO_PUSHT_CHECKPOINT`、`DINO_PUSHT_DATA_ROOT`、`DINO_PUSHT_DEPS_ROOT` 和 `DINO_RUNTIME_ARCHIVE` 指向它们。若目录布局改变，请只在提交作业时覆盖这些环境变量，不要修改历史 freeze 或 summary。
