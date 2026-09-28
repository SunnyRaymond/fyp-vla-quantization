# E0：动作响应数学单元测试

本目录中的 E0 使用合成函数检查中心有限差分和一阶 Taylor residual 的基本行为。测试只依赖 Python 标准库，不加载 LeWM、PushT、GPU 或外部数据。

覆盖四类函数：

- **仿射函数**：中心二阶差分和 Taylor residual 接近零；以平方距离为 criterion 时，线性化得分与精确得分相同。
- **二次函数**：方向二阶差分除以半径平方后保持为常数，二阶差分平方除以半径四次方也保持为常数；一阶 Taylor residual 随半径平方增长。
- **原点处的三次函数** `f(x)=x³`：中心二阶差分恰为零，但有限步长 Taylor residual 为 `δ³`。因此，某个中心和半径上的二阶差分低，不代表 Taylor residual 低。
- **交互项** `f(x,y)=xy`：沿两个坐标轴分别扰动时二阶差分为零，沿混合方向扰动时可观察到非零二阶差分。

## 运行

在项目根目录运行：

```powershell
python -m unittest discover -s meeting-ready/03-action-response-straightening -p "test_*.py" -v
```

这些测试验证数学计算，不验证 LeWM 的自动微分、动作维度与 normalization、history 传递或完整 rollout 接口；这些属于后续模型接口 gate。
