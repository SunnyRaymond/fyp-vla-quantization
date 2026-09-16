# CEM-Update PTQ：CCDS Wall 快速验证结果

2026-09-12。结论：**当前冻结配方 Stage B no-go，不进入更贵的闭环 Stage C**。实验完整运行，独立CPU核验工程通过，机制门槛失败且无临界歧义。不是超时或环境失败，也不能推广为所有CEM-update量化方法均无效。

## 已有证据

正式GPU job64670完成，TC1N05、1张Tesla V100-PCIE-32GB、22分39秒。四方法均完成两轮，每方法71次配置评价，共享缓存实际83种CAL配置。CPU独立核验64676完成，2CPU、无GPU、7秒：`engineering_pass=true`、`mechanism_gate_pass=false`、`mechanism_gate_status=fail`。

核验从原始候选、分数和搜索轨迹重新计算；检查CAL/DEV reference与workload逐pool一致、跨方法second-step reference一致。新增检查的self-tests也在该CPU allocation通过。独立核验未重新加载模型推理，模型恢复仍依据runner exact-weight assertion；这些限制保留在verification.json中。

以下为独立CPU核验值，均越低越好。DEV仅4 episodes、每episode CEM第1和第5轮各一个pool；two-step为配对候选重放，不是环境闭环。

| 方法 | DEV L_update | DEV L_mu | two-step final-mu MSE |
|---|---:|---:|---:|
| shared start | 0.054280 | 0.046051 | 0.063511 |
| CEM-Update | 0.044866 | 0.037586 | 0.060577 |
| MeanOnly | 0.044866 | 0.037586 | 0.060577 |
| ScoreError | 0.043242 | 0.037774 | 0.045147 |
| Rank | 0.043242 | 0.037774 | 0.045147 |

CEM-Update与MeanOnly选中相同map `1964a65ab303`；ScoreError与Rank选中相同map `b8146ca05adb`。因此本轮没有显示sigma项的增量价值。

相对shared start，CEM-Update的L_update约降低17.3%，4/4 episodes改善；two-step误差仅降低约4.6%，虽3/4 episodes改善，仍未达到预定5%平均改善门槛。更实质的问题是它与MeanOnly持平，并且L_update及two-step误差都劣于ScoreError/Rank。two-step误差相对ScoreError/Rank约高34.2%。结论不只依赖4.6%与5%的窄差距。

## 研究假设与判断

假设是用实际elite均值与标准差更新误差指导完整混合精度配置，能获得比均值单项、分数误差或排序目标更好的新episode更新与后续搜索保真度。本轮CAL目标确有稳定两轮改进，但DEV未显示优于对照，sigma项没有改变最终配置，后续搜索误差也没有优于简单对照。按事前经济性门槛，应停止当前配方，优先换idea，而不是继续调DEV权重或扩大闭环预算。

这是一个小样本、局部两轮搜索的早停决策，不是统计显著性检验；也未穷尽起点、搜索空间或量化器。不能声称证明整类方法不可能有效。若未来修改，应作为新假设另行冻结，并使用新验证目标。

## 待定与未做

未进行量化方法的Stage C新闭环成功率比较。FP32轨迹采集仅供候选池构造，不能充当方法闭环收益。weight-only RTN使用FP32 operators，未验证native低比特速度或显存收益。不会为当前配方追加GPU实验。

## 资源与保留记录

CCDS GPU使用：预检64668为4分54秒，正式搜索64670为22分39秒，合计27分33秒、约**0.459 GPU hours**。CPU准备21分42秒，CPU核验7秒，均不计GPU hours；另有秒级CPU探测及失败后修复的NFS cache准备。墙钟总耗时还包含排队、控制和定时检查间隔。

ASPIRE2A旧目录`../cem-update-ptq/`完整保留，本轮未连接NSCC。旧搜索17078256最终结果与实际用量仍未知，不能计为已完成或并入CCDS结论。此前确认的ASPIRE预检453秒另计，不包含在上述CCDS用量内。

证据文件：`artifacts/64670/summary.json`、`dev_metrics.json`、`selected_maps.json`、`slurm_status.txt`；`artifacts/64676/verification.json`、`verify.log`、`slurm_status.txt`。完整大数组仍留在CCDS compute存储。credentials读取路径已统一至项目根credentials.env；未上传secret。
