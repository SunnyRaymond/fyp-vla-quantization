# C80 R128 fused VQ：最终实现与测量

本轮实现和测量已完成。主job `25943391.pbs101`（normal→g1，申请6h、实际54m45s）与precision job `25944399.pbs101`（normal→gdev，30min上限）均PBS F/Exit0、artifact0、PIPELINE_COMPLETE及对应完整gate通过。监控subagents已停止。

[完整中文结果](pulled/25943391.pbs101/RESULTS.zh.md)；[原始summary](pulled/25943391.pbs101/summary.json)；[精度诊断](precision_probe/pulled/25944399.pbs101/precision.json)。五模式各80 queries、同一A100、614目标Linears、原C80 R128 A+Z+KL checkpoint不变、无重训/新closed-loop。

- Input fusion：eager p50 6.528s→3.504s，1.863×，完整80条action误差0，peak25.139GB。
- Packed native VQ：p50 7.093s、query peak16.330GB；含direct/meta loader时p50 7.095s、query peak16.438GB、load peak16.139GB。614 encoded modules、0 dense target weights。
- Packed并非等价替代：相对eager first10 motor RMSE0.0449623，gripper分歧6/800。Direct loader相对同packed算术完整80条误差0。自动pipeline成功不表示packed相对checkpoint数值等价。
- Weight storage：target含transform2.991409GiB、4.271052bits/weight；whole archive14.895614GiB；whole tensor payload压缩1.5516×。
- 独立6-shape诊断：切换cuBLAS BF16 reduction flag未改变输出；移除VQ decode的dense Triton仍有主要偏差，GEMM backend/reduction/bias epilogue是后续优先匹配对象。

代码：fused_input.py、fused_vq.py、kernel_runtime.py、kernel_worker.py、tune_worker.py、finalize.py；precision_probe中为无整模型forward的数值诊断。Torch/Triton/model IO与所有编译仅在获批PBS compute allocation执行。GPU利用率/显存按15秒写入各job log。

控制：python control.py status|verify|logs|pull。Pull将远程自动报告保存为RESULTS.generated.md，保留本地评阅的RESULTS.zh.md。未下载checkpoint或fulltrace。

旧25943033因parent25942669失败未进入allocation；旧源码/状态在attempts。新main修复了PyTorch scalar /7与Triton div_rn的不等价，并完成全部重新验证。Precision初始25944338在Q时发现队列guard预期不符后取消并核验F，无allocation；最终precision25944399成功。没有修改已运行artifacts。
