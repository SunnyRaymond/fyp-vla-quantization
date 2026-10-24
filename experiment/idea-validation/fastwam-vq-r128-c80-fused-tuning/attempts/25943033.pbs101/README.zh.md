# C80 R128 fused VQ 布局与tiling调优

同一goal线程01a167e5-84ea-7133-a45f-de17f7cb4280。保持冻结C80 R128 A+Z+KL checkpoint、614目标Linears、40 contexts×2 seeds、10+10steps、BF16KV，无重训。

PBS `25943033.pbs101` / `vqft0001`，06:00:00，1GPU/16CPUs/120GB，normal→g1；依赖`afterok:25942669.pbs101`，需首版pipeline全部成功后才运行。monitor为`/root/fused_tuning_watch`，gpt-6-luna/xhigh。首检已确认H及精确依赖，是正常等待，不等于阻塞或失败。

触发依据：首版614层decoder已BF16 bitwise通过，但video FFN的packed GEMM microbench比dense慢约11–16倍。新kernel改为[N,K]连续lookup后转置给MMA，并增加M tile复用；12代表shapes×12候选，先GPU数值gate再按3次计时block的median选择。配置在完整模型测量前冻结，所有主mode记录实际dispatch。

五个fresh-process模式各80query：BF16/eager/input_fused/packed_fused/packed_direct。timed614模块调用数应为每mode514880。输入fusion需完整80输出parity，614decoder比较BF16原始位模式；packedGEMM输出误差单独汇报，不能承诺闭环准确率不变。Direct loader必须与相同packed算术完整80输出一致。GPU利用率/显存每15秒记录。

当前仅源码/stdlb/PBS语法检查通过，未取得本任务GPU结果；首版microbench属于另一job，不能当本任务E2E speedup。代码和protocol已冻结上传。Controller只处理bounded小文件和PBS控制，所有编译/推理/模型IO都在compute allocation内。

控制：python control.py status|logs|verify|pull。最终summary必须完整五组80、同GPU、kernel和tuning gate成立，PBSF/Exit0且PIPELINE_COMPLETE才确认完成。
