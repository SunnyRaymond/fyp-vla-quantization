# ROUND5 root gate

TD-MPC2 policy-prior support保留为待独立设计审查的窄候选。原代码确实把policy trajectories插入candidate pool；当前讨论只改变proposal generation，评分器和terminal bootstrap actor必须保持FP。没有通过设计gate前不运行科学screen，不能把普通actor drift或同candidate score fidelity改名。

DINO-WM action-token reinjection **identifiability_no_go / GPU0**。若唯一被量化的是action encoder，再用FP action embedding替换全部action输入，基本是在移除唯一干预；恢复FP是negative control，不是独立PTQ机制收益。更关键的是，action slice被重插入并不清除已经传播到visual/proprio slice的误差。将此案与predictor量化比较还会改变locus/Jacobian，不能由所提对照识别reset boundary的额外作用。本次不退化成普通module sensitivity实验。

Online success-mask feedback **identifiability_no_go / GPU0**，保留原brief；真实环境结果、action drift和分支时刻共同变化，所提replay无法隔离新的量化机制。未运行这些候选，不是经验no-go。

以上是执行前判断，与已完成的rounding-persistence go、teacher-bias no-go无关，不复用其DEV调整新阈值。
