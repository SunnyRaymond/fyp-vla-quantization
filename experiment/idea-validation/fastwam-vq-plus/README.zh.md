# Row-scale VQ 与 Microscaling VQ：LIBERO-Plus closed-loop evaluation

当前作业：`25766937.pbs101`。一次PBS申请4张GPU、24小时，分别评测两组已冻结VQ bank，每组280 episodes，共560。Worker 0/1评测row-scale VQ，worker 2/3评测B32 power-of-two microscaling VQ；各worker负责140个variants，正常单次运行总计4次模型加载。每卡保持一个persistent worker，20秒错开启动以减小加载峰值。

直接复用原评测的280个variants：4 suites × 7 perturbation dimensions × 10 variants。每个variant使用相同state 0、30 settling steps、environment和sampler seeds；checkpoint为clean Optional-IDM，IDM模式，10个video和10个action denoising steps、32-step action chunk，每次执行前10步后replan，普通suite最多400控制步，Long最多700步。原BF16、QuaRot、SpinQuant与Smooth/Hadamard Scalar的1120条episode结果用于配对汇总，不重新运行。

两组VQ复用原books、scales、indices与Smooth α=0.5/H128变换，不重新拟合。沿用其原动态per-row A4、decoded-BF16计算路径，KV保持BF16；这轮验证closed-loop成功率，不将VQ标为native INT4。显存记录反映decoded weights实际驻留，不能由约4-bit编码存储直接推断GPU显存节约。

每个worker在同一次模型加载内，重放其bank的一个已保存selection action（case 1000、seed 0），与源reference一致后才进入正式episodes。正式query计数从0开始。每个完整episode立即保存；接近24小时软截止时保留完结结果并正常checkpoint退出，下一次allocation仅补未完成episode。GPU利用率与显存每15秒写入job log；任一worker失败时终止本作业的其余workers。

当前来源：row-scale VQ `25743693.pbs101`；microscaling VQ `25755896.pbs101`。新任务保留原来源配置，两个配置各有614个目标Linear。成功率、suite/dimension分项、逐variant配对胜负、query耗时及tensor显存将在compute allocation内汇总。

四组旧基线与两组VQ的训练数据、变换及执行路径有所不同；结果比较完整配置。Closed-loop query来自不同轨迹，不是同输入microbenchmark。原四基线已经看过，新增配置的复用对照也不是独立预注册的新cohort。

- [固定协议](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-vq-plus/protocol.json>)
- [当前作业状态](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-vq-plus/RUN_STATE.json>)
- [PBS入口](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-vq-plus/run.pbs>)
