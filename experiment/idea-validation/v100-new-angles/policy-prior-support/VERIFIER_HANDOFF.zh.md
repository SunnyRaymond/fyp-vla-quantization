# 最终 CPU verifier 合同

2026-09-13，root按实际GPU64833合同收敛。此前草案中额外的schema aliases、episode_ids和transaction JSON要求已移除；科学门槛与GPU数据未改动，也没有为适配草案重跑GPU。

CPU64834已提交，INPUT=/tc1home/UG/yguo017/v100_newangles_ccds/artifacts/64833。独立replay、verifier与launcher均先复制到当前job目录再执行，SLURM guard检查实际allocation，CPU内读取hash/raw/snapshot。GPU完整8states、elapsed11.614s，所有producer guards true；科学结论待CPU verification.json。

- raw_policy_support.npz：actions[8,3,3,512,1]、values[8,3,512]、elite_indices/weights[8,3,64]、mu/std[8,3,3,1]、mu_values/masses/counts[8,3]、completed[8]、observations[8,5]、reset_seeds、arm_names、metadata_json。
- quant_snapshot.pt：三个_pi.N.weight映射到fp/scale/codes/dequant/readback_q/readback_fp，均保存CPU tensor。CPU验证实际tensor的形状、有限性、integer grid、per-row scale、dequant/readback和FP恢复。源代码与执行hash绑定CUDA RTN recipe；不声称CPU精确重演所有CUDA舍入指令。
- engineering.json、runtime.json、rng.json，以及exact bytes的input/parent manifest、protocol/inputfreeze/runner/helper：独立核对hash、source/checkpoint/runtime、actualallocation、seed和RNG前后状态、non-actor/model digest恢复。所有requiredgates显式列举。
- policy_support_replay.py：独立NumPy重算top64、weights、mu/std、source slotmass/count和冻结G/binding/joint gate；model J仅为绑定的FP scorer输出，CPU不重跑模型。

Verifier LF SHA925b92295c5da0f817a22a0be7c8e6f4f0523cb2b0e0b664d16975ecc4f023d9；replay LF SHA1a69ee65256f70c8656520ff5a4c7948d303e025aebe4ef7a806b975ab35b9fc；producer LF SHA0fd129c569663c3321485e9ab2d770b4833ad3a01547fb4b2eda0e850cf1eece。各report小于64KiB，原始数据留在compute存储。

Producer未完成/缺失时先归档budget或implementation inconclusive，不要求partial raw齐全或计算partial科学结果。最终完整结果无论正负均STOP，不扩seed、bitwidth、完整planner或closed-loop。
