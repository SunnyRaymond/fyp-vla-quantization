# Cached KL candidate cross-check

这是同一冻结 C80、同一 MSE80 起点上的两个已有候选集交叉评分。只读取 JSON；没有 held-out、forward、backward、teacher 查询或训练更新。各套每个 condition 80 cases，四个 suite 各 20。

公式：`Jm = action_mse + 0.0424545268 × latent_nmse`；`Jk = 0.0123024793 × action_kl + 0.0424545268 × 0.312978952 × latent_kl`。两公式分别交叉评分；各自系数固定来自原有 C80 标定。

## 候选集得分与原始指标均值

| 候选集 | family | fraction | Jm mean | Jk mean | mean case motor RMSE | latent NMSE | action KL | latent KL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| original MSE80 | source/no-step | 0 | 0.0028503231 | 0.0027439962 | 0.029800742 | 0.04468495 | 0.068850856 | 0.14276411 |
| mse | saved_adam | 1.0 | 0.0030819553 | 0.0029558163 | 0.031161736 | 0.04782168 | 0.074420549 | 0.15354871 |
| mse | saved_adam | 0.5 | 0.0031277291 | 0.0030377859 | 0.030287354 | 0.050542181 | 0.072199287 | 0.16177431 |
| mse | saved_adam | 0.25 | 0.0029187005 | 0.002842453 | 0.029296802 | 0.047382962 | 0.066377502 | 0.15246393 |
| mse | fresh_adam | 1.0 | 0.00307094 | 0.0029629072 | 0.030436942 | 0.049479258 | 0.069700974 | 0.15845212 |
| mse | fresh_adam | 0.5 | 0.0028740869 | 0.0028131626 | 0.02930273 | 0.046296013 | 0.068360937 | 0.14842313 |
| mse | fresh_adam | 0.25 | 0.0027972171 | 0.0026590234 | 0.029974242 | 0.043212995 | 0.067908279 | 0.13724182 |
| kl | saved_adam | 1.0 | 0.0027679682 | 0.0026860427 | 0.029925598 | 0.043066042 | 0.069094483 | 0.13817699 |
| kl | saved_adam | 0.5 | 0.0030969412 | 0.0029944666 | 0.030443463 | 0.049696684 | 0.070880428 | 0.15973522 |
| kl | saved_adam | 0.25 | 0.0030793945 | 0.0030329826 | 0.029414606 | 0.050939844 | 0.071396826 | 0.1621558 |
| kl | fresh_adam | 1.0 | 0.0029874804 | 0.002911756 | 0.030212905 | 0.047178868 | 0.07452419 | 0.1501368 |
| kl | fresh_adam | 0.5 | 0.0029606759 | 0.0028911979 | 0.02975038 | 0.047752216 | 0.06994157 | 0.15283256 |
| kl | fresh_adam | 0.25 | 0.0029104676 | 0.0028227625 | 0.029979834 | 0.045915773 | 0.071792908 | 0.14596803 |

## 相对共同 no-step 的配对变化

Δ 为每 case 候选减 baseline；负值表示公式分数下降。suite 列按四个 suite 等权列出 mean Δ。

| 候选 | 公式 | mean Δ | mean Δ% | median case Δ | improved cases | suite mean Δ |
|---|---|---:|---:|---:|---:|---|
| mse/saved_adam/1.0 | mse_nmse | 0.00023163218 | 8.126523523602433 | 4.1994964e-05 | 39 | libero_10:0.000207272, libero_goal:0.000282709, libero_object:-4.55864e-05, libero_spatial:0.000482134 |
| mse/saved_adam/1.0 | dual_kl | 0.00021182006 | 7.71940047724552 | 2.4524452e-05 | 39 | libero_10:0.000213876, libero_goal:0.0003062, libero_object:-3.21658e-05, libero_spatial:0.000359371 |
| mse/saved_adam/0.5 | mse_nmse | 0.00027740606 | 9.732442576799219 | 0.00010676369 | 36 | libero_10:0.000337149, libero_goal:-0.000117178, libero_object:0.000582327, libero_spatial:0.000307326 |
| mse/saved_adam/0.5 | dual_kl | 0.00029378969 | 10.70663631369622 | 6.5686167e-05 | 36 | libero_10:0.000437657, libero_goal:-7.30939e-05, libero_object:0.00052979, libero_spatial:0.000280806 |
| mse/saved_adam/0.25 | mse_nmse | 6.8377403e-05 | 2.3989351917617983 | 3.3937445e-06 | 40 | libero_10:6.14482e-05, libero_goal:0.000232228, libero_object:-2.46456e-05, libero_spatial:4.47938e-06 |
| mse/saved_adam/0.25 | dual_kl | 9.8456792e-05 | 3.588080472446181 | 8.8923919e-05 | 38 | libero_10:8.10384e-05, libero_goal:0.000350633, libero_object:-6.72654e-05, libero_spatial:2.94211e-05 |
| mse/fresh_adam/1.0 | mse_nmse | 0.00022061689 | 7.740065951468874 | 9.8980243e-05 | 38 | libero_10:0.000148913, libero_goal:0.000299175, libero_object:0.000356009, libero_spatial:7.83707e-05 |
| mse/fresh_adam/1.0 | dual_kl | 0.00021891101 | 7.977817550691307 | 7.9701225e-06 | 40 | libero_10:0.000188695, libero_goal:0.000338614, libero_object:0.000302044, libero_spatial:4.62908e-05 |
| mse/fresh_adam/0.5 | mse_nmse | 2.3763822e-05 | 0.8337237949860005 | 6.3289965e-05 | 38 | libero_10:3.77198e-06, libero_goal:-0.000141393, libero_object:0.000288326, libero_spatial:-5.56488e-05 |
| mse/fresh_adam/0.5 | dual_kl | 6.9166366e-05 | 2.520643658168413 | 8.2196564e-05 | 35 | libero_10:7.38866e-05, libero_goal:-6.98934e-05, libero_object:0.000288524, libero_spatial:-1.58515e-05 |
| mse/fresh_adam/0.25 | mse_nmse | -5.3105994e-05 | -1.8631570097977468 | 1.9139719e-05 | 38 | libero_10:3.6103e-06, libero_goal:-0.000518129, libero_object:0.000212363, libero_spatial:8.97325e-05 |
| mse/fresh_adam/0.25 | dual_kl | -8.4972759e-05 | -3.096679201062062 | -8.3429571e-05 | 44 | libero_10:-1.41114e-05, libero_goal:-0.000515365, libero_object:0.000177112, libero_spatial:1.24726e-05 |
| kl/saved_adam/1.0 | mse_nmse | -8.2354835e-05 | -2.8893157979847386 | -9.6279852e-05 | 44 | libero_10:0.000170384, libero_goal:-0.000439227, libero_object:-0.000287118, libero_spatial:0.000226541 |
| kl/saved_adam/1.0 | dual_kl | -5.7953501e-05 | -2.112010998560477 | -0.00017154178 | 43 | libero_10:0.00020008, libero_goal:-0.000317537, libero_object:-0.000263939, libero_spatial:0.000149582 |
| kl/saved_adam/0.5 | mse_nmse | 0.00024661816 | 8.652288042844999 | 9.3760027e-05 | 39 | libero_10:0.00010126, libero_goal:0.000312118, libero_object:0.000168638, libero_spatial:0.000404457 |
| kl/saved_adam/0.5 | dual_kl | 0.00025047037 | 9.127941484631133 | 6.7986252e-05 | 37 | libero_10:0.000115896, libero_goal:0.000407749, libero_object:0.000148512, libero_spatial:0.000329724 |
| kl/saved_adam/0.25 | mse_nmse | 0.00022907145 | 8.036683618055111 | -1.1440921e-05 | 41 | libero_10:-7.69092e-06, libero_goal:0.000134509, libero_object:-5.04829e-05, libero_spatial:0.00083995 |
| kl/saved_adam/0.25 | dual_kl | 0.00028898639 | 10.531588404620637 | 2.2513692e-05 | 40 | libero_10:7.52515e-05, libero_goal:0.000251354, libero_object:-9.95461e-06, libero_spatial:0.000839295 |
| kl/fresh_adam/1.0 | mse_nmse | 0.00013715729 | 4.811991171066425 | -6.1240481e-05 | 45 | libero_10:-0.000137624, libero_goal:0.000476995, libero_object:4.26467e-05, libero_spatial:0.000166612 |
| kl/fresh_adam/1.0 | dual_kl | 0.00016775977 | 6.1137028841494745 | -7.7368391e-06 | 40 | libero_10:-0.000113271, libero_goal:0.000607282, libero_object:1.64915e-05, libero_spatial:0.000160536 |
| kl/fresh_adam/0.5 | mse_nmse | 0.00011035278 | 3.871588373039532 | 0.00014524102 | 35 | libero_10:-0.000121155, libero_goal:5.2799e-06, libero_object:0.000176306, libero_spatial:0.00038098 |
| kl/fresh_adam/0.5 | dual_kl | 0.00014720174 | 5.364502435987386 | 0.00014208107 | 34 | libero_10:-0.000120501, libero_goal:0.000110036, libero_object:0.000233807, libero_spatial:0.000365465 |
| kl/fresh_adam/0.25 | mse_nmse | 6.0144563e-05 | 2.1100963353499353 | 1.1291253e-05 | 39 | libero_10:-5.88743e-05, libero_goal:-2.91083e-05, libero_object:-0.000258247, libero_spatial:0.000586808 |
| kl/fresh_adam/0.25 | dual_kl | 7.8766291e-05 | 2.8704956239841493 | 9.0174504e-05 | 36 | libero_10:-4.89642e-05, libero_goal:5.12112e-05, libero_object:-0.00021302, libero_spatial:0.000525839 |

## 同 family/fraction 下 KL 候选减 MSE 候选

目标差值定义为 KL 候选减 MSE 候选，负值代表对应公式分数较低。配对 80 个 C80 case；这些是固定队列的描述统计，不是显著性或跨训练轨迹推断。

| family | fraction | Jm mean/median Δ | Jm improved | Jm suite Δ | Jk mean/median Δ | Jk improved | Jk suite Δ | motor RMSE Δ | latent NMSE Δ | action lower bound mean/max | latent lower bound mean/max |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| saved_adam | 1.0 | -0.00031398701/-0.00015012031 | 44 | libero_10:-3.68871e-05, libero_goal:-0.000721936, libero_object:-0.000241531, libero_spatial:-0.000255593 | -0.00026977356/-0.00011223561 | 48 | libero_10:-1.37959e-05, libero_goal:-0.000623737, libero_object:-0.000231773, libero_spatial:-0.000209788 | -0.0012361379 | -0.0047556384 | action 0.004076377/0.031001438 | 0.036177011/0.26181648 |
| saved_adam | 0.5 | -3.0787894e-05/7.0255824e-05 | 35 | libero_10:-0.00023589, libero_goal:0.000429296, libero_object:-0.000413689, libero_spatial:9.71304e-05 | -4.3319326e-05/4.1652487e-05 | 37 | libero_10:-0.00032176, libero_goal:0.000480843, libero_object:-0.000381278, libero_spatial:4.8918e-05 | 0.00015610882 | -0.00084549696 | action 0.0046129337/0.023337986 | 0.047139299/0.22739551 |
| saved_adam | 0.25 | 0.00016069404/-9.4935102e-06 | 40 | libero_10:-6.91391e-05, libero_goal:-9.77184e-05, libero_object:-2.58374e-05, libero_spatial:0.000835471 | 0.00019052959/5.3329349e-05 | 39 | libero_10:-5.78692e-06, libero_goal:-9.92794e-05, libero_object:5.73108e-05, libero_spatial:0.000809874 | 0.00011780461 | 0.0035568815 | action 0.0039997537/0.016409923 | 0.038256012/0.33124544 |
| fresh_adam | 1.0 | -8.3459591e-05/-0.00014277557 | 46 | libero_10:-0.000286537, libero_goal:0.000177819, libero_object:-0.000313362, libero_spatial:8.82411e-05 | -5.1151236e-05/-9.3993029e-05 | 45 | libero_10:-0.000301966, libero_goal:0.000268668, libero_object:-0.000285553, libero_spatial:0.000114246 | -0.00022403714 | -0.0023003908 | action 0.0036006901/0.033061977 | 0.041843286/0.36762795 |
| fresh_adam | 0.5 | 8.6588955e-05/3.7424271e-05 | 38 | libero_10:-0.000124927, libero_goal:0.000146673, libero_object:-0.000112019, libero_spatial:0.000436629 | 7.8035377e-05/3.0477358e-05 | 39 | libero_10:-0.000194388, libero_goal:0.00017993, libero_object:-5.47167e-05, libero_spatial:0.000381316 | 0.00044765021 | 0.0014562028 | action 0.0039847888/0.023908375 | 0.041849091/0.19859978 |
| fresh_adam | 0.25 | 0.00011325056/1.0187117e-05 | 40 | libero_10:-6.24846e-05, libero_goal:0.000489021, libero_object:-0.00047061, libero_spatial:0.000497076 | 0.00016373905/0.00012521454 | 34 | libero_10:-3.48528e-05, libero_goal:0.000566576, libero_object:-0.000390133, libero_spatial:0.000513366 | 5.5913685e-06 | 0.0027027774 | action 0.0040741476/0.022393502 | 0.041803947/0.17634446 |

下界来自同 target 上 RMSE 差的三角不等式；KL 记录没有 raw action/latent 输出，因此不能由此得到 student 间实际距离。参数步距离只在 ratio/channel 参数子空间计算，不是全模型参数比例；cosine 接近 1 也不证明两个目标具有相同损失变化。没有据此将差异归因为 STE、D 或任何单一机制。

## 主任务解读与下一步

本轮CPU作业25882534.pbs101已核实F/Exit0、artifact exit0、完成marker与全部覆盖门，耗时00:00:35。使用两套候选各480条已有C80记录和80条共同baseline，无新增model/teacher forward、backward、模型训练更新，也没有读取held-out数据或运行GPU任务。

共同MSE80起点的12个候选中，两个公式对每个候选的上升/下降方向均一致：2个候选同时下降，另外10个同时上升。因而这些反例不能只解释成MSE与KL评价口径不同。特别是saved Adam原步长：MSE梯度候选的Jm/Jk分别变差8.1265%/7.7194%，KL梯度候选却改善2.8893%/2.1120%；两候选投影更新向量的距离仅为MSE更新范数的0.58925%。这里的百分比是相对于本次更新向量，不能误读为相对全模型参数或输出的比例。

这两个saved/full候选之间，motor输出RMSE距离的逐case下界均值为0.00407638、最大0.03100144；latent相对输出距离下界均值为0.0361770。它们说明已有误差记录所支持的输出差异不可忽略，但只是下界，不是直接输出测距，也不构成全路径Lipschitz或某个quantizer的因果结论。

候选的最优排名也不随生成梯度的loss固定：仅在这12个共同起点候选中，最低Jk来自MSE梯度fresh Adam四分之一步（0.0026590234），而KL梯度自身六条件最低Jk为0.0026860427；最低Jm却来自KL梯度saved Adam原步长（0.0027679682），低于MSE梯度自身六条件最小值0.0027972171。这是事后完整C80交叉诊断，不是新增checkpoint选择策略，也不据此重选held-out表现最好的模型。

结合同参数training/evaluation前向parity为0、MSE与KL完整平均梯度高度同向、但真实有限步经常反向变化，目前最直接的问题是surrogate/Adam提议与真实有限步损失之间的不稳定对应。它可以解释为什么有限80步没有稳定超过已知MSE可行点，但还不能分离STE偏差、曲率、D/amax变化、BF16补偿及数值执行的各自贡献。原held-out上精修退化是另一个已经观察到的问题，两者不能混为一谈。

下一步仍以保留原MSE的per-channel D、固定D仅开放614 ratios的MSE versus dual-KL对照隔离更新子空间。允许的独立准备工作是把复用缓存梯度、真实完整C80选步、原held-out80评估及<2h资源限制落实为可审阅代码；按用户最后明确指令，尚不提交或启动新训练。现有结果没有证明信息论loss路线的联合泛化目标已经达到。
