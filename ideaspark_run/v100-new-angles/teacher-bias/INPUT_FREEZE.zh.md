# Teacher-bias 实际输入冻结

CPU64827，TC1N05，10s完成；未加载模型或执行推理。Manifest来自远端job artifact的小型原始JSON，SHA256：`6dae677a0e7763896d735495500aea78a9254e59b67fa47817bb9bb9bb1f5c14`。使用路径 `teacher_bias_ready2/manifest.json`，第一失败准备目录保留。

| valid-local | underlying trajectory | frames | model action blocks |
|---|---|---|---|
|124|1035|0,5,25|5×10|
|125|1534|0,5,25|5×10|
|126|1158|0,5,25|5×10|
|127|203|0,5,25|5×10|
|128|1837|0,5,25|5×10|
|129|1095|0,5,25|5×10|

所有visual为dataset已处理float `[3,3,224,224]`。实际raw state/proprio均为2D；learned proprio embedding为10D，故辅助raw target为 `[6,2,2]`，encoded target/prediction保留10D。之前静态审查的raw4假设错误，已在任何GPU输出之前按真实CPU manifest修正。主visual metric、样本、H1/H5、RTN recipe和4/6 gate不变。

Checkpoint与source/DINOv2身份通过原冻结hash/commit检查。当前config SHA256 `4ed3109219ed6820ce51fd19e4c97199552b3443ed58c502fddd6441784967e1`。实际CCDS env adapter确实为compact形式，SHA `c90f6a26d5f23c8dd1c448a0190fcc5d9550ddd4682456ac9e5f61748a37268c`；dino adapter SHA `597aac0e561abe5c223acd26d1c0928378ea15c4a3a8196008d3fff8e5fb659a`。

GPU64828按此freeze提交；runner LF SHA256 `6e475ff4c75dda269b6d916ac28f8d63879abe6aae6427da6955bb35d36ed559`，独立CPU verifier绑定同值。Protocol SHA仍为 `b553d6a011bfc990a55c3f41dc1319fc3990b7e6449b27c3cd4b0b60a35e0140`。
