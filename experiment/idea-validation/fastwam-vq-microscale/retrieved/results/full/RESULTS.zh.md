# Microscaling + additive VQ：配对数值验证

Microscaled VQ test motor RMSE：0.05240395。
Row-scale VQ 对照：0.05705206。
相对改善：8.147%；改善 cases：14/22。
工程筛选门槛通过：True。

全部614个Linear；selection 4 cases/8 contexts，test 22 cases/44 contexts。
BF16 additive books，d4/M2/K256；A4、Smooth0.5、H128固定。B32 exponent scaling，row gain及exponents固定。
已有test结果已被看过，本轮为配对机制诊断。没有native kernel、INT4 books或closed-loop成功率结论。
