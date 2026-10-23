# C8 frozen-reader reader_kl：完整配对结果

原 C8 A+Z、rho1.5、8 updates；原 baseline 88 queries 和 teacher 全部复用。
只训练原 codebooks/log row gains，无 activation clip/per-channel；未运行闭环。

主评测为实际量化 student，非 frozen-reader 自身得分。

## calibration

| Metric | Baseline | Candidate | Delta | Paired median | W/L/T |
|---|---:|---:|---:|---:|---:|
| first10motor_rmse | 0.0421332419 | 0.0387150622 | -0.00341817969 | -0.00139309652 | 6/2/0 |
| full32motor_rmse | 0.0443880437 | 0.0412583584 | -0.0031296853 | -0.00171203166 | 5/3/0 |
| late22motor_rmse | 0.0446109003 | 0.0421239366 | -0.00248696376 | -0.00119904429 | 5/3/0 |
| first10gripper_rmse | 0.0200497055 | 0.0181073576 | -0.00194234797 | -0.00151934661 | 5/3/0 |
| future_latent_nmse | 0.0544809815 | 0.0454062065 | -0.00907477504 | -0.0048930943 | 6/2/0 |

## heldout

| Metric | Baseline | Candidate | Delta | Paired median | W/L/T |
|---|---:|---:|---:|---:|---:|
| first10motor_rmse | 0.0442967744 | 0.0451499174 | +0.000853142957 | +0.00127490563 | 12/28/0 |
| full32motor_rmse | 0.0475009795 | 0.0495732677 | +0.00207228819 | +0.000774967484 | 17/23/0 |
| late22motor_rmse | 0.048528504 | 0.0508427202 | +0.00231421622 | +0.00123874936 | 15/25/0 |
| first10gripper_rmse | 0.0570540305 | 0.0408736746 | -0.016180356 | -0.000512814848 | 25/15/0 |
| future_latent_nmse | 0.060877371 | 0.0573564747 | -0.0035208963 | -0.00282795262 | 22/18/0 |

冻结 gate：motor 均值下降、case 配对中位数下降、至少21/40改善，且 latent NMSE 均值不退步。
Gate passed: False；决定：stop_this_recipe_no_parameter_sweep

beta=0.005622540816813253；仅用 C8 初始输入标定一次。
不使用 heldout 选择 checkpoint、系数或训练顺序；两个 sampler seeds 不是独立训练重复。
这是固定输入保真验证；不代表闭环、native INT4 或独立泛化收益。
