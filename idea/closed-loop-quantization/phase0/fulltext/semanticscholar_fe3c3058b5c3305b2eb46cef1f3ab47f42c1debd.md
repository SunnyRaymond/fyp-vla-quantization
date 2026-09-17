# Joint-Torque Trajectory Tracking Control of the Manipulator Based on Operator Theory and SVR Feedforward Residual Compensation

paper_id: semanticscholar:fe3c3058b5c3305b2eb46cef1f3ab47f42c1debd
tier: T2
source_used: failed
warning: fetch failed across all paths; intro filled with abstract; method empty

## Intro

End effector tracking errors in complex trajectory tracking of the UR5e manipulator are mainly caused by inter joint dynamic coupling and model approximation errors. This paper proposes a joint torque control method that combines operator theory with support vector regression (SVR) based residual compensation. First, a gravity compensated multi joint viscoelastic feedback controller is constructed. Then, SVR is used to learn the nonlinear mapping from the desired joint states to the full gravity decoupled feedforward torque. Finally, under a nominal right coprime factorization framework, the full feedforward torque is decomposed into a diagonal nominal feedforward term and an SVR estimated residual term. The residual term is introduced at the input of the nominal error system, rather than being treated as an independent feedforward module. In simulation, the UR5e end effector tracks a multi period composite cosine trajectory in the vertical $y z$ plane. Among the compared controllers, the proposed scheme, S3, achieves the lowest task space tracking error. The tool center point (TCP) RMSE decreases from 6.393 mm under viscoelastic feedback alone, S1, to 1.256 mm under full SVR feedforward, S2, and further to $\mathbf{1. 1 4 2 ~ m m}$ under S3. The maximum TCP error is reduced to 4.989 mm. Residual analysis shows that SVR reduces the RMSE of the torque residual not compensated by the nominal model from $\mathbf{1. 0 2 9 6 ~ N} \cdot \mathbf{m}$ to $\mathbf{0. 4 2 8 2 ~ N} \cdot \mathbf{m}$. These results indicate that operator theory based structured residual compensation can further improve UR5e task space tracking accuracy beyond direct SVR feedforward compensation.

## Method


