# Architecture-independent error bookkeeping (not a proposed new method)

This is a standard first-order recurrence, supplied to keep the proposal's mechanism claims precise. It is not evidence of novelty or empirical failure. No model was executed.

For model m, fix an observation/context c, initial noise, solver schedule and all stochastic innovations. Let x_k contain every evolving state that the action query needs: just action latents for Fast-WAM Base with a fixed video cache, or coupled video/action states for a joint/shared model. Write one actual solver update as F_{m,k}; noise/velocity/x0 parameterization is absorbed into this update, not assumed identical across models. Extract/denormalize the action chunk with the model's native map D_m and the controller's executed prefix selector E_m. Compare each model to its own BF16 reference.

Exact identity, defining r_k(x) = F_Q,k(x,c) - F_FP,k(x,c):

    delta_(k+1) = F_FP,k(x_Q,k,c) - F_FP,k(x_FP,k,c) + r_k(x_Q,k).

If the FP map is differentiable on the relevant neighborhood:

    delta_(k+1) = J_k delta_k + r_k(x_Q,k) + O(||delta_k||^2),
    J_k = d F_FP,k / d x evaluated at x_FP,k.

For equal initial states, its linearized endpoint error is a sum of transported residuals, not just a sum of residual norms. Its squared norm generally includes cross-step terms. This is algebra; it does not establish that those terms matter in any WAM or justify a particular estimator. SteerQuant already uses downstream-action Jacobians for PTQ sensitivity, so merely restating Jacobian transport is not novelty.

Important restrictions:

- Replacing r_k(x_Q,k) by r_k(x_FP,k) is an additional approximation. The teacher-state residual does not automatically describe quantized-student visited states; quantizer clipping/bin changes can make this approximation poor.
- In coupled models, J_k includes video/action off-diagonal blocks. Dropping them requires a proved dependency separation or an explicitly tested approximation; freezing video parameters alone does not make them zero.
- E_m D_m may include quantile/min-max normalization, clipping and discrete gripper decisions. A continuous latent endpoint norm is not identical to the actuator command deviation, much less task success.
- Jacobian propagation/endpoint fitting provides fixed-input evidence. Environmental state drift under closed-loop execution is a separate process, with a separate evaluation requirement.
- A deployable action query uses the quantized model alone. A teacher may be used during training or diagnosis, but teacher-paid corrections at evaluation do not prove standalone QAT performance.

Primary interface evidence is in MODEL_INTERFACES.md. Direct downstream-action sensitivity prior: https://arxiv.org/abs/2609.39056 . The proposal must independently identify and falsify any stronger contribution beyond this bookkeeping.
