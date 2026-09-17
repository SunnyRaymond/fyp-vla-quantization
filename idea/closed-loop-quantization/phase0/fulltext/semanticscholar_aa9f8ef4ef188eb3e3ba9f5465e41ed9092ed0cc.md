# Model-Assisted Variable-Bandwidth ESO Based Feedback Linearization for Trajectory Tracking of Flexible-Joint Robot Arms

paper_id: semanticscholar:aa9f8ef4ef188eb3e3ba9f5465e41ed9092ed0cc
tier: T2
source_used: failed
warning: fetch failed across all paths; intro filled with abstract; method empty

## Intro

Flexible-joint manipulators are difficult to track precisely because of elastic dynamics, strong coupling, and time-varying disturbances.Although feedback linearization (FL) cancels nominal nonlinearities, its performance degrades under model uncertainty.Fixed-bandwidth extended state observers (ESOs) improve robustness, but high observer gains may induce peaking, actuator jitter, and saturation. This letter proposes a model-assisted variable-bandwidth ESO (MA-VBESO) for FL-based trajectory tracking of strongly coupled multi-input multi-output (MIMO) flexible-joint manipulators. Although the dominant link-side disturbances are mismatched in the original coordinates, their effects, together with motor-side disturbances and model uncertainty, are transformed into equivalent lumped disturbances in the highest-order output-channel dynamics under the Byrnes–Isidori normal form. By embedding the nominal model, the observer estimates only residual dynamics and exogenous perturbations, while a filtered error-driven bandwidth law improves transients and mitigates peaking. Input-to-state stability of the estimation and tracking errors is established under bounded disturbances. Experiments on the Quanser 2-DOF platform show that, under load, FL+MA-VBESO reduces the RMSE relative to FL by 59.23% and 69.72% for Joints 1 and 2, and further reduces the RMSE relative to fixed-bandwidth ESO by 5.31% and 3.54%. The motor current-rate RMS decreases from 25.874 to 9.064 A/s for Joint 1 and from 48.659 to 11.777 A/s for Joint 2.

## Method


