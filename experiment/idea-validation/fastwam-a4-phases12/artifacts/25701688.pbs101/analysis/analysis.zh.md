# Fast-WAM A4 Phases 1/2 机制诊断汇总

状态：`complete`。本结果覆盖 22 cases、44 sampler contexts、
440 full queries（Phase 1 176；factorial 352，
其中端点复用 88、新增 mixed queries 264）。
episodes = 0，query 后环境步数 = 0。此诊断不提供成功率或 formal latency 结论。

## 同 case BF16 action 对照

每个 query 仅与相同 case 和 sampler seed 的 BF16 action 比较。Primary 为前 10 步 × 6 motor 维 RMSE；
first10 gripper RMSE 单独报告，secondary global RMSE 覆盖 32×7。

| Domain | Dimension | Query | Contexts | Primary motor RMSE | First10 gripper RMSE | Secondary global 32×7 RMSE |
|---|---|---:|---:|---:|---:|---:|
| original | unperturbed | all_a4 | 20 | 0.347171 | 0.31493 | 0.370231 |
| original | unperturbed | all_a8 | 20 | 0.0203364 | 0.0059656 | 0.0263388 |
| original | unperturbed | independent_reference | 20 | 0.348263 | 0.347941 | 0.375704 |
| original | unperturbed | mixed_v4a4p8 | 20 | 0.348467 | 0.28091 | 0.369453 |
| original | unperturbed | mixed_v4a8p4 | 20 | 0.32391 | 0.0470992 | 0.30383 |
| original | unperturbed | mixed_v4a8p8 | 20 | 0.325478 | 0.0473573 | 0.303732 |
| original | unperturbed | mixed_v8a4p4 | 20 | 0.113593 | 0.192018 | 0.13028 |
| original | unperturbed | mixed_v8a4p8 | 20 | 0.112994 | 0.184642 | 0.128529 |
| original | unperturbed | mixed_v8a8p4 | 20 | 0.0209738 | 0.0064303 | 0.0287835 |
| plus | all_dimensions | all_a4 | 24 | 0.345348 | 0.208075 | 0.388895 |
| plus | all_dimensions | all_a8 | 24 | 0.0285153 | 0.0066386 | 0.0367693 |
| plus | all_dimensions | independent_reference | 24 | 0.346502 | 0.211838 | 0.384807 |
| plus | all_dimensions | mixed_v4a4p8 | 24 | 0.34513 | 0.188191 | 0.382638 |
| plus | all_dimensions | mixed_v4a8p4 | 24 | 0.313006 | 0.0465922 | 0.325472 |
| plus | all_dimensions | mixed_v4a8p8 | 24 | 0.312332 | 0.0458046 | 0.326114 |
| plus | all_dimensions | mixed_v8a4p4 | 24 | 0.129575 | 0.207571 | 0.139925 |
| plus | all_dimensions | mixed_v8a4p8 | 24 | 0.128543 | 0.200285 | 0.137097 |
| plus | all_dimensions | mixed_v8a8p4 | 24 | 0.028368 | 0.00651186 | 0.0366038 |
| plus | camera_viewpoints | all_a4 | 4 | 0.332788 | 0.148316 | 0.3155 |
| plus | camera_viewpoints | all_a8 | 4 | 0.0453701 | 0.00437897 | 0.0695938 |
| plus | camera_viewpoints | independent_reference | 4 | 0.333361 | 0.259997 | 0.324014 |
| plus | camera_viewpoints | mixed_v4a4p8 | 4 | 0.326249 | 0.14815 | 0.300773 |
| plus | camera_viewpoints | mixed_v4a8p4 | 4 | 0.303228 | 0.0450193 | 0.263205 |
| plus | camera_viewpoints | mixed_v4a8p8 | 4 | 0.301526 | 0.0457224 | 0.262363 |
| plus | camera_viewpoints | mixed_v8a4p4 | 4 | 0.128381 | 0.153657 | 0.148947 |
| plus | camera_viewpoints | mixed_v8a4p8 | 4 | 0.127563 | 0.162076 | 0.14453 |
| plus | camera_viewpoints | mixed_v8a8p4 | 4 | 0.0455906 | 0.00521208 | 0.0701447 |
| plus | light_conditions | all_a4 | 4 | 0.392563 | 0.12716 | 0.396729 |
| plus | light_conditions | all_a8 | 4 | 0.0200908 | 0.00849411 | 0.0227614 |
| plus | light_conditions | independent_reference | 4 | 0.389005 | 0.240793 | 0.395539 |
| plus | light_conditions | mixed_v4a4p8 | 4 | 0.393419 | 0.214518 | 0.399724 |
| plus | light_conditions | mixed_v4a8p4 | 4 | 0.359887 | 0.0470078 | 0.358594 |
| plus | light_conditions | mixed_v4a8p8 | 4 | 0.359949 | 0.048748 | 0.327131 |
| plus | light_conditions | mixed_v8a4p4 | 4 | 0.137815 | 0.128351 | 0.128823 |
| plus | light_conditions | mixed_v8a4p8 | 4 | 0.134459 | 0.12538 | 0.125716 |
| plus | light_conditions | mixed_v8a8p4 | 4 | 0.0192728 | 0.00795725 | 0.0229649 |
| plus | background_textures | all_a4 | 4 | 0.318816 | 0.433863 | 0.435154 |
| plus | background_textures | all_a8 | 4 | 0.0339368 | 0.00497241 | 0.0327878 |
| plus | background_textures | independent_reference | 4 | 0.318676 | 0.25062 | 0.407755 |
| plus | background_textures | mixed_v4a4p8 | 4 | 0.318782 | 0.247944 | 0.393278 |
| plus | background_textures | mixed_v4a8p4 | 4 | 0.280551 | 0.0463372 | 0.323963 |
| plus | background_textures | mixed_v4a8p8 | 4 | 0.279388 | 0.0434808 | 0.370699 |
| plus | background_textures | mixed_v8a4p4 | 4 | 0.13455 | 0.247931 | 0.134838 |
| plus | background_textures | mixed_v8a4p8 | 4 | 0.133416 | 0.213409 | 0.131972 |
| plus | background_textures | mixed_v8a8p4 | 4 | 0.0337365 | 0.00487401 | 0.031018 |
| plus | objects_layout | all_a4 | 4 | 0.317476 | 0.15427 | 0.385639 |
| plus | objects_layout | all_a8 | 4 | 0.0300037 | 0.00791834 | 0.0532452 |
| plus | objects_layout | independent_reference | 4 | 0.325517 | 0.139434 | 0.389853 |
| plus | objects_layout | mixed_v4a4p8 | 4 | 0.322466 | 0.138538 | 0.386727 |
| plus | objects_layout | mixed_v4a8p4 | 4 | 0.273837 | 0.047582 | 0.306749 |
| plus | objects_layout | mixed_v4a8p8 | 4 | 0.273471 | 0.0462085 | 0.308077 |
| plus | objects_layout | mixed_v8a4p4 | 4 | 0.140018 | 0.228153 | 0.140091 |
| plus | objects_layout | mixed_v8a4p8 | 4 | 0.140731 | 0.196965 | 0.135932 |
| plus | objects_layout | mixed_v8a8p4 | 4 | 0.0278799 | 0.0084759 | 0.0510536 |
| plus | robot_initial_states | all_a4 | 4 | 0.387588 | 0.124308 | 0.421392 |
| plus | robot_initial_states | all_a8 | 4 | 0.0211944 | 0.00779606 | 0.0215417 |
| plus | robot_initial_states | independent_reference | 4 | 0.391025 | 0.124976 | 0.415777 |
| plus | robot_initial_states | mixed_v4a4p8 | 4 | 0.384165 | 0.12369 | 0.419119 |
| plus | robot_initial_states | mixed_v4a8p4 | 4 | 0.352499 | 0.0467093 | 0.377697 |
| plus | robot_initial_states | mixed_v4a8p8 | 4 | 0.355526 | 0.0453814 | 0.369614 |
| plus | robot_initial_states | mixed_v8a4p4 | 4 | 0.114923 | 0.126113 | 0.117883 |
| plus | robot_initial_states | mixed_v8a4p8 | 4 | 0.120639 | 0.132501 | 0.117333 |
| plus | robot_initial_states | mixed_v8a8p4 | 4 | 0.0230578 | 0.0070686 | 0.0221535 |
| plus | language_instructions | all_a4 | 4 | 0.322858 | 0.26053 | 0.378954 |
| plus | language_instructions | all_a8 | 4 | 0.020496 | 0.0062717 | 0.0206861 |
| plus | language_instructions | independent_reference | 4 | 0.321427 | 0.255206 | 0.375902 |
| plus | language_instructions | mixed_v4a4p8 | 4 | 0.325696 | 0.256304 | 0.396207 |
| plus | language_instructions | mixed_v4a8p4 | 4 | 0.308031 | 0.0468978 | 0.322622 |
| plus | language_instructions | mixed_v4a8p8 | 4 | 0.304131 | 0.0452865 | 0.318801 |
| plus | language_instructions | mixed_v8a4p4 | 4 | 0.121764 | 0.361223 | 0.168966 |
| plus | language_instructions | mixed_v8a4p8 | 4 | 0.114451 | 0.371378 | 0.167098 |
| plus | language_instructions | mixed_v8a8p4 | 4 | 0.0206702 | 0.00548329 | 0.0222884 |

## Native all-A4 对 independent reference 的直接 action 差

逐 case/seed 直接比较两组 raw action；不通过它们各自相对 BF16 的 RMSE 推算。

44 contexts：first10 motor RMSE mean 0.04551，first10 gripper RMSE mean 0.123051，secondary global 32×7 RMSE mean 0.0852427。

## A4 factorial 描述效应

Primary response 是相对同 context BF16 的前 10 步 × 6 motor 维绝对 RMSE；对每个 case/seed 完整配对 2³ cells。仅作描述统计，不做 p-values。

| Factor | Mean at A4 | Mean at A8 | A4 − A8 | Paired contexts |
|---|---:|---:|---:|---:|
| video | 0.332273 | 0.0733976 | 0.258876 | 44 |
| action | 0.234152 | 0.171519 | 0.0626337 | 44 |
| proprio | 0.202864 | 0.202807 | 5.73941e-05 | 44 |

Interactions (paired within context):

- `video_x_action`: mean contrast -0.0687139; median -0.0640836; range [-0.126371, -0.017989].
- `video_x_proprio`: mean contrast -0.000930154; median -0.00070443; range [-0.0205541, 0.0116322].
- `action_x_proprio`: mean contrast 0.000250481; median 2.06098e-05; range [-0.0101563, 0.0124444].
- `video_x_action_x_proprio`: mean contrast -0.000751478; median -0.00312462; range [-0.0179712, 0.0205548].

## Per-layer trace summaries

仅统计 44 个 native all-A4 query 实际调用的 Linear；不要求未调用模块有 coverage。指标在同一层调用级别累计。

| Scope | Stage | Step | Calls | A4 scale mean | A8 scale mean | A4 zero fraction | A4 input rel. RMSE | A4/A8 local RMSE | A4/A8 local NRMSE | Native/reference NRMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| action | action | 0 | 13508 | 0.637994 | 0.035165 | 0.559834 | 0.314945 | 0.222371 | 0.215517 | 2.52905e-05 |
| action | action | 1 | 13508 | 0.62164 | 0.0342636 | 0.565703 | 0.314305 | 0.214955 | 0.210073 | 2.54549e-05 |
| action | action | 2 | 13508 | 0.673614 | 0.0371283 | 0.578007 | 0.32901 | 0.230085 | 0.216302 | 2.62643e-05 |
| action | action | 3 | 13508 | 0.701724 | 0.0386777 | 0.586238 | 0.339045 | 0.240127 | 0.21997 | 2.65507e-05 |
| action | action | 4 | 13508 | 0.723664 | 0.039887 | 0.594168 | 0.34598 | 0.247824 | 0.220114 | 2.65685e-05 |
| action | action | 5 | 13508 | 0.705806 | 0.0389027 | 0.5928 | 0.343365 | 0.244024 | 0.215625 | 2.69903e-05 |
| action | action | 6 | 13508 | 0.697706 | 0.0384563 | 0.595184 | 0.344909 | 0.243739 | 0.214598 | 2.61264e-05 |
| action | action | 7 | 13508 | 0.691215 | 0.0380984 | 0.596977 | 0.3426 | 0.242427 | 0.215189 | 2.63372e-05 |
| action | action | 8 | 13508 | 0.669436 | 0.036898 | 0.594385 | 0.334747 | 0.234516 | 0.212231 | 2.62283e-05 |
| action | action | 9 | 13508 | 0.614457 | 0.0338677 | 0.587654 | 0.314931 | 0.213827 | 0.205775 | 2.61285e-05 |
| proprio | conditioning | -1 | 44 | 0.126268 | 0.00695967 | 0.136364 | 0.0650472 | 0.019922 | 0.0569622 | 8.78508e-06 |
| video | video | 0 | 13464 | 0.89275 | 0.0492067 | 0.693709 | 0.386922 | 0.377516 | 0.278704 | 2.95986e-05 |
| video | video | 1 | 13464 | 0.887841 | 0.0489361 | 0.684657 | 0.385544 | 0.378072 | 0.276579 | 2.91297e-05 |
| video | video | 2 | 13464 | 0.881272 | 0.048574 | 0.674497 | 0.380282 | 0.376337 | 0.273136 | 2.86663e-05 |
| video | video | 3 | 13464 | 0.867096 | 0.0477927 | 0.662902 | 0.373982 | 0.372372 | 0.26923 | 2.80802e-05 |
| video | video | 4 | 13464 | 0.864838 | 0.0476682 | 0.659401 | 0.372061 | 0.372952 | 0.268209 | 2.80716e-05 |
| video | video | 5 | 13464 | 0.845747 | 0.046616 | 0.64544 | 0.365915 | 0.368148 | 0.264977 | 2.78423e-05 |
| video | video | 6 | 13464 | 0.86574 | 0.0477179 | 0.657512 | 0.372427 | 0.377787 | 0.269127 | 2.77254e-05 |
| video | video | 7 | 13464 | 0.937071 | 0.0516496 | 0.696513 | 0.400409 | 0.416209 | 0.288127 | 3.0406e-05 |
| video | video | 8 | 13464 | 0.915602 | 0.0504663 | 0.687852 | 0.395121 | 0.406258 | 0.284499 | 2.86871e-05 |
| video | video | 9 | 13464 | 0.858746 | 0.0473325 | 0.653893 | 0.373932 | 0.384362 | 0.272897 | 2.85661e-05 |
| video | video_conditioning_prefill | -1 | 13420 | 0.954758 | 0.0526244 | 0.715165 | 0.415432 | 0.44851 | 0.300989 | 2.88129e-05 |

Worst observed module/stage/step local normalized errors:

- `video_expert.time_embedding.0` (video, video step 7): mean NRMSE 0.000505673, max 0.000505673, calls 44.
- `action_expert.time_embedding.0` (action, action step 5): mean NRMSE 0.000216203, max 0.000216203, calls 44.
- `video_expert.time_embedding.0` (video, video step 0): mean NRMSE 0.000147478, max 0.000147478, calls 44.
- `video_expert.time_embedding.0` (video, video_conditioning_prefill step -1): mean NRMSE 0.000132394, max 0.000132394, calls 44.
- `action_expert.time_embedding.0` (action, action step 9): mean NRMSE 0.000130427, max 0.000130427, calls 44.
- `video_expert.time_embedding.0` (video, video step 2): mean NRMSE 0.00011854, max 0.00011854, calls 44.
- `video_expert.time_embedding.0` (video, video step 4): mean NRMSE 0.000111076, max 0.000111076, calls 44.
- `video_expert.time_embedding.0` (video, video step 1): mean NRMSE 0.000107631, max 0.000107631, calls 44.
- `video_expert.blocks.6.cross_attn.q` (video, video step 3): mean NRMSE 0.000102736, max 0.000153764, calls 44.
- `video_expert.time_embedding.0` (video, video step 9): mean NRMSE 9.97796e-05, max 9.97796e-05, calls 44.

所有 case 的 PBS exit status、artifact exit code、PIPELINE_COMPLETE marker、case summary、query 计数、
counter 边界、actions 形状/有限性与 all-A4 实际 trace coverage 均通过聚合门禁。
