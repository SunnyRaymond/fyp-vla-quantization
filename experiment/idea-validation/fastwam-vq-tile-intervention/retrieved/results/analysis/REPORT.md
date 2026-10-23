# Tile intervention results

Correlations are descriptive within4predefined modules; contexts/tiles are repeated measurements, not independent population replicates. Signed scores predict MSE/NMSE change; Fisher only ranked on its corresponding native/joint objective. Loss nonadditivity is decomposed because quadratic cross terms exist even for additive outputs.

| Target / module | N | Spearman | Sign correct / informative | Mean prediction | Mean true delta |
|---|---:|---:|---:|---:|---:|
| native_mse/video_expert.blocks.0.ffn.0 | 56 | -0.026999316473000683 | 24/56 | -7.453587195e-08 | -0.0001688057801 |
| native_mse/video_expert.blocks.29.ffn.2 | 56 | -0.3307587149692413 | 21/56 | 3.979725512e-07 | -0.0001309702258 |
| native_mse/action_expert.blocks.0.ffn.0 | 56 | 0.0758714969241285 | 30/56 | 2.347941821e-07 | -7.074760139e-05 |
| native_mse/action_expert.blocks.29.ffn.2 | 56 | -0.07074504442925496 | 26/56 | 2.394595568e-07 | -2.374229856e-05 |
| future_nmse/video_expert.blocks.0.ffn.0 | 56 | 0.09295967190704033 | 30/56 | -5.85732319e-07 | 0.0004484319026 |
| future_nmse/video_expert.blocks.29.ffn.2 | 56 | -0.03595352016404648 | 25/56 | -3.900876958e-05 | -0.001931829337 |
| future_nmse/action_expert.blocks.0.ffn.0 | 56 | None | 0/0 | 0 | 0 |
| future_nmse/action_expert.blocks.29.ffn.2 | 56 | None | 0/0 | 0 | 0 |
| reader_mse/video_expert.blocks.0.ffn.0 | 56 | -0.03773069036226931 | 26/56 | -3.775336547e-07 | -0.0002269704889 |
| reader_mse/video_expert.blocks.29.ffn.2 | 56 | -0.13779904306220095 | 30/56 | -2.27985358e-06 | -0.0001595226844 |
| reader_mse/action_expert.blocks.0.ffn.0 | 56 | None | 0/0 | 0 | 0 |
| reader_mse/action_expert.blocks.29.ffn.2 | 56 | None | 0/0 | 0 | 0 |

| Target / simultaneous size | N | Median abs interaction / sum abs singles | Median output nonadditivity ratio | Sign flips |
|---|---:|---:|---:|---:|
| native_mse/size2 | 144 | 0.629075 | 0.701608 | 43/144 |
| native_mse/size4 | 40 | 0.608816 | 0.668599 | 14/40 |
| native_mse/size8 | 8 | 0.435588 | 0.658928 | 4/8 |
| future_nmse/size2 | 144 | 0.00309856 | 0.210272 | 15/80 |
| future_nmse/size4 | 40 | 0.104526 | 0.574272 | 5/24 |
| future_nmse/size8 | 8 | 0.652267 | 0.586586 | 2/8 |
| reader_mse/size2 | 144 | 0.0242611 | 0.0958156 | 18/80 |
| reader_mse/size4 | 40 | 0.198684 | 0.396871 | 6/24 |
| reader_mse/size8 | 8 | 0.604921 | 0.671467 | 2/8 |

Action-module future/reader negative controls: {"n": 112, "max_future_abs": 0.0, "max_reader_abs": 0.0}

Baseline repeated noise: [{"case_id": 0, "baseline": {"native_mse": 0.0016438893973827362, "reader_mse": 0.00020816077788670856, "future_nmse": 0.04401194861582232}, "repeat": {"native_mse": 0.0016438893973827362, "reader_mse": 0.00020816077788670856, "future_nmse": 0.04401194861582232}, "noise_floor": {"native_mse": 1e-12, "future_nmse": 1e-12, "reader_mse": 1e-12}, "repeat_output_max_abs": {"native_mse": 0.0, "future_nmse": 0.0, "reader_mse": 0.0}}, {"case_id": 2, "baseline": {"native_mse": 0.0019143817946314812, "reader_mse": 0.0006736976404984792, "future_nmse": 0.049670707654229665}, "repeat": {"native_mse": 0.0019143817946314812, "reader_mse": 0.0006736976404984792, "future_nmse": 0.049670707654229665}, "noise_floor": {"native_mse": 1e-12, "future_nmse": 1e-12, "reader_mse": 1e-12}, "repeat_output_max_abs": {"native_mse": 0.0, "future_nmse": 0.0, "reader_mse": 0.0}}, {"case_id": 4, "baseline": {"native_mse": 0.0018453312416871388, "reader_mse": 0.0006947162250677745, "future_nmse": 0.10220324840075777}, "repeat": {"native_mse": 0.0018453312416871388, "reader_mse": 0.0006947162250677745, "future_nmse": 0.10220324840075777}, "noise_floor": {"native_mse": 1e-12, "future_nmse": 1e-12, "reader_mse": 1e-12}, "repeat_output_max_abs": {"native_mse": 0.0, "future_nmse": 0.0, "reader_mse": 0.0}}, {"case_id": 6, "baseline": {"native_mse": 0.001532157650217414, "reader_mse": 0.0002672825163851182, "future_nmse": 0.047150757289897205}, "repeat": {"native_mse": 0.001532157650217414, "reader_mse": 0.0002672825163851182, "future_nmse": 0.047150757289897205}, "noise_floor": {"native_mse": 1e-12, "future_nmse": 1e-12, "reader_mse": 1e-12}, "repeat_output_max_abs": {"native_mse": 0.0, "future_nmse": 0.0, "reader_mse": 0.0}}, {"case_id": 10, "baseline": {"native_mse": 0.002957378828432411, "reader_mse": 0.0017441441033345959, "future_nmse": 0.07776454609774498}, "repeat": {"native_mse": 0.002957378828432411, "reader_mse": 0.0017441441033345959, "future_nmse": 0.07776454609774498}, "noise_floor": {"native_mse": 1e-12, "future_nmse": 1e-12, "reader_mse": 1e-12}, "repeat_output_max_abs": {"native_mse": 0.0, "future_nmse": 0.0, "reader_mse": 0.0}}, {"case_id": 12, "baseline": {"native_mse": 0.002930573622385661, "reader_mse": 0.002843772371610006, "future_nmse": 0.20937072892588166}, "repeat": {"native_mse": 0.002930573622385661, "reader_mse": 0.002843772371610006, "future_nmse": 0.20937072892588166}, "noise_floor": {"native_mse": 1e-12, "future_nmse": 1e-12, "reader_mse": 1e-12}, "repeat_output_max_abs": {"native_mse": 0.0, "future_nmse": 0.0, "reader_mse": 0.0}}, {"case_id": 14, "baseline": {"native_mse": 0.0016125102837880453, "reader_mse": 0.0006814196705818177, "future_nmse": 0.04440261051631411}, "repeat": {"native_mse": 0.0016125102837880453, "reader_mse": 0.0006814196705818177, "future_nmse": 0.04440261051631411}, "noise_floor": {"native_mse": 1e-12, "future_nmse": 1e-12, "reader_mse": 1e-12}, "repeat_output_max_abs": {"native_mse": 0.0, "future_nmse": 0.0, "reader_mse": 0.0}}, {"case_id": 16, "baseline": {"native_mse": 0.0015651213626066843, "reader_mse": 0.0004939449951052666, "future_nmse": 0.0814960772682438}, "repeat": {"native_mse": 0.0015651213626066843, "reader_mse": 0.0004939449951052666, "future_nmse": 0.0814960772682438}, "noise_floor": {"native_mse": 1e-12, "future_nmse": 1e-12, "reader_mse": 1e-12}, "repeat_output_max_abs": {"native_mse": 0.0, "future_nmse": 0.0, "reader_mse": 0.0}}]
