# Full-member codeword checks

All8 action-only calibration contexts; selected4 modules and32 book0 codewords.
Gradients are summed per vector over sampler calls before grouping. Book1 is fixed.
Full-member sums match exact autograd codeword gradients. Zero-active rows are reported explicitly.

| Module | Codeword | Members | Full cancellation | Sample cancellation | Full pair cosine | Shared-step opposed members | Retained descent fraction | Zero-scale gradient energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| video_expert.time_embedding.0 | 32 | 1287 | 0.924074 | 0.293161 | -0.000033 | 0.496892 | 0.001670 | 0.000000 |
| video_expert.time_embedding.0 | 195 | 1227 | 0.933888 | 0.520855 | 0.000248 | 0.489201 | 0.001762 | 0.000000 |
| video_expert.time_embedding.0 | 49 | 1030 | 0.933808 | 0.532457 | 0.000248 | 0.487257 | 0.001597 | 0.000000 |
| video_expert.time_embedding.0 | 150 | 1061 | 0.933255 | 0.384945 | -0.000140 | 0.499529 | 0.001411 | 0.000000 |
| video_expert.time_embedding.0 | 127 | 894 | 0.878587 | 0.413495 | -0.000583 | 0.503775 | 0.002466 | 0.000000 |
| video_expert.time_embedding.0 | 223 | 480 | 0.869335 | 0.447154 | -0.000431 | 0.499740 | 0.004259 | 0.000000 |
| video_expert.time_embedding.0 | 197 | 1007 | 0.930194 | 0.629433 | -0.000054 | 0.486966 | 0.001399 | 0.000000 |
| video_expert.time_embedding.0 | 249 | 802 | 0.924453 | 0.577528 | -0.000302 | 0.483167 | 0.001573 | 0.000000 |
| video_expert.text_embedding.0 | 182 | 14914 | 0.987277 | 0.760131 | 0.000028 | 0.492784 | 0.000116 | 0.000000 |
| video_expert.text_embedding.0 | 93 | 9218 | 0.989347 | 0.714988 | -0.000011 | 0.493206 | 0.000082 | 0.000000 |
| video_expert.text_embedding.0 | 24 | 12118 | 0.988630 | 0.709379 | 0.000041 | 0.494677 | 0.000091 | 0.000000 |
| video_expert.text_embedding.0 | 203 | 23224 | 0.991745 | 0.757328 | -0.000007 | 0.495150 | 0.000049 | 0.000000 |
| video_expert.text_embedding.0 | 66 | 6588 | 0.981319 | 0.702881 | 0.000053 | 0.489944 | 0.000263 | 0.000000 |
| video_expert.text_embedding.0 | 148 | 16549 | 0.988911 | 0.697661 | 0.000036 | 0.493716 | 0.000093 | 0.000000 |
| video_expert.text_embedding.0 | 155 | 15389 | 0.990183 | 0.725863 | -0.000004 | 0.494590 | 0.000078 | 0.000000 |
| video_expert.text_embedding.0 | 194 | 15907 | 0.987741 | 0.815460 | 0.000011 | 0.494421 | 0.000110 | 0.000000 |
| video_expert.blocks.24.ffn.0 | 59 | 69376 | 0.994150 | 0.767708 | 0.000011 | 0.496409 | 0.000020 | 0.000000 |
| video_expert.blocks.24.ffn.0 | 208 | 70893 | 0.994112 | 0.757537 | 0.000005 | 0.496498 | 0.000022 | 0.000000 |
| video_expert.blocks.24.ffn.0 | 182 | 56257 | 0.995118 | 0.726583 | -0.000001 | 0.498482 | 0.000015 | 0.000000 |
| video_expert.blocks.24.ffn.0 | 187 | 46204 | 0.994482 | 0.742945 | 0.000004 | 0.496846 | 0.000020 | 0.000000 |
| video_expert.blocks.24.ffn.0 | 160 | 72952 | 0.995162 | 0.769212 | 0.000003 | 0.497310 | 0.000014 | 0.000000 |
| video_expert.blocks.24.ffn.0 | 184 | 67486 | 0.995057 | 0.758552 | -0.000002 | 0.497742 | 0.000015 | 0.000000 |
| video_expert.blocks.24.ffn.0 | 21 | 73436 | 0.994542 | 0.772818 | 0.000004 | 0.498066 | 0.000019 | 0.000000 |
| video_expert.blocks.24.ffn.0 | 107 | 42290 | 0.993036 | 0.731791 | -0.000001 | 0.497694 | 0.000032 | 0.000000 |
| video_expert.blocks.17.ffn.0 | 123 | 75529 | 0.995990 | 0.701811 | -0.000002 | 0.497762 | 0.000012 | 0.000000 |
| video_expert.blocks.17.ffn.0 | 163 | 48448 | 0.995646 | 0.711946 | -0.000001 | 0.498049 | 0.000015 | 0.000000 |
| video_expert.blocks.17.ffn.0 | 220 | 52802 | 0.995841 | 0.750759 | -0.000001 | 0.496743 | 0.000014 | 0.000000 |
| video_expert.blocks.17.ffn.0 | 229 | 65601 | 0.995720 | 0.836217 | 0.000002 | 0.497266 | 0.000014 | 0.000000 |
| video_expert.blocks.17.ffn.0 | 177 | 69444 | 0.996413 | 0.793089 | -0.000004 | 0.497721 | 0.000010 | 0.000000 |
| video_expert.blocks.17.ffn.0 | 190 | 79148 | 0.996506 | 0.749583 | -0.000006 | 0.498765 | 0.000009 | 0.000000 |
| video_expert.blocks.17.ffn.0 | 3 | 48697 | 0.994680 | 0.800596 | -0.000002 | 0.496779 | 0.000020 | 0.000000 |
| video_expert.blocks.17.ffn.0 | 175 | 58552 | 0.995447 | 0.732498 | 0.000003 | 0.497131 | 0.000016 | 0.000000 |

Pooled-calibration projection and actual BF16 perturbation norms are recorded in analysis.json.
Nonzero zero-scale gradient energy would mean untied updates also revive rows held fixed by zero scales; their benefit could not be attributed solely to codeword sharing.
Untied directions increase degrees of freedom; no fixed-bit assignment improvement is established by this check.
