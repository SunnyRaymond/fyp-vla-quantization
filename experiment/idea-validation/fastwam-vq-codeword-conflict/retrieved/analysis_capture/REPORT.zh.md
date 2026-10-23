# Codeword gradient conflict: sampled screen

All8 calibration contexts; all3 checkpoints; all614 Linears.
Within-codeword metrics use fixed stratified samples. Exact shared-codeword gradients use all members/calls.

| Checkpoint | Latent cancellation | Random grouping | Mean pair cosine | Latent shared-step opposed member fraction | Action/latent conflicting codeword fraction |
|---|---:|---:|---:|---:|---:|
| parent | 0.774074 | 0.774109 | 0.000083 | 0.495992 | 0.244012 |
| action_only | 0.774829 | 0.774572 | 0.000015 | 0.495857 | 0.364685 |
| action_latent | 0.774358 | 0.774221 | 0.000020 | 0.495937 | 0.286600 |

Sampling cancellation alone is not evidence of harmful assignment. Full-member and finite-forward checks remain required.

Selected modules:
- video_expert.time_embedding.0, book0, codewords=[32, 195, 49, 150, 127, 223, 197, 249]
- video_expert.text_embedding.0, book0, codewords=[182, 93, 24, 203, 66, 148, 155, 194]
- video_expert.blocks.24.ffn.0, book0, codewords=[59, 208, 182, 187, 160, 184, 21, 107]
- video_expert.blocks.17.ffn.0, book0, codewords=[123, 163, 220, 229, 177, 190, 3, 175]
