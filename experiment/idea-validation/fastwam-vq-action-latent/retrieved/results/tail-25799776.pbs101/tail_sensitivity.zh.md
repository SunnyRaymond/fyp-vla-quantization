# Tail-case sensitivity（post hoc）

- Source analysis job: `25799647.pbs101`；experiment job: `25798439.pbs101`。
- Frozen primary cohort remains all 22 test cases / 44 contexts. Every omission below is a post hoc sensitivity view; it does not replace the 22-case result.
- Relative mean change = mean paired delta / mean baseline × 100%；negative values indicate lower error. No p-values or new model forwards.
- All action and future values come from the stored `case_rows` and their existing paired-delta fields in the one source `analysis.json`.

## Full cohort and omission sensitivities

| Comparison | Metric | View | Cases | Baseline mean | Chosen mean | Mean paired delta | Relative change (%) |
|---|---|---|---:|---:|---:|---:|---:|
| parent | First-10 motor RMSE | all_22 | 22 | 0.057255758 | 0.05295101 | -0.0043047488 | -7.5185 |
| parent | First-10 motor RMSE | omit_1015 | 21 | 0.057560386 | 0.053181668 | -0.0043787183 | -7.6072 |
| parent | First-10 motor RMSE | omit_1016 | 21 | 0.055297085 | 0.051261496 | -0.004035589 | -7.2980 |
| parent | First-10 motor RMSE | omit_1025 | 21 | 0.05506358 | 0.052487689 | -0.0025758909 | -4.6780 |
| parent | First-10 motor RMSE | omit_1016_1025 | 20 | 0.052897364 | 0.050690533 | -0.0022068303 | -4.1719 |
| parent | Full-32 motor RMSE | all_22 | 22 | 0.066828375 | 0.061331654 | -0.0054967205 | -8.2251 |
| parent | Full-32 motor RMSE | omit_1015 | 21 | 0.060297254 | 0.059969037 | -0.00032821729 | -0.5443 |
| parent | Full-32 motor RMSE | omit_1016 | 21 | 0.065722528 | 0.060268986 | -0.0054535424 | -8.2978 |
| parent | Full-32 motor RMSE | omit_1025 | 21 | 0.063488722 | 0.059218207 | -0.0042705142 | -6.7264 |
| parent | Full-32 motor RMSE | omit_1016_1025 | 20 | 0.0621606 | 0.057996733 | -0.0041638669 | -6.6986 |
| parent | Late-22 motor RMSE | all_22 | 22 | 0.069300306 | 0.063866255 | -0.0054340512 | -7.8413 |
| parent | Late-22 motor RMSE | omit_1015 | 21 | 0.061000978 | 0.061987837 | 0.00098685915 | 1.6178 |
| parent | Late-22 motor RMSE | omit_1016 | 21 | 0.068548595 | 0.063175896 | -0.0053726984 | -7.8378 |
| parent | Late-22 motor RMSE | omit_1025 | 21 | 0.065487131 | 0.061196489 | -0.0042906419 | -6.5519 |
| parent | Late-22 motor RMSE | omit_1016_1025 | 20 | 0.064507175 | 0.060338124 | -0.0041690511 | -6.4629 |
| parent | Future latent NMSE | all_22 | 22 | 0.084435442 | 0.071786944 | -0.012648498 | -14.9801 |
| parent | Future latent NMSE | omit_1015 | 21 | 0.078063905 | 0.070361685 | -0.0077022205 | -9.8666 |
| parent | Future latent NMSE | omit_1016 | 21 | 0.080576 | 0.070298375 | -0.010277625 | -12.7552 |
| parent | Future latent NMSE | omit_1025 | 21 | 0.07890121 | 0.068171109 | -0.010730101 | -13.5994 |
| parent | Future latent NMSE | omit_1016_1025 | 20 | 0.074572085 | 0.066427319 | -0.0081447657 | -10.9220 |
| parent | Temporal-delta global NMSE | all_22 | 22 | 0.15307188 | 0.13384145 | -0.019230439 | -12.5630 |
| parent | Temporal-delta global NMSE | omit_1015 | 21 | 0.14434771 | 0.13195972 | -0.012387988 | -8.5820 |
| parent | Temporal-delta global NMSE | omit_1016 | 21 | 0.14275723 | 0.1291754 | -0.013581824 | -9.5139 |
| parent | Temporal-delta global NMSE | omit_1025 | 21 | 0.14113707 | 0.12693508 | -0.014201986 | -10.0625 |
| parent | Temporal-delta global NMSE | omit_1016_1025 | 20 | 0.12970993 | 0.12169042 | -0.0080195168 | -6.1827 |
| old_action_only | First-10 motor RMSE | all_22 | 22 | 0.058110442 | 0.05295101 | -0.0051594321 | -8.8787 |
| old_action_only | First-10 motor RMSE | omit_1015 | 21 | 0.0574675 | 0.053181668 | -0.0042858319 | -7.4578 |
| old_action_only | First-10 motor RMSE | omit_1016 | 21 | 0.052815582 | 0.051261496 | -0.0015540858 | -2.9425 |
| old_action_only | First-10 motor RMSE | omit_1025 | 21 | 0.055438616 | 0.052487689 | -0.0029509277 | -5.3229 |
| old_action_only | First-10 motor RMSE | omit_1016_1025 | 20 | 0.049745422 | 0.050690533 | 0.00094511116 | 1.8999 |
| old_action_only | Full-32 motor RMSE | all_22 | 22 | 0.069571863 | 0.061331654 | -0.0082402092 | -11.8442 |
| old_action_only | Full-32 motor RMSE | omit_1015 | 21 | 0.068378749 | 0.059969037 | -0.0084097124 | -12.2987 |
| old_action_only | Full-32 motor RMSE | omit_1016 | 21 | 0.057823017 | 0.060268986 | 0.0024459687 | 4.2301 |
| old_action_only | Full-32 motor RMSE | omit_1025 | 21 | 0.067036927 | 0.059218207 | -0.0078187199 | -11.6633 |
| old_action_only | Full-32 motor RMSE | omit_1016_1025 | 20 | 0.054573892 | 0.057996733 | 0.0034228414 | 6.2719 |
| old_action_only | Late-22 motor RMSE | all_22 | 22 | 0.07334153 | 0.063866255 | -0.0094752752 | -12.9194 |
| old_action_only | Late-22 motor RMSE | omit_1015 | 21 | 0.071914321 | 0.061987837 | -0.0099264842 | -13.8032 |
| old_action_only | Late-22 motor RMSE | omit_1016 | 21 | 0.059514644 | 0.063175896 | 0.0036612519 | 6.1519 |
| old_action_only | Late-22 motor RMSE | omit_1025 | 21 | 0.070833518 | 0.061196489 | -0.0096370291 | -13.6052 |
| old_action_only | Late-22 motor RMSE | omit_1016_1025 | 20 | 0.056189887 | 0.060338124 | 0.0041482367 | 7.3825 |
| old_action_only | Future latent NMSE | all_22 | 22 | 0.093010997 | 0.071786944 | -0.021224053 | -22.8189 |
| old_action_only | Future latent NMSE | omit_1015 | 21 | 0.089901025 | 0.070361685 | -0.01953934 | -21.7343 |
| old_action_only | Future latent NMSE | omit_1016 | 21 | 0.079005535 | 0.070298375 | -0.0087071604 | -11.0209 |
| old_action_only | Future latent NMSE | omit_1025 | 21 | 0.087696519 | 0.068171109 | -0.01952541 | -22.2647 |
| old_action_only | Future latent NMSE | omit_1016_1025 | 20 | 0.07272506 | 0.066427319 | -0.0062977411 | -8.6597 |
| old_action_only | Temporal-delta global NMSE | all_22 | 22 | 0.16885551 | 0.13384145 | -0.03501406 | -20.7361 |
| old_action_only | Temporal-delta global NMSE | omit_1015 | 21 | 0.16314914 | 0.13195972 | -0.031189421 | -19.1171 |
| old_action_only | Temporal-delta global NMSE | omit_1016 | 21 | 0.14098943 | 0.1291754 | -0.011814033 | -8.3794 |
| old_action_only | Temporal-delta global NMSE | omit_1025 | 21 | 0.15684102 | 0.12693508 | -0.02990594 | -19.0677 |
| old_action_only | Temporal-delta global NMSE | omit_1016_1025 | 20 | 0.12698092 | 0.12169042 | -0.0052905057 | -4.1664 |

## Focused checks

| Question | Full 22 mean delta | After omission mean delta | Full relative change (%) | After omission relative change (%) | Direction after omission |
|---|---:|---:|---:|---:|---|
| Full-32 motor vs old action-only: omit case 1016 | -0.0082402092 | 0.0024459687 | -11.8442 | 4.2301 | deterioration |
| Late-22 motor vs old action-only: omit case 1016 | -0.0094752752 | 0.0036612519 | -12.9194 | 6.1519 | deterioration |
| Late-22 motor vs parent: omit case 1015 | -0.0054340512 | 0.00098685915 | -7.8413 | 1.6178 | deterioration |

## Action / future overlap counts

Negative paired delta means improvement. Categories use strict signs; zero-boundary cases are reported separately.

| Comparison | Both improve | Action only improves | Future only improves | Both deteriorate | Boundary / unchanged | Total |
|---|---:|---:|---:|---:|---:|---:|
| parent | 14 | 1 | 3 | 4 | 0 | 22 |
| old_action_only | 12 | 0 | 4 | 6 | 0 | 22 |

## Figure

`paired_recovery_case_ids.png` redraws the existing paired-recovery values from the same case rows. The upper x-axis labels are actual case IDs, not row indices; the original analysis and image remain untouched.
