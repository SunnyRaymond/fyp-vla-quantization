# Shared-codeword finite intervention

完整60contexts；test22cases/44contexts。所有方向仅由8个calibration contexts决定。
Untied是增加自由度的容量诊断，不是同bits可部署方法。

| Split | Arm | Future NMSE | Teacher first10 | Teacher full32 | Teacher late22 | Teacher gripper | Native first10 |
|---|---|---:|---:|---:|---:|---:|---:|
| calibration | shared_0 | 0.0881727329 | 0.0265346949 | 0.0505865472 | 0.0571379156 | 0.0027321934 | 0.043398567 |
| calibration | baseline_0 | 0.0820088238 | 0.0283135731 | 0.0457387164 | 0.0506437209 | 0.00230038406 | 0.0443291143 |
| calibration | untied_0 | 0.0713763814 | 0.0266474984 | 0.0422280838 | 0.0471170454 | 0.00230038406 | 0.0435211305 |
| calibration | random_0 | 0.0829685186 | 0.0253973654 | 0.0489970215 | 0.0552751548 | 0.00299300834 | 0.0431182897 |
| calibration | shared_1 | 0.0767219132 | 0.0256255367 | 0.0474969905 | 0.0533166393 | 0.00201313739 | 0.0431278897 |
| calibration | baseline_1 | 0.0820088238 | 0.0283135731 | 0.0457387164 | 0.0506437209 | 0.00230038406 | 0.0443291143 |
| calibration | untied_1 | 0.105596013 | 0.0428169458 | 0.0565262486 | 0.0602600814 | 0.00258191164 | 0.0582016311 |
| calibration | random_1 | 0.0833436609 | 0.025200356 | 0.0478452377 | 0.0535251589 | 0.00279444632 | 0.0417129095 |
| selection | shared_0 | 0.0370542482 | 0.0192719748 | 0.0195022486 | 0.0192609645 | 0.0010543644 | 0.0434682262 |
| selection | baseline_0 | 0.0345952888 | 0.0162318083 | 0.0173395558 | 0.0175890005 | 0.000990406465 | 0.0415074304 |
| selection | untied_0 | 0.0333491254 | 0.0198723301 | 0.0194276519 | 0.0190792659 | 0.00101299089 | 0.0430281386 |
| selection | random_0 | 0.0334407871 | 0.0175568571 | 0.019074499 | 0.0189575532 | 0.00102685727 | 0.0402343012 |
| selection | shared_1 | 0.0349304639 | 0.0176590312 | 0.0190757664 | 0.0193411318 | 0.000772040439 | 0.0441370476 |
| selection | baseline_1 | 0.0345952888 | 0.0162318083 | 0.0173395558 | 0.0175890005 | 0.000990406465 | 0.0415074304 |
| selection | untied_1 | 0.0646597939 | 0.0323936332 | 0.0329213787 | 0.0326645398 | 0.0010394831 | 0.0524079897 |
| selection | random_1 | 0.0436045183 | 0.0176072899 | 0.0187233873 | 0.018832993 | 0.000885075016 | 0.0394955045 |
| test | shared_0 | 0.0832546743 | 0.039111998 | 0.0467319518 | 0.0486325334 | 0.00165287544 | 0.0556034299 |
| test | baseline_0 | 0.0930109973 | 0.0412432187 | 0.0529077989 | 0.0562831438 | 0.00163013973 | 0.0581104419 |
| test | untied_0 | 0.0730160971 | 0.0333009363 | 0.0417523264 | 0.0439867867 | 0.00181447881 | 0.0519148836 |
| test | random_0 | 0.0822009249 | 0.0390220974 | 0.0478189374 | 0.0504736252 | 0.00168252506 | 0.0553539609 |
| test | shared_1 | 0.0777559782 | 0.0359131415 | 0.0432696822 | 0.0451431809 | 0.00169448237 | 0.0535368901 |
| test | baseline_1 | 0.0930109973 | 0.0412432187 | 0.0529077989 | 0.0562831438 | 0.00163013973 | 0.0581104419 |
| test | untied_1 | 0.121805413 | 0.0590209782 | 0.0609870222 | 0.060024035 | 0.00206500849 | 0.0737578399 |
| test | random_1 | 0.0863554274 | 0.0414429962 | 0.050888958 | 0.0530919765 | 0.00173645307 | 0.0582130104 |

| Test contrast | Future mean delta | Future improved | Teacher first10 delta | Motor improved |
|---|---:|---:|---:|---:|
| test/0/untied-minus-shared | -0.0102385772 | 14/22 | -0.0058110617 | 16/22 |
| test/0/untied-minus-random | -0.00918482777 | 11/22 | -0.00572116111 | 15/22 |
| test/0/shared-minus-baseline | -0.00975632299 | 13/22 | -0.00213122076 | 13/22 |
| test/0/untied-minus-baseline | -0.0199949002 | 13/22 | -0.00794228246 | 15/22 |
| test/0/random-minus-baseline | -0.0108100724 | 10/22 | -0.00222112135 | 13/22 |
| test/1/untied-minus-shared | 0.0440494346 | 2/22 | 0.0231078367 | 2/22 |
| test/1/untied-minus-random | 0.0354499853 | 2/22 | 0.017577982 | 2/22 |
| test/1/shared-minus-baseline | -0.0152550191 | 11/22 | -0.00533007725 | 15/22 |
| test/1/untied-minus-baseline | 0.0287944155 | 4/22 | 0.0177777595 | 5/22 |
| test/1/random-minus-baseline | -0.00665556982 | 11/22 | 0.000199777476 | 13/22 |

| Untied versus shared epsilon | Both improve | Future improves / motor worsens | Future worsens / motor improves | Both worsen | Any exact tie |
|---|---:|---:|---:|---:|---:|
| 0.005 | 12 | 2 | 4 | 4 | 0 |
| 0.015 | 1 | 1 | 1 | 19 | 0 |

Quadrants are descriptive signs for22 held-out cases after averaging paired seeds, without an uncertainty or significance claim.

| Module | Shared tangent rank | Support dimension | Isotropic rank/D | Actual retained energy | Actual/reference |
|---|---:|---:|---:|---:|---:|
| video_expert.time_embedding.0 | 32 | 31152 | 0.00102722137 | 0.00226644997 | 2.20639 |
| video_expert.text_embedding.0 | 32 | 455628 | 7.02327337e-05 | 8.47869519e-05 | 1.20723 |
| video_expert.blocks.24.ffn.0 | 32 | 1995576 | 1.60354705e-05 | 1.85337701e-05 | 1.1558 |
| video_expert.blocks.17.ffn.0 | 32 | 1992884 | 1.60571313e-05 | 9.06413767e-06 | 0.564493 |

The isotropic rank/D reference explains dimensional restriction only. It is not a matched alternative assignment, empirical gradient null, significance test, or finite-BF16 performance prediction.

Predefined engineering progression gate=False; not a statistical significance test.
Internal gradient/finite-perturbation diagnostic; within-codeword samples distinct from exact codebook gradients; STE backward distinct from finite BF16 forward; no proof of historical training cause or fixed-bit assignment improvement.
