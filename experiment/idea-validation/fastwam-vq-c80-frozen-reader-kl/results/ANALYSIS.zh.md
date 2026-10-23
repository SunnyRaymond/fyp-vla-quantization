# C80 frozen-reader KL：已有缓存分析

用户取消同预算 C80 A+Z；以下比较参照是历史 C8 A+Z，不证明 KL 的独立收益。
缓存分析新增训练、模型/teacher forward、闭环均为0。

## calibration

| Metric | C8 A+Z | C80 KL | Change | W/L |
|---|---:|---:|---:|---:|
| first10motor_rmse | 0.0441469606 | 0.0436468461 | -1.133% | 45/35 |
| full32motor_rmse | 0.0464244542 | 0.0461106773 | -0.676% | 41/39 |
| late22motor_rmse | 0.0469743735 | 0.0469145932 | -0.127% | 42/38 |
| first10gripper_rmse | 0.041486297 | 0.0519310361 | +25.176% | 21/59 |
| future_latent_nmse | 0.0547353715 | 0.0522391139 | -4.561% | 46/34 |

## heldout

| Metric | C8 A+Z | C80 KL | Change | W/L |
|---|---:|---:|---:|---:|
| first10motor_rmse | 0.0442967744 | 0.04444355 | +0.331% | 20/20 |
| full32motor_rmse | 0.0475009795 | 0.0484203074 | +1.935% | 23/17 |
| late22motor_rmse | 0.048528504 | 0.0496173107 | +2.244% | 21/19 |
| first10gripper_rmse | 0.0570540305 | 0.0541784765 | -5.040% | 12/28 |
| future_latent_nmse | 0.060877371 | 0.0543611403 | -10.704% | 23/17 |

## Frozen-reader endpoints

```json
{
  "calibration": {
    "reader_kl": {
      "mean": 0.030807435155293206,
      "median_case": 0.013246686663478613
    },
    "reader_mse": {
      "mean": 0.009976665235808469,
      "median_case": 0.004276492865756154
    },
    "reader_first10motor_rmse": {
      "mean": 0.01982347443699837,
      "median_case": 0.015431986656039953
    },
    "reader_gap_first10motor_rmse": {
      "mean": 0.03720550467260182,
      "median_case": 0.037437804043293
    },
    "decomposition": {
      "student_motor_mse": 0.001998400157026481,
      "future_reader_effect_mse": 0.0005431532464319831,
      "quantized_reader_gap_mse": 0.0013960202821155132,
      "cross_term": 5.922662808686141e-05
    }
  },
  "heldout": {
    "reader_kl": {
      "mean": 0.030650933357537723,
      "median_case": 0.01388291537296027
    },
    "reader_mse": {
      "mean": 0.009888074985792628,
      "median_case": 0.004481532727368176
    },
    "reader_first10motor_rmse": {
      "mean": 0.019625502813141793,
      "median_case": 0.01633460819721222
    },
    "reader_gap_first10motor_rmse": {
      "mean": 0.03782240708824247,
      "median_case": 0.03785501327365637
    },
    "decomposition": {
      "student_motor_mse": 0.0020554475078824906,
      "future_reader_effect_mse": 0.0005478375576719124,
      "quantized_reader_gap_mse": 0.0014408646941533893,
      "cross_term": 6.674524795897696e-05
    }
  }
}
```

## Initial to final calibration reader fit

```json
{
  "reader_kl": {
    "initial_raw_c80": 0.027481458333932098,
    "final_c80": 0.030807435155293206,
    "relative_change": 0.12102621269026459
  },
  "reader_mse": {
    "initial_raw_c80": 0.009132323118683417,
    "final_c80": 0.009976665235808469,
    "relative_change": 0.09245644357432448
  }
}
```

## Recorded update durations

```json
{
  "original_c8_az": {
    "records": 8,
    "per_update_seconds": [
      230.4393630521372,
      224.4571682261303,
      225.0392103716731,
      224.52288391580805,
      225.44341875007376,
      224.2129775248468,
      224.28616922721267,
      225.56973976874724
    ],
    "median_all_seconds": 224.78104714374058,
    "median_excluding_first_seconds": 224.52288391580805,
    "scope": "descriptive recorded update durations; different timing boundaries and cases"
  },
  "c80_reader_kl": {
    "records": 80,
    "per_update_seconds": [
      282.65358909079805,
      249.4754243097268,
      243.59813841013238,
      241.23469305364415,
      242.71140269516036,
      239.05377888213843,
      239.49887498887256,
      239.61867505917326,
      238.7586259529926,
      239.24165661400184,
      240.35471570305526,
      245.53700945526361,
      239.9852765048854,
      240.38554694876075,
      239.56191871035844,
      261.39023797633126,
      266.124134354759,
      250.7434585080482,
      259.3002426601015,
      255.99643960082904,
      255.6417546980083,
      253.66778461216018,
      256.5491315680556,
      254.5879347040318,
      251.34709871467203,
      254.08220470603555,
      251.59512746566907,
      248.2134526516311,
      247.91547590261325,
      264.06939605111256,
      263.0632861992344,
      274.7694305428304,
      269.0171283329837,
      261.54222220927477,
      263.9180097160861,
      261.1699824030511,
      262.82285178406164,
      260.29136175615713,
      260.17807333916426,
      256.47600710205734,
      256.956479833927,
      258.04635706730187,
      256.67901062080637,
      254.34062386816368,
      258.6322845611721,
      271.3048695488833,
      261.0054668779485,
      258.9672024836764,
      255.40997740393505,
      255.86843754118308,
      255.43976512597874,
      260.3772422298789,
      258.02947178902104,
      260.00825904496014,
      262.769360056147,
      261.3314302507788,
      259.4196605896577,
      261.48730813805014,
      263.4463001610711,
      271.6497066956945,
      306.013748108875,
      262.0549124972895,
      256.3506549419835,
      251.57801971584558,
      253.76060771383345,
      259.86172222392634,
      252.7318737222813,
      248.26848458312452,
      266.1561827259138,
      281.19276597490534,
      290.88833614392206,
      265.1545387208462,
      246.0589799550362,
      247.83354125218466,
      251.1920514414087,
      251.36522366013378,
      247.5786144998856,
      246.6619717241265,
      247.4206668571569,
      251.8498848201707
    ],
    "median_all_seconds": 256.1735472714063,
    "median_excluding_first_seconds": 255.99643960082904,
    "scope": "descriptive recorded update durations; different timing boundaries and cases"
  },
  "cancelled_c80_az": {
    "records": 2,
    "per_update_seconds": [
      283.06923201400787,
      258.5616631619632
    ],
    "median_all_seconds": 270.81544758798555,
    "median_excluding_first_seconds": 258.5616631619632,
    "scope": "descriptive recorded update durations; different timing boundaries and cases"
  },
  "cancelled_c80_az_committed_updates": 2,
  "implementation": {
    "original_c8_az": "one differentiable student forward/backward per update",
    "c80_az": "one no-grad student forward plus differentiable student replay/VJP",
    "c80_kl": "same staged student path plus frozen-reader backward and dense-weight parking"
  }
}
```

KL causal attribution remains unresolved because the matched C80 A+Z arm was cancelled.
