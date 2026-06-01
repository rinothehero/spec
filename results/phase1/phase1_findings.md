# Phase 1 Findings

작성일: 2026-06-01

## Scope

1차 목표는 DSD에서 gamma/workload가 정해졌을 때 verifier/drafter 주파수에 따라 latency와 energy/token이 어떻게 달라지는지 확인하는 것이다.

이번 실행은 여섯 묶음으로 진행했다.

1. Main verifier-frequency sweep
2. Extra drafter-frequency sweep
3. Larger-verifier validation sweep
4. Cross-model-family validation sweep
5. Drafter-main frequency sweep
6. Fine-grained verifier sweep with drafter fixed at high frequency

## Environment

```text
Drafter pod: Nano
Verifier pod: AGX
Driver pod: Mas
Decoding: distributed speculative decoding
Power scope: measured drafter + verifier device energy
Primary rail for analysis: tot_power
```

Model pairs tested:

```text
Main:              Qwen/Qwen2.5-0.5B -> Qwen/Qwen2.5-3B
Larger verifier:   Qwen/Qwen2.5-0.5B -> Qwen/Qwen2.5-7B
Model-family check: facebook/opt-125m -> facebook/opt-1.3b
```

최종 주파수 상태는 원복 확인했다.

```text
AGX GPU: 408000000 Hz, min=408000000, max=408000000
Nano GPU: 624750000 Hz, min=306000000, max=624750000
```

현재 떠 있는 서버는 drafter-main 실험 이후 Qwen pair이다.

```text
Nano drafter: Qwen/Qwen2.5-0.5B
AGX verifier: Qwen/Qwen2.5-3B
```

## Result Files

Main raw CSVs:

```text
SPEC/results/phase1/raw/phase1_fv408mhz_qwen25_0p5b_to_3b_g1_2_4_8_p10_r1_w0_t64.csv
SPEC/results/phase1/raw/phase1_fv816mhz_qwen25_0p5b_to_3b_g1_2_4_8_p10_r1_w0_t64.csv
SPEC/results/phase1/raw/phase1_fv1300p5mhz_qwen25_0p5b_to_3b_g1_2_4_8_p10_r1_w0_t64.csv
```

Main summaries:

```text
SPEC/results/phase1/summary/phase1_summary.md
SPEC/results/phase1/summary/phase1_summary_by_gamma_freq.csv
SPEC/results/phase1/summary/phase1_best_frequency_by_gamma.csv
SPEC/results/phase1/summary/phase1_filtered_tot_power_rows.csv
```

Main plots:

```text
SPEC/results/phase1/summary/plots/energy_per_token_by_gamma.svg
SPEC/results/phase1/summary/plots/latency_by_gamma.svg
SPEC/results/phase1/summary/plots/pareto_energy_latency.svg
```

Extra drafter-frequency files:

```text
SPEC/results/phase1/extra/raw/phase1_extra_fd408_624_fv816_g1_4_8_p5_r1_w0_t64.csv
SPEC/results/phase1/extra/summary/phase1_extra_drafter_freq_summary.md
SPEC/results/phase1/extra/summary/phase1_extra_drafter_freq_summary.csv
SPEC/results/phase1/extra/summary/phase1_extra_drafter_freq_best.csv
SPEC/results/phase1/extra/summary/plots/extra_drafter_energy_per_token.svg
SPEC/results/phase1/extra/summary/plots/extra_drafter_latency.svg
```

Larger-verifier validation files:

```text
SPEC/results/phase1/qwen25_0p5b_to_7b/raw/
SPEC/results/phase1/qwen25_0p5b_to_7b/summary/phase1_summary.md
SPEC/results/phase1/qwen25_0p5b_to_7b/summary/phase1_best_frequency_by_gamma.csv
SPEC/results/phase1/qwen25_0p5b_to_7b/summary/plots/
```

Cross-model-family validation files:

```text
SPEC/results/phase1/opt_125m_to_1p3b/raw/
SPEC/results/phase1/opt_125m_to_1p3b/summary/phase1_summary.md
SPEC/results/phase1/opt_125m_to_1p3b/summary/phase1_best_frequency_by_gamma.csv
SPEC/results/phase1/opt_125m_to_1p3b/summary/plots/
```

Drafter-main files:

```text
SPEC/results/phase1/drafter_main/raw/phase1_drafter_main_qwen25_0p5b_to_3b_fd306_408_510_612_624_fv816_g1_2_4_8_p10_r1_w0_t64.csv
SPEC/results/phase1/drafter_main/summary/drafter_main_summary.md
SPEC/results/phase1/drafter_main/summary/drafter_main_summary_by_gamma_freq.csv
SPEC/results/phase1/drafter_main/summary/drafter_main_best_frequency_by_gamma.csv
SPEC/results/phase1/drafter_main/summary/drafter_main_prompt_best_frequency.csv
SPEC/results/phase1/drafter_main/summary/plots/
```

Fine-grained verifier sweep files:

```text
SPEC/results/phase1/verifier_fd624p75/raw/
SPEC/results/phase1/verifier_fd624p75/summary/phase1_summary.md
SPEC/results/phase1/verifier_fd624p75/summary/phase1_summary_by_gamma_freq.csv
SPEC/results/phase1/verifier_fd624p75/summary/phase1_best_frequency_by_gamma.csv
SPEC/results/phase1/verifier_fd624p75/summary/plots/
```

Issue log:

```text
SPEC/results/phase1/logs/issues.md
```

Partial/probe files are intentionally separated from main raw analysis:

```text
SPEC/results/phase1/partial/
SPEC/results/phase1/probes/
```

## Data Integrity

Main sweep:

```text
408 MHz CSV:    360 rows, 40 measured tot_power rows, 40 complete
816 MHz CSV:    360 rows, 40 measured tot_power rows, 40 complete
1300.5 MHz CSV: 360 rows, 40 measured tot_power rows, 40 complete
```

Extra drafter sweep:

```text
270 rows, 30 measured tot_power rows, 30 complete
```

Larger-verifier validation sweep:

```text
408 MHz CSV:    180 rows, 20 measured tot_power rows, 20 complete
816 MHz CSV:    180 rows, 20 measured tot_power rows, 20 complete
1300.5 MHz CSV: 180 rows, 20 measured tot_power rows, 20 complete
```

Cross-model-family validation sweep:

```text
408 MHz CSV:    180 rows, 20 measured tot_power rows, 20 complete
816 MHz CSV:    180 rows, 20 measured tot_power rows, 20 complete
1300.5 MHz CSV: 180 rows, 20 measured tot_power rows, 20 complete
```

Drafter-main sweep:

```text
1800 rows, 200 measured tot_power rows, 200 complete
```

Fine-grained verifier sweep with drafter fixed at 624.75 MHz:

```text
408 MHz CSV:    360 rows, 40 measured tot_power rows, 40 complete
612 MHz CSV:    360 rows, 40 measured tot_power rows, 40 complete
816 MHz CSV:    360 rows, 40 measured tot_power rows, 40 complete
1300.5 MHz CSV: 360 rows, 40 measured tot_power rows, 40 complete
```

`system_energy_complete=1` for every measured `tot_power` row used in the summaries.

## Main Experiment

Experiment matrix:

```text
prompts = 10
runs = 1
warmup = 0
max_new_tokens = 64
gamma = 1, 2, 4, 8
verifier GPU frequency = 408 MHz, 816 MHz, 1300.5 MHz
drafter GPU frequency = not controlled / not logged
```

Total measured sessions:

```text
10 prompts * 4 gamma values * 3 verifier frequencies = 120
```

### Main Summary

| gamma | verifier freq | mean latency ms | mean tokens/s | mean energy/token mJ | mean accept rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 408 MHz | 17344.095 | 3.570110 | 4532.408 | 0.788142 |
| 1 | 816 MHz | 16460.913 | 3.773200 | 4183.237 | 0.788142 |
| 1 | 1300.5 MHz | 16304.039 | 3.802790 | 4724.088 | 0.788142 |
| 2 | 408 MHz | 15476.302 | 4.031540 | 3608.787 | 0.676108 |
| 2 | 816 MHz | 14754.731 | 4.231770 | 3336.717 | 0.676108 |
| 2 | 1300.5 MHz | 14522.610 | 4.294040 | 3772.049 | 0.676108 |
| 4 | 408 MHz | 14076.544 | 4.485630 | 2861.189 | 0.559512 |
| 4 | 816 MHz | 13255.583 | 4.777110 | 2602.638 | 0.559512 |
| 4 | 1300.5 MHz | 13371.148 | 4.710620 | 2880.001 | 0.559512 |
| 8 | 408 MHz | 15919.441 | 4.153020 | 2746.709 | 0.395535 |
| 8 | 816 MHz | 15097.970 | 4.373350 | 2548.515 | 0.395535 |
| 8 | 1300.5 MHz | 15213.305 | 4.338270 | 2773.158 | 0.395535 |

### Best Frequency By Gamma

5 percent latency overhead constraint 기준에서도 energy-best는 다음과 같다.

| gamma | fastest freq | energy-best freq | latency-constrained energy-best freq | energy saving vs high |
|---:|---:|---:|---:|---:|
| 1 | 1300.5 MHz | 816 MHz | 816 MHz | 11.449% |
| 2 | 1300.5 MHz | 816 MHz | 816 MHz | 11.541% |
| 4 | 816 MHz | 816 MHz | 816 MHz | 9.631% |
| 8 | 816 MHz | 816 MHz | 816 MHz | 8.101% |

## Main Insights

### Insight 1: Maximum frequency is not energy-optimal

1300.5 MHz was sometimes the fastest, especially for gamma 1 and 2, but it was never the best energy/token setting.

For gamma 1:

```text
1300.5 MHz latency: 16304.039 ms
816 MHz latency:    16460.913 ms

1300.5 MHz energy/token: 4724.088 mJ
816 MHz energy/token:    4183.237 mJ
```

The latency difference was small, but energy/token was much worse at 1300.5 MHz.

### Insight 2: Minimum frequency is also not energy-optimal

408 MHz was lower power, but it was slower enough that energy/token was worse than 816 MHz for every gamma in the 10-prompt average.

For gamma 4:

```text
408 MHz energy/token: 2861.189 mJ
816 MHz energy/token: 2602.638 mJ
```

So simply lowering frequency as much as possible is also not optimal.

### Insight 3: A mid-frequency operating point is robust for this model pair

In this Qwen2.5-0.5B -> Qwen2.5-3B setup, 816 MHz was the best verifier GPU frequency for all tested gamma values.

This is slightly different from the earlier one-prompt pilot where gamma 4 favored 408 MHz. The broader 10-prompt result is more reliable for the current setup.

The immediate statement should therefore be:

```text
For this DSD workload, the best energy point is an intermediate frequency, not the maximum or minimum.
```

The stronger statement:

```text
Each gamma has a different best frequency.
```

is not supported by this initial 3-point frequency grid. A later fine-grained verifier sweep added 612 MHz and found that the verifier energy-best frequency can differ by gamma.

### Insight 4: gamma still changes the workload strongly

Accept rate decreases as gamma increases:

```text
gamma=1: 0.788142
gamma=2: 0.676108
gamma=4: 0.559512
gamma=8: 0.395535
```

Energy/token also changes with gamma:

```text
gamma=1 best: 4183.237 mJ/token
gamma=2 best: 3336.717 mJ/token
gamma=4 best: 2602.638 mJ/token
gamma=8 best: 2548.515 mJ/token
```

So gamma clearly changes the DSD workload and system behavior, even though the verifier-frequency optimum was the same 816 MHz in this run.

## Fine-Grained Verifier Sweep: Drafter Fixed At 624.75 MHz

After the initial verifier sweep, a finer verifier-frequency experiment was run with the drafter fixed at the Nano maximum frequency.

Experiment matrix:

```text
drafter = Qwen/Qwen2.5-0.5B
verifier = Qwen/Qwen2.5-3B
drafter GPU frequency = 624.75 MHz fixed
prompts = 10
runs = 1
warmup = 0
max_new_tokens = 64
gamma = 1, 2, 4, 8
verifier GPU frequency = 408 MHz, 612 MHz, 816 MHz, 1300.5 MHz
```

Total measured sessions:

```text
10 prompts * 4 gamma values * 4 verifier frequencies = 160
```

Data integrity:

```text
160 measured tot_power rows, 160 complete
```

### Fine-Grained Verifier Summary

| gamma | verifier freq | mean latency ms | mean tokens/s | mean energy/token mJ | mean accept rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 408 MHz | 17106.437 | 3.620080 | 4484.171 | 0.788142 |
| 1 | 612 MHz | 16355.471 | 3.791420 | 4189.182 | 0.788142 |
| 1 | 816 MHz | 16106.858 | 3.836130 | 4210.161 | 0.788142 |
| 1 | 1300.5 MHz | 16096.542 | 3.858850 | 4760.574 | 0.788142 |
| 2 | 408 MHz | 15322.438 | 4.072640 | 3570.869 | 0.676108 |
| 2 | 612 MHz | 14869.283 | 4.200120 | 3379.709 | 0.676108 |
| 2 | 816 MHz | 14242.151 | 4.361030 | 3353.411 | 0.676108 |
| 2 | 1300.5 MHz | 14535.348 | 4.296040 | 3765.500 | 0.676108 |
| 4 | 408 MHz | 13894.606 | 4.537990 | 2802.549 | 0.559512 |
| 4 | 612 MHz | 13439.493 | 4.699100 | 2661.239 | 0.559512 |
| 4 | 816 MHz | 13242.520 | 4.758370 | 2659.786 | 0.559512 |
| 4 | 1300.5 MHz | 13294.593 | 4.740220 | 2939.585 | 0.559512 |
| 8 | 408 MHz | 15746.389 | 4.201860 | 2713.182 | 0.395535 |
| 8 | 612 MHz | 15169.775 | 4.341130 | 2552.485 | 0.395535 |
| 8 | 816 MHz | 15108.984 | 4.353930 | 2563.986 | 0.395535 |
| 8 | 1300.5 MHz | 15055.948 | 4.367470 | 2772.464 | 0.395535 |

### Fine-Grained Verifier Best Frequency

| gamma | fastest verifier freq | energy-best verifier freq | energy/token |
|---:|---:|---:|---:|
| 1 | 1300.5 MHz | 612 MHz | 4189.182 mJ/token |
| 2 | 816 MHz | 816 MHz | 3353.411 mJ/token |
| 4 | 816 MHz | 816 MHz | 2659.786 mJ/token |
| 8 | 1300.5 MHz | 612 MHz | 2552.485 mJ/token |

### Fine-Grained Verifier Insight

Adding 612 MHz changed the conclusion for verifier control:

```text
gamma=1 -> 612 MHz energy-best
gamma=2 -> 816 MHz energy-best
gamma=4 -> 816 MHz energy-best
gamma=8 -> 612 MHz energy-best
```

The initial 408/816/1300.5 MHz sweep made 816 MHz look universally best. The finer grid shows that gamma 1 and gamma 8 have a better energy point at 612 MHz. This is important because it means the frequency optimum can be missed if the frequency grid is too coarse.

Implementation note:

```text
summarize_phase1.py was updated to recognize the fv612mhz filename pattern.
```

## Extra Experiment: Drafter Frequency

Extra experiment matrix:

```text
prompts = 5
runs = 1
warmup = 0
max_new_tokens = 64
gamma = 1, 4, 8
verifier GPU frequency = 816 MHz fixed
drafter GPU frequency = 408 MHz, 624.75 MHz
```

### Extra Summary

| gamma | drafter freq | mean latency ms | mean tokens/s | mean energy/token mJ |
|---:|---:|---:|---:|---:|
| 1 | 408 MHz | 17254.483 | 3.712820 | 4228.712 |
| 1 | 624.75 MHz | 17052.649 | 3.762600 | 4346.035 |
| 4 | 408 MHz | 14887.742 | 4.387700 | 2795.151 |
| 4 | 624.75 MHz | 14279.935 | 4.583480 | 2696.811 |
| 8 | 408 MHz | 16787.121 | 4.089560 | 2688.521 |
| 8 | 624.75 MHz | 16455.194 | 4.200200 | 2712.384 |

### Extra Insight

Drafter frequency does interact with gamma, but the pattern is not monotonic.

Energy-best drafter frequency:

```text
gamma=1 -> 408 MHz
gamma=4 -> 624.75 MHz
gamma=8 -> 408 MHz
```

Fastest drafter frequency:

```text
gamma=1 -> 624.75 MHz
gamma=4 -> 624.75 MHz
gamma=8 -> 624.75 MHz
```

This supports the broader motivation for joint drafter/verifier frequency control. The verifier-only sweep found a robust mid-frequency optimum, while the drafter sweep shows gamma-dependent energy tradeoffs.

## Drafter-Main Experiment: Qwen2.5-0.5B -> Qwen2.5-3B

Because the research focus is drafter-side frequency control, a larger drafter sweep was run after the verifier calibration.

Experiment matrix:

```text
drafter = Qwen/Qwen2.5-0.5B
verifier = Qwen/Qwen2.5-3B
verifier GPU frequency = 816 MHz fixed
prompts = 10
runs = 1
warmup = 0
max_new_tokens = 64
gamma = 1, 2, 4, 8
drafter GPU frequency = 306 MHz, 408 MHz, 510 MHz, 612 MHz, 624.75 MHz
```

Total measured sessions:

```text
10 prompts * 4 gamma values * 5 drafter frequencies = 200
```

### Drafter-Main Summary

| gamma | drafter freq | mean latency ms | mean tokens/s | mean energy/token mJ | mean accept rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 306 MHz | 17061.672 | 3.635640 | 4220.596 | 0.788142 |
| 1 | 408 MHz | 16669.819 | 3.711190 | 4308.490 | 0.788142 |
| 1 | 510 MHz | 16283.967 | 3.799990 | 4234.551 | 0.788142 |
| 1 | 612 MHz | 16451.994 | 3.770190 | 4238.200 | 0.788142 |
| 1 | 624.75 MHz | 16398.027 | 3.783900 | 4245.794 | 0.788142 |
| 2 | 306 MHz | 15261.543 | 4.096240 | 3349.473 | 0.676108 |
| 2 | 408 MHz | 14807.763 | 4.232720 | 3403.883 | 0.676108 |
| 2 | 510 MHz | 14765.273 | 4.229570 | 3409.186 | 0.676108 |
| 2 | 612 MHz | 14634.240 | 4.261000 | 3396.208 | 0.676108 |
| 2 | 624.75 MHz | 14541.215 | 4.289550 | 3380.581 | 0.676108 |
| 4 | 306 MHz | 14332.151 | 4.409340 | 2698.086 | 0.559512 |
| 4 | 408 MHz | 13741.691 | 4.586270 | 2674.964 | 0.559512 |
| 4 | 510 MHz | 13596.222 | 4.652360 | 2690.256 | 0.559512 |
| 4 | 612 MHz | 13261.800 | 4.781960 | 2693.787 | 0.559512 |
| 4 | 624.75 MHz | 13414.672 | 4.715690 | 2695.777 | 0.559512 |
| 8 | 306 MHz | 16025.408 | 4.125970 | 2637.186 | 0.395535 |
| 8 | 408 MHz | 15516.675 | 4.264850 | 2591.576 | 0.395535 |
| 8 | 510 MHz | 15109.834 | 4.354830 | 2559.323 | 0.395535 |
| 8 | 612 MHz | 15039.587 | 4.396600 | 2603.643 | 0.395535 |
| 8 | 624.75 MHz | 15082.978 | 4.396090 | 2562.756 | 0.395535 |

### Drafter-Main Best Frequency

| gamma | fastest drafter freq | energy-only best drafter freq | latency-constrained best drafter freq | saving vs high |
|---:|---:|---:|---:|---:|
| 1 | 510 MHz | 306 MHz | 306 MHz | 0.593% |
| 2 | 624.75 MHz | 306 MHz | 306 MHz | 0.920% |
| 4 | 612 MHz | 408 MHz | 408 MHz | 0.772% |
| 8 | 612 MHz | 510 MHz | 510 MHz | 0.134% |

### Drafter-Main Insight

This is the strongest result for the research direction so far.

The energy-best drafter frequency changes with gamma:

```text
gamma=1 -> 306 MHz
gamma=2 -> 306 MHz
gamma=4 -> 408 MHz
gamma=8 -> 510 MHz
```

The fastest drafter frequency also changes:

```text
gamma=1 -> 510 MHz
gamma=2 -> 624.75 MHz
gamma=4 -> 612 MHz
gamma=8 -> 612 MHz
```

The absolute energy savings versus the highest drafter frequency are small in this p10/r1 average, roughly 0.1-0.9 percent. However, the direction is important: the best energy point is not always the highest drafter frequency, and it moves upward as gamma increases.

This directly supports the drafter-centered research statement:

```text
Given a DSD gamma/workload, a drafter-aware controller can choose a better drafter GPU frequency than a fixed maximum-frequency policy.
```

## Larger-Verifier Validation: Qwen2.5-0.5B -> Qwen2.5-7B

To test whether a larger verifier exposes stronger frequency sensitivity, an additional sweep was run with the same drafter and a larger verifier.

Experiment matrix:

```text
drafter = Qwen/Qwen2.5-0.5B
verifier = Qwen/Qwen2.5-7B
prompts = 5
runs = 1
warmup = 0
max_new_tokens = 64
gamma = 1, 2, 4, 8
verifier GPU frequency = 408 MHz, 816 MHz, 1300.5 MHz
```

### Larger-Verifier Summary

| gamma | verifier freq | mean latency ms | mean tokens/s | mean energy/token mJ | mean accept rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 408 MHz | 24734.819 | 2.566340 | 9385.190 | 0.763462 |
| 1 | 816 MHz | 18170.877 | 3.494120 | 5947.395 | 0.763462 |
| 1 | 1300.5 MHz | 16478.974 | 3.849500 | 6573.895 | 0.763462 |
| 2 | 408 MHz | 20037.827 | 3.179540 | 6868.999 | 0.705308 |
| 2 | 816 MHz | 15072.716 | 4.226240 | 4345.846 | 0.705308 |
| 2 | 1300.5 MHz | 14439.871 | 4.422600 | 4786.269 | 0.705308 |
| 4 | 408 MHz | 17854.519 | 3.615840 | 5239.230 | 0.569632 |
| 4 | 816 MHz | 14132.332 | 4.571840 | 3416.163 | 0.569632 |
| 4 | 1300.5 MHz | 13485.989 | 4.791800 | 3652.962 | 0.569632 |
| 8 | 408 MHz | 18074.923 | 3.782680 | 4368.396 | 0.429295 |
| 8 | 816 MHz | 15440.193 | 4.417740 | 2992.945 | 0.429295 |
| 8 | 1300.5 MHz | 14801.884 | 4.567120 | 3159.104 | 0.429295 |

### Larger-Verifier Best Frequency

| gamma | fastest freq | energy-only best freq | latency-constrained best freq | energy saving vs high |
|---:|---:|---:|---:|---:|
| 1 | 1300.5 MHz | 816 MHz | 1300.5 MHz | 0.000% |
| 2 | 1300.5 MHz | 816 MHz | 816 MHz | 9.202% |
| 4 | 1300.5 MHz | 816 MHz | 816 MHz | 6.482% |
| 8 | 1300.5 MHz | 816 MHz | 816 MHz | 5.260% |

### Larger-Verifier Insight

The 7B verifier makes the latency cost of low frequency much clearer.

For gamma 1:

```text
408 MHz latency:    24734.819 ms
816 MHz latency:    18170.877 ms
1300.5 MHz latency: 16478.974 ms
```

Energy-only best remains 816 MHz for all gamma values, but the latency-constrained policy differs:

```text
gamma=1 -> 1300.5 MHz under 5% latency constraint
gamma=2 -> 816 MHz
gamma=4 -> 816 MHz
gamma=8 -> 816 MHz
```

This strengthens the case that the definition of "optimal" must include latency constraints. Without a latency constraint, 816 MHz is the robust energy point. With a strict latency constraint, gamma 1 on the larger verifier needs high frequency to stay close to the fastest latency.

## Cross-Model-Family Validation: OPT-125M -> OPT-1.3B

To check whether the result was only a Qwen-specific artifact, a smaller OPT model pair was also tested.

Experiment matrix:

```text
drafter = facebook/opt-125m
verifier = facebook/opt-1.3b
prompts = 5
runs = 1
warmup = 0
max_new_tokens = 64
gamma = 1, 2, 4, 8
verifier GPU frequency = 408 MHz, 816 MHz, 1300.5 MHz
```

### OPT Summary

| gamma | verifier freq | mean latency ms | mean tokens/s | mean energy/token mJ | mean accept rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 408 MHz | 9976.246 | 6.473820 | 2315.495 | 0.873203 |
| 1 | 816 MHz | 9342.539 | 6.859880 | 2039.494 | 0.873203 |
| 1 | 1300.5 MHz | 9133.595 | 7.019760 | 2197.610 | 0.873203 |
| 2 | 408 MHz | 7991.814 | 8.034380 | 1717.916 | 0.792602 |
| 2 | 816 MHz | 7971.486 | 8.037380 | 1593.609 | 0.792602 |
| 2 | 1300.5 MHz | 7922.015 | 8.155440 | 1810.910 | 0.792602 |
| 4 | 408 MHz | 6838.789 | 9.589260 | 1287.407 | 0.682329 |
| 4 | 816 MHz | 6544.049 | 10.038420 | 1188.727 | 0.682329 |
| 4 | 1300.5 MHz | 6563.556 | 9.930320 | 1299.229 | 0.682329 |
| 8 | 408 MHz | 6429.425 | 10.403740 | 1041.972 | 0.515900 |
| 8 | 816 MHz | 6390.108 | 10.474940 | 956.786 | 0.515900 |
| 8 | 1300.5 MHz | 6273.809 | 10.648440 | 1064.864 | 0.515900 |

### OPT Best Frequency

| gamma | fastest freq | energy-only best freq | latency-constrained best freq | energy saving vs high |
|---:|---:|---:|---:|---:|
| 1 | 1300.5 MHz | 816 MHz | 816 MHz | 7.195% |
| 2 | 1300.5 MHz | 816 MHz | 816 MHz | 12.000% |
| 4 | 816 MHz | 816 MHz | 816 MHz | 8.505% |
| 8 | 1300.5 MHz | 816 MHz | 816 MHz | 10.149% |

### OPT Insight

The OPT run repeats the main pattern on a different model family:

```text
408 MHz is usually slower enough to lose energy/token.
1300.5 MHz is often fastest, but wastes energy/token.
816 MHz is the best energy/token point for every tested gamma.
```

This strengthens the first objective because the "middle frequency beats both max and min" result now appears in both Qwen and OPT runs. It still does not prove that verifier energy-only best frequency changes by gamma.

## Issues Encountered

Three issues occurred and were handled.

1. `--shuffle-conditions --shuffle-runs --resume` on the full p20/r3 run generated plan/trace but did not start measured output after about 15 minutes.
2. The full p20/r3/warmup1 sequential run was reliable but too slow for interactive progress.
3. The first OPT server-restart script used `pkill -f` inside a shell command whose own command line contained the same server string. It killed the shell before starting the experiment. No measured output was produced in that failed attempt.

Actions taken:

```text
1. Confirmed drafter/verifier health with an independent one-prompt probe.
2. Preserved partial/probe files outside the main raw directory.
3. Switched to p10/r1/warmup0 for broad verifier-frequency coverage.
4. Added a smaller drafter-frequency experiment for diversity.
5. Restarted OPT servers with a safer stdin-based kubectl exec path, then reran the OPT matrix successfully.
```

The issue log is stored at:

```text
SPEC/results/phase1/logs/issues.md
```

## Current Claim Strength

Supported:

```text
1. DSD gamma/workload affects accept rate, latency, throughput, and energy/token.
2. Maximum GPU frequency is not energy-optimal for this DSD setup.
3. Minimum GPU frequency is also not energy-optimal for this DSD setup.
4. A mid verifier frequency gives the best energy/token under the current Qwen2.5-0.5B -> 3B setup.
5. Drafter frequency shows gamma-dependent energy/latency tradeoffs in the larger p10/r1 drafter-main sweep.
6. With a larger Qwen2.5-7B verifier, latency-constrained optimal verifier frequency can differ from energy-only optimal frequency.
7. The mid-frequency energy optimum also appears on a different model family, OPT-125M -> OPT-1.3B.
8. With the finer verifier grid, verifier energy-best frequency differs by gamma: 612 MHz for gamma 1/8 and 816 MHz for gamma 2/4.
```

Not yet fully supported:

```text
1. Default hardware governor is worse than a DSD-aware governor.
2. A complete online DSD-aware governor improves over all baselines.
3. The gamma-dependent best-frequency pattern is statistically stable across repeated runs.
```

The initial 3-point verifier sweep should not be overstated as "816 MHz is always the verifier optimum." A more accurate statement is:

```text
In the tested DSD workload, the verifier energy-optimal operating point is an intermediate frequency rather than max/min frequency. When 612 MHz is included, the energy-best verifier frequency differs by gamma.
```

For the drafter-centered research question, the current strongest statement is:

```text
With verifier frequency fixed at the calibrated 816 MHz point, the energy-best drafter frequency changes by gamma: 306 MHz for gamma 1/2, 408 MHz for gamma 4, and 510 MHz for gamma 8.
```

After the 7B validation, a stronger but still careful statement is also supported:

```text
When latency constraints are included, the best verifier frequency can depend on gamma and model size. In the 7B verifier run, gamma=1 required high frequency under a 5% latency constraint, while gamma=2/4/8 preferred 816 MHz.
```

After the OPT validation, another careful statement is supported:

```text
The intermediate-frequency energy optimum is not only observed on Qwen; it also appears on an OPT model pair in the current hardware setup.
```

After the fine-grained verifier sweep, another careful statement is supported:

```text
The frequency grid matters. A coarse 408/816/1300.5 MHz grid missed the 612 MHz point that became energy-best for gamma 1 and gamma 8.
```

## Next Experiments

Recommended next steps:

```text
1. Repeat the drafter-main p10 matrix with runs=3 for statistical stability.
2. Repeat the fine-grained verifier sweep with runs=3 for statistical stability.
3. Add gamma=16 to the drafter-main and fine-grained verifier sweeps.
4. Repeat drafter-main with Qwen2.5-7B verifier fixed at its calibrated frequency.
5. Run prompt-type breakdown to see whether prompt class changes the best frequency.
6. Implement a simple lookup policy only after the calibration evidence is stable.
```

The most valuable next experiment is likely:

```text
Drafter-main repeat
Qwen2.5-0.5B -> Qwen2.5-3B
verifier freq = 816 MHz fixed
drafter freq = 306, 408, 510, 612, 624.75 MHz
gamma = 1, 2, 4, 8
prompts = 10
runs = 3
```

The p10/r1 version has already been run and shows gamma-dependent drafter best frequency. Repeating it improves confidence.
