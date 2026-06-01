# Phase 1 Experiment Artifacts

This directory holds the reproducible inputs and local analysis helpers for the first research goal:

```text
Show that DSD gamma/workload has its own frequency-energy behavior, and that max/min frequency is not always energy-optimal under a latency constraint.
```

Primary prompt set:

```text
SPEC/experiments/prompts/phase1_20_prompts.jsonl
```

Expected local result layout:

```text
SPEC/results/phase1/raw/      raw CSVs copied back from Mas
SPEC/results/phase1/logs/     run logs and issue notes
SPEC/results/phase1/summary/  merged summaries and best-frequency tables
SPEC/results/phase1/plots/    generated figures
```

The remote Mas pod stores raw runs under:

```text
/home/xronos/spec/results/phase1/raw/
/home/xronos/spec/results/phase1/logs/
```

## Joint Drafter/Verifier Frequency Sweep

When the verifier runs on AGX/Jetson, the driver `--verifier-clocks-mhz` path is
not the right control surface because it targets `nvidia-smi`. Use the remote
shell wrapper to lock the AGX devfreq once per verifier frequency, then let the
driver sweep drafter frequencies through gRPC:

```bash
bash experiments/phase1/run_joint_frequency_remote.sh
```

Default matrix:

```text
prompts = phase1_10_prompts.jsonl
gamma = 1,2,4,8
drafter GPU frequency = 306,408,510,612,624.75 MHz
verifier GPU frequency = 408,612,816,1300.5 MHz
runs = 1
warmup = 0
```

Useful overrides:

```bash
VERIFIER_FREQS_HZ=408000000,816000000 \
DRAFTER_FREQS_HZ=306000000,408000000,510000000,612000000,624750000 \
RUNS=3 \
RESUME=1 \
bash experiments/phase1/run_joint_frequency_remote.sh
```

Outputs are written under:

```text
results/phase1/joint_frequency/raw/
results/phase1/joint_frequency/plans/
results/phase1/joint_frequency/traces/
results/phase1/joint_frequency/logs/
results/phase1/joint_frequency/summary/
```
