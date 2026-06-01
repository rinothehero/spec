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
