# Distributed Speculative Decoding Experiment

This path is separate from the existing Xronos training loop. It implements a
two-device speculative decoding experiment:

- Jetson drafter: runs a small causal LM and returns `gamma` draft tokens.
- GPU verifier: runs a larger causal LM and verifies the draft tokens.
- Driver: sweeps gamma values, coordinates both services, and writes CSV rows.
- Verifier-only baseline: runs the large model autoregressively for comparison.

The implementation is stateful:

- The driver creates one session per gamma/run.
- Drafter and verifier prefill once and keep KV cache for that session.
- Each step runs `Draft -> Verify -> Commit`.
- Each stateful RPC carries the expected committed sequence length, and servers
  reject requests when their local KV-cache length does not match that value.
- Step traces include the same base and committed lengths so the report can
  audit KV-cache state after the run.
- Drafter power is sampled from Jetson INA3221 sysfs for prefill, draft, and
  commit phases.
- Verifier GPU power is sampled with NVML when available, with `nvidia-smi` as
  a fallback, for prefill and verify phases.

Current scope:

- Greedy decoding only.
- The verifier checks gamma draft tokens in one forward pass.
- Frequency can be fixed at server startup for controlled experiments.
- Sampling policies and adaptive gamma controllers are not implemented yet.

## Generate Proto Bindings

```bash
make protobuf
```

Install inference extras on the drafter, verifier, and driver images:

```bash
pip install -e ".[infer]"
```

Build reusable experiment images with git/image provenance embedded:

```bash
make docker-gpu GPU_IMAGE=xronos:gpu
```

For Jetson, use an NVIDIA L4T/PyTorch base image that already matches the
device's JetPack/CUDA version:

```bash
make docker-jetson \
  JETSON_BASE_IMAGE=<l4t-pytorch-base-image-for-your-jetpack> \
  JETSON_IMAGE=xronos:jetson
```

The Makefile passes the current git commit into the image as
`XRONOS_GIT_COMMIT` and records the image name as `XRONOS_IMAGE`; the Dockerfile
also writes matching OCI labels and includes the Kubernetes template used by the
manifest-audit job. Use the same clean git commit and generated proto files for
the Jetson drafter, GPU verifier, baseline, driver, and report jobs. The final
report rejects dirty driver checkouts and role git-commit mismatches.

## Preflight Checks

These commands only inspect the host. They do not start decoding or run the
experiment.

On the Jetson drafter:

```bash
python -m xronos.infer.experiment_doctor \
  --role drafter \
  --model meta-llama/Llama-3.2-1B \
  --device cuda:0 \
  --jetson-gpu-freq-hz 612000000 \
  --jetson-gpu-devfreq-root "" \
  --require-hf-token \
  --require-power \
  --require-frequency-control
```

On the GPU verifier:

```bash
python -m xronos.infer.experiment_doctor \
  --role verifier \
  --model meta-llama/Llama-3.2-3B \
  --device cuda:0 \
  --gpu-index 0 \
  --gpu-clock-mhz 1410 \
  --require-hf-token \
  --require-power \
  --require-frequency-control
```

Use the same verifier `--gpu-index` for power sampling and `--gpu-clock-mhz`
locking. The verifier server passes that index to `nvidia-smi -i`, so the
measured GPU and the fixed-clock GPU stay aligned on multi-GPU desktops.

On the driver image that will launch the sweep:

```bash
python -m xronos.infer.experiment_doctor \
  --role driver \
  --tokenizer meta-llama/Llama-3.2-1B \
  --prompts-jsonl xronos/infer/prompts_example.jsonl \
  --min-prompts 4 \
  --k8s-manifest k8s/spec-decoding.yaml \
  --require-k8s-manifest \
  --require-hf-token
```

This checks driver dependencies, generated-proto runtime versions, tokenizer
reference shape, prompt-set syntax, unique prompt count, and the packaged
Kubernetes experiment template without starting the drafter/verifier services
or decoding. `--require-hf-token` makes the
preflight fail when neither `HF_TOKEN` nor `HUGGING_FACE_HUB_TOKEN` is present,
which is useful for gated models such as Llama.

Check the sweep matrix without contacting the devices:

```bash
python -m xronos.infer.spec_driver \
  --dry-run \
  --tokenizer meta-llama/Llama-3.2-1B \
  --prompts-jsonl xronos/infer/prompts_example.jsonl \
  --gammas 1,2,4,8,16 \
  --drafter-freqs-hz 408000000,612000000,828000000 \
  --verifier-clocks-mhz 810,1050,1410 \
  --idle-baseline-s 5 \
  --idle-baseline-policy run \
  --max-start-temp-c 85 \
  --warmup-runs 1 \
  --runs 5 \
  --max-new-tokens 64 \
  --shuffle-conditions --seed 42 \
  --shuffle-runs \
  --plan-out spec_plan.json
```

Create the matching verifier-only baseline plan without contacting the verifier.
Passing the drafter address/frequency list makes this a two-device baseline:
the verifier generates normally while Jetson drafter idle power is accounted
under the same drafter-frequency levels used by the speculative sweep.

```bash
python -m xronos.infer.verifier_baseline_driver \
  --dry-run \
  --drafter-addr spec-drafter:50061 \
  --tokenizer meta-llama/Llama-3.2-1B \
  --prompts-jsonl xronos/infer/prompts_example.jsonl \
  --drafter-freqs-hz 408000000,612000000,828000000 \
  --verifier-clocks-mhz 810,1050,1410 \
  --idle-baseline-s 5 \
  --idle-baseline-policy run \
  --max-start-temp-c 85 \
  --warmup-runs 1 \
  --runs 5 \
  --max-new-tokens 64 \
  --shuffle-conditions --seed 42 \
  --shuffle-runs \
  --plan-out verifier_baseline_plan.json
```

Audit both plan JSONs before launching a long run:

```bash
python -m xronos.infer.plan_audit \
  --plan spec_plan.json verifier_baseline_plan.json \
  --min-runs 5 \
  --min-prompts 4 \
  --min-gammas 5 \
  --summary-energy-key mean_drafter_active_energy_mj_per_token \
  --paired-energy-key system_active_energy_mj_per_generated_token \
  --require-two-device-boundary \
  --out plan_audit.json
```

This fails early if the design lacks `gamma=1`, has too few gamma values,
uses too few repeats or unique prompts, omits randomized run schedules, misses
verifier-only baseline coverage, has mismatched baseline/spec repeat counts, or
uses an idle-baseline policy that cannot support active-energy claims. It also
checks that each prompt has the full declared gamma x drafter-frequency x
verifier-clock grid, so interaction-analysis gaps are caught before a long
device run. When the verifier-only baseline is configured as a two-device
idle-drafter baseline, the audit/report also require the baseline to cover the
same Jetson `drafter_freq_hz` levels as the speculative plan. With
`--require-two-device-boundary`, the audit also fails if the baseline is only a
GPU-side verifier-only baseline, because that would compare different system
power boundaries. The final report can require this audit and compares its
`plan_design_sha256` fingerprints against the measured run plans. Keep
design-affecting arguments such as `--max-start-temp-c`, prompts, gamma values,
frequencies, warmups, and repeat counts identical between the dry-run plan jobs
and the measured jobs.

Use a prompt set instead of a single prompt when you want a less prompt-specific
measurement. Each JSONL line can be a string or an object with `id` and
`prompt`:

```bash
python -m xronos.infer.spec_driver \
  --dry-run \
  --prompts-jsonl xronos/infer/prompts_example.jsonl \
  --gammas 1,2,4,8,16 \
  --runs 5 \
  --shuffle-runs \
  --plan-out spec_plan.json
```

Run pure local checks that do not import torch/grpc and do not contact devices:

```bash
python -m xronos.infer.self_test
```

## Start Drafter On Jetson

```bash
python -m xronos.infer.drafter_server \
  --model meta-llama/Llama-3.2-1B \
  --host "[::]" --port 50061 \
  --device cuda:0 --dtype float16 \
  --power-interval 0.01 \
  --jetson-gpu-freq-hz 612000000 \
  --jetson-gpu-devfreq-root ""
```

Leave `--jetson-gpu-devfreq-root` empty to auto-discover the board's GPU
devfreq path. Set it explicitly on Jetson boards whose GPU path is not exposed
under the usual `/sys/class/devfreq/*gpu*` locations.

## Start Verifier On GPU Desktop

```bash
python -m xronos.infer.verifier_server \
  --model meta-llama/Llama-3.2-3B \
  --host "[::]" --port 50062 \
  --device cuda:0 --dtype float16 \
  --power-interval 0.01 \
  --gpu-index 0 \
  --gpu-clock-mhz 1410
```

## Run Gamma Sweep

Run this from any machine that can reach both services:

```bash
python -m xronos.infer.spec_driver \
  --drafter-addr spec-drafter:50061 \
  --verifier-addr spec-verifier:50062 \
  --tokenizer meta-llama/Llama-3.2-1B \
  --prompts-jsonl xronos/infer/prompts_example.jsonl \
  --gammas 1,2,4,8,16 \
  --max-new-tokens 64 \
  --warmup-runs 1 \
  --runs 3 \
  --shuffle-runs \
  --sample-runtime-metadata \
  --max-start-temp-c 85 \
  --out spec_gamma_sweep.csv \
  --resume
```

Measured rows are checkpointed to `--out` after every completed measured run
with an atomic replace, so a failed long sweep should still leave a readable CSV
through the last completed session. Add `--resume` when restarting the same
planned sweep; the driver verifies the existing `plan_sha256` and skips already
completed measurement orders. Warmup runs are not written to the CSV.

## Run Verifier-Only Baseline

This is the autoregressive baseline for the verifier model. It uses the same
prompt set, token budget, warmups, repeats, and verifier clock sweep. If
`--drafter-addr` is set, it also locks/sweeps the Jetson drafter frequency and
adds Jetson idle energy to `system_total_energy_mj`, while keeping drafter
active energy at zero. That gives a fair two-device system boundary without
running speculative verification:

```bash
python -m xronos.infer.verifier_baseline_driver \
  --drafter-addr spec-drafter:50061 \
  --verifier-addr spec-verifier:50062 \
  --tokenizer meta-llama/Llama-3.2-1B \
  --prompts-jsonl xronos/infer/prompts_example.jsonl \
  --drafter-freqs-hz 408000000,612000000,828000000 \
  --verifier-clocks-mhz 810,1050,1410 \
  --max-new-tokens 64 \
  --stop-token-ids "" \
  --idle-baseline-s 5 \
  --idle-baseline-policy run \
  --shuffle-conditions --seed 42 \
  --shuffle-runs \
  --sample-runtime-metadata \
  --max-start-temp-c 85 \
  --warmup-runs 1 \
  --runs 5 \
  --out verifier_baseline.csv \
  --resume \
  --plan-out verifier_baseline_plan.json \
  --trace-out verifier_baseline_trace.jsonl
```

## Run Gamma And Frequency Sweep

This is the experiment shape that matches the proposal's co-optimization goal:

```bash
python -m xronos.infer.spec_driver \
  --drafter-addr spec-drafter:50061 \
  --verifier-addr spec-verifier:50062 \
  --tokenizer meta-llama/Llama-3.2-1B \
  --prompts-jsonl xronos/infer/prompts_example.jsonl \
  --gammas 1,2,4,8,16 \
  --drafter-freqs-hz 408000000,612000000,828000000 \
  --verifier-clocks-mhz 810,1050,1410 \
  --max-new-tokens 64 \
  --idle-baseline-s 5 \
  --idle-baseline-policy run \
  --shuffle-conditions --seed 42 \
  --shuffle-runs \
  --sample-runtime-metadata \
  --max-start-temp-c 85 \
  --warmup-runs 1 \
  --runs 5 \
  --out spec_gamma_freq_sweep.csv \
  --resume \
  --plan-out spec_gamma_freq_plan.json \
  --trace-out spec_gamma_freq_trace.jsonl
```

The CSV is long-format over power rails. Key columns:

- `gamma`
- `result_schema_version`
- `algorithm_version`
- `plan_sha256`
- `algorithm`
- `decoding_mode`
- `prompt_id`
- `prompt_sha256`
- `prompt_set_sha256`
- `prompt_tokens`
- `prompt_token_sha256`
- `max_new_tokens`
- `stop_token_policy`
- `stop_token_ids`
- `drafter_freq_hz`
- `verifier_clock_mhz`
- `drafter_jetson_gpu_freq_hz`
- `verifier_gpu_clock_mhz`
- `drafter_frequency_lock_ok`
- `verifier_frequency_lock_ok`
- `drafter_power_interval_s`
- `verifier_power_interval_s`
- `driver_result_schema_version`
- `driver_spec_rpc_schema_version`
- `drafter_spec_rpc_schema_version`
- `verifier_spec_rpc_schema_version`
- `drafter_primary_power_rail`
- `verifier_primary_power_rail`
- `system_primary_power_rails`
- `idle_baseline_policy`
- `drafter_model_bos_token_id`
- `drafter_model_eos_token_id`
- `drafter_model_pad_token_id`
- `verifier_model_bos_token_id`
- `verifier_model_eos_token_id`
- `verifier_model_pad_token_id`
- `drafter_runtime_fingerprint`
- `verifier_runtime_fingerprint`
- `tokenizer_name_or_path`
- `tokenizer_vocab_size`
- `tokenizer_base_vocab_size`
- `drafter_pod_name`
- `verifier_pod_name`
- `drafter_node_name`
- `verifier_node_name`
- `driver_hostname`
- `driver_pod_name`
- `driver_node_name`
- `driver_git_commit`
- `driver_git_dirty`
- `driver_xronos_git_commit`
- `driver_xronos_image`
- `drafter_xronos_git_commit`
- `verifier_xronos_git_commit`
- `driver_command_sha256`
- `driver_plan_sha256`
- `rail`
- `generated_tokens`
- `stop_reason`
- `output_token_sha256`
- `draft_tokens`
- `accepted_draft_tokens`
- `accept_rate`
- `drafter_init_latency_ms`
- `drafter_draft_latency_ms`
- `drafter_commit_latency_ms`
- `verifier_latency_ms`
- `client_rpc_latency_ms`
- `server_compute_latency_ms`
- `estimated_rpc_overhead_ms`
- `rpc_request_bytes`
- `rpc_response_bytes`
- `rpc_total_bytes`
- `rpc_bytes_per_generated_token`
- `wall_latency_ms`
- `drafter_prefill_energy_mj`
- `drafter_draft_energy_mj`
- `drafter_commit_energy_mj`
- `drafter_active_energy_mj`
- `system_active_energy_mj`
- `system_active_energy_mj_per_generated_token`
- `verifier_prefill_energy_mj`
- `verifier_verify_energy_mj`
- `role_total_energy_mj`
- `drafter_prefill_total_energy_mj`
- `drafter_draft_total_energy_mj`
- `drafter_commit_total_energy_mj`
- `verifier_prefill_total_energy_mj`
- `verifier_verify_total_energy_mj`
- `drafter_total_energy_mj`
- `verifier_total_energy_mj`
- `system_total_energy_mj`
- `system_total_energy_mj_per_generated_token`
- `idle_baseline_s`
- `drafter_idle_power_mw`
- `verifier_idle_power_mw`
- `system_idle_power_mw`
- `drafter_active_energy_mj`
- `verifier_active_energy_mj`
- `system_active_energy_mj`
- `system_active_energy_mj_per_generated_token`
- `n_power_samples`
- `drafter_prefill_power_samples`
- `drafter_draft_power_samples`
- `drafter_commit_power_samples`
- `verifier_prefill_power_samples`
- `verifier_verify_power_samples`
- `verifier_decode_power_samples`
- `drafter_power_samples`
- `verifier_power_samples`
- `system_energy_complete`

For Jetson INA3221 measurements, `tot_power` is the drafter-side primary rail
used by the analysis. It uses an input/module rail such as `VDD_IN` when that
rail is exposed by the board; otherwise it falls back to `sum_rails_power`, the
sum of the visible non-input rails. The raw per-rail rows are still preserved so
the final plots can state exactly which Jetson rail was used.

`spec_gamma_freq_plan.json` records the exact sweep matrix, command, git commit,
schema version, algorithm version, and runtime metadata. Each plan also records
a stable `plan_design_sha256` that excludes volatile command/time/host metadata,
so a dry-run plan audit can be compared with the later measured run plan. The
raw CSV keeps both full-plan and design-plan hashes plus driver/server
provenance columns so a detached CSV can still be traced back to the code,
image, and Kubernetes node/pod that produced it.
`spec_gamma_freq_trace.jsonl` records one JSON object per decode step with draft
tokens, accepted tokens, replacement tokens, server-side phase latencies,
client-side RPC latencies, and phase power summaries. It also records a final
run-summary event with generated-token count and output-token hash, which the
report can compare against the raw CSV. Use the trace when a run has an unusual
accept rate, latency, or energy value.

`--idle-baseline-s` is optional but recommended for energy experiments. It
samples idle power and writes both raw total energy and active-minus-idle
energy. Use raw energy for whole-system cost and active energy when you want to
isolate decode work from background idle power. Use `--idle-baseline-policy run`
for claim-ready active-energy experiments; this samples idle power immediately
before every measured run so slow background-power drift is less likely to be
mistaken for a gamma effect. The lighter `condition` policy samples once per
condition and is mainly for quick debugging.
Very short phases can be shorter than the sampler interval; in that case the
sampler uses the nearest non-empty power sample so the phase is still marked
with measured power instead of silently becoming an incomplete energy row.

`--shuffle-conditions --seed N` randomizes the order of prompt/frequency/gamma
conditions while keeping the order reproducible in the plan JSON. This reduces
thermal and time-order bias in the sweep.

`--shuffle-runs` additionally randomizes the measured repeat schedule across
conditions. The plan records the full `measurement_schedule`, and each CSV row
gets a 1-based `measurement_order`, so postprocessing can detect missing or
duplicated scheduled measurements. It also verifies that each row's
`measurement_order` matches the exact planned prompt, gamma, frequency, and run
index for that order, so shuffled CSV rows cannot be silently interpreted as a
different schedule. The final experiment report requires this run-level
randomization and a recorded seed whenever a plan has more than one measured
session. It also rejects schedules that are still sorted by condition order even
when `shuffle_runs` is set, because blocked condition order can make thermal
drift look like a gamma effect.

`--warmup-runs` defaults to `1` and should stay positive for real
measurements. Warmup sessions contact the same services before measured rows are
written, which keeps first-use model/KV/cache/power-sampler effects out of the
CSV. The plan records a `warmup_schedule`; when `--shuffle-runs` is enabled,
the warmup schedule is randomized with the same seed namespace instead of being
blocked by condition order. The final experiment report rejects measured plans
with missing or zero warmup runs, missing warmup schedules, or blocked warmup
schedules.

`--sample-runtime-metadata` refreshes server health metadata before each
measured run and records temperature/throttling columns such as
`drafter_runtime_temp_c`, `verifier_runtime_temp_c`, and
`verifier_nvidia_throttle_active`. Use these columns to flag runs where thermal
state changed enough to confound gamma comparisons.

`--max-start-temp-c T` is a stronger guard: before each measured run, the driver
refreshes runtime metadata and waits until the recorded maximum runtime
temperature is at or below `T`. If the device does not cool within
`--thermal-wait-timeout-s`, the run fails instead of mixing hot-device behavior
into the gamma sweep.

Each measured repeat is written with a 1-based `run` index. Validation rejects
duplicate or missing run indices for a planned condition, so `runs=5` means
exactly runs `1..5` must exist, not just any five rows.

For the proposal's first experiment, use the gamma-effect summary to inspect how
Jetson drafter energy changes with gamma. Keep frequency fixed unless the
experiment is explicitly sweeping frequency, and include `gamma=1` so every
gamma-effect row has a natural baseline:

```bash
python -m xronos.infer.analyze_gamma_effect \
  --input spec_gamma_freq_sweep.csv \
  --out gamma_effect_summary.csv
```

This writes one row per gamma/frequency/model condition with drafter total
energy per generated token, drafter draft energy per generated token, drafter
draft energy per draft token, ratios against the smallest measured gamma,
prompt-overlap checks, prompt-paired energy changes against the smallest
measured gamma, prompt-paired bootstrap 95% confidence intervals, sign-test
p-values, and log2(gamma) trend slopes.

For the full proposal, group by
`(gamma, drafter_freq_hz, verifier_clock_mhz, decoding_mode)` and compare
`system_total_energy_mj` under the same prompt, model pair, max token count,
and warmup/run count.

Validate that the raw CSVs cover the planned sweep before summarizing:

```bash
python -m xronos.infer.validate_results \
  --plan spec_gamma_freq_plan.json verifier_baseline_plan.json \
  --input spec_gamma_freq_sweep.csv verifier_baseline.csv \
  --min-power-samples 3 \
  --out validation_report.json
```

This checks missing/extra conditions, token budget, measured run counts,
complete energy rows, matching stop-token configuration, valid generated-token
counts, stop reasons for early termination, and idle-baseline samples when the plan requested
`--idle-baseline-s`. `--min-power-samples` rejects sessions where any required
active phase, such as drafter draft or verifier verify, is too sparsely sampled
to support an energy claim. A run with `generated_tokens < max_new_tokens` is
accepted only when `stop_reason` starts with `eos` or `stop`; otherwise it is
treated as an invalid early stop.

Summarize raw runs:

```bash
python -m xronos.infer.analyze_spec_results \
  --input spec_gamma_freq_sweep.csv verifier_baseline.csv \
  --out combined_summary.csv
```

The summary command excludes runs where either drafter or verifier power was
not sampled for speculative runs. The verifier-only baseline writes compatible
CSV columns and can be summarized with the same command. Use
`--allow-incomplete-energy` only for debugging. Summary rows keep
`max_new_tokens`, `stop_token_policy`, `stop_token_ids`, `prompt_set_sha256`,
and model metadata, and include standard deviation, standard error, and an
approximate 95% confidence interval for the main energy, throughput, and
accept-rate metrics.

Compare speculative configurations against the verifier-only baseline with the
same verifier clock, decoding mode, token budget, prompt set, and verifier
model. The comparison key also includes `stop_token_policy` and
`stop_token_ids`, so an EOS-default run and an explicit custom stop-token run
are not silently paired:

```bash
python -m xronos.infer.compare_to_baseline \
  --input combined_summary.csv \
  --energy-key mean_active_system_energy_mj_per_token \
  --out baseline_comparison.csv
```

Use this for system-level claims like "gamma 4 at this frequency saved N%
active system energy per token versus verifier-only decoding." Drafter-only
gamma claims should use the gamma-effect report and the
`mean_drafter_active_energy_mj_per_token` summary metric.

For a stricter prompt-paired comparison, compare raw speculative and baseline
runs prompt by prompt before averaging:

```bash
python -m xronos.infer.paired_prompt_compare \
  --input spec_gamma_freq_sweep.csv verifier_baseline.csv \
  --energy-key system_active_energy_mj_per_generated_token \
  --out paired_prompt_summary.csv \
  --pairs-out paired_prompt_rows.csv \
  --unpaired-out unpaired_prompt_rows.csv \
  --bootstrap-samples 1000
```

Use this when making the main experimental claim, because each speculative
measurement is paired with the verifier-only result for the same prompt,
prompt hash, token budget, verifier model, verifier clock, and decoding mode.
When the baseline includes Jetson idle power, pairing is also exact on
`drafter_freq_hz`; otherwise it falls back to the verifier-only baseline that
has no drafter-frequency level.
For greedy decoding, the paired output also reports `mean_output_token_match`.
It should be `1.0`; otherwise the speculative run did not reproduce the
verifier-only token sequence and should not be used for energy claims. The
paired summary includes mean and median savings, normal and bootstrap 95%
confidence intervals, and an exact sign-test p-value over prompt-level savings.

Generate a complete postprocess bundle from the plan files and raw CSVs:

```bash
python -m xronos.infer.experiment_report \
  --plan spec_gamma_freq_plan.json verifier_baseline_plan.json \
  --input spec_gamma_freq_sweep.csv verifier_baseline.csv \
  --doctor-json drafter_doctor.json verifier_doctor.json driver_doctor.json \
  --plan-audit-json plan_audit.json \
  --network-json network_probe.json \
  --k8s-manifest-audit-json k8s_manifest_audit.json \
  --trace-jsonl spec_trace.jsonl verifier_baseline_trace.jsonl \
  --require-doctor \
  --require-driver-doctor \
  --require-plan-audit \
  --require-k8s-manifest-audit \
  --require-network-probe \
  --require-trace \
  --require-interaction-analysis \
  --require-claim-readiness \
  --require-two-device-boundary \
  --summary-energy-key mean_drafter_active_energy_mj_per_token \
  --paired-energy-key system_active_energy_mj_per_generated_token \
  --min-runs 5 \
  --min-prompts 4 \
  --min-gammas 5 \
  --min-power-samples 3 \
  --paired-bootstrap-samples 1000 \
  --max-runtime-temp-c 85 \
  --max-energy-cv 0.50 \
  --max-latency-cv 0.50 \
  --fail-on-throttle \
  --out-dir report_gamma_freq
```

This writes validation, pre-run plan-audit status, Kubernetes manifest-audit
input tracking, doctor-check status, provenance checks, model-role checks,
plan-hash integrity, input-tokenization
checks, network probe status, timing/accounting checks, summary, prompt-paired
comparison, runtime temperature/throttling status, Pareto configs, and a short
`REPORT.md` into one folder. It also writes `system_boundary_report.json`,
which verifies that speculative rows use `two_device_active` and their paired
baseline rows use `two_device_idle_drafter`. The report first checks
that the baseline plan covers every speculative prompt/verifier-clock/token-budget condition. If
active energy metrics are requested, it also requires an idle-baseline plan. For
claim-ready active-energy reports, that plan must use
`idle_baseline_policy=run`, which means idle power was sampled immediately
before each measured run rather than once and reused across the sweep. For
reproducibility, it rejects plan files whose recorded `plan_sha256` no longer
matches the plan content, result rows whose `plan_sha256` does not match the
loaded plan, result CSVs with missing or mixed result schema versions,
mismatched plan/result schema versions, missing algorithm versions, mixed
algorithm versions for the same algorithm, unknown algorithm labels, or
driver/drafter/verifier RPC schema mismatches. For
speculative decoding compatibility, it records drafter/verifier model config
metadata from server health checks and fails if both servers report incompatible
vocabulary sizes. It also records the driver tokenizer metadata and fails when
the tokenizer vocabulary size does not match the drafter/verifier model
vocabulary sizes, when tokenizer metadata is mixed across result files, or when
driver tokenizer bos/eos/pad ids conflict with reported model config token ids.
This catches the most obvious "same token id means different text" or "driver
can emit token ids the model cannot embed" setup errors before energy claims
are made. The server health metadata also includes model parameter counts, and
the report fails speculative energy claims unless the drafter reports fewer
parameters than the verifier. It records `prompt_token_sha256` for the tokenized
context sent to both services and fails if the same prompt hash is tokenized
differently across speculative and baseline rows. For
gamma-effect claims, it requires at least one multi-gamma configuration with the
same prompts present at every gamma value, `gamma=1` as the baseline, enough
distinct gamma values for `--min-gammas`, enough prompt-paired gamma rows for
`--min-prompts`, and prompt-paired confidence
intervals plus sign-test
statistics for every non-baseline gamma row. It also requires valid
log2-gamma trend slope and correlation statistics for drafter total, draft, and
active energy when active-energy claims are enabled. The default summary metric is
`mean_drafter_active_energy_mj_per_token`, so best-config and Pareto selection
follow the Jetson drafter energy question rather than the combined
drafter-plus-verifier energy. With `--max-energy-cv` and `--max-latency-cv`,
it also rejects summary groups whose repeat measurements are too noisy to
support a stable claim. It also rejects mixed plan schema versions, mixed
tokenizers between the speculative and baseline plans, and plans that only test
one gamma value. With `--require-doctor`, it fails unless drafter and verifier
preflight reports are present and contain no failed checks; with
`--require-driver-doctor`, it also requires the driver preflight report. For each
prompt, drafter frequency, verifier clock, decoding mode, and token budget, the
speculative plan must contain the same gamma set and the same measured repeat
count at every gamma. The matching verifier-only baseline must use the same
measured repeat and warmup counts for each paired prompt/verifier-clock/token
budget condition. Each plan must also record enough unique prompt hashes for
`--min-prompts`; duplicate prompt texts are rejected so repeated copies of one
prompt cannot be counted as prompt diversity. It fails if recorded speculative
output hashes disagree
with the matching verifier-only baseline, if paired rows lack output hashes, if
greedy runs produce multiple output hashes for the same prompt pair, if any
planned measured run index is missing or duplicated, if any
planned measurement order is missing or duplicated, if any requested
drafter/verifier frequency differs from the server-reported applied frequency,
if frequency locking reports failure, if server power-sampling interval metadata
is missing/invalid/mixed, if role git commit metadata is missing/mixed, if the
driver checkout was dirty, if the drafter/verifier model names are missing or
identical, if drafter/verifier parameter counts are missing or the drafter is
not smaller than the verifier, if prompt-tokenization metadata is missing or
inconsistent, if drafter/verifier host identity is missing, if drafter and
verifier report the same host identity, if measured runs were not randomized
with a recorded seed or the recorded
schedule is still condition-blocked, if RPC payload byte metrics are missing or
invalid, if throughput does not match generated tokens divided by wall latency,
if required trace JSONL files are missing, missing final run summaries,
orphaned, or inconsistent with the raw CSV session rows,
if RPC/server timing accounting is inconsistent, if speculative token accounting
is inconsistent, if gamma-effect statistics are missing or malformed, if primary
power-rail metadata is missing or the selected long-format rail does not match
the configured primary rail, if role/system energy sums or per-token energy
columns disagree, if complete measured sessions contain nonpositive primary
energy or idle-power signals, if any runtime temperature exceeds
`--max-runtime-temp-c` when that limit is provided, if throttling is observed
with `--fail-on-throttle`, if plan hashes are missing or mismatched between the
plan JSONs and result CSV rows, if any speculative prompt row lacks a matching
verifier-only baseline, or if no speculative config satisfies the requested
run/prompt/latency constraints. Use this as the main artifact to inspect before
writing experimental claims.

The same report bundle also writes dependency-free SVG figures under
`report_gamma_freq/figures/`: drafter energy vs gamma, accept rate vs gamma,
prompt-paired energy savings, and the energy/throughput tradeoff. The
`plot_manifest.json` file records which plots were produced or skipped.
The bundle also writes `artifact_manifest.json` with SHA256 hashes for the
input plan/result/doctor files and every generated report artifact, so a figure
or summary can be traced back to the exact raw files used to create it. The
audit also fails if a required raw input or report output exists on disk but is
missing from the manifest.

Audit the final result folder before archiving or copying it into a paper
artifact:

```bash
python -m xronos.infer.artifact_audit \
  --results-dir /results \
  --report-dir /results/report_gamma_freq \
  --require-report-ok \
  --out /results/artifact_audit.json
```

This checks that the root result files, pre-run plan audit, Kubernetes manifest
audit, report bundle files, figure files, JSON reports, raw traces, required
manifest entries, and manifest-recorded SHA256 hashes for both inputs and
outputs are all present and consistent.

Select the lowest-energy feasible configuration and write the Pareto front:

```bash
python -m xronos.infer.select_best_config \
  --input combined_summary.csv \
  --energy-key mean_drafter_active_energy_mj_per_token \
  --min-tokens-per-s 10 \
  --min-runs 5 \
  --min-prompts 4 \
  --out pareto_configs.csv \
  --report-json optimization_report.json \
  --policy-out gamma_frequency_policy.csv
```

This step is where an experimental claim such as "best `(gamma, f_draft,
f_verify)` under a throughput constraint" should come from. The optimization
report records the constrained joint best config, the best feasible `gamma=1`
reference config, and the energy/throughput/latency ratios between them. It
also records the runner-up config, percent energy margin over the runner-up,
and whether the mean-energy gap is larger than the two configs' reported 95%
CI widths, so a tiny numerical win is visible instead of silently becoming a
strong claim.
The full `experiment_report` always also writes
`system_optimization_report.json`, using active system energy when idle
baselines are present, so the proposal's total-energy objective is kept
separate from the Jetson-drafter energy view.
It also writes `gamma_policy_report.json` and `gamma_frequency_policy.csv`,
which map each measured gamma to the best feasible `(f_draft, f_verify)` pair
under the selected system-energy metric. That is the table to use when gamma is
chosen adaptively by another controller. Policy rows also include the
runner-up frequency pair and energy margin for that gamma when another measured
pair exists.
For the full gamma/frequency experiment, the report also writes
`interaction_report.json`. With `--require-interaction-analysis`, it requires a
complete gamma x drafter-frequency x verifier-clock grid and records whether
the best verifier clock changes with gamma, plus how far a marginal
factor-by-factor choice is from the true joint best config.
Finally, `claim_readiness_report.json` groups the evidence into concrete claim
types: drafter gamma-energy, system energy vs verifier-only baseline, joint
system-energy optimization, gamma/frequency interaction, and adaptive gamma
frequency policy. Use that file to decide which statements are supported by the
current artifact bundle and which are still blocked. Use
`--require-claim-readiness` for the final paper run so `report_ok=1` means all
proposal claim categories are ready, not just that individual checks produced
files.

## Network Probe

Run this after the drafter/verifier services are up and before the measured
sweep. It records gRPC health-check round-trip times to both services without
running model decode:

```bash
python -m xronos.infer.network_probe \
  --drafter-addr spec-drafter:50061 \
  --verifier-addr spec-verifier:50062 \
  --samples 20 \
  --warmup-samples 3 \
  --out network_probe.json
```

Pass `network_probe.json` to `experiment_report` with `--require-network-probe`
when making distributed-system claims, so a bad network day is visible in the
artifact bundle instead of being mistaken for an algorithm effect.

## Kubernetes Sketch

Use `k8s/spec-decoding.yaml` as a starting template. It assumes separate
`xronos:jetson` and `xronos:gpu` images because the Jetson and desktop GPU
usually need different CUDA/PyTorch base images. It writes verifier-only
baseline results, speculative results, preflight doctor reports, and a
postprocess report under
`/results`, backed by a PVC named `spec-results`. It also mounts a second PVC
named `hf-cache` at `/models` and sets `HF_HOME=/models/huggingface`, so model
and tokenizer downloads are cached across jobs. For gated Hugging Face models,
create the optional token secret before resuming the doctor/server jobs:

```bash
kubectl create secret generic hf-token \
  -n xronos-spec \
  --from-literal=token="$HF_TOKEN"
```

Label the Jetson node and GPU desktop node before applying:

```bash
kubectl label node <jetson-node> xronos-role=jetson-drafter
kubectl label node <gpu-node> xronos-role=gpu-verifier
kubectl apply -f k8s/spec-decoding.yaml
```

All Jobs in the manifest are suspended by default so nothing runs immediately
after `kubectl apply`. The verifier container is privileged because verifier GPU
clock locking uses `nvidia-smi -lgc`; remove that only if you are not controlling
the verifier clock. Jobs that use the `xronos:gpu` image are pinned to the
`gpu-verifier` node label so an x86 desktop image is not accidentally scheduled
onto the Jetson. The drafter and verifier deployments use TCP startup,
readiness, and liveness probes. Because each server opens its port only after
model loading finishes, the Service should not route driver traffic to a pod
while the model is still loading; the driver also retries health checks at
startup as an extra guard. Both the Jetson drafter and GPU verifier pods request
`nvidia.com/gpu: 1`; keep that resource request if your cluster uses the NVIDIA
device plugin so the container receives the intended accelerator. Edit the image
names, model IDs, prompt count, gamma/frequency lists, `MAX_START_TEMP_C`, and
`EXPERIMENT_RUNS` consistently across the plan and measured jobs for your
cluster. Each Job also carries `xronos.run-order`, `xronos.run-after`,
`xronos.produces`, and `xronos.consumes` annotations. The Kubernetes manifest
audit checks those annotations against the command paths, so the result bundle
keeps a machine-readable runbook instead of relying only on this README. Render
that runbook before operating the cluster:

```bash
python -m xronos.infer.k8s_runbook \
  --manifest k8s/spec-decoding.yaml \
  --out k8s_runbook.json \
  --markdown-out k8s_runbook.md
```

This command only writes command lists; it never executes `kubectl`. Then resume
one Job at a time:

```bash
# 1. Check the Jetson drafter node prerequisites.
kubectl patch job drafter-doctor -n xronos-spec --type merge -p '{"spec":{"suspend":false}}'

# 2. Check the GPU verifier node prerequisites.
kubectl patch job verifier-doctor -n xronos-spec --type merge -p '{"spec":{"suspend":false}}'

# 3. Check the driver image, tokenizer reference, and prompt set.
kubectl patch job driver-doctor -n xronos-spec --type merge -p '{"spec":{"suspend":false}}'

# 4. Audit that the Kubernetes template preserves the planned design.
kubectl patch job k8s-manifest-audit -n xronos-spec --type merge -p '{"spec":{"suspend":false}}'

# 5. Record driver-to-service network health.
kubectl patch job network-probe -n xronos-spec --type merge -p '{"spec":{"suspend":false}}'

# 6. Generate the verifier-only baseline plan without running inference.
kubectl patch job verifier-baseline-plan -n xronos-spec --type merge -p '{"spec":{"suspend":false}}'

# 7. Generate the speculative sweep plan without running inference.
kubectl patch job spec-plan -n xronos-spec --type merge -p '{"spec":{"suspend":false}}'

# 8. Fail early if the planned design is not claim-ready.
kubectl patch job plan-audit -n xronos-spec --type merge -p '{"spec":{"suspend":false}}'

# 9. Run the verifier-only baseline and wait for completion.
kubectl patch job verifier-baseline -n xronos-spec --type merge -p '{"spec":{"suspend":false}}'

# 10. Run the speculative gamma/frequency sweep and wait for completion.
kubectl patch job spec-driver -n xronos-spec --type merge -p '{"spec":{"suspend":false}}'

# 11. Generate validation, summaries, paired comparison, Pareto configs, and REPORT.md.
kubectl patch job spec-report -n xronos-spec --type merge -p '{"spec":{"suspend":false}}'

# 12. Audit that the final result folder is complete and hash-consistent.
kubectl patch job artifact-audit -n xronos-spec --type merge -p '{"spec":{"suspend":false}}'
```
