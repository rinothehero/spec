#!/usr/bin/env bash
set -euo pipefail

# Run this script on the Mas host where kubectl can reach the experiment pods.
# The drafter/verifier servers should already be running in the Nano/AGX pods.

NS=${NS:-xronos-sjp}
AGX_POD=${AGX_POD:-xronos-client-sjp-68cb785c48-hqwdh}
MAS_POD=${MAS_POD:-xronos-server-sjp-mas-d8764467b-ng4cd}

DRAFTER_ADDR=${DRAFTER_ADDR:-143.0.6.109:50061}
VERIFIER_ADDR=${VERIFIER_ADDR:-143.0.2.248:50062}

REMOTE_SPEC=${REMOTE_SPEC:-/home/xronos/spec}
PROMPTS_JSONL=${PROMPTS_JSONL:-$REMOTE_SPEC/experiments/prompts/phase1_10_prompts.jsonl}
PROMPT_LABEL=${PROMPT_LABEL:-p10}
TOKENIZER=${TOKENIZER:-Qwen/Qwen2.5-3B}
MODEL_PAIR_LABEL=${MODEL_PAIR_LABEL:-qwen25_0p5b_to_3b}

GAMMAS=${GAMMAS:-1,2,4,8}
DRAFTER_FREQS_HZ=${DRAFTER_FREQS_HZ:-306000000,408000000,510000000,612000000,624750000}
VERIFIER_FREQS_HZ=${VERIFIER_FREQS_HZ:-408000000,612000000,816000000,1300500000}

RUNS=${RUNS:-1}
WARMUP_RUNS=${WARMUP_RUNS:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
SEED_BASE=${SEED_BASE:-20260602}

OUT_ROOT=${OUT_ROOT:-results/phase1/joint_frequency}
EXPERIMENT_PREFIX=${EXPERIMENT_PREFIX:-phase1_joint_frequency}
RUN_SUMMARY=${RUN_SUMMARY:-1}
RESUME=${RESUME:-0}

AGX_GPU_ROOT=${AGX_GPU_ROOT:-/sys/class/devfreq/17000000.gpu}
AGX_FREQ_SETTLE_S=${AGX_FREQ_SETTLE_S:-1}
RESTORE_AGX_FREQ=${RESTORE_AGX_FREQ:-}

# Optional raw driver flags, for example:
#   EXTRA_DRIVER_ARGS="--shuffle-runs --max-start-temp-c 55"
EXTRA_DRIVER_ARGS=${EXTRA_DRIVER_ARGS:-}

trim() {
  local value=$1
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

csv_token() {
  printf '%s' "$1" | tr -d '[:space:]' | tr ',' '_'
}

freq_token() {
  awk -v f="$1" 'BEGIN {
    mhz = f / 1000000.0
    s = sprintf("%.6f", mhz)
    sub(/0+$/, "", s)
    sub(/\.$/, "", s)
    gsub(/\./, "p", s)
    print s
  }'
}

freq_label() {
  printf '%smhz' "$(freq_token "$1")"
}

freq_list_label() {
  local csv=$1
  local prefix=$2
  local result=""
  local sep=""
  local freq
  IFS=',' read -r -a freqs <<< "$csv"
  for freq in "${freqs[@]}"; do
    freq=$(trim "$freq")
    if [[ -z "$freq" ]]; then
      continue
    fi
    result="${result}${sep}$(freq_token "$freq")"
    sep="_"
  done
  printf '%s%s' "$prefix" "$result"
}

read_agx_cur_freq() {
  kubectl exec -n "$NS" "$AGX_POD" -- bash -lc "cat '$AGX_GPU_ROOT/cur_freq'"
}

set_agx_freq() {
  local freq=$1
  kubectl exec -n "$NS" "$AGX_POD" -- bash -lc "
    set -euo pipefail
    root='$AGX_GPU_ROOT'
    freq='$freq'
    test -r \"\$root/min_freq\"
    test -r \"\$root/max_freq\"
    cur_min=\$(cat \"\$root/min_freq\")
    cur_max=\$(cat \"\$root/max_freq\")
    if [ \"\$freq\" -gt \"\$cur_max\" ]; then
      printf '%s\n' \"\$freq\" > \"\$root/max_freq\"
      printf '%s\n' \"\$freq\" > \"\$root/min_freq\"
    elif [ \"\$freq\" -lt \"\$cur_min\" ]; then
      printf '%s\n' \"\$freq\" > \"\$root/min_freq\"
      printf '%s\n' \"\$freq\" > \"\$root/max_freq\"
    else
      printf '%s\n' \"\$freq\" > \"\$root/min_freq\"
      printf '%s\n' \"\$freq\" > \"\$root/max_freq\"
    fi
    sleep '$AGX_FREQ_SETTLE_S'
    printf 'AGX_FREQ cur=%s min=%s max=%s\n' \
      \"\$(cat \"\$root/cur_freq\")\" \
      \"\$(cat \"\$root/min_freq\")\" \
      \"\$(cat \"\$root/max_freq\")\"
  "
}

cleanup() {
  if [[ -n "${RESTORE_AGX_FREQ:-}" ]]; then
    set_agx_freq "$RESTORE_AGX_FREQ" >/dev/null 2>&1 || true
  fi
}

check_remote_inputs() {
  kubectl exec -n "$NS" "$MAS_POD" -- bash -lc "
    set -euo pipefail
    cd '$REMOTE_SPEC'
    test -f '$PROMPTS_JSONL'
    python3 - <<'PY'
import json
from pathlib import Path

p = Path('$PROMPTS_JSONL')
count = 0
for line in p.read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    if isinstance(obj, dict):
        assert 'id' in obj or 'prompt_id' in obj
        assert 'prompt' in obj or 'text' in obj
    elif not isinstance(obj, str):
        raise AssertionError(f'unexpected prompt row type: {type(obj).__name__}')
    count += 1
print(f'prompts_ok={count}')
PY
  "
}

make_remote_dirs() {
  kubectl exec -n "$NS" "$MAS_POD" -- bash -lc "
    set -euo pipefail
    cd '$REMOTE_SPEC'
    mkdir -p '$OUT_ROOT/raw' '$OUT_ROOT/plans' '$OUT_ROOT/traces' '$OUT_ROOT/logs' '$OUT_ROOT/summary'
  "
}

run_verifier_block() {
  local verifier_freq=$1
  local verifier_label
  local fd_label
  local gamma_label
  local experiment
  local base
  local out
  local plan
  local trace
  local log
  local seed
  local resume_arg=""

  verifier_label=$(freq_label "$verifier_freq")
  fd_label=$(freq_list_label "$DRAFTER_FREQS_HZ" "fd")
  gamma_label="g$(csv_token "$GAMMAS")"
  experiment="${EXPERIMENT_PREFIX}_${MODEL_PAIR_LABEL}_${fd_label}_fv${verifier_label}"
  base="${experiment}_${gamma_label}_${PROMPT_LABEL}_r${RUNS}_w${WARMUP_RUNS}_t${MAX_NEW_TOKENS}"
  out="$OUT_ROOT/raw/${base}.csv"
  plan="$OUT_ROOT/plans/${base}.plan.json"
  trace="$OUT_ROOT/traces/${base}.trace.jsonl"
  log="$OUT_ROOT/logs/${base}.log"
  seed=$((SEED_BASE + verifier_freq / 1000000))

  if [[ "$RESUME" == "1" ]]; then
    resume_arg="--resume"
  fi

  echo "===== joint sweep verifier=${verifier_freq}Hz (${verifier_label}) drafter_freqs=${DRAFTER_FREQS_HZ} ====="
  set_agx_freq "$verifier_freq"

  kubectl exec -n "$NS" "$MAS_POD" -- bash -lc "
    set -euo pipefail
    cd '$REMOTE_SPEC'
    mkdir -p '$OUT_ROOT/raw' '$OUT_ROOT/plans' '$OUT_ROOT/traces' '$OUT_ROOT/logs'
    python3 -m xronos.infer.spec_driver \
      --drafter-addr '$DRAFTER_ADDR' \
      --verifier-addr '$VERIFIER_ADDR' \
      --tokenizer '$TOKENIZER' \
      --prompts-jsonl '$PROMPTS_JSONL' \
      --gammas '$GAMMAS' \
      --drafter-freqs-hz '$DRAFTER_FREQS_HZ' \
      --runs '$RUNS' \
      --warmup-runs '$WARMUP_RUNS' \
      --max-new-tokens '$MAX_NEW_TOKENS' \
      --local-files-only \
      --sample-runtime-metadata \
      --seed '$seed' \
      --experiment '$experiment' \
      --plan-out '$plan' \
      --trace-out '$trace' \
      --out '$out' \
      $resume_arg \
      $EXTRA_DRIVER_ARGS \
      2>&1 | tee '$log'
  "
}

summarize_remote() {
  if [[ "$RUN_SUMMARY" != "1" ]]; then
    return
  fi
  kubectl exec -n "$NS" "$MAS_POD" -- bash -lc "
    set -euo pipefail
    cd '$REMOTE_SPEC'
    if [ ! -f experiments/phase1/summarize_joint_frequency_sweep.py ]; then
      echo 'summary script not found in Mas pod; raw/plans/traces are still complete.'
      exit 0
    fi
    python3 experiments/phase1/summarize_joint_frequency_sweep.py \
      --raw-dir '$OUT_ROOT/raw' \
      --out-dir '$OUT_ROOT/summary'
  "
}

main() {
  if [[ -z "${RESTORE_AGX_FREQ:-}" ]]; then
    RESTORE_AGX_FREQ=$(read_agx_cur_freq 2>/dev/null || true)
  fi
  trap cleanup EXIT

  check_remote_inputs
  make_remote_dirs

  local verifier_freq
  IFS=',' read -r -a verifier_freqs <<< "$VERIFIER_FREQS_HZ"
  for verifier_freq in "${verifier_freqs[@]}"; do
    verifier_freq=$(trim "$verifier_freq")
    if [[ -z "$verifier_freq" ]]; then
      continue
    fi
    run_verifier_block "$verifier_freq"
  done

  summarize_remote
}

main "$@"
