#!/usr/bin/env bash
set -euo pipefail

# Run this script on the Mas host, not inside the pod.

NS=${NS:-xronos-sjp}
NANO_POD=${NANO_POD:-xronos-client-sjp-68cb785c48-f8r29}
AGX_POD=${AGX_POD:-xronos-client-sjp-68cb785c48-hqwdh}
MAS_POD=${MAS_POD:-xronos-server-sjp-mas-d8764467b-ng4cd}

DRAFTER_ADDR=${DRAFTER_ADDR:-143.0.6.109:50061}
VERIFIER_ADDR=${VERIFIER_ADDR:-143.0.2.248:50062}

REMOTE_SPEC=${REMOTE_SPEC:-/home/xronos/spec}
PROMPTS=${PROMPTS:-$REMOTE_SPEC/experiments/prompts/phase1_20_prompts.jsonl}
PROMPT_LABEL=${PROMPT_LABEL:-p20}
GAMMAS=${GAMMAS:-1,2,4,8}
RUNS=${RUNS:-3}
WARMUP_RUNS=${WARMUP_RUNS:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
SEED_BASE=${SEED_BASE:-20260601}

AGX_GPU_ROOT=${AGX_GPU_ROOT:-/sys/class/devfreq/17000000.gpu}
ORIG_AGX_FREQ=${ORIG_AGX_FREQ:-408000000}

set_agx_freq() {
  local freq=$1
  kubectl exec -n "$NS" "$AGX_POD" -- bash -lc "
    set -euo pipefail
    root=$AGX_GPU_ROOT
    freq=$freq
    cur_min=\$(cat \"\$root/min_freq\")
    cur_max=\$(cat \"\$root/max_freq\")
    if [ \"\$freq\" -gt \"\$cur_max\" ]; then
      printf '%s' \"\$freq\" > \"\$root/max_freq\"
      printf '%s' \"\$freq\" > \"\$root/min_freq\"
    elif [ \"\$freq\" -lt \"\$cur_min\" ]; then
      printf '%s' \"\$freq\" > \"\$root/min_freq\"
      printf '%s' \"\$freq\" > \"\$root/max_freq\"
    else
      printf '%s' \"\$freq\" > \"\$root/min_freq\"
      printf '%s' \"\$freq\" > \"\$root/max_freq\"
    fi
    sleep 1
    printf 'AGX_FREQ cur=%s min=%s max=%s\n' \"\$(cat \"\$root/cur_freq\")\" \"\$(cat \"\$root/min_freq\")\" \"\$(cat \"\$root/max_freq\")\"
  "
}

check_remote_inputs() {
  kubectl exec -n "$NS" "$MAS_POD" -- bash -lc "
    set -euo pipefail
    test -f '$PROMPTS'
    cd '$REMOTE_SPEC'
    python3 - <<'PY'
import json
from pathlib import Path
p=Path('$PROMPTS')
count=0
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    obj=json.loads(line)
    assert 'id' in obj and 'prompt' in obj
    count+=1
print(f'prompts_ok={count}')
PY
  "
}

run_block() {
  local freq=$1
  local label=$2
  local seed=$3
  local out="results/phase1/raw/phase1_fv${label}_qwen25_0p5b_to_3b_g1_2_4_8_${PROMPT_LABEL}_r${RUNS}_w${WARMUP_RUNS}_t${MAX_NEW_TOKENS}.csv"
  local trace="results/phase1/traces/phase1_fv${label}_qwen25_0p5b_to_3b_g1_2_4_8_${PROMPT_LABEL}_r${RUNS}_w${WARMUP_RUNS}_t${MAX_NEW_TOKENS}.trace.jsonl"
  local plan="results/phase1/plans/phase1_fv${label}_qwen25_0p5b_to_3b_g1_2_4_8_${PROMPT_LABEL}_r${RUNS}_w${WARMUP_RUNS}_t${MAX_NEW_TOKENS}.plan.json"
  echo "===== phase1 verifier_gpu_${label} freq=${freq} ====="
  set_agx_freq "$freq"
  kubectl exec -n "$NS" "$MAS_POD" -- bash -lc "
    set -euo pipefail
    cd '$REMOTE_SPEC'
    mkdir -p results/phase1/raw results/phase1/logs results/phase1/traces results/phase1/plans
    python3 -m xronos.infer.spec_driver \
      --drafter-addr '$DRAFTER_ADDR' \
      --verifier-addr '$VERIFIER_ADDR' \
      --tokenizer Qwen/Qwen2.5-3B \
      --prompts-jsonl '$PROMPTS' \
      --gammas '$GAMMAS' \
      --warmup-runs '$WARMUP_RUNS' \
      --runs '$RUNS' \
      --max-new-tokens '$MAX_NEW_TOKENS' \
      --local-files-only \
      --sample-runtime-metadata \
      --seed '$seed' \
      --experiment 'phase1_fv${label}_qwen25_0p5b_to_3b' \
      --plan-out '$plan' \
      --trace-out '$trace' \
      --out '$out'
  "
}

main() {
  check_remote_inputs
  trap 'set_agx_freq "$ORIG_AGX_FREQ" >/dev/null 2>&1 || true' EXIT
  kubectl exec -n "$NS" "$MAS_POD" -- bash -lc "mkdir -p '$REMOTE_SPEC/results/phase1/raw' '$REMOTE_SPEC/results/phase1/logs' '$REMOTE_SPEC/results/phase1/traces' '$REMOTE_SPEC/results/phase1/plans'"
  run_block 408000000 408mhz "$((SEED_BASE + 408))"
  run_block 816000000 816mhz "$((SEED_BASE + 816))"
  run_block 1300500000 1300p5mhz "$((SEED_BASE + 1300))"
  set_agx_freq "$ORIG_AGX_FREQ"
}

main "$@"
