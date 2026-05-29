import argparse
import csv
import json
import math
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_ENERGY_KEY = "mean_drafter_active_energy_mj_per_token"
THROUGHPUT_KEY = "mean_tokens_per_s"
LATENCY_KEY = "mean_wall_latency_ms"
CONFIG_KEYS = [
    "algorithm",
    "gamma",
    "drafter_freq_hz",
    "verifier_clock_mhz",
    "decoding_mode",
    "max_new_tokens",
    "stop_token_policy",
    "stop_token_ids",
    "prompt_set_sha256",
]


def _float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        parsed = float(value) if value not in ("", None) else 0.0
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _int(row: Dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value not in ("", None) else 0


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_csvs(paths: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["_source"] = path
                rows.append(row)
    return rows


def passes_constraints(
    row: Dict[str, str],
    algorithms: Optional[List[str]],
    min_tokens_per_s: float,
    max_wall_latency_ms: Optional[float],
    min_runs: int,
    min_prompts: int,
    energy_key: str = DEFAULT_ENERGY_KEY,
) -> bool:
    if algorithms and row.get("algorithm", "") not in algorithms:
        return False
    if _float(row, THROUGHPUT_KEY) < min_tokens_per_s:
        return False
    if max_wall_latency_ms is not None and _float(row, LATENCY_KEY) > max_wall_latency_ms:
        return False
    if _int(row, "runs") < min_runs:
        return False
    if _int(row, "prompts") < min_prompts:
        return False
    if _float(row, energy_key) <= 0:
        return False
    return True


def dominated(
    candidate: Dict[str, str],
    other: Dict[str, str],
    energy_key: str = DEFAULT_ENERGY_KEY,
) -> bool:
    candidate_energy = _float(candidate, energy_key)
    other_energy = _float(other, energy_key)
    candidate_throughput = _float(candidate, THROUGHPUT_KEY)
    other_throughput = _float(other, THROUGHPUT_KEY)

    no_worse = (
        other_energy <= candidate_energy
        and other_throughput >= candidate_throughput
    )
    strictly_better = (
        other_energy < candidate_energy
        or other_throughput > candidate_throughput
    )
    return no_worse and strictly_better


def pareto_front(
    rows: Iterable[Dict[str, str]],
    energy_key: str = DEFAULT_ENERGY_KEY,
) -> List[Dict[str, str]]:
    items = list(rows)
    front = [
        row
        for row in items
        if not any(
            dominated(row, other, energy_key=energy_key)
            for other in items
            if other is not row
        )
    ]
    return sorted(
        front,
        key=lambda row: (
            _float(row, energy_key),
            -_float(row, THROUGHPUT_KEY),
            row.get("algorithm", ""),
            row.get("gamma", ""),
            row.get("drafter_freq_hz", ""),
            row.get("verifier_clock_mhz", ""),
        ),
    )


def best_energy(
    rows: Iterable[Dict[str, str]],
    energy_key: str = DEFAULT_ENERGY_KEY,
) -> Optional[Dict[str, str]]:
    items = list(rows)
    if not items:
        return None
    return min(
        items,
        key=lambda row: (
            _float(row, energy_key),
            -_float(row, THROUGHPUT_KEY),
            _float(row, LATENCY_KEY),
        ),
    )


def companion_metric_key(energy_key: str, prefix: str) -> str:
    if energy_key.startswith("mean_"):
        return f"{prefix}_{energy_key[len('mean_'):]}"
    return f"{prefix}_{energy_key}"


def compact_config(row: Optional[Dict[str, str]], energy_key: str) -> Dict[str, str]:
    if row is None:
        return {}
    energy_stdev_key = companion_metric_key(energy_key, "stdev")
    energy_stderr_key = companion_metric_key(energy_key, "stderr")
    energy_ci95_key = companion_metric_key(energy_key, "ci95")
    keys = CONFIG_KEYS + [
        energy_key,
        energy_stdev_key,
        energy_stderr_key,
        energy_ci95_key,
        THROUGHPUT_KEY,
        "stdev_tokens_per_s",
        "ci95_tokens_per_s",
        LATENCY_KEY,
        "stdev_wall_latency_ms",
        "ci95_wall_latency_ms",
        "runs",
        "prompts",
        "mean_accept_rate",
    ]
    return {key: str(row.get(key, "")) for key in keys if key in row}


def _group_sort_key(value: str) -> tuple:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def best_by_key(
    rows: Iterable[Dict[str, str]],
    group_key: str,
    energy_key: str = DEFAULT_ENERGY_KEY,
) -> List[Dict[str, str]]:
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        value = str(row.get(group_key, ""))
        if not value:
            continue
        groups.setdefault(value, []).append(row)
    result = []
    for value, items in groups.items():
        selected = best_energy(items, energy_key=energy_key)
        if selected is None:
            continue
        record = {group_key: value}
        record.update(compact_config(selected, energy_key))
        result.append(record)
    return sorted(result, key=lambda row: _group_sort_key(str(row.get(group_key, ""))))


def _positive_pct_savings(selected: float, reference: float) -> Optional[float]:
    if reference <= 0 or selected <= 0:
        return None
    return (reference - selected) / reference * 100.0


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    if numerator <= 0 or denominator <= 0:
        return None
    return numerator / denominator


def ranked_by_energy(
    rows: Iterable[Dict[str, str]],
    energy_key: str = DEFAULT_ENERGY_KEY,
) -> List[Dict[str, str]]:
    return sorted(
        [row for row in rows if _float(row, energy_key) > 0],
        key=lambda row: (
            _float(row, energy_key),
            -_float(row, THROUGHPUT_KEY),
            _float(row, LATENCY_KEY),
        ),
    )


def selection_margin(
    rows: Iterable[Dict[str, str]],
    selected: Optional[Dict[str, str]],
    energy_key: str = DEFAULT_ENERGY_KEY,
) -> Dict[str, str]:
    if selected is None:
        return {
            "runner_up_config": {},
            "energy_margin_pct_vs_runner_up": "",
            "energy_ci95_margin_clear": "",
        }
    ranked = ranked_by_energy(rows, energy_key=energy_key)
    runner_up = next((row for row in ranked if row is not selected), None)
    selected_energy = _float(selected, energy_key)
    runner_energy = _float(runner_up or {}, energy_key)
    margin = (
        (runner_energy - selected_energy) / selected_energy * 100.0
        if selected_energy > 0 and runner_energy > 0
        else None
    )
    ci_key = companion_metric_key(energy_key, "ci95")
    selected_ci = _float(selected, ci_key)
    runner_ci = _float(runner_up or {}, ci_key)
    ci_clear = (
        runner_energy - selected_energy > selected_ci + runner_ci
        if runner_up is not None and selected_ci > 0 and runner_ci > 0
        else None
    )
    return {
        "runner_up_config": compact_config(runner_up, energy_key),
        "energy_margin_pct_vs_runner_up": (
            f"{margin:.6f}" if margin is not None else ""
        ),
        "energy_ci95_margin_clear": (
            str(int(ci_clear)) if ci_clear is not None else ""
        ),
    }


def optimization_summary(
    feasible_rows: List[Dict[str, str]],
    pareto_rows: List[Dict[str, str]],
    selected: Optional[Dict[str, str]],
    energy_key: str = DEFAULT_ENERGY_KEY,
    constraints: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    gamma_one_rows = [
        row
        for row in feasible_rows
        if str(row.get("algorithm", "")) == "speculative"
        and str(row.get("gamma", "")) == "1"
    ]
    gamma_one_best = best_energy(gamma_one_rows, energy_key=energy_key)
    selected_energy = _float(selected or {}, energy_key)
    gamma_one_energy = _float(gamma_one_best or {}, energy_key)
    selected_throughput = _float(selected or {}, THROUGHPUT_KEY)
    gamma_one_throughput = _float(gamma_one_best or {}, THROUGHPUT_KEY)
    selected_latency = _float(selected or {}, LATENCY_KEY)
    gamma_one_latency = _float(gamma_one_best or {}, LATENCY_KEY)
    energy_savings = _positive_pct_savings(selected_energy, gamma_one_energy)
    throughput_ratio = _ratio(selected_throughput, gamma_one_throughput)
    latency_ratio = _ratio(selected_latency, gamma_one_latency)
    errors = []
    if selected is None:
        errors.append("missing_best_feasible_config")
    if gamma_one_best is None:
        errors.append("missing_gamma_one_reference_config")
    margin = selection_margin(feasible_rows, selected, energy_key=energy_key)
    return {
        "ok": not errors,
        "energy_key": energy_key,
        "constraints": constraints or {},
        "feasible_configs": len(feasible_rows),
        "pareto_configs": len(pareto_rows),
        "best_joint_config": compact_config(selected, energy_key),
        "runner_up_config": margin["runner_up_config"],
        "energy_margin_pct_vs_runner_up": margin[
            "energy_margin_pct_vs_runner_up"
        ],
        "energy_ci95_margin_clear": margin["energy_ci95_margin_clear"],
        "best_gamma_one_config": compact_config(gamma_one_best, energy_key),
        "energy_savings_pct_vs_best_gamma_one": (
            f"{energy_savings:.6f}" if energy_savings is not None else ""
        ),
        "throughput_ratio_vs_best_gamma_one": (
            f"{throughput_ratio:.6f}" if throughput_ratio is not None else ""
        ),
        "latency_ratio_vs_best_gamma_one": (
            f"{latency_ratio:.6f}" if latency_ratio is not None else ""
        ),
        "best_by_gamma": best_by_key(feasible_rows, "gamma", energy_key=energy_key),
        "best_by_drafter_freq_hz": best_by_key(
            feasible_rows,
            "drafter_freq_hz",
            energy_key=energy_key,
        ),
        "best_by_verifier_clock_mhz": best_by_key(
            feasible_rows,
            "verifier_clock_mhz",
            energy_key=energy_key,
        ),
        "errors": errors,
    }


def policy_by_gamma(
    feasible_rows: List[Dict[str, str]],
    energy_key: str = DEFAULT_ENERGY_KEY,
) -> List[Dict[str, str]]:
    policies = []
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in feasible_rows:
        gamma = str(row.get("gamma", ""))
        if gamma:
            groups.setdefault(gamma, []).append(row)
    for gamma in sorted(groups, key=_group_sort_key):
        gamma_row = best_energy(groups[gamma], energy_key=energy_key)
        if gamma_row is None:
            continue
        margin = selection_margin(groups[gamma], gamma_row, energy_key=energy_key)
        policy = {
            "gamma": gamma_row.get("gamma", ""),
            "drafter_freq_hz": gamma_row.get("drafter_freq_hz", ""),
            "verifier_clock_mhz": gamma_row.get("verifier_clock_mhz", ""),
            "decoding_mode": gamma_row.get("decoding_mode", ""),
            "max_new_tokens": gamma_row.get("max_new_tokens", ""),
            "stop_token_policy": gamma_row.get("stop_token_policy", ""),
            "stop_token_ids": gamma_row.get("stop_token_ids", ""),
            "prompt_set_sha256": gamma_row.get("prompt_set_sha256", ""),
            "energy_key": energy_key,
            "energy": gamma_row.get(energy_key, ""),
            "energy_ci95": gamma_row.get(companion_metric_key(energy_key, "ci95"), ""),
            "runner_up_drafter_freq_hz": margin["runner_up_config"].get(
                "drafter_freq_hz",
                "",
            ),
            "runner_up_verifier_clock_mhz": margin["runner_up_config"].get(
                "verifier_clock_mhz",
                "",
            ),
            "energy_margin_pct_vs_runner_up": margin[
                "energy_margin_pct_vs_runner_up"
            ],
            "energy_ci95_margin_clear": margin["energy_ci95_margin_clear"],
            "tokens_per_s": gamma_row.get(THROUGHPUT_KEY, ""),
            "wall_latency_ms": gamma_row.get(LATENCY_KEY, ""),
            "runs": gamma_row.get("runs", ""),
            "prompts": gamma_row.get("prompts", ""),
        }
        if "mean_accept_rate" in gamma_row:
            policy["mean_accept_rate"] = gamma_row.get("mean_accept_rate", "")
        policies.append(policy)
    return policies


def policy_report(
    feasible_rows: List[Dict[str, str]],
    energy_key: str = DEFAULT_ENERGY_KEY,
    constraints: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    policy = policy_by_gamma(feasible_rows, energy_key=energy_key)
    verifier_clocks = {
        str(row.get("verifier_clock_mhz", ""))
        for row in policy
        if row.get("verifier_clock_mhz", "")
    }
    drafter_freqs = {
        str(row.get("drafter_freq_hz", ""))
        for row in policy
        if row.get("drafter_freq_hz", "")
    }
    errors = []
    if not policy:
        errors.append("missing_gamma_policy_rows")
    return {
        "ok": not errors,
        "energy_key": energy_key,
        "constraints": constraints or {},
        "policy_rows": len(policy),
        "policy_by_gamma": policy,
        "uses_gamma_dependent_verifier_clock": len(verifier_clocks) > 1,
        "uses_gamma_dependent_drafter_freq": len(drafter_freqs) > 1,
        "errors": errors,
    }


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        fieldnames = [
            "algorithm",
            "gamma",
            "drafter_freq_hz",
            "verifier_clock_mhz",
            "decoding_mode",
            "max_new_tokens",
            "stop_token_policy",
            "stop_token_ids",
            "prompt_set_sha256",
            DEFAULT_ENERGY_KEY,
            THROUGHPUT_KEY,
            LATENCY_KEY,
        ]
    else:
        seen = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        fieldnames = seen

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def describe(row: Dict[str, str], energy_key: str = DEFAULT_ENERGY_KEY) -> str:
    return (
        f"algorithm={row.get('algorithm', '')} "
        f"gamma={row.get('gamma', '')} "
        f"f_draft={row.get('drafter_freq_hz', '')} "
        f"f_verify={row.get('verifier_clock_mhz', '')} "
        f"{energy_key}={row.get(energy_key, '')}mJ "
        f"tok/s={row.get(THROUGHPUT_KEY, '')} "
        f"wall_ms={row.get(LATENCY_KEY, '')} "
        f"runs={row.get('runs', '')} "
        f"prompts={row.get('prompts', '')}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select best and Pareto-optimal configs from summary CSV files."
    )
    parser.add_argument("--input", nargs="+", required=True, help="Summary CSV file(s)")
    parser.add_argument("--out", default="selected_pareto.csv", help="Pareto CSV output")
    parser.add_argument(
        "--algorithm",
        default="",
        help="Optional comma-separated algorithm filter, e.g. speculative,verifier_only",
    )
    parser.add_argument("--min-tokens-per-s", type=float, default=0.0)
    parser.add_argument("--max-wall-latency-ms", type=float)
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-prompts", type=int, default=1)
    parser.add_argument(
        "--energy-key",
        default=DEFAULT_ENERGY_KEY,
        help="Summary CSV energy metric to minimize.",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional JSON output describing the constrained optimization result.",
    )
    parser.add_argument(
        "--policy-out",
        default="",
        help="Optional CSV output mapping each gamma to its best frequency pair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csvs(args.input)
    algorithms = _split_csv(args.algorithm) if args.algorithm else None
    feasible = [
        row
        for row in rows
        if passes_constraints(
            row=row,
            algorithms=algorithms,
            min_tokens_per_s=args.min_tokens_per_s,
            max_wall_latency_ms=args.max_wall_latency_ms,
            min_runs=args.min_runs,
            min_prompts=args.min_prompts,
            energy_key=args.energy_key,
        )
    ]
    front = pareto_front(feasible, energy_key=args.energy_key)
    selected = best_energy(feasible, energy_key=args.energy_key)
    write_csv(args.out, front)
    report = optimization_summary(
        feasible_rows=feasible,
        pareto_rows=front,
        selected=selected,
        energy_key=args.energy_key,
        constraints={
            "algorithm": args.algorithm,
            "min_tokens_per_s": args.min_tokens_per_s,
            "max_wall_latency_ms": args.max_wall_latency_ms,
            "min_runs": args.min_runs,
            "min_prompts": args.min_prompts,
        },
    )
    gamma_policy = policy_report(
        feasible_rows=feasible,
        energy_key=args.energy_key,
        constraints=report["constraints"],
    )
    report["gamma_policy"] = gamma_policy
    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    if args.policy_out:
        write_csv(args.policy_out, gamma_policy["policy_by_gamma"])

    print(f"feasible_configs={len(feasible)}")
    print(f"pareto_configs={len(front)}")
    print(f"policy_rows={gamma_policy['policy_rows']}")
    if selected is None:
        print("best_energy_per_token: none")
    else:
        print(f"best_energy_per_token: {describe(selected, energy_key=args.energy_key)}")
    print(f"Wrote {len(front)} Pareto rows to {args.out}")


if __name__ == "__main__":
    main()
