import argparse
import csv
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SPEC_ALGORITHM = "speculative"
DEFAULT_BASELINE_ALGORITHM = "verifier_only"
DEFAULT_ENERGY_KEY = "mean_active_system_energy_mj_per_token"


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


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def read_csvs(paths: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["_source"] = path
                rows.append(row)
    return rows


def baseline_key(
    row: Dict[str, str],
    drafter_freq_hz: Optional[str] = None,
) -> Tuple[str, str, str, str, str, str, str, str, str]:
    return (
        row.get("drafter_freq_hz", "") if drafter_freq_hz is None else drafter_freq_hz,
        row.get("verifier_clock_mhz", ""),
        row.get("decoding_mode", "greedy"),
        row.get("max_new_tokens", ""),
        row.get("stop_token_policy", ""),
        row.get("stop_token_ids", ""),
        row.get("prompt_set_sha256", ""),
        row.get("verifier_model", ""),
        row.get("verifier_runtime_fingerprint", ""),
    )


def better_baseline(
    candidate: Dict[str, str],
    current: Optional[Dict[str, str]],
    energy_key: str,
) -> bool:
    if current is None:
        return True
    candidate_size = (_int(candidate, "runs"), _int(candidate, "prompts"))
    current_size = (_int(current, "runs"), _int(current, "prompts"))
    if candidate_size != current_size:
        return candidate_size > current_size
    return _float(candidate, energy_key) < _float(current, energy_key)


def baseline_index(
    rows: Iterable[Dict[str, str]],
    baseline_algorithm: str,
    energy_key: str,
) -> Dict[Tuple[str, str, str, str, str, str, str, str], Dict[str, str]]:
    baselines: Dict[Tuple[str, str, str, str, str, str, str, str], Dict[str, str]] = {}
    for row in rows:
        if row.get("algorithm", "") != baseline_algorithm:
            continue
        key = baseline_key(row)
        if better_baseline(row, baselines.get(key), energy_key=energy_key):
            baselines[key] = row
    return baselines


def compare_rows(
    rows: Iterable[Dict[str, str]],
    spec_algorithm: str = DEFAULT_SPEC_ALGORITHM,
    baseline_algorithm: str = DEFAULT_BASELINE_ALGORITHM,
    energy_key: str = DEFAULT_ENERGY_KEY,
) -> Tuple[List[Dict[str, str]], int]:
    items = list(rows)
    baselines = baseline_index(items, baseline_algorithm, energy_key=energy_key)
    compared: List[Dict[str, str]] = []
    unmatched = 0

    for row in items:
        if row.get("algorithm", "") != spec_algorithm:
            continue
        baseline = baselines.get(baseline_key(row))
        if baseline is None:
            baseline = baselines.get(baseline_key(row, drafter_freq_hz=""))
        if baseline is None:
            unmatched += 1
            continue

        spec_energy = _float(row, energy_key)
        baseline_energy = _float(baseline, energy_key)
        energy_ratio = _ratio(spec_energy, baseline_energy)
        energy_savings = (
            (1.0 - energy_ratio) * 100.0 if energy_ratio is not None else None
        )

        spec_total_energy = _float(row, "mean_system_energy_mj_per_token")
        baseline_total_energy = _float(baseline, "mean_system_energy_mj_per_token")
        total_energy_ratio = _ratio(spec_total_energy, baseline_total_energy)
        total_energy_savings = (
            (1.0 - total_energy_ratio) * 100.0
            if total_energy_ratio is not None
            else None
        )

        throughput_ratio = _ratio(
            _float(row, "mean_tokens_per_s"),
            _float(baseline, "mean_tokens_per_s"),
        )
        latency_ratio = _ratio(
            _float(row, "mean_wall_latency_ms"),
            _float(baseline, "mean_wall_latency_ms"),
        )

        compared.append(
            {
                "algorithm": row.get("algorithm", ""),
                "gamma": row.get("gamma", ""),
                "drafter_freq_hz": row.get("drafter_freq_hz", ""),
                "verifier_clock_mhz": row.get("verifier_clock_mhz", ""),
                "decoding_mode": row.get("decoding_mode", ""),
                "max_new_tokens": row.get("max_new_tokens", ""),
                "stop_token_policy": row.get("stop_token_policy", ""),
                "stop_token_ids": row.get("stop_token_ids", ""),
                "prompt_set_sha256": row.get("prompt_set_sha256", ""),
                "drafter_model": row.get("drafter_model", ""),
                "verifier_model": row.get("verifier_model", ""),
                "drafter_runtime_fingerprint": row.get("drafter_runtime_fingerprint", ""),
                "verifier_runtime_fingerprint": row.get("verifier_runtime_fingerprint", ""),
                "runs": row.get("runs", ""),
                "prompts": row.get("prompts", ""),
                "baseline_algorithm": baseline.get("algorithm", ""),
                "baseline_runs": baseline.get("runs", ""),
                "baseline_prompts": baseline.get("prompts", ""),
                "baseline_verifier_model": baseline.get("verifier_model", ""),
                "energy_key": energy_key,
                "spec_energy_mj_per_token": _fmt(spec_energy),
                "baseline_energy_mj_per_token": _fmt(baseline_energy),
                "energy_ratio_vs_baseline": _fmt(energy_ratio),
                "energy_savings_pct_vs_baseline": _fmt(energy_savings),
                "spec_system_energy_mj_per_token": _fmt(spec_total_energy),
                "baseline_system_energy_mj_per_token": _fmt(baseline_total_energy),
                "system_energy_ratio_vs_baseline": _fmt(total_energy_ratio),
                "system_energy_savings_pct_vs_baseline": _fmt(total_energy_savings),
                "spec_tokens_per_s": row.get("mean_tokens_per_s", ""),
                "baseline_tokens_per_s": baseline.get("mean_tokens_per_s", ""),
                "tokens_per_s_ratio_vs_baseline": _fmt(throughput_ratio),
                "spec_wall_latency_ms": row.get("mean_wall_latency_ms", ""),
                "baseline_wall_latency_ms": baseline.get("mean_wall_latency_ms", ""),
                "wall_latency_ratio_vs_baseline": _fmt(latency_ratio),
                "mean_accept_rate": row.get("mean_accept_rate", ""),
            }
        )
    compared.sort(
        key=lambda item: (
            -_float(item, "energy_savings_pct_vs_baseline"),
            -_float(item, "tokens_per_s_ratio_vs_baseline"),
            item.get("gamma", ""),
            item.get("drafter_freq_hz", ""),
            item.get("verifier_clock_mhz", ""),
        )
    )
    return compared, unmatched


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
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
        "drafter_model",
        "verifier_model",
        "drafter_runtime_fingerprint",
        "verifier_runtime_fingerprint",
        "runs",
        "prompts",
        "baseline_algorithm",
        "baseline_runs",
        "baseline_prompts",
        "baseline_verifier_model",
        "energy_key",
        "spec_energy_mj_per_token",
        "baseline_energy_mj_per_token",
        "energy_ratio_vs_baseline",
        "energy_savings_pct_vs_baseline",
        "spec_system_energy_mj_per_token",
        "baseline_system_energy_mj_per_token",
        "system_energy_ratio_vs_baseline",
        "system_energy_savings_pct_vs_baseline",
        "spec_tokens_per_s",
        "baseline_tokens_per_s",
        "tokens_per_s_ratio_vs_baseline",
        "spec_wall_latency_ms",
        "baseline_wall_latency_ms",
        "wall_latency_ratio_vs_baseline",
        "mean_accept_rate",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare speculative summary rows against verifier-only baselines."
    )
    parser.add_argument("--input", nargs="+", required=True, help="Summary CSV file(s)")
    parser.add_argument("--out", default="baseline_comparison.csv")
    parser.add_argument("--spec-algorithm", default=DEFAULT_SPEC_ALGORITHM)
    parser.add_argument("--baseline-algorithm", default=DEFAULT_BASELINE_ALGORITHM)
    parser.add_argument(
        "--energy-key",
        default=DEFAULT_ENERGY_KEY,
        help="Summary energy metric for the primary savings calculation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csvs(args.input)
    compared, unmatched = compare_rows(
        rows,
        spec_algorithm=args.spec_algorithm,
        baseline_algorithm=args.baseline_algorithm,
        energy_key=args.energy_key,
    )
    write_csv(args.out, compared)
    print(f"compared_configs={len(compared)}")
    print(f"unmatched_spec_configs={unmatched}")
    if compared:
        best = compared[0]
        print(
            "best_savings: "
            f"gamma={best['gamma']} "
            f"f_draft={best['drafter_freq_hz']} "
            f"f_verify={best['verifier_clock_mhz']} "
            f"savings={best['energy_savings_pct_vs_baseline']}% "
            f"tok/s_ratio={best['tokens_per_s_ratio_vs_baseline']}"
        )
    print(f"Wrote {len(compared)} comparison rows to {args.out}")


if __name__ == "__main__":
    main()
