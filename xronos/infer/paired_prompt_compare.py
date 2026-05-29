import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SPEC_ALGORITHM = "speculative"
DEFAULT_BASELINE_ALGORITHM = "verifier_only"
DEFAULT_ENERGY_KEY = "system_active_energy_mj_per_generated_token"


def _float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        parsed = float(value) if value not in ("", None) else 0.0
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _truthy(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "y")


def _preferred_rail(row: Dict[str, str]) -> str:
    if row.get("algorithm", "") == DEFAULT_BASELINE_ALGORITHM:
        return row.get("verifier_primary_power_rail", "") or "verifier_gpu_power"
    return row.get("drafter_primary_power_rail", "") or "tot_power"


def _session_row_score(row: Dict[str, str]) -> Tuple[int, int, int]:
    rail = row.get("rail", "")
    preferred_rail = _preferred_rail(row)
    primary_rail = int(bool(preferred_rail) and rail == preferred_rail)
    has_total_energy = int(bool(row.get("system_total_energy_mj_per_generated_token", "")))
    has_active_energy = int(bool(row.get("system_active_energy_mj_per_generated_token", "")))
    return primary_rail, has_total_energy, has_active_energy


def _mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _ci95(values: List[float]) -> float:
    return 1.96 * _stdev(values) / (len(values) ** 0.5) if len(values) >= 2 else 0.0


def _median(values: List[float]) -> float:
    return statistics.median(values) if values else 0.0


def _percentile(sorted_values: List[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = percentile * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap_mean_ci95(
    values: List[float],
    samples: int,
    seed: int,
) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1 or samples <= 0:
        mean = _mean(values)
        return mean, mean

    rng = random.Random(seed)
    n_values = len(values)
    boot_means = [
        _mean([values[rng.randrange(n_values)] for _ in range(n_values)])
        for _ in range(samples)
    ]
    boot_means.sort()
    return _percentile(boot_means, 0.025), _percentile(boot_means, 0.975)


def _sign_test_p_value(values: List[float]) -> Optional[float]:
    positives = sum(1 for value in values if value > 0)
    negatives = sum(1 for value in values if value < 0)
    n_trials = positives + negatives
    if n_trials == 0:
        return None
    tail = min(positives, negatives)
    p_value = 2.0 * sum(
        math.comb(n_trials, k) * (0.5 ** n_trials)
        for k in range(tail + 1)
    )
    return min(1.0, p_value)


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def _output_hashes(rows: Iterable[Dict[str, str]]) -> List[str]:
    return sorted(
        {
            row.get("output_token_sha256", "")
            for row in rows
            if row.get("output_token_sha256", "")
        }
    )


def _single_value(rows: Iterable[Dict[str, str]], key: str) -> str:
    values = sorted({row.get(key, "") for row in rows if row.get(key, "")})
    if not values:
        return ""
    return values[0] if len(values) == 1 else "mixed"


def _output_match(
    spec_hashes: Sequence[str],
    baseline_hashes: Sequence[str],
) -> Optional[float]:
    if not spec_hashes or not baseline_hashes:
        return None
    return 1.0 if list(spec_hashes) == list(baseline_hashes) else 0.0


def read_csvs(paths: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["_source"] = path
                rows.append(row)
    return rows


def session_rows(
    rows: Iterable[Dict[str, str]],
    allow_incomplete_energy: bool,
) -> List[Dict[str, str]]:
    by_session: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if not allow_incomplete_energy and not _truthy(
            row.get("system_energy_complete", "")
        ):
            continue
        session_id = row.get("session_id", "")
        if not session_id:
            continue
        current = by_session.get(session_id)
        if current is None or _session_row_score(row) > _session_row_score(current):
            by_session[session_id] = row
    return list(by_session.values())


def prompt_key(
    row: Dict[str, str],
    include_drafter_freq: bool = False,
) -> Tuple[str, ...]:
    key = (
        row.get("verifier_clock_mhz", ""),
        row.get("decoding_mode", "greedy"),
        row.get("max_new_tokens", ""),
        row.get("stop_token_policy", ""),
        row.get("stop_token_ids", ""),
        row.get("prompt_set_sha256", ""),
        row.get("verifier_model", ""),
        row.get("verifier_runtime_fingerprint", ""),
        row.get("prompt_id", ""),
        row.get("prompt_sha256", ""),
    )
    if include_drafter_freq:
        return (row.get("drafter_freq_hz", ""),) + key
    return key


def spec_config_key(row: Dict[str, str]) -> Tuple[str, ...]:
    return (
        row.get("gamma", ""),
        row.get("drafter_freq_hz", ""),
        row.get("verifier_clock_mhz", ""),
        row.get("decoding_mode", "greedy"),
        row.get("max_new_tokens", ""),
        row.get("stop_token_policy", ""),
        row.get("stop_token_ids", ""),
        row.get("prompt_set_sha256", ""),
        row.get("drafter_model", ""),
        row.get("verifier_model", ""),
        row.get("drafter_runtime_fingerprint", ""),
        row.get("verifier_runtime_fingerprint", ""),
    )


def grouped_means(
    rows: Iterable[Dict[str, str]],
    energy_key: str,
    algorithm: str,
    include_spec_config: bool,
    include_drafter_freq_in_prompt_key: bool = False,
) -> Dict[Tuple[str, ...], Dict[str, object]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("algorithm", "") != algorithm:
            continue
        if _float(row, energy_key) <= 0:
            continue
        key = prompt_key(
            row,
            include_drafter_freq=include_drafter_freq_in_prompt_key,
        )
        if include_spec_config:
            key = spec_config_key(row) + key
        groups[key].append(row)

    means: Dict[Tuple[str, ...], Dict[str, object]] = {}
    for key, items in groups.items():
        means[key] = {
            "runs": len(items),
            "system_boundary": _single_value(items, "system_boundary"),
            "energy": _mean([_float(row, energy_key) for row in items]),
            "system_energy": _mean(
                [
                    _float(row, "system_total_energy_mj_per_generated_token")
                    for row in items
                ]
            ),
            "tokens_per_s": _mean([_float(row, "tokens_per_s") for row in items]),
            "wall_latency_ms": _mean([_float(row, "wall_latency_ms") for row in items]),
            "accept_rate": _mean([_float(row, "accept_rate") for row in items]),
            "output_hashes": _output_hashes(items),
        }
    return means


def aggregate_pairs(
    rows: List[Dict[str, str]],
    energy_key: str = DEFAULT_ENERGY_KEY,
    spec_algorithm: str = DEFAULT_SPEC_ALGORITHM,
    baseline_algorithm: str = DEFAULT_BASELINE_ALGORITHM,
    allow_incomplete_energy: bool = False,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    sessions = session_rows(rows, allow_incomplete_energy)
    baselines = grouped_means(
        sessions,
        energy_key=energy_key,
        algorithm=baseline_algorithm,
        include_spec_config=False,
        include_drafter_freq_in_prompt_key=True,
    )
    specs = grouped_means(
        sessions,
        energy_key=energy_key,
        algorithm=spec_algorithm,
        include_spec_config=True,
    )

    per_config: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    pair_rows: List[Dict[str, str]] = []
    for spec_key, spec in specs.items():
        config_key = spec_key[:12]
        base_prompt_key = spec_key[12:]
        baseline = baselines.get((config_key[1],) + base_prompt_key)
        if baseline is None:
            baseline = baselines.get(("",) + base_prompt_key)
        if baseline is None:
            continue
        spec_energy = float(spec["energy"])
        baseline_energy = float(baseline["energy"])
        if baseline_energy <= 0:
            continue

        energy_ratio = spec_energy / baseline_energy
        energy_savings_pct = (1.0 - energy_ratio) * 100.0
        spec_system = float(spec["system_energy"])
        baseline_system = float(baseline["system_energy"])
        system_ratio = spec_system / baseline_system if baseline_system > 0 else 0.0
        throughput_ratio = (
            float(spec["tokens_per_s"]) / float(baseline["tokens_per_s"])
            if float(baseline["tokens_per_s"]) > 0
            else 0.0
        )
        latency_ratio = (
            float(spec["wall_latency_ms"]) / float(baseline["wall_latency_ms"])
            if float(baseline["wall_latency_ms"]) > 0
            else 0.0
        )
        output_match = _output_match(
            spec["output_hashes"],
            baseline["output_hashes"],
        )
        row = {
            "gamma": config_key[0],
            "drafter_freq_hz": config_key[1],
            "verifier_clock_mhz": config_key[2],
            "decoding_mode": config_key[3],
            "max_new_tokens": config_key[4],
            "stop_token_policy": config_key[5],
            "stop_token_ids": config_key[6],
            "prompt_set_sha256": config_key[7],
            "drafter_model": config_key[8],
            "verifier_model": config_key[9],
            "drafter_runtime_fingerprint": config_key[10],
            "verifier_runtime_fingerprint": config_key[11],
            "prompt_id": base_prompt_key[8],
            "prompt_sha256": base_prompt_key[9],
            "spec_runs": str(spec["runs"]),
            "baseline_runs": str(baseline["runs"]),
            "spec_system_boundary": str(spec.get("system_boundary", "")),
            "baseline_system_boundary": str(baseline.get("system_boundary", "")),
            "energy_key": energy_key,
            "spec_energy_mj_per_token": _fmt(spec_energy),
            "baseline_energy_mj_per_token": _fmt(baseline_energy),
            "energy_ratio_vs_baseline": _fmt(energy_ratio),
            "energy_savings_pct_vs_baseline": _fmt(energy_savings_pct),
            "spec_system_energy_mj_per_token": _fmt(spec_system),
            "baseline_system_energy_mj_per_token": _fmt(baseline_system),
            "system_energy_ratio_vs_baseline": _fmt(system_ratio),
            "system_energy_savings_pct_vs_baseline": _fmt(
                (1.0 - system_ratio) * 100.0 if system_ratio > 0 else None
            ),
            "tokens_per_s_ratio_vs_baseline": _fmt(throughput_ratio),
            "wall_latency_ratio_vs_baseline": _fmt(latency_ratio),
            "mean_accept_rate": _fmt(float(spec["accept_rate"])),
            "output_token_match": _fmt(output_match),
            "spec_output_hash_count": str(len(spec["output_hashes"])),
            "baseline_output_hash_count": str(len(baseline["output_hashes"])),
        }
        pair_rows.append(row)
        per_config[config_key].append(row)

    summary_rows: List[Dict[str, str]] = []
    for config_key, items in sorted(per_config.items()):
        energy_savings = [
            _float(row, "energy_savings_pct_vs_baseline") for row in items
        ]
        system_savings = [
            _float(row, "system_energy_savings_pct_vs_baseline") for row in items
        ]
        throughput_ratios = [
            _float(row, "tokens_per_s_ratio_vs_baseline") for row in items
        ]
        latency_ratios = [
            _float(row, "wall_latency_ratio_vs_baseline") for row in items
        ]
        accept_rates = [_float(row, "mean_accept_rate") for row in items]
        positive_savings = sum(1 for value in energy_savings if value > 0)
        bootstrap_low, bootstrap_high = _bootstrap_mean_ci95(
            energy_savings,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        output_matches = [
            _float(row, "output_token_match")
            for row in items
            if row.get("output_token_match", "") not in ("", None)
        ]
        spec_runs = sum(int(row["spec_runs"]) for row in items)
        baseline_runs = sum(int(row["baseline_runs"]) for row in items)
        baseline_boundaries = sorted(
            {
                row.get("baseline_system_boundary", "")
                for row in items
                if row.get("baseline_system_boundary", "")
            }
        )
        summary_rows.append(
            {
                "algorithm": spec_algorithm,
                "baseline_algorithm": baseline_algorithm,
                "gamma": config_key[0],
                "drafter_freq_hz": config_key[1],
                "verifier_clock_mhz": config_key[2],
                "decoding_mode": config_key[3],
                "max_new_tokens": config_key[4],
                "stop_token_policy": config_key[5],
                "stop_token_ids": config_key[6],
                "prompt_set_sha256": config_key[7],
                "drafter_model": config_key[8],
                "verifier_model": config_key[9],
                "drafter_runtime_fingerprint": config_key[10],
                "verifier_runtime_fingerprint": config_key[11],
                "paired_prompts": str(len(items)),
                "spec_runs": str(spec_runs),
                "baseline_runs": str(baseline_runs),
                "baseline_system_boundary": (
                    baseline_boundaries[0]
                    if len(baseline_boundaries) == 1
                    else "mixed"
                    if baseline_boundaries
                    else ""
                ),
                "energy_key": energy_key,
                "mean_energy_savings_pct_vs_baseline": _fmt(_mean(energy_savings)),
                "median_energy_savings_pct_vs_baseline": _fmt(_median(energy_savings)),
                "stdev_energy_savings_pct_vs_baseline": _fmt(_stdev(energy_savings)),
                "ci95_energy_savings_pct_vs_baseline": _fmt(_ci95(energy_savings)),
                "bootstrap_ci95_low_energy_savings_pct_vs_baseline": _fmt(
                    bootstrap_low
                ),
                "bootstrap_ci95_high_energy_savings_pct_vs_baseline": _fmt(
                    bootstrap_high
                ),
                "positive_energy_savings_prompts": str(positive_savings),
                "positive_energy_savings_fraction": _fmt(
                    positive_savings / len(energy_savings) if energy_savings else None
                ),
                "sign_test_p_value_energy_savings": _fmt(
                    _sign_test_p_value(energy_savings)
                ),
                "mean_system_energy_savings_pct_vs_baseline": _fmt(_mean(system_savings)),
                "mean_tokens_per_s_ratio_vs_baseline": _fmt(_mean(throughput_ratios)),
                "mean_wall_latency_ratio_vs_baseline": _fmt(_mean(latency_ratios)),
                "mean_accept_rate": _fmt(_mean(accept_rates)),
                "mean_output_token_match": _fmt(
                    _mean(output_matches) if output_matches else None
                ),
                "output_checked_prompts": str(len(output_matches)),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            -_float(row, "mean_energy_savings_pct_vs_baseline"),
            -_float(row, "mean_tokens_per_s_ratio_vs_baseline"),
            row.get("gamma", ""),
        )
    )
    return summary_rows, pair_rows


def unpaired_spec_prompts(
    rows: List[Dict[str, str]],
    energy_key: str = DEFAULT_ENERGY_KEY,
    spec_algorithm: str = DEFAULT_SPEC_ALGORITHM,
    baseline_algorithm: str = DEFAULT_BASELINE_ALGORITHM,
    allow_incomplete_energy: bool = False,
) -> List[Dict[str, str]]:
    sessions = session_rows(rows, allow_incomplete_energy)
    baselines = grouped_means(
        sessions,
        energy_key=energy_key,
        algorithm=baseline_algorithm,
        include_spec_config=False,
        include_drafter_freq_in_prompt_key=True,
    )
    specs = grouped_means(
        sessions,
        energy_key=energy_key,
        algorithm=spec_algorithm,
        include_spec_config=True,
    )

    unpaired: List[Dict[str, str]] = []
    for spec_key in sorted(specs):
        config_key = spec_key[:12]
        base_prompt_key = spec_key[12:]
        baseline = baselines.get((config_key[1],) + base_prompt_key)
        if baseline is None:
            baseline = baselines.get(("",) + base_prompt_key)
        if baseline is not None and float(baseline["energy"]) > 0:
            continue
        unpaired.append(
            {
                "gamma": config_key[0],
                "drafter_freq_hz": config_key[1],
                "verifier_clock_mhz": config_key[2],
                "decoding_mode": config_key[3],
                "max_new_tokens": config_key[4],
                "stop_token_policy": config_key[5],
                "stop_token_ids": config_key[6],
                "prompt_set_sha256": config_key[7],
                "drafter_model": config_key[8],
                "verifier_model": config_key[9],
                "drafter_runtime_fingerprint": config_key[10],
                "verifier_runtime_fingerprint": config_key[11],
                "prompt_id": base_prompt_key[8],
                "prompt_sha256": base_prompt_key[9],
                "spec_runs": str(specs[spec_key]["runs"]),
                "reason": "missing_or_invalid_baseline",
            }
        )
    return unpaired


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


SUMMARY_FIELDNAMES = [
    "algorithm",
    "baseline_algorithm",
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
    "paired_prompts",
    "spec_runs",
    "baseline_runs",
    "baseline_system_boundary",
    "energy_key",
    "mean_energy_savings_pct_vs_baseline",
    "median_energy_savings_pct_vs_baseline",
    "stdev_energy_savings_pct_vs_baseline",
    "ci95_energy_savings_pct_vs_baseline",
    "bootstrap_ci95_low_energy_savings_pct_vs_baseline",
    "bootstrap_ci95_high_energy_savings_pct_vs_baseline",
    "positive_energy_savings_prompts",
    "positive_energy_savings_fraction",
    "sign_test_p_value_energy_savings",
    "mean_system_energy_savings_pct_vs_baseline",
    "mean_tokens_per_s_ratio_vs_baseline",
    "mean_wall_latency_ratio_vs_baseline",
    "mean_accept_rate",
    "mean_output_token_match",
    "output_checked_prompts",
]


PAIR_FIELDNAMES = [
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
    "prompt_id",
    "prompt_sha256",
    "spec_runs",
    "baseline_runs",
    "spec_system_boundary",
    "baseline_system_boundary",
    "energy_key",
    "spec_energy_mj_per_token",
    "baseline_energy_mj_per_token",
    "energy_ratio_vs_baseline",
    "energy_savings_pct_vs_baseline",
    "spec_system_energy_mj_per_token",
    "baseline_system_energy_mj_per_token",
    "system_energy_ratio_vs_baseline",
    "system_energy_savings_pct_vs_baseline",
    "tokens_per_s_ratio_vs_baseline",
    "wall_latency_ratio_vs_baseline",
    "mean_accept_rate",
    "output_token_match",
    "spec_output_hash_count",
    "baseline_output_hash_count",
]

UNPAIRED_FIELDNAMES = [
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
    "prompt_id",
    "prompt_sha256",
    "spec_runs",
    "reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair speculative and verifier-only raw runs by prompt."
    )
    parser.add_argument("--input", nargs="+", required=True, help="Raw result CSV file(s)")
    parser.add_argument("--out", default="paired_prompt_summary.csv")
    parser.add_argument("--pairs-out", default="")
    parser.add_argument("--unpaired-out", default="")
    parser.add_argument("--spec-algorithm", default=DEFAULT_SPEC_ALGORITHM)
    parser.add_argument("--baseline-algorithm", default=DEFAULT_BASELINE_ALGORITHM)
    parser.add_argument("--energy-key", default=DEFAULT_ENERGY_KEY)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--allow-incomplete-energy",
        action="store_true",
        help="Include runs without complete required energy samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csvs(args.input)
    summary_rows, pair_rows = aggregate_pairs(
        rows,
        energy_key=args.energy_key,
        spec_algorithm=args.spec_algorithm,
        baseline_algorithm=args.baseline_algorithm,
        allow_incomplete_energy=args.allow_incomplete_energy,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    unpaired_rows = unpaired_spec_prompts(
        rows,
        energy_key=args.energy_key,
        spec_algorithm=args.spec_algorithm,
        baseline_algorithm=args.baseline_algorithm,
        allow_incomplete_energy=args.allow_incomplete_energy,
    )
    write_csv(args.out, summary_rows, SUMMARY_FIELDNAMES)
    if args.pairs_out:
        write_csv(args.pairs_out, pair_rows, PAIR_FIELDNAMES)
    if args.unpaired_out:
        write_csv(args.unpaired_out, unpaired_rows, UNPAIRED_FIELDNAMES)
    print(f"paired_configs={len(summary_rows)}")
    print(f"paired_prompt_rows={len(pair_rows)}")
    print(f"unpaired_spec_prompt_rows={len(unpaired_rows)}")
    if summary_rows:
        best = summary_rows[0]
        print(
            "best_paired_savings: "
            f"gamma={best['gamma']} "
            f"f_draft={best['drafter_freq_hz']} "
            f"f_verify={best['verifier_clock_mhz']} "
            f"paired_prompts={best['paired_prompts']} "
            f"savings={best['mean_energy_savings_pct_vs_baseline']}%"
        )
    print(f"Wrote {len(summary_rows)} paired summary rows to {args.out}")
    if args.pairs_out:
        print(f"Wrote {len(pair_rows)} prompt-pair rows to {args.pairs_out}")
    if args.unpaired_out:
        print(f"Wrote {len(unpaired_rows)} unpaired prompt rows to {args.unpaired_out}")


if __name__ == "__main__":
    main()
