import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        parsed = float(value) if value not in ("", None) else 0.0
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _truthy(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "y")


def _mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _ci95(values: List[float]) -> float:
    return 1.96 * _stdev(values) / (len(values) ** 0.5) if len(values) >= 2 else 0.0


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


def _ratio(value: float, baseline: float) -> Optional[float]:
    if baseline <= 0:
        return None
    return value / baseline


def _percent_change(value: float, baseline: float) -> Optional[float]:
    ratio = _ratio(value, baseline)
    if ratio is None:
        return None
    return (ratio - 1.0) * 100.0


def _slope(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0 or y_var <= 0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return cov / ((x_var * y_var) ** 0.5)


def _session_row_score(row: Dict[str, str]) -> Tuple[int, int, int]:
    preferred_rail = row.get("drafter_primary_power_rail", "") or "tot_power"
    primary_rail = int(row.get("rail", "") == preferred_rail)
    has_total_energy = int(bool(row.get("system_total_energy_mj_per_generated_token", "")))
    has_active_energy = int(bool(row.get("system_active_energy_mj_per_generated_token", "")))
    return primary_rail, has_total_energy, has_active_energy


def session_rows(
    rows: Iterable[Dict[str, str]],
    allow_incomplete_energy: bool = False,
) -> List[Dict[str, str]]:
    by_session: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if row.get("algorithm", "speculative") != "speculative":
            continue
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


def _group_key(row: Dict[str, str]) -> Tuple[str, ...]:
    return (
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


def _prompt_key(row: Dict[str, str]) -> Tuple[str, str]:
    return row.get("prompt_id", ""), row.get("prompt_sha256", "")


def _per_generated(row: Dict[str, str], key: str) -> float:
    generated = _float(row, "generated_tokens")
    if generated <= 0:
        return 0.0
    return _float(row, key) / generated


def _per_draft_token(row: Dict[str, str], key: str) -> float:
    draft_tokens = _float(row, "draft_tokens")
    if draft_tokens <= 0:
        return 0.0
    return _float(row, key) / draft_tokens


def _gamma_value(row: Dict[str, str]) -> Optional[int]:
    try:
        gamma = int(float(row.get("gamma", "") or "0"))
    except ValueError:
        return None
    return gamma if gamma > 0 else None


def _metric_summary(items: List[Dict[str, str]]) -> Dict[str, float]:
    drafter_total_per_token = [
        _per_generated(row, "drafter_total_energy_mj") for row in items
    ]
    drafter_active_per_token = [
        _per_generated(row, "drafter_active_energy_mj") for row in items
    ]
    drafter_draft_per_generated_token = [
        _per_generated(row, "drafter_draft_total_energy_mj") for row in items
    ]
    drafter_draft_active_per_generated_token = [
        _per_generated(row, "drafter_draft_active_energy_mj") for row in items
    ]
    drafter_draft_per_draft_token = [
        _per_draft_token(row, "drafter_draft_total_energy_mj") for row in items
    ]
    drafter_draft_active_per_draft_token = [
        _per_draft_token(row, "drafter_draft_active_energy_mj") for row in items
    ]
    return {
        "runs": float(len(items)),
        "prompts": float(len({_prompt_key(row) for row in items})),
        "generated_tokens": _mean([_float(row, "generated_tokens") for row in items]),
        "draft_tokens": _mean([_float(row, "draft_tokens") for row in items]),
        "accept_rate": _mean([_float(row, "accept_rate") for row in items]),
        "tokens_per_s": _mean([_float(row, "tokens_per_s") for row in items]),
        "wall_latency_ms": _mean([_float(row, "wall_latency_ms") for row in items]),
        "drafter_total_energy_mj_per_generated_token": _mean(
            drafter_total_per_token
        ),
        "ci95_drafter_total_energy_mj_per_generated_token": _ci95(
            drafter_total_per_token
        ),
        "drafter_active_energy_mj_per_generated_token": _mean(
            drafter_active_per_token
        ),
        "drafter_draft_energy_mj_per_generated_token": _mean(
            drafter_draft_per_generated_token
        ),
        "drafter_draft_active_energy_mj_per_generated_token": _mean(
            drafter_draft_active_per_generated_token
        ),
        "drafter_draft_energy_mj_per_draft_token": _mean(
            drafter_draft_per_draft_token
        ),
        "drafter_draft_active_energy_mj_per_draft_token": _mean(
            drafter_draft_active_per_draft_token
        ),
    }


def _by_prompt_metrics(items: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    by_prompt: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in items:
        by_prompt[_prompt_key(row)].append(row)
    return {
        prompt_key: _metric_summary(prompt_items)
        for prompt_key, prompt_items in by_prompt.items()
    }


def _paired_changes(
    metrics: Dict[Tuple[str, str], Dict[str, float]],
    baseline_metrics: Dict[Tuple[str, str], Dict[str, float]],
    key: str,
) -> List[float]:
    changes = []
    for prompt_key in sorted(set(metrics) & set(baseline_metrics)):
        change = _percent_change(
            metrics[prompt_key].get(key, 0.0),
            baseline_metrics[prompt_key].get(key, 0.0),
        )
        if change is not None:
            changes.append(change)
    return changes


def summarize(
    rows: List[Dict[str, str]],
    allow_incomplete_energy: bool = False,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> List[Dict[str, str]]:
    sessions = session_rows(rows, allow_incomplete_energy=allow_incomplete_energy)
    groups: Dict[Tuple[str, ...], Dict[int, List[Dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in sessions:
        gamma = _gamma_value(row)
        if gamma is None:
            continue
        groups[_group_key(row)][gamma].append(row)

    output_rows: List[Dict[str, str]] = []
    for group_key, by_gamma in sorted(groups.items()):
        if not by_gamma:
            continue
        baseline_gamma = min(by_gamma)
        gamma_prompt_sets = {
            gamma: {_prompt_key(row) for row in items}
            for gamma, items in by_gamma.items()
        }
        common_prompts = set.intersection(*gamma_prompt_sets.values())
        complete_prompt_overlap = all(
            prompt_set == common_prompts for prompt_set in gamma_prompt_sets.values()
        )

        gamma_metrics: Dict[int, Dict[str, float]] = {}
        prompt_metrics_by_gamma: Dict[int, Dict[Tuple[str, str], Dict[str, float]]] = {}
        for gamma, items in by_gamma.items():
            gamma_metrics[gamma] = _metric_summary(items)
            prompt_metrics_by_gamma[gamma] = _by_prompt_metrics(items)

        trend_x = [math.log2(gamma) for gamma in sorted(gamma_metrics)]
        total_trend_y = [
            gamma_metrics[gamma]["drafter_total_energy_mj_per_generated_token"]
            for gamma in sorted(gamma_metrics)
        ]
        active_trend_y = [
            gamma_metrics[gamma]["drafter_active_energy_mj_per_generated_token"]
            for gamma in sorted(gamma_metrics)
        ]
        draft_trend_y = [
            gamma_metrics[gamma]["drafter_draft_energy_mj_per_generated_token"]
            for gamma in sorted(gamma_metrics)
        ]
        draft_active_trend_y = [
            gamma_metrics[gamma]["drafter_draft_active_energy_mj_per_generated_token"]
            for gamma in sorted(gamma_metrics)
        ]
        total_slope = _slope(trend_x, total_trend_y)
        active_slope = _slope(trend_x, active_trend_y)
        draft_slope = _slope(trend_x, draft_trend_y)
        draft_active_slope = _slope(trend_x, draft_active_trend_y)
        total_corr = _pearson(trend_x, total_trend_y)
        active_corr = _pearson(trend_x, active_trend_y)
        draft_corr = _pearson(trend_x, draft_trend_y)
        draft_active_corr = _pearson(trend_x, draft_active_trend_y)

        baseline_metrics = gamma_metrics[baseline_gamma]
        baseline_prompt_metrics = prompt_metrics_by_gamma[baseline_gamma]
        baseline_prompt_set = set(baseline_prompt_metrics)
        for gamma in sorted(gamma_metrics):
            metrics = gamma_metrics[gamma]
            prompt_metrics = prompt_metrics_by_gamma[gamma]
            paired_prompt_set = set(prompt_metrics) & baseline_prompt_set
            paired_total_changes = _paired_changes(
                prompt_metrics,
                baseline_prompt_metrics,
                "drafter_total_energy_mj_per_generated_token",
            )
            paired_active_changes = _paired_changes(
                prompt_metrics,
                baseline_prompt_metrics,
                "drafter_active_energy_mj_per_generated_token",
            )
            paired_draft_changes = _paired_changes(
                prompt_metrics,
                baseline_prompt_metrics,
                "drafter_draft_energy_mj_per_generated_token",
            )
            paired_draft_active_changes = _paired_changes(
                prompt_metrics,
                baseline_prompt_metrics,
                "drafter_draft_active_energy_mj_per_generated_token",
            )
            paired_draft_per_draft_changes = _paired_changes(
                prompt_metrics,
                baseline_prompt_metrics,
                "drafter_draft_energy_mj_per_draft_token",
            )
            paired_draft_active_per_draft_changes = _paired_changes(
                prompt_metrics,
                baseline_prompt_metrics,
                "drafter_draft_active_energy_mj_per_draft_token",
            )
            total_bootstrap_low, total_bootstrap_high = _bootstrap_mean_ci95(
                paired_total_changes,
                samples=bootstrap_samples,
                seed=bootstrap_seed + gamma,
            )
            active_bootstrap_low, active_bootstrap_high = _bootstrap_mean_ci95(
                paired_active_changes,
                samples=bootstrap_samples,
                seed=bootstrap_seed + (gamma * 31),
            )
            draft_bootstrap_low, draft_bootstrap_high = _bootstrap_mean_ci95(
                paired_draft_changes,
                samples=bootstrap_samples,
                seed=bootstrap_seed + (gamma * 17),
            )
            draft_active_bootstrap_low, draft_active_bootstrap_high = (
                _bootstrap_mean_ci95(
                    paired_draft_active_changes,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + (gamma * 19),
                )
            )
            draft_per_draft_bootstrap_low, draft_per_draft_bootstrap_high = (
                _bootstrap_mean_ci95(
                    paired_draft_per_draft_changes,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + (gamma * 43),
                )
            )
            (
                draft_active_per_draft_bootstrap_low,
                draft_active_per_draft_bootstrap_high,
            ) = _bootstrap_mean_ci95(
                paired_draft_active_per_draft_changes,
                samples=bootstrap_samples,
                seed=bootstrap_seed + (gamma * 47),
            )
            total_metric = metrics["drafter_total_energy_mj_per_generated_token"]
            active_metric = metrics["drafter_active_energy_mj_per_generated_token"]
            draft_metric = metrics["drafter_draft_energy_mj_per_generated_token"]
            draft_active_metric = metrics[
                "drafter_draft_active_energy_mj_per_generated_token"
            ]
            draft_per_draft = metrics["drafter_draft_energy_mj_per_draft_token"]
            draft_active_per_draft = metrics[
                "drafter_draft_active_energy_mj_per_draft_token"
            ]
            output_rows.append(
                {
                    "gamma": str(gamma),
                    "baseline_gamma": str(baseline_gamma),
                    "drafter_freq_hz": group_key[0],
                    "verifier_clock_mhz": group_key[1],
                    "decoding_mode": group_key[2],
                    "max_new_tokens": group_key[3],
                    "stop_token_policy": group_key[4],
                    "stop_token_ids": group_key[5],
                    "prompt_set_sha256": group_key[6],
                    "drafter_model": group_key[7],
                    "verifier_model": group_key[8],
                    "drafter_runtime_fingerprint": group_key[9],
                    "verifier_runtime_fingerprint": group_key[10],
                    "gamma_count": str(len(gamma_metrics)),
                    "runs": str(int(metrics["runs"])),
                    "prompts": str(int(metrics["prompts"])),
                    "common_prompts": str(len(common_prompts)),
                    "complete_prompt_overlap": str(int(complete_prompt_overlap)),
                    "paired_prompts_vs_baseline_gamma": str(len(paired_prompt_set)),
                    "complete_prompt_overlap_vs_baseline_gamma": str(
                        int(set(prompt_metrics) == baseline_prompt_set)
                    ),
                    "mean_generated_tokens": _fmt(metrics["generated_tokens"]),
                    "mean_draft_tokens": _fmt(metrics["draft_tokens"]),
                    "mean_accept_rate": _fmt(metrics["accept_rate"]),
                    "mean_tokens_per_s": _fmt(metrics["tokens_per_s"]),
                    "mean_wall_latency_ms": _fmt(metrics["wall_latency_ms"]),
                    "mean_drafter_total_energy_mj_per_generated_token": _fmt(
                        total_metric
                    ),
                    "ci95_drafter_total_energy_mj_per_generated_token": _fmt(
                        metrics["ci95_drafter_total_energy_mj_per_generated_token"]
                    ),
                    "drafter_total_energy_ratio_vs_baseline_gamma": _fmt(
                        _ratio(
                            total_metric,
                            baseline_metrics[
                                "drafter_total_energy_mj_per_generated_token"
                            ],
                        )
                    ),
                    "drafter_total_energy_change_pct_vs_baseline_gamma": _fmt(
                        _percent_change(
                            total_metric,
                            baseline_metrics[
                                "drafter_total_energy_mj_per_generated_token"
                            ],
                        )
                    ),
                    "paired_mean_drafter_total_energy_change_pct_vs_baseline_gamma": _fmt(
                        _mean(paired_total_changes) if paired_total_changes else None
                    ),
                    "paired_ci95_drafter_total_energy_change_pct_vs_baseline_gamma": _fmt(
                        _ci95(paired_total_changes) if paired_total_changes else None
                    ),
                    "paired_bootstrap_ci95_low_drafter_total_energy_change_pct_vs_baseline_gamma": _fmt(
                        total_bootstrap_low
                    ),
                    "paired_bootstrap_ci95_high_drafter_total_energy_change_pct_vs_baseline_gamma": _fmt(
                        total_bootstrap_high
                    ),
                    "paired_sign_test_p_value_drafter_total_energy_change_pct_vs_baseline_gamma": _fmt(
                        _sign_test_p_value(paired_total_changes)
                    ),
                    "mean_drafter_active_energy_mj_per_generated_token": _fmt(
                        active_metric
                    ),
                    "drafter_active_energy_ratio_vs_baseline_gamma": _fmt(
                        _ratio(
                            active_metric,
                            baseline_metrics[
                                "drafter_active_energy_mj_per_generated_token"
                            ],
                        )
                    ),
                    "paired_mean_drafter_active_energy_change_pct_vs_baseline_gamma": _fmt(
                        _mean(paired_active_changes) if paired_active_changes else None
                    ),
                    "paired_ci95_drafter_active_energy_change_pct_vs_baseline_gamma": _fmt(
                        _ci95(paired_active_changes) if paired_active_changes else None
                    ),
                    "paired_bootstrap_ci95_low_drafter_active_energy_change_pct_vs_baseline_gamma": _fmt(
                        active_bootstrap_low
                    ),
                    "paired_bootstrap_ci95_high_drafter_active_energy_change_pct_vs_baseline_gamma": _fmt(
                        active_bootstrap_high
                    ),
                    "paired_sign_test_p_value_drafter_active_energy_change_pct_vs_baseline_gamma": _fmt(
                        _sign_test_p_value(paired_active_changes)
                    ),
                    "mean_drafter_draft_energy_mj_per_generated_token": _fmt(
                        draft_metric
                    ),
                    "drafter_draft_energy_ratio_vs_baseline_gamma": _fmt(
                        _ratio(
                            draft_metric,
                            baseline_metrics[
                                "drafter_draft_energy_mj_per_generated_token"
                            ],
                        )
                    ),
                    "paired_mean_drafter_draft_energy_change_pct_vs_baseline_gamma": _fmt(
                        _mean(paired_draft_changes) if paired_draft_changes else None
                    ),
                    "paired_ci95_drafter_draft_energy_change_pct_vs_baseline_gamma": _fmt(
                        _ci95(paired_draft_changes) if paired_draft_changes else None
                    ),
                    "paired_bootstrap_ci95_low_drafter_draft_energy_change_pct_vs_baseline_gamma": _fmt(
                        draft_bootstrap_low
                    ),
                    "paired_bootstrap_ci95_high_drafter_draft_energy_change_pct_vs_baseline_gamma": _fmt(
                        draft_bootstrap_high
                    ),
                    "paired_sign_test_p_value_drafter_draft_energy_change_pct_vs_baseline_gamma": _fmt(
                        _sign_test_p_value(paired_draft_changes)
                    ),
                    "mean_drafter_draft_active_energy_mj_per_generated_token": _fmt(
                        draft_active_metric
                    ),
                    "drafter_draft_active_energy_ratio_vs_baseline_gamma": _fmt(
                        _ratio(
                            draft_active_metric,
                            baseline_metrics[
                                "drafter_draft_active_energy_mj_per_generated_token"
                            ],
                        )
                    ),
                    "paired_mean_drafter_draft_active_energy_change_pct_vs_baseline_gamma": _fmt(
                        _mean(paired_draft_active_changes)
                        if paired_draft_active_changes
                        else None
                    ),
                    "paired_ci95_drafter_draft_active_energy_change_pct_vs_baseline_gamma": _fmt(
                        _ci95(paired_draft_active_changes)
                        if paired_draft_active_changes
                        else None
                    ),
                    "paired_bootstrap_ci95_low_drafter_draft_active_energy_change_pct_vs_baseline_gamma": _fmt(
                        draft_active_bootstrap_low
                    ),
                    "paired_bootstrap_ci95_high_drafter_draft_active_energy_change_pct_vs_baseline_gamma": _fmt(
                        draft_active_bootstrap_high
                    ),
                    "paired_sign_test_p_value_drafter_draft_active_energy_change_pct_vs_baseline_gamma": _fmt(
                        _sign_test_p_value(paired_draft_active_changes)
                    ),
                    "mean_drafter_draft_energy_mj_per_draft_token": _fmt(
                        draft_per_draft
                    ),
                    "drafter_draft_per_draft_token_ratio_vs_baseline_gamma": _fmt(
                        _ratio(
                            draft_per_draft,
                            baseline_metrics[
                                "drafter_draft_energy_mj_per_draft_token"
                            ],
                        )
                    ),
                    "paired_mean_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma": _fmt(
                        _mean(paired_draft_per_draft_changes)
                        if paired_draft_per_draft_changes
                        else None
                    ),
                    "paired_ci95_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma": _fmt(
                        _ci95(paired_draft_per_draft_changes)
                        if paired_draft_per_draft_changes
                        else None
                    ),
                    "paired_bootstrap_ci95_low_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma": _fmt(
                        draft_per_draft_bootstrap_low
                    ),
                    "paired_bootstrap_ci95_high_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma": _fmt(
                        draft_per_draft_bootstrap_high
                    ),
                    "paired_sign_test_p_value_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma": _fmt(
                        _sign_test_p_value(paired_draft_per_draft_changes)
                    ),
                    "mean_drafter_draft_active_energy_mj_per_draft_token": _fmt(
                        draft_active_per_draft
                    ),
                    "drafter_draft_active_per_draft_token_ratio_vs_baseline_gamma": _fmt(
                        _ratio(
                            draft_active_per_draft,
                            baseline_metrics[
                                "drafter_draft_active_energy_mj_per_draft_token"
                            ],
                        )
                    ),
                    "paired_mean_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma": _fmt(
                        _mean(paired_draft_active_per_draft_changes)
                        if paired_draft_active_per_draft_changes
                        else None
                    ),
                    "paired_ci95_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma": _fmt(
                        _ci95(paired_draft_active_per_draft_changes)
                        if paired_draft_active_per_draft_changes
                        else None
                    ),
                    "paired_bootstrap_ci95_low_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma": _fmt(
                        draft_active_per_draft_bootstrap_low
                    ),
                    "paired_bootstrap_ci95_high_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma": _fmt(
                        draft_active_per_draft_bootstrap_high
                    ),
                    "paired_sign_test_p_value_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma": _fmt(
                        _sign_test_p_value(paired_draft_active_per_draft_changes)
                    ),
                    "log2_gamma_slope_drafter_total_energy_mj_per_token": _fmt(
                        total_slope
                    ),
                    "log2_gamma_slope_drafter_active_energy_mj_per_token": _fmt(
                        active_slope
                    ),
                    "log2_gamma_slope_drafter_draft_energy_mj_per_token": _fmt(
                        draft_slope
                    ),
                    "log2_gamma_slope_drafter_draft_active_energy_mj_per_token": _fmt(
                        draft_active_slope
                    ),
                    "pearson_log2_gamma_drafter_total_energy": _fmt(total_corr),
                    "pearson_log2_gamma_drafter_active_energy": _fmt(active_corr),
                    "pearson_log2_gamma_drafter_draft_energy": _fmt(draft_corr),
                    "pearson_log2_gamma_drafter_draft_active_energy": _fmt(
                        draft_active_corr
                    ),
                }
            )
    return output_rows


FIELDNAMES = [
    "gamma",
    "baseline_gamma",
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
    "gamma_count",
    "runs",
    "prompts",
    "common_prompts",
    "complete_prompt_overlap",
    "paired_prompts_vs_baseline_gamma",
    "complete_prompt_overlap_vs_baseline_gamma",
    "mean_generated_tokens",
    "mean_draft_tokens",
    "mean_accept_rate",
    "mean_tokens_per_s",
    "mean_wall_latency_ms",
    "mean_drafter_total_energy_mj_per_generated_token",
    "ci95_drafter_total_energy_mj_per_generated_token",
    "drafter_total_energy_ratio_vs_baseline_gamma",
    "drafter_total_energy_change_pct_vs_baseline_gamma",
    "paired_mean_drafter_total_energy_change_pct_vs_baseline_gamma",
    "paired_ci95_drafter_total_energy_change_pct_vs_baseline_gamma",
    "paired_bootstrap_ci95_low_drafter_total_energy_change_pct_vs_baseline_gamma",
    "paired_bootstrap_ci95_high_drafter_total_energy_change_pct_vs_baseline_gamma",
    "paired_sign_test_p_value_drafter_total_energy_change_pct_vs_baseline_gamma",
    "mean_drafter_active_energy_mj_per_generated_token",
    "drafter_active_energy_ratio_vs_baseline_gamma",
    "paired_mean_drafter_active_energy_change_pct_vs_baseline_gamma",
    "paired_ci95_drafter_active_energy_change_pct_vs_baseline_gamma",
    "paired_bootstrap_ci95_low_drafter_active_energy_change_pct_vs_baseline_gamma",
    "paired_bootstrap_ci95_high_drafter_active_energy_change_pct_vs_baseline_gamma",
    "paired_sign_test_p_value_drafter_active_energy_change_pct_vs_baseline_gamma",
    "mean_drafter_draft_energy_mj_per_generated_token",
    "drafter_draft_energy_ratio_vs_baseline_gamma",
    "paired_mean_drafter_draft_energy_change_pct_vs_baseline_gamma",
    "paired_ci95_drafter_draft_energy_change_pct_vs_baseline_gamma",
    "paired_bootstrap_ci95_low_drafter_draft_energy_change_pct_vs_baseline_gamma",
    "paired_bootstrap_ci95_high_drafter_draft_energy_change_pct_vs_baseline_gamma",
    "paired_sign_test_p_value_drafter_draft_energy_change_pct_vs_baseline_gamma",
    "mean_drafter_draft_active_energy_mj_per_generated_token",
    "drafter_draft_active_energy_ratio_vs_baseline_gamma",
    "paired_mean_drafter_draft_active_energy_change_pct_vs_baseline_gamma",
    "paired_ci95_drafter_draft_active_energy_change_pct_vs_baseline_gamma",
    "paired_bootstrap_ci95_low_drafter_draft_active_energy_change_pct_vs_baseline_gamma",
    "paired_bootstrap_ci95_high_drafter_draft_active_energy_change_pct_vs_baseline_gamma",
    "paired_sign_test_p_value_drafter_draft_active_energy_change_pct_vs_baseline_gamma",
    "mean_drafter_draft_energy_mj_per_draft_token",
    "drafter_draft_per_draft_token_ratio_vs_baseline_gamma",
    "paired_mean_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma",
    "paired_ci95_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma",
    "paired_bootstrap_ci95_low_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma",
    "paired_bootstrap_ci95_high_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma",
    "paired_sign_test_p_value_drafter_draft_per_draft_token_change_pct_vs_baseline_gamma",
    "mean_drafter_draft_active_energy_mj_per_draft_token",
    "drafter_draft_active_per_draft_token_ratio_vs_baseline_gamma",
    "paired_mean_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma",
    "paired_ci95_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma",
    "paired_bootstrap_ci95_low_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma",
    "paired_bootstrap_ci95_high_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma",
    "paired_sign_test_p_value_drafter_draft_active_per_draft_token_change_pct_vs_baseline_gamma",
    "log2_gamma_slope_drafter_total_energy_mj_per_token",
    "log2_gamma_slope_drafter_active_energy_mj_per_token",
    "log2_gamma_slope_drafter_draft_energy_mj_per_token",
    "log2_gamma_slope_drafter_draft_active_energy_mj_per_token",
    "pearson_log2_gamma_drafter_total_energy",
    "pearson_log2_gamma_drafter_active_energy",
    "pearson_log2_gamma_drafter_draft_energy",
    "pearson_log2_gamma_drafter_draft_active_energy",
]


def read_csvs(paths: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["_source"] = path
                rows.append(row)
    return rows


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze how gamma changes Jetson drafter energy."
    )
    parser.add_argument("--input", nargs="+", required=True, help="Raw spec CSV file(s)")
    parser.add_argument("--out", default="gamma_effect_summary.csv")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--allow-incomplete-energy",
        action="store_true",
        help="Include sessions without complete drafter/verifier energy samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize(
        read_csvs(args.input),
        allow_incomplete_energy=args.allow_incomplete_energy,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_csv(args.out, rows)
    print(f"gamma_effect_rows={len(rows)}")
    if rows:
        best = min(
            rows,
            key=lambda row: float(
                row["mean_drafter_total_energy_mj_per_generated_token"]
            ),
        )
        print(
            "best_drafter_energy: "
            f"gamma={best['gamma']} "
            f"f_draft={best['drafter_freq_hz']} "
            f"f_verify={best['verifier_clock_mhz']} "
            "E_draft_total/token="
            f"{best['mean_drafter_total_energy_mj_per_generated_token']}mJ"
        )
    print(f"Wrote {len(rows)} gamma-effect rows to {args.out}")


if __name__ == "__main__":
    main()
