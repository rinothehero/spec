import argparse
import csv
import hashlib
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def _float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def _float_first(row: Dict[str, str], *keys: str) -> float:
    for key in keys:
        value = row.get(key, "")
        if value not in ("", None):
            return float(value)
    return 0.0


def _is_truthy(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "y")


def _preferred_rail(row: Dict[str, str]) -> str:
    if row.get("algorithm", "speculative") == "verifier_only":
        return row.get("verifier_primary_power_rail", "") or "verifier_gpu_power"
    return row.get("drafter_primary_power_rail", "") or "tot_power"


def _session_row_score(row: Dict[str, str]) -> Tuple[int, int, int]:
    rail = row.get("rail", "")
    preferred_rail = _preferred_rail(row)
    primary_rail = int(bool(preferred_rail) and rail == preferred_rail)
    has_total_energy = int(bool(row.get("system_total_energy_mj_per_generated_token", "")))
    has_active_energy = int(bool(row.get("system_active_energy_mj_per_generated_token", "")))
    return primary_rail, has_total_energy, has_active_energy


def _session_rows(
    rows: Iterable[Dict[str, str]],
    allow_incomplete_energy: bool,
) -> List[Dict[str, str]]:
    by_session: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if not allow_incomplete_energy and not _is_truthy(
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


def _mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _stderr(values: List[float]) -> float:
    return _stdev(values) / (len(values) ** 0.5) if len(values) >= 2 else 0.0


def _ci95_normal(values: List[float]) -> float:
    return 1.96 * _stderr(values)


def _mean_ratio(numerators: List[float], denominators: List[float]) -> float:
    ratios = [
        numerator / denominator
        for numerator, denominator in zip(numerators, denominators)
        if denominator > 0
    ]
    return _mean(ratios)


def _per_token(values: List[float], tokens: List[float]) -> List[float]:
    return [
        value / token
        for value, token in zip(values, tokens)
        if token > 0
    ]


def _single_value(rows: Iterable[Dict[str, str]], key: str) -> str:
    values = sorted({row.get(key, "") for row in rows if row.get(key, "") != ""})
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return "mixed"


def _prompt_set_sha256(rows: Iterable[Dict[str, str]]) -> str:
    explicit = sorted(
        {row.get("prompt_set_sha256", "") for row in rows if row.get("prompt_set_sha256", "")}
    )
    if len(explicit) == 1:
        return explicit[0]
    if len(explicit) > 1:
        return "mixed"

    prompt_hashes = sorted({row.get("prompt_sha256", "") for row in rows})
    prompt_hashes = [value for value in prompt_hashes if value]
    if not prompt_hashes:
        return ""
    return hashlib.sha256("\n".join(prompt_hashes).encode("utf-8")).hexdigest()


def summarize(
    rows: List[Dict[str, str]],
    allow_incomplete_energy: bool = False,
) -> List[Dict[str, object]]:
    sessions = _session_rows(rows, allow_incomplete_energy)
    groups: Dict[
        Tuple[str, str, str, str, str, str, str, str, str, str, str, str, str],
        List[Dict[str, str]],
    ] = defaultdict(list)
    for row in sessions:
        groups[
            (
                row.get("algorithm", "speculative"),
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
        ].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (
        algorithm,
        gamma,
        drafter_freq_hz,
        verifier_clock_mhz,
        decoding_mode,
        max_new_tokens,
        stop_token_policy,
        stop_token_ids,
        prompt_set_sha256,
        drafter_model,
        verifier_model,
        drafter_runtime_fingerprint,
        verifier_runtime_fingerprint,
    ), items in sorted(groups.items()):
        system_energy = [_float(row, "system_total_energy_mj") for row in items]
        energy_per_token = [
            _float(row, "system_total_energy_mj_per_generated_token") for row in items
        ]
        drafter_total_energy = [_float(row, "drafter_total_energy_mj") for row in items]
        verifier_total_energy = [_float(row, "verifier_total_energy_mj") for row in items]
        drafter_draft_energy = [
            _float_first(row, "drafter_draft_total_energy_mj", "drafter_draft_energy_mj")
            for row in items
        ]
        drafter_draft_active_energy = [
            _float_first(
                row,
                "drafter_draft_active_energy_mj",
                "drafter_draft_total_energy_mj",
                "drafter_draft_energy_mj",
            )
            for row in items
        ]
        verifier_verify_energy = [
            _float_first(row, "verifier_verify_total_energy_mj", "verifier_verify_energy_mj")
            for row in items
        ]
        active_system_energy = [
            _float_first(row, "system_active_energy_mj", "system_total_energy_mj")
            for row in items
        ]
        active_energy_per_token = [
            _float_first(
                row,
                "system_active_energy_mj_per_generated_token",
                "system_total_energy_mj_per_generated_token",
            )
            for row in items
        ]
        drafter_active_energy = [
            _float_first(row, "drafter_active_energy_mj", "drafter_total_energy_mj")
            for row in items
        ]
        verifier_active_energy = [
            _float_first(row, "verifier_active_energy_mj", "verifier_total_energy_mj")
            for row in items
        ]
        idle_baseline_count = sum(1 for row in items if row.get("idle_baseline_s", ""))
        tokens_per_s = [_float(row, "tokens_per_s") for row in items]
        accept_rate = [_float(row, "accept_rate") for row in items]
        wall_latency = [_float(row, "wall_latency_ms") for row in items]
        client_rpc_latency = [_float(row, "client_rpc_latency_ms") for row in items]
        server_compute_latency = [
            _float(row, "server_compute_latency_ms") for row in items
        ]
        estimated_rpc_overhead = [
            _float(row, "estimated_rpc_overhead_ms") for row in items
        ]
        rpc_total_bytes = [_float(row, "rpc_total_bytes") for row in items]
        rpc_bytes_per_token = [
            _float(row, "rpc_bytes_per_generated_token") for row in items
        ]
        generated = [_float(row, "generated_tokens") for row in items]
        drafter_total_energy_per_token = _per_token(drafter_total_energy, generated)
        drafter_active_energy_per_token = _per_token(drafter_active_energy, generated)
        drafter_draft_energy_per_token = _per_token(drafter_draft_energy, generated)
        drafter_draft_active_energy_per_token = _per_token(
            drafter_draft_active_energy,
            generated,
        )
        stop_reasons = sorted(
            {
                row.get("stop_reason", "")
                for row in items
                if row.get("stop_reason", "")
            }
        )
        eos_stop_runs = sum(
            1 for row in items if row.get("stop_reason", "").startswith("eos")
        )
        stop_token_stop_runs = sum(
            1 for row in items if row.get("stop_reason", "").startswith("stop")
        )
        max_token_stop_runs = sum(
            1 for row in items if row.get("stop_reason", "") == "max_new_tokens"
        )
        prompt_ids = {row.get("prompt_id", "") for row in items}
        summary_rows.append(
            {
                "algorithm": algorithm,
                "system_boundary": _single_value(items, "system_boundary"),
                "gamma": gamma,
                "drafter_freq_hz": drafter_freq_hz,
                "verifier_clock_mhz": verifier_clock_mhz,
                "decoding_mode": decoding_mode,
                "max_new_tokens": max_new_tokens or _single_value(items, "max_new_tokens"),
                "stop_token_policy": (
                    stop_token_policy or _single_value(items, "stop_token_policy")
                ),
                "stop_token_ids": stop_token_ids or _single_value(items, "stop_token_ids"),
                "prompt_set_sha256": prompt_set_sha256 or _prompt_set_sha256(items),
                "drafter_model": drafter_model or _single_value(items, "drafter_model"),
                "verifier_model": verifier_model or _single_value(items, "verifier_model"),
                "drafter_runtime_fingerprint": (
                    drafter_runtime_fingerprint
                    or _single_value(items, "drafter_runtime_fingerprint")
                ),
                "verifier_runtime_fingerprint": (
                    verifier_runtime_fingerprint
                    or _single_value(items, "verifier_runtime_fingerprint")
                ),
                "runs": len(items),
                "prompts": len(prompt_ids),
                "mean_generated_tokens": f"{_mean(generated):.3f}",
                "stop_reasons": ",".join(stop_reasons),
                "eos_stop_runs": eos_stop_runs,
                "stop_token_stop_runs": stop_token_stop_runs,
                "max_token_stop_runs": max_token_stop_runs,
                "mean_accept_rate": f"{_mean(accept_rate):.6f}",
                "stdev_accept_rate": f"{_stdev(accept_rate):.6f}",
                "stderr_accept_rate": f"{_stderr(accept_rate):.6f}",
                "ci95_accept_rate": f"{_ci95_normal(accept_rate):.6f}",
                "mean_tokens_per_s": f"{_mean(tokens_per_s):.6f}",
                "stdev_tokens_per_s": f"{_stdev(tokens_per_s):.6f}",
                "stderr_tokens_per_s": f"{_stderr(tokens_per_s):.6f}",
                "ci95_tokens_per_s": f"{_ci95_normal(tokens_per_s):.6f}",
                "mean_wall_latency_ms": f"{_mean(wall_latency):.6f}",
                "stdev_wall_latency_ms": f"{_stdev(wall_latency):.6f}",
                "stderr_wall_latency_ms": f"{_stderr(wall_latency):.6f}",
                "ci95_wall_latency_ms": f"{_ci95_normal(wall_latency):.6f}",
                "mean_client_rpc_latency_ms": f"{_mean(client_rpc_latency):.6f}",
                "mean_server_compute_latency_ms": f"{_mean(server_compute_latency):.6f}",
                "mean_estimated_rpc_overhead_ms": f"{_mean(estimated_rpc_overhead):.6f}",
                "mean_rpc_total_bytes": f"{_mean(rpc_total_bytes):.6f}",
                "mean_rpc_bytes_per_token": f"{_mean(rpc_bytes_per_token):.6f}",
                "mean_system_total_energy_mj": f"{_mean(system_energy):.6f}",
                "stdev_system_total_energy_mj": f"{_stdev(system_energy):.6f}",
                "mean_system_energy_mj_per_token": f"{_mean(energy_per_token):.6f}",
                "stdev_system_energy_mj_per_token": f"{_stdev(energy_per_token):.6f}",
                "stderr_system_energy_mj_per_token": f"{_stderr(energy_per_token):.6f}",
                "ci95_system_energy_mj_per_token": f"{_ci95_normal(energy_per_token):.6f}",
                "mean_active_system_energy_mj": f"{_mean(active_system_energy):.6f}",
                "stdev_active_system_energy_mj": f"{_stdev(active_system_energy):.6f}",
                "mean_active_system_energy_mj_per_token": f"{_mean(active_energy_per_token):.6f}",
                "stdev_active_system_energy_mj_per_token": f"{_stdev(active_energy_per_token):.6f}",
                "stderr_active_system_energy_mj_per_token": f"{_stderr(active_energy_per_token):.6f}",
                "ci95_active_system_energy_mj_per_token": f"{_ci95_normal(active_energy_per_token):.6f}",
                "mean_drafter_total_energy_mj": f"{_mean(drafter_total_energy):.6f}",
                "mean_verifier_total_energy_mj": f"{_mean(verifier_total_energy):.6f}",
                "mean_drafter_active_energy_mj": f"{_mean(drafter_active_energy):.6f}",
                "mean_verifier_active_energy_mj": f"{_mean(verifier_active_energy):.6f}",
                "mean_drafter_draft_energy_mj": f"{_mean(drafter_draft_energy):.6f}",
                "mean_drafter_draft_active_energy_mj": f"{_mean(drafter_draft_active_energy):.6f}",
                "mean_verifier_verify_energy_mj": f"{_mean(verifier_verify_energy):.6f}",
                "mean_drafter_total_energy_mj_per_token": f"{_mean(drafter_total_energy_per_token):.6f}",
                "stdev_drafter_total_energy_mj_per_token": f"{_stdev(drafter_total_energy_per_token):.6f}",
                "stderr_drafter_total_energy_mj_per_token": f"{_stderr(drafter_total_energy_per_token):.6f}",
                "ci95_drafter_total_energy_mj_per_token": f"{_ci95_normal(drafter_total_energy_per_token):.6f}",
                "mean_drafter_active_energy_mj_per_token": f"{_mean(drafter_active_energy_per_token):.6f}",
                "stdev_drafter_active_energy_mj_per_token": f"{_stdev(drafter_active_energy_per_token):.6f}",
                "stderr_drafter_active_energy_mj_per_token": f"{_stderr(drafter_active_energy_per_token):.6f}",
                "ci95_drafter_active_energy_mj_per_token": f"{_ci95_normal(drafter_active_energy_per_token):.6f}",
                "mean_drafter_draft_energy_mj_per_token": f"{_mean(drafter_draft_energy_per_token):.6f}",
                "stdev_drafter_draft_energy_mj_per_token": f"{_stdev(drafter_draft_energy_per_token):.6f}",
                "stderr_drafter_draft_energy_mj_per_token": f"{_stderr(drafter_draft_energy_per_token):.6f}",
                "ci95_drafter_draft_energy_mj_per_token": f"{_ci95_normal(drafter_draft_energy_per_token):.6f}",
                "mean_drafter_draft_active_energy_mj_per_token": f"{_mean(drafter_draft_active_energy_per_token):.6f}",
                "stdev_drafter_draft_active_energy_mj_per_token": f"{_stdev(drafter_draft_active_energy_per_token):.6f}",
                "stderr_drafter_draft_active_energy_mj_per_token": f"{_stderr(drafter_draft_active_energy_per_token):.6f}",
                "ci95_drafter_draft_active_energy_mj_per_token": f"{_ci95_normal(drafter_draft_active_energy_per_token):.6f}",
                "mean_drafter_energy_share": f"{_mean_ratio(drafter_total_energy, system_energy):.6f}",
                "idle_baseline_runs": idle_baseline_count,
            }
        )
    return summary_rows


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "gamma",
        "algorithm",
        "system_boundary",
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
        "mean_generated_tokens",
        "stop_reasons",
        "eos_stop_runs",
        "stop_token_stop_runs",
        "max_token_stop_runs",
        "mean_accept_rate",
        "stdev_accept_rate",
        "stderr_accept_rate",
        "ci95_accept_rate",
        "mean_tokens_per_s",
        "stdev_tokens_per_s",
        "stderr_tokens_per_s",
        "ci95_tokens_per_s",
        "mean_wall_latency_ms",
        "stdev_wall_latency_ms",
        "stderr_wall_latency_ms",
        "ci95_wall_latency_ms",
        "mean_client_rpc_latency_ms",
        "mean_server_compute_latency_ms",
        "mean_estimated_rpc_overhead_ms",
        "mean_rpc_total_bytes",
        "mean_rpc_bytes_per_token",
        "mean_system_total_energy_mj",
        "stdev_system_total_energy_mj",
        "mean_system_energy_mj_per_token",
        "stdev_system_energy_mj_per_token",
        "stderr_system_energy_mj_per_token",
        "ci95_system_energy_mj_per_token",
        "mean_active_system_energy_mj",
        "stdev_active_system_energy_mj",
        "mean_active_system_energy_mj_per_token",
        "stdev_active_system_energy_mj_per_token",
        "stderr_active_system_energy_mj_per_token",
        "ci95_active_system_energy_mj_per_token",
        "mean_drafter_total_energy_mj",
        "mean_verifier_total_energy_mj",
        "mean_drafter_active_energy_mj",
        "mean_verifier_active_energy_mj",
        "mean_drafter_draft_energy_mj",
        "mean_drafter_draft_active_energy_mj",
        "mean_verifier_verify_energy_mj",
        "mean_drafter_total_energy_mj_per_token",
        "stdev_drafter_total_energy_mj_per_token",
        "stderr_drafter_total_energy_mj_per_token",
        "ci95_drafter_total_energy_mj_per_token",
        "mean_drafter_active_energy_mj_per_token",
        "stdev_drafter_active_energy_mj_per_token",
        "stderr_drafter_active_energy_mj_per_token",
        "ci95_drafter_active_energy_mj_per_token",
        "mean_drafter_draft_energy_mj_per_token",
        "stdev_drafter_draft_energy_mj_per_token",
        "stderr_drafter_draft_energy_mj_per_token",
        "ci95_drafter_draft_energy_mj_per_token",
        "mean_drafter_draft_active_energy_mj_per_token",
        "stdev_drafter_draft_active_energy_mj_per_token",
        "stderr_drafter_draft_active_energy_mj_per_token",
        "ci95_drafter_draft_active_energy_mj_per_token",
        "mean_drafter_energy_share",
        "idle_baseline_runs",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csvs(paths: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def print_best(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("No rows to summarize.")
        return
    energy_key = "mean_drafter_active_energy_mj_per_token"
    candidates = [
        row
        for row in rows
        if float(row.get(energy_key, 0.0) or 0.0) > 0
    ]
    if not candidates:
        energy_key = "mean_system_energy_mj_per_token"
        candidates = rows
    best = min(candidates, key=lambda row: float(row[energy_key]))
    print(
        "best_energy_per_token: "
        f"algorithm={best['algorithm']} "
        f"gamma={best['gamma']} "
        f"f_draft={best['drafter_freq_hz']} "
        f"f_verify={best['verifier_clock_mhz']} "
        f"{energy_key}={best[energy_key]}mJ "
        f"tok/s={best['mean_tokens_per_s']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize spec decoding CSV results")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Raw CSV file(s) from spec_driver or verifier_baseline_driver",
    )
    parser.add_argument("--out", default="spec_summary.csv", help="Summary CSV")
    parser.add_argument(
        "--allow-incomplete-energy",
        action="store_true",
        help="Include runs without both drafter and verifier power samples",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csvs(args.input)
    all_sessions = len(_session_rows(rows, allow_incomplete_energy=True))
    complete_sessions = len(_session_rows(rows, allow_incomplete_energy=False))
    if not args.allow_incomplete_energy and all_sessions != complete_sessions:
        print(
            "Dropped "
            f"{all_sessions - complete_sessions}/{all_sessions} sessions "
            "without complete required energy."
        )
    summary_rows = summarize(rows, allow_incomplete_energy=args.allow_incomplete_energy)
    write_csv(args.out, summary_rows)
    print_best(summary_rows)
    print(f"Wrote {len(summary_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
