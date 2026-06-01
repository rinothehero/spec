#!/usr/bin/env python3
"""Summarize joint drafter/verifier frequency sweeps for phase 1."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


PRIMARY_RAIL = "tot_power"
LATENCY_SLACK = 0.05


SUMMARY_FIELDS = [
    "gamma",
    "verifier_freq_hz",
    "drafter_freq_hz",
    "n",
    "generated_tokens_mean",
    "accept_rate_mean",
    "wall_latency_ms_mean",
    "wall_latency_ms_std",
    "tokens_per_s_mean",
    "system_energy_mj_mean",
    "energy_per_token_mj_mean",
    "energy_per_token_mj_std",
    "drafter_energy_mj_mean",
    "verifier_energy_mj_mean",
    "system_energy_complete_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--primary-rail", default=PRIMARY_RAIL)
    parser.add_argument("--latency-slack", type=float, default=LATENCY_SLACK)
    return parser.parse_args()


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.{digits}f}"


def freq_label_to_hz(label: str) -> int:
    return int(round(float(label.replace("p", ".")) * 1_000_000))


def verifier_freq_from_name(path: Path) -> int:
    match = re.search(r"fv([0-9]+(?:p[0-9]+)?)(?:mhz)?(?:_|$)", path.name)
    if match:
        return freq_label_to_hz(match.group(1))
    match = re.search(r"fv([0-9]+)hz", path.name)
    if match:
        return int(match.group(1))
    return 0


def row_drafter_freq_hz(row: dict[str, str]) -> int:
    return as_int(row, "drafter_freq_hz") or as_int(row, "drafter_jetson_gpu_freq_hz")


def row_verifier_freq_hz(row: dict[str, str], source_path: Path) -> int:
    explicit = as_int(row, "verifier_freq_hz")
    if explicit:
        return explicit
    clock_mhz = as_float(row, "verifier_clock_mhz")
    if not math.isnan(clock_mhz) and clock_mhz > 0:
        return int(round(clock_mhz * 1_000_000))
    gpu_clock_mhz = as_float(row, "verifier_gpu_clock_mhz")
    if not math.isnan(gpu_clock_mhz) and gpu_clock_mhz > 0:
        return int(round(gpu_clock_mhz * 1_000_000))
    return verifier_freq_from_name(source_path)


def energy_complete(row: dict[str, str]) -> float:
    return (
        1.0
        if str(row.get("system_energy_complete", "")).strip()
        in ("1", "1.0", "true", "True")
        else 0.0
    )


def read_rows(raw_dir: Path, primary_rail: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(raw_dir.glob("*.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("rail") != primary_rail:
                    continue
                row = dict(row)
                row["source_file"] = path.name
                row["drafter_freq_hz"] = str(row_drafter_freq_hz(row))
                row["verifier_freq_hz"] = str(row_verifier_freq_hz(row, path))
                rows.append(row)
    return rows


def write_csv(
    path: Path,
    rows: Iterable[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                as_int(row, "gamma"),
                as_int(row, "verifier_freq_hz"),
                as_int(row, "drafter_freq_hz"),
            )
        ].append(row)

    out: list[dict[str, object]] = []
    for (gamma, verifier_freq, drafter_freq), items in sorted(groups.items()):
        latencies = [as_float(r, "wall_latency_ms") for r in items]
        epts = [as_float(r, "system_total_energy_mj_per_generated_token") for r in items]
        out.append(
            {
                "gamma": gamma,
                "verifier_freq_hz": verifier_freq,
                "drafter_freq_hz": drafter_freq,
                "n": len(items),
                "generated_tokens_mean": fmt(mean(as_float(r, "generated_tokens") for r in items)),
                "accept_rate_mean": fmt(mean(as_float(r, "accept_rate") for r in items), 6),
                "wall_latency_ms_mean": fmt(mean(latencies)),
                "wall_latency_ms_std": fmt(pstdev(latencies) if len(latencies) > 1 else 0.0),
                "tokens_per_s_mean": fmt(mean(as_float(r, "tokens_per_s") for r in items), 6),
                "system_energy_mj_mean": fmt(mean(as_float(r, "system_total_energy_mj") for r in items)),
                "energy_per_token_mj_mean": fmt(mean(epts)),
                "energy_per_token_mj_std": fmt(pstdev(epts) if len(epts) > 1 else 0.0),
                "drafter_energy_mj_mean": fmt(mean(as_float(r, "drafter_total_energy_mj") for r in items)),
                "verifier_energy_mj_mean": fmt(mean(as_float(r, "verifier_total_energy_mj") for r in items)),
                "system_energy_complete_rate": fmt(mean(energy_complete(r) for r in items), 6),
            }
        )
    return out


def row_float(row: dict[str, object], key: str) -> float:
    return float(str(row[key]))


def max_freq_pair(rows: list[dict[str, object]]) -> dict[str, object]:
    max_verifier = max(int(r["verifier_freq_hz"]) for r in rows)
    max_drafter = max(int(r["drafter_freq_hz"]) for r in rows)
    candidates = [
        r
        for r in rows
        if int(r["verifier_freq_hz"]) == max_verifier
        and int(r["drafter_freq_hz"]) == max_drafter
    ]
    if candidates:
        return candidates[0]
    return max(rows, key=lambda r: (int(r["verifier_freq_hz"]), int(r["drafter_freq_hz"])))


def saving_percent(reference: dict[str, object], candidate: dict[str, object]) -> str:
    ref = row_float(reference, "energy_per_token_mj_mean")
    cand = row_float(candidate, "energy_per_token_mj_mean")
    if ref <= 0:
        return ""
    return fmt((ref - cand) / ref * 100.0, 3)


def best_pair_by_gamma(
    summary_rows: list[dict[str, object]],
    latency_slack: float,
) -> list[dict[str, object]]:
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        groups[int(row["gamma"])].append(row)

    out: list[dict[str, object]] = []
    for gamma, rows in sorted(groups.items()):
        fastest = min(rows, key=lambda r: row_float(r, "wall_latency_ms_mean"))
        energy_best = min(rows, key=lambda r: row_float(r, "energy_per_token_mj_mean"))
        budget = row_float(fastest, "wall_latency_ms_mean") * (1.0 + latency_slack)
        feasible = [r for r in rows if row_float(r, "wall_latency_ms_mean") <= budget]
        constrained = min(feasible, key=lambda r: row_float(r, "energy_per_token_mj_mean"))
        high = max_freq_pair(rows)
        out.append(
            {
                "gamma": gamma,
                "latency_budget_ms": fmt(budget),
                "fastest_verifier_freq_hz": fastest["verifier_freq_hz"],
                "fastest_drafter_freq_hz": fastest["drafter_freq_hz"],
                "fastest_latency_ms": fastest["wall_latency_ms_mean"],
                "energy_best_verifier_freq_hz": energy_best["verifier_freq_hz"],
                "energy_best_drafter_freq_hz": energy_best["drafter_freq_hz"],
                "energy_best_mj_per_token": energy_best["energy_per_token_mj_mean"],
                "latency_constrained_best_verifier_freq_hz": constrained["verifier_freq_hz"],
                "latency_constrained_best_drafter_freq_hz": constrained["drafter_freq_hz"],
                "latency_constrained_best_mj_per_token": constrained["energy_per_token_mj_mean"],
                "energy_saving_vs_max_freq_pair_percent": saving_percent(high, constrained),
            }
        )
    return out


def best_within_groups(
    summary_rows: list[dict[str, object]],
    group_fields: tuple[str, str],
    tuned_freq_field: str,
    latency_slack: float,
) -> list[dict[str, object]]:
    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        groups[(int(row[group_fields[0]]), int(row[group_fields[1]]))].append(row)

    out: list[dict[str, object]] = []
    for key, rows in sorted(groups.items()):
        fastest = min(rows, key=lambda r: row_float(r, "wall_latency_ms_mean"))
        energy_best = min(rows, key=lambda r: row_float(r, "energy_per_token_mj_mean"))
        budget = row_float(fastest, "wall_latency_ms_mean") * (1.0 + latency_slack)
        feasible = [r for r in rows if row_float(r, "wall_latency_ms_mean") <= budget]
        constrained = min(feasible, key=lambda r: row_float(r, "energy_per_token_mj_mean"))
        high = max(rows, key=lambda r: int(r[tuned_freq_field]))
        out_row: dict[str, object] = {
            group_fields[0]: key[0],
            group_fields[1]: key[1],
            "latency_budget_ms": fmt(budget),
            f"fastest_{tuned_freq_field}": fastest[tuned_freq_field],
            "fastest_latency_ms": fastest["wall_latency_ms_mean"],
            f"energy_best_{tuned_freq_field}": energy_best[tuned_freq_field],
            "energy_best_mj_per_token": energy_best["energy_per_token_mj_mean"],
            f"latency_constrained_best_{tuned_freq_field}": constrained[tuned_freq_field],
            "latency_constrained_best_mj_per_token": constrained["energy_per_token_mj_mean"],
            "energy_saving_vs_high_freq_percent": saving_percent(high, constrained),
        }
        out.append(out_row)
    return out


def write_markdown(
    path: Path,
    summary_rows: list[dict[str, object]],
    best_rows: list[dict[str, object]],
    source_count: int,
    primary_rail: str,
) -> None:
    lines = [
        "# Joint Frequency Sweep Summary",
        "",
        f"Primary rail: `{primary_rail}`.",
        "",
        f"Measured rows used: `{source_count}`.",
        "",
        "## Best Pair By Gamma",
        "",
        "| gamma | fastest vf | fastest fd | energy best vf | energy best fd | latency-constrained vf | latency-constrained fd | saving vs max pair |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            "| {gamma} | {fastest_verifier_freq_hz} | {fastest_drafter_freq_hz} | "
            "{energy_best_verifier_freq_hz} | {energy_best_drafter_freq_hz} | "
            "{latency_constrained_best_verifier_freq_hz} | "
            "{latency_constrained_best_drafter_freq_hz} | "
            "{energy_saving_vs_max_freq_pair_percent}% |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Aggregated Conditions",
            "",
            "| gamma | verifier freq Hz | drafter freq Hz | n | latency ms | tokens/s | energy/token mJ | accept rate |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {gamma} | {verifier_freq_hz} | {drafter_freq_hz} | {n} | "
            "{wall_latency_ms_mean} | {tokens_per_s_mean} | "
            "{energy_per_token_mj_mean} | {accept_rate_mean} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    rows = read_rows(raw_dir, args.primary_rail)
    if not rows:
        raise SystemExit(f"No {args.primary_rail} rows found in {raw_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "joint_filtered_tot_power_rows.csv", rows, list(rows[0].keys()))

    summary_rows = summarize(rows)
    write_csv(
        out_dir / "joint_summary_by_gamma_vf_fd.csv",
        summary_rows,
        SUMMARY_FIELDS,
    )

    best_pairs = best_pair_by_gamma(summary_rows, args.latency_slack)
    write_csv(
        out_dir / "joint_best_pair_by_gamma.csv",
        best_pairs,
        [
            "gamma",
            "latency_budget_ms",
            "fastest_verifier_freq_hz",
            "fastest_drafter_freq_hz",
            "fastest_latency_ms",
            "energy_best_verifier_freq_hz",
            "energy_best_drafter_freq_hz",
            "energy_best_mj_per_token",
            "latency_constrained_best_verifier_freq_hz",
            "latency_constrained_best_drafter_freq_hz",
            "latency_constrained_best_mj_per_token",
            "energy_saving_vs_max_freq_pair_percent",
        ],
    )

    best_drafter = best_within_groups(
        summary_rows,
        group_fields=("gamma", "verifier_freq_hz"),
        tuned_freq_field="drafter_freq_hz",
        latency_slack=args.latency_slack,
    )
    write_csv(
        out_dir / "joint_best_drafter_by_gamma_verifier.csv",
        best_drafter,
        [
            "gamma",
            "verifier_freq_hz",
            "latency_budget_ms",
            "fastest_drafter_freq_hz",
            "fastest_latency_ms",
            "energy_best_drafter_freq_hz",
            "energy_best_mj_per_token",
            "latency_constrained_best_drafter_freq_hz",
            "latency_constrained_best_mj_per_token",
            "energy_saving_vs_high_freq_percent",
        ],
    )

    best_verifier = best_within_groups(
        summary_rows,
        group_fields=("gamma", "drafter_freq_hz"),
        tuned_freq_field="verifier_freq_hz",
        latency_slack=args.latency_slack,
    )
    write_csv(
        out_dir / "joint_best_verifier_by_gamma_drafter.csv",
        best_verifier,
        [
            "gamma",
            "drafter_freq_hz",
            "latency_budget_ms",
            "fastest_verifier_freq_hz",
            "fastest_latency_ms",
            "energy_best_verifier_freq_hz",
            "energy_best_mj_per_token",
            "latency_constrained_best_verifier_freq_hz",
            "latency_constrained_best_mj_per_token",
            "energy_saving_vs_high_freq_percent",
        ],
    )

    write_markdown(
        out_dir / "joint_summary.md",
        summary_rows,
        best_pairs,
        len(rows),
        args.primary_rail,
    )
    print(f"rows={len(rows)} groups={len(summary_rows)} out={out_dir}")


if __name__ == "__main__":
    main()
