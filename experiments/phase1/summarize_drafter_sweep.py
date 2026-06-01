#!/usr/bin/env python3
"""Summarize phase-1 drafter-frequency sweep outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


PRIMARY_RAIL = "tot_power"
HIGH_DRAFTER_FREQ_HZ = 624_750_000
LATENCY_OVERHEAD = 0.05


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value not in ("", None) else 0


def read_rows(raw_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(raw_dir.glob("*.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("rail") == PRIMARY_RAIL:
                    row = dict(row)
                    row["source_file"] = path.name
                    rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(as_int(row, "gamma"), as_int(row, "drafter_freq_hz"))].append(row)

    out: list[dict[str, object]] = []
    for (gamma, freq), group_rows in sorted(groups.items()):
        latency = [as_float(r, "wall_latency_ms") for r in group_rows]
        tps = [as_float(r, "tokens_per_s") for r in group_rows]
        energy = [as_float(r, "system_total_energy_mj_per_generated_token") for r in group_rows]
        drafter_energy = [as_float(r, "drafter_total_energy_mj") for r in group_rows]
        verifier_energy = [as_float(r, "verifier_total_energy_mj") for r in group_rows]
        accept = [as_float(r, "accept_rate") for r in group_rows]
        generated = [as_float(r, "generated_tokens") for r in group_rows]
        complete = [
            1.0 if str(r.get("system_energy_complete", "")).strip() in ("1", "1.0", "true", "True") else 0.0
            for r in group_rows
        ]
        out.append(
            {
                "gamma": gamma,
                "drafter_freq_hz": freq,
                "n": len(group_rows),
                "generated_tokens_mean": f"{mean(generated):.3f}",
                "accept_rate_mean": f"{mean(accept):.6f}",
                "wall_latency_ms_mean": f"{mean(latency):.3f}",
                "wall_latency_ms_std": f"{maybe_stdev(latency):.3f}",
                "tokens_per_s_mean": f"{mean(tps):.6f}",
                "system_energy_mj_mean": f"{mean([as_float(r, 'system_total_energy_mj') for r in group_rows]):.3f}",
                "energy_per_token_mj_mean": f"{mean(energy):.3f}",
                "energy_per_token_mj_std": f"{maybe_stdev(energy):.3f}",
                "drafter_energy_mj_mean": f"{mean(drafter_energy):.3f}",
                "verifier_energy_mj_mean": f"{mean(verifier_energy):.3f}",
                "system_energy_complete_rate": f"{mean(complete):.3f}",
            }
        )
    return out


def best_by_gamma(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        groups[int(row["gamma"])].append(row)

    out: list[dict[str, object]] = []
    for gamma, rows in sorted(groups.items()):
        fastest = min(rows, key=lambda r: float(r["wall_latency_ms_mean"]))
        energy_best = min(rows, key=lambda r: float(r["energy_per_token_mj_mean"]))
        latency_budget = float(fastest["wall_latency_ms_mean"]) * (1 + LATENCY_OVERHEAD)
        eligible = [r for r in rows if float(r["wall_latency_ms_mean"]) <= latency_budget]
        constrained = min(eligible, key=lambda r: float(r["energy_per_token_mj_mean"]))
        high = next((r for r in rows if int(r["drafter_freq_hz"]) == HIGH_DRAFTER_FREQ_HZ), None)
        if high is None:
            high = max(rows, key=lambda r: int(r["drafter_freq_hz"]))
        saving = (
            (float(high["energy_per_token_mj_mean"]) - float(constrained["energy_per_token_mj_mean"]))
            / float(high["energy_per_token_mj_mean"])
            * 100
        )
        out.append(
            {
                "gamma": gamma,
                "latency_budget_ms": f"{latency_budget:.3f}",
                "fastest_drafter_freq_hz": fastest["drafter_freq_hz"],
                "fastest_latency_ms": fastest["wall_latency_ms_mean"],
                "energy_best_drafter_freq_hz": energy_best["drafter_freq_hz"],
                "energy_best_mj_per_token": energy_best["energy_per_token_mj_mean"],
                "latency_constrained_best_drafter_freq_hz": constrained["drafter_freq_hz"],
                "latency_constrained_best_mj_per_token": constrained["energy_per_token_mj_mean"],
                "energy_saving_vs_high_percent": f"{saving:.3f}",
            }
        )
    return out


def prompt_best(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("prompt_id", ""), as_int(row, "gamma"), as_int(row, "drafter_freq_hz"))].append(row)

    by_prompt_gamma: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for (prompt_id, gamma, freq), group_rows in groups.items():
        energy = mean([as_float(r, "system_total_energy_mj_per_generated_token") for r in group_rows])
        latency = mean([as_float(r, "wall_latency_ms") for r in group_rows])
        by_prompt_gamma[(prompt_id, gamma)].append(
            {
                "prompt_id": prompt_id,
                "gamma": gamma,
                "drafter_freq_hz": freq,
                "energy_per_token_mj_mean": energy,
                "wall_latency_ms_mean": latency,
            }
        )

    out: list[dict[str, object]] = []
    for (prompt_id, gamma), candidates in sorted(by_prompt_gamma.items()):
        energy_best = min(candidates, key=lambda r: r["energy_per_token_mj_mean"])
        fastest = min(candidates, key=lambda r: r["wall_latency_ms_mean"])
        out.append(
            {
                "prompt_id": prompt_id,
                "gamma": gamma,
                "energy_best_drafter_freq_hz": energy_best["drafter_freq_hz"],
                "energy_best_mj_per_token": f"{energy_best['energy_per_token_mj_mean']:.3f}",
                "fastest_drafter_freq_hz": fastest["drafter_freq_hz"],
                "fastest_latency_ms": f"{fastest['wall_latency_ms_mean']:.3f}",
            }
        )
    return out


def plot_svg(path: Path, summary_rows: list[dict[str, object]], metric: str, title: str, y_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 820, 430
    margin_left, margin_right, margin_top, margin_bottom = 78, 28, 48, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    freqs = sorted({int(r["drafter_freq_hz"]) for r in summary_rows})
    gammas = sorted({int(r["gamma"]) for r in summary_rows})
    values = [float(r[metric]) for r in summary_rows]
    y_min = min(values)
    y_max = max(values)
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 1
    y_min -= y_pad
    y_max += y_pad

    def x(freq: int) -> float:
        if len(freqs) == 1:
            return margin_left + plot_w / 2
        return margin_left + plot_w * freqs.index(freq) / (len(freqs) - 1)

    def y(value: float) -> float:
        return margin_top + plot_h * (1 - (value - y_min) / (y_max - y_min))

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="26" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width-margin_right}" y2="{margin_top + plot_h}" stroke="#444"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#444"/>',
        f'<text x="{width/2:.1f}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">Drafter GPU frequency (MHz)</text>',
        f'<text x="18" y="{height/2:.1f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.1f})" font-family="Arial" font-size="13">{y_label}</text>',
    ]
    for freq in freqs:
        label = f"{freq/1_000_000:g}"
        lines.append(f'<text x="{x(freq):.1f}" y="{margin_top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>')
    for i in range(5):
        value = y_min + (y_max - y_min) * i / 4
        yy = y(value)
        lines.append(f'<line x1="{margin_left}" y1="{yy:.1f}" x2="{width-margin_right}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{margin_left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.0f}</text>')

    rows_by_gamma: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        rows_by_gamma[int(row["gamma"])].append(row)
    for idx, gamma in enumerate(gammas):
        color = colors[idx % len(colors)]
        points = []
        for row in sorted(rows_by_gamma[gamma], key=lambda r: int(r["drafter_freq_hz"])):
            points.append((x(int(row["drafter_freq_hz"])), y(float(row[metric]))))
        point_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{point_str}"/>')
        for px, py in points:
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{color}"/>')
        legend_x = margin_left + idx * 110
        legend_y = height - 43
        lines.append(f'<rect x="{legend_x}" y="{legend_y-10}" width="18" height="3" fill="{color}"/>')
        lines.append(f'<text x="{legend_x+24}" y="{legend_y-5}" font-family="Arial" font-size="12">gamma={gamma}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def write_markdown(path: Path, summary_rows: list[dict[str, object]], best_rows: list[dict[str, object]], source_count: int) -> None:
    lines = [
        "# Drafter Main Sweep Summary",
        "",
        f"Primary rail: `{PRIMARY_RAIL}`.",
        "",
        f"Measured rows used: `{source_count}`.",
        "",
        "## Best Drafter Frequency By Gamma",
        "",
        "| gamma | fastest freq | energy best freq | latency-constrained best freq | saving vs high |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            "| {gamma} | {fastest_drafter_freq_hz} | {energy_best_drafter_freq_hz} | "
            "{latency_constrained_best_drafter_freq_hz} | {energy_saving_vs_high_percent}% |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Aggregated Conditions",
            "",
            "| gamma | drafter freq Hz | n | latency ms | tokens/s | energy/token mJ | accept rate |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {gamma} | {drafter_freq_hz} | {n} | {wall_latency_ms_mean} | "
            "{tokens_per_s_mean} | {energy_per_token_mj_mean} | {accept_rate_mean} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    rows = read_rows(raw_dir)
    if not rows:
        raise SystemExit(f"No {PRIMARY_RAIL} rows found in {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered_fields = list(rows[0].keys())
    write_csv(out_dir / "drafter_main_filtered_tot_power_rows.csv", rows, filtered_fields)

    summary_rows = aggregate(rows)
    summary_fields = [
        "gamma",
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
    write_csv(out_dir / "drafter_main_summary_by_gamma_freq.csv", summary_rows, summary_fields)

    best_rows = best_by_gamma(summary_rows)
    best_fields = [
        "gamma",
        "latency_budget_ms",
        "fastest_drafter_freq_hz",
        "fastest_latency_ms",
        "energy_best_drafter_freq_hz",
        "energy_best_mj_per_token",
        "latency_constrained_best_drafter_freq_hz",
        "latency_constrained_best_mj_per_token",
        "energy_saving_vs_high_percent",
    ]
    write_csv(out_dir / "drafter_main_best_frequency_by_gamma.csv", best_rows, best_fields)

    prompt_rows = prompt_best(rows)
    prompt_fields = [
        "prompt_id",
        "gamma",
        "energy_best_drafter_freq_hz",
        "energy_best_mj_per_token",
        "fastest_drafter_freq_hz",
        "fastest_latency_ms",
    ]
    write_csv(out_dir / "drafter_main_prompt_best_frequency.csv", prompt_rows, prompt_fields)

    plot_svg(
        out_dir / "plots" / "drafter_energy_per_token_by_gamma.svg",
        summary_rows,
        "energy_per_token_mj_mean",
        "System energy per generated token by gamma",
        "Energy/token (mJ)",
    )
    plot_svg(
        out_dir / "plots" / "drafter_latency_by_gamma.svg",
        summary_rows,
        "wall_latency_ms_mean",
        "Latency by gamma",
        "Latency (ms)",
    )
    plot_svg(
        out_dir / "plots" / "drafter_total_energy_by_gamma.svg",
        summary_rows,
        "drafter_energy_mj_mean",
        "Drafter total energy by gamma",
        "Drafter energy (mJ)",
    )
    write_markdown(out_dir / "drafter_main_summary.md", summary_rows, best_rows, len(rows))
    print(f"rows={len(rows)} groups={len(summary_rows)} out={out_dir}")


if __name__ == "__main__":
    main()
