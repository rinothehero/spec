#!/usr/bin/env python3
"""Summarize phase-1 DSD gamma/frequency experiments.

The script expects SPEC result CSV files and filters to the `tot_power` rows,
which are the primary system-energy rows used in the phase-1 plan.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


PRIMARY_RAIL = "tot_power"
KEY_FIELDS = ("experiment", "gamma", "verifier_freq_hz")
SUMMARY_FIELDS = [
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
    parser.add_argument("--latency-slack", type=float, default=0.05)
    return parser.parse_args()


def verifier_freq_from_name(path: Path) -> str:
    name = path.name
    if "fv408" in name or "408mhz" in name:
        return "408000000"
    if "fv612" in name or "612mhz" in name:
        return "612000000"
    if "fv816" in name or "816mhz" in name:
        return "816000000"
    if "fv1300p5" in name or "1300p5" in name:
        return "1300500000"
    return ""


def read_rows(raw_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(raw_dir.glob("*.csv")):
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("rail") != PRIMARY_RAIL:
                    continue
                row = dict(row)
                row["source_file"] = path.name
                row["verifier_freq_hz"] = row.get("verifier_freq_hz") or verifier_freq_from_name(path)
                row["experiment"] = row.get("experiment") or path.stem
                rows.append(row)
    return rows


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k, "") for k in KEY_FIELDS)
        groups[key].append(row)

    summary: list[dict[str, str]] = []
    for key, items in sorted(groups.items(), key=lambda kv: (kv[0][0], int(kv[0][1] or 0), int(kv[0][2] or 0))):
        latencies = [f(r, "wall_latency_ms") for r in items]
        epts = [f(r, "system_total_energy_mj_per_generated_token") for r in items]
        record = dict(zip(KEY_FIELDS, key))
        record.update(
            {
                "n": str(len(items)),
                "generated_tokens_mean": fmt(mean(f(r, "generated_tokens") for r in items)),
                "accept_rate_mean": fmt(mean(f(r, "accept_rate") for r in items), 6),
                "wall_latency_ms_mean": fmt(mean(latencies)),
                "wall_latency_ms_std": fmt(pstdev(latencies) if len(latencies) > 1 else 0.0),
                "tokens_per_s_mean": fmt(mean(f(r, "tokens_per_s") for r in items), 6),
                "system_energy_mj_mean": fmt(mean(f(r, "system_total_energy_mj") for r in items)),
                "energy_per_token_mj_mean": fmt(mean(epts)),
                "energy_per_token_mj_std": fmt(pstdev(epts) if len(epts) > 1 else 0.0),
                "drafter_energy_mj_mean": fmt(mean(f(r, "drafter_total_energy_mj") for r in items)),
                "verifier_energy_mj_mean": fmt(mean(f(r, "verifier_total_energy_mj") for r in items)),
                "system_energy_complete_rate": fmt(mean(f(r, "system_energy_complete", 0.0) for r in items), 6),
            }
        )
        summary.append(record)
    return summary


def best_by_gamma(summary: list[dict[str, str]], latency_slack: float) -> list[dict[str, str]]:
    by_gamma: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary:
        by_gamma[row["gamma"]].append(row)

    result: list[dict[str, str]] = []
    for gamma, rows in sorted(by_gamma.items(), key=lambda kv: int(kv[0] or 0)):
        high_rows = [r for r in rows if r["verifier_freq_hz"] == "1300500000"]
        high_latency = f(high_rows[0], "wall_latency_ms_mean") if high_rows else min(
            f(r, "wall_latency_ms_mean") for r in rows
        )
        latency_budget = high_latency * (1.0 + latency_slack)
        fastest = min(rows, key=lambda r: f(r, "wall_latency_ms_mean"))
        energy_best = min(rows, key=lambda r: f(r, "energy_per_token_mj_mean"))
        feasible = [r for r in rows if f(r, "wall_latency_ms_mean") <= latency_budget]
        constrained = min(feasible, key=lambda r: f(r, "energy_per_token_mj_mean")) if feasible else energy_best
        high_energy = f(high_rows[0], "energy_per_token_mj_mean") if high_rows else math.nan
        saving = (high_energy - f(constrained, "energy_per_token_mj_mean")) / high_energy * 100 if high_energy else math.nan
        result.append(
            {
                "gamma": gamma,
                "latency_budget_ms": fmt(latency_budget),
                "fastest_freq_hz": fastest["verifier_freq_hz"],
                "fastest_latency_ms": fastest["wall_latency_ms_mean"],
                "energy_best_freq_hz": energy_best["verifier_freq_hz"],
                "energy_best_mj_per_token": energy_best["energy_per_token_mj_mean"],
                "latency_constrained_best_freq_hz": constrained["verifier_freq_hz"],
                "latency_constrained_best_mj_per_token": constrained["energy_per_token_mj_mean"],
                "energy_saving_vs_high_percent": fmt(saving, 3),
            }
        )
    return result


def fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]], best: list[dict[str, str]]) -> None:
    lines = [
        "# Phase 1 Summary",
        "",
        "Primary rail: `tot_power`.",
        "",
        "## Best Frequency By Gamma",
        "",
        "| gamma | fastest freq | energy best freq | latency-constrained best freq | saving vs high |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            "| {gamma} | {fastest_freq_hz} | {energy_best_freq_hz} | {latency_constrained_best_freq_hz} | {energy_saving_vs_high_percent}% |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Aggregated Conditions",
            "",
            "| gamma | verifier freq Hz | n | latency ms | tokens/s | energy/token mJ | accept rate |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda r: (int(r["gamma"] or 0), int(r["verifier_freq_hz"] or 0))):
        lines.append(
            "| {gamma} | {verifier_freq_hz} | {n} | {wall_latency_ms_mean} | {tokens_per_s_mean} | {energy_per_token_mj_mean} | {accept_rate_mean} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n")


def maybe_write_plots(out_dir: Path, rows: list[dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        write_svg_plots(out_dir, rows)
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    by_gamma: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_gamma[row["gamma"]].append(row)

    for metric, ylabel, filename in [
        ("energy_per_token_mj_mean", "Energy per token (mJ)", "energy_per_token_by_gamma.png"),
        ("wall_latency_ms_mean", "Latency (ms)", "latency_by_gamma.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for gamma, items in sorted(by_gamma.items(), key=lambda kv: int(kv[0] or 0)):
            items = sorted(items, key=lambda r: int(r["verifier_freq_hz"] or 0))
            xs = [int(r["verifier_freq_hz"]) / 1e6 for r in items]
            ys = [f(r, metric) for r in items]
            plt.plot(xs, ys, marker="o", label=f"gamma={gamma}")
        plt.xlabel("Verifier GPU frequency (MHz)")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=180)
        plt.close()

    plt.figure(figsize=(8, 5))
    for row in rows:
        plt.scatter(
            f(row, "wall_latency_ms_mean"),
            f(row, "energy_per_token_mj_mean"),
            label=f"g{row['gamma']} {int(row['verifier_freq_hz'])/1e6:.0f}MHz",
        )
    plt.xlabel("Latency (ms)")
    plt.ylabel("Energy per token (mJ)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "pareto_energy_latency.png", dpi=180)
    plt.close()


def write_svg_plots(out_dir: Path, rows: list[dict[str, str]]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    line_specs = [
        ("energy_per_token_mj_mean", "Energy per token (mJ)", "energy_per_token_by_gamma.svg"),
        ("wall_latency_ms_mean", "Latency (ms)", "latency_by_gamma.svg"),
    ]
    for metric, ylabel, filename in line_specs:
        series: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            series[row["gamma"]].append((float(row["verifier_freq_hz"]) / 1e6, f(row, metric)))
        write_line_svg(plot_dir / filename, series, "Verifier GPU frequency (MHz)", ylabel)

    points: list[tuple[float, float, str]] = []
    for row in rows:
        label = f"g{row['gamma']} {float(row['verifier_freq_hz']) / 1e6:.0f}MHz"
        points.append((f(row, "wall_latency_ms_mean"), f(row, "energy_per_token_mj_mean"), label))
    write_scatter_svg(plot_dir / "pareto_energy_latency.svg", points, "Latency (ms)", "Energy per token (mJ)")


def write_line_svg(
    path: Path,
    series: dict[str, list[tuple[float, float]]],
    xlabel: str,
    ylabel: str,
) -> None:
    width, height = 900, 560
    left, right, top, bottom = 90, 30, 40, 80
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    values = [point for points in series.values() for point in points]
    xs = [x for x, _ in values]
    ys = [y for _, y in values]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    ypad = (ymax - ymin) * 0.08 or 1.0
    ymin -= ypad
    ymax += ypad

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin or 1.0) * (width - left - right)

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin or 1.0) * (height - top - bottom)

    parts = svg_header(width, height)
    parts.append(axis_svg(width, height, left, right, top, bottom, xlabel, ylabel))
    for tick in [xmin, (xmin + xmax) / 2, xmax]:
        x = sx(tick)
        parts.append(f'<text x="{x:.1f}" y="{height - 45}" text-anchor="middle" font-size="12">{tick:.0f}</text>')
    for tick in [ymin, (ymin + ymax) / 2, ymax]:
        y = sy(tick)
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{tick:.0f}</text>')

    for idx, (name, points) in enumerate(sorted(series.items(), key=lambda kv: int(kv[0]))):
        points = sorted(points)
        color = colors[idx % len(colors)]
        poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{color}"/>')
        legend_y = top + idx * 22
        parts.append(f'<rect x="{width - 180}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{width - 160}" y="{legend_y + 2}" font-size="13">gamma={name}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def write_scatter_svg(
    path: Path,
    points: list[tuple[float, float, str]],
    xlabel: str,
    ylabel: str,
) -> None:
    width, height = 900, 560
    left, right, top, bottom = 90, 30, 40, 80
    xs = [x for x, _, _ in points]
    ys = [y for _, y, _ in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = (xmax - xmin) * 0.08 or 1.0
    ypad = (ymax - ymin) * 0.08 or 1.0
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin or 1.0) * (width - left - right)

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin or 1.0) * (height - top - bottom)

    parts = svg_header(width, height)
    parts.append(axis_svg(width, height, left, right, top, bottom, xlabel, ylabel))
    for x, y, label in points:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="#1f77b4"/>')
        parts.append(f'<title>{label}: latency={x:.1f}, energy/token={y:.1f}</title>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def axis_svg(
    width: int,
    height: int,
    left: int,
    right: int,
    top: int,
    bottom: int,
    xlabel: str,
    ylabel: str,
) -> str:
    x2 = width - right
    y2 = height - bottom
    return "\n".join(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{y2}" stroke="#222" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{y2}" x2="{x2}" y2="{y2}" stroke="#222" stroke-width="1.5"/>',
            f'<text x="{(left + x2) / 2:.1f}" y="{height - 15}" text-anchor="middle" font-size="15">{xlabel}</text>',
            f'<text x="20" y="{(top + y2) / 2:.1f}" text-anchor="middle" font-size="15" transform="rotate(-90 20 {(top + y2) / 2:.1f})">{ylabel}</text>',
        ]
    )


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    rows = read_rows(raw_dir)
    summary = summarize(rows)
    best = best_by_gamma(summary, args.latency_slack)
    write_csv(out_dir / "phase1_filtered_tot_power_rows.csv", rows, sorted(rows[0].keys()) if rows else [])
    write_csv(out_dir / "phase1_summary_by_gamma_freq.csv", summary, list(KEY_FIELDS) + SUMMARY_FIELDS)
    write_csv(
        out_dir / "phase1_best_frequency_by_gamma.csv",
        best,
        [
            "gamma",
            "latency_budget_ms",
            "fastest_freq_hz",
            "fastest_latency_ms",
            "energy_best_freq_hz",
            "energy_best_mj_per_token",
            "latency_constrained_best_freq_hz",
            "latency_constrained_best_mj_per_token",
            "energy_saving_vs_high_percent",
        ],
    )
    write_markdown(out_dir / "phase1_summary.md", summary, best)
    maybe_write_plots(out_dir, summary)
    print(f"rows={len(rows)} groups={len(summary)} out={out_dir}")


if __name__ == "__main__":
    main()
