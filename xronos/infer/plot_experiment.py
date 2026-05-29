import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


WIDTH = 960
HEIGHT = 560
MARGIN_LEFT = 88
MARGIN_RIGHT = 36
MARGIN_TOP = 56
MARGIN_BOTTOM = 88
COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
]


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _float(row: Dict[str, str], key: str) -> Optional[float]:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _fmt(value: float) -> str:
    return f"{value:.3g}"


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _plot_area() -> Tuple[int, int, int, int]:
    return (
        MARGIN_LEFT,
        MARGIN_TOP,
        WIDTH - MARGIN_RIGHT,
        HEIGHT - MARGIN_BOTTOM,
    )


def _scale(values: Sequence[float], start: float, end: float, pad_fraction: float = 0.08):
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        delta = abs(vmin) * 0.1 if vmin else 1.0
        vmin -= delta
        vmax += delta
    else:
        pad = (vmax - vmin) * pad_fraction
        vmin -= pad
        vmax += pad

    def scale(value: float) -> float:
        return start + (value - vmin) * (end - start) / (vmax - vmin)

    return scale, vmin, vmax


def _base_svg(title: str, x_label: str, y_label: str) -> List[str]:
    x0, y0, x1, y1 = _plot_area()
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{WIDTH / 2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{_escape(title)}</text>',
        f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#111827" stroke-width="1.2"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#111827" stroke-width="1.2"/>',
        f'<text x="{(x0 + x1) / 2:.1f}" y="{HEIGHT - 22}" text-anchor="middle" font-family="Arial" font-size="14">{_escape(x_label)}</text>',
        f'<text x="22" y="{(y0 + y1) / 2:.1f}" transform="rotate(-90 22 {(y0 + y1) / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="14">{_escape(y_label)}</text>',
    ]


def _finish_svg(parts: List[str]) -> str:
    return "\n".join(parts + ["</svg>", ""])


def _group_label(row: Dict[str, str]) -> str:
    pieces = []
    for key, label in (
        ("drafter_freq_hz", "fd"),
        ("verifier_clock_mhz", "fv"),
        ("max_new_tokens", "tok"),
        ("stop_token_policy", "stop"),
    ):
        value = row.get(key, "")
        if value not in ("", None):
            pieces.append(f"{label}={value}")
    return ", ".join(pieces) if pieces else "default"


def _line_plot(
    rows: List[Dict[str, str]],
    title: str,
    y_key: str,
    y_label: str,
    out_path: Path,
    error_key: str = "",
) -> Dict[str, object]:
    valid = [
        row
        for row in rows
        if _float(row, "gamma") is not None and _float(row, y_key) is not None
    ]
    if not valid:
        return {"name": out_path.stem, "path": str(out_path), "status": "skipped", "rows": 0}

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in valid:
        grouped.setdefault(_group_label(row), []).append(row)

    xs = [math.log2(_float(row, "gamma") or 1.0) for row in valid]
    ys = [_float(row, y_key) or 0.0 for row in valid]
    errors = [_float(row, error_key) or 0.0 for row in valid] if error_key else []
    y_values = ys + [y + e for y, e in zip(ys, errors)] + [y - e for y, e in zip(ys, errors)]
    x0, y0, x1, y1 = _plot_area()
    xscale, xmin, xmax = _scale(xs, x0, x1, pad_fraction=0.04)
    yscale_raw, ymin, ymax = _scale(y_values, y1, y0)
    parts = _base_svg(title, "gamma", y_label)

    for tick in sorted({_float(row, "gamma") for row in valid if _float(row, "gamma") is not None}):
        x = xscale(math.log2(tick or 1.0))
        parts.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y1 + 5}" stroke="#111827"/>')
        parts.append(f'<text x="{x:.1f}" y="{y1 + 22}" text-anchor="middle" font-family="Arial" font-size="12">{_escape(_fmt(tick or 0.0))}</text>')
    for i in range(5):
        value = ymin + (ymax - ymin) * i / 4
        y = yscale_raw(value)
        parts.append(f'<line x1="{x0 - 5}" y1="{y:.1f}" x2="{x0}" y2="{y:.1f}" stroke="#111827"/>')
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{_escape(_fmt(value))}</text>')

    legend_y = MARGIN_TOP
    for index, (label, items) in enumerate(sorted(grouped.items())):
        color = COLORS[index % len(COLORS)]
        items = sorted(items, key=lambda row: _float(row, "gamma") or 0.0)
        points = []
        for row in items:
            gamma = _float(row, "gamma") or 1.0
            y_value = _float(row, y_key) or 0.0
            x = xscale(math.log2(gamma))
            y = yscale_raw(y_value)
            points.append(f"{x:.1f},{y:.1f}")
            if error_key:
                err = _float(row, error_key) or 0.0
                y_hi = yscale_raw(y_value + err)
                y_lo = yscale_raw(y_value - err)
                parts.append(f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="{color}" stroke-width="1"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        if len(points) >= 2:
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<rect x="{x1 - 185}" y="{legend_y + index * 20}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x1 - 168}" y="{legend_y + index * 20 + 11}" font-family="Arial" font-size="12">{_escape(label)}</text>')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_finish_svg(parts), encoding="utf-8")
    return {"name": out_path.stem, "path": str(out_path), "status": "ok", "rows": len(valid)}


def _bar_plot(
    rows: List[Dict[str, str]],
    title: str,
    value_key: str,
    y_label: str,
    out_path: Path,
) -> Dict[str, object]:
    valid = [row for row in rows if _float(row, value_key) is not None]
    if not valid:
        return {"name": out_path.stem, "path": str(out_path), "status": "skipped", "rows": 0}
    valid = valid[:20]
    values = [_float(row, value_key) or 0.0 for row in valid]
    x0, y0, x1, y1 = _plot_area()
    yscale, ymin, ymax = _scale(values + [0.0], y1, y0)
    parts = _base_svg(title, "configuration", y_label)
    bar_gap = 8
    bar_width = max(8, (x1 - x0 - bar_gap * (len(valid) - 1)) / max(1, len(valid)))

    for i in range(5):
        value = ymin + (ymax - ymin) * i / 4
        y = yscale(value)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{_escape(_fmt(value))}</text>')

    zero_y = yscale(0.0)
    for index, row in enumerate(valid):
        value = _float(row, value_key) or 0.0
        x = x0 + index * (bar_width + bar_gap)
        y = yscale(value)
        color = "#16a34a" if value >= 0 else "#dc2626"
        top = min(y, zero_y)
        height = abs(zero_y - y)
        parts.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_width:.1f}" height="{height:.1f}" fill="{color}"/>')
        label = f"g={row.get('gamma', '')}"
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{y1 + 26}" text-anchor="middle" font-family="Arial" font-size="11" transform="rotate(45 {x + bar_width / 2:.1f} {y1 + 26})">{_escape(label)}</text>')
    parts.append(f'<line x1="{x0}" y1="{zero_y:.1f}" x2="{x1}" y2="{zero_y:.1f}" stroke="#111827" stroke-width="1"/>')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_finish_svg(parts), encoding="utf-8")
    return {"name": out_path.stem, "path": str(out_path), "status": "ok", "rows": len(valid)}


def _scatter_plot(
    rows: List[Dict[str, str]],
    title: str,
    x_key: str,
    y_key: str,
    out_path: Path,
) -> Dict[str, object]:
    valid = [
        row
        for row in rows
        if _float(row, x_key) is not None and _float(row, y_key) is not None
    ]
    if not valid:
        return {"name": out_path.stem, "path": str(out_path), "status": "skipped", "rows": 0}
    xs = [_float(row, x_key) or 0.0 for row in valid]
    ys = [_float(row, y_key) or 0.0 for row in valid]
    x0, y0, x1, y1 = _plot_area()
    xscale, xmin, xmax = _scale(xs, x0, x1)
    yscale, ymin, ymax = _scale(ys, y1, y0)
    parts = _base_svg(title, "tokens/s", y_key)
    for i in range(5):
        x_value = xmin + (xmax - xmin) * i / 4
        x = xscale(x_value)
        parts.append(f'<text x="{x:.1f}" y="{y1 + 22}" text-anchor="middle" font-family="Arial" font-size="12">{_escape(_fmt(x_value))}</text>')
        y_value = ymin + (ymax - ymin) * i / 4
        y = yscale(y_value)
        parts.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{_escape(_fmt(y_value))}</text>')
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e5e7eb"/>')
    for index, row in enumerate(valid):
        x = xscale(_float(row, x_key) or 0.0)
        y = yscale(_float(row, y_key) or 0.0)
        color = COLORS[index % len(COLORS)]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{x + 7:.1f}" y="{y - 7:.1f}" font-family="Arial" font-size="11">g={_escape(row.get("gamma", ""))}</text>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_finish_svg(parts), encoding="utf-8")
    return {"name": out_path.stem, "path": str(out_path), "status": "ok", "rows": len(valid)}


def write_plot_bundle(
    summary_rows: List[Dict[str, str]],
    gamma_effect_rows: List[Dict[str, str]],
    paired_summary_rows: List[Dict[str, str]],
    out_dir: Path,
    summary_energy_key: str = "mean_drafter_active_energy_mj_per_token",
) -> Dict[str, object]:
    plots = [
        _line_plot(
            gamma_effect_rows,
            title="Drafter Energy vs Gamma",
            y_key="mean_drafter_total_energy_mj_per_generated_token",
            y_label="drafter mJ / generated token",
            error_key="ci95_drafter_total_energy_mj_per_generated_token",
            out_path=out_dir / "gamma_drafter_energy.svg",
        ),
        _line_plot(
            gamma_effect_rows,
            title="Accept Rate vs Gamma",
            y_key="mean_accept_rate",
            y_label="accept rate",
            out_path=out_dir / "gamma_accept_rate.svg",
        ),
        _bar_plot(
            paired_summary_rows,
            title="Prompt-Paired Energy Savings vs Baseline",
            value_key="mean_energy_savings_pct_vs_baseline",
            y_label="energy savings (%)",
            out_path=out_dir / "paired_energy_savings.svg",
        ),
        _scatter_plot(
            summary_rows,
            title="Energy and Throughput Tradeoff",
            x_key="mean_tokens_per_s",
            y_key=summary_energy_key,
            out_path=out_dir / "energy_throughput.svg",
        ),
    ]
    manifest = {
        "plots": plots,
        "ok_plots": sum(1 for plot in plots if plot["status"] == "ok"),
        "skipped_plots": sum(1 for plot in plots if plot["status"] == "skipped"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write SVG plots from Xronos report CSVs.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--gamma-effect", required=True)
    parser.add_argument("--paired-summary", required=True)
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument(
        "--summary-energy-key",
        default="mean_drafter_active_energy_mj_per_token",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_plot_bundle(
        summary_rows=read_csv(args.summary),
        gamma_effect_rows=read_csv(args.gamma_effect),
        paired_summary_rows=read_csv(args.paired_summary),
        out_dir=Path(args.out_dir),
        summary_energy_key=args.summary_energy_key,
    )
    print(f"ok_plots={manifest['ok_plots']}")
    print(f"skipped_plots={manifest['skipped_plots']}")
    print(f"Wrote plot bundle to {args.out_dir}")


if __name__ == "__main__":
    main()
