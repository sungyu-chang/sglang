#!/usr/bin/env python3
"""Draw two-expert even-split DeepGEMM MoE comparison figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_rows(run_dir: Path) -> list[dict[str, str]]:
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_csv}")
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("active_experts") != "2" or row.get("distribution") != "even":
            raise ValueError(
                f"Expected two-expert even-split rows in {metrics_csv}, got "
                f"active_experts={row.get('active_experts')} distribution={row.get('distribution')}."
            )
        row["run_dir"] = str(run_dir)
    return rows


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model_preset"]].append(row)
    for series in grouped.values():
        series.sort(key=lambda row: int(row["batch_size"]))
    return dict(grouped)


def pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import academia_style
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for DeepGEMM MoE comparison figures. "
            "Install matplotlib in the active environment and retry."
        ) from exc
    return plt, academia_style


def save_metric_plot(
    output_dir: Path,
    grouped: dict[str, list[dict[str, str]]],
    *,
    metric: str,
    y_label: str,
    stem: str,
) -> None:
    plt, academia_style = pyplot()
    fig, ax = plt.subplots(figsize=(9.4, 5.2), dpi=120)
    for series_index, (label, series) in enumerate(grouped.items()):
        x_values = [int(row["batch_size"]) for row in series]
        y_values = [float(row[metric]) for row in series]
        ax.plot(
            x_values,
            y_values,
            marker=academia_style.MARKERS[series_index % len(academia_style.MARKERS)],
            linestyle=academia_style.LINESTYLES[series_index % len(academia_style.LINESTYLES)],
            color=academia_style.PAIRED[series_index % len(academia_style.PAIRED)],
            label=label,
        )
    ax.set_xscale("log", base=2)
    all_x = sorted({int(row["batch_size"]) for series in grouped.values() for row in series})
    ax.set_xticks(all_x)
    ax.set_xticklabels([str(x) for x in all_x])
    ax.set_xlabel("Input batch size")
    ax.set_ylabel(y_label)
    ax.set_title("DeepGEMM two experts even split")
    academia_style.style_fig(fig, legend_ncol=1, enforce=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"{stem}.{suffix}")
    plt.close(fig)


def write_combined_csv(output_dir: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with (output_dir / "combined_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(output_dir: Path, run_dirs: list[Path], rows: list[dict[str, str]]) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "two_experts_even_split",
        "run_dirs": [str(path) for path in run_dirs],
        "num_rows": len(rows),
        "series": sorted({row["model_preset"] for row in rows}),
    }
    (output_dir / "comparison_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# DeepGEMM Two-Expert Even-Split Comparison", "", "## Runs", ""]
    for run_dir in run_dirs:
        lines.append(f"- `{run_dir}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `combined_metrics.csv`",
            "- `comparison_config.json`",
            "- `latency_us_vs_batch.png` / `latency_us_vs_batch.pdf`",
            "- `throughput_vs_batch.png` / `throughput_vs_batch.pdf`",
            "- `tflops_vs_batch.png` / `tflops_vs_batch.pdf`",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Two-expert even-split run dirs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for comparison figures. Defaults to results/deepgemm_moe/compare_two_experts_<timestamp>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = [path.resolve() for path in args.run_dirs]
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("results") / "deepgemm_moe" / f"compare_two_experts_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, str]] = []
    for run_dir in run_dirs:
        rows.extend(load_rows(run_dir))
    if not rows:
        raise RuntimeError("No metrics rows found.")

    grouped = group_rows(rows)
    write_combined_csv(output_dir, rows)
    write_summary(output_dir, run_dirs, rows)
    save_metric_plot(
        output_dir,
        grouped,
        metric="latency_us_mean",
        y_label="Mean latency (us)",
        stem="latency_us_vs_batch",
    )
    save_metric_plot(
        output_dir,
        grouped,
        metric="throughput_tokens_per_s",
        y_label="Throughput (tokens/s)",
        stem="throughput_vs_batch",
    )
    save_metric_plot(
        output_dir,
        grouped,
        metric="effective_tflops",
        y_label="Effective TFLOP/s",
        stem="tflops_vs_batch",
    )
    print(f"Two-expert comparison figures written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())