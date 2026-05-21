#!/usr/bin/env python3
"""Draw DeepGEMM MoE benchmark figures from a run directory."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_rows(run_dir: Path) -> list[dict[str, str]]:
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_csv}")
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        backend = row.get("backend", "deep_gemm_wrapper")
        graph_suffix = " / cuda graph" if row.get("use_cuda_graph") == "True" else ""
        label = f"{row['model_preset']} / {backend} / {row['op']}{graph_suffix}"
        grouped[label].append(row)
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
            "matplotlib is required for DeepGEMM MoE benchmark figures. "
            "Install matplotlib in the active environment and retry."
        ) from exc
    return plt, academia_style


def save_metric_plot(
    run_dir: Path,
    grouped: dict[str, list[dict[str, str]]],
    *,
    metric: str,
    y_label: str,
    stem: str,
) -> None:
    plt, academia_style = pyplot()
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=120)
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
    ax.set_title("DeepGEMM MoE Expert Compute")
    academia_style.style_fig(fig, legend_ncol=max(1, len(grouped)), enforce=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    for suffix in ["png", "pdf"]:
        fig.savefig(run_dir / f"{stem}.{suffix}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory under results/deepgemm_moe/.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    rows = load_rows(run_dir)
    if not rows:
        raise RuntimeError(f"No rows found in {run_dir / 'metrics.csv'}")
    grouped = group_rows(rows)
    save_metric_plot(
        run_dir,
        grouped,
        metric="latency_us_mean",
        y_label="Mean latency (us)",
        stem="latency_us_vs_batch",
    )
    save_metric_plot(
        run_dir,
        grouped,
        metric="throughput_tokens_per_s",
        y_label="Throughput (tokens/s)",
        stem="throughput_vs_batch",
    )
    save_metric_plot(
        run_dir,
        grouped,
        metric="effective_tflops",
        y_label="Effective TFLOP/s",
        stem="tflops_vs_batch",
    )
    print(f"Figures written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
