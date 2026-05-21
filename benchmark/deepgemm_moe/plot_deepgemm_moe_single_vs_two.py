#!/usr/bin/env python3
"""Compare single-expert and two-expert even-split DeepGEMM MoE runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


METRICS = [
    ("latency_us_mean", "Mean latency (us)", "latency_us_vs_batch"),
    ("throughput_tokens_per_s", "Throughput (tokens/s)", "throughput_vs_batch"),
    ("effective_tflops", "Effective TFLOP/s", "tflops_vs_batch"),
]


def load_rows(run_dir: Path, expected_experts: str, expected_distribution: str) -> list[dict[str, str]]:
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_csv}")
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("active_experts") != expected_experts or row.get("distribution") != expected_distribution:
            raise ValueError(
                f"Unexpected row in {metrics_csv}: active_experts={row.get('active_experts')} "
                f"distribution={row.get('distribution')}; expected active_experts={expected_experts} "
                f"distribution={expected_distribution}."
            )
        row["run_dir"] = str(run_dir)
    return rows


def rows_by_model(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
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
    single_by_model: dict[str, list[dict[str, str]]],
    two_by_model: dict[str, list[dict[str, str]]],
    *,
    metric: str,
    y_label: str,
    stem: str,
) -> None:
    plt, academia_style = pyplot()
    fig, ax = plt.subplots(figsize=(9.8, 5.4), dpi=120)
    models = sorted(set(single_by_model) | set(two_by_model))
    for model_index, model in enumerate(models):
        color = academia_style.PAIRED[model_index % len(academia_style.PAIRED)]
        marker = academia_style.MARKERS[model_index % len(academia_style.MARKERS)]
        if model in single_by_model:
            series = single_by_model[model]
            ax.plot(
                [int(row["batch_size"]) for row in series],
                [float(row[metric]) for row in series],
                marker=marker,
                linestyle=":",
                color=color,
                label=f"{model} single expert",
            )
        if model in two_by_model:
            series = two_by_model[model]
            ax.plot(
                [int(row["batch_size"]) for row in series],
                [float(row[metric]) for row in series],
                marker=marker,
                linestyle="-",
                color=color,
                label=f"{model} two experts",
            )
    ax.set_xscale("log", base=2)
    all_x = sorted(
        {
            int(row["batch_size"])
            for grouped in [single_by_model, two_by_model]
            for series in grouped.values()
            for row in series
        }
    )
    ax.set_xticks(all_x)
    ax.set_xticklabels([str(x) for x in all_x])
    ax.set_xlabel("Input batch size")
    ax.set_ylabel(y_label)
    ax.set_title("DeepGEMM single vs two experts")
    academia_style.style_fig(fig, legend_ncol=2, enforce=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
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


def write_difference_csv(
    output_dir: Path,
    single_rows: list[dict[str, str]],
    two_rows: list[dict[str, str]],
) -> None:
    single_index = {
        (row["model_preset"], row["batch_size"]): row
        for row in single_rows
    }
    two_index = {
        (row["model_preset"], row["batch_size"]): row
        for row in two_rows
    }
    keys = sorted(set(single_index) & set(two_index), key=lambda item: (item[0], int(item[1])))
    rows: list[dict[str, object]] = []
    for model, batch_size in keys:
        single = single_index[(model, batch_size)]
        two = two_index[(model, batch_size)]
        row: dict[str, object] = {
            "model_preset": model,
            "batch_size": int(batch_size),
        }
        for metric, _, _ in METRICS:
            single_value = float(single[metric])
            two_value = float(two[metric])
            row[f"single_{metric}"] = single_value
            row[f"two_expert_{metric}"] = two_value
            row[f"delta_{metric}"] = two_value - single_value
            row[f"ratio_two_over_single_{metric}"] = two_value / single_value if single_value else 0.0
        rows.append(row)
    if not rows:
        return
    with (output_dir / "single_vs_two_differences.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    output_dir: Path,
    single_dirs: list[Path],
    two_dirs: list[Path],
    rows: list[dict[str, str]],
) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "single_vs_two_experts",
        "single_expert_run_dirs": [str(path) for path in single_dirs],
        "two_expert_run_dirs": [str(path) for path in two_dirs],
        "num_rows": len(rows),
        "series": sorted({row["model_preset"] for row in rows}),
    }
    (output_dir / "comparison_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# DeepGEMM Single vs Two Experts",
        "",
        "Dotted lines are single-expert runs. Solid lines are two-expert even-split runs.",
        "",
        "## Artifacts",
        "",
        "- `combined_metrics.csv`",
        "- `single_vs_two_differences.csv`",
        "- `comparison_config.json`",
        "- `latency_us_vs_batch.png` / `latency_us_vs_batch.pdf`",
        "- `throughput_vs_batch.png` / `throughput_vs_batch.pdf`",
        "- `tflops_vs_batch.png` / `tflops_vs_batch.pdf`",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--two-run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for comparison figures. Defaults to results/deepgemm_moe/compare_single_vs_two_<timestamp>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    single_dirs = [path.resolve() for path in args.single_run_dirs]
    two_dirs = [path.resolve() for path in args.two_run_dirs]
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("results") / "deepgemm_moe" / f"compare_single_vs_two_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    single_rows: list[dict[str, str]] = []
    for run_dir in single_dirs:
        single_rows.extend(load_rows(run_dir, expected_experts="1", expected_distribution="single"))
    two_rows: list[dict[str, str]] = []
    for run_dir in two_dirs:
        two_rows.extend(load_rows(run_dir, expected_experts="2", expected_distribution="even"))
    if not single_rows or not two_rows:
        raise RuntimeError("Both single-expert and two-expert rows are required.")

    all_rows = single_rows + two_rows
    single_by_model = rows_by_model(single_rows)
    two_by_model = rows_by_model(two_rows)
    write_combined_csv(output_dir, all_rows)
    write_difference_csv(output_dir, single_rows, two_rows)
    write_summary(output_dir, single_dirs, two_dirs, all_rows)
    for metric, y_label, stem in METRICS:
        save_metric_plot(
            output_dir,
            single_by_model,
            two_by_model,
            metric=metric,
            y_label=y_label,
            stem=stem,
        )
    print(f"Single-vs-two comparison figures written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())