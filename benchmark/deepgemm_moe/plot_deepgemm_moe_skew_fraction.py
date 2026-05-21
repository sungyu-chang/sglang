#!/usr/bin/env python3
"""Compare DeepGEMM MoE skewed token distributions with different skew fractions."""

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

COLORS_BY_EXPERTS = {
    2: "#1f77b4",
    4: "#2ca02c",
    8: "#d62728",
}

MARKERS_BY_EXPERTS = {
    2: "o",
    4: "s",
    8: "^",
}

LINESTYLES_BY_FRACTION = {
    0.0625: "-",
    0.5: "--",
}


def parse_group(value: str) -> tuple[int, float, list[Path]]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Groups must use EXPERT_COUNT:SKEW_MINOR_FRACTION:path1,path2,... format."
        )
    expert_count_text, fraction_text, paths_text = parts
    try:
        expert_count = int(expert_count_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid expert count: {expert_count_text!r}") from exc
    try:
        fraction = float(fraction_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid skew minor fraction: {fraction_text!r}") from exc
    if expert_count <= 1:
        raise argparse.ArgumentTypeError("Skew comparison expects at least two active experts.")
    if not 0.0 < fraction < 1.0:
        raise argparse.ArgumentTypeError("Skew minor fraction must be in (0, 1).")
    paths = [Path(item) for item in paths_text.split(",") if item]
    if not paths:
        raise argparse.ArgumentTypeError("Each group needs at least one run directory.")
    return expert_count, fraction, paths


def load_rows(run_dir: Path, active_experts: int, skew_minor_fraction: float) -> list[dict[str, str]]:
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_csv}")
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row_fraction = float(row.get("skew_minor_fraction", "nan"))
        if (
            row.get("active_experts") != str(active_experts)
            or row.get("distribution") != "skewed"
            or abs(row_fraction - skew_minor_fraction) > 1e-9
        ):
            raise ValueError(
                f"Unexpected row in {metrics_csv}: active_experts={row.get('active_experts')} "
                f"distribution={row.get('distribution')} skew_minor_fraction={row.get('skew_minor_fraction')}; "
                f"expected active_experts={active_experts} distribution=skewed "
                f"skew_minor_fraction={skew_minor_fraction}."
            )
        row["run_dir"] = str(run_dir)
    return rows


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, int, float], list[dict[str, str]]]:
    grouped: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["model_preset"], int(row["active_experts"]), float(row["skew_minor_fraction"]))
        ].append(row)
    for series in grouped.values():
        series.sort(key=lambda row: int(row["batch_size"]))
    return dict(grouped)


def pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for DeepGEMM MoE comparison figures.") from exc
    return plt


def fraction_label(fraction: float) -> str:
    return f"{fraction * 100:g}% minor"


def series_label(active_experts: int, fraction: float) -> str:
    return f"{active_experts} experts / {fraction_label(fraction)}"


def save_metric_plot(
    output_dir: Path,
    grouped: dict[tuple[str, int, float], list[dict[str, str]]],
    *,
    metric: str,
    y_label: str,
    stem: str,
) -> None:
    plt = pyplot()
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "lines.markersize": 5,
            "lines.linewidth": 1.6,
        }
    )
    models = sorted({model for model, _, _ in grouped})
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(4.8 * len(models), 3.8),
        dpi=120,
        sharey=True,
        squeeze=False,
    )
    for model_index, model in enumerate(models):
        ax = axes[0][model_index]
        keys = sorted(
            (active_experts, fraction)
            for item_model, active_experts, fraction in grouped
            if item_model == model
        )
        for active_experts, fraction in keys:
            series = grouped[(model, active_experts, fraction)]
            ax.plot(
                [int(row["batch_size"]) for row in series],
                [float(row[metric]) for row in series],
                color=COLORS_BY_EXPERTS.get(active_experts, "#333333"),
                marker=MARKERS_BY_EXPERTS.get(active_experts, "o"),
                linestyle=LINESTYLES_BY_FRACTION.get(fraction, "-"),
                label=series_label(active_experts, fraction),
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Input batch size")
        if model_index == 0:
            ax.set_ylabel(y_label)
        ax.set_title(model)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
        ax.grid(False, axis="x")
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    all_x = sorted({int(row["batch_size"]) for series in grouped.values() for row in series})
    for ax in axes[0]:
        ax.set_xticks(all_x)
        ax.set_xticklabels([str(x) for x in all_x], rotation=45, ha="right")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.94), frameon=False)
    fig.suptitle("DeepGEMM skewness sensitivity", y=0.995, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82), w_pad=2.0)
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"{stem}.{suffix}")
    plt.close(fig)


def write_combined_csv(output_dir: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with (output_dir / "combined_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(output_dir: Path, groups: list[tuple[int, float, list[Path]]], rows: list[dict[str, str]]) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "skew_fraction_comparison",
        "groups": [
            {
                "active_experts": active_experts,
                "skew_minor_fraction": fraction,
                "run_dirs": [str(path.resolve()) for path in paths],
            }
            for active_experts, fraction, paths in groups
        ],
        "num_rows": len(rows),
        "series": sorted({row["model_preset"] for row in rows}),
    }
    (output_dir / "comparison_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# DeepGEMM Skewness Sensitivity",
        "",
        "Compares skewed token distributions with different `--skew-minor-fraction` values.",
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
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        type=parse_group,
        help="Group as EXPERT_COUNT:SKEW_MINOR_FRACTION:path1,path2,... .",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for comparison figures. Defaults to results/deepgemm_moe/compare_skew_fraction_<timestamp>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("results") / "deepgemm_moe" / f"compare_skew_fraction_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, str]] = []
    for active_experts, fraction, paths in args.group:
        for path in paths:
            rows.extend(load_rows(path.resolve(), active_experts, fraction))
    if not rows:
        raise RuntimeError("No metrics rows found.")
    grouped = group_rows(rows)
    write_combined_csv(output_dir, rows)
    write_summary(output_dir, args.group, rows)
    for metric, y_label, stem in METRICS:
        save_metric_plot(output_dir, grouped, metric=metric, y_label=y_label, stem=stem)
    print(f"Skew fraction comparison figures written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())