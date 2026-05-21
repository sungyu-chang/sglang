#!/usr/bin/env python3
"""Compare DeepGEMM MoE runs across active expert counts."""

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

LINESTYLES_BY_EXPERTS = {
    1: ":",
    2: "--",
    4: "-",
}


def parse_group(value: str) -> tuple[int, list[Path]]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "Groups must use EXPERT_COUNT:path1,path2,... format, for example 4:run_a,run_b,run_c."
        )
    expert_count_text, paths_text = value.split(":", 1)
    try:
        expert_count = int(expert_count_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid expert count: {expert_count_text!r}") from exc
    if expert_count <= 0:
        raise argparse.ArgumentTypeError("Expert count must be positive.")
    paths = [Path(item) for item in paths_text.split(",") if item]
    if not paths:
        raise argparse.ArgumentTypeError("Each group needs at least one run directory.")
    return expert_count, paths


def expected_distribution(active_experts: int) -> str:
    return "single" if active_experts == 1 else "even"


def load_rows(run_dir: Path, active_experts: int) -> list[dict[str, str]]:
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_csv}")
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    distribution = expected_distribution(active_experts)
    for row in rows:
        if row.get("active_experts") != str(active_experts) or row.get("distribution") != distribution:
            raise ValueError(
                f"Unexpected row in {metrics_csv}: active_experts={row.get('active_experts')} "
                f"distribution={row.get('distribution')}; expected active_experts={active_experts} "
                f"distribution={distribution}."
            )
        row["run_dir"] = str(run_dir)
    return rows


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, int], list[dict[str, str]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_preset"], int(row["active_experts"]))].append(row)
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


def label_for(active_experts: int) -> str:
    suffix = "expert" if active_experts == 1 else "experts"
    return f"{active_experts} {suffix}"


def save_metric_plot(
    output_dir: Path,
    grouped: dict[tuple[str, int], list[dict[str, str]]],
    *,
    metric: str,
    y_label: str,
    stem: str,
) -> None:
    plt, academia_style = pyplot()
    fig, ax = plt.subplots(figsize=(10.4, 5.8), dpi=120)
    models = sorted({model for model, _ in grouped})
    model_colors = {
        model: academia_style.PAIRED[index % len(academia_style.PAIRED)]
        for index, model in enumerate(models)
    }
    model_markers = {
        model: academia_style.MARKERS[index % len(academia_style.MARKERS)]
        for index, model in enumerate(models)
    }
    for model in models:
        expert_counts = sorted(active_experts for item_model, active_experts in grouped if item_model == model)
        for active_experts in expert_counts:
            series = grouped[(model, active_experts)]
            ax.plot(
                [int(row["batch_size"]) for row in series],
                [float(row[metric]) for row in series],
                marker=model_markers[model],
                linestyle=LINESTYLES_BY_EXPERTS.get(active_experts, "-"),
                color=model_colors[model],
                label=f"{model} / {label_for(active_experts)}",
            )
    ax.set_xscale("log", base=2)
    all_x = sorted({int(row["batch_size"]) for series in grouped.values() for row in series})
    ax.set_xticks(all_x)
    ax.set_xticklabels([str(x) for x in all_x])
    ax.set_xlabel("Input batch size")
    ax.set_ylabel(y_label)
    ax.set_title("DeepGEMM expert scaling")
    academia_style.style_fig(fig, legend_ncol=3, enforce=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.76))
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


def write_summary(output_dir: Path, groups: list[tuple[int, list[Path]]], rows: list[dict[str, str]]) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "expert_scaling",
        "line_styles": {"1": "dotted", "2": "dashed", "4": "solid"},
        "groups": [
            {"active_experts": active_experts, "run_dirs": [str(path.resolve()) for path in paths]}
            for active_experts, paths in groups
        ],
        "num_rows": len(rows),
        "series": sorted({row["model_preset"] for row in rows}),
    }
    (output_dir / "comparison_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# DeepGEMM Expert Scaling Comparison",
        "",
        "Line styles: 1 expert dotted, 2 experts dashed, 4 experts solid.",
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
        help="Expert-count group as EXPERT_COUNT:path1,path2,... . Repeat for 1, 2, 4, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for comparison figures. Defaults to results/deepgemm_moe/compare_expert_scaling_<timestamp>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("results") / "deepgemm_moe" / f"compare_expert_scaling_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, str]] = []
    for active_experts, paths in args.group:
        for path in paths:
            rows.extend(load_rows(path.resolve(), active_experts))
    if not rows:
        raise RuntimeError("No metrics rows found.")

    grouped = group_rows(rows)
    write_combined_csv(output_dir, rows)
    write_summary(output_dir, args.group, rows)
    for metric, y_label, stem in METRICS:
        save_metric_plot(output_dir, grouped, metric=metric, y_label=y_label, stem=stem)
    print(f"Expert-scaling comparison figures written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())