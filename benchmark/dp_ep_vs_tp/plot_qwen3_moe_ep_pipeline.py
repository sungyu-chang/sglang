#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draw figures and derived CSVs for completed Qwen3-MoE DP+EP runs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    from matplotlib_plots import save_line_plot, save_stacked_bar_plot
except ModuleNotFoundError:
    from benchmark.dp_ep_vs_tp.matplotlib_plots import (
        save_line_plot,
        save_stacked_bar_plot,
    )


CASE_RE = re.compile(r"^dp(?P<gpu_count>\d+)_ep$")
ATTENTION_MODULE = "vllm:attention"
FUSED_MOE_MODULE = "vllm:fused_moe"
MOE_COMM_MODULE = "vllm:moe_comm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render Qwen3-MoE DP+EP figures from an existing result directory."
        )
    )
    parser.add_argument(
        "result_root",
        type=Path,
        help=(
            "Completed run directory, for example "
            "results/dp_ep_vs_tp/qwen3_moe_ep_pipeline/YYYY-MM-DD_HH-MM-SS."
        ),
    )
    parser.add_argument(
        "--throughput-metric",
        default="total_token_throughput",
        help="Metric column from throughput/summary.csv to plot.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def case_gpu_count(case_name: str) -> int:
    match = CASE_RE.match(case_name)
    if match is None:
        raise ValueError(f"Unexpected DP+EP case name: {case_name}")
    return int(match.group("gpu_count"))


def plot_throughput(summary_csv: Path, output_path: Path, metric: str) -> None:
    rows = sorted(read_csv(summary_csv), key=lambda row: int(row["gpu_count"]))
    x_values = [int(row["gpu_count"]) for row in rows]
    y_values = [float(row[metric]) for row in rows]
    save_line_plot(
        output_path,
        x_values,
        y_values,
        title="Qwen3-MoE DP+EP Online Throughput",
        x_label="Number of GPUs (DP+EP size)",
        y_label=metric.replace("_", " "),
    )


def load_case_gpu_counts(summary_csv: Path) -> dict[str, int]:
    return {
        row["case"]: int(row["gpu_count"] or case_gpu_count(row["case"]))
        for row in read_csv(summary_csv)
    }


def plot_module_latency(
    module_summary_csv: Path,
    profile_summary_csv: Path,
    output_path: Path,
) -> None:
    case_gpus = load_case_gpu_counts(profile_summary_csv)
    values: dict[int, dict[str, float]] = {}
    for row in read_csv(module_summary_csv):
        if row["rank"] != "all":
            continue
        module = row["module"]
        if module not in (ATTENTION_MODULE, FUSED_MOE_MODULE, MOE_COMM_MODULE):
            continue
        gpu_count = case_gpus.get(row["case"], case_gpu_count(row["case"]))
        values.setdefault(gpu_count, {})[module] = float(row["avg_cuda_ms"])

    x_values = sorted(values)
    attention_values = [values[gpu].get(ATTENTION_MODULE, 0.0) for gpu in x_values]
    communication_values = [
        values[gpu].get(MOE_COMM_MODULE, 0.0) for gpu in x_values
    ]
    expert_values = [
        max(values[gpu].get(FUSED_MOE_MODULE, 0.0) - comm_value, 0.0)
        for gpu, comm_value in zip(x_values, communication_values)
    ]
    series = [
        ("Attention", "#356f8c", attention_values),
        ("Expert computation", "#d18f2f", expert_values),
        ("MoE communication", "#7c3aed", communication_values),
    ]
    save_stacked_bar_plot(
        output_path,
        x_values,
        series,
        title="Qwen3-MoE Module Latency Breakdown",
        x_label="Number of GPUs (DP+EP size)",
        y_label="Latency (ms)",
    )


def write_profile_breakdowns(profile_root: Path) -> None:
    module_summary = profile_root / "module_summary.csv"
    profile_summary = profile_root / "summary.csv"
    case_gpus = load_case_gpu_counts(profile_summary)

    layer_rows: list[dict[str, object]] = []
    comm_rows: list[dict[str, object]] = []
    for row in read_csv(module_summary):
        if row["rank"] != "all":
            continue
        module = row["module"]
        gpu_count = case_gpus.get(row["case"], case_gpu_count(row["case"]))
        common = {
            "case": row["case"],
            "gpu_count": gpu_count,
            "count": row["count"],
            "total_cuda_ms": row["total_cuda_ms"],
            "avg_cuda_ms": row["avg_cuda_ms"],
        }
        for prefix, kind in (
            ("vllm:attention:", "attention"),
            ("vllm:fused_moe:", "fused_moe"),
        ):
            if module.startswith(prefix):
                layer_rows.append({
                    **common,
                    "module_kind": kind,
                    "layer": module.removeprefix(prefix),
                })
        if module.startswith("vllm:moe_comm"):
            comm_rows.append({**common, "module": module})

    write_csv(
        profile_root / "per_layer_module_summary.csv",
        [
            "case",
            "gpu_count",
            "module_kind",
            "layer",
            "count",
            "total_cuda_ms",
            "avg_cuda_ms",
        ],
        layer_rows,
    )
    write_csv(
        profile_root / "moe_comm_summary.csv",
        [
            "case",
            "gpu_count",
            "module",
            "count",
            "total_cuda_ms",
            "avg_cuda_ms",
        ],
        comm_rows,
    )


def draw_qwen_results(result_root: Path, throughput_metric: str) -> None:
    if not result_root.is_dir():
        raise SystemExit(f"Result directory does not exist: {result_root}")

    throughput_summary = result_root / "throughput" / "summary.csv"
    if throughput_summary.is_file():
        throughput_output = result_root / "qwen3_moe_dp_ep_throughput.png"
        plot_throughput(throughput_summary, throughput_output, throughput_metric)
        print(f"Throughput figure: {throughput_output}")
    else:
        print(f"Skipping throughput plot: missing {throughput_summary}.")

    profile_root = result_root / "profile"
    module_summary = profile_root / "module_summary.csv"
    profile_summary = profile_root / "summary.csv"
    if module_summary.is_file() and profile_summary.is_file():
        write_profile_breakdowns(profile_root)
        module_output = result_root / "qwen3_moe_module_latency_stacked.png"
        plot_module_latency(module_summary, profile_summary, module_output)
        print(f"Module latency figure: {module_output}")
        print(f"Per-layer summary: {profile_root / 'per_layer_module_summary.csv'}")
        print(f"MoE communication summary: {profile_root / 'moe_comm_summary.csv'}")
    else:
        print(
            "Skipping module profile plot: missing "
            f"{module_summary} or {profile_summary}."
        )


def main() -> None:
    args = parse_args()
    draw_qwen_results(args.result_root, args.throughput_metric)


if __name__ == "__main__":
    main()
