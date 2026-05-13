#!/usr/bin/env python3
"""Plot total token throughput vs GPU count for TP and DP+EP runs.

This script reads one or more run directories or summary CSV files produced by
benchmark/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py and writes a PNG figure.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

try:
    from matplotlib_plots import save_line_plot, save_series_panels
except ModuleNotFoundError:
    from benchmark.dp_ep_vs_tp.matplotlib_plots import (
        save_line_plot,
        save_series_panels,
    )


SeriesMap = dict[str, list[tuple[int, float]]]
ModelMap = dict[str, SeriesMap]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw total token throughput vs GPU count from one or more "
            "dp_ep_vs_tp summary CSVs or run directories."
        ))
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Run directories containing summary.csv or explicit summary.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Defaults to "
            "results/dp_ep_vs_tp/total_token_throughput.png"
        ),
    )
    parser.add_argument(
        "--title",
        default="Total Token Throughput vs GPU Count",
        help="Figure title.",
    )
    return parser.parse_args()


def resolve_summary_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            summary_path = input_path / "summary.csv"
            if not summary_path.is_file():
                raise SystemExit(
                    f"Missing summary.csv under run directory: {input_path}"
                )
            paths.append(summary_path)
        elif input_path.is_file():
            paths.append(input_path)
        else:
            raise SystemExit(f"Input path does not exist: {input_path}")
    return paths


def series_name(case_name: str) -> str | None:
    if case_name.startswith("tp"):
        return "TP"
    if case_name.startswith("dp") and case_name.endswith("_ep"):
        return "DP+EP"
    return None


def load_rows(summary_paths: list[Path]) -> ModelMap:
    model_series: ModelMap = defaultdict(lambda: defaultdict(list))
    for summary_path in summary_paths:
        with summary_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                case_name = (row.get("case") or "").strip()
                series = series_name(case_name)
                if series is None:
                    continue

                model_name = (row.get("model") or "unknown").strip()
                try:
                    gpu_count = int(float(row["gpu_count"]))
                    total_token_throughput = float(row["total_token_throughput"])
                except (TypeError, ValueError, KeyError) as exc:
                    raise SystemExit(
                        f"Invalid row in {summary_path}: {row}"
                    ) from exc

                model_series[model_name][series].append(
                    (gpu_count, total_token_throughput)
                )

    if not model_series:
        raise SystemExit("No TP or DP+EP rows found in the provided summaries.")

    for series_map in model_series.values():
        for points in series_map.values():
            points.sort(key=lambda item: item[0])

    return dict(model_series)


def render_png(title: str, model_series: ModelMap, output_path: Path) -> None:
    if len(model_series) == 1:
        series_map = next(iter(model_series.values()))
        if len(series_map) == 1:
            series_name, points = next(iter(series_map.items()))
            x_values = [gpu_count for gpu_count, _ in points]
            y_values = [throughput for _, throughput in points]
            x_label = "Number of GPUs"
            if series_name == "DP+EP":
                x_label = "Number of GPUs (DP+EP size)"
            elif series_name == "TP":
                x_label = "Number of GPUs (TP size)"
            save_line_plot(
                output_path,
                x_values,
                y_values,
                title=title,
                x_label=x_label,
                y_label="total token throughput",
                color="#1f5f5b",
            )
            return

    save_series_panels(
        output_path,
        title=title,
        model_series=model_series,
        colors={"TP": "#0f766e", "DP+EP": "#b45309"},
    )


def default_output_path() -> Path:
    return Path("results/dp_ep_vs_tp/total_token_throughput.png")


def main() -> None:
    args = parse_args()
    summary_paths = resolve_summary_paths(args.inputs)
    model_series = load_rows(summary_paths)
    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_png(args.title, model_series, output_path)
    print(f"Wrote figure to {output_path}")


if __name__ == "__main__":
    main()
