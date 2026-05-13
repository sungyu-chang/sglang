#!/usr/bin/env python3
"""Analyze one-node TP vs DP+EP benchmark runs.

This script consumes one or more run directories produced by
benchmark/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py, writes a compact combined CSV,
emits a Markdown summary comparing TP and DP+EP at each GPU count, and renders
a PNG figure of total token throughput vs GPU count.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from results_layout import build_result_root
except ModuleNotFoundError:
    from benchmark.dp_ep_vs_tp.results_layout import build_result_root

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Row:
    run_dir: str
    model: str
    case: str
    series: str
    gpu_count: int
    request_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    mean_tpot_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze completed one-node TP vs DP+EP run directories.")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Completed run directories under results/dp_ep_vs_tp/one_node_online/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for combined CSV, Markdown summary, and PNG. "
            "Defaults to results/dp_ep_vs_tp/analysis/YYYY-MM-DD_HH-MM-SS."
        ),
    )
    parser.add_argument(
        "--title",
        default="Total Token Throughput vs GPU Count",
        help="Figure title.",
    )
    return parser.parse_args()


def classify_series(case_name: str) -> str:
    if case_name.startswith("tp"):
        return "TP"
    if case_name.startswith("dp") and case_name.endswith("_ep"):
        return "DP+EP"
    raise SystemExit(f"Unsupported case name: {case_name}")


def load_rows(run_dirs: list[Path]) -> list[Row]:
    rows: list[Row] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.csv"
        if not summary_path.is_file():
            raise SystemExit(f"Missing summary.csv in run directory: {run_dir}")

        with summary_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for record in reader:
                case_name = (record.get("case") or "").strip()
                if not case_name:
                    continue

                rows.append(
                    Row(
                        run_dir=run_dir.name,
                        model=(record.get("model") or "unknown").strip(),
                        case=case_name,
                        series=classify_series(case_name),
                        gpu_count=int(float(record["gpu_count"])),
                        request_throughput=float(record["request_throughput"]),
                        total_token_throughput=float(
                            record["total_token_throughput"]),
                        mean_ttft_ms=float(record["mean_ttft_ms"]),
                        mean_tpot_ms=float(record["mean_tpot_ms"]),
                    ))
    if not rows:
        raise SystemExit("No benchmark rows found in the provided run directories.")
    rows.sort(key=lambda row: (row.model, row.gpu_count, row.series, row.case))
    return rows


def default_output_dir() -> Path:
    return build_result_root("analysis")


def write_combined_csv(output_path: Path, rows: list[Row]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "run_dir",
            "model",
            "case",
            "series",
            "gpu_count",
            "request_throughput",
            "total_token_throughput",
            "mean_ttft_ms",
            "mean_tpot_ms",
        ])
        for row in rows:
            writer.writerow([
                row.run_dir,
                row.model,
                row.case,
                row.series,
                row.gpu_count,
                f"{row.request_throughput:.6f}",
                f"{row.total_token_throughput:.6f}",
                f"{row.mean_ttft_ms:.6f}",
                f"{row.mean_tpot_ms:.6f}",
            ])


def write_markdown_summary(output_path: Path, rows: list[Row]) -> None:
    by_model_gpu: dict[str, dict[int, dict[str, Row]]] = defaultdict(
        lambda: defaultdict(dict))
    run_dir_by_model: dict[str, str] = {}
    for row in rows:
        by_model_gpu[row.model][row.gpu_count][row.series] = row
        run_dir_by_model[row.model] = row.run_dir

    lines: list[str] = [
        "# One-node TP vs DP+EP Analysis",
        "",
        "This summary compares total token throughput and latency for TP and DP+EP",
        "at the same GPU count for each completed model run.",
    ]

    for model, gpu_map in sorted(by_model_gpu.items()):
        lines.extend([
            "",
            f"## {model}",
            "",
            f"- Run directory: `{run_dir_by_model[model]}`",
            "",
            "| GPUs | TP tok/s | DP+EP tok/s | DP+EP vs TP | TP TTFT ms | DP+EP TTFT ms | TP TPOT ms | DP+EP TPOT ms |",
            "| ---: | --------: | ----------: | ----------: | ---------: | ------------: | ---------: | ------------: |",
        ])
        for gpu_count in sorted(gpu_map):
            tp_row = gpu_map[gpu_count].get("TP")
            dp_ep_row = gpu_map[gpu_count].get("DP+EP")
            if tp_row is None and dp_ep_row is None:
                continue

            tp_tok = f"{tp_row.total_token_throughput:.2f}" if tp_row else "-"
            dp_ep_tok = (
                f"{dp_ep_row.total_token_throughput:.2f}" if dp_ep_row else "-")
            if tp_row and dp_ep_row and tp_row.total_token_throughput:
                speedup = dp_ep_row.total_token_throughput / tp_row.total_token_throughput
                speedup_text = f"{speedup:.2f}x"
            else:
                speedup_text = "-"
            tp_ttft = f"{tp_row.mean_ttft_ms:.2f}" if tp_row else "-"
            dp_ep_ttft = f"{dp_ep_row.mean_ttft_ms:.2f}" if dp_ep_row else "-"
            tp_tpot = f"{tp_row.mean_tpot_ms:.2f}" if tp_row else "-"
            dp_ep_tpot = f"{dp_ep_row.mean_tpot_ms:.2f}" if dp_ep_row else "-"
            lines.append(
                f"| {gpu_count} | {tp_tok} | {dp_ep_tok} | {speedup_text} | "
                f"{tp_ttft} | {dp_ep_ttft} | {tp_tpot} | {dp_ep_tpot} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_plot(run_dirs: list[Path], output_path: Path, title: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "benchmark/dp_ep_vs_tp/plot_total_token_throughput.py",
            *[str(run_dir) for run_dir in run_dirs],
            "--output",
            str(output_path),
            "--title",
            title,
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> None:
    args = parse_args()
    run_dirs = [run_dir.resolve() for run_dir in args.run_dirs]
    rows = load_rows(run_dirs)
    output_dir = args.output_dir.resolve() if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_combined_csv(output_dir / "combined_summary.csv", rows)
    write_markdown_summary(output_dir / "analysis.md", rows)
    render_plot(run_dirs, output_dir / "total_token_throughput.png", args.title)

    print(f"Analysis directory: {output_dir}")
    print(f"Combined CSV: {output_dir / 'combined_summary.csv'}")
    print(f"Markdown summary: {output_dir / 'analysis.md'}")
    print(f"Figure: {output_dir / 'total_token_throughput.png'}")


if __name__ == "__main__":
    main()
