#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matplotlib helpers for dp_ep_vs_tp benchmark figures."""

from __future__ import annotations

from pathlib import Path


def pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import academia_style
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for benchmark figures. Install the "
            "benchmark extra, for example: uv pip install -e '.[bench]'"
        ) from exc
    return plt, academia_style


def save_line_plot(
    output_path: Path,
    x_values: list[int],
    y_values: list[float],
    *,
    title: str,
    x_label: str,
    y_label: str,
    color: str = "#1f5f5b",
) -> None:
    plt, academia_style = pyplot()
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=120)
    ax.plot(x_values, y_values, marker=academia_style.MARKERS[0], color=color)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_values)
    academia_style.style_fig(fig, enforce=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_stacked_bar_plot(
    output_path: Path,
    x_values: list[int],
    series: list[tuple[str, str, list[float]]],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    plt, academia_style = pyplot()
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=120)
    fig.suptitle(title, fontproperties=academia_style.fp, y=0.98)
    bottoms = [0.0 for _ in x_values]
    for label, color, values in series:
        ax.bar(x_values, values, bottom=bottoms, label=label, color=color)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_values)
    academia_style.style_fig(
        fig,
        legend_ncol=len(series),
        enforce=True,
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    fig.savefig(output_path)
    plt.close(fig)


def save_series_panels(
    output_path: Path,
    *,
    title: str,
    model_series: dict[str, dict[str, list[tuple[int, float]]]],
    colors: dict[str, str],
) -> None:
    plt, academia_style = pyplot()
    models = list(model_series.items())
    fig_height = max(3.2 * len(models), 3.2)
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(11.0, fig_height),
        dpi=120,
        squeeze=False,
        sharex=True,
    )
    axes_flat = [axis for row in axes for axis in row]

    all_gpu_counts = sorted({
        gpu_count
        for series_map in model_series.values()
        for points in series_map.values()
        for gpu_count, _ in points
    })

    for ax, (model_name, series_map) in zip(axes_flat, models):
        for series_index, (series_name, color) in enumerate(colors.items()):
            points = series_map.get(series_name, [])
            if not points:
                continue
            x_values = [gpu_count for gpu_count, _ in points]
            y_values = [throughput for _, throughput in points]
            ax.plot(
                x_values,
                y_values,
                marker=academia_style.MARKERS[
                    series_index % len(academia_style.MARKERS)
                ],
                linestyle=academia_style.LINESTYLES[
                    series_index % len(academia_style.LINESTYLES)
                ],
                label=series_name,
                color=color,
            )
        ax.set_title(model_name, loc="left")
        ax.set_ylabel("Total token throughput")
        ax.legend()

    axes_flat[-1].set_xlabel("GPUs involved")
    if all_gpu_counts:
        axes_flat[-1].set_xticks(all_gpu_counts)

    fig.suptitle(title)
    academia_style.style_fig(fig, legend_ncol=len(colors), enforce=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
