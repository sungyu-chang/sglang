#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for organizing benchmark results under the repo-level results tree."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOP_LEVEL_RESULTS_DIR = REPO_ROOT / "results"
BENCHMARK_NAME = "dp_ep_vs_tp"


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_result_root(run_group: str) -> Path:
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if env_flag("SMOKE_RUN"):
        run_name = f"{run_name}_smoke"
    return TOP_LEVEL_RESULTS_DIR / BENCHMARK_NAME / run_group / run_name


def _format_notes(raw_notes: str | None) -> list[str]:
    if raw_notes is None:
        return []
    return [line.strip() for line in raw_notes.splitlines() if line.strip()]


def write_run_readme(
    run_root: Path,
    *,
    title: str,
    script_path: str,
    status: str,
    started_at: str,
    setup: Mapping[str, str],
    planned_cases: Sequence[str],
    completed_at: str | None = None,
    artifact_paths: Mapping[str, str] | None = None,
    failure_reason: str | None = None,
    run_notes: str | None = None,
    fix_notes: str | None = None,
) -> None:
    lines: list[str] = [
        f"# {title}",
        "",
        f"- Status: `{status}`",
        f"- Started At: `{started_at}`",
        f"- Script: `{script_path}`",
    ]
    if completed_at is not None:
        lines.append(f"- Completed At: `{completed_at}`")

    lines.extend([
        "",
        "## Experiment Setup",
        "",
    ])
    for key, value in setup.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend([
        "",
        "## Planned Cases",
        "",
    ])
    for case_name in planned_cases:
        lines.append(f"- `{case_name}`")

    note_lines = _format_notes(run_notes)
    if note_lines:
        lines.extend([
            "",
            "## Run Notes",
            "",
        ])
        for note in note_lines:
            lines.append(f"- {note}")

    fix_lines = _format_notes(fix_notes)
    if fix_lines:
        lines.extend([
            "",
            "## Fix Notes",
            "",
        ])
        for note in fix_lines:
            lines.append(f"- {note}")

    if failure_reason is not None:
        lines.extend([
            "",
            "## Failure Reason",
            "",
            "```text",
            failure_reason.rstrip(),
            "```",
        ])

    if artifact_paths:
        lines.extend([
            "",
            "## Artifacts",
            "",
        ])
        for label, rel_path in artifact_paths.items():
            lines.append(f"- {label}: `{rel_path}`")

    lines.append("")
    (run_root / "README.md").write_text("\n".join(lines), encoding="utf-8")