#!/usr/bin/env python3
"""Run one-node DP+EP benchmarks for Qwen MoE presets.

This wrapper launches separate one-node runs for Qwen1.5 MoE and Qwen3 MoE,
using DP+EP sizes from 1 through the detected GPU count by default, and then
invokes the analysis script after all requested runs complete. TP baselines can
be included explicitly with --include-tp.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from results_layout import build_result_root
except ModuleNotFoundError:
    from benchmark.dp_ep_vs_tp.results_layout import build_result_root

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "results" / "dp_ep_vs_tp" / "one_node_online"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str


MODEL_PRESETS = {
    "qwen1.5": ModelSpec(
        name="qwen1.5",
        model_id="Qwen/Qwen1.5-MoE-A2.7B",
    ),
    "qwen3": ModelSpec(
        name="qwen3",
        model_id="Qwen/Qwen3-30B-A3B",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen one-node DP+EP experiments and analyze them.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_PRESETS),
        default=["qwen1.5", "qwen3"],
        help="Model presets to run. Defaults to both Qwen1.5 and Qwen3.",
    )
    parser.add_argument(
        "--gpu-count",
        default=None,
        help="GPU count to use. Defaults to CUDA_VISIBLE_DEVICES or nvidia-smi.",
    )
    parser.add_argument(
        "--include-tp",
        action="store_true",
        help="Also run TP baselines. Defaults to DP+EP only.",
    )
    parser.add_argument(
        "--tp-sizes",
        default=None,
        help=(
            "Space-separated TP sizes. Defaults to powers of two when "
            "--include-tp is set."
        ),
    )
    parser.add_argument(
        "--dp-sizes",
        default=None,
        help=(
            "Space-separated DP+EP sizes. Defaults to 1 through the detected "
            "GPU count."
        ),
    )
    parser.add_argument("--input-len", default="1")
    parser.add_argument("--output-len", default="256")
    parser.add_argument(
        "--num-warmups",
        default="100",
        help="Number of warmup requests for each sglang.bench_serving run.",
    )
    parser.add_argument(
        "--num-prompts",
        default=None,
        help="Override per-case prompt count. Defaults to 1000 * GPUs used.",
    )
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument(
        "--max-concurrency",
        default=None,
        help="Absolute benchmark client max concurrency for every case.",
    )
    parser.add_argument(
        "--max-concurrency-per-gpu",
        default=None,
        help="Benchmark client max concurrency per GPU/DP rank, scaled by case size.",
    )
    parser.add_argument("--server-extra-args", default="--dtype bfloat16")
    parser.add_argument(
        "--all2all-backend",
        "--moe-a2a-backend",
        dest="moe_a2a_backend",
        default=None,
        help="Explicit SGLang MoE A2A backend. Defaults to SGLang's backend.",
    )
    return parser.parse_args()


def build_run_notes(spec: ModelSpec, args: argparse.Namespace) -> str:
    gpu_scope = args.gpu_count or "detected GPU count"
    dp_scope = args.dp_sizes or f"1..{gpu_scope}"
    tp_scope = args.tp_sizes if args.include_tp else "disabled"
    if args.include_tp and args.tp_sizes is None:
        tp_scope = f"powers of two up to {gpu_scope}"
    return (
        f"Single-node DP+EP comparison for {spec.model_id}, "
        f"input {args.input_len}, output {args.output_len}, "
        f"prompts={args.num_prompts or '1000 * GPUs used'}, "
        f"warmups={args.num_warmups}, "
        f"max_concurrency={args.max_concurrency or 'unset'}, "
        f"max_concurrency_per_gpu={args.max_concurrency_per_gpu or 'unset'}, "
        f"TP={tp_scope}, DP+EP={dp_scope}, ignore_eos=true, "
        "disable_radix_cache=true, "
        f"moe_a2a_backend={args.moe_a2a_backend or 'SGLang default'}"
    )


def existing_run_dirs() -> set[Path]:
    if not RUN_ROOT.is_dir():
        return set()
    return {path.resolve() for path in RUN_ROOT.iterdir() if path.is_dir()}


def detect_new_run_dir(before: set[Path]) -> Path:
    after = existing_run_dirs()
    new_dirs = sorted(after - before)
    if len(new_dirs) == 1:
        return new_dirs[0]
    if not new_dirs:
        raise RuntimeError("Benchmark finished but no new run directory was created.")
    return max(new_dirs, key=lambda path: path.stat().st_mtime)


def run_model(spec: ModelSpec, args: argparse.Namespace) -> Path:
    env = os.environ.copy()
    env["PATH"] = f"{REPO_ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
    env.update({
        "MODEL": spec.model_id,
        "SERVER_EXTRA_ARGS": args.server_extra_args,
        "RUN_TP": "1" if args.include_tp else "0",
        "RUN_DP_EP": "1",
        "INPUT_LEN": args.input_len,
        "OUTPUT_LEN": args.output_len,
        "NUM_WARMUPS": args.num_warmups,
        "REQUEST_RATE": args.request_rate,
        "RUN_NOTES": build_run_notes(spec, args),
        "PYTHON_BIN": str(REPO_ROOT / ".venv" / "bin" / "python"),
    })
    if args.moe_a2a_backend is not None:
        env["MOE_A2A_BACKEND"] = args.moe_a2a_backend
    if args.gpu_count is not None:
        env["GPU_COUNT"] = args.gpu_count
    if args.num_prompts is not None:
        env["NUM_PROMPTS"] = args.num_prompts
    if args.max_concurrency is not None:
        env["MAX_CONCURRENCY"] = args.max_concurrency
    if args.max_concurrency_per_gpu is not None:
        env["MAX_CONCURRENCY_PER_GPU"] = args.max_concurrency_per_gpu
    if args.include_tp and args.tp_sizes is not None:
        env["TP_SIZES"] = args.tp_sizes
    if args.dp_sizes is not None:
        env["DP_SIZES"] = args.dp_sizes

    before = existing_run_dirs()
    subprocess.run(
        [sys.executable, "benchmark/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    return detect_new_run_dir(before)


def analyze_runs(run_dirs: list[Path]) -> Path:
    analysis_dir = build_result_root("analysis")
    subprocess.run(
        [
            sys.executable,
            "benchmark/dp_ep_vs_tp/analyze_one_node_runs.py",
            *[str(run_dir) for run_dir in run_dirs],
            "--output-dir",
            str(analysis_dir),
            "--title",
            "Qwen DP+EP Total Token Throughput",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return analysis_dir


def main() -> int:
    args = parse_args()
    run_dirs: list[Path] = []
    for model_key in args.models:
        spec = MODEL_PRESETS[model_key]
        print(f"=== Running {spec.model_id} ===", flush=True)
        run_dir = run_model(spec, args)
        run_dirs.append(run_dir)
        print(f"Run directory for {spec.model_id}: {run_dir}", flush=True)

    analysis_dir = analyze_runs(run_dirs)
    print(f"Analysis directory: {analysis_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
