#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one-node online TP vs DP+EP serving benchmarks."""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.dp_ep_vs_tp.results_layout import build_result_root, write_run_readme
from single_node_common import (
    SingleNodeBenchmarkConfig,
    SingleNodeBenchmarkRunner,
    configured_sizes,
    default_dp_sizes,
    default_tp_sizes,
    detect_gpu_ids,
    env,
    env_bool,
    moe_a2a_backend_from_env,
    shlex_env,
    validate_sizes,
)


def build_config(gpu_ids: list[str]) -> SingleNodeBenchmarkConfig:
    return SingleNodeBenchmarkConfig(
        model=env("MODEL", "deepseek-ai/DeepSeek-V2-Lite"),
        served_model_name=env("SERVED_MODEL_NAME", "bench-model"),
        host=env("HOST", "127.0.0.1"),
        base_port=int(env("BASE_PORT", "8100")),
        gpu_ids=gpu_ids,
        num_prompts=env("NUM_PROMPTS", ""),
        prompts_per_gpu=int(env("PROMPTS_PER_GPU", "1000")),
        input_len=env("INPUT_LEN", "1"),
        output_len=env("OUTPUT_LEN", "256"),
        num_warmups=int(env("NUM_WARMUPS", "100")),
        request_rate=env("REQUEST_RATE", "inf"),
        max_concurrency=env("MAX_CONCURRENCY", ""),
        max_concurrency_per_gpu=env("MAX_CONCURRENCY_PER_GPU", ""),
        max_model_len=env("MAX_MODEL_LEN", ""),
        result_root=build_result_root("one_node_online"),
        server_start_timeout=int(env("SERVER_START_TIMEOUT", "900")),
        server_extra_args=shlex_env("SERVER_EXTRA_ARGS"),
        bench_extra_args=shlex_env("BENCH_EXTRA_ARGS"),
        python_bin=env("PYTHON_BIN", ".venv/bin/python"),
        profile_modules=env_bool("PROFILE_MODULES"),
        profile_delay_iterations=int(env("PROFILE_DELAY_ITERATIONS", "5")),
        profile_max_iterations=int(env("PROFILE_MAX_ITERATIONS", "20")),
        profile_with_stack=env_bool("PROFILE_WITH_STACK"),
        profile_layer_scopes=env_bool("PROFILE_LAYER_SCOPES"),
        disable_prefix_caching=env_bool("DISABLE_PREFIX_CACHING", True),
        port_release_timeout=int(env("PORT_RELEASE_TIMEOUT", "60")),
    )


def write_run_summary(
    *,
    config: SingleNodeBenchmarkConfig,
    tp_sizes: list[int],
    dp_sizes: list[int],
    moe_a2a_backend: str,
    status: str,
    started_at: str,
    completed_at: str | None = None,
    failure_reason: str | None = None,
) -> None:
    planned_cases = [f"tp{size}" for size in tp_sizes]
    planned_cases.extend(f"dp{size}_ep" for size in dp_sizes)
    artifact_paths = {
        "server logs": "server_logs/",
        "bench logs": "bench_logs/",
        "json results": "json/",
        "summary": "summary.csv",
    }
    if config.profile_modules:
        artifact_paths.update({
            "profiler traces": "profiler_traces/",
            "module summary": "module_summary.csv",
        })

    write_run_readme(
        config.result_root,
        title="One-node TP vs DP+EP Run",
        script_path="benchmark/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py",
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        setup={
            "model": config.model,
            "served_model_name": config.served_model_name,
            "host": config.host,
            "base_port": str(config.base_port),
            "gpu_ids": ",".join(config.gpu_ids),
            "gpu_count": str(config.gpu_count),
            "tp_sizes": " ".join(str(size) for size in tp_sizes),
            "dp_sizes": " ".join(str(size) for size in dp_sizes),
            "num_prompts": config.num_prompts or "prompts_per_gpu * case_gpu_count",
            "prompts_per_gpu": str(config.prompts_per_gpu),
            "input_len": config.input_len,
            "output_len": config.output_len,
            "num_warmups": str(config.num_warmups),
            "request_rate": config.request_rate,
            "max_concurrency": config.max_concurrency or "unset",
            "max_concurrency_per_gpu": config.max_concurrency_per_gpu or "unset",
            "max_model_len": config.max_model_len or "SGLang model default",
            "moe_a2a_backend": moe_a2a_backend or "SGLang default",
            "server_start_timeout": str(config.server_start_timeout),
            "port_release_timeout": str(config.port_release_timeout),
            "server_extra_args": " ".join(config.server_extra_args) or "(none)",
            "bench_extra_args": " ".join(config.bench_extra_args) or "(none)",
            "python_bin": config.python_bin,
            "profile_modules": str(config.profile_modules).lower(),
            "profile_layer_scopes": str(config.profile_layer_scopes).lower(),
            "disable_prefix_caching": str(config.disable_prefix_caching).lower(),
            "smoke_run": str(config.result_root.name.endswith("_smoke")).lower(),
        },
        planned_cases=planned_cases,
        artifact_paths=artifact_paths,
        failure_reason=failure_reason,
        run_notes=env("RUN_NOTES", ""),
        fix_notes=env("FIX_NOTES", ""),
    )


def main() -> int:
    gpu_ids = detect_gpu_ids()
    gpu_count = len(gpu_ids)
    run_tp = env_bool("RUN_TP", True)
    run_dp_ep = env_bool("RUN_DP_EP", True)
    if not run_tp and not run_dp_ep:
        raise SystemExit("At least one of RUN_TP or RUN_DP_EP must be enabled")

    tp_sizes = (configured_sizes("TP_SIZES", default_tp_sizes(gpu_count))
                if run_tp else [])
    dp_sizes = (configured_sizes("DP_SIZES", default_dp_sizes(gpu_count))
                if run_dp_ep else [])
    if run_tp:
        validate_sizes("TP_SIZES", tp_sizes, gpu_count)
    if run_dp_ep:
        validate_sizes("DP_SIZES", dp_sizes, gpu_count)

    moe_a2a_backend = moe_a2a_backend_from_env("")
    config = build_config(gpu_ids)
    runner = SingleNodeBenchmarkRunner(config)
    runner.require_python()
    runner.setup_dirs()
    runner.install_signal_handlers()

    started_at = datetime.now().isoformat(timespec="seconds")
    write_run_summary(
        config=config,
        tp_sizes=tp_sizes,
        dp_sizes=dp_sizes,
        moe_a2a_backend=moe_a2a_backend,
        status="running",
        started_at=started_at,
    )

    try:
        print(
            "Detected benchmark matrix: "
            f"GPU_IDS={gpu_ids}, TP_SIZES={tp_sizes}, DP_SIZES={dp_sizes}",
            flush=True,
        )
        case_index = 0
        for tp_size in tp_sizes:
            runner.run_case(
                case_name=f"tp{tp_size}",
                gpu_count=tp_size,
                port=config.base_port + case_index,
                server_args=["--tp-size", str(tp_size)],
                metadata={"parallelism": "tp", "tp_size": tp_size},
            )
            case_index += 1

        for dp_size in dp_sizes:
            server_args = [
                "--tp-size",
                str(dp_size),
                "--dp-size",
                str(dp_size),
                "--ep-size",
                str(dp_size),
                "--enable-dp-attention",
            ]
            if moe_a2a_backend:
                server_args.extend(["--moe-a2a-backend", moe_a2a_backend])
            runner.run_case(
                case_name=f"dp{dp_size}_ep",
                gpu_count=dp_size,
                port=config.base_port + case_index,
                server_args=server_args,
                metadata={
                    "parallelism": "dp_ep",
                    "dp_size": dp_size,
                    "ep_size": dp_size,
                    "moe_a2a_backend": moe_a2a_backend or "default",
                },
            )
            case_index += 1

        runner.summarize_results()
        if config.profile_modules:
            runner.summarize_module_profiles()
        write_run_summary(
            config=config,
            tp_sizes=tp_sizes,
            dp_sizes=dp_sizes,
            moe_a2a_backend=moe_a2a_backend,
            status="completed",
            started_at=started_at,
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        print(f"Results: {config.result_root}")
        return 0
    except Exception:
        write_run_summary(
            config=config,
            tp_sizes=tp_sizes,
            dp_sizes=dp_sizes,
            moe_a2a_backend=moe_a2a_backend,
            status="failed",
            started_at=started_at,
            completed_at=datetime.now().isoformat(timespec="seconds"),
            failure_reason=traceback.format_exc(),
        )
        raise
    finally:
        runner.cleanup_server()


if __name__ == "__main__":
    raise SystemExit(main())
