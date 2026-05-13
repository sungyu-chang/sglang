#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one-node Qwen3-MoE DP+EP throughput and module-profile pipeline."""

from __future__ import annotations

from pathlib import Path

try:
    from results_layout import build_result_root
    from single_node_common import (
        SingleNodeBenchmarkConfig,
        SingleNodeBenchmarkRunner,
        configured_sizes,
        default_dp_sizes,
        detect_gpu_ids,
        env,
        env_bool,
        moe_a2a_backend_from_env,
        shlex_env,
        validate_sizes,
    )
except ModuleNotFoundError:
    from benchmark.dp_ep_vs_tp.results_layout import build_result_root
    from benchmark.dp_ep_vs_tp.single_node_common import (
        SingleNodeBenchmarkConfig,
        SingleNodeBenchmarkRunner,
        configured_sizes,
        default_dp_sizes,
        detect_gpu_ids,
        env,
        env_bool,
        moe_a2a_backend_from_env,
        shlex_env,
        validate_sizes,
    )


def default_qwen_result_root() -> Path:
    return build_result_root("qwen3_moe_ep_pipeline")


def prompt_count_description() -> str:
    num_prompts = env("NUM_PROMPTS", "")
    if num_prompts:
        return f"NUM_PROMPTS_PER_CASE={num_prompts}"
    return f"NUM_PROMPTS_PER_CASE=DP_SIZE*{env('PROMPTS_PER_GPU', '1000')}"


def dp_ep_server_args(dp_size: int) -> list[str]:
    args = [
        "--tp-size",
        str(dp_size),
        "--dp-size",
        str(dp_size),
        "--ep-size",
        str(dp_size),
        "--enable-dp-attention",
    ]
    moe_a2a_backend = moe_a2a_backend_from_env()
    if moe_a2a_backend:
        args.extend(["--moe-a2a-backend", moe_a2a_backend])
    return args


def make_config(
    *,
    result_root: Path,
    gpu_ids: list[str],
    profile_modules: bool,
    base_port: int,
) -> SingleNodeBenchmarkConfig:
    return SingleNodeBenchmarkConfig(
        model=env("MODEL", "Qwen/Qwen3-30B-A3B"),
        served_model_name=env("SERVED_MODEL_NAME", "qwen3-moe-bench"),
        host=env("HOST", "127.0.0.1"),
        base_port=base_port,
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
        result_root=result_root,
        server_start_timeout=int(env("SERVER_START_TIMEOUT", "900")),
        server_extra_args=shlex_env("SERVER_EXTRA_ARGS", "--dtype bfloat16"),
        bench_extra_args=shlex_env("BENCH_EXTRA_ARGS"),
        python_bin=env("PYTHON_BIN", ".venv/bin/python"),
        profile_modules=profile_modules,
        profile_delay_iterations=int(env("PROFILE_DELAY_ITERATIONS", "5")),
        profile_max_iterations=int(env("PROFILE_MAX_ITERATIONS", "20")),
        profile_with_stack=env_bool("PROFILE_WITH_STACK"),
        profile_layer_scopes=env_bool("PROFILE_LAYER_SCOPES"),
        disable_prefix_caching=True,
    )


def run_dp_ep_matrix(
    *,
    result_root: Path,
    gpu_ids: list[str],
    dp_sizes: list[int],
    profile_modules: bool,
    base_port: int,
) -> None:
    config = make_config(
        result_root=result_root,
        gpu_ids=gpu_ids,
        profile_modules=profile_modules,
        base_port=base_port,
    )
    runner = SingleNodeBenchmarkRunner(config)
    runner.require_python()
    runner.setup_dirs()
    runner.install_signal_handlers()

    try:
        for index, dp_size in enumerate(dp_sizes):
            runner.run_case(
                case_name=f"dp{dp_size}_ep",
                gpu_count=dp_size,
                port=config.base_port + index,
                server_args=dp_ep_server_args(dp_size),
                metadata={
                    "parallelism": "dp_ep",
                    "dp_size": dp_size,
                    "ep_size": dp_size,
                    "tp_size": dp_size,
                    "moe_a2a_backend": moe_a2a_backend_from_env() or "default",
                    "disable_prefix_caching": config.disable_prefix_caching,
                },
            )
        runner.summarize_results()
        if profile_modules:
            runner.summarize_module_profiles()
    finally:
        runner.cleanup_server()


def main() -> int:
    gpu_ids = detect_gpu_ids()
    gpu_count = len(gpu_ids)
    dp_sizes = configured_sizes("DP_SIZES", default_dp_sizes(gpu_count))
    validate_sizes("DP_SIZES", dp_sizes, gpu_count)
    run_throughput = env_bool("RUN_THROUGHPUT", True)
    run_profile = env_bool("RUN_PROFILE", True)
    if not run_throughput and not run_profile:
        raise SystemExit("At least one of RUN_THROUGHPUT or RUN_PROFILE must be 1.")

    result_root = default_qwen_result_root()
    throughput_root = result_root / "throughput"
    profile_root = result_root / "profile"

    print(
        "Qwen3-MoE DP+EP pipeline: "
        f"GPU_IDS={gpu_ids}, DP_SIZES={dp_sizes}, "
        f"{prompt_count_description()}, "
        f"RUN_THROUGHPUT={run_throughput}, RUN_PROFILE={run_profile}",
        flush=True,
    )
    print(
        "DISABLE_PREFIX_CACHING disables radix-cache reuse; SGLang serving still "
        "uses KV cache for autoregressive decoding.",
        flush=True,
    )
    print(
        "Qwen3-MoE pipeline always passes --disable-radix-cache. "
        "SGLang bench_serving ignores EOS by default.",
        flush=True,
    )

    if run_throughput:
        run_dp_ep_matrix(
            result_root=throughput_root,
            gpu_ids=gpu_ids,
            dp_sizes=dp_sizes,
            profile_modules=False,
            base_port=int(env("BASE_PORT", "8100")),
        )

    if run_profile:
        run_dp_ep_matrix(
            result_root=profile_root,
            gpu_ids=gpu_ids,
            dp_sizes=dp_sizes,
            profile_modules=True,
            base_port=int(env("PROFILE_BASE_PORT", env("BASE_PORT", "8100"))),
        )

    print(f"Results: {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
