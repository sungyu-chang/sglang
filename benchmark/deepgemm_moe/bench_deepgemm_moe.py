#!/usr/bin/env python3
"""Benchmark SGLang DeepGEMM MoE expert compute without dispatch/combine.

This benchmark mocks the expert assignment that would normally be produced by
MoE routing and times execution on a single CUDA device. The default backend
uses SGLang's DeepGemmRunnerCore and skips top-k computation entirely.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch


BENCHMARK_NAME = "deepgemm_moe"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / BENCHMARK_NAME
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
FP8_BLOCK_SIZE = 128

# This benchmark sweeps explicit batch sizes. Avoid SGLang's server-oriented
# all-M precompile path unless the caller intentionally enables it.
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "false")

# SGLang JIT kernels may spawn build tools such as ninja. When this script is
# launched as `.venv/bin/python` without activating the environment, those tools
# can be present in the venv but absent from PATH.
VENV_BIN = Path(sys.prefix) / "bin"
if (VENV_BIN / "ninja").exists():
    os.environ["PATH"] = f"{VENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}"


@dataclass(frozen=True)
class ModelPreset:
    name: str
    hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    topk: int
    source: str


MODEL_PRESETS = {
    "deepseek-v3": ModelPreset(
        name="deepseek-v3",
        hidden_size=7168,
        moe_intermediate_size=2048,
        num_experts=256,
        topk=8,
        source="DeepSeek-V3 / DeepSeek-R1 MoE config",
    ),
    "qwen3-235b-a22b": ModelPreset(
        name="qwen3-235b-a22b",
        hidden_size=4096,
        moe_intermediate_size=1536,
        num_experts=128,
        topk=8,
        source="Qwen3-235B-A22B MoE config",
    ),
    "qwen3-30b-a3b": ModelPreset(
        name="qwen3-30b-a3b",
        hidden_size=2048,
        moe_intermediate_size=768,
        num_experts=128,
        topk=8,
        source="Qwen3-30B-A3B MoE config",
    ),
}


@dataclass(frozen=True)
class GemmShape:
    op: str
    n: int
    k: int


@dataclass
class DeepGemmInputs:
    lhs: tuple[torch.Tensor, torch.Tensor]
    rhs: tuple[torch.Tensor, torch.Tensor]
    out: torch.Tensor
    m_indices: torch.Tensor
    token_counts: list[int]
    padded_token_counts: list[int]
    logical_tokens: int
    padded_tokens: int


@dataclass
class DeepGemmRunnerBenchInputs:
    runner_input: object
    quant_info: object
    running_state: dict[str, object]
    token_counts: list[int]
    padded_token_counts: list[int]
    logical_tokens: int
    padded_tokens: int


@dataclass
class BenchmarkResult:
    model_preset: str
    backend: str
    op: str
    batch_size: int
    active_experts: int
    distribution: str
    skew_minor_fraction: float
    tp_size: int
    hidden_size: int
    moe_intermediate_size: int
    intermediate_per_partition: int
    num_experts: int
    topk: int
    n: int
    k: int
    logical_tokens: int
    padded_tokens: int
    latency_us_mean: float
    latency_us_p50: float
    latency_us_p90: float
    latency_us_p99: float
    throughput_tokens_per_s: float
    effective_tflops: float
    warmup_iters: int
    timed_warmup_iters: int
    measure_iters: int
    use_cuda_graph: bool
    cuda_graph_inner_iters: int


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def per_token_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, k = x.shape
    pad_size = (FP8_BLOCK_SIZE - (k % FP8_BLOCK_SIZE)) % FP8_BLOCK_SIZE
    if pad_size:
        x = torch.nn.functional.pad(x, (0, pad_size), value=0)
    x_view = x.view(m, -1, FP8_BLOCK_SIZE)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    x_fp8 = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return x_fp8.view(m, k + pad_size)[:, :k].contiguous(), (x_amax / 448.0).contiguous()


def per_block_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    n, k = x.shape
    padded_n = ceil_div(n, FP8_BLOCK_SIZE) * FP8_BLOCK_SIZE
    padded_k = ceil_div(k, FP8_BLOCK_SIZE) * FP8_BLOCK_SIZE
    x_padded = torch.zeros((padded_n, padded_k), dtype=x.dtype, device=x.device)
    x_padded[:n, :k] = x
    x_view = x_padded.view(-1, FP8_BLOCK_SIZE, padded_k // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_fp8 = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    scales = (x_amax / 448.0).view(x_view.size(0), x_view.size(2)).contiguous()
    return x_fp8.view_as(x_padded)[:n, :k].contiguous(), scales


def per_expert_block_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3
    data = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (x.size(0), ceil_div(x.size(1), FP8_BLOCK_SIZE), ceil_div(x.size(2), FP8_BLOCK_SIZE)),
        device=x.device,
        dtype=torch.float32,
    )
    for expert_id in range(x.size(0)):
        data[expert_id], scales[expert_id] = per_block_cast_to_fp8(x[expert_id])
    return data, scales


def resolve_device(device_arg: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; DeepGEMM MoE benchmark requires one CUDA GPU.")
    device = torch.device(device_arg)
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device_arg!r}.")
    index = 0 if device.index is None else device.index
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def require_deepgemm() -> tuple[object, Callable[[], int]]:
    from sglang.srt.layers import deep_gemm_wrapper

    if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        raise RuntimeError(
            "SGLang DeepGEMM wrapper reports ENABLE_JIT_DEEPGEMM=False. "
            "Check GPU SM version, deep_gemm installation, and SGLANG_ENABLE_JIT_DEEPGEMM."
        )
    from deep_gemm.utils.layout import get_mk_alignment_for_contiguous_layout

    return deep_gemm_wrapper, get_mk_alignment_for_contiguous_layout


def make_token_counts(
    batch_size: int,
    active_experts: int,
    distribution: str,
    skew_minor_fraction: float,
) -> list[int]:
    if distribution == "single":
        return [batch_size] + [0 for _ in range(active_experts - 1)]
    if distribution == "even":
        if batch_size % active_experts != 0:
            raise ValueError(
                f"batch_size={batch_size} cannot be evenly split across active_experts={active_experts}."
            )
        return [batch_size // active_experts for _ in range(active_experts)]
    if distribution == "uniform":
        base = batch_size // active_experts
        extra = batch_size % active_experts
        return [base + (1 if idx < extra else 0) for idx in range(active_experts)]
    if distribution == "skewed":
        if active_experts == 1:
            return [batch_size]
        if not 0.0 < skew_minor_fraction < 1.0:
            raise ValueError("skew_minor_fraction must be in (0, 1).")
        num_minor_experts = active_experts - 1
        min_minor_tokens = min(num_minor_experts, max(0, batch_size - 1))
        total_minor_tokens = round(batch_size * skew_minor_fraction)
        total_minor_tokens = max(min_minor_tokens, total_minor_tokens)
        total_minor_tokens = min(batch_size - 1, total_minor_tokens)
        base = total_minor_tokens // num_minor_experts
        extra = total_minor_tokens % num_minor_experts
        minor_counts = [base + (1 if idx < extra else 0) for idx in range(num_minor_experts)]
        return [batch_size - total_minor_tokens] + minor_counts
    raise ValueError(f"Unsupported distribution: {distribution}")


def make_inputs(
    *,
    shape: GemmShape,
    batch_size: int,
    active_experts: int,
    distribution: str,
    skew_minor_fraction: float,
    device: torch.device,
    alignment: int,
) -> DeepGemmInputs:
    token_counts = make_token_counts(batch_size, active_experts, distribution, skew_minor_fraction)
    padded_token_counts = [ceil_div(count, alignment) * alignment if count else 0 for count in token_counts]
    padded_tokens = sum(padded_token_counts)
    logical_tokens = sum(token_counts)
    if logical_tokens != batch_size:
        raise RuntimeError(f"Token count mismatch: {logical_tokens=} {batch_size=}")
    if padded_tokens == 0:
        raise ValueError("No tokens to benchmark.")

    lhs_bf16 = torch.randn((padded_tokens, shape.k), device=device, dtype=torch.bfloat16)
    rhs_bf16 = torch.randn((active_experts, shape.n, shape.k), device=device, dtype=torch.bfloat16)
    m_indices = torch.empty((padded_tokens,), device=device, dtype=torch.int32)
    out = torch.empty((padded_tokens, shape.n), device=device, dtype=torch.bfloat16)

    start = 0
    for expert_id, (count, padded_count) in enumerate(zip(token_counts, padded_token_counts)):
        actual_end = start + count
        padded_end = start + padded_count
        if count:
            m_indices[start:actual_end] = expert_id
        if padded_count > count:
            m_indices[actual_end:padded_end] = -1
        start = padded_end

    lhs = per_token_cast_to_fp8(lhs_bf16)
    rhs_data = torch.empty_like(rhs_bf16, dtype=torch.float8_e4m3fn)
    rhs_scales = torch.empty(
        (active_experts, ceil_div(shape.n, FP8_BLOCK_SIZE), ceil_div(shape.k, FP8_BLOCK_SIZE)),
        device=device,
        dtype=torch.float32,
    )
    for expert_id in range(active_experts):
        rhs_data[expert_id], rhs_scales[expert_id] = per_block_cast_to_fp8(rhs_bf16[expert_id])

    return DeepGemmInputs(
        lhs=lhs,
        rhs=(rhs_data, rhs_scales),
        out=out,
        m_indices=m_indices,
        token_counts=token_counts,
        padded_token_counts=padded_token_counts,
        logical_tokens=logical_tokens,
        padded_tokens=padded_tokens,
    )


def make_runner_inputs(
    *,
    preset: ModelPreset,
    batch_size: int,
    active_experts: int,
    distribution: str,
    skew_minor_fraction: float,
    tp_size: int,
    device: torch.device,
    alignment: int,
) -> DeepGemmRunnerBenchInputs:
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        DeepGemmMoeQuantInfo,
        DeepGemmRunnerInput,
    )

    if preset.moe_intermediate_size % tp_size != 0:
        raise ValueError(
            f"moe_intermediate_size={preset.moe_intermediate_size} is not divisible by tp_size={tp_size}."
        )

    intermediate = preset.moe_intermediate_size // tp_size
    token_counts = make_token_counts(batch_size, active_experts, distribution, skew_minor_fraction)
    padded_token_counts = [ceil_div(count, alignment) * alignment if count else 0 for count in token_counts]
    logical_tokens = sum(token_counts)
    padded_tokens = sum(padded_token_counts)
    if logical_tokens != batch_size:
        raise RuntimeError(f"Token count mismatch: {logical_tokens=} {batch_size=}")
    if padded_tokens == 0:
        raise ValueError("No tokens to benchmark.")

    hidden_states_bf16 = torch.randn(
        (padded_tokens, preset.hidden_size), device=device, dtype=torch.bfloat16
    )
    hidden_states, hidden_states_scale = per_token_cast_to_fp8(hidden_states_bf16)
    m_indices = torch.empty((padded_tokens,), device=device, dtype=torch.int32)

    start = 0
    for expert_id, (count, padded_count) in enumerate(zip(token_counts, padded_token_counts)):
        actual_end = start + count
        padded_end = start + padded_count
        if count:
            m_indices[start:actual_end] = expert_id
        if padded_count > count:
            m_indices[actual_end:padded_end] = -1
        start = padded_end

    w13_bf16 = torch.randn(
        (active_experts, 2 * intermediate, preset.hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )
    w2_bf16 = torch.randn(
        (active_experts, preset.hidden_size, intermediate),
        device=device,
        dtype=torch.bfloat16,
    )
    w13_weight, w13_scale = per_expert_block_cast_to_fp8(w13_bf16)
    w2_weight, w2_scale = per_expert_block_cast_to_fp8(w2_bf16)

    runner_input = DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=False,
        m_indices=m_indices,
    )
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        use_fp8=True,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=[FP8_BLOCK_SIZE, FP8_BLOCK_SIZE],
        is_fp4_experts=False,
    )
    running_state = {
        "all_tokens": padded_tokens,
        "hidden_states_device": device,
        "hidden_states_dtype": torch.bfloat16,
        "hidden_states_shape": (padded_tokens, preset.hidden_size),
    }

    return DeepGemmRunnerBenchInputs(
        runner_input=runner_input,
        quant_info=quant_info,
        running_state=running_state,
        token_counts=token_counts,
        padded_token_counts=padded_token_counts,
        logical_tokens=logical_tokens,
        padded_tokens=padded_tokens,
    )


def shape_for_op(preset: ModelPreset, tp_size: int, op: str) -> GemmShape:
    if preset.moe_intermediate_size % tp_size != 0:
        raise ValueError(
            f"moe_intermediate_size={preset.moe_intermediate_size} is not divisible by tp_size={tp_size}."
        )
    intermediate = preset.moe_intermediate_size // tp_size
    if op == "gateup":
        return GemmShape(op=op, n=2 * intermediate, k=preset.hidden_size)
    if op == "down":
        return GemmShape(op=op, n=preset.hidden_size, k=intermediate)
    raise ValueError(f"Unsupported single GEMM op: {op}")


def run_timed(
    callable_op: Callable[[], None],
    warmup_iters: int,
    timed_warmup_iters: int,
    measure_iters: int,
    *,
    use_cuda_graph: bool,
    cuda_graph_inner_iters: int,
) -> list[float]:
    if cuda_graph_inner_iters <= 0:
        raise ValueError("cuda_graph_inner_iters must be positive.")
    if timed_warmup_iters < 0:
        raise ValueError("timed_warmup_iters must be non-negative.")

    for _ in range(warmup_iters):
        callable_op()
    torch.cuda.synchronize()

    timed_op = callable_op
    latency_divisor = 1
    if use_cuda_graph:
        if warmup_iters == 0:
            callable_op()
            torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(cuda_graph_inner_iters):
                callable_op()
        torch.cuda.synchronize()
        timed_op = graph.replay
        latency_divisor = cuda_graph_inner_iters

    for _ in range(timed_warmup_iters):
        timed_op()
    torch.cuda.synchronize()

    latencies_us: list[float] = []
    for _ in range(measure_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        timed_op()
        end_event.record()
        end_event.synchronize()
        latencies_us.append((start_event.elapsed_time(end_event) * 1000.0) / latency_divisor)
    torch.cuda.synchronize()
    return latencies_us


def summarize_result(
    *,
    preset: ModelPreset,
    backend: str,
    op: str,
    batch_size: int,
    active_experts: int,
    distribution: str,
    skew_minor_fraction: float,
    tp_size: int,
    shape: GemmShape,
    logical_tokens: int,
    padded_tokens: int,
    latencies_us: list[float],
    warmup_iters: int,
    timed_warmup_iters: int,
    measure_iters: int,
    use_cuda_graph: bool,
    cuda_graph_inner_iters: int,
) -> BenchmarkResult:
    mean_us = statistics.fmean(latencies_us)
    intermediate = preset.moe_intermediate_size // tp_size
    if op == "mlp":
        gateup_flops = 2 * logical_tokens * (2 * intermediate) * preset.hidden_size
        down_flops = 2 * logical_tokens * preset.hidden_size * intermediate
        flops = gateup_flops + down_flops
    else:
        flops = 2 * logical_tokens * shape.n * shape.k
    seconds = mean_us / 1_000_000.0
    return BenchmarkResult(
        model_preset=preset.name,
        backend=backend,
        op=op,
        batch_size=batch_size,
        active_experts=active_experts,
        distribution=distribution,
        skew_minor_fraction=skew_minor_fraction,
        tp_size=tp_size,
        hidden_size=preset.hidden_size,
        moe_intermediate_size=preset.moe_intermediate_size,
        intermediate_per_partition=intermediate,
        num_experts=preset.num_experts,
        topk=preset.topk,
        n=shape.n,
        k=shape.k,
        logical_tokens=logical_tokens,
        padded_tokens=padded_tokens,
        latency_us_mean=mean_us,
        latency_us_p50=percentile(latencies_us, 50),
        latency_us_p90=percentile(latencies_us, 90),
        latency_us_p99=percentile(latencies_us, 99),
        throughput_tokens_per_s=logical_tokens / seconds if seconds else 0.0,
        effective_tflops=(flops / seconds / 1e12) if seconds else 0.0,
        warmup_iters=warmup_iters,
        timed_warmup_iters=timed_warmup_iters,
        measure_iters=measure_iters,
        use_cuda_graph=use_cuda_graph,
        cuda_graph_inner_iters=cuda_graph_inner_iters,
    )


def benchmark_runner_mlp(
    *,
    preset: ModelPreset,
    batch_size: int,
    active_experts: int,
    distribution: str,
    skew_minor_fraction: float,
    tp_size: int,
    device: torch.device,
    alignment: int,
    warmup_iters: int,
    timed_warmup_iters: int,
    measure_iters: int,
    use_cuda_graph: bool,
    cuda_graph_inner_iters: int,
) -> tuple[BenchmarkResult, list[float]]:
    from sglang.srt.compilation.piecewise_context_manager import enable_piecewise_cuda_graph
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmRunnerCore

    intermediate = preset.moe_intermediate_size // tp_size
    inputs = make_runner_inputs(
        preset=preset,
        batch_size=batch_size,
        active_experts=active_experts,
        distribution=distribution,
        skew_minor_fraction=skew_minor_fraction,
        tp_size=tp_size,
        device=device,
        alignment=alignment,
    )
    runner = DeepGemmRunnerCore(
        MoeRunnerConfig(
            num_experts=preset.num_experts,
            num_local_experts=active_experts,
            hidden_size=preset.hidden_size,
            intermediate_size_per_partition=intermediate,
            top_k=1,
            params_dtype=torch.bfloat16,
            activation="silu",
            is_gated=True,
            inplace=False,
        )
    )

    def run_once() -> None:
        with enable_piecewise_cuda_graph():
            runner.run(inputs.runner_input, inputs.quant_info, inputs.running_state)

    latencies_us = run_timed(
        run_once,
        warmup_iters,
        timed_warmup_iters,
        measure_iters,
        use_cuda_graph=use_cuda_graph,
        cuda_graph_inner_iters=cuda_graph_inner_iters,
    )
    result = summarize_result(
        preset=preset,
        backend="deep_gemm_runner",
        op="mlp",
        batch_size=batch_size,
        active_experts=active_experts,
        distribution=distribution,
        skew_minor_fraction=skew_minor_fraction,
        tp_size=tp_size,
        shape=GemmShape(op="mlp", n=(3 * intermediate) + preset.hidden_size, k=0),
        logical_tokens=inputs.logical_tokens,
        padded_tokens=inputs.padded_tokens,
        latencies_us=latencies_us,
        warmup_iters=warmup_iters,
        timed_warmup_iters=timed_warmup_iters,
        measure_iters=measure_iters,
        use_cuda_graph=use_cuda_graph,
        cuda_graph_inner_iters=cuda_graph_inner_iters,
    )
    return result, latencies_us


def benchmark_single_gemm(
    *,
    deep_gemm_wrapper: object,
    preset: ModelPreset,
    op: str,
    batch_size: int,
    active_experts: int,
    distribution: str,
    skew_minor_fraction: float,
    tp_size: int,
    device: torch.device,
    alignment: int,
    warmup_iters: int,
    timed_warmup_iters: int,
    measure_iters: int,
    use_cuda_graph: bool,
    cuda_graph_inner_iters: int,
) -> tuple[BenchmarkResult, list[float]]:
    shape = shape_for_op(preset, tp_size, op)
    inputs = make_inputs(
        shape=shape,
        batch_size=batch_size,
        active_experts=active_experts,
        distribution=distribution,
        skew_minor_fraction=skew_minor_fraction,
        device=device,
        alignment=alignment,
    )

    def run_once() -> None:
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            inputs.lhs, inputs.rhs, inputs.out, inputs.m_indices
        )

    latencies_us = run_timed(
        run_once,
        warmup_iters,
        timed_warmup_iters,
        measure_iters,
        use_cuda_graph=use_cuda_graph,
        cuda_graph_inner_iters=cuda_graph_inner_iters,
    )
    result = summarize_result(
        preset=preset,
        backend="deep_gemm_wrapper",
        op=op,
        batch_size=batch_size,
        active_experts=active_experts,
        distribution=distribution,
        skew_minor_fraction=skew_minor_fraction,
        tp_size=tp_size,
        shape=shape,
        logical_tokens=inputs.logical_tokens,
        padded_tokens=inputs.padded_tokens,
        latencies_us=latencies_us,
        warmup_iters=warmup_iters,
        timed_warmup_iters=timed_warmup_iters,
        measure_iters=measure_iters,
        use_cuda_graph=use_cuda_graph,
        cuda_graph_inner_iters=cuda_graph_inner_iters,
    )
    return result, latencies_us


def benchmark_mlp(
    *,
    deep_gemm_wrapper: object,
    preset: ModelPreset,
    batch_size: int,
    active_experts: int,
    distribution: str,
    skew_minor_fraction: float,
    tp_size: int,
    device: torch.device,
    alignment: int,
    warmup_iters: int,
    timed_warmup_iters: int,
    measure_iters: int,
    use_cuda_graph: bool,
    cuda_graph_inner_iters: int,
) -> tuple[BenchmarkResult, list[float]]:
    gateup_shape = shape_for_op(preset, tp_size, "gateup")
    down_shape = shape_for_op(preset, tp_size, "down")
    gateup_inputs = make_inputs(
        shape=gateup_shape,
        batch_size=batch_size,
        active_experts=active_experts,
        distribution=distribution,
        skew_minor_fraction=skew_minor_fraction,
        device=device,
        alignment=alignment,
    )
    down_rhs_bf16 = torch.randn(
        (active_experts, down_shape.n, down_shape.k), device=device, dtype=torch.bfloat16
    )
    down_rhs_data = torch.empty_like(down_rhs_bf16, dtype=torch.float8_e4m3fn)
    down_rhs_scales = torch.empty(
        (
            active_experts,
            ceil_div(down_shape.n, FP8_BLOCK_SIZE),
            ceil_div(down_shape.k, FP8_BLOCK_SIZE),
        ),
        device=device,
        dtype=torch.float32,
    )
    for expert_id in range(active_experts):
        down_rhs_data[expert_id], down_rhs_scales[expert_id] = per_block_cast_to_fp8(
            down_rhs_bf16[expert_id]
        )
    down_out = torch.empty(
        (gateup_inputs.padded_tokens, down_shape.n), device=device, dtype=torch.bfloat16
    )

    def run_once() -> None:
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            gateup_inputs.lhs, gateup_inputs.rhs, gateup_inputs.out, gateup_inputs.m_indices
        )
        gate, up = gateup_inputs.out.chunk(2, dim=1)
        down_input_bf16 = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        down_lhs = per_token_cast_to_fp8(down_input_bf16)
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            down_lhs,
            (down_rhs_data, down_rhs_scales),
            down_out,
            gateup_inputs.m_indices,
        )

    latencies_us = run_timed(
        run_once,
        warmup_iters,
        timed_warmup_iters,
        measure_iters,
        use_cuda_graph=use_cuda_graph,
        cuda_graph_inner_iters=cuda_graph_inner_iters,
    )
    result = summarize_result(
        preset=preset,
        backend="deep_gemm_wrapper",
        op="mlp",
        batch_size=batch_size,
        active_experts=active_experts,
        distribution=distribution,
        skew_minor_fraction=skew_minor_fraction,
        tp_size=tp_size,
        shape=GemmShape(op="mlp", n=gateup_shape.n + down_shape.n, k=0),
        logical_tokens=gateup_inputs.logical_tokens,
        padded_tokens=gateup_inputs.padded_tokens,
        latencies_us=latencies_us,
        warmup_iters=warmup_iters,
        timed_warmup_iters=timed_warmup_iters,
        measure_iters=measure_iters,
        use_cuda_graph=use_cuda_graph,
        cuda_graph_inner_iters=cuda_graph_inner_iters,
    )
    return result, latencies_us


def create_run_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir
    run_dir = RESULTS_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def collect_env(device: torch.device, deep_gemm_wrapper: object) -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for module_name in ["sglang", "torch", "deep_gemm", "triton", "flashinfer"]:
        try:
            module = __import__(module_name)
            packages[module_name] = getattr(module, "__version__", None)
        except Exception:
            packages[module_name] = None
    index = device.index or 0
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(index),
        "cuda_device_capability": list(torch.cuda.get_device_capability(index)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": packages,
        "deepgemm": {
            "ENABLE_JIT_DEEPGEMM": bool(deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM),
            "DEEPGEMM_BLACKWELL": bool(deep_gemm_wrapper.DEEPGEMM_BLACKWELL),
            "DEEPGEMM_SCALE_UE8M0": bool(deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0),
            "DEEPGEMM_NEED_TMA_ALIGNED_SCALES": bool(
                deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES
            ),
        },
    }


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_metrics_csv(path: Path, results: list[BenchmarkResult]) -> None:
    if not results:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_raw_latencies(path: Path, raw_rows: list[dict[str, object]]) -> None:
    if not raw_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_preset",
                "backend",
                "op",
                "batch_size",
                "iteration",
                "latency_us",
                "use_cuda_graph",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)


def write_readme(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    env: dict[str, object],
    status: str,
    started_at: str,
    completed_at: str | None,
    failure: str | None = None,
) -> None:
    artifacts = [
        "README.md",
        "config.json",
        "env.json",
        "metrics.json",
        "metrics.csv",
        "raw_latencies.csv",
    ]
    lines = [
        "# DeepGEMM MoE Benchmark Run",
        "",
        f"- Status: `{status}`",
        f"- Started At: `{started_at}`",
        f"- Completed At: `{completed_at or 'n/a'}`",
        f"- Benchmark: `{BENCHMARK_NAME}`",
        f"- Command: `{' '.join(sys.argv)}`",
        "",
        "## Parameters",
        "",
        f"- Model preset: `{args.model_preset}`",
        f"- Backend: `{args.backend}`",
        f"- Operation: `{args.op}`",
        f"- Batch sizes: `{args.batch_sizes}`",
        f"- Active experts: `{args.active_experts}`",
        f"- Distribution: `{args.distribution}`",
        f"- Skew minor fraction: `{args.skew_minor_fraction}`",
        f"- TP size for shape derivation: `{args.tp_size}`",
        f"- Warmup iterations: `{args.warmup_iters}`",
        f"- Timed callable warmup iterations: `{args.timed_warmup_iters}`",
        f"- Measure iterations: `{args.measure_iters}`",
        f"- CUDA Graph: `{args.use_cuda_graph}`",
        f"- CUDA Graph inner iterations: `{args.cuda_graph_inner_iters}`",
        "",
        "## Environment",
        "",
        f"- CUDA device: `{env.get('cuda_device')}`",
        f"- GPU: `{env.get('cuda_device_name')}`",
        f"- Compute capability: `{env.get('cuda_device_capability')}`",
        f"- Torch: `{env.get('torch_version')}`",
        f"- Torch CUDA: `{env.get('torch_cuda')}`",
        "",
        "## Artifacts",
        "",
    ]
    for artifact in artifacts:
        lines.append(f"- `{artifact}`")
    if failure is not None:
        lines.extend(["", "## Failure", "", "```text", failure.rstrip(), "```"])
    lines.append("")
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESETS),
        default="deepseek-v3",
        help="MoE model shape preset.",
    )
    parser.add_argument(
        "--backend",
        choices=["deep_gemm_runner", "deep_gemm_wrapper"],
        default="deep_gemm_runner",
        help="Backend to benchmark. The default uses SGLang DeepGemmRunnerCore.",
    )
    parser.add_argument(
        "--op",
        choices=["gateup", "down", "mlp"],
        default="mlp",
        help="Expert compute to benchmark. deep_gemm_runner supports mlp only.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_BATCH_SIZES,
        help="Logical input batch sizes to sweep.",
    )
    parser.add_argument("--active-experts", type=int, default=1)
    parser.add_argument("--distribution", choices=["single", "even", "uniform", "skewed"], default="single")
    parser.add_argument(
        "--skew-minor-fraction",
        type=float,
        default=0.0625,
        help="For --distribution skewed, target total token fraction assigned to non-major experts.",
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Shape-only tensor parallel divisor.")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument(
        "--timed-warmup-iters",
        type=int,
        default=5,
        help="Warmups of the exact timed callable after CUDA Graph capture, per batch size.",
    )
    parser.add_argument("--measure-iters", type=int, default=100)
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--cuda-graph-inner-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-raw-latencies", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("All batch sizes must be positive.")
    if args.active_experts <= 0:
        raise ValueError("--active-experts must be positive.")
    preset = MODEL_PRESETS[args.model_preset]
    if args.active_experts > preset.num_experts:
        raise ValueError("--active-experts cannot exceed the preset's num_experts.")
    if args.tp_size <= 0:
        raise ValueError("--tp-size must be positive.")
    if not 0.0 < args.skew_minor_fraction < 1.0:
        raise ValueError("--skew-minor-fraction must be in (0, 1).")
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be non-negative.")
    if args.timed_warmup_iters < 0:
        raise ValueError("--timed-warmup-iters must be non-negative.")
    if args.measure_iters <= 0:
        raise ValueError("--measure-iters must be positive.")
    if args.cuda_graph_inner_iters <= 0:
        raise ValueError("--cuda-graph-inner-iters must be positive.")
    if args.backend == "deep_gemm_runner" and args.op != "mlp":
        raise ValueError("--backend deep_gemm_runner supports --op mlp only.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    started_at = datetime.now().isoformat(timespec="seconds")
    run_dir = create_run_dir(args.output_dir)

    try:
        device = resolve_device(args.device)
        deep_gemm_wrapper, get_alignment = require_deepgemm()
        alignment = int(get_alignment())
        torch.manual_seed(args.seed)
        preset = MODEL_PRESETS[args.model_preset]
        env = collect_env(device, deep_gemm_wrapper)
        config = {
            "benchmark_name": BENCHMARK_NAME,
            "started_at": started_at,
            "model_preset": asdict(preset),
            "args": vars(args) | {"output_dir": str(run_dir)},
            "deepgemm_alignment": alignment,
            "result_dir": str(run_dir),
        }
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "env.json", env)
        write_readme(
            run_dir,
            args=args,
            env=env,
            status="running",
            started_at=started_at,
            completed_at=None,
        )

        results: list[BenchmarkResult] = []
        raw_rows: list[dict[str, object]] = []
        for batch_size in args.batch_sizes:
            print(
                f"Running {args.model_preset} {args.backend} {args.op} batch_size={batch_size}",
                flush=True,
            )
            if args.backend == "deep_gemm_runner":
                result, latencies = benchmark_runner_mlp(
                    preset=preset,
                    batch_size=batch_size,
                    active_experts=args.active_experts,
                    distribution=args.distribution,
                    skew_minor_fraction=args.skew_minor_fraction,
                    tp_size=args.tp_size,
                    device=device,
                    alignment=alignment,
                    warmup_iters=args.warmup_iters,
                    timed_warmup_iters=args.timed_warmup_iters,
                    measure_iters=args.measure_iters,
                    use_cuda_graph=args.use_cuda_graph,
                    cuda_graph_inner_iters=args.cuda_graph_inner_iters,
                )
            elif args.op == "mlp":
                result, latencies = benchmark_mlp(
                    deep_gemm_wrapper=deep_gemm_wrapper,
                    preset=preset,
                    batch_size=batch_size,
                    active_experts=args.active_experts,
                    distribution=args.distribution,
                    skew_minor_fraction=args.skew_minor_fraction,
                    tp_size=args.tp_size,
                    device=device,
                    alignment=alignment,
                    warmup_iters=args.warmup_iters,
                    timed_warmup_iters=args.timed_warmup_iters,
                    measure_iters=args.measure_iters,
                    use_cuda_graph=args.use_cuda_graph,
                    cuda_graph_inner_iters=args.cuda_graph_inner_iters,
                )
            else:
                result, latencies = benchmark_single_gemm(
                    deep_gemm_wrapper=deep_gemm_wrapper,
                    preset=preset,
                    op=args.op,
                    batch_size=batch_size,
                    active_experts=args.active_experts,
                    distribution=args.distribution,
                    skew_minor_fraction=args.skew_minor_fraction,
                    tp_size=args.tp_size,
                    device=device,
                    alignment=alignment,
                    warmup_iters=args.warmup_iters,
                    timed_warmup_iters=args.timed_warmup_iters,
                    measure_iters=args.measure_iters,
                    use_cuda_graph=args.use_cuda_graph,
                    cuda_graph_inner_iters=args.cuda_graph_inner_iters,
                )
            results.append(result)
            if not args.no_raw_latencies:
                for iteration, latency_us in enumerate(latencies):
                    raw_rows.append(
                        {
                            "model_preset": args.model_preset,
                            "backend": args.backend,
                            "op": args.op,
                            "batch_size": batch_size,
                            "iteration": iteration,
                            "latency_us": latency_us,
                            "use_cuda_graph": args.use_cuda_graph,
                        }
                    )
            print(
                "  latency_mean_us="
                f"{result.latency_us_mean:.2f} throughput_tokens_s={result.throughput_tokens_per_s:.2f} "
                f"effective_tflops={result.effective_tflops:.2f}",
                flush=True,
            )

        write_json(run_dir / "metrics.json", [asdict(result) for result in results])
        write_metrics_csv(run_dir / "metrics.csv", results)
        if not args.no_raw_latencies:
            write_raw_latencies(run_dir / "raw_latencies.csv", raw_rows)
        completed_at = datetime.now().isoformat(timespec="seconds")
        write_readme(
            run_dir,
            args=args,
            env=env,
            status="success",
            started_at=started_at,
            completed_at=completed_at,
        )
        print(f"Results written to {run_dir}")
        return 0
    except Exception as exc:
        completed_at = datetime.now().isoformat(timespec="seconds")
        failure = f"{type(exc).__name__}: {exc}"
        fallback_env = {"cuda_device": args.device, "cuda_device_name": "unknown"}
        write_readme(
            run_dir,
            args=args,
            env=fallback_env,
            status="failure",
            started_at=started_at,
            completed_at=completed_at,
            failure=failure,
        )
        print(f"Benchmark failed; failure recorded in {run_dir / 'README.md'}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
