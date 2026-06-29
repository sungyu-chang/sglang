#!/usr/bin/env python3
"""Measure fused MoE and single GEMM latency versus batch size.

This benchmark uses synthetic hidden states, synthetic expert weights, and
synthetic routing ids. It is intended to make padding-related latency jumps
visible by sweeping token batch sizes while keeping model dimensions fixed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "python" / "sglang").is_dir()
)
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
os.environ.setdefault("SGLANG_OPT_USE_JIT_EP_ACTIVATION", "0")

import torch
import triton
import triton.language as tl

import sglang.srt.layers.moe.moe_runner.deep_gemm as deep_gemm_runner_module
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.ep_moe.kernels import moe_ep_deepgemm_preprocess
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    _prepare_fused_moe_run,
    fused_moe,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_config_dtype_str,
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

# The DeepGEMM runner normally disposes intermediate tensors, including the
# dispatch input. This benchmark reuses the same synthetic input across timing
# iterations, so disposal must not release it.
deep_gemm_runner_module.dispose_tensor = lambda tensor: None


@dataclass(frozen=True)
class ModelShape:
    name: str
    hidden_size: int
    moe_intermediate_size: int

    @property
    def fused_w1_n(self) -> int:
        return 2 * self.moe_intermediate_size


@dataclass(frozen=True)
class WeightBundle:
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None


@dataclass(frozen=True)
class SingleGemmWeightBundle:
    weight: torch.Tensor
    weight_scale: torch.Tensor | None = None


@dataclass(frozen=True)
class ResultRow:
    benchmark: str
    model: str
    backend: str
    batch_size: int
    gemm: str
    gemm_m: int
    gemm_n: int
    gemm_k: int
    route_mode: str
    num_experts: int
    route_counts: str
    triton_block_size_m: int
    triton_tokens_post_padded_est: int
    deepgemm_m_max: int
    deepgemm_expected_m: int
    warmups: int
    iters: int
    cuda_graph: bool
    latency_ms: float


MODEL_SHAPES: Dict[str, ModelShape] = {
    "deepseek-v3": ModelShape(
        name="deepseek-v3",
        hidden_size=7168,
        moe_intermediate_size=2048,
    ),
    "qwen3-235b-a22b": ModelShape(
        name="qwen3-235b-a22b",
        hidden_size=4096,
        moe_intermediate_size=1536,
    ),
    "qwen3-30b-a3b": ModelShape(
        name="qwen3-30b-a3b",
        hidden_size=2048,
        moe_intermediate_size=768,
    ),
}


def parse_int_list(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    try:
        return mapping[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {dtype_name!r}") from exc


def selected_models(names: Iterable[str]) -> List[ModelShape]:
    models = []
    for name in names:
        key = name.lower()
        if key not in MODEL_SHAPES:
            raise ValueError(
                f"unknown model preset {name!r}; choices: {', '.join(MODEL_SHAPES)}"
            )
        models.append(MODEL_SHAPES[key])
    return models


def quantize_deepgemm_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    qweights = []
    scales = []
    for expert_id in range(weight.shape[0]):
        qweight, scale = per_block_cast_to_fp8(weight[expert_id].contiguous())
        qweights.append(qweight)
        scales.append(scale)
    return torch.stack(qweights, dim=0), torch.stack(scales, dim=0)


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n
    ), (x_amax / 448.0).view(m, -1)


def make_weights(
    model: ModelShape,
    num_experts: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    backend: str,
) -> WeightBundle:
    w1_bf16 = torch.randn(
        (num_experts, model.fused_w1_n, model.hidden_size),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    w2_bf16 = torch.randn(
        (num_experts, model.hidden_size, model.moe_intermediate_size),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if backend == "deep_gemm":
        w1, w1_scale = quantize_deepgemm_weight(w1_bf16)
        w2, w2_scale = quantize_deepgemm_weight(w2_bf16)
        del w1_bf16, w2_bf16
        return WeightBundle(w1=w1, w2=w2, w1_scale=w1_scale, w2_scale=w2_scale)
    return WeightBundle(w1=w1_bf16, w2=w2_bf16)


def single_gemm_shape(model: ModelShape, gemm: str) -> Tuple[int, int]:
    if gemm == "up":
        return model.fused_w1_n, model.hidden_size
    if gemm == "down":
        return model.hidden_size, model.moe_intermediate_size
    raise ValueError(f"unknown single GEMM kind {gemm!r}")


def fused_moe_gemm_shape(model: ModelShape, gemm: str) -> Tuple[int, int]:
    return single_gemm_shape(model, gemm)


def make_single_gemm_weight(
    *,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    backend: str,
) -> SingleGemmWeightBundle:
    weight = torch.randn((n, k), device=device, dtype=dtype, generator=generator)
    if backend == "deep_gemm":
        weight_fp8, weight_scale = per_block_cast_to_fp8(weight)
        del weight
        return SingleGemmWeightBundle(weight=weight_fp8, weight_scale=weight_scale)
    return SingleGemmWeightBundle(weight=weight)


def prepare_deepgemm_scale_for_masked_gemm(scale: torch.Tensor) -> torch.Tensor:
    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        if scale.dtype != torch.int:
            return deep_gemm_runner_module._cast_to_e8m0_with_rounding_up(scale)
        return scale
    if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
        return deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(scale)
    return scale


def make_topk_ids(batch_size: int, num_experts: int, mode: str, device: torch.device):
    if mode == "single":
        ids = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    elif mode == "round_robin":
        ids = torch.arange(batch_size, dtype=torch.int64, device=device) % num_experts
    elif mode == "two_expert_split":
        ids = torch.zeros((batch_size,), dtype=torch.int64, device=device)
        ids[batch_size // 2 :] = 1 if num_experts > 1 else 0
    else:
        raise ValueError(f"unknown route mode {mode!r}")
    return ids.view(-1, 1)


def route_counts(topk_ids: torch.Tensor, num_experts: int) -> List[int]:
    counts = torch.bincount(topk_ids.view(-1), minlength=num_experts)
    return [int(x) for x in counts.tolist()]


def make_topk_output(
    topk_ids: torch.Tensor, num_experts: int, device: torch.device
) -> StandardTopKOutput:
    batch_size = topk_ids.shape[0]
    topk_weights = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
    router_logits = torch.zeros(
        (batch_size, num_experts), dtype=torch.float32, device=device
    )
    return StandardTopKOutput(topk_weights, topk_ids, router_logits)


def ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def estimate_triton_padding(
    model: ModelShape,
    weights: WeightBundle,
    topk_ids: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[int, int]:
    dtype_str = get_config_dtype_str(
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        dtype=dtype,
    )
    config, _ = try_get_optimal_moe_config(
        weights.w1.shape,
        weights.w2.shape,
        top_k=1,
        dtype=dtype_str,
        M=topk_ids.shape[0],
        block_shape=None,
        return_down_config=True,
    )
    block_m = int(config["BLOCK_SIZE_M"])
    counts = route_counts(topk_ids, weights.w1.shape[0])
    tokens_post_padded = sum(
        ceil_to_multiple(count, block_m) for count in counts if count > 0
    )
    return block_m, tokens_post_padded


def estimate_deepgemm_padding(batch_size: int, num_experts: int) -> Tuple[int, int]:
    m_max = (batch_size // 256 + 1) * 256
    expected_m = (batch_size - 1) // num_experts + 1
    return m_max, expected_m


def run_triton(
    hidden_states: torch.Tensor,
    topk_output: StandardTopKOutput,
    weights: WeightBundle,
) -> torch.Tensor:
    return fused_moe(
        hidden_states,
        weights.w1,
        weights.w2,
        topk_output,
        moe_runner_config=MoeRunnerConfig(
            num_experts=weights.w1.shape[0],
            num_local_experts=weights.w1.shape[0],
            top_k=1,
            inplace=False,
        ),
    )


def run_deep_gemm(
    hidden_states: torch.Tensor,
    topk_output: StandardTopKOutput,
    weights: WeightBundle,
    runner: MoeRunner,
) -> torch.Tensor:
    assert weights.w1_scale is not None and weights.w2_scale is not None
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=weights.w1,
        w2_weight=weights.w2,
        use_fp8=True,
        w13_scale=weights.w1_scale,
        w2_scale=weights.w2_scale,
        block_shape=[128, 128],
    )
    dispatch_output = StandardDispatchOutput(hidden_states, None, topk_output)
    return runner.run(dispatch_output, quant_info).hidden_states


@triton.jit
def _single_gemm_bf16_nt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_mask = k_start + offs_k < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run_single_gemm_triton_bf16(
    activations: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    m, k = activations.shape
    n, weight_k = weight.shape
    assert k == weight_k
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 64))
    _single_gemm_bf16_nt_kernel[grid](
        activations,
        weight,
        output,
        m,
        n,
        k,
        activations.stride(0),
        activations.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=32,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
        num_stages=4,
    )
    return output


def run_single_gemm_deep_gemm(
    activations: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: SingleGemmWeightBundle,
    output: torch.Tensor,
) -> torch.Tensor:
    assert weight.weight_scale is not None
    deep_gemm_wrapper.gemm_nt_f8f8bf16(
        (activations, activation_scale),
        (weight.weight, weight.weight_scale),
        output,
    )
    return output


def make_triton_fused_moe_metadata(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    weights: WeightBundle,
):
    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        weights.w1,
        weights.w2,
        topk_ids,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
    )
    return (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )


def triton_fused_moe_total_tokens(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    weights: WeightBundle,
    config: Dict,
    down_moe_use_tma: bool,
) -> int:
    num_tokens = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    padded_tokens = (
        min(num_tokens * topk, weights.w1.shape[0] + 1) * (config["BLOCK_SIZE_M"] - 1)
        if down_moe_use_tma
        else 0
    )
    return num_tokens * topk + padded_tokens


def run_triton_fused_moe_gemm(
    *,
    gemm: str,
    hidden_states: torch.Tensor,
    topk_output: StandardTopKOutput,
    weights: WeightBundle,
    config: Dict,
    down_config: Dict | None,
    down_moe_use_tma: bool,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    intermediate_cache2: torch.Tensor | None,
    output: torch.Tensor,
) -> torch.Tensor:
    topk_weights, topk_ids, _ = topk_output
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
    if gemm == "up":
        invoke_fused_moe_kernel(
            hidden_states,
            weights.w1,
            None,
            output,
            None,
            None,
            None,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            topk_ids.shape[1],
            config,
            compute_type,
            False,
            False,
            False,
            False,
            False,
            filter_expert=False,
        )
        return output

    if gemm != "down":
        raise ValueError(f"unknown fused MoE GEMM kind {gemm!r}")
    assert intermediate_cache2 is not None
    invoke_fused_moe_kernel(
        intermediate_cache2,
        weights.w2,
        None,
        output.unsqueeze(0),
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        1,
        down_config or config,
        compute_type,
        False,
        False,
        False,
        False,
        False,
        a_use_tma=down_moe_use_tma,
        b_use_tma=down_moe_use_tma,
        filter_expert=False,
        router_topk=topk_ids.shape[1],
    )
    return output


def run_deepgemm_fused_moe_gemm(
    *,
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
) -> torch.Tensor:
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
        lhs,
        rhs,
        output,
        masked_m,
        expected_m,
    )
    return output


def time_call(
    fn,
    *,
    warmups: int,
    iters: int,
    device: torch.device,
    use_cuda_graph: bool,
) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize(device)

    graph = None
    if use_cuda_graph:
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream(device).wait_stream(side_stream)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        if graph is None:
            fn()
        else:
            graph.replay()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / iters


def benchmark_one(
    *,
    model: ModelShape,
    backend: str,
    weights: WeightBundle,
    batch_size: int,
    route_mode: str,
    num_experts: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    warmups: int,
    iters: int,
    cuda_graph: bool,
) -> ResultRow:
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden_states = torch.randn(
        (batch_size, model.hidden_size),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    topk_ids = make_topk_ids(batch_size, num_experts, route_mode, device)
    topk_output = make_topk_output(topk_ids, num_experts, device)
    counts = route_counts(topk_ids, num_experts)
    triton_block_m, triton_tokens_post_padded = estimate_triton_padding(
        model, weights, topk_ids, dtype
    )
    deepgemm_m_max, deepgemm_expected_m = estimate_deepgemm_padding(
        batch_size, num_experts
    )

    if backend == "triton":
        fn = lambda: run_triton(hidden_states, topk_output, weights)
        runner = None
    elif backend == "deep_gemm":
        runner = MoeRunner(
            MoeRunnerBackend.DEEP_GEMM,
            MoeRunnerConfig(
                num_experts=num_experts,
                num_local_experts=num_experts,
                hidden_size=model.hidden_size,
                intermediate_size_per_partition=model.moe_intermediate_size,
                top_k=1,
                params_dtype=dtype,
                inplace=False,
            ),
        )
        fn = lambda: run_deep_gemm(hidden_states, topk_output, weights, runner)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    latency_ms = time_call(
        fn,
        warmups=warmups,
        iters=iters,
        device=device,
        use_cuda_graph=cuda_graph,
    )

    del hidden_states, topk_output, topk_ids, runner
    torch.cuda.empty_cache()
    gc.collect()

    return ResultRow(
        benchmark="fused_moe",
        model=model.name,
        backend=backend,
        batch_size=batch_size,
        gemm="moe",
        gemm_m=batch_size,
        gemm_n=0,
        gemm_k=0,
        route_mode=route_mode,
        num_experts=num_experts,
        route_counts=",".join(str(count) for count in counts),
        triton_block_size_m=triton_block_m,
        triton_tokens_post_padded_est=triton_tokens_post_padded,
        deepgemm_m_max=deepgemm_m_max,
        deepgemm_expected_m=deepgemm_expected_m,
        warmups=warmups,
        iters=iters,
        cuda_graph=cuda_graph,
        latency_ms=latency_ms,
    )


def benchmark_single_gemm_one(
    *,
    model: ModelShape,
    backend: str,
    gemm: str,
    weight: SingleGemmWeightBundle,
    batch_size: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    warmups: int,
    iters: int,
    cuda_graph: bool,
) -> ResultRow:
    generator = torch.Generator(device=device).manual_seed(seed)
    output_dtype = torch.bfloat16 if backend == "deep_gemm" else dtype
    output = torch.empty((batch_size, n), device=device, dtype=output_dtype)

    if backend == "triton":
        activations = torch.randn(
            (batch_size, k),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        activation_scale = None
        fn = lambda: run_single_gemm_triton_bf16(activations, weight.weight, output)
    elif backend == "deep_gemm":
        if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            raise RuntimeError(
                "single_gemm backend=deep_gemm requires the deep_gemm package"
            )
        activations_bf16 = torch.randn(
            (batch_size, k),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        activations, activation_scale = per_token_cast_to_fp8(activations_bf16)
        del activations_bf16
        fn = lambda: run_single_gemm_deep_gemm(
            activations, activation_scale, weight, output
        )
    else:
        raise ValueError(f"unknown backend {backend!r}")

    latency_ms = time_call(
        fn,
        warmups=warmups,
        iters=iters,
        device=device,
        use_cuda_graph=cuda_graph,
    )

    del activations, activation_scale, output
    torch.cuda.empty_cache()
    gc.collect()

    return ResultRow(
        benchmark="single_gemm",
        model=model.name,
        backend=backend,
        batch_size=batch_size,
        gemm=gemm,
        gemm_m=batch_size,
        gemm_n=n,
        gemm_k=k,
        route_mode="",
        num_experts=0,
        route_counts="",
        triton_block_size_m=0,
        triton_tokens_post_padded_est=0,
        deepgemm_m_max=0,
        deepgemm_expected_m=0,
        warmups=warmups,
        iters=iters,
        cuda_graph=cuda_graph,
        latency_ms=latency_ms,
    )


def benchmark_fused_moe_gemm_one(
    *,
    model: ModelShape,
    backend: str,
    gemm: str,
    weights: WeightBundle,
    batch_size: int,
    route_mode: str,
    num_experts: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    warmups: int,
    iters: int,
    cuda_graph: bool,
) -> ResultRow:
    n, k = fused_moe_gemm_shape(model, gemm)
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden_states = torch.randn(
        (batch_size, model.hidden_size),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    topk_ids = make_topk_ids(batch_size, num_experts, route_mode, device)
    topk_output = make_topk_output(topk_ids, num_experts, device)
    counts = route_counts(topk_ids, num_experts)
    triton_block_m, triton_tokens_post_padded = estimate_triton_padding(
        model, weights, topk_ids, dtype
    )
    deepgemm_m_max, deepgemm_expected_m = estimate_deepgemm_padding(
        batch_size, num_experts
    )

    if backend == "triton":
        (
            config,
            down_config,
            down_moe_use_tma,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
        ) = make_triton_fused_moe_metadata(hidden_states, topk_ids, weights)
        total_tokens = triton_fused_moe_total_tokens(
            hidden_states, topk_ids, weights, config, down_moe_use_tma
        )
        if gemm == "up":
            output = torch.empty(
                (total_tokens, model.fused_w1_n), device=device, dtype=dtype
            )
            intermediate_cache2 = None
        elif gemm == "down":
            output = torch.empty(
                (batch_size, model.hidden_size), device=device, dtype=dtype
            )
            intermediate_cache2 = torch.randn(
                (total_tokens, model.moe_intermediate_size),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        else:
            raise ValueError(f"unknown fused MoE GEMM kind {gemm!r}")

        fn = lambda: run_triton_fused_moe_gemm(
            gemm=gemm,
            hidden_states=hidden_states,
            topk_output=topk_output,
            weights=weights,
            config=config,
            down_config=down_config,
            down_moe_use_tma=down_moe_use_tma,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            intermediate_cache2=intermediate_cache2,
            output=output,
        )
    elif backend == "deep_gemm":
        if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            raise RuntimeError(
                "fused_moe_gemm backend=deep_gemm requires the deep_gemm package"
            )
        assert weights.w1_scale is not None and weights.w2_scale is not None
        masked_m, expected_m, _, gateup_input, gateup_input_scale = (
            moe_ep_deepgemm_preprocess(
                topk_ids,
                num_experts,
                hidden_states,
                1,
                [128, 128],
            )
        )
        del hidden_states

        if gemm == "up":
            lhs_data = gateup_input
            lhs_scale = prepare_deepgemm_scale_for_masked_gemm(gateup_input_scale)
            rhs = (weights.w1, weights.w1_scale)
            output = torch.empty(
                (num_experts, gateup_input.shape[1], model.fused_w1_n),
                device=device,
                dtype=torch.bfloat16,
            )
        elif gemm == "down":
            del gateup_input, gateup_input_scale
            gateup_output = torch.randn(
                (num_experts, deepgemm_m_max, model.fused_w1_n),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            lhs_data, lhs_scale = (
                deep_gemm_runner_module._varlen_deep_gemm_silu_mul_quant(
                    gateup_output,
                    masked_m,
                    group_size=128,
                    topk=1,
                )
            )
            del gateup_output
            if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
                lhs_scale = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(lhs_scale)
            rhs = (weights.w2, weights.w2_scale)
            output = torch.empty(
                (num_experts, deepgemm_m_max, model.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
        else:
            raise ValueError(f"unknown fused MoE GEMM kind {gemm!r}")

        fn = lambda: run_deepgemm_fused_moe_gemm(
            lhs=(lhs_data, lhs_scale),
            rhs=rhs,
            output=output,
            masked_m=masked_m,
            expected_m=expected_m,
        )
    else:
        raise ValueError(f"unknown backend {backend!r}")

    latency_ms = time_call(
        fn,
        warmups=warmups,
        iters=iters,
        device=device,
        use_cuda_graph=cuda_graph,
    )

    del topk_output, topk_ids, output
    torch.cuda.empty_cache()
    gc.collect()

    return ResultRow(
        benchmark="fused_moe_gemm",
        model=model.name,
        backend=backend,
        batch_size=batch_size,
        gemm=gemm,
        gemm_m=batch_size,
        gemm_n=n,
        gemm_k=k,
        route_mode=route_mode,
        num_experts=num_experts,
        route_counts=",".join(str(count) for count in counts),
        triton_block_size_m=triton_block_m,
        triton_tokens_post_padded_est=triton_tokens_post_padded,
        deepgemm_m_max=deepgemm_m_max,
        deepgemm_expected_m=deepgemm_expected_m,
        warmups=warmups,
        iters=iters,
        cuda_graph=cuda_graph,
        latency_ms=latency_ms,
    )


def write_csv(path: Path, rows: Sequence[ResultRow]):
    fieldnames = list(ResultRow.__dataclass_fields__)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def csv_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes"}


def load_existing_rows(path: Path) -> List[ResultRow]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(ResultRow.__dataclass_fields__):
            raise ValueError(
                f"cannot resume from {path}: unexpected CSV columns {reader.fieldnames}"
            )
        rows = []
        for row in reader:
            rows.append(
                ResultRow(
                    benchmark=row["benchmark"],
                    model=row["model"],
                    backend=row["backend"],
                    batch_size=int(row["batch_size"]),
                    gemm=row["gemm"],
                    gemm_m=int(row["gemm_m"]),
                    gemm_n=int(row["gemm_n"]),
                    gemm_k=int(row["gemm_k"]),
                    route_mode=row["route_mode"],
                    num_experts=int(row["num_experts"]),
                    route_counts=row["route_counts"],
                    triton_block_size_m=int(row["triton_block_size_m"]),
                    triton_tokens_post_padded_est=int(
                        row["triton_tokens_post_padded_est"]
                    ),
                    deepgemm_m_max=int(row["deepgemm_m_max"]),
                    deepgemm_expected_m=int(row["deepgemm_expected_m"]),
                    warmups=int(row["warmups"]),
                    iters=int(row["iters"]),
                    cuda_graph=csv_bool(row["cuda_graph"]),
                    latency_ms=float(row["latency_ms"]),
                )
            )
    return rows


def result_key(row: ResultRow) -> Tuple[object, ...]:
    return (
        row.benchmark,
        row.model,
        row.backend,
        row.batch_size,
        row.gemm,
        row.route_mode,
        row.num_experts,
        row.warmups,
        row.iters,
        row.cuda_graph,
    )


def candidate_key(
    *,
    benchmark: str,
    model: str,
    backend: str,
    batch_size: int,
    gemm: str,
    route_mode: str,
    num_experts: int,
    warmups: int,
    iters: int,
    cuda_graph: bool,
) -> Tuple[object, ...]:
    return (
        benchmark,
        model,
        backend,
        batch_size,
        gemm,
        route_mode,
        num_experts,
        warmups,
        iters,
        cuda_graph,
    )


def markdown_table(rows: Sequence[ResultRow]) -> str:
    headers = [
        "Benchmark",
        "Model",
        "Backend",
        "Batch",
        "GEMM",
        "M",
        "N",
        "K",
        "Route counts",
        "Triton block M",
        "Triton padded",
        "DeepGEMM m_max",
        "Latency ms",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.benchmark,
                    row.model,
                    row.backend,
                    str(row.batch_size),
                    row.gemm,
                    str(row.gemm_m),
                    str(row.gemm_n),
                    str(row.gemm_k),
                    row.route_counts,
                    str(row.triton_block_size_m),
                    str(row.triton_tokens_post_padded_est),
                    str(row.deepgemm_m_max),
                    f"{row.latency_ms:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton and DeepGEMM fused MoE and single GEMM latency versus batch size."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["fused_moe", "single_gemm", "fused_moe_gemm"],
        choices=["fused_moe", "single_gemm", "fused_moe_gemm"],
        help="Benchmark families to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SHAPES),
        help=f"Model dimension presets to run. Choices: {', '.join(MODEL_SHAPES)}",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["triton", "deep_gemm"],
        choices=["triton", "deep_gemm"],
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=list(range(1, 513)),
        help="Comma-separated token batch sizes. Defaults to every size from 1 to 512.",
    )
    parser.add_argument(
        "--route-mode",
        default="single",
        choices=["single", "round_robin", "two_expert_split"],
        help="Synthetic routing pattern used to generate topk_ids.",
    )
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument(
        "--single-gemm-kinds",
        nargs="+",
        default=["up", "down"],
        choices=["up", "down"],
        help=(
            "Single GEMM shapes to benchmark. up uses "
            "[batch, hidden] x [2 * intermediate, hidden]^T; down uses "
            "[batch, intermediate] x [hidden, intermediate]^T."
        ),
    )
    parser.add_argument(
        "--fused-moe-gemm-kinds",
        nargs="+",
        default=["up", "down"],
        choices=["up", "down"],
        help=(
            "GEMM kernels to isolate from the fused MoE code path. up times the "
            "gate/up GEMM launcher; down times the down-projection GEMM launcher."
        ),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16"],
    )
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture each benchmark case in a CUDA graph before timing.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("results/eplb_vs_manual/latency_vs_batch_size.csv"),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse rows already present in --csv-out and only run missing cases.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_experts <= 0:
        raise ValueError("--num-experts must be positive")
    if min(args.batch_sizes) <= 0:
        raise ValueError("--batch-sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if torch.cuda.device_count() <= args.device:
        raise RuntimeError(
            f"requested cuda:{args.device}, but only {torch.cuda.device_count()} CUDA devices are visible"
        )

    server_args = ServerArgs(model_path="dummy")
    server_args.enable_fused_moe_sum_all_reduce = False
    set_global_server_args_for_scheduler(server_args)

    dtype = torch_dtype(args.dtype)
    device = torch.device(f"cuda:{args.device}")

    rows: List[ResultRow] = load_existing_rows(args.csv_out) if args.resume else []
    completed = {result_key(row) for row in rows}
    if rows:
        print(f"Loaded {len(rows)} existing rows from {args.csv_out}", flush=True)

    def record(row: ResultRow) -> None:
        rows.append(row)
        completed.add(result_key(row))
        write_csv(args.csv_out, rows)

    def skip_or_run(key: Tuple[object, ...], message: str) -> bool:
        if key in completed:
            print(f"Skipping existing {message}", flush=True)
            return True
        return False

    for model in selected_models(args.models):
        for backend in args.backends:
            if "fused_moe" in args.benchmarks:
                weight_generator = torch.Generator(device=device).manual_seed(args.seed)
                with torch.cuda.device(device):
                    weights = make_weights(
                        model,
                        args.num_experts,
                        dtype,
                        device,
                        weight_generator,
                        backend,
                    )
                for batch_size in args.batch_sizes:
                    print(
                        f"Running benchmark=fused_moe model={model.name} "
                        f"backend={backend} batch_size={batch_size} "
                        f"route_mode={args.route_mode}",
                        flush=True,
                    )
                    key = candidate_key(
                        benchmark="fused_moe",
                        model=model.name,
                        backend=backend,
                        batch_size=batch_size,
                        gemm="moe",
                        route_mode=args.route_mode,
                        num_experts=args.num_experts,
                        warmups=args.warmups,
                        iters=args.iters,
                        cuda_graph=args.cuda_graph,
                    )
                    if skip_or_run(
                        key,
                        f"benchmark=fused_moe model={model.name} backend={backend} "
                        f"batch_size={batch_size} route_mode={args.route_mode}",
                    ):
                        continue
                    with torch.cuda.device(device):
                        record(
                            benchmark_one(
                                model=model,
                                backend=backend,
                                weights=weights,
                                batch_size=batch_size,
                                route_mode=args.route_mode,
                                num_experts=args.num_experts,
                                dtype=dtype,
                                device=device,
                                seed=args.seed,
                                warmups=args.warmups,
                                iters=args.iters,
                                cuda_graph=args.cuda_graph,
                            )
                        )
                del weights
                torch.cuda.empty_cache()

            if "fused_moe_gemm" in args.benchmarks:
                weight_generator = torch.Generator(device=device).manual_seed(args.seed)
                with torch.cuda.device(device):
                    weights = make_weights(
                        model,
                        args.num_experts,
                        dtype,
                        device,
                        weight_generator,
                        backend,
                    )
                for gemm in args.fused_moe_gemm_kinds:
                    for batch_size in args.batch_sizes:
                        print(
                            f"Running benchmark=fused_moe_gemm model={model.name} "
                            f"backend={backend} gemm={gemm} batch_size={batch_size} "
                            f"route_mode={args.route_mode}",
                            flush=True,
                        )
                        key = candidate_key(
                            benchmark="fused_moe_gemm",
                            model=model.name,
                            backend=backend,
                            batch_size=batch_size,
                            gemm=gemm,
                            route_mode=args.route_mode,
                            num_experts=args.num_experts,
                            warmups=args.warmups,
                            iters=args.iters,
                            cuda_graph=args.cuda_graph,
                        )
                        if skip_or_run(
                            key,
                            f"benchmark=fused_moe_gemm model={model.name} "
                            f"backend={backend} gemm={gemm} batch_size={batch_size} "
                            f"route_mode={args.route_mode}",
                        ):
                            continue
                        with torch.cuda.device(device):
                            record(
                                benchmark_fused_moe_gemm_one(
                                    model=model,
                                    backend=backend,
                                    gemm=gemm,
                                    weights=weights,
                                    batch_size=batch_size,
                                    route_mode=args.route_mode,
                                    num_experts=args.num_experts,
                                    dtype=dtype,
                                    device=device,
                                    seed=args.seed,
                                    warmups=args.warmups,
                                    iters=args.iters,
                                    cuda_graph=args.cuda_graph,
                                )
                            )
                del weights
                torch.cuda.empty_cache()

            if "single_gemm" in args.benchmarks:
                for gemm_index, gemm in enumerate(args.single_gemm_kinds):
                    n, k = single_gemm_shape(model, gemm)
                    weight_generator = torch.Generator(device=device).manual_seed(
                        args.seed + 1000 + gemm_index
                    )
                    with torch.cuda.device(device):
                        weight = make_single_gemm_weight(
                            n=n,
                            k=k,
                            dtype=dtype,
                            device=device,
                            generator=weight_generator,
                            backend=backend,
                        )
                    for batch_size in args.batch_sizes:
                        print(
                            f"Running benchmark=single_gemm model={model.name} "
                            f"backend={backend} gemm={gemm} "
                            f"shape=({batch_size},{n},{k})",
                            flush=True,
                        )
                        key = candidate_key(
                            benchmark="single_gemm",
                            model=model.name,
                            backend=backend,
                            batch_size=batch_size,
                            gemm=gemm,
                            route_mode="",
                            num_experts=0,
                            warmups=args.warmups,
                            iters=args.iters,
                            cuda_graph=args.cuda_graph,
                        )
                        if skip_or_run(
                            key,
                            f"benchmark=single_gemm model={model.name} "
                            f"backend={backend} gemm={gemm} batch_size={batch_size}",
                        ):
                            continue
                        with torch.cuda.device(device):
                            record(
                                benchmark_single_gemm_one(
                                    model=model,
                                    backend=backend,
                                    gemm=gemm,
                                    weight=weight,
                                    batch_size=batch_size,
                                    n=n,
                                    k=k,
                                    dtype=dtype,
                                    device=device,
                                    seed=args.seed,
                                    warmups=args.warmups,
                                    iters=args.iters,
                                    cuda_graph=args.cuda_graph,
                                )
                            )
                    del weight
                    torch.cuda.empty_cache()

    print()
    print(markdown_table(rows))
    write_csv(args.csv_out, rows)
    print(f"\nWrote CSV results to {args.csv_out}")


if __name__ == "__main__":
    main()
