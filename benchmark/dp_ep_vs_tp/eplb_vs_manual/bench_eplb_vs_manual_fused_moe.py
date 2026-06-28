#!/usr/bin/env python3
"""Benchmark EPLB placement against a manual expert placement.

Toy topology:
  - 2 GPUs
  - 3 physical expert slots per GPU
  - 4 logical experts with token counts [100, 90, 30, 20]

The benchmark uses SGLang's fused_moe kernel with deterministic random inputs and
weights. Logical expert weights and token embeddings are generated once per model
shape and reused for both placement strategies.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "python" / "sglang").is_dir()
)
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

# Keep this microbenchmark from triggering broad first-use DeepGEMM precompile
# over thousands of M values. DeepGEMM still JIT-compiles the shapes it runs.
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
# The JIT EP activation path requires some intermediate sizes to satisfy extra
# packing constraints. The fallback path supports all three requested presets.
os.environ.setdefault("SGLANG_OPT_USE_JIT_EP_ACTIVATION", "0")

import triton
import torch

import sglang.srt.layers.moe.moe_runner.deep_gemm as deep_gemm_runner_module
from sglang.srt.layers.moe.ep_moe.kernels import (
    compute_masked_m_triton_kernel,
    compute_seg_indptr_triton_kernel,
    deepgemm_compute_src2dst_triton_kernel,
    fill_gateup_input_triton_kernel,
    post_reorder_triton_kernel,
)
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerInput,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

# Activation per-token-group quantization block size used throughout the
# deep_gemm path (matches DeepGemmMoeQuantInfo.block_shape == [128, 128]).
DEEPGEMM_ACT_GROUP_SIZE = 128

# The production DeepGEMM runner aggressively disposes intermediate tensors,
# including the dispatch input. This benchmark intentionally reuses a stable
# random input tensor across warmup/measurement iterations, so make disposal a
# no-op instead of cloning inputs inside the measured region.
deep_gemm_runner_module.dispose_tensor = lambda tensor: None


NUM_GPUS = 2
SLOTS_PER_GPU = 3
NUM_LOGICAL_EXPERTS = 4
DEFAULT_TOKEN_COUNTS = [100, 90, 30, 20]
EMPTY_SLOT = -1


@dataclass(frozen=True)
class ModelShape:
    name: str
    hidden_size: int
    moe_intermediate_size: int
    dtype_name: str = "bfloat16"

    @property
    def fused_w1_n(self) -> int:
        return 2 * self.moe_intermediate_size


@dataclass(frozen=True)
class Placement:
    name: str
    physical_to_logical: Tuple[int, ...]


@dataclass
class DeviceBatch:
    backend: str
    hidden_states: torch.Tensor
    topk_output: StandardTopKOutput
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor | None
    w2_scale: torch.Tensor | None
    runner: MoeRunner | None
    num_tokens: int
    token_partition: List[int]
    materialized_slots: List[int]
    # When True, `hidden_states` is already fp8 (paired with
    # `hidden_states_scale`) and the deep_gemm path must skip its usual
    # runtime bf16->fp8 activation quantization step.
    skip_quant: bool = False
    hidden_states_scale: torch.Tensor | None = None


@dataclass
class ResultRow:
    model: str
    backend: str
    placement: str
    gpu0_tokens: int
    gpu1_tokens: int
    gpu0_ms: float
    gpu1_ms: float
    critical_path_ms: float


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


def parse_devices(raw: str) -> List[int]:
    devices = parse_int_list(raw)
    if len(devices) != NUM_GPUS:
        raise argparse.ArgumentTypeError(f"expected exactly {NUM_GPUS} CUDA devices")
    return devices


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


def split_evenly(total: int, parts: int) -> List[int]:
    base, remainder = divmod(total, parts)
    return [base + (idx < remainder) for idx in range(parts)]


def compute_eplb_placement(token_counts: Sequence[int]) -> Placement:
    from sglang.srt.eplb.eplb_algorithms.deepseek import rebalance_experts

    weight = torch.tensor([list(token_counts)], dtype=torch.float32)
    physical_to_logical, _, _ = rebalance_experts(
        weight=weight,
        num_replicas=NUM_GPUS * SLOTS_PER_GPU,
        num_groups=1,
        num_nodes=1,
        num_gpus=NUM_GPUS,
        enable_hierarchical=False,
    )
    return Placement(
        name="eplb",
        physical_to_logical=tuple(int(x) for x in physical_to_logical[0].tolist()),
    )


def manual_placement() -> Placement:
    return Placement(
        name="manual",
        physical_to_logical=(0, 1, EMPTY_SLOT, 2, 3, EMPTY_SLOT),
    )


def manual_balanced_placement() -> Placement:
    return Placement(
        name="manual_balanced",
        physical_to_logical=(0, 3, EMPTY_SLOT, 1, 2, EMPTY_SLOT),
    )


def quantize_deepgemm_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    qweights = []
    scales = []
    for expert_id in range(weight.shape[0]):
        qweight, scale = per_block_cast_to_fp8(weight[expert_id].contiguous())
        qweights.append(qweight)
        scales.append(scale)
    return torch.stack(qweights, dim=0), torch.stack(scales, dim=0)


def make_fp8_hidden_direct(
    total_tokens: int,
    hidden_size: int,
    generator: torch.Generator,
    block_k: int = DEEPGEMM_ACT_GROUP_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthesize activations directly as fp8 plus a paired per-token-group(128)
    scale, bypassing `sglang_per_token_group_quant_fp8` entirely. Models a
    pipeline stage where the activation arrives already quantized (e.g. from a
    previous fp8 op) instead of being produced by this layer's runtime
    bf16->fp8 cast."""
    hidden_fp8_cpu = (
        torch.randn((total_tokens, hidden_size), generator=generator, dtype=torch.float32)
        .clamp_(-1.0, 1.0)
        .to(torch.float8_e4m3fn)
        .pin_memory()
    )
    scale_cpu = torch.empty(
        (total_tokens, hidden_size // block_k), dtype=torch.float32
    )
    scale_cpu.uniform_(0.5, 1.5, generator=generator)
    return hidden_fp8_cpu, scale_cpu.pin_memory()


def moe_ep_deepgemm_preprocess_no_quant(
    topk_ids: torch.Tensor,
    num_local_experts: int,
    hidden_states_fp8: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirrors `moe_ep_deepgemm_preprocess` (sglang.srt.layers.moe.ep_moe.kernels)
    but skips its `per_token_group_quant_fp8` call -- `hidden_states_fp8` and
    `hidden_states_scale` are assumed to already be in DeepGEMM's fp8 +
    per-token-group(128) layout."""
    reorder_topk_ids, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)
    seg_indptr = torch.zeros(
        num_local_experts + 1, device=topk_ids.device, dtype=torch.int64
    )
    src2dst = torch.empty(topk_ids.numel(), device=topk_ids.device, dtype=torch.int32)
    masked_m = torch.empty(num_local_experts, device=topk_ids.device, dtype=torch.int32)

    compute_seg_indptr_triton_kernel[(num_local_experts + 1,)](
        reorder_topk_ids, seg_indptr, topk_ids.numel()
    )
    grid = lambda meta: (triton.cdiv(topk_ids.numel(), meta["BLOCK_SIZE"]),)
    compute_masked_m_triton_kernel[(num_local_experts,)](seg_indptr, masked_m)

    m_max = (hidden_states_fp8.size(0) // 256 + 1) * 256
    expected_m = (topk_ids.numel() - 1) // num_local_experts + 1
    gateup_input = torch.empty(
        (num_local_experts, m_max, hidden_states_fp8.size(1)),
        device=hidden_states_fp8.device,
        dtype=hidden_states_fp8.dtype,
    )

    deepgemm_compute_src2dst_triton_kernel[grid](
        topk_ids,
        reorder_ids,
        seg_indptr,
        src2dst,
        m_max,
        topk_ids.numel(),
        BLOCK_SIZE=256,
    )

    gateup_input_scale = torch.empty(
        (gateup_input.size(0), gateup_input.size(1), hidden_states_scale.size(1)),
        device=hidden_states_fp8.device,
        dtype=hidden_states_scale.dtype,
    )

    fill_gateup_input_triton_kernel[(hidden_states_fp8.shape[0],)](
        hidden_states_fp8,
        hidden_states_scale,
        gateup_input,
        gateup_input_scale,
        src2dst,
        topk_ids,
        top_k,
        hidden_states_fp8.size(1),
        hidden_states_scale.size(1),
        BLOCK_SIZE=1024,
    )

    return masked_m, expected_m, src2dst, gateup_input, gateup_input_scale


def token_rows_by_expert(token_counts: Sequence[int]) -> List[List[int]]:
    rows: List[List[int]] = []
    offset = 0
    for count in token_counts:
        rows.append(list(range(offset, offset + count)))
        offset += count
    return rows


def partition_tokens(
    placement: Placement, token_counts: Sequence[int]
) -> Dict[int, List[int]]:
    rows_by_expert = token_rows_by_expert(token_counts)
    rows_by_physical: Dict[int, List[int]] = {
        physical_id: []
        for physical_id in range(NUM_GPUS * SLOTS_PER_GPU)
    }

    for expert_id, expert_rows in enumerate(rows_by_expert):
        physical_slots = [
            slot
            for slot, logical in enumerate(placement.physical_to_logical)
            if logical == expert_id
        ]
        if not physical_slots:
            raise ValueError(
                f"placement {placement.name!r} has no slot for expert {expert_id + 1}"
            )
        split_counts = split_evenly(len(expert_rows), len(physical_slots))
        offset = 0
        for physical_slot, split_count in zip(physical_slots, split_counts):
            rows_by_physical[physical_slot].extend(
                expert_rows[offset : offset + split_count]
            )
            offset += split_count

    return rows_by_physical


def build_device_batch(
    *,
    device_id: int,
    gpu_idx: int,
    placement: Placement,
    rows_by_physical: Dict[int, List[int]],
    hidden_cpu: torch.Tensor,
    logical_w1_cpu: torch.Tensor,
    logical_w2_cpu: torch.Tensor,
    compact_empty_slots: bool,
    backend: str,
    hidden_scale_cpu: Optional[torch.Tensor] = None,
    skip_quant: bool = False,
) -> DeviceBatch:
    if skip_quant and backend != "deep_gemm":
        raise ValueError("skip_quant is only supported for the deep_gemm backend")
    if skip_quant and hidden_scale_cpu is None:
        raise ValueError("skip_quant requires hidden_scale_cpu")
    device = torch.device(f"cuda:{device_id}")
    local_start = gpu_idx * SLOTS_PER_GPU
    local_slots = range(local_start, local_start + SLOTS_PER_GPU)
    local_map = placement.physical_to_logical[local_start : local_start + SLOTS_PER_GPU]
    materialized_slots = [
        local_slot
        for local_slot, logical_expert in enumerate(local_map)
        if not (compact_empty_slots and logical_expert == EMPTY_SLOT)
    ]
    if not materialized_slots:
        raise ValueError(f"GPU {gpu_idx} has no materialized expert slots")
    compact_slot_id = {
        local_slot: compact_id for compact_id, local_slot in enumerate(materialized_slots)
    }

    with torch.cuda.device(device):
        w1_bf16 = torch.empty(
            (len(materialized_slots), logical_w1_cpu.shape[1], logical_w1_cpu.shape[2]),
            dtype=logical_w1_cpu.dtype,
            device=device,
        )
        w2_bf16 = torch.empty(
            (len(materialized_slots), logical_w2_cpu.shape[1], logical_w2_cpu.shape[2]),
            dtype=logical_w2_cpu.dtype,
            device=device,
        )
        for compact_id, local_slot in enumerate(materialized_slots):
            logical_expert = local_map[local_slot]
            if logical_expert == EMPTY_SLOT:
                w1_bf16[compact_id].zero_()
                w2_bf16[compact_id].zero_()
            else:
                w1_bf16[compact_id].copy_(
                    logical_w1_cpu[logical_expert], non_blocking=True
                )
                w2_bf16[compact_id].copy_(
                    logical_w2_cpu[logical_expert], non_blocking=True
                )

        if backend == "deep_gemm":
            w1, w1_scale = quantize_deepgemm_weight(w1_bf16)
            w2, w2_scale = quantize_deepgemm_weight(w2_bf16)
        else:
            w1, w2 = w1_bf16, w2_bf16
            w1_scale = None
            w2_scale = None

        ordered_rows: List[int] = []
        local_topk_ids: List[int] = []
        token_partition: List[int] = []
        for local_slot, physical_slot in enumerate(local_slots):
            slot_rows = rows_by_physical[physical_slot]
            token_partition.append(len(slot_rows))
            ordered_rows.extend(slot_rows)
            if slot_rows:
                local_topk_ids.extend([compact_slot_id[local_slot]] * len(slot_rows))

        hidden_states = hidden_cpu[ordered_rows].to(device=device, non_blocking=True)
        hidden_states_scale = (
            hidden_scale_cpu[ordered_rows].to(device=device, non_blocking=True)
            if skip_quant
            else None
        )
        topk_ids = torch.tensor(local_topk_ids, dtype=torch.int64, device=device).view(
            -1, 1
        )
        topk_weights = torch.ones(
            (len(ordered_rows), 1), dtype=torch.float32, device=device
        )
        router_logits = torch.zeros(
            (len(ordered_rows), len(materialized_slots)),
            dtype=torch.float32,
            device=device,
        )

    runner = (
        MoeRunner(
            MoeRunnerBackend.DEEP_GEMM,
            MoeRunnerConfig(
                num_experts=w1.shape[0],
                num_local_experts=w1.shape[0],
                hidden_size=logical_w1_cpu.shape[2],
                intermediate_size_per_partition=logical_w2_cpu.shape[2],
                top_k=1,
                params_dtype=logical_w1_cpu.dtype,
                inplace=False,
            ),
        )
        if backend == "deep_gemm"
        else None
    )

    return DeviceBatch(
        backend=backend,
        hidden_states=hidden_states,
        topk_output=StandardTopKOutput(topk_weights, topk_ids, router_logits),
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        runner=runner,
        num_tokens=len(ordered_rows),
        token_partition=token_partition,
        materialized_slots=materialized_slots,
        skip_quant=skip_quant,
        hidden_states_scale=hidden_states_scale,
    )


def _run_deep_gemm_skip_quant(
    batch: DeviceBatch, quant_info: DeepGemmMoeQuantInfo
) -> torch.Tensor:
    assert batch.hidden_states_scale is not None
    assert batch.runner is not None and batch.runner.runner_core is not None
    topk_ids = batch.topk_output.topk_ids
    topk_weights = batch.topk_output.topk_weights
    num_local_experts = batch.w1.shape[0]

    masked_m, expected_m, src2dst, gateup_input, gateup_input_scale = (
        moe_ep_deepgemm_preprocess_no_quant(
            topk_ids,
            num_local_experts,
            batch.hidden_states,
            batch.hidden_states_scale,
            top_k=1,
        )
    )
    runner_input = DeepGemmRunnerInput(
        hidden_states=gateup_input,
        hidden_states_scale=gateup_input_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
    )
    running_state = {"hidden_states_device": batch.hidden_states.device}
    runner_output = batch.runner.runner_core.run(runner_input, quant_info, running_state)

    output = torch.empty(
        batch.hidden_states.shape,
        dtype=torch.bfloat16,
        device=batch.hidden_states.device,
    )
    post_reorder_triton_kernel[(batch.hidden_states.shape[0],)](
        runner_output.hidden_states,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        1,
        batch.hidden_states.shape[1],
        BLOCK_SIZE=512,
    )
    return output


def run_fused_moe(batch: DeviceBatch) -> torch.Tensor:
    if batch.backend == "deep_gemm":
        assert batch.runner is not None
        assert batch.w1_scale is not None and batch.w2_scale is not None
        quant_info = DeepGemmMoeQuantInfo(
            w13_weight=batch.w1,
            w2_weight=batch.w2,
            use_fp8=True,
            w13_scale=batch.w1_scale,
            w2_scale=batch.w2_scale,
            block_shape=[128, 128],
        )
        if batch.skip_quant:
            return _run_deep_gemm_skip_quant(batch, quant_info)

        dispatch_output = StandardDispatchOutput(
            batch.hidden_states,
            None,
            batch.topk_output,
        )
        return batch.runner.run(dispatch_output, quant_info).hidden_states

    return fused_moe(
        batch.hidden_states,
        batch.w1,
        batch.w2,
        batch.topk_output,
        moe_runner_config=MoeRunnerConfig(
            num_experts=batch.w1.shape[0],
            num_local_experts=batch.w1.shape[0],
            top_k=1,
            inplace=False,
        ),
    )


def synchronize_batches(batches: Sequence[DeviceBatch]):
    for batch in batches:
        torch.cuda.synchronize(batch.hidden_states.device)


def capture_cuda_graph(batch: DeviceBatch) -> torch.cuda.CUDAGraph:
    device = batch.hidden_states.device
    side_stream = torch.cuda.Stream(device=device)
    side_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            run_fused_moe(batch)
    torch.cuda.current_stream(device).wait_stream(side_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_fused_moe(batch)
    return graph


def time_batches(
    batches: Sequence[DeviceBatch],
    warmups: int,
    iters: int,
    use_cuda_graph: bool = True,
) -> List[float]:
    active_batches = [batch for batch in batches if batch.num_tokens > 0]
    if not active_batches:
        return [0.0 for _ in batches]

    for _ in range(warmups):
        for batch in active_batches:
            with torch.cuda.device(batch.hidden_states.device):
                run_fused_moe(batch)
    synchronize_batches(active_batches)

    graphs: Dict[int, torch.cuda.CUDAGraph] = {}
    if use_cuda_graph:
        for batch in active_batches:
            device = batch.hidden_states.device
            with torch.cuda.device(device):
                graphs[device.index] = capture_cuda_graph(batch)
        synchronize_batches(active_batches)

    def run_iteration(batch: DeviceBatch) -> None:
        if use_cuda_graph:
            graphs[batch.hidden_states.device.index].replay()
        else:
            run_fused_moe(batch)

    starts: Dict[int, torch.cuda.Event] = {}
    ends: Dict[int, torch.cuda.Event] = {}
    for batch in active_batches:
        device = batch.hidden_states.device
        with torch.cuda.device(device):
            starts[device.index] = torch.cuda.Event(enable_timing=True)
            ends[device.index] = torch.cuda.Event(enable_timing=True)
            starts[device.index].record()

    for _ in range(iters):
        for batch in active_batches:
            with torch.cuda.device(batch.hidden_states.device):
                run_iteration(batch)

    for batch in active_batches:
        device = batch.hidden_states.device
        with torch.cuda.device(device):
            ends[device.index].record()
    synchronize_batches(active_batches)

    latencies = []
    for batch in batches:
        if batch.num_tokens == 0:
            latencies.append(0.0)
        else:
            device_idx = batch.hidden_states.device.index
            latencies.append(starts[device_idx].elapsed_time(ends[device_idx]) / iters)
    return latencies


def worker_placement(name: str, token_counts: Sequence[int]) -> Placement:
    if name == "eplb":
        return compute_eplb_placement(token_counts)
    if name == "manual":
        return manual_placement()
    if name == "manual_balanced":
        return manual_balanced_placement()
    raise ValueError(f"unknown worker placement {name!r}")


def parse_worker_result(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"worker did not emit a JSON result; stdout was:\n{stdout}")


def run_single_batch_worker(args: argparse.Namespace):
    if len(args.token_counts) != NUM_LOGICAL_EXPERTS:
        raise ValueError(f"--token-counts must contain {NUM_LOGICAL_EXPERTS} values")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SGLang fused_moe benchmark")

    server_args = ServerArgs(model_path="dummy")
    server_args.enable_fused_moe_sum_all_reduce = False
    set_global_server_args_for_scheduler(server_args)

    model = selected_models([args.worker_model])[0]
    dtype = torch_dtype(args.dtype)
    placement = worker_placement(args.worker_placement, args.token_counts)
    rows_by_physical = partition_tokens(placement, args.token_counts)
    skip_quant = args.deepgemm_skip_quant and args.worker_backend == "deep_gemm"
    hidden_cpu, logical_w1_cpu, logical_w2_cpu, hidden_scale_cpu = make_logical_tensors(
        model, args.token_counts, dtype, args.seed, skip_quant=skip_quant
    )
    batch = build_device_batch(
        device_id=0,
        gpu_idx=args.worker_gpu_idx,
        placement=placement,
        rows_by_physical=rows_by_physical,
        hidden_cpu=hidden_cpu,
        hidden_scale_cpu=hidden_scale_cpu,
        logical_w1_cpu=logical_w1_cpu,
        logical_w2_cpu=logical_w2_cpu,
        compact_empty_slots=args.compact_empty_slots,
        backend=args.worker_backend,
        skip_quant=skip_quant,
    )
    latency = time_batches([batch], args.warmups, args.iters, args.cuda_graph)[0]
    print(
        json.dumps(
            {
                "latency_ms": latency,
                "num_tokens": batch.num_tokens,
                "token_partition": batch.token_partition,
                "materialized_slots": batch.materialized_slots,
            }
        ),
        flush=True,
    )


def time_placement_isolated(
    *,
    model: ModelShape,
    placement: Placement,
    token_counts: Sequence[int],
    dtype_name: str,
    seed: int,
    devices: Sequence[int],
    warmups: int,
    iters: int,
    compact_empty_slots: bool,
    backend: str,
    cuda_graph: bool,
    deepgemm_skip_quant: bool,
) -> Tuple[List[float], List[int]]:
    processes = []
    script_path = Path(__file__).resolve()
    token_counts_arg = ",".join(str(count) for count in token_counts)
    for gpu_idx, physical_device in enumerate(devices):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(physical_device)
        cmd = [
            sys.executable,
            str(script_path),
            "--worker-single-batch",
            "--worker-model",
            model.name,
            "--worker-placement",
            placement.name,
            "--worker-gpu-idx",
            str(gpu_idx),
            "--worker-backend",
            backend,
            "--token-counts",
            token_counts_arg,
            "--dtype",
            dtype_name,
            "--warmups",
            str(warmups),
            "--iters",
            str(iters),
            "--seed",
            str(seed),
        ]
        if compact_empty_slots:
            cmd.append("--compact-empty-slots")
        cmd.append("--cuda-graph" if cuda_graph else "--no-cuda-graph")
        if deepgemm_skip_quant:
            cmd.append("--deepgemm-skip-quant")
        processes.append(
            subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    worker_results = []
    for gpu_idx, process in enumerate(processes):
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"{backend} worker failed for {model.name} {placement.name} gpu{gpu_idx} "
                f"with exit code {process.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        worker_results.append(parse_worker_result(stdout))

    return (
        [float(result["latency_ms"]) for result in worker_results],
        [int(result["num_tokens"]) for result in worker_results],
    )


def make_logical_tensors(
    model: ModelShape,
    token_counts: Sequence[int],
    dtype: torch.dtype,
    seed: int,
    skip_quant: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total_tokens = sum(token_counts)
    hidden_scale_cpu = None
    if skip_quant:
        hidden_cpu, hidden_scale_cpu = make_fp8_hidden_direct(
            total_tokens, model.hidden_size, generator
        )
    else:
        hidden_cpu = torch.randn(
            (total_tokens, model.hidden_size), generator=generator, dtype=dtype
        ).pin_memory()
    logical_w1_cpu = torch.randn(
        (NUM_LOGICAL_EXPERTS, model.fused_w1_n, model.hidden_size),
        generator=generator,
        dtype=dtype,
    ).pin_memory()
    logical_w2_cpu = torch.randn(
        (NUM_LOGICAL_EXPERTS, model.hidden_size, model.moe_intermediate_size),
        generator=generator,
        dtype=dtype,
    ).pin_memory()
    return hidden_cpu, logical_w1_cpu, logical_w2_cpu, hidden_scale_cpu


def physical_slot_name(physical_slot: int) -> str:
    gpu_id, local_slot = divmod(physical_slot, SLOTS_PER_GPU)
    return f"gpu{gpu_id}:slot{local_slot}"


def format_placement(placement: Placement) -> str:
    entries = []
    for physical_slot, logical_expert in enumerate(placement.physical_to_logical):
        expert_name = (
            "empty" if logical_expert == EMPTY_SLOT else f"E{logical_expert + 1}"
        )
        entries.append(f"{physical_slot_name(physical_slot)}={expert_name}")
    return ", ".join(entries)


def format_token_partition(
    placement: Placement, rows_by_physical: Dict[int, List[int]]
) -> str:
    entries = []
    for physical_slot, logical_expert in enumerate(placement.physical_to_logical):
        expert_name = (
            "empty" if logical_expert == EMPTY_SLOT else f"E{logical_expert + 1}"
        )
        entries.append(
            f"{physical_slot_name(physical_slot)}:{expert_name}:{len(rows_by_physical[physical_slot])}"
        )
    return ", ".join(entries)


def markdown_table(rows: Sequence[ResultRow]) -> str:
    headers = [
        "Model",
        "Backend",
        "Placement",
        "GPU0 tokens",
        "GPU1 tokens",
        "GPU0 ms",
        "GPU1 ms",
        "Critical path ms",
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
                    row.model,
                    row.backend,
                    row.placement,
                    str(row.gpu0_tokens),
                    str(row.gpu1_tokens),
                    f"{row.gpu0_ms:.4f}",
                    f"{row.gpu1_ms:.4f}",
                    f"{row.critical_path_ms:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


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


def benchmark_model(
    *,
    model: ModelShape,
    placements: Sequence[Placement],
    token_counts: Sequence[int],
    dtype: torch.dtype,
    seed: int,
    devices: Sequence[int],
    warmups: int,
    iters: int,
    compact_empty_slots: bool,
    backend: str,
    dtype_name: str,
    isolate_devices: bool,
    cuda_graph: bool,
    deepgemm_skip_quant: bool,
) -> List[ResultRow]:
    results: List[ResultRow] = []
    skip_quant = deepgemm_skip_quant and backend == "deep_gemm"

    for placement in placements:
        if isolate_devices:
            latencies, token_totals = time_placement_isolated(
                model=model,
                placement=placement,
                token_counts=token_counts,
                dtype_name=dtype_name,
                seed=seed,
                devices=devices,
                warmups=warmups,
                iters=iters,
                compact_empty_slots=compact_empty_slots,
                backend=backend,
                cuda_graph=cuda_graph,
                deepgemm_skip_quant=skip_quant,
            )
        else:
            hidden_cpu, logical_w1_cpu, logical_w2_cpu, hidden_scale_cpu = (
                make_logical_tensors(
                    model, token_counts, dtype, seed, skip_quant=skip_quant
                )
            )
            rows_by_physical = partition_tokens(placement, token_counts)
            batches = [
                build_device_batch(
                    device_id=devices[gpu_idx],
                    gpu_idx=gpu_idx,
                    placement=placement,
                    rows_by_physical=rows_by_physical,
                    hidden_cpu=hidden_cpu,
                    hidden_scale_cpu=hidden_scale_cpu,
                    logical_w1_cpu=logical_w1_cpu,
                    logical_w2_cpu=logical_w2_cpu,
                    compact_empty_slots=compact_empty_slots,
                    backend=backend,
                    skip_quant=skip_quant,
                )
                for gpu_idx in range(NUM_GPUS)
            ]
            latencies = time_batches(batches, warmups, iters, cuda_graph)
            token_totals = [batch.num_tokens for batch in batches]

            del batches
            del hidden_cpu, logical_w1_cpu, logical_w2_cpu, hidden_scale_cpu
            for device_id in devices:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()

        results.append(
            ResultRow(
                model=model.name,
                backend=backend,
                placement=placement.name,
                gpu0_tokens=token_totals[0],
                gpu1_tokens=token_totals[1],
                gpu0_ms=latencies[0],
                gpu1_ms=latencies[1],
                critical_path_ms=max(latencies),
            )
        )

    gc.collect()
    return results


def write_csv(path: Path, rows: Sequence[ResultRow], placements: Sequence[Placement]):
    placement_maps = {
        placement.name: ",".join(str(logical) for logical in placement.physical_to_logical)
        for placement in placements
    }
    fieldnames = [
        "model",
        "backend",
        "placement",
        "physical_to_logical",
        "gpu0_tokens",
        "gpu1_tokens",
        "gpu0_ms",
        "gpu1_ms",
        "critical_path_ms",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row.__dict__,
                    "physical_to_logical": placement_maps[row.placement],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SGLang EPLB placement with a manual placement using fused_moe."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SHAPES),
        help=f"Model dimension presets to run. Choices: {', '.join(MODEL_SHAPES)}",
    )
    parser.add_argument(
        "--token-counts",
        type=parse_int_list,
        default=DEFAULT_TOKEN_COUNTS,
        help="Comma-separated token counts for logical experts 1..4.",
    )
    parser.add_argument("--devices", type=parse_devices, default=[0, 1])
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16"],
    )
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["triton"],
        choices=["triton", "deep_gemm"],
        help="MoE compute backends to benchmark.",
    )
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument(
        "--compact-empty-slots",
        action="store_true",
        help="Skip allocating all-zero fused_moe weights for empty physical slots.",
    )
    parser.add_argument(
        "--isolate-devices",
        action="store_true",
        help="Run each GPU batch in its own CUDA_VISIBLE_DEVICES-isolated worker process.",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture each device's fused_moe iteration into a CUDA graph and replay it "
        "during timing (default: enabled). Use --no-cuda-graph to fall back to eager dispatch.",
    )
    parser.add_argument(
        "--deepgemm-skip-quant",
        action="store_true",
        help="deep_gemm backend only: synthesize activations directly as fp8 plus a "
        "paired per-token-group(128) scale, and skip the runtime bf16->fp8 activation "
        "quantization kernel entirely. No effect on the triton backend.",
    )
    parser.add_argument("--worker-single-batch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-model", choices=list(MODEL_SHAPES), help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-placement",
        choices=["eplb", "manual", "manual_balanced"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-gpu-idx", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-backend",
        choices=["triton", "deep_gemm"],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.token_counts) != NUM_LOGICAL_EXPERTS:
        raise ValueError(f"--token-counts must contain {NUM_LOGICAL_EXPERTS} values")
    if min(args.token_counts) < 0:
        raise ValueError("--token-counts must be non-negative")
    if args.worker_single_batch:
        run_single_batch_worker(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SGLang fused_moe benchmark")
    if torch.cuda.device_count() <= max(args.devices):
        raise RuntimeError(
            f"requested devices {args.devices}, but only {torch.cuda.device_count()} CUDA devices are visible"
        )

    server_args = ServerArgs(model_path="dummy")
    server_args.enable_fused_moe_sum_all_reduce = False
    set_global_server_args_for_scheduler(server_args)

    dtype = torch_dtype(args.dtype)
    placements = [
        compute_eplb_placement(args.token_counts),
        manual_placement(),
        manual_balanced_placement(),
    ]
    # Capturing a CUDA graph on one GPU corrupts Triton's launch state for a
    # second GPU warmed up afterward in the same process, so CUDA graph runs
    # always isolate each GPU into its own subprocess.
    isolate_devices = (
        args.isolate_devices or "deep_gemm" in args.backends or args.cuda_graph
    )

    print(f"Logical expert token counts: {args.token_counts}")
    for placement in placements:
        rows_by_physical = partition_tokens(placement, args.token_counts)
        print(f"{placement.name} placement: {format_placement(placement)}")
        print(
            f"{placement.name} token partition: "
            f"{format_token_partition(placement, rows_by_physical)}"
        )
    if isolate_devices:
        print(
            "Timing mode: isolated worker process per GPU "
            "(used automatically for deep_gemm and/or CUDA graph capture)."
        )
    print(
        "CUDA graph: "
        + ("each device's iteration is captured and replayed." if args.cuda_graph else "disabled (eager dispatch).")
    )
    if args.deepgemm_skip_quant:
        print(
            "deep_gemm activation quant: SKIPPED -- hidden_states synthesized directly "
            "as fp8 + per-token-group(128) scale (no effect on triton backend)."
        )
    print()

    rows: List[ResultRow] = []
    for backend in args.backends:
        for model in selected_models(args.models):
            rows.extend(
                benchmark_model(
                    model=model,
                    placements=placements,
                    token_counts=args.token_counts,
                    dtype=dtype,
                    seed=args.seed,
                    devices=args.devices,
                    warmups=args.warmups,
                    iters=args.iters,
                    compact_empty_slots=args.compact_empty_slots,
                    backend=backend,
                    dtype_name=args.dtype,
                    isolate_devices=isolate_devices,
                    cuda_graph=args.cuda_graph,
                    deepgemm_skip_quant=args.deepgemm_skip_quant,
                )
            )

    print(markdown_table(rows))
    if args.csv_out is not None:
        write_csv(args.csv_out, rows, placements)
        print(f"\nWrote CSV results to {args.csv_out}")


if __name__ == "__main__":
    main()
