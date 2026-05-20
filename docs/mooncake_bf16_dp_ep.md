# Mooncake EP BF16 DP+EP Path

This document describes the SGLang-side path for testing DP+EP performance on
A100 with Mooncake EP communication and BF16 expert computation. Mooncake EP
itself already supports BF16 dispatch through `Buffer.dispatch(..., use_fp8=False)`
and BF16 combine inputs, so this change only wires SGLang to use that mode when
the MoE runner is Triton.

## Design

The original Mooncake low-latency dispatcher always requested FP8 dispatch. That
matched the DeepGEMM runner because DeepGEMM consumes FP8 activations plus
per-token scales. For BF16 A100 testing, that coupling is undesirable: the
communication path should send BF16 tensors and the expert FFN should run with
the BF16-capable Triton MoE kernels.

The new dispatch selector is controlled by:

```bash
SGLANG_MOONCAKE_EP_DISPATCH_DTYPE=auto|bf16|fp8
```

`auto` keeps FP8 dispatch for `--moe-runner-backend deep_gemm` and uses BF16
dispatch for explicit `--moe-runner-backend triton`. For unresolved
`--moe-runner-backend auto`, Mooncake keeps the legacy FP8 dispatch default
because FP8 checkpoints may resolve to DeepGEMM after the dispatcher is built.
For BF16 DP+EP testing, either pass `--moe-runner-backend triton` or force
`SGLANG_MOONCAKE_EP_DISPATCH_DTYPE=bf16`. `bf16` and `fp8` force the selected
Mooncake EP dispatch dtype.

For `mooncake + triton`, SGLang registers a `deepep_ll -> triton` bridge:

- Mooncake dispatch returns packed BF16 expert input shaped
  `[num_local_experts, max_tokens_all_ranks, hidden]`.
- The bridge compacts the valid rows using Mooncake's per-expert token counts.
- Each compacted row is routed to exactly one local expert and processed by the
  existing Triton BF16 MoE kernel sequence.
- The output is scattered back to Mooncake's packed layout and passed to
  Mooncake combine with the original `topk_ids` and `topk_weights`.
- The DeepEP-family MoE layer now allows the unquantized `mooncake + triton`
  low-latency path to use its normal quant-method runner instead of requiring
  DeepGEMM.

DeepGEMM registrations and FP8 behavior are unchanged.

## Server Smoke Test

Use a BF16/unquantized MoE checkpoint for this path. Example:

```bash
export SGLANG_MOONCAKE_EP_DISPATCH_DTYPE=bf16
python -m sglang.launch_server \
  --model-path <bf16-moe-model> \
  --tp-size <N> \
  --dp-size <N> \
  --ep-size <N> \
  --enable-dp-attention \
  --moe-a2a-backend mooncake \
  --moe-runner-backend triton \
  --deepep-mode low_latency \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000
```

`SGLANG_MOONCAKE_EP_DISPATCH_DTYPE=auto` should resolve to the same BF16
dispatch mode when `--moe-runner-backend triton` is used.

## DP+EP Benchmark

The existing one-node benchmark can exercise this path:

```bash
export MOE_A2A_BACKEND=mooncake
export SERVER_EXTRA_ARGS="--dtype bfloat16 --moe-runner-backend triton --deepep-mode low_latency"
export SGLANG_MOONCAKE_EP_DISPATCH_DTYPE=bf16
python benchmark/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py
```

For the Qwen wrapper:

```bash
export SGLANG_MOONCAKE_EP_DISPATCH_DTYPE=bf16
python benchmark/dp_ep_vs_tp/run_qwen_one_node_suite.py \
  --moe-a2a-backend mooncake \
  --server-extra-args "--dtype bfloat16 --moe-runner-backend triton --deepep-mode low_latency"
```

## Validation Checklist

On an A100 machine, verify:

- The server starts with `--moe-a2a-backend mooncake --moe-runner-backend triton`.
- Logs report Mooncake EP dispatch dtype as `bf16`.
- Profiling shows Mooncake EP dispatch/combine kernels.
- Profiling shows Triton MoE kernels.
- DeepGEMM grouped GEMM kernels are absent.
- Accuracy smoke tests pass for the selected BF16 checkpoint.

This repository change was prepared on a machine without GPUs, so GPU execution
must be validated separately on the target A100 platform.
