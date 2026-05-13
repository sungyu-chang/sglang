# DP+EP vs TP Online Benchmarks for SGLang

This directory contains SGLang online serving benchmark harnesses adapted from
the vLLM `benchmarks/dp_ep_vs_tp` workflow. The main experiments are:

1. Single-node Qwen MoE TP vs DP+EP experiments.
2. Qwen3-MoE DP+EP throughput, torch-profile, and Nsight Systems pipelines.
3. Two-node Ray TP1x2 vs DP+EP experiments.

Use these EP comparisons for MoE models. For dense models, expert parallelism
does not create useful expert sharding.

## SGLang Mapping

The vLLM DP+EP shape is mapped to SGLang as:

```text
TP baseline: --tp-size N
DP+EP:       --tp-size N --dp-size N --ep-size N --enable-dp-attention
```

With `--enable-dp-attention`, attention is data parallel while FFN/MoE work is
sharded across the TP/EP ranks. Prefix caching is disabled with
`--disable-radix-cache`. SGLang `bench_serving` ignores EOS by default.

The old `ALL2ALL_BACKEND` environment variable is still accepted for
compatibility, but SGLang's preferred knob is `MOE_A2A_BACKEND`. Supported
values are SGLang server values such as `none`, `deepep`, `mooncake`, `nixl`,
`mori`, `ascend_fuseep`, and `flashinfer`. The vLLM value
`allgather_reducescatter` is treated as SGLang `none`.

## Setup

Run commands from the repo root after setting up the SGLang environment:

```bash
PATH="$(pwd)/.venv/bin:$PATH"
```

Figure generation uses matplotlib:

```bash
uv pip install matplotlib
```

## Single-Node TP vs DP+EP

Run both Qwen presets and include TP baselines:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
.venv/bin/python benchmark/dp_ep_vs_tp/run_qwen_one_node_suite.py \
  --include-tp \
  --gpu-count 8 \
  --tp-sizes "1 2 4 8" \
  --input-len 1 \
  --output-len 256 \
  --num-warmups 100 \
  --max-concurrency-per-gpu 1001 \
  --moe-a2a-backend none
```

Run one model preset:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
.venv/bin/python benchmark/dp_ep_vs_tp/run_qwen_one_node_suite.py \
  --models qwen3 \
  --include-tp \
  --gpu-count 8 \
  --tp-sizes "1 2 4 8" \
  --input-len 1 \
  --output-len 256 \
  --num-warmups 100 \
  --max-concurrency-per-gpu 1001
```

Results are written under:

```text
results/dp_ep_vs_tp/one_node_online/YYYY-MM-DD_HH-MM-SS/
results/dp_ep_vs_tp/analysis/YYYY-MM-DD_HH-MM-SS/
```

## Qwen3-MoE DP+EP Pipeline

Clean throughput sweep:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
RUN_PROFILE=0 \
MODEL=Qwen/Qwen3-30B-A3B \
SERVER_EXTRA_ARGS="--dtype bfloat16" \
GPU_COUNT=8 \
PROMPTS_PER_GPU=1000 \
INPUT_LEN=1 \
OUTPUT_LEN=256 \
NUM_WARMUPS=100 \
REQUEST_RATE=inf \
MAX_CONCURRENCY_PER_GPU=1001 \
MOE_A2A_BACKEND=none \
.venv/bin/python benchmark/dp_ep_vs_tp/run_qwen3_moe_ep_pipeline.py
```

Torch profiling sweep:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
RUN_THROUGHPUT=0 \
RUN_PROFILE=1 \
MODEL=Qwen/Qwen3-30B-A3B \
SERVER_EXTRA_ARGS="--dtype bfloat16" \
GPU_COUNT=8 \
PROMPTS_PER_GPU=1000 \
INPUT_LEN=1 \
OUTPUT_LEN=256 \
NUM_WARMUPS=100 \
REQUEST_RATE=inf \
MAX_CONCURRENCY_PER_GPU=1001 \
MOE_A2A_BACKEND=none \
.venv/bin/python benchmark/dp_ep_vs_tp/run_qwen3_moe_ep_pipeline.py
```

The pipeline writes:

```text
results/dp_ep_vs_tp/qwen3_moe_ep_pipeline/YYYY-MM-DD_HH-MM-SS/
```

Throughput artifacts are under `throughput/`; torch profiler traces are under
`profile/profiler_traces/<case>/`.

## Nsight Systems

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
.venv/bin/python benchmark/dp_ep_vs_tp/run_qwen3_moe_ep_nsys.py \
  --model Qwen/Qwen3-30B-A3B \
  --server-extra-args "--dtype bfloat16" \
  --gpu-count 8 \
  --prompts-per-gpu 1000 \
  --input-len 1 \
  --output-len 256 \
  --num-warmups 100 \
  --request-rate inf \
  --max-concurrency-per-gpu 1001 \
  --moe-a2a-backend none
```

Nsight results are written under:

```text
results/dp_ep_vs_tp/nsys_profiling/YYYY-MM-DD_HH-MM-SS/
```

Use `--nsys-bin` if `nsys` is not on `PATH`. By default, the harness captures
`cuda,nvtx`, exports `cuda_gpu_kern_sum` and `cuda_gpu_trace` CSVs, and writes
one `.nsys-rep` base per DP+EP case.

## Plotting

Rerender a Qwen3 pipeline throughput plot:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
.venv/bin/python benchmark/dp_ep_vs_tp/plot_qwen3_moe_ep_pipeline.py \
  results/dp_ep_vs_tp/qwen3_moe_ep_pipeline/YYYY-MM-DD_HH-MM-SS
```

Rerender a TP vs DP+EP throughput figure:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
.venv/bin/python benchmark/dp_ep_vs_tp/plot_total_token_throughput.py \
  results/dp_ep_vs_tp/one_node_online/YYYY-MM-DD_HH-MM-SS \
  --output results/dp_ep_vs_tp/analysis/YYYY-MM-DD_HH-MM-SS/total_token_throughput.png
```

## Two-Node Ray

This script assumes a Ray cluster is already running across two one-GPU nodes
and is launched from the Ray head node. It compares:

- `tp1x2`: SGLang Ray deployment with `TP=1`, `DP=2`, EP disabled.
- `dp2_ep`: SGLang Ray deployment with `TP=2`, `DP=2`, `EP=2`, DP attention.

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
MODEL=allenai/OLMoE-1B-7B-0924-Instruct \
NUM_WARMUPS=100 \
SERVER_EXTRA_ARGS="--dtype float16" \
MOE_A2A_BACKEND=none \
.venv/bin/python benchmark/dp_ep_vs_tp/run_two_node_ray_tp1_vs_dp_ep.py
```

## Common Knobs

```bash
MODEL=deepseek-ai/DeepSeek-V2-Lite
GPU_COUNT=8
TP_SIZES="1 2 4 8"
DP_SIZES=
NUM_PROMPTS=
PROMPTS_PER_GPU=1000
INPUT_LEN=1
OUTPUT_LEN=256
NUM_WARMUPS=100
REQUEST_RATE=inf
MAX_CONCURRENCY=
MAX_CONCURRENCY_PER_GPU=
MAX_MODEL_LEN=
MOE_A2A_BACKEND=none
BASE_PORT=8100
HOST=127.0.0.1
SERVER_EXTRA_ARGS="--trust-remote-code --dtype bfloat16"
BENCH_EXTRA_ARGS=
PYTHON_BIN=.venv/bin/python
SMOKE_RUN=0
RUN_NOTES=
FIX_NOTES=
PROFILE_MODULES=0
RUN_THROUGHPUT=1
RUN_PROFILE=1
PROFILE_DELAY_ITERATIONS=5
PROFILE_MAX_ITERATIONS=20
PROFILE_LAYER_SCOPES=0
DISABLE_PREFIX_CACHING=1
```

Every run directory includes a generated `README.md` recording setup, planned
cases, artifacts, run notes, fix notes, and failure details if the run aborts.
