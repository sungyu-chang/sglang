set shell := ["bash", "-cu"]

# Run Qwen1.5-MoE DP+EP on one node with Mooncake A2A.
# Uses the Qwen DP+EP pipeline only; it does not run TP baselines.
#
# Usage:
#   just qwen15-moe-mooncake              # DP sizes: 1..8
#   just qwen15-moe-mooncake 4            # DP sizes: 1..4
#   just qwen15-moe-mooncake 4 "1 2 4"
qwen15-moe-mooncake gpu_count="8" dp_sizes="":
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH="$PWD/.venv/bin:$PATH"
    export MODEL="Qwen/Qwen1.5-MoE-A2.7B"
    export SERVER_EXTRA_ARGS="--dtype bfloat16 --deepep-mode low_latency"
    export MOE_A2A_BACKEND="mooncake"
    export GPU_COUNT="{{gpu_count}}"
    export RUN_THROUGHPUT="${RUN_THROUGHPUT:-1}"
    export RUN_PROFILE="${RUN_PROFILE:-0}"
    export INPUT_LEN="${INPUT_LEN:-1}"
    export OUTPUT_LEN="${OUTPUT_LEN:-256}"
    export NUM_WARMUPS="${NUM_WARMUPS:-100}"
    export PROMPTS_PER_GPU="${PROMPTS_PER_GPU:-1000}"
    export REQUEST_RATE="${REQUEST_RATE:-inf}"
    export MAX_CONCURRENCY_PER_GPU="${MAX_CONCURRENCY_PER_GPU:-1001}"
    export SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
    if [[ -n "{{dp_sizes}}" ]]; then
      export DP_SIZES="{{dp_sizes}}"
    else
      export DP_SIZES="$(seq -s ' ' 1 "{{gpu_count}}")"
    fi
    python3 benchmark/dp_ep_vs_tp/run_qwen3_moe_ep_pipeline.py

# Profile Qwen1.5-MoE DP+EP on one node with Mooncake A2A via Nsight Systems.
# CUDA graph tracing is set to node mode.
#
# Usage:
#   just qwen15-moe-mooncake-profile      # DP sizes: 1..8
#   just qwen15-moe-mooncake-profile 4    # DP sizes: 1..4
#   just qwen15-moe-mooncake-profile 4 "1 2 4"
qwen15-moe-mooncake-profile gpu_count="8" dp_sizes="":
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH="$PWD/.venv/bin:$PATH"
    export SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
    cmd=(
      python3 benchmark/dp_ep_vs_tp/run_qwen3_moe_ep_nsys.py
      --model Qwen/Qwen1.5-MoE-A2.7B
      --server-extra-args "--dtype bfloat16 --deepep-mode low_latency"
      --gpu-count "{{gpu_count}}"
      --prompts-per-gpu "${PROMPTS_PER_GPU:-1000}"
      --input-len "${INPUT_LEN:-1}"
      --output-len "${OUTPUT_LEN:-256}"
      --num-warmups "${NUM_WARMUPS:-100}"
      --request-rate "${REQUEST_RATE:-inf}"
      --max-concurrency-per-gpu "${MAX_CONCURRENCY_PER_GPU:-1001}"
      --moe-a2a-backend mooncake
      --cuda-graph-trace node
    )
    if [[ -n "{{dp_sizes}}" ]]; then
      cmd+=(--dp-sizes "{{dp_sizes}}")
    else
      cmd+=(--dp-sizes "$(seq -s ' ' 1 "{{gpu_count}}")")
    fi
    "${cmd[@]}"

install_mooncake:
    uv pip install mooncake-transfer-engine-cuda13

install_source:
    pip install -e "python"
