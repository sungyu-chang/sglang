source /home/exouser/sglang-bench-dp-ep-vs-tp-sglang/.venv/bin/activate && python3 benchmark/eplb_vs_manual/latency_vs_batch_size.py \
  --benchmarks fused_moe_gemm \
  --models deepseek-v3 qwen3-235b-a22b qwen3-30b-a3b \
  --backends triton deep_gemm \
  --fused-moe-gemm-kinds up down \
  --route-mode single \
  --num-experts 4 \
  --warmups 100 \
  --iters 1000 \
  --csv-out results/eplb_vs_manual/latency_vs_batch_size.csv
