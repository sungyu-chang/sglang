source /home/exouser/sglang-bench-dp-ep-vs-tp-sglang/.venv/bin/activate && python3 benchmark/dp_ep_vs_tp/eplb_vs_manual/bench_eplb_vs_manual_fused_moe.py \
  --models deepseek-v3 qwen3-235b-a22b qwen3-30b-a3b \
  --backends triton deep_gemm \
  --devices 0,1 \
  --no-cuda-graph \
  --iters 1000 \
  --csv-out results/dp_ep_vs_tp/eplb_vs_manual/cuda_graph_off.csv

source /home/exouser/sglang-bench-dp-ep-vs-tp-sglang/.venv/bin/activate && python3 benchmark/dp_ep_vs_tp/eplb_vs_manual/bench_eplb_vs_manual_fused_moe.py \
  --models deepseek-v3 qwen3-235b-a22b qwen3-30b-a3b \
  --backends triton deep_gemm \
  --devices 0,1 \
  --iters 1000 \
  --csv-out results/dp_ep_vs_tp/eplb_vs_manual/cuda_graph_on.csv

source /home/exouser/sglang-bench-dp-ep-vs-tp-sglang/.venv/bin/activate && python3 benchmark/dp_ep_vs_tp/eplb_vs_manual/bench_eplb_vs_manual_fused_moe.py \
  --models deepseek-v3 qwen3-235b-a22b qwen3-30b-a3b \
  --backends triton deep_gemm \
  --devices 0,1 \
  --iters 1000 \
  --deepgemm-skip-quant \
  --csv-out results/dp_ep_vs_tp/eplb_vs_manual/cuda_graph_on_skip_quant.csv
