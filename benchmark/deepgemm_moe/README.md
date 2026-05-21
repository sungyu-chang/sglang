# DeepGEMM MoE Benchmark

This benchmark measures DeepGEMM MoE expert compute on a single GPU without routing dispatch, A2A, or combine. It directly mocks expert assignment and, by default, calls SGLang's `DeepGemmRunnerCore` instead of hand-assembling the expert MLP from individual DeepGEMM wrapper calls.

The default path does not compute top-k. It builds the input tensor, constructs mocked `m_indices`, and lets the SGLang DeepGEMM MoE runner execute gate/up, activation plus FP8 quantization, and down projection.

The first intended experiment is single-expert performance over input batch sizes `1 2 4 8 16 32 64 128 256`.

For the two-expert experiment, use `--active-experts 2 --distribution even` and even batch sizes such as `2 4 8 16 32 64 128 256`.

## Presets

- `deepseek-v3`: hidden size `7168`, MoE intermediate size `2048`, experts `256`, top-k `8`.
- `qwen3-235b-a22b`: hidden size `4096`, MoE intermediate size `1536`, experts `128`, top-k `8`.
- `qwen3-30b-a3b`: hidden size `2048`, MoE intermediate size `768`, experts `128`, top-k `8`.

By default `--tp-size 1` uses full one-GPU expert dimensions. Set `--tp-size` only when you want to reproduce a tensor-parallel sharded GEMM shape while still running the experiment on one GPU.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark/deepgemm_moe/bench_deepgemm_moe.py \
  --model-preset deepseek-v3 \
  --op mlp \
  --use-cuda-graph
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark/deepgemm_moe/bench_deepgemm_moe.py \
  --model-preset qwen3-235b-a22b \
  --op mlp \
  --use-cuda-graph
```

Useful quick smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark/deepgemm_moe/bench_deepgemm_moe.py \
  --model-preset qwen3-30b-a3b \
  --op mlp \
  --batch-sizes 1 2 \
  --warmup-iters 1 \
  --measure-iters 2
```

CUDA Graph replay is opt-in with `--use-cuda-graph`. Use `--cuda-graph-inner-iters` when you want each captured replay to contain multiple MoE runner invocations and report per-invocation latency.

Each batch size has two warmup phases by default: `--warmup-iters 10` before CUDA Graph capture, then `--timed-warmup-iters 5` on the exact callable being measured. For CUDA Graph runs, the second phase warms up `graph.replay()` so the first measured replay is not included. The default measurement count is `--measure-iters 100`.

The historical direct wrapper path is still available for comparison:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark/deepgemm_moe/bench_deepgemm_moe.py \
  --backend deep_gemm_wrapper \
  --model-preset deepseek-v3 \
  --op gateup \
  --batch-sizes 1 2
```

Each run writes to:

```text
results/deepgemm_moe/YYYY-MM-DD_HH-MM-SS/
```

The run directory includes `README.md`, `config.json`, `env.json`, `metrics.json`, `metrics.csv`, and `raw_latencies.csv` unless raw latencies are disabled.

## Plot

```bash
.venv/bin/python benchmark/deepgemm_moe/plot_deepgemm_moe.py \
  results/deepgemm_moe/YYYY-MM-DD_HH-MM-SS
```

The plotting script writes latency, throughput, and effective TFLOP/s figures as both PNG and PDF into the same run directory.

To overlay multiple model runs on the same figures:

```bash
.venv/bin/python benchmark/deepgemm_moe/plot_deepgemm_moe_compare.py \
  results/deepgemm_moe/RUN_DEEPSEEK \
  results/deepgemm_moe/RUN_QWEN3_235B \
  results/deepgemm_moe/RUN_QWEN3_30B
```

The comparison script creates `results/deepgemm_moe/compare_YYYY-MM-DD_HH-MM-SS/` with combined CSV data and overlaid PNG/PDF figures.

For the two-expert even-split experiment, use the dedicated plotter:

```bash
.venv/bin/python benchmark/deepgemm_moe/plot_deepgemm_moe_two_experts.py \
  results/deepgemm_moe/RUN_DEEPSEEK_TWO_EXPERTS \
  results/deepgemm_moe/RUN_QWEN3_235B_TWO_EXPERTS \
  results/deepgemm_moe/RUN_QWEN3_30B_TWO_EXPERTS
```

For any even-split expert count, use the generic plotter:

```bash
.venv/bin/python benchmark/deepgemm_moe/plot_deepgemm_moe_even_experts.py \
  --active-experts 4 \
  results/deepgemm_moe/RUN_DEEPSEEK_FOUR_EXPERTS \
  results/deepgemm_moe/RUN_QWEN3_235B_FOUR_EXPERTS \
  results/deepgemm_moe/RUN_QWEN3_30B_FOUR_EXPERTS
```

To overlay single-expert and two-expert results on the same figures, with dotted lines for single expert and solid lines for two experts:

```bash
.venv/bin/python benchmark/deepgemm_moe/plot_deepgemm_moe_single_vs_two.py \
  --single-run-dirs RUN_DEEPSEEK_SINGLE RUN_QWEN3_235B_SINGLE RUN_QWEN3_30B_SINGLE \
  --two-run-dirs RUN_DEEPSEEK_TWO RUN_QWEN3_235B_TWO RUN_QWEN3_30B_TWO
```

To overlay several active expert counts on the same figures:

```bash
.venv/bin/python benchmark/deepgemm_moe/plot_deepgemm_moe_expert_scaling.py \
  --group 1:RUN_DEEPSEEK_SINGLE,RUN_QWEN3_235B_SINGLE,RUN_QWEN3_30B_SINGLE \
  --group 2:RUN_DEEPSEEK_TWO,RUN_QWEN3_235B_TWO,RUN_QWEN3_30B_TWO \
  --group 4:RUN_DEEPSEEK_FOUR,RUN_QWEN3_235B_FOUR,RUN_QWEN3_30B_FOUR
```

To compare even and skewed token distributions across expert counts:

```bash
.venv/bin/python benchmark/deepgemm_moe/plot_deepgemm_moe_distribution_scaling.py \
  --group 2:even:RUN_DEEPSEEK_2_EVEN,RUN_QWEN3_235B_2_EVEN,RUN_QWEN3_30B_2_EVEN \
  --group 2:skewed:RUN_DEEPSEEK_2_SKEW,RUN_QWEN3_235B_2_SKEW,RUN_QWEN3_30B_2_SKEW \
  --group 4:even:RUN_DEEPSEEK_4_EVEN,RUN_QWEN3_235B_4_EVEN,RUN_QWEN3_30B_4_EVEN \
  --group 4:skewed:RUN_DEEPSEEK_4_SKEW,RUN_QWEN3_235B_4_SKEW,RUN_QWEN3_30B_4_SKEW
```
