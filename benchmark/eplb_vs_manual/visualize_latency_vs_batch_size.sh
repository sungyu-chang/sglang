#!/usr/bin/env bash
set -euo pipefail

CSV_PATH=${1:-results/eplb_vs_manual/latency_vs_batch_size.csv}
OUT_DIR=${2:-results/eplb_vs_manual/plots}
DUCKDB_BIN=${DUCKDB:-duckdb}
UPLOT_BIN=${UPLOT:-}
UPLOT_GEM_HOME=

usage() {
  cat <<EOF
Usage: $0 [csv_path] [out_dir]

Summarize and plot latency-vs-batch-size benchmark results.

Defaults:
  csv_path: results/eplb_vs_manual/latency_vs_batch_size.csv
  out_dir:  results/eplb_vs_manual/plots

Outputs:
  - DuckDB summary tables on stdout
  - Miniplot HTML files under out_dir
  - Combined Plotly HTML files under out_dir, one per model
  - Combined YouPlot terminal charts if uplot is installed

Environment:
  DUCKDB=/path/to/duckdb overrides the DuckDB executable
  UPLOT=/path/to/uplot overrides the YouPlot executable
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v "$DUCKDB_BIN" >/dev/null 2>&1; then
  echo "error: duckdb command not found. Set DUCKDB=/path/to/duckdb or add it to PATH." >&2
  exit 1
fi

if [[ ! -f "$CSV_PATH" ]]; then
  echo "error: CSV not found: $CSV_PATH" >&2
  exit 1
fi

sql_quote() {
  local value=${1//\'/\'\'}
  printf "'%s'" "$value"
}

slugify() {
  printf "%s" "$1" | tr -c "A-Za-z0-9_" "_" | sed "s/_*$//"
}

resolve_uplot() {
  if [[ -n "$UPLOT_BIN" ]]; then
    command -v "$UPLOT_BIN"
    return
  fi

  if command -v uplot >/dev/null 2>&1; then
    command -v uplot
    return
  fi

  local candidate
  for candidate in \
    "$HOME/miniforge3/share/rubygems/bin/uplot" \
    "$HOME/.gem/ruby"/*"/bin/uplot"; do
    if [[ -x "$candidate" ]]; then
      printf "%s\n" "$candidate"
      return
    fi
  done
}

run_uplot() {
  if [[ -n "$UPLOT_GEM_HOME" ]]; then
    GEM_HOME="$UPLOT_GEM_HOME" GEM_PATH="$UPLOT_GEM_HOME" "$UPLOT_BIN" "$@"
  else
    "$UPLOT_BIN" "$@"
  fi
}

write_combined_html() {
  local benchmark=$1
  local gemm=$2
  local model=$3
  local benchmark_sql gemm_sql model_sql title title_sql title_json traces_js slug html_path

  benchmark_sql=$(sql_quote "$benchmark")
  gemm_sql=$(sql_quote "$gemm")
  model_sql=$(sql_quote "$model")
  title="$benchmark / $gemm / $model latency vs batch size by kernel"
  title_sql=$(sql_quote "$title")
  title_json=$("$DUCKDB_BIN" -no-init -noheader -list -separator "" -c "
SELECT to_json($title_sql::VARCHAR);
")
  traces_js=$("$DUCKDB_BIN" -no-init -noheader -list -separator "" -c "
WITH traces AS (
  SELECT
    backend,
    to_json(list(batch_size ORDER BY batch_size)) AS x_json,
    to_json(list(latency_ms ORDER BY batch_size)) AS y_json
  FROM $RESULTS_SQL
  WHERE benchmark = $benchmark_sql
    AND gemm = $gemm_sql
    AND model = $model_sql
  GROUP BY backend
)
SELECT string_agg(
  '  { name: ' || to_json(backend) ||
  ', x: ' || x_json ||
  ', y: ' || y_json ||
  ', type: ''scatter'', mode: ''lines'', line: { width: 2 } }',
  ',' || chr(10)
  ORDER BY backend
)
FROM traces;
")
  traces_js=${traces_js//\\n/$'\n'}
  slug=$(slugify "${benchmark}__${gemm}__${model}__kernels")
  html_path="$OUT_DIR/${slug}.html"

  cat > "$html_path" <<EOF
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Latency vs Batch Size</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; margin: 0; background: #f5f5f5; padding: 20px; }
    .container { max-width: 1400px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    h1 { margin: 0 0 20px; color: #333; font-size: 28px; text-align: center; }
    #chart { width: 100%; height: 650px; }
  </style>
</head>
<body>
  <div class="container">
    <h1 id="page-title">Latency vs Batch Size</h1>
    <div id="chart"></div>
  </div>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <script>
    var title = $title_json;
    document.title = title;
    document.getElementById('page-title').textContent = title;

    var data = [
$traces_js
    ];
    var layout = {
      xaxis: { title: 'Batch size', showgrid: true, gridcolor: '#e5e5e5' },
      yaxis: { title: 'Latency ms', showgrid: true, gridcolor: '#e5e5e5' },
      legend: { orientation: 'h', y: 1.08, x: 0.5, xanchor: 'center' },
      plot_bgcolor: '#fff',
      paper_bgcolor: '#fff',
      margin: { t: 50, r: 40, b: 60, l: 70 },
      autosize: true
    };
    Plotly.newPlot('chart', data, layout, { responsive: true, displayModeBar: true });
    window.addEventListener('resize', function() { Plotly.Plots.resize('chart'); });
  </script>
</body>
</html>
EOF

  printf "%s\n" "$html_path"
}

write_operation_html() {
  local model=$1
  local backend=$2
  local model_sql backend_sql title title_sql title_json traces_js slug html_path

  model_sql=$(sql_quote "$model")
  backend_sql=$(sql_quote "$backend")
  title="$model / $backend GEMM operation latency vs batch size"
  title_sql=$(sql_quote "$title")
  title_json=$("$DUCKDB_BIN" -no-init -noheader -list -separator "" -c "
SELECT to_json($title_sql::VARCHAR);
")
  traces_js=$("$DUCKDB_BIN" -no-init -noheader -list -separator "" -c "
WITH traces AS (
  SELECT
    benchmark || '/' || gemm AS operation,
    to_json(list(batch_size ORDER BY batch_size)) AS x_json,
    to_json(list(latency_ms ORDER BY batch_size)) AS y_json
  FROM $RESULTS_SQL
  WHERE model = $model_sql
    AND backend = $backend_sql
    AND benchmark IN ('single_gemm', 'fused_moe_gemm')
  GROUP BY operation
)
SELECT string_agg(
  '  { name: ' || to_json(operation) ||
  ', x: ' || x_json ||
  ', y: ' || y_json ||
  ', type: ''scatter'', mode: ''lines'', line: { width: 2 } }',
  ',' || chr(10)
  ORDER BY operation
)
FROM traces;
")
  traces_js=${traces_js//\\n/$'\n'}
  slug=$(slugify "gemm_operations__${model}__${backend}")
  html_path="$OUT_DIR/${slug}.html"

  cat > "$html_path" <<EOF
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GEMM Operation Latency vs Batch Size</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; margin: 0; background: #f5f5f5; padding: 20px; }
    .container { max-width: 1400px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    h1 { margin: 0 0 20px; color: #333; font-size: 28px; text-align: center; }
    #chart { width: 100%; height: 650px; }
  </style>
</head>
<body>
  <div class="container">
    <h1 id="page-title">GEMM Operation Latency vs Batch Size</h1>
    <div id="chart"></div>
  </div>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <script>
    var title = $title_json;
    document.title = title;
    document.getElementById('page-title').textContent = title;

    var data = [
$traces_js
    ];
    var layout = {
      xaxis: { title: 'Batch size', showgrid: true, gridcolor: '#e5e5e5' },
      yaxis: { title: 'Latency ms', showgrid: true, gridcolor: '#e5e5e5' },
      legend: { orientation: 'h', y: 1.08, x: 0.5, xanchor: 'center' },
      plot_bgcolor: '#fff',
      paper_bgcolor: '#fff',
      margin: { t: 50, r: 40, b: 60, l: 70 },
      autosize: true
    };
    Plotly.newPlot('chart', data, layout, { responsive: true, displayModeBar: true });
    window.addEventListener('resize', function() { Plotly.Plots.resize('chart'); });
  </script>
</body>
</html>
EOF

  printf "%s\n" "$html_path"
}

CSV_SQL=$(sql_quote "$CSV_PATH")
if head -n 1 "$CSV_PATH" | tr "," "\n" | grep -qx "benchmark"; then
  RESULTS_SQL="read_csv_auto($CSV_SQL)"
else
  RESULTS_SQL="(
    SELECT
      'fused_moe' AS benchmark,
      'moe' AS gemm,
      batch_size AS gemm_m,
      0 AS gemm_n,
      0 AS gemm_k,
      *
    FROM read_csv_auto($CSV_SQL)
  )"
fi
mkdir -p "$OUT_DIR"

if UPLOT_BIN=$(resolve_uplot); then
  case "$UPLOT_BIN" in
    */share/rubygems/bin/uplot|*/share/rubygems/bin/youplot)
      UPLOT_GEM_HOME=${UPLOT_BIN%/bin/*}
      ;;
    */.gem/ruby/*/bin/uplot|*/.gem/ruby/*/bin/youplot)
      UPLOT_GEM_HOME=${UPLOT_BIN%/bin/*}
      ;;
  esac
else
  UPLOT_BIN=
fi

echo "== Result summary =="
"$DUCKDB_BIN" -no-init -c "
SELECT
  benchmark,
  gemm,
  model,
  backend,
  min(batch_size) AS min_batch,
  max(batch_size) AS max_batch,
  count(*) AS rows,
  round(min(latency_ms), 6) AS min_latency_ms,
  round(avg(latency_ms), 6) AS avg_latency_ms,
  round(max(latency_ms), 6) AS max_latency_ms
FROM $RESULTS_SQL
GROUP BY 1, 2, 3, 4
ORDER BY 1, 2, 3, 4;
"

echo
echo "== Fastest backend by model =="
"$DUCKDB_BIN" -no-init -c "
WITH per_backend AS (
  SELECT benchmark, gemm, model, backend, avg(latency_ms) AS avg_latency_ms
  FROM $RESULTS_SQL
  GROUP BY 1, 2, 3, 4
),
ranked AS (
  SELECT
    *,
    row_number() OVER (
      PARTITION BY benchmark, gemm, model ORDER BY avg_latency_ms
    ) AS rank
  FROM per_backend
)
SELECT
  benchmark,
  gemm,
  model,
  backend AS fastest_backend,
  round(avg_latency_ms, 6) AS avg_latency_ms
FROM ranked
WHERE rank = 1
ORDER BY benchmark, gemm, model;
"

echo
echo "== Writing Miniplot HTML charts =="
PAIRS=$("$DUCKDB_BIN" -no-init -csv -c "
SELECT DISTINCT benchmark, gemm, model, backend
FROM $RESULTS_SQL
ORDER BY 1, 2, 3, 4;
")

printf "%s\n" "$PAIRS" | tail -n +2 | while IFS=, read -r benchmark gemm model backend; do
  [[ -n "$benchmark" && -n "$gemm" && -n "$model" && -n "$backend" ]] || continue

  benchmark_sql=$(sql_quote "$benchmark")
  gemm_sql=$(sql_quote "$gemm")
  model_sql=$(sql_quote "$model")
  backend_sql=$(sql_quote "$backend")
  title_sql=$(sql_quote "$benchmark / $gemm / $model / $backend latency vs batch size")
  slug=$(slugify "${benchmark}__${gemm}__${model}__${backend}")
  html_path="$OUT_DIR/${slug}.html"
  html_sql=$(sql_quote "$html_path")

  "$DUCKDB_BIN" -no-init -csv -c "
LOAD miniplot;
SELECT line_chart(
  list(batch_size::VARCHAR ORDER BY batch_size),
  list(latency_ms ORDER BY batch_size),
  $title_sql,
  $html_sql
) AS html_path
FROM $RESULTS_SQL
WHERE benchmark = $benchmark_sql
  AND gemm = $gemm_sql
  AND model = $model_sql
  AND backend = $backend_sql;
" | tail -n +2
done

echo
echo "== Writing combined kernel comparison HTML charts =="
SERIES_GROUPS=$("$DUCKDB_BIN" -no-init -csv -c "
SELECT DISTINCT benchmark, gemm, model
FROM $RESULTS_SQL
ORDER BY 1, 2, 3;
")

printf "%s\n" "$SERIES_GROUPS" | tail -n +2 | while IFS=, read -r benchmark gemm model; do
  [[ -n "$benchmark" && -n "$gemm" && -n "$model" ]] || continue
  write_combined_html "$benchmark" "$gemm" "$model"
done

echo
echo "== Writing GEMM operation comparison HTML charts =="
OPERATION_GROUPS=$("$DUCKDB_BIN" -no-init -csv -c "
SELECT DISTINCT model, backend
FROM $RESULTS_SQL
WHERE benchmark IN ('single_gemm', 'fused_moe_gemm')
ORDER BY 1, 2;
")

printf "%s\n" "$OPERATION_GROUPS" | tail -n +2 | while IFS=, read -r model backend; do
  [[ -n "$model" && -n "$backend" ]] || continue
  write_operation_html "$model" "$backend"
done

if [[ -n "$UPLOT_BIN" ]]; then
  echo
  echo "== YouPlot combined terminal charts ($UPLOT_BIN) =="
  printf "%s\n" "$SERIES_GROUPS" | tail -n +2 | while IFS=, read -r benchmark gemm model; do
    [[ -n "$benchmark" && -n "$gemm" && -n "$model" ]] || continue

    benchmark_sql=$(sql_quote "$benchmark")
    gemm_sql=$(sql_quote "$gemm")
    model_sql=$(sql_quote "$model")
    title="$benchmark / $gemm / $model latency vs batch size by kernel"

    echo
    echo "-- $title --"
    "$DUCKDB_BIN" -no-init -c "
COPY (
  SELECT
    batch_size,
    max(CASE WHEN backend = 'deep_gemm' THEN latency_ms END) AS deep_gemm,
    max(CASE WHEN backend = 'triton' THEN latency_ms END) AS triton
  FROM $RESULTS_SQL
  WHERE benchmark = $benchmark_sql
    AND gemm = $gemm_sql
    AND model = $model_sql
  GROUP BY batch_size
  ORDER BY batch_size
) TO '/dev/stdout' WITH (FORMAT csv, HEADER);
" | run_uplot lines -d, -H --fmt xyy -t "$title" --xlabel batch_size --ylabel latency_ms -o /dev/stdout
  done
else
  echo
  echo "uplot not found; skipped YouPlot terminal charts."
  echo "Tip: install YouPlot and either add uplot to PATH or run with UPLOT=/path/to/uplot."
fi

INDEX_PATH="$OUT_DIR/index.html"
{
  cat <<EOF
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Latency vs Batch Size</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.4; }
    h1 { font-size: 1.5rem; }
    h2 { font-size: 1.1rem; margin-top: 1.6rem; }
    li { margin: 0.4rem 0; }
  </style>
</head>
<body>
  <h1>Latency vs Batch Size</h1>
  <h2>Combined Kernel Comparisons</h2>
  <ul>
EOF

  for html_file in "$OUT_DIR"/*__kernels.html; do
    [[ -e "$html_file" ]] || continue
    label=$(basename "$html_file" .html)
    printf '    <li><a href="%s">%s</a></li>\n' "$(basename "$html_file")" "$label"
  done

  cat <<EOF
  </ul>
  <h2>GEMM Operation Comparisons</h2>
  <ul>
EOF

  for html_file in "$OUT_DIR"/gemm_operations__*.html; do
    [[ -e "$html_file" ]] || continue
    label=$(basename "$html_file" .html)
    printf '    <li><a href="%s">%s</a></li>\n' "$(basename "$html_file")" "$label"
  done

  cat <<EOF
  </ul>
  <h2>Individual Kernel Charts</h2>
  <ul>
EOF

  for html_file in "$OUT_DIR"/*.html; do
    [[ -e "$html_file" ]] || continue
    [[ "$(basename "$html_file")" == "index.html" ]] && continue
    [[ "$(basename "$html_file")" == *__kernels.html ]] && continue
    [[ "$(basename "$html_file")" == gemm_operations__*.html ]] && continue
    label=$(basename "$html_file" .html)
    printf '    <li><a href="%s">%s</a></li>\n' "$(basename "$html_file")" "$label"
  done

  cat <<EOF
  </ul>
</body>
</html>
EOF
} > "$INDEX_PATH"

echo
echo "Done. Open the HTML charts in: $OUT_DIR"
echo "Index: $INDEX_PATH"
