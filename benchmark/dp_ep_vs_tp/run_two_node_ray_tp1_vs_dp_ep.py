#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run two-node Ray online benchmarks for TP1x2 vs DP+EP2.

Assumes a Ray cluster is already running across two one-GPU nodes and this
script is launched from the head node.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from benchmark.dp_ep_vs_tp.results_layout import build_result_root, write_run_readme
except ModuleNotFoundError:
    from results_layout import build_result_root, write_run_readme


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


MODEL = env("MODEL", "allenai/OLMoE-1B-7B-0924-Instruct")
SERVED_MODEL_NAME = env("SERVED_MODEL_NAME", "bench-model")
HOST = env("HOST", "127.0.0.1")
BASE_PORT = int(env("BASE_PORT", "8200"))
PROMPTS_PER_GPU = int(env("PROMPTS_PER_GPU", "1000"))
NUM_PROMPTS = env("NUM_PROMPTS", str(PROMPTS_PER_GPU * 2))
INPUT_LEN = env("INPUT_LEN", "1")
OUTPUT_LEN = env("OUTPUT_LEN", "256")
NUM_WARMUPS = int(env("NUM_WARMUPS", "100"))
if NUM_WARMUPS < 0:
    raise SystemExit("NUM_WARMUPS must be non-negative")
REQUEST_RATE = env("REQUEST_RATE", "inf")
MAX_CONCURRENCY = env("MAX_CONCURRENCY", "")
MAX_MODEL_LEN = env("MAX_MODEL_LEN", "")
MOE_A2A_BACKEND = env("MOE_A2A_BACKEND", env("ALL2ALL_BACKEND", "none"))
if MOE_A2A_BACKEND == "allgather_reducescatter":
    MOE_A2A_BACKEND = "none"
RUN_ROOT = build_result_root("two_node_ray")
SERVER_START_TIMEOUT = int(env("SERVER_START_TIMEOUT", "900"))
SERVER_START_RETRIES = int(env("SERVER_START_RETRIES", "3"))
SERVER_RETRY_DELAY_SECONDS = int(env("SERVER_RETRY_DELAY_SECONDS", "10"))
PORT_RELEASE_TIMEOUT = int(env("PORT_RELEASE_TIMEOUT", "60"))
SERVER_EXTRA_ARGS = shlex.split(env("SERVER_EXTRA_ARGS", "--dtype float16"))
BENCH_EXTRA_ARGS = shlex.split(env("BENCH_EXTRA_ARGS", ""))
DISABLE_PREFIX_CACHING = env("DISABLE_PREFIX_CACHING", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
PYTHON_BIN = env("PYTHON_BIN", sys.executable)
RUN_NOTES = env("RUN_NOTES", "")
FIX_NOTES = env("FIX_NOTES", "")

SERVER_LOG_DIR = RUN_ROOT / "server_logs"
BENCH_LOG_DIR = RUN_ROOT / "bench_logs"
JSON_DIR = RUN_ROOT / "json"

server_proc: subprocess.Popen[bytes] | None = None


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(
            f"{command} command not found. Activate/install the environment first."
        )


def require_python() -> None:
    python_path = Path(PYTHON_BIN)
    if not python_path.is_file() and shutil.which(PYTHON_BIN) is None:
        raise SystemExit(
            f"{PYTHON_BIN} is not executable. Set PYTHON_BIN to the Python "
            "executable for this SGLang checkout."
        )


def cleanup_server() -> None:
    global server_proc
    proc = server_proc
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    server_proc = None


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((HOST, port)) == 0


def wait_for_port_to_close(port: int) -> None:
    deadline = time.monotonic() + PORT_RELEASE_TIMEOUT
    while time.monotonic() < deadline:
        if not is_port_open(port):
            return
        time.sleep(1)
    raise TimeoutError(
        f"Timed out waiting for port {port} on {HOST} to close. "
        "A stale API server may still be running."
    )


def handle_signal(_signum: int, _frame: object) -> None:
    cleanup_server()
    raise SystemExit(128 + int(signal.SIGTERM))


def wait_for_server(port: int, log_file: Path) -> None:
    assert server_proc is not None
    health_url = f"http://{HOST}:{port}/health"
    deadline = time.monotonic() + SERVER_START_TIMEOUT
    while time.monotonic() < deadline:
        if server_proc.poll() is not None:
            raise RuntimeError(f"Server exited before becoming ready. Log: {log_file}")
        try:
            with urlopen(health_url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except URLError:
            pass
        except TimeoutError:
            pass
        time.sleep(5)
    raise TimeoutError(
        f"Timed out waiting for server on {HOST}:{port}. Log: {log_file}"
    )


def should_retry_server_start(log_file: Path) -> bool:
    if not log_file.is_file():
        return False
    log_text = log_file.read_text(encoding="utf-8", errors="replace")
    return "AssertionError: No GPU found in Ray cluster." in log_text


def run_case(case_name: str, port: int, server_args: list[str]) -> None:
    global server_proc

    server_log = SERVER_LOG_DIR / f"{case_name}.log"
    bench_log = BENCH_LOG_DIR / f"{case_name}.log"
    result_json = JSON_DIR / f"{case_name}.json"

    print(f"=== {case_name} on Ray DP=2 port {port} ===", flush=True)
    cleanup_server()
    wait_for_port_to_close(port)

    server_cmd = [
        PYTHON_BIN,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        HOST,
        "--port",
        str(port),
        "--use-ray",
        *server_args,
        *SERVER_EXTRA_ARGS,
    ]
    if MAX_MODEL_LEN:
        server_cmd.extend(["--context-length", MAX_MODEL_LEN])
    if DISABLE_PREFIX_CACHING:
        server_cmd.append("--disable-radix-cache")
    server_env = {**os.environ}
    last_error: Exception | None = None
    for attempt in range(1, SERVER_START_RETRIES + 1):
        with server_log.open("wb") as log:
            server_proc = subprocess.Popen(
                server_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=server_env,
            )

        try:
            wait_for_server(port, server_log)
            break
        except RuntimeError as exc:
            last_error = exc
            cleanup_server()
            wait_for_port_to_close(port)
            if attempt == SERVER_START_RETRIES or not should_retry_server_start(
                server_log
            ):
                raise
            print(
                f"Server startup hit transient Ray GPU discovery failure on attempt "
                f"{attempt}/{SERVER_START_RETRIES}; retrying in "
                f"{SERVER_RETRY_DELAY_SECONDS}s.",
                flush=True,
            )
            time.sleep(SERVER_RETRY_DELAY_SECONDS)
    else:
        if last_error is not None:
            raise last_error

    bench_cmd = [
        PYTHON_BIN,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--model",
        MODEL,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--tokenizer",
        MODEL,
        "--host",
        HOST,
        "--port",
        str(port),
        "--dataset-name",
        "random",
        "--random-input-len",
        INPUT_LEN,
        "--random-output-len",
        OUTPUT_LEN,
        "--num-prompts",
        NUM_PROMPTS,
        "--warmup-requests",
        str(NUM_WARMUPS),
        "--request-rate",
        REQUEST_RATE,
        "--output-file",
        str(result_json),
        "--tag",
        case_name,
        *BENCH_EXTRA_ARGS,
    ]
    if MAX_CONCURRENCY:
        bench_cmd.extend(["--max-concurrency", MAX_CONCURRENCY])

    try:
        with bench_log.open("wb") as log:
            subprocess.run(
                bench_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    finally:
        cleanup_server()

    if not result_json.is_file() or result_json.stat().st_size == 0:
        raise RuntimeError(f"Missing benchmark result JSON: {result_json}")
    annotate_result_json(result_json, case_name)


def annotate_result_json(result_json: Path, case_name: str) -> None:
    import json

    rows = []
    with result_json.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            record.update(
                {
                    "case": case_name,
                    "model": MODEL,
                    "gpu_count": 2,
                    "num_prompts": int(NUM_PROMPTS),
                }
            )
            record["metadata"] = {
                "case": case_name,
                "model": MODEL,
                "gpu_count": 2,
                "num_prompts": NUM_PROMPTS,
                "num_warmups": NUM_WARMUPS,
                "prompts_per_gpu": PROMPTS_PER_GPU,
                "nodes": 2,
                "gpus_per_node": 1,
                "input_len": INPUT_LEN,
                "output_len": OUTPUT_LEN,
            }
            rows.append(record)
    with result_json.open("w", encoding="utf-8") as file:
        for record in rows:
            file.write(json.dumps(record) + "\n")


def summarize_results() -> None:
    python_path = Path(PYTHON_BIN)
    if not python_path.is_file() and shutil.which(PYTHON_BIN) is None:
        print(f"Skipping summary: {PYTHON_BIN} is not executable.", file=sys.stderr)
        return
    summary_path = RUN_ROOT / "summary.csv"
    subprocess.run(
        [
            PYTHON_BIN,
            "benchmark/dp_ep_vs_tp/summarize_results.py",
            str(JSON_DIR),
            "--output",
            str(summary_path),
        ],
        check=True,
    )
    print(f"Summary: {summary_path}")


def write_run_summary(*, status: str, started_at: str,
                      completed_at: str | None = None,
                      failure_reason: str | None = None) -> None:
    write_run_readme(
        RUN_ROOT,
        title="Two-node Ray TP1x2 vs DP+EP Run",
        script_path="benchmark/dp_ep_vs_tp/run_two_node_ray_tp1_vs_dp_ep.py",
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        setup={
            "model": MODEL,
            "served_model_name": SERVED_MODEL_NAME,
            "host": HOST,
            "base_port": str(BASE_PORT),
            "num_prompts": NUM_PROMPTS,
            "prompts_per_gpu": str(PROMPTS_PER_GPU),
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
            "num_warmups": str(NUM_WARMUPS),
            "request_rate": REQUEST_RATE,
            "max_concurrency": MAX_CONCURRENCY or "unset",
            "max_model_len": MAX_MODEL_LEN or "SGLang model default",
            "moe_a2a_backend": MOE_A2A_BACKEND,
            "server_start_timeout": str(SERVER_START_TIMEOUT),
            "server_start_retries": str(SERVER_START_RETRIES),
            "server_retry_delay_seconds": str(SERVER_RETRY_DELAY_SECONDS),
            "port_release_timeout": str(PORT_RELEASE_TIMEOUT),
            "server_extra_args": " ".join(SERVER_EXTRA_ARGS) or "(none)",
            "bench_extra_args": " ".join(BENCH_EXTRA_ARGS) or "(none)",
            "disable_prefix_caching": str(DISABLE_PREFIX_CACHING).lower(),
            "python_bin": PYTHON_BIN,
            "smoke_run": str(RUN_ROOT.name.endswith("_smoke")).lower(),
        },
        planned_cases=["tp1x2", "dp2_ep"],
        artifact_paths={
            "server logs": "server_logs/",
            "bench logs": "bench_logs/",
            "json results": "json/",
            "summary": "summary.csv",
        },
        failure_reason=failure_reason,
        run_notes=RUN_NOTES,
        fix_notes=FIX_NOTES,
    )


def main() -> int:
    require_python()
    SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")
    write_run_summary(status="running", started_at=started_at)

    signal.signal(signal.SIGTERM, handle_signal)
    try:
        print(
            "Detected benchmark matrix: "
            "tp1x2 baseline vs dp2_ep, SGLang Ray backend",
            flush=True,
        )
        run_case(
            case_name="tp1x2",
            port=BASE_PORT,
            server_args=["--tp-size", "1", "--dp-size", "2"],
        )
        run_case(
            case_name="dp2_ep",
            port=BASE_PORT + 1,
            server_args=[
                "--tp-size",
                "2",
                "--dp-size",
                "2",
                "--ep-size",
                "2",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                MOE_A2A_BACKEND,
            ],
        )
        summarize_results()
        write_run_summary(
            status="completed",
            started_at=started_at,
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        print(f"Results: {RUN_ROOT}")
        return 0
    except Exception:
        write_run_summary(
            status="failed",
            started_at=started_at,
            completed_at=datetime.now().isoformat(timespec="seconds"),
            failure_reason=traceback.format_exc(),
        )
        raise
    finally:
        cleanup_server()


if __name__ == "__main__":
    raise SystemExit(main())
