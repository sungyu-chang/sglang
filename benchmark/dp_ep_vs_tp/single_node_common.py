#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for one-node online serving benchmarks."""

from __future__ import annotations

import csv
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

try:
    from results_layout import build_result_root
except ModuleNotFoundError:
    from benchmark.dp_ep_vs_tp.results_layout import build_result_root


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_list(name: str, default: str) -> list[int]:
    raw = env(name, default)
    try:
        return [int(item) for item in raw.split()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be a space-separated list of integers") from exc


def shlex_env(name: str, default: str = "") -> list[str]:
    return shlex.split(env(name, default))


def with_default_flag(args: list[str], flag: str) -> list[str]:
    if flag in args:
        return args
    return [flag, *args]


def moe_a2a_backend_from_env(default: str = "none") -> str:
    """Return the SGLang MoE A2A backend, accepting the old vLLM env name."""
    backend = env("MOE_A2A_BACKEND", env("ALL2ALL_BACKEND", default)).strip()
    if backend == "allgather_reducescatter":
        return "none"
    return backend


def parse_cuda_visible_devices() -> list[str] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() in ("", "-1", "NoDevFiles"):
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def detect_gpu_ids() -> list[str]:
    visible_ids = parse_cuda_visible_devices()
    if "GPU_COUNT" in os.environ:
        try:
            count = int(os.environ["GPU_COUNT"])
        except ValueError as exc:
            raise SystemExit("GPU_COUNT must be an integer") from exc
        if count < 1:
            raise SystemExit("GPU_COUNT must be >= 1")
        if visible_ids is not None:
            if count > len(visible_ids):
                raise SystemExit(
                    "GPU_COUNT exceeds CUDA_VISIBLE_DEVICES length: "
                    f"{count} > {len(visible_ids)}"
                )
            return visible_ids[:count]
        return [str(index) for index in range(count)]

    if visible_ids is not None:
        return visible_ids

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise SystemExit(
            "Unable to detect GPU count: set GPU_COUNT or install nvidia-smi."
        )

    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Unable to detect GPU count from nvidia-smi; set GPU_COUNT."
        ) from exc

    count = sum(1 for line in result.stdout.splitlines() if line.startswith("GPU "))
    if count < 1:
        raise SystemExit("No GPUs detected; set GPU_COUNT to override.")
    return [str(index) for index in range(count)]


def default_tp_sizes(gpu_count: int) -> list[int]:
    sizes: list[int] = []
    size = 1
    while size <= gpu_count:
        sizes.append(size)
        size *= 2
    return sizes


def default_dp_sizes(gpu_count: int) -> list[int]:
    return list(range(1, gpu_count + 1))


def configured_sizes(name: str, default_sizes: list[int]) -> list[int]:
    if name in os.environ:
        return env_list(name, "")
    return default_sizes


def validate_sizes(name: str, sizes: list[int], gpu_count: int) -> None:
    if not sizes:
        raise SystemExit(f"{name} must not be empty")
    invalid = [size for size in sizes if size < 1 or size > gpu_count]
    if invalid:
        raise SystemExit(
            f"{name} contains sizes outside available GPU count "
            f"{gpu_count}: {invalid}"
        )


def default_result_root(name: str = "one_node_online") -> Path:
    return build_result_root(name or "one_node_online")


@dataclass
class SingleNodeBenchmarkConfig:
    model: str
    served_model_name: str = "bench-model"
    host: str = "127.0.0.1"
    base_port: int = 8100
    gpu_ids: list[str] = field(default_factory=detect_gpu_ids)
    num_prompts: str = ""
    prompts_per_gpu: int = 1000
    input_len: str = "1"
    output_len: str = "256"
    num_warmups: int = 100
    request_rate: str = "inf"
    max_concurrency: str = ""
    max_concurrency_per_gpu: str = ""
    max_model_len: str = ""
    result_root: Path = field(default_factory=default_result_root)
    server_start_timeout: int = 900
    server_extra_args: list[str] = field(default_factory=list)
    bench_extra_args: list[str] = field(default_factory=list)
    python_bin: str = sys.executable
    profile_modules: bool = False
    profile_delay_iterations: int = 5
    profile_max_iterations: int = 20
    profile_with_stack: bool = False
    profile_layer_scopes: bool = False
    disable_prefix_caching: bool = True
    port_release_timeout: int = 60

    def __post_init__(self) -> None:
        if self.num_warmups < 0:
            raise SystemExit("NUM_WARMUPS must be non-negative")

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_ids)

    def num_prompts_for_gpu_count(self, gpu_count: int) -> str:
        return str(self.num_prompts_count_for_gpu_count(gpu_count))

    def num_prompts_count_for_gpu_count(self, gpu_count: int) -> int:
        if self.num_prompts:
            explicit_num_prompts = int(self.num_prompts)
            if explicit_num_prompts <= 0:
                raise SystemExit("NUM_PROMPTS must be positive when set")
            return explicit_num_prompts
        if self.prompts_per_gpu <= 0:
            raise SystemExit("PROMPTS_PER_GPU must be positive")
        return self.prompts_per_gpu * gpu_count

    def max_concurrency_for_gpu_count(self, gpu_count: int) -> str:
        max_concurrency = self.max_concurrency_count_for_gpu_count(gpu_count)
        return str(max_concurrency) if max_concurrency is not None else ""

    def max_concurrency_count_for_gpu_count(self, gpu_count: int) -> int | None:
        if self.max_concurrency:
            max_concurrency = int(self.max_concurrency)
            if max_concurrency <= 0:
                raise SystemExit("MAX_CONCURRENCY must be positive when set")
            return max_concurrency
        if self.max_concurrency_per_gpu:
            per_gpu = int(self.max_concurrency_per_gpu)
            if per_gpu <= 0:
                raise SystemExit(
                    "MAX_CONCURRENCY_PER_GPU must be positive when set"
                )
            return per_gpu * gpu_count
        return None

    def validate_case_concurrency(self, case_name: str, gpu_count: int) -> None:
        max_concurrency = self.max_concurrency_count_for_gpu_count(gpu_count)
        if max_concurrency is None:
            return
        num_prompts = self.num_prompts_count_for_gpu_count(gpu_count)
        if max_concurrency <= num_prompts:
            raise SystemExit(
                f"{case_name}: configured max concurrency ({max_concurrency}) "
                f"must be greater than num_prompts ({num_prompts}). "
                "Increase MAX_CONCURRENCY/MAX_CONCURRENCY_PER_GPU or reduce "
                "NUM_PROMPTS/PROMPTS_PER_GPU."
            )


class SingleNodeBenchmarkRunner:
    def __init__(self, config: SingleNodeBenchmarkConfig) -> None:
        self.config = config
        self.server_log_dir = config.result_root / "server_logs"
        self.bench_log_dir = config.result_root / "bench_logs"
        self.json_dir = config.result_root / "json"
        self.profile_dir = config.result_root / "profiler_traces"
        self.server_proc: subprocess.Popen[bytes] | None = None

    def require_command(self, command: str) -> None:
        if shutil.which(command) is None:
            raise SystemExit(
                f"{command} command not found. Activate/install SGLang first."
            )

    def require_python(self) -> None:
        python_path = Path(self.config.python_bin)
        if not python_path.is_file() and shutil.which(self.config.python_bin) is None:
            raise SystemExit(
                f"{self.config.python_bin} is not executable. Set PYTHON_BIN to "
                "the Python executable for this SGLang checkout."
            )

    def setup_dirs(self) -> None:
        self.server_log_dir.mkdir(parents=True, exist_ok=True)
        self.bench_log_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        if self.config.profile_modules:
            self.profile_dir.mkdir(parents=True, exist_ok=True)

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, _signum: int, _frame: object) -> None:
        self.cleanup_server()
        raise SystemExit(128 + int(signal.SIGTERM))

    def gpu_list(self, count: int) -> str:
        return ",".join(self.config.gpu_ids[:count])

    def cleanup_server(self) -> None:
        proc = self.server_proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        self.server_proc = None

    def is_port_open(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex((self.config.host, port)) == 0

    def wait_for_port_to_close(self, port: int) -> None:
        deadline = time.monotonic() + self.config.port_release_timeout
        while time.monotonic() < deadline:
            if not self.is_port_open(port):
                return
            time.sleep(1)
        raise TimeoutError(
            f"Timed out waiting for port {port} on {self.config.host} to close. "
            "A stale API server may still be running."
        )

    def wait_for_server(self, port: int, log_file: Path) -> None:
        assert self.server_proc is not None
        health_url = f"http://{self.config.host}:{port}/health"
        deadline = time.monotonic() + self.config.server_start_timeout
        while time.monotonic() < deadline:
            if self.server_proc.poll() is not None:
                raise RuntimeError(
                    f"Server exited before becoming ready. Log: {log_file}"
                )
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
            f"Timed out waiting for server on {self.config.host}:{port}. "
            f"Log: {log_file}"
        )

    def profiler_config(self, case_name: str) -> Path:
        profile_dir = self.profile_dir / case_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        return profile_dir

    def build_server_cmd(self, port: int, server_args: list[str]) -> list[str]:
        cmd = [
            self.config.python_bin,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.config.model,
            "--served-model-name",
            self.config.served_model_name,
            "--host",
            self.config.host,
            "--port",
            str(port),
        ]
        if self.config.disable_prefix_caching:
            cmd.append("--disable-radix-cache")
        if self.config.max_model_len:
            cmd.extend(["--context-length", self.config.max_model_len])
        cmd.extend(server_args)
        cmd.extend(self.config.server_extra_args)
        return cmd

    def build_bench_cmd(
        self,
        case_name: str,
        port: int,
        gpu_count: int,
        metadata: dict[str, object] | None = None,
    ) -> list[str]:
        num_prompts = self.config.num_prompts_for_gpu_count(gpu_count)
        max_concurrency = self.config.max_concurrency_for_gpu_count(gpu_count)

        cmd = [
            self.config.python_bin,
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang",
            "--model",
            self.config.model,
            "--served-model-name",
            self.config.served_model_name,
            "--tokenizer",
            self.config.model,
            "--host",
            self.config.host,
            "--port",
            str(port),
            "--dataset-name",
            "random",
            "--random-input-len",
            self.config.input_len,
            "--random-output-len",
            self.config.output_len,
            "--num-prompts",
            num_prompts,
            "--warmup-requests",
            str(self.config.num_warmups),
            "--request-rate",
            self.config.request_rate,
            "--output-file",
            str(self.json_dir / f"{case_name}.json"),
            "--tag",
            case_name,
            *self.config.bench_extra_args,
        ]
        if max_concurrency:
            cmd.extend(["--max-concurrency", max_concurrency])
        if self.config.profile_modules:
            cmd.extend(
                [
                    "--profile",
                    "--profile-output-dir",
                    str((self.profile_dir / case_name).resolve()),
                    "--profile-start-step",
                    str(self.config.profile_delay_iterations),
                    "--profile-steps",
                    str(self.config.profile_max_iterations),
                    "--profile-prefix",
                    case_name,
                ]
            )
            if self.config.profile_layer_scopes:
                cmd.append("--profile-by-stage")
        return cmd

    def annotate_result_json(
        self,
        result_json: Path,
        case_name: str,
        gpu_count: int,
        metadata: dict[str, object] | None,
    ) -> None:
        metadata = metadata or {}
        num_prompts = self.config.num_prompts_for_gpu_count(gpu_count)
        rows = []
        with result_json.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                record.update(
                    {
                        "case": case_name,
                        "model": self.config.model,
                        "gpu_count": gpu_count,
                        "num_prompts": int(num_prompts),
                    }
                )
                record["metadata"] = {
                    "case": case_name,
                    "model": self.config.model,
                    "gpu_count": gpu_count,
                    "num_prompts": num_prompts,
                    "prompts_per_gpu": self.config.prompts_per_gpu,
                    "input_len": self.config.input_len,
                    "output_len": self.config.output_len,
                    "num_warmups": self.config.num_warmups,
                    "profile_modules": self.config.profile_modules,
                    **metadata,
                }
                rows.append(record)

        with result_json.open("w", encoding="utf-8") as file:
            for record in rows:
                file.write(json.dumps(record) + "\n")

    def run_case(
        self,
        case_name: str,
        gpu_count: int,
        port: int,
        server_args: list[str],
        metadata: dict[str, object] | None = None,
    ) -> None:
        server_log = self.server_log_dir / f"{case_name}.log"
        bench_log = self.bench_log_dir / f"{case_name}.log"
        result_json = self.json_dir / f"{case_name}.json"
        cuda_devices = self.gpu_list(gpu_count)

        print(f"=== {case_name} on GPUs {cuda_devices} port {port} ===", flush=True)
        self.config.validate_case_concurrency(case_name, gpu_count)
        self.cleanup_server()
        self.wait_for_port_to_close(port)

        server_cmd = self.build_server_cmd(port, server_args)
        server_env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_devices}
        profile_dir: Path | None = None
        if self.config.profile_modules:
            profile_dir = self.profiler_config(case_name)
            server_env["SGLANG_TORCH_PROFILER_DIR"] = str(profile_dir.resolve())

        with server_log.open("wb") as log:
            self.server_proc = subprocess.Popen(
                server_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=server_env,
            )

        self.wait_for_server(port, server_log)
        bench_cmd = self.build_bench_cmd(case_name, port, gpu_count, metadata)

        try:
            with bench_log.open("wb") as log:
                subprocess.run(
                    bench_cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        finally:
            self.cleanup_server()

        if not result_json.is_file() or result_json.stat().st_size == 0:
            raise RuntimeError(f"Missing benchmark result JSON: {result_json}")
        self.annotate_result_json(result_json, case_name, gpu_count, metadata)
        if profile_dir is not None:
            print(f"Profiler: {profile_dir}", flush=True)

    def summarize_results(self) -> None:
        python_path = Path(self.config.python_bin)
        if not python_path.is_file() and shutil.which(self.config.python_bin) is None:
            print(
                f"Skipping summary: {self.config.python_bin} is not executable.",
                file=sys.stderr,
            )
            return
        summary_path = self.config.result_root / "summary.csv"
        subprocess.run(
            [
                self.config.python_bin,
                "benchmark/dp_ep_vs_tp/summarize_results.py",
                str(self.json_dir),
                "--output",
                str(summary_path),
            ],
            check=True,
        )
        print(f"Summary: {summary_path}")

    def summarize_module_profiles(self) -> Path | None:
        if not self.profile_dir.is_dir():
            return None

        rows: list[dict[str, str]] = []
        totals: dict[tuple[str, str], dict[str, float]] = {}
        for profile_file in sorted(
            self.profile_dir.glob("*/module_profiler_out_*.txt")
        ):
            case_name = profile_file.parent.name
            rank = profile_file.stem.removeprefix("module_profiler_out_")
            with profile_file.open() as file:
                for row in csv.DictReader(file):
                    row_with_case = {"case": case_name, "rank": rank, **row}
                    rows.append(row_with_case)

                    key = (case_name, row["module"])
                    total = totals.setdefault(
                        key,
                        {
                            "count": 0.0,
                            "total_cuda_ms": 0.0,
                            "total_cpu_ms": 0.0,
                        },
                    )
                    total["count"] += float(row["count"])
                    total["total_cuda_ms"] += float(row["total_cuda_ms"])
                    total["total_cpu_ms"] += float(row["total_cpu_ms"])

        for (case_name, module), total in sorted(totals.items()):
            count = max(total["count"], 1.0)
            rows.append(
                {
                    "case": case_name,
                    "rank": "all",
                    "module": module,
                    "count": f"{int(total['count'])}",
                    "total_cuda_ms": f"{total['total_cuda_ms']:.3f}",
                    "avg_cuda_ms": f"{total['total_cuda_ms'] / count:.3f}",
                    "total_cpu_ms": f"{total['total_cpu_ms']:.3f}",
                    "avg_cpu_ms": f"{total['total_cpu_ms'] / count:.3f}",
                }
            )

        if not rows:
            print(
                "Skipping module summary: no module_profiler_out_*.txt files "
                f"found under {self.profile_dir}."
            )
            return None

        summary_path = self.config.result_root / "module_summary.csv"
        fieldnames = [
            "case",
            "rank",
            "module",
            "count",
            "total_cuda_ms",
            "avg_cuda_ms",
            "total_cpu_ms",
            "avg_cpu_ms",
        ]
        with summary_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Module summary: {summary_path}")
        return summary_path
