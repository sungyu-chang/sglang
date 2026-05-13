#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one-node Qwen3-MoE DP+EP serving benchmarks under Nsight Systems."""

from __future__ import annotations

import argparse
import os
import signal
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.dp_ep_vs_tp.results_layout import build_result_root, write_run_readme
from benchmark.dp_ep_vs_tp.run_qwen3_moe_ep_pipeline import dp_ep_server_args
from benchmark.dp_ep_vs_tp.single_node_common import (
    SingleNodeBenchmarkConfig,
    SingleNodeBenchmarkRunner,
    configured_sizes,
    default_dp_sizes,
    detect_gpu_ids,
    env,
    moe_a2a_backend_from_env,
    shlex_env,
    validate_sizes,
)

DEFAULT_NSYS_BIN = "/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys"
CLI_ARGS: argparse.Namespace | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-node Qwen MoE DP+EP benchmarks under Nsight Systems."
    )
    parser.add_argument("--model")
    parser.add_argument("--server-extra-args")
    parser.add_argument("--gpu-count", type=int)
    parser.add_argument("--gpu-ids")
    parser.add_argument("--dp-sizes")
    parser.add_argument("--prompts-per-gpu", type=int)
    parser.add_argument("--input-len")
    parser.add_argument("--output-len")
    parser.add_argument("--num-warmups", type=int)
    parser.add_argument("--request-rate")
    parser.add_argument("--max-concurrency-per-gpu")
    parser.add_argument("--all2all-backend", "--moe-a2a-backend", dest="moe_a2a_backend")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--nsys-bin")
    parser.add_argument("--nsys-trace")
    parser.add_argument("--nsys-wait")
    parser.add_argument("--nsys-extra-args")
    parser.add_argument("--capture-range")
    parser.add_argument("--capture-range-end")
    parser.add_argument("--duration", type=int)
    parser.add_argument("--capture-timeout", type=int)
    parser.add_argument("--cuda-graph-trace")
    parser.add_argument("--stop-profile-timeout", type=int)
    parser.add_argument("--stats-reports")
    parser.add_argument("--report-timeout", type=int)
    parser.add_argument("--server-shutdown-timeout", type=int)
    parser.add_argument("--server-exit-timeout", type=int)
    parser.add_argument("--server-terminate-timeout", type=int)
    parser.add_argument("--flush-delay", type=int)
    return parser.parse_args()


def cli_or_env(name: str, env_name: str, default: str) -> str:
    if CLI_ARGS is not None:
        value = getattr(CLI_ARGS, name)
        if value is not None:
            return str(value)
    return env(env_name, default)


def cli_int_or_env(name: str, env_name: str, default: str) -> int:
    return int(cli_or_env(name, env_name, default))


def apply_cli_env_overrides(args: argparse.Namespace) -> None:
    overrides = {
        "model": "MODEL",
        "server_extra_args": "SERVER_EXTRA_ARGS",
        "gpu_count": "GPU_COUNT",
        "gpu_ids": "GPU_IDS",
        "dp_sizes": "DP_SIZES",
        "prompts_per_gpu": "PROMPTS_PER_GPU",
        "input_len": "INPUT_LEN",
        "output_len": "OUTPUT_LEN",
        "num_warmups": "NUM_WARMUPS",
        "request_rate": "REQUEST_RATE",
        "max_concurrency_per_gpu": "MAX_CONCURRENCY_PER_GPU",
        "moe_a2a_backend": "MOE_A2A_BACKEND",
    }
    for arg_name, env_name in overrides.items():
        value = getattr(args, arg_name)
        if value is not None:
            os.environ[env_name] = str(value)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def resolve_nsys_bin() -> str:
    configured = cli_or_env("nsys_bin", "NSYS_BIN", "")
    if configured:
        return configured
    nsys_on_path = shutil.which("nsys")
    if nsys_on_path is not None:
        return nsys_on_path
    if Path(DEFAULT_NSYS_BIN).is_file():
        return DEFAULT_NSYS_BIN
    return "nsys"


def build_result_dir() -> Path:
    return build_result_root("nsys_profiling")


def validate_enforce_eager_disabled(server_extra_args: list[str]) -> None:
    if "--enforce-eager" in server_extra_args:
        raise SystemExit(
            "Nsight Systems profiling should run with enforce eager disabled. "
            "Remove --enforce-eager from --server-extra-args."
        )


def nsys_capture_range() -> str:
    return cli_or_env("capture_range", "NSYS_CAPTURE_RANGE", "none")


def use_sglang_profile_endpoint() -> bool:
    return nsys_capture_range() == "cudaProfilerApi"


def nsys_wait_mode() -> str:
    return cli_or_env("nsys_wait", "NSYS_WAIT", "all")


def nsys_extra_args() -> list[str]:
    default_args = "--sample=none --backtrace=none --resolve-symbols=false"
    return shlex.split(cli_or_env("nsys_extra_args", "NSYS_EXTRA_ARGS", default_args))


def nsys_duration() -> int | None:
    duration = cli_int_or_env("duration", "NSYS_DURATION", "0")
    return duration if duration > 0 else None


def nsys_capture_timeout() -> int | None:
    default_timeout = "300" if use_sglang_profile_endpoint() else "0"
    timeout = cli_int_or_env("capture_timeout", "NSYS_CAPTURE_TIMEOUT", default_timeout)
    return timeout if timeout > 0 else None


def stop_profile_timeout() -> int:
    return cli_int_or_env("stop_profile_timeout", "NSYS_STOP_PROFILE_TIMEOUT", "30")


def build_config(*, result_root: Path, gpu_ids: list[str]) -> SingleNodeBenchmarkConfig:
    server_extra_args = shlex_env("SERVER_EXTRA_ARGS", "--dtype bfloat16")
    validate_enforce_eager_disabled(server_extra_args)

    return SingleNodeBenchmarkConfig(
        model=env("MODEL", "Qwen/Qwen3-30B-A3B"),
        served_model_name=env("SERVED_MODEL_NAME", "qwen3-moe-bench"),
        host=env("HOST", "127.0.0.1"),
        base_port=int(env("BASE_PORT", "8100")),
        gpu_ids=gpu_ids,
        num_prompts=env("NUM_PROMPTS", ""),
        prompts_per_gpu=int(env("PROMPTS_PER_GPU", "1000")),
        input_len=env("INPUT_LEN", "1"),
        output_len=env("OUTPUT_LEN", "256"),
        num_warmups=int(env("NUM_WARMUPS", "100")),
        request_rate=env("REQUEST_RATE", "inf"),
        max_concurrency=env("MAX_CONCURRENCY", ""),
        max_concurrency_per_gpu=env("MAX_CONCURRENCY_PER_GPU", "1001"),
        max_model_len=env("MAX_MODEL_LEN", ""),
        result_root=result_root,
        server_start_timeout=int(env("SERVER_START_TIMEOUT", "900")),
        server_extra_args=server_extra_args,
        bench_extra_args=shlex_env("BENCH_EXTRA_ARGS"),
        python_bin=env("PYTHON_BIN", ".venv/bin/python"),
        profile_modules=False,
        disable_prefix_caching=True,
        port_release_timeout=int(env("PORT_RELEASE_TIMEOUT", "60")),
    )


class NsysDpEpRunner(SingleNodeBenchmarkRunner):
    def __init__(self, config: SingleNodeBenchmarkConfig, nsys_bin: str) -> None:
        super().__init__(config)
        self.nsys_bin = nsys_bin
        self.nsys_dir = config.result_root / "nsys"
        self.nsys_stats_dir = config.result_root / "nsys_stats"
        self.server_process_group_snapshot: set[int] = set()

    def setup_dirs(self) -> None:
        super().setup_dirs()
        self.nsys_dir.mkdir(parents=True, exist_ok=True)
        self.nsys_stats_dir.mkdir(parents=True, exist_ok=True)

    def require_psutil(self) -> None:
        if psutil is None:
            raise SystemExit(
                "psutil is required for Nsight Systems process cleanup. "
                "Install it in the SGLang environment before running this harness."
            )

    def require_command(self, command: str) -> None:
        if command != "nsys":
            super().require_command(command)
        if shutil.which(self.nsys_bin) is None and not Path(self.nsys_bin).is_file():
            raise SystemExit(
                f"nsys command not found: {self.nsys_bin}. Pass --nsys-bin with the "
                "full Nsight Systems CLI path."
            )

    def build_nsys_cmd(self, case_name: str, server_cmd: list[str]) -> list[str]:
        output_base = self.nsys_dir / case_name
        cmd = [
            self.nsys_bin,
            "profile",
            "--force-overwrite=true",
            "--trace-fork-before-exec=true",
            f"--wait={nsys_wait_mode()}",
            "--output",
            str(output_base.resolve()),
            "--trace",
            cli_or_env("nsys_trace", "NSYS_TRACE", "cuda,nvtx"),
            *nsys_extra_args(),
            *server_cmd,
        ]
        cuda_graph_trace = cli_or_env("cuda_graph_trace", "NSYS_CUDA_GRAPH_TRACE", "")
        if cuda_graph_trace:
            cmd[4:4] = [f"--cuda-graph-trace={cuda_graph_trace}"]
        duration = nsys_duration()
        if duration is not None:
            cmd[4:4] = [f"--duration={duration}"]
        capture_range = nsys_capture_range()
        if capture_range != "none":
            capture_range_end = cli_or_env(
                "capture_range_end", "NSYS_CAPTURE_RANGE_END", "repeat"
            )
            cmd[4:4] = [
                f"--capture-range={capture_range}",
                f"--capture-range-end={capture_range_end}",
            ]
        return cmd

    def build_bench_cmd(
        self,
        case_name: str,
        port: int,
        gpu_count: int,
        metadata: dict[str, object] | None = None,
    ) -> list[str]:
        metadata = metadata or {}
        metadata = {
            **metadata,
            "nsys": True,
            "nsys_trace": cli_or_env("nsys_trace", "NSYS_TRACE", "cuda,nvtx"),
            "nsys_capture_range": nsys_capture_range(),
            "nsys_capture_timeout": nsys_capture_timeout() or "none",
            "nsys_duration": nsys_duration() or "none",
        }
        cmd = super().build_bench_cmd(case_name, port, gpu_count, metadata)
        if use_sglang_profile_endpoint() and "--profile" not in cmd:
            cmd.extend(["--profile", "--profile-activities", "CUDA_PROFILER"])
        return cmd

    def stop_profile(self, port: int, log) -> None:
        if not use_sglang_profile_endpoint():
            return
        profile_url = f"http://{self.config.host}:{port}/stop_profile"
        request = Request(profile_url, method="POST")
        try:
            with urlopen(request, timeout=stop_profile_timeout()) as response:
                log.write(
                    "\nForced /stop_profile returned "
                    f"HTTP {response.status}.\n".encode()
                )
        except (TimeoutError, URLError, OSError) as exc:
            log.write(
                "\nFailed to force /stop_profile before cleanup: "
                f"{exc!r}\n".encode()
            )

    def cleanup_server(self) -> None:
        proc = self.server_proc
        if proc is None:
            return

        shutdown_timeout = cli_int_or_env(
            "server_shutdown_timeout", "NSYS_SERVER_SHUTDOWN_TIMEOUT", "300"
        )
        server_exit_timeout = cli_int_or_env(
            "server_exit_timeout", "NSYS_SERVER_EXIT_TIMEOUT", "60"
        )
        flush_delay = cli_int_or_env("flush_delay", "NSYS_FLUSH_DELAY", "90")
        terminate_timeout = cli_int_or_env(
            "server_terminate_timeout", "NSYS_SERVER_TERMINATE_TIMEOUT", "60"
        )
        server_process_groups = set(self.server_process_group_snapshot)
        server_process_groups.update(self.server_process_groups(proc.pid))
        nsys_process_groups = self.nsys_process_groups(proc.pid)

        if proc.poll() is not None:
            self.terminate_process_groups(server_process_groups, terminate_timeout)
            self.server_process_group_snapshot.clear()
            self.server_proc = None
            return

        self.signal_process_groups(server_process_groups, signal.SIGINT)
        if not self.wait_for_process_groups(server_process_groups, server_exit_timeout):
            server_process_groups.update(self.server_process_groups(proc.pid))
            self.terminate_process_groups(server_process_groups, terminate_timeout)

        time.sleep(flush_delay)
        self.signal_process_groups(
            self.nsys_root_process_group(proc.pid), signal.SIGINT
        )
        try:
            proc.wait(timeout=shutdown_timeout)
        except subprocess.TimeoutExpired:
            server_process_groups.update(self.server_process_groups(proc.pid))
            self.terminate_process_groups(server_process_groups, terminate_timeout)
            try:
                proc.wait(timeout=terminate_timeout)
            except subprocess.TimeoutExpired:
                nsys_process_groups.update(self.nsys_process_groups(proc.pid))
                self.signal_process_groups(nsys_process_groups, signal.SIGTERM)
                try:
                    proc.wait(timeout=terminate_timeout)
                except subprocess.TimeoutExpired:
                    nsys_process_groups.update(self.nsys_process_groups(proc.pid))
                    self.signal_process_groups(nsys_process_groups, signal.SIGKILL)
                    proc.wait(timeout=terminate_timeout)
        finally:
            self.terminate_process_groups(server_process_groups, terminate_timeout)
            self.server_process_group_snapshot.clear()
            self.server_proc = None

    def terminate_process_groups(self, process_groups: set[int], timeout: int) -> None:
        if not process_groups:
            return
        self.signal_process_groups(process_groups, signal.SIGTERM)
        if self.wait_for_process_groups(process_groups, timeout):
            return
        self.signal_process_groups(process_groups, signal.SIGKILL)
        self.wait_for_process_groups(process_groups, timeout)

    def wait_for_process_groups(self, process_groups: set[int], timeout: int) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.process_groups_alive(process_groups):
                return True
            time.sleep(1)
        return not self.process_groups_alive(process_groups)

    def process_groups_alive(self, process_groups: set[int]) -> bool:
        current_pgid = os.getpgrp()
        for process in psutil.process_iter(["pid", "status"]):
            try:
                pgid = os.getpgid(process.info["pid"])
            except (ProcessLookupError, psutil.Error):
                continue
            if process.info.get("status") == psutil.STATUS_ZOMBIE:
                continue
            if pgid in process_groups and pgid != current_pgid:
                return True
        return False

    def nsys_root_process_group(self, pid: int) -> set[int]:
        try:
            return {os.getpgid(pid)}
        except ProcessLookupError:
            return set()

    def nsys_process_groups(self, pid: int) -> set[int]:
        process_groups = self.nsys_agent_process_groups(pid)
        try:
            process_groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
        return process_groups

    def server_process_groups(self, pid: int) -> set[int]:
        process_groups: set[int] = set()
        try:
            root = psutil.Process(pid)
            root_pgid = os.getpgid(pid)
            processes = [root, *root.children(recursive=True)]
        except psutil.Error:
            processes = []
            root_pgid = -1

        for process in processes:
            try:
                pgid = os.getpgid(process.pid)
            except ProcessLookupError:
                continue
            except psutil.Error:
                continue
            if pgid != root_pgid:
                process_groups.add(pgid)
        return process_groups

    def nsys_agent_process_groups(self, pid: int) -> set[int]:
        process_groups: set[int] = set()
        session_marker = f"profile-{pid}"
        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = process.info.get("cmdline") or []
                if not any(session_marker in arg for arg in cmdline):
                    continue
                process_groups.add(os.getpgid(process.pid))
            except (ProcessLookupError, psutil.Error):
                continue
        return process_groups

    def signal_process_groups(self, process_groups: set[int], sig: int) -> None:
        current_pgid = os.getpgrp()
        for pgid in sorted(process_groups):
            if pgid == current_pgid:
                continue
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                continue

    def wait_for_nsys_reports(self, case_name: str) -> list[Path]:
        timeout = cli_int_or_env("report_timeout", "NSYS_REPORT_TIMEOUT", "900")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            report_files = sorted(self.nsys_dir.glob(f"{case_name}*.nsys-rep"))
            if report_files:
                return report_files
            time.sleep(1)
        raise RuntimeError(f"Missing Nsight Systems report for case: {case_name}")

    def export_nsys_stats(self, case_name: str) -> None:
        reports = cli_or_env(
            "stats_reports", "NSYS_STATS_REPORTS", "cuda_gpu_kern_sum,cuda_gpu_trace"
        )
        report_files = self.wait_for_nsys_reports(case_name)

        for report_file in report_files:
            output_base = self.nsys_stats_dir / report_file.stem
            stats_log = self.nsys_stats_dir / f"{report_file.stem}.log"
            with stats_log.open("wb") as log:
                subprocess.run(
                    [
                        self.nsys_bin,
                        "stats",
                        "--force-export=true",
                        "--force-overwrite=true",
                        "--report",
                        reports,
                        "--format",
                        "csv",
                        "--output",
                        str(output_base),
                        str(report_file),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print(f"Nsight Systems stats: {output_base}_*.csv", flush=True)

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
        server_cmd = self.build_nsys_cmd(case_name, server_cmd)
        server_env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_devices}
        server_env.setdefault("SGLANG_TORCH_PROFILER_DIR", str(self.nsys_dir.resolve()))

        with server_log.open("wb") as log:
            self.server_proc = subprocess.Popen(
                server_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=server_env,
                start_new_session=True,
            )

        self.wait_for_server(port, server_log)
        self.server_process_group_snapshot = self.server_process_groups(
            self.server_proc.pid
        )
        bench_cmd = self.build_bench_cmd(case_name, port, gpu_count, metadata)

        try:
            with bench_log.open("wb") as log:
                try:
                    subprocess.run(
                        bench_cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=nsys_capture_timeout(),
                    )
                except subprocess.TimeoutExpired as exc:
                    log.write(
                        f"\nTimed out after {exc.timeout} seconds while running "
                        "the benchmark/profile capture.\n".encode()
                    )
                    self.stop_profile(port, log)
                    raise RuntimeError(
                        f"Timed out after {exc.timeout} seconds while running "
                        f"Nsight Systems ranged capture for case: {case_name}"
                    ) from exc
        finally:
            self.server_process_group_snapshot.update(
                self.server_process_groups(self.server_proc.pid)
            )
            self.cleanup_server()

        if not result_json.is_file() or result_json.stat().st_size == 0:
            raise RuntimeError(f"Missing benchmark result JSON: {result_json}")
        self.annotate_result_json(result_json, case_name, gpu_count, metadata)
        self.export_nsys_stats(case_name)
        print(f"Nsight Systems output base: {self.nsys_dir / case_name}", flush=True)


def write_run_summary(
    *,
    config: SingleNodeBenchmarkConfig,
    dp_sizes: list[int],
    nsys_bin: str,
    status: str,
    started_at: str,
    completed_at: str | None = None,
    failure_reason: str | None = None,
) -> None:
    write_run_readme(
        config.result_root,
        title="Qwen3-MoE DP+EP Nsight Systems Profiling Run",
        script_path="benchmark/dp_ep_vs_tp/run_qwen3_moe_ep_nsys.py",
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        setup={
            "model": config.model,
            "served_model_name": config.served_model_name,
            "host": config.host,
            "base_port": str(config.base_port),
            "gpu_ids": ",".join(config.gpu_ids),
            "gpu_count": str(config.gpu_count),
            "dp_sizes": " ".join(str(size) for size in dp_sizes),
            "num_prompts": config.num_prompts or "prompts_per_gpu * dp_size",
            "prompts_per_gpu": str(config.prompts_per_gpu),
            "input_len": config.input_len,
            "output_len": config.output_len,
            "num_warmups": str(config.num_warmups),
            "request_rate": config.request_rate,
            "max_concurrency": config.max_concurrency or "unset",
            "max_concurrency_per_gpu": config.max_concurrency_per_gpu or "unset",
            "max_model_len": config.max_model_len or "SGLang model default",
            "moe_a2a_backend": moe_a2a_backend_from_env(),
            "server_start_timeout": str(config.server_start_timeout),
            "port_release_timeout": str(config.port_release_timeout),
            "server_extra_args": " ".join(config.server_extra_args) or "(none)",
            "bench_extra_args": " ".join(config.bench_extra_args) or "(none)",
            "python_bin": config.python_bin,
            "nsys_bin": nsys_bin,
            "nsys_trace": cli_or_env("nsys_trace", "NSYS_TRACE", "cuda,nvtx"),
            "nsys_capture_range": nsys_capture_range(),
            "nsys_capture_range_end": cli_or_env(
                "capture_range_end", "NSYS_CAPTURE_RANGE_END", "repeat"
            ),
            "nsys_capture_timeout": str(nsys_capture_timeout() or "none"),
            "nsys_duration": str(nsys_duration() or "none"),
            "nsys_cuda_graph_trace": cli_or_env(
                "cuda_graph_trace", "NSYS_CUDA_GRAPH_TRACE", ""
            )
            or "unset",
            "nsys_wait": nsys_wait_mode(),
            "sglang_profile_endpoint": str(use_sglang_profile_endpoint()).lower(),
            "nsys_extra_args": " ".join(nsys_extra_args()) or "(none)",
            "nsys_stats_reports": cli_or_env(
                "stats_reports",
                "NSYS_STATS_REPORTS", "cuda_gpu_kern_sum,cuda_gpu_trace"
            ),
            "nsys_server_shutdown_timeout": cli_or_env(
                "server_shutdown_timeout", "NSYS_SERVER_SHUTDOWN_TIMEOUT", "300"
            ),
            "nsys_flush_delay": cli_or_env("flush_delay", "NSYS_FLUSH_DELAY", "90"),
            "nsys_server_terminate_timeout": cli_or_env(
                "server_terminate_timeout", "NSYS_SERVER_TERMINATE_TIMEOUT", "60"
            ),
            "nsys_report_timeout": cli_or_env(
                "report_timeout", "NSYS_REPORT_TIMEOUT", "900"
            ),
            "enforce_eager": "false",
            "disable_prefix_caching": str(config.disable_prefix_caching).lower(),
            "smoke_run": str(config.result_root.name.endswith("_smoke")).lower(),
        },
        planned_cases=[f"dp{dp_size}_ep" for dp_size in dp_sizes],
        artifact_paths={
            "nsys reports": "nsys/",
            "nsys stats": "nsys_stats/",
            "server logs": "server_logs/",
            "bench logs": "bench_logs/",
            "json results": "json/",
            "summary": "summary.csv",
        },
        failure_reason=failure_reason,
        run_notes=env("RUN_NOTES", ""),
        fix_notes=env("FIX_NOTES", ""),
    )


def main() -> int:
    global CLI_ARGS
    CLI_ARGS = parse_args()
    apply_cli_env_overrides(CLI_ARGS)

    gpu_ids = detect_gpu_ids()
    gpu_count = len(gpu_ids)
    dp_sizes = configured_sizes("DP_SIZES", default_dp_sizes(gpu_count))
    validate_sizes("DP_SIZES", dp_sizes, gpu_count)

    result_root = build_result_dir()
    config = build_config(result_root=result_root, gpu_ids=gpu_ids)
    nsys_bin = resolve_nsys_bin()
    runner = NsysDpEpRunner(config, nsys_bin)
    runner.require_python()
    runner.require_command("nsys")
    runner.require_psutil()
    runner.setup_dirs()
    runner.install_signal_handlers()

    started_at = datetime.now().isoformat(timespec="seconds")
    write_run_summary(
        config=config,
        dp_sizes=dp_sizes,
        nsys_bin=nsys_bin,
        status="running",
        started_at=started_at,
    )

    try:
        print(
            "Qwen3-MoE DP+EP Nsight Systems profiling: "
            f"GPU_IDS={gpu_ids}, DP_SIZES={dp_sizes}, RESULTS={result_root}",
            flush=True,
        )
        for index, dp_size in enumerate(dp_sizes):
            runner.run_case(
                case_name=f"dp{dp_size}_ep",
                gpu_count=dp_size,
                port=config.base_port + index,
                server_args=dp_ep_server_args(dp_size),
                metadata={
                    "parallelism": "dp_ep",
                    "dp_size": dp_size,
                    "ep_size": dp_size,
                    "tp_size": dp_size,
                    "moe_a2a_backend": moe_a2a_backend_from_env(),
                    "disable_prefix_caching": config.disable_prefix_caching,
                },
            )
        runner.summarize_results()
        write_run_summary(
            config=config,
            dp_sizes=dp_sizes,
            nsys_bin=nsys_bin,
            status="completed",
            started_at=started_at,
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        print(f"Results: {result_root}")
        return 0
    except Exception:
        write_run_summary(
            config=config,
            dp_sizes=dp_sizes,
            nsys_bin=nsys_bin,
            status="failed",
            started_at=started_at,
            completed_at=datetime.now().isoformat(timespec="seconds"),
            failure_reason=traceback.format_exc(),
        )
        raise
    finally:
        runner.cleanup_server()


if __name__ == "__main__":
    raise SystemExit(main())
