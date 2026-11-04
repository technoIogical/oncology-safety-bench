"""Launches local vLLM OpenAI-compatible model servers with per-model GPU and context settings."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_registry import ModelSpec, get_spec, is_registered, local_models

DEFAULT_PORT = 12345
GPU_MEMORY_UTILIZATION = "0.95"
DEFAULT_LOG = "vllm_server.log"


def detect_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=10, check=True).stdout
        return max(1, len([ln for ln in out.splitlines() if ln.strip()]))
    except Exception:
        return 1


def build_command(spec: ModelSpec, *, num_gpus: int = 1, port: int = DEFAULT_PORT, context=None) -> list:
    if spec.provider != "local" or not spec.hf_id:
        raise ValueError(f"{spec.name!r} is not a local model; only local models are served with vLLM")
    cmd = ["vllm", "serve", spec.hf_id, "--served-model-name", spec.name,
           "--dtype", "bfloat16" if spec.precision == "bf16" else "auto",
           "--max-model-len", str(context or spec.max_context or 65536),
           "--tensor-parallel-size", str(max(1, int(num_gpus))),
           "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION, "--port", str(port)]
    if spec.prefix_caching:
        cmd.append("--enable-prefix-caching")
    if num_gpus > 1:
        cmd.append("--disable-custom-all-reduce")
    if spec.revision:
        cmd += ["--revision", spec.revision]
    cmd += list(spec.serve_args)
    if spec.native_tools and spec.tool_parser:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", spec.tool_parser]
    if spec.reasoning_parser:
        cmd += ["--reasoning-parser", spec.reasoning_parser]
    return cmd


def run_command(cmd: list, log_path: str = DEFAULT_LOG) -> int:
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    except FileNotFoundError:
        print(f"error: {cmd[0]!r} not found; activate the venv that has vLLM installed", file=sys.stderr)
        return 127
    with open(log_path, "w") as log:
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            proc.terminate()
        return proc.wait()


def describe_route(spec: ModelSpec) -> str:
    if not spec.native_tools:
        return "text route"
    return f"native tools ({spec.tool_parser})" + ("" if spec.native_tools_confirmed else ", unconfirmed")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=None)
    ap.add_argument("--model", help="registry name or HF id of a local model")
    ap.add_argument("--list", action="store_true", help="list the local roster")
    ap.add_argument("--names", action="store_true", help="print only the registry names, one per line")
    ap.add_argument("--exec", dest="do_exec", action="store_true", help="run the server instead of printing the command")
    ap.add_argument("--gpus", type=int, default=None, help="tensor-parallel size (default: GPUs found by nvidia-smi)")
    ap.add_argument("--context", default=None, help="served window in tokens, or 'auto' for the longest that fits "
                                                    "(default: the registry window, VLLM_SERVED_CONTEXT or 65536)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--log", default=DEFAULT_LOG, help="server log written by --exec")
    args = ap.parse_args(argv)

    if args.names or args.list:
        for s in local_models():
            print(s.name if args.names else f"{s.name}\t{s.hf_id}\t{s.role}\t{describe_route(s)}")
        return 0
    if not args.model:
        ap.error("--model is required (or --list)")
    if not is_registered(args.model):
        print(f"error: {args.model!r} is not in the registry; use --list", file=sys.stderr)
        return 2
    spec = get_spec(args.model)
    cmd = build_command(spec, num_gpus=args.gpus or detect_gpus(), port=args.port, context=args.context)
    if not args.do_exec:
        print(shlex.join(cmd))
        return 0
    print(f"Launching: {shlex.join(cmd)}", flush=True)
    return run_command(cmd, args.log)

if __name__ == "__main__":
    sys.exit(main())
