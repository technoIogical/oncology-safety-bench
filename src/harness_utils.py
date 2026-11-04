from __future__ import annotations

import atexit
from dataclasses import dataclass

from dotenv import load_dotenv
import datetime
import hashlib
import json
import os
import re
import time

from src.harness_config import DEFAULT_TEMPERATURE
from src.model_registry import CACHE_READ_MULT, CACHE_WRITE_MULT, get_spec

_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SPECIAL = re.compile(r"<\|(?:im_end|eot_id|end)\|>")
_CACHE_READ_KEYS = ("cache_read_tokens", "cache_read_input_tokens", "cached_tokens")
_CACHE_WRITE_KEYS = ("cache_write_tokens", "cache_creation_input_tokens", "cache_creation_tokens")

MAX_WEIGHT_SHARE = 0.70
ENV_TIER, ENV_RATE = "GPU_TIER", "GPU_USD_PER_HOUR"
LOCK_NAME = ".run.lock"

PRIMARY_ARMS = ("base", "rag", "agent")
STAGE1_EXPLORATORY_ARMS = ("checklist", "oracle")
BOTH_STAGE_EXPLORATORY_ARMS = ("agent_sweep",)
PLAN_CHECK_EXPLORATORY_ARMS = ("rag_check", "base_check", "agent_check")
EXPLORATORY_ARMS = STAGE1_EXPLORATORY_ARMS + BOTH_STAGE_EXPLORATORY_ARMS + PLAN_CHECK_EXPLORATORY_ARMS
REPLICATES = {"ladder": 2, "gemini": 2, "descriptive": 2}


class RunLockedError(RuntimeError):
    pass


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class GpuTier:
    name: str
    label: str
    gpus: int
    vram_gb_each: int

    @property
    def vram_gb(self) -> int:
        return self.gpus * self.vram_gb_each

    def max_weights_gb(self) -> float:
        return MAX_WEIGHT_SHARE * self.vram_gb


TIERS = {t.name: t for t in (
    GpuTier("a6000", "1x RTX A6000 48 GB", 1, 48),
    GpuTier("2xa40", "2x A40 48 GB (tensor parallel)", 2, 48),
    GpuTier("a100-80gb", "1x A100 80 GB", 1, 80),
)}
BASELINE_ORDER = ("a6000", "2xa40")


@dataclass(frozen=True)
class RunPlan:
    model: str
    group: str
    replicates: int
    arm_stages: tuple

    @property
    def arms(self) -> tuple:
        return tuple(a for a, _ in self.arm_stages)

    def stages_for(self, arm: str) -> str:
        return dict(self.arm_stages)[arm]

    def describe(self) -> str:
        return (f"{self.model}: group {self.group}, {self.replicates} replicate(s), arms "
                + ", ".join(f"{a}[{'1+2' if s == 'both' else 'stage 1'}]" for a, s in self.arm_stages))


def _read_pid(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return int((f.readline() or "0").strip())
    except (OSError, ValueError):
        return 0


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    def __init__(self, directory: str):
        self.path = os.path.join(directory, LOCK_NAME)
        self.held = False

    def acquire(self) -> "RunLock":
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                pid = _read_pid(self.path)
                if pid and pid != os.getpid() and pid_alive(pid):
                    raise RunLockedError(
                        f"another process (pid {pid}) is running in {os.path.dirname(self.path)}; two processes on one results "
                        f"directory write every row twice. Stop it, or wait for it to finish ({self.path} says who holds it).")
                try:
                    os.remove(self.path)
                except FileNotFoundError:
                    pass
                time.sleep(0.05)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()}\n")
            self.held = True
            atexit.register(self.release)
            return self
        raise RunLockedError(f"could not take {self.path}")

    def release(self) -> None:
        if self.held and _read_pid(self.path) == os.getpid():
            try:
                os.remove(self.path)
            except OSError:
                pass
        self.held = False

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def _registry_spec(model_name: str):
    try:
        return get_spec(model_name)
    except Exception:
        return None


def _first_int(usage: dict, keys) -> int:
    for k in keys:
        v = (usage or {}).get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return 0


def strip_reasoning(text: str) -> str:
    out = _THINK_CLOSED.sub("", text or "")
    low = out.lower()
    if "<think>" in low:
        start = low.index("<think>")
        tag = out.find("FINAL_RECOMMENDATION:", start)
        out = out[:start] + (out[tag:] if tag != -1 else "")
    return _SPECIAL.sub("", out).strip()


def conversation_chars(messages: list) -> int:
    total = 0
    for m in messages:
        total += len(m.get("content") or "")
        if m.get("tool_calls"):
            total += len(json.dumps(m["tool_calls"], default=str))
    return total


def truncate_result(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n[TRUNCATED: result exceeded {cap} characters]"


def is_truncated(usage: dict) -> bool:
    return str((usage or {}).get("finish_reason", "")).upper().split(".")[-1] in {"LENGTH", "MAX_TOKENS"}


def derive_seed(*parts) -> int:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def model_run_settings(model_name: str) -> dict:
    spec = _registry_spec(model_name)
    temperature = getattr(spec, "temperature", None)
    interval = getattr(spec, "min_interval_s", None)
    return {"temperature": DEFAULT_TEMPERATURE if temperature is None else float(temperature),
            "min_interval_s": 0.0 if interval is None else max(0.0, float(interval))}


def model_prices(model_name: str) -> dict | None:
    spec = _registry_spec(model_name)
    price = getattr(spec, "price", None)
    if price is None and getattr(spec, "price_in_per_mtok", None) is not None and getattr(spec, "price_out_per_mtok", None) is not None:
        price = {"input": spec.price_in_per_mtok, "output": spec.price_out_per_mtok,
                 "cache_read": (spec.price_cache_read_per_mtok if spec.price_cache_read_per_mtok is not None
                                else CACHE_READ_MULT * spec.price_in_per_mtok),
                 "cache_write": CACHE_WRITE_MULT * spec.price_in_per_mtok}
    if price is None:
        return None
    key_names = {k: (k, f"{k}_per_mtok", f"{k}_per_million") for k in ("input", "output", "cache_read", "cache_write")}
    out = {}
    for key, names in key_names.items():
        for n in names:
            v = price.get(n) if isinstance(price, dict) else getattr(price, n, None)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[key] = float(v)
                break
    return out if "input" in out and "output" in out else None


def usage_tokens(usage: dict | None) -> dict:
    u = usage or {}
    return {"prompt_tokens": _first_int(u, ("prompt_tokens",)), "completion_tokens": _first_int(u, ("completion_tokens",)),
            "thoughts_tokens": _first_int(u, ("thoughts_tokens",)),
            "cache_read_tokens": _first_int(u, _CACHE_READ_KEYS), "cache_write_tokens": _first_int(u, _CACHE_WRITE_KEYS)}


def aggregate_usage(calls: list) -> dict:
    keys = ("prompt_tokens", "completion_tokens", "thoughts_tokens", "cache_read_tokens", "cache_write_tokens")
    return {k: sum(int(c.get(k) or 0) for c in calls) for k in keys}


def cost_usd(totals: dict, prices: dict | None, batch: bool = False) -> float | None:
    if not prices:
        return None
    read, write = totals.get("cache_read_tokens", 0), totals.get("cache_write_tokens", 0)
    plain = max(0, totals.get("prompt_tokens", 0) - read - write)
    usd = (plain * prices["input"] + read * prices.get("cache_read", prices["input"])
           + write * prices.get("cache_write", prices["input"])
           + (totals.get("completion_tokens", 0) + totals.get("thoughts_tokens", 0)) * prices["output"]) / 1e6
    return round(usd * (0.5 if batch else 1.0), 6)


def load_env() -> None:
    load_dotenv(os.environ.get("DOTENV_PATH") or None)
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CLAUDE_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_API_KEY"]


def hardware_record(tier: str | None = None, usd_per_hour: float | str | None = None) -> dict | None:
    tier = (tier or os.environ.get(ENV_TIER) or "").strip().lower() or None
    if tier is None:
        return None
    if tier not in TIERS:
        raise ValueError(f"unknown GPU tier {tier!r}; expected one of {sorted(TIERS)}")
    raw = usd_per_hour if usd_per_hour is not None else os.environ.get(ENV_RATE)
    rate = None
    if raw not in (None, ""):
        rate = float(raw)
        if rate < 0:
            raise ValueError("the hourly rate must not be negative")
    t = TIERS[tier]
    return {"tier": t.name, "label": t.label, "gpus": t.gpus, "vram_gb": t.vram_gb, "usd_per_hour": rate,
            "rate_basis": "pod price per hour as billed by the provider" if rate is not None else None}


def plan_for(model_name: str, *, include_check: bool = False) -> RunPlan:
    spec = get_spec(model_name)
    if spec.family == "gemini":
        group = "gemini" if spec.role == "anchor" else "descriptive"
    elif spec.role == "ladder":
        group = "ladder"
    else:
        group = "descriptive"
    arm_stages = [(a, "both") for a in PRIMARY_ARMS]
    arm_stages += [(a, "both") for a in STAGE1_EXPLORATORY_ARMS]
    if group in ("ladder", "gemini"):
        arm_stages += [(a, "both") for a in BOTH_STAGE_EXPLORATORY_ARMS]
    if include_check:
        arm_stages += [(a, "both") for a in PLAN_CHECK_EXPLORATORY_ARMS]
    return RunPlan(spec.name, group, REPLICATES[group], tuple(arm_stages))
