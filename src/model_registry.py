"""Model roster, sampling hyperparameters, context windows, vLLM flags, and API pricing."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from src.harness_config import (CLAUDE_EFFORT, CLAUDE_OUTPUT_ALLOWANCE_TOKENS, CLAUDE_THINKING_TEMPERATURE, DEFAULT_TEMPERATURE,
                                GEMINI_THINKING, LOCAL_THINKING_BUDGET_TOKENS)

PROVIDERS = ("local", "gemini", "anthropic")
ROLES = ("", "ladder", "descriptive", "anchor", "reference")
SERVED_CONTEXT = int(os.environ.get("VLLM_SERVED_CONTEXT") or 65_536)
GEMINI_PACING_S = 1.5
CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25
BATCH_DISCOUNT = 0.5
_LOCAL_THINK = {"budget_tokens": LOCAL_THINKING_BUDGET_TOKENS}
_GEMMA_SAMPLING = {"top_p": 0.95, "top_k": 64}
_QWEN_SAMPLING = {"top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
_MUSE_SAMPLING = {"top_p": 0.95, "top_k": 64}
_NEMOTRON_SAMPLING = {"top_p": 0.95}
_MEDGEMMA_SAMPLING = {"top_p": 0.95, "top_k": 64}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    family: str = ""
    params_b: float | None = None
    active_params_b: float | None = None
    role: str = ""
    hf_id: str | None = None
    revision: str | None = None
    precision: str = "unknown"
    native_tools: bool = True
    native_tools_confirmed: bool = True
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    chat_template_kwargs: dict = field(default_factory=dict, compare=False, hash=False)
    thinking: dict = field(default_factory=dict, compare=False, hash=False)
    temperature: float = DEFAULT_TEMPERATURE
    min_interval_s: float = 0.0
    max_context: int | None = None
    price_in_per_mtok: float | None = None
    price_out_per_mtok: float | None = None
    price_cache_read_per_mtok: float | None = None
    min_vllm: str | None = None
    prefix_caching: bool = True
    serve_args: tuple = ()
    sampling: dict = field(default_factory=dict, compare=False, hash=False)
    output_to_window: bool = False
    notes: str = ""

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; expected one of {PROVIDERS}")
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; expected one of {ROLES}")

    @property
    def api_model_id(self) -> str:
        return self.name

    @property
    def thinking_budget_tokens(self) -> int:
        t = self.thinking or {}
        return int(t.get("budget_tokens") or t.get("allowance_tokens") or 0)


def _local(hf_id: str, **kw) -> ModelSpec:
    kw.setdefault("precision", "bf16")
    kw.setdefault("max_context", SERVED_CONTEXT)
    return ModelSpec(name=hf_id.split("/")[-1], provider="local", hf_id=hf_id, **kw)

_ROSTER: tuple = (
    ModelSpec(name="gemini-4-argon", provider="gemini", family="gemini", role="anchor", precision="api",
              thinking=dict(GEMINI_THINKING), min_interval_s=GEMINI_PACING_S,
              notes=""),
    ModelSpec(name="gemini-3.1-pro-preview", provider="gemini", family="gemini", role="descriptive", precision="api",
              thinking=dict(GEMINI_THINKING), min_interval_s=GEMINI_PACING_S,
              price_in_per_mtok=2.0, price_out_per_mtok=12.0, price_cache_read_per_mtok=0.20,
              notes=""),
    ModelSpec(name="claude-sonnet-5-5", provider="anthropic", family="claude", role="anchor", precision="api",
              thinking={"type": "adaptive", "effort": CLAUDE_EFFORT, "allowance_tokens": CLAUDE_OUTPUT_ALLOWANCE_TOKENS},
              temperature=CLAUDE_THINKING_TEMPERATURE,
              price_in_per_mtok=3.0, price_out_per_mtok=15.0, price_cache_read_per_mtok=0.30,
              notes=""),
    ModelSpec(name="claude-opus-5-5", provider="anthropic", family="claude", role="reference", precision="api",
              thinking={"type": "adaptive", "effort": CLAUDE_EFFORT, "allowance_tokens": CLAUDE_OUTPUT_ALLOWANCE_TOKENS},
              temperature=CLAUDE_THINKING_TEMPERATURE,
              price_in_per_mtok=4.0, price_out_per_mtok=20.0, price_cache_read_per_mtok=0.20,
              notes=""),
    _local("google/gemma-4-31B-it", revision="842da3794eaa0b77d5f08bae87a17459d91ff475", family="gemma", role="ladder", params_b=31, tool_parser="gemma4",
           reasoning_parser="gemma4", chat_template_kwargs={"enable_thinking": True}, thinking=dict(_LOCAL_THINK),
           temperature=1.0, sampling=dict(_GEMMA_SAMPLING), output_to_window=True,
           serve_args=("--max-num-batched-tokens", "2496", "--max-num-seqs", "64", "--gpu-memory-utilization", "0.98", "--limit-mm-per-prompt", '{"image":0,"video":0}')),
    _local("google/gemma-4-12B-it", revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", family="gemma", role="ladder", params_b=12, tool_parser="gemma4",
           reasoning_parser="gemma4", chat_template_kwargs={"enable_thinking": True}, thinking=dict(_LOCAL_THINK),
           temperature=1.0, sampling=dict(_GEMMA_SAMPLING), output_to_window=True,
           notes=""),
    _local("google/gemma-4-E4B-it", revision="ee0ef6023621cff504d758262d4e04895a5af4a2", family="gemma", role="ladder", params_b=4, tool_parser="gemma4",
           reasoning_parser="gemma4", chat_template_kwargs={"enable_thinking": True}, thinking=dict(_LOCAL_THINK),
           temperature=1.0, sampling=dict(_GEMMA_SAMPLING), output_to_window=True),
    _local("google/gemma-4-E2B-it", revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7", family="gemma", role="ladder", params_b=2, tool_parser="gemma4",
           reasoning_parser="gemma4", chat_template_kwargs={"enable_thinking": True}, thinking=dict(_LOCAL_THINK),
           temperature=1.0, sampling=dict(_GEMMA_SAMPLING), output_to_window=True),
    _local("Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", family="qwen", role="descriptive", params_b=27, tool_parser="qwen3_coder",
           reasoning_parser="qwen3", chat_template_kwargs={"enable_thinking": True}, thinking=dict(_LOCAL_THINK),
           temperature=1.0, sampling=dict(_QWEN_SAMPLING), output_to_window=True, serve_args=("--max-num-seqs", "16")),
    _local("Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a", family="qwen", role="descriptive", params_b=9, tool_parser="qwen3_coder",
           reasoning_parser="qwen3", chat_template_kwargs={"enable_thinking": True}, thinking=dict(_LOCAL_THINK),
           temperature=1.0, sampling=dict(_QWEN_SAMPLING), output_to_window=True, serve_args=("--max-num-seqs", "64")),
    _local("meta-models/Muse-Glimmer-30B", revision="a4e59da52a7bc87ae7251dd5545c0dd437c44b68", family="muse", role="descriptive", params_b=30,
           active_params_b=None,
           tool_parser="muse_glimmer", reasoning_parser="muse_glimmer", min_vllm="0.28.0",
           chat_template_kwargs={"reasoning_strength": "xhigh"}, thinking=dict(_LOCAL_THINK),
           temperature=1.0, sampling=dict(_MUSE_SAMPLING), output_to_window=True,
           notes=""),
    _local("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", revision="a9904d24bcc1d289a1950fa9d2b978c47cf903b9", family="nemotron", role="descriptive", params_b=30,
           active_params_b=3, tool_parser="qwen3_coder", reasoning_parser="nemotron_v3",
           chat_template_kwargs={"enable_thinking": True}, thinking=dict(_LOCAL_THINK),
           temperature=1.0, sampling=dict(_NEMOTRON_SAMPLING), output_to_window=True,
           notes=""),
    _local("mistralai/Ministral-3-14B-Instruct-2512", revision="29439f81c2be264d8d393273f99e7db9c0961120", family="ministral", role="descriptive", params_b=14,
           tool_parser="mistral", native_tools_confirmed=False, temperature=0.05,
           serve_args=("--tokenizer-mode", "mistral", "--config-format", "mistral", "--load-format", "mistral"),
           notes=""),
    _local("google/medgemma-1.5-4b-it", revision="91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b", family="medgemma", role="descriptive", params_b=4, native_tools=False,
           temperature=1.0, sampling=dict(_MEDGEMMA_SAMPLING),
           notes=""),
    _local("google/medgemma-27b-text-it", revision="5b667cf2ddcf064085bc90952edb35a0edbfb79c", family="medgemma", role="descriptive", params_b=27, native_tools=False,
           temperature=1.0, sampling=dict(_MEDGEMMA_SAMPLING),
           notes=""),

)

ANCHORS = {"gemini": "gemini-4-argon", "claude": "claude-sonnet-5-5"}
LADDER: tuple = ("gemma-4-31B-it", "gemma-4-12B-it", "gemma-4-E4B-it", "gemma-4-E2B-it")
REGISTRY: dict = {s.name: s for s in _ROSTER}
_INDEX: dict = {k: s for s in _ROSTER for k in ((s.name or "").strip().lower(), (s.hf_id or "").strip().lower()) if k}


def _norm(name: str) -> str:
    return (name or "").strip().lower()


def _infer_provider(name: str) -> str:
    low = (name or "").lower()
    if low.startswith("claude") or "/claude" in low:
        return "anthropic"
    if low.startswith("gemini") or "/gemini" in low:
        return "gemini"
    return "local"


def _inferred(name: str) -> ModelSpec:
    provider = _infer_provider(name)
    if provider == "gemini":
        return ModelSpec(name=name, provider=provider, family="gemini", precision="api", min_interval_s=GEMINI_PACING_S)
    if provider == "anthropic":
        return ModelSpec(name=name, provider=provider, family="claude", precision="api")
    return ModelSpec(name=name, provider=provider, native_tools_confirmed=False, notes="")


def is_registered(name: str) -> bool:
    key = _norm(name)
    return key in _INDEX or key.split("/")[-1].split("\\")[-1] in _INDEX


def get_spec(name: str) -> ModelSpec:

    name = name or "local-model"
    key = _norm(name)
    hit = _INDEX.get(key) or _INDEX.get(key.split("/")[-1].split("\\")[-1])
    return hit if hit is not None else _inferred(name)


def provider_of(name: str) -> str:
    return get_spec(name).provider


def is_gemini(name: str) -> bool:
    return provider_of(name) == "gemini"


def is_claude(name: str) -> bool:
    return provider_of(name) == "anthropic"


def is_local(name: str) -> bool:
    return provider_of(name) == "local"


def local_models(*, include_reference: bool = True) -> list:
    return [s for s in _ROSTER if s.provider == "local" and (include_reference or s.role != "reference")]


def descriptor(name: str) -> dict:
    spec = get_spec(name)
    d = json.loads(json.dumps(asdict(spec)))
    d["registered"] = is_registered(name)
    d["thinking_budget_tokens"] = spec.thinking_budget_tokens
    return d
