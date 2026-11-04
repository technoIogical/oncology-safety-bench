"""Judge configuration, content-addressed verdict cache, and synchronous/batch judge transports."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import httpx
from google import genai
from google.genai import types
from openai import OpenAI

from src.harness_config import (
    JUDGE_GEMINI_FLASH_ENV,
    JUDGE_MAX_OUTPUT_TOKENS,
    JUDGE_MAX_TOTAL_WAIT_S,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    JUDGE_REQUEST_TIMEOUT_S,
    JUDGE_TEMPERATURE,
    JUDGE_THINKING_BUDGET,
)

JUDGE_CACHE_VERSION = 2
CACHE_MODES = ("readwrite", "replay", "off")
PROVIDERS = ("gemini", "openai_compatible", "anthropic")
SCHEMA_NAME = "JudgeVerdict"
VERDICT_TOOL = "submit_verdict"
ENFORCEMENT = {"gemini": "gemini_response_schema", "anthropic": "anthropic_forced_tool",
               "openai_compatible": "openai_json_schema"}
ANCHOR_FAMILIES = ("gemini", "claude", "local_only")
ANTHROPIC_VERSION = "2023-06-01"
BATCH_DISCOUNT = 0.5
_FAMILIES = (("medgemma", "medgemma"), ("gemini", "gemini"), ("gemma", "gemma"), ("claude", "claude"),
             ("anthropic", "claude"), ("qwen", "qwen"), ("phi", "phi"), ("llama", "llama"), ("ministral", "mistral"),
             ("mistral", "mistral"), ("gpt", "gpt"))
JUDGE_ASSIGNMENT = {"gemini": ("claude", "gemini"), "claude": ("gemini", "claude"), "local_only": ("gemini", "claude")}


class JudgeCacheMiss(RuntimeError):
    pass


class JudgeConfigError(ValueError):
    pass


class JudgeAssignmentError(ValueError):
    pass


class GeminiHTTPError(RuntimeError):
    def __init__(self, status_code, text):
        super().__init__(f"HTTP {status_code}: {text[:500]}")
        self.status_code = status_code


class AnthropicHTTPError(RuntimeError):
    def __init__(self, status_code, text):
        super().__init__(f"HTTP {status_code}: {text[:500]}")
        self.status_code = status_code


class BatchTimeout(RuntimeError):
    pass


@dataclass(frozen=True)
class JudgeConfig:
    provider: str = JUDGE_PROVIDER
    model: str = JUDGE_MODEL
    thinking_budget: int | None = JUDGE_THINKING_BUDGET
    temperature: float = JUDGE_TEMPERATURE
    api_base: str | None = None
    key_env: str | None = None
    request_timeout_s: float = JUDGE_REQUEST_TIMEOUT_S
    max_total_wait_s: float = JUDGE_MAX_TOTAL_WAIT_S

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown judge provider {self.provider!r}; expected one of {PROVIDERS}")
        if self.provider != "gemini" and self.thinking_budget is not None:
            object.__setattr__(self, "thinking_budget", None)

    def identity(self) -> dict:
        return {"provider": self.provider, "model": self.model, "thinking_budget": self.thinking_budget,
                "temperature": self.temperature, "api_base": self.api_base,
                "schema_enforcement": ENFORCEMENT[self.provider]}

    @property
    def family(self) -> str:
        if self.provider == "anthropic":
            return "claude"
        if self.provider == "gemini":
            return "gemini"
        return model_family(self.model, default=self.provider)


@dataclass(frozen=True)
class BatchResult:
    custom_id: str
    ok: bool
    text: str = ""
    model_version: str | None = None
    error: str = ""


def model_family(model: str, default: str = "other") -> str:
    m = str(model or "").lower()
    for needle, fam in _FAMILIES:
        if needle in m:
            return fam
    return default


def judge_config_from_spec(spec: str, env=None, **overrides) -> JudgeConfig:
    env = os.environ if env is None else env
    spec = (spec or "").strip()
    if not spec:
        raise JudgeConfigError("empty judge spec")
    provider, model = (spec.split(":", 1) if ":" in spec else ("", spec))
    provider = {"claude": "anthropic", "anthropic": "anthropic", "gemini": "gemini", "google": "gemini",
                "openai_compatible": "openai_compatible"}.get(provider, provider)
    if provider and provider not in PROVIDERS:
        raise JudgeConfigError(f"unknown judge provider in {spec!r}; write claude:<model> or gemini:<model>")
    if model.lower() in ("flash", "gemini-flash"):
        model = env.get(JUDGE_GEMINI_FLASH_ENV, "")
        if not model:
            raise JudgeConfigError(f"the Gemini Flash judge id is not configured: set {JUDGE_GEMINI_FLASH_ENV} to the "
                                   f"model id (it is deliberately not hard-coded)")
        provider = provider or "gemini"
    provider = provider or {"claude": "anthropic", "gemini": "gemini"}.get(model_family(model), "")
    if not provider:
        raise JudgeConfigError(f"cannot tell the provider of {spec!r}; write claude:<model> or gemini:<model>")
    return JudgeConfig(provider=provider, model=model, **overrides)


def primary_judge_config(model_name: str | None = None) -> JudgeConfig:
    return JudgeConfig(model=model_name or JUDGE_MODEL)


def second_judge_config(env=None) -> JudgeConfig | None:
    env = os.environ if env is None else env
    provider, model = env.get("JUDGE2_PROVIDER"), env.get("JUDGE2_MODEL")
    if not provider or not model:
        return None
    tb = env.get("JUDGE2_THINKING_BUDGET")
    return JudgeConfig(provider=provider, model=model, thinking_budget=int(tb) if tb not in (None, "") else None,
                       api_base=env.get("JUDGE2_API_BASE") or None, key_env=env.get("JUDGE2_KEY_ENV") or None)


def resolve_second_judge(spec: str | None = None, env=None) -> JudgeConfig | None:
    env = os.environ if env is None else env
    if spec:
        return judge_config_from_spec(spec, env)
    return second_judge_config(env)


def source_row_key(model, arm, variant, replicate, case_id) -> str:
    return "|".join(str(x) for x in (model, arm, variant or "standard", replicate if str(replicate) != "" else 0, case_id))


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def cache_key(cfg: JudgeConfig, prompt: str, replicate: int = 0, schema_tag: str = "") -> str:
    """Computes a content-addressed SHA-256 cache key over judge identity, replicate, schema, and prompt hash."""
    body = json.dumps({"v": JUDGE_CACHE_VERSION, **cfg.identity(), "replicate": int(replicate), "schema": schema_tag,
                       "max_output_tokens": JUDGE_MAX_OUTPUT_TOKENS, "prompt": prompt_hash(prompt)}, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class JudgeCache:
    def __init__(self, directory, mode: str = "readwrite"):
        if mode not in CACHE_MODES:
            raise ValueError(f"unknown judge cache mode {mode!r}")
        self.dir, self.mode = Path(directory) if directory else None, mode if directory else "off"

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        if self.mode == "off":
            return None
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, payload: dict) -> dict:
        if self.mode != "readwrite":
            return payload
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.parent / f".{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        body = {**payload, "key": key, "cache_version": JUDGE_CACHE_VERSION,
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(body, ensure_ascii=False, sort_keys=True))
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, p)
            stored = body
        except FileExistsError:
            existing = self.get(key)
            stored = existing if existing is not None else body
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return stored


# Recursively expands JSON Schema $ref pointers inline so provider tool/response schemas have no $defs.
def _inline_refs(node, defs):
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            return _inline_refs({**defs[name], **{k: v for k, v in node.items() if k != "$ref"}}, defs)
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def verdict_schema() -> dict:
    from src.evaluator import JudgeVerdict
    raw = JudgeVerdict.model_json_schema()
    return _inline_refs(raw, raw.get("$defs", {}))


def _key_for(cfg: JudgeConfig, default_env: str) -> str:
    env = cfg.key_env or default_env
    key = os.environ.get(env)
    if not key:
        raise ValueError(f"{env} environment variable is missing; cannot initialise the {cfg.provider} judge")
    return key


def _is_format_unsupported(e: Exception) -> bool:
    code = getattr(e, "status_code", None)
    msg = str(e).lower()
    return code in (400, 404, 422, 501) and ("response_format" in msg or "json_schema" in msg or "schema" in msg
                                             or "unsupported" in msg or code in (404, 501))


def openai_compatible_call(cfg: JudgeConfig, prompt: str, client=None, info: dict | None = None) -> tuple[str, str | None]:
    if client is None:
        client = OpenAI(base_url=cfg.api_base or "https://api.openai.com/v1", api_key=_key_for(cfg, "JUDGE2_API_KEY"),
                        max_retries=0, http_client=httpx.Client(timeout=httpx.Timeout(cfg.request_timeout_s)))
    common = dict(model=cfg.model, temperature=cfg.temperature, max_tokens=JUDGE_MAX_OUTPUT_TOKENS,
                  messages=[{"role": "user", "content": prompt}])
    enforcement = "openai_json_schema"
    try:
        resp = client.chat.completions.create(
            **common, response_format={"type": "json_schema",
                                       "json_schema": {"name": SCHEMA_NAME, "schema": verdict_schema()}})
    except Exception as e:
        if not _is_format_unsupported(e):
            raise
        enforcement = "openai_json_object"
        resp = client.chat.completions.create(**common, response_format={"type": "json_object"})
    if info is not None:
        info["enforcement"] = enforcement
    return resp.choices[0].message.content or "", getattr(resp, "model", None)


def anthropic_request_body(cfg: JudgeConfig, prompt: str) -> dict:
    return {"model": cfg.model, "max_tokens": JUDGE_MAX_OUTPUT_TOKENS, "temperature": cfg.temperature,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{"name": VERDICT_TOOL, "description": "Submit the complete grading verdict for the AI recommendation.",
                       "input_schema": verdict_schema()}],
            "tool_choice": {"type": "tool", "name": VERDICT_TOOL}}


def anthropic_message_to_verdict(data: dict) -> tuple[str, str | None]:
    for block in data.get("content") or []:
        if block.get("type") == "tool_use" and block.get("name") == VERDICT_TOOL:
            if data.get("stop_reason") == "max_tokens":
                raise ValueError("judge output was cut at the output limit (stop_reason max_tokens)")
            return json.dumps(block.get("input") or {}, ensure_ascii=False), data.get("model")
    raise ValueError(f"judge response has no {VERDICT_TOOL} tool call (stop_reason {data.get('stop_reason')!r})")


def _anthropic_headers(cfg: JudgeConfig) -> dict:
    return {"x-api-key": _key_for(cfg, "ANTHROPIC_API_KEY"), "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json"}


def _gemini_headers(cfg: JudgeConfig) -> dict:
    return {"x-goog-api-key": _key_for(cfg, "GEMINI_API_KEY"), "content-type": "application/json"}


def _gemini_httpx_transport(method, url, headers, json_body, timeout):
    last_exc = None
    for attempt in range(6):
        try:
            r = httpx.request(method, url, headers=headers, json=json_body, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(min(30.0, 2.0 * (2 ** attempt)))
                continue
            if r.status_code >= 400:
                raise GeminiHTTPError(r.status_code, r.text)
            return r.json()
        except httpx.TransportError as e:
            last_exc = e
            if attempt < 5:
                time.sleep(min(30.0, 2.0 * (2 ** attempt)))
                continue
            raise
    raise last_exc


def _httpx_post(url, headers, json, timeout):
    r = httpx.post(url, headers=headers, json=json, timeout=timeout)
    if r.status_code >= 400:
        raise AnthropicHTTPError(r.status_code, r.text)
    return r.json()


def anthropic_call(cfg: JudgeConfig, prompt: str, post=None, info: dict | None = None) -> tuple[str, str | None]:
    base = (cfg.api_base or "https://api.anthropic.com").rstrip("/")
    body = anthropic_request_body(cfg, prompt)
    data = (post or _httpx_post)(f"{base}/v1/messages", _anthropic_headers(cfg), body, cfg.request_timeout_s)
    if info is not None:
        info["enforcement"] = "anthropic_forced_tool"
    return anthropic_message_to_verdict(data)


def call_other_provider(cfg: JudgeConfig, prompt: str, info: dict | None = None) -> tuple[str, str | None]:
    if cfg.provider == "openai_compatible":
        return openai_compatible_call(cfg, prompt, info=info)
    if cfg.provider == "anthropic":
        return anthropic_call(cfg, prompt, info=info)
    raise ValueError(f"{cfg.provider} is handled by the Gemini path of ClinicalEvaluator")


def _httpx_transport(method, url, headers, json_body, timeout):
    r = httpx.request(method, url, headers=headers, json=json_body, timeout=timeout)
    if r.status_code >= 400:
        raise AnthropicHTTPError(r.status_code, r.text)
    return r.text if url.rstrip("/").endswith("/results") else r.json()


class AnthropicBatch:

    MAX_REQUESTS = 100_000

    def __init__(self, cfg: JudgeConfig, transport: Callable | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        if cfg.provider != "anthropic":
            raise ValueError("the Message Batches API is only available for Anthropic judges")
        self.cfg, self.transport, self.sleep, self.clock = cfg, transport or _httpx_transport, sleep, clock
        self.base = (cfg.api_base or "https://api.anthropic.com").rstrip("/")

    def _call(self, method: str, url: str, body=None):
        return self.transport(method, url, _anthropic_headers(self.cfg), body, self.cfg.request_timeout_s)

    def submit(self, requests: Sequence[tuple[str, str]]) -> str:
        if not requests or len(requests) > self.MAX_REQUESTS:
            raise ValueError(f"a batch needs 1..{self.MAX_REQUESTS} requests, got {len(requests)}")
        payload = {"requests": [{"custom_id": cid, "params": anthropic_request_body(self.cfg, prompt)}
                                for cid, prompt in requests]}
        data = self._call("POST", f"{self.base}/v1/messages/batches", payload)
        return data["id"]

    def status(self, batch_id: str) -> dict:
        return self._call("GET", f"{self.base}/v1/messages/batches/{batch_id}")

    def wait(self, batch_id: str, poll_s: float = 60.0, max_wait_s: float = 26 * 3600.0,
             on_poll: Callable[[dict], None] | None = None) -> dict:
        start = self.clock()
        while True:
            st = self.status(batch_id)
            if on_poll:
                on_poll(st)
            if st.get("processing_status") == "ended":
                return st
            if self.clock() - start > max_wait_s:
                raise BatchTimeout(f"batch {batch_id} still {st.get('processing_status')!r} after {max_wait_s:.0f}s")
            self.sleep(poll_s)

    def results(self, batch_id: str, status: dict | None = None) -> dict[str, BatchResult]:
        st = status or self.status(batch_id)
        url = st.get("results_url") or f"{self.base}/v1/messages/batches/{batch_id}/results"
        raw = self._call("GET", url)
        lines = raw.splitlines() if isinstance(raw, str) else list(raw)
        out: dict[str, BatchResult] = {}
        for line in lines:
            if isinstance(line, str):
                if not line.strip():
                    continue
                line = json.loads(line)
            cid, res = line.get("custom_id"), line.get("result") or {}
            kind = res.get("type")
            if kind == "succeeded":
                try:
                    text, version = anthropic_message_to_verdict(res.get("message") or {})
                    out[cid] = BatchResult(cid, True, text, version)
                except ValueError as e:
                    out[cid] = BatchResult(cid, False, error=str(e))
            else:
                err = (res.get("error") or {}).get("error") or res.get("error") or {}
                out[cid] = BatchResult(cid, False, error=f"{kind}: {err.get('message', err) if isinstance(err, dict) else err}")
        return out


def gemini_request_body(cfg: JudgeConfig, prompt: str) -> dict:
    gen_config: dict = {
        "temperature": cfg.temperature,
        "maxOutputTokens": JUDGE_MAX_OUTPUT_TOKENS,
        "responseMimeType": "application/json",
        "responseSchema": verdict_schema(),
    }
    if cfg.thinking_budget is not None:
        gen_config["thinkingConfig"] = {"thinking_budget": cfg.thinking_budget}
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
        "safetySettings": [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }


def gemini_batch_request_item(cfg: JudgeConfig, prompt: str, cid: str) -> dict:
    return {
        "request": gemini_request_body(cfg, prompt),
        "metadata": {"custom_id": cid},
    }


def gemini_batch_payload(cfg: JudgeConfig, requests: Sequence[tuple[str, str]]) -> dict:
    return {
        "batch": {
            "inputConfig": {
                "requests": {
                    "requests": [gemini_batch_request_item(cfg, prompt, cid) for cid, prompt in requests]
                }
            }
        }
    }


def gemini_response_to_verdict(data: dict) -> tuple[str, str | None]:
    from src.evaluator import JudgeVerdict
    candidates = data.get("candidates") or []
    version = data.get("modelVersion") or data.get("model_version")
    if not candidates:
        if data.get("text"):
            text = str(data["text"]).strip()
            JudgeVerdict.model_validate(json.loads(text, strict=False))
            return text, version
        feedback = data.get("promptFeedback") or data.get("prompt_feedback")
        raise ValueError(f"gemini judge returned no candidates (prompt_feedback: {feedback})")
    first = candidates[0]
    finish = first.get("finishReason") or first.get("finish_reason")
    if finish == "MAX_TOKENS":
        raise ValueError("judge output was cut at the output limit (finish_reason MAX_TOKENS)")
    if finish == "SAFETY":
        raise ValueError("judge output was blocked by safety settings")
    content = first.get("content") or {}
    parts = content.get("parts") or []
    visible_parts = [p for p in parts if not (isinstance(p, dict) and p.get("thought"))] or parts
    text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in visible_parts).strip()
    if not text:
        raise ValueError("gemini judge returned empty text")
    try:
        val = json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        from src.model_clients import extract_and_parse_json
        parsed = extract_and_parse_json(text)
        if parsed is None:
            raise ValueError(f"judge response is not JSON: {e}") from e
        val = parsed
    JudgeVerdict.model_validate(val)
    return text, version


class GeminiBatch:

    MAX_REQUESTS = 100_000

    def __init__(self, cfg: JudgeConfig, transport: Callable | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        if cfg.provider != "gemini":
            raise ValueError("GeminiBatch is only available for Gemini judges")
        self.cfg, self.transport, self.sleep, self.clock = cfg, transport or _gemini_httpx_transport, sleep, clock
        self.base = (cfg.api_base or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

    def _call(self, method: str, url: str, body=None):
        return self.transport(method, url, _gemini_headers(self.cfg), body, self.cfg.request_timeout_s)

    def submit(self, requests: Sequence[tuple[str, str]]) -> str:
        if not requests or len(requests) > self.MAX_REQUESTS:
            raise ValueError(f"a batch needs 1..{self.MAX_REQUESTS} requests, got {len(requests)}")
        if self.transport is _gemini_httpx_transport and not self.cfg.api_base:
            client = genai.Client(api_key=_key_for(self.cfg, "GEMINI_API_KEY"))
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
                for cid, prompt in requests:
                    fh.write(json.dumps({"key": cid, "request": gemini_request_body(self.cfg, prompt)}, ensure_ascii=False) + "\n")
                tpath = fh.name
            try:
                up = client.files.upload(file=tpath, config=types.UploadFileConfig(mime_type="jsonl"))
            finally:
                try:
                    os.remove(tpath)
                except OSError:
                    pass
            bjob = None
            for attempt in range(30):
                try:
                    bjob = client.batches.create(model=self.cfg.model, src=up.name)
                    break
                except Exception as exc:
                    if ("429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)) and attempt < 29:
                        self.sleep(30.0)
                        continue
                    raise
            if not bjob or not bjob.name:
                raise ValueError(f"Gemini batch submit did not return a batch id: {bjob}")
            return bjob.name
        model_name = self.cfg.model if self.cfg.model.startswith("models/") else f"models/{self.cfg.model}"
        payload = gemini_batch_payload(self.cfg, requests)
        data = self._call("POST", f"{self.base}/{model_name}:batchGenerateContent", payload)
        bid = data.get("name") or data.get("id") or ""
        if not bid:
            raise ValueError(f"Gemini batch submit did not return a batch id: {data}")
        return bid

    def status(self, batch_id: str) -> dict:
        bname = batch_id.split("/")[-1]
        data = self._call("GET", f"{self.base}/batches/{bname}")
        raw_state = (
            data.get("state")
            or (data.get("metadata") or {}).get("state")
            or data.get("processing_status")
            or ""
        )
        ended_states = {
            "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
            "JOB_STATE_PARTIALLY_SUCCEEDED", "BATCH_STATE_SUCCEEDED", "BATCH_STATE_FAILED",
            "BATCH_STATE_CANCELLED", "BATCH_STATE_EXPIRED", "BATCH_STATE_PARTIALLY_SUCCEEDED",
            "SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED", "ENDED",
        }
        is_done = bool(data.get("done")) or str(raw_state).upper() in ended_states
        data["processing_status"] = "ended" if is_done else "in_progress"
        data["id"] = data.get("id") or data.get("name") or batch_id
        return data

    def wait(self, batch_id: str, poll_s: float = 60.0, max_wait_s: float = 26 * 3600.0,
             on_poll: Callable[[dict], None] | None = None) -> dict:
        start = self.clock()
        while True:
            st = self.status(batch_id)
            if on_poll:
                on_poll(st)
            if st.get("processing_status") == "ended":
                return st
            if self.clock() - start > max_wait_s:
                raise BatchTimeout(f"batch {batch_id} still {st.get('processing_status')!r} after {max_wait_s:.0f}s")
            self.sleep(poll_s)

    def results(self, batch_id: str, status: dict | None = None) -> dict[str, BatchResult]:
        st = status or self.status(batch_id)
        resp_obj = st.get("response") or {}
        responses = (
            (st.get("metadata") or {}).get("output", {}).get("inlinedResponses", {}).get("inlinedResponses")
            or (resp_obj.get("inlinedResponses") or {}).get("inlinedResponses")
            or resp_obj.get("inlined_responses")
            or (st.get("dest") or {}).get("inlined_responses")
            or st.get("inlined_responses")
            or st.get("inlinedResponses")
            or st.get("results")
            or st.get("responses")
        )
        if responses is None:
            res_url = st.get("results_url")
            fname = (
                (st.get("dest") or {}).get("file_name")
                or (st.get("dest") or {}).get("fileName")
                or (resp_obj.get("responsesFile"))
                or st.get("file_name")
                or (st.get("metadata") or {}).get("output", {}).get("responsesFile")
            )
            if fname and self.transport is _gemini_httpx_transport and not self.cfg.api_base:
                client = genai.Client(api_key=_key_for(self.cfg, "GEMINI_API_KEY"))
                raw_bytes = client.files.download(file=fname)
                raw_str = raw_bytes.decode("utf-8") if isinstance(raw_bytes, (bytes, bytearray)) else str(raw_bytes)
                responses = [json.loads(line) for line in raw_str.splitlines() if line.strip()]
            else:
                if not res_url and fname:
                    fname = fname.split("/")[-1]
                    res_url = f"{self.base}/files/{fname}"
                if res_url:
                    raw = self._call("GET", res_url)
                    if isinstance(raw, str):
                        responses = [json.loads(line) for line in raw.splitlines() if line.strip()]
                    elif isinstance(raw, dict):
                        responses = raw.get("responses") or raw.get("inlined_responses") or [raw]
                    elif isinstance(raw, list):
                        responses = raw
        if responses is None:
            responses = []

        out: dict[str, BatchResult] = {}
        for item in responses:
            cid = (
                item.get("key")
                or (item.get("metadata") or {}).get("custom_id")
                or item.get("custom_id")
                or (item.get("metadata") or {}).get("id")
                or item.get("id")
            )
            if not cid:
                continue
            err = item.get("error")
            if err:
                err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                out[cid] = BatchResult(cid, False, error=err_msg)
                continue
            resp = item.get("response") or item
            try:
                text, version = gemini_response_to_verdict(resp)
                out[cid] = BatchResult(cid, True, text, version)
            except ValueError as e:
                out[cid] = BatchResult(cid, False, error=str(e))
        return out
