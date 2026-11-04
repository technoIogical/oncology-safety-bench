"""Unified inference router for local vLLM (OpenAI-compatible), Google Gemini, and Anthropic Claude models."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid

from google import genai
from google.genai import types
from google.genai.errors import APIError
import httpx
from openai import OpenAI

from src.harness_config import MAX_ANSWER_TOKENS, PROVIDER_BACKOFF_S
from src.harness_utils import strip_reasoning
from src.model_registry import descriptor, get_spec, is_registered

GEMINI_REQUEST_TIMEOUT_MS = 300_000
LOCAL_REQUEST_TIMEOUT_S = float(os.environ.get("VLLM_REQUEST_TIMEOUT_S") or 600.0)
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_REQUEST_TIMEOUT_S = 900.0
MAX_REASONING_TEXT_CHARS = 50_000
TRANSPORT_ATTEMPTS = 5
BACKOFF_CAP_S = 60.0
GEMINI_BLOCK_REASONS = frozenset({"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY",
                                  "IMAGE_PROHIBITED_CONTENT", "LANGUAGE"})
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_CONTEXT_ERROR_MARKERS = ("prompt is too long", "context window", "context length", "maximum context", "too many tokens",
                          "request_too_large", "exceeds the maximum")


class ModelNotFoundError(RuntimeError):
    pass


class AnthropicHTTPError(RuntimeError):

    def __init__(self, status_code: int, body: str, retry_after: float | None = None):
        super().__init__(f"HTTP {status_code}: {body[:500]}")
        self.status_code, self.body, self.retry_after = status_code, body, retry_after


def _backoff(attempt: int) -> float:
    return min(BACKOFF_CAP_S, PROVIDER_BACKOFF_S * (2 ** attempt))


def _match_served_model(model_name: str, ids: list, base_url: str) -> str:
    if model_name in ids:
        return model_name
    if model_name not in ("local-model", "", None):
        tail = [i for i in ids if i.split("/")[-1].split("\\")[-1] == model_name]
        if len(tail) == 1:
            return tail[0]
        raise ModelNotFoundError(f"model {model_name!r} is not served at {base_url}/models (served: {ids})")
    raw_id = ids[0]
    if "/" in raw_id and "\\" not in raw_id:
        return raw_id
    return raw_id.split("/")[-1].split("\\")[-1]


def get_actual_model_name(model_name: str) -> str:
    if get_spec(model_name).provider != "local":
        return model_name

    base_url = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:12345/v1").rstrip("/")

    max_retries = 6
    for attempt in range(max_retries):
        try:
            resp = httpx.get(f"{base_url}/models", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                ids = [str(m.get("id")) for m in data.get("data") or [] if m.get("id")]
                if ids:
                    return _match_served_model(model_name, ids, base_url)
            elif resp.status_code == 503 and attempt < max_retries - 1:
                print(f"  [Model Detection] Server still loading into VRAM... waiting 10s (Attempt {attempt+1}/{max_retries})")
                time.sleep(10)
                continue
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [Model Detection] Waiting for server to come online... (Attempt {attempt+1}/{max_retries})")
                time.sleep(10)
                continue
            pass

    if model_name in ("local-model", "", None):
        raise ModelNotFoundError(f"could not detect a loaded model at {base_url}/models (alias {model_name!r}); "
                                 f"start the server or pass the exact model id")
    return model_name


def extract_and_parse_json(text: str) -> dict:
    if not text:
        return None

    start_idx = text.find("{")
    if start_idx == -1:
        return None

    raw_json = text[start_idx:]

    if "```" in raw_json:
        markdown_end = raw_json.find("```")
        if markdown_end != -1:
            raw_json = raw_json[:markdown_end]

    raw_json = raw_json.strip()

    in_string = False
    escape = False
    stack = []
    balanced_json = ""
    for char in raw_json:
        if char == '"' and not escape:
            in_string = not in_string
        if char == '\\' and in_string:
            escape = not escape
        else:
            escape = False

        if not in_string:
            if char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in "}]":
                if stack:
                    if (char == "}" and stack[-1] == "}") or (char == "]" and stack[-1] == "]"):
                        stack.pop()
        balanced_json += char

    if in_string:
        balanced_json += '"'
    for closing_char in reversed(stack):
        balanced_json += closing_char

    try:
        return json.loads(balanced_json)
    except json.JSONDecodeError:
        fixed_json = re.sub(r',\s*([\]}])', r'\1', balanced_json)
        try:
            return json.loads(fixed_json)
        except json.JSONDecodeError:
            return None


def _message_reasoning(message) -> str:
    for attr in ("reasoning_content", "reasoning"):
        v = getattr(message, attr, None)
        if v is None:
            extra = getattr(message, "model_extra", None)
            v = extra.get(attr) if isinstance(extra, dict) else None
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _split_inline_reasoning(text: str) -> tuple[str, str]:
    if not text:
        return text, ""
    low = text.lower()
    # vLLM reasoning parsers sometimes strip the opening <think> tag while leaving </think> in content.
    if "</think>" in low and "<think>" not in low:
        cut = low.rfind("</think>")
        return text[cut + len("</think>"):].strip(), text[:cut].strip()
    clean = strip_reasoning(text)
    if clean == text.strip():
        return text, ""
    return clean, " ".join(m.strip() for m in _THINK_BLOCK.findall(text))


def _anthropic_http_post(url: str, headers: dict, json: dict, timeout: float) -> dict:
    r = httpx.post(url, headers=headers, json=json, timeout=timeout)
    if r.status_code >= 400:
        try:
            ra = float(r.headers.get("retry-after")) if r.headers.get("retry-after") else None
        except ValueError:
            ra = None
        raise AnthropicHTTPError(r.status_code, r.text, ra)
    return r.json()


def _is_context_error(e: Exception) -> bool:
    status = ModelClientRouter._status_of(e)
    text = str(getattr(e, "body", "") or e).lower()
    return status == 413 or (status in (None, 400) and any(m in text for m in _CONTEXT_ERROR_MARKERS))


def _anthropic_tools(tools) -> list:
    out = []
    for t in tools or []:
        if t.get("type") == "function":
            f = t["function"]
            out.append({"name": f["name"], "description": f.get("description", ""),
                        "input_schema": f.get("parameters") or {"type": "object", "properties": {}}})
    return out


def _tool_use_id(raw, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", str(raw or ""))[:64]
    return cleaned or fallback


def _thinking_blocks_of(tool_calls, lookup=None) -> list:
    blocks, seen = [], set()
    for tc in tool_calls or []:
        for b in tc.get("thinking_blocks") or (lookup or {}).get(tc.get("id")) or []:
            key = b.get("signature") or b.get("data") or b.get("thinking")
            if key in seen:
                continue
            seen.add(key)
            if b.get("type") == "redacted_thinking":
                blocks.append({"type": "redacted_thinking", "data": b.get("data", "")})
            else:
                blocks.append({"type": "thinking", "thinking": b.get("thinking", ""), "signature": b.get("signature", "")})
    return blocks


def anthropic_messages(messages: list, thinking_by_id: dict | None = None) -> tuple[list, list]:
    system, out = [], []

    def push(role: str, blocks: list):
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for n, m in enumerate(messages):
        role, content = m.get("role"), m.get("content")
        text = content if isinstance(content, str) else ("" if content is None else json.dumps(content))
        if role == "system":
            if text.strip():
                system.append({"type": "text", "text": text})
        elif role == "user":
            if text.strip():
                push("user", [{"type": "text", "text": text}])
        elif role == "assistant":
            blocks = _thinking_blocks_of(m.get("tool_calls"), thinking_by_id)
            if text.strip():
                blocks.append({"type": "text", "text": text})
            for i, tc in enumerate(m.get("tool_calls") or []):
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except json.JSONDecodeError:
                        args = {}
                blocks.append({"type": "tool_use", "id": _tool_use_id(tc.get("id"), f"toolu_{n}_{i}"),
                               "name": fn.get("name"), "input": args if isinstance(args, dict) else {}})
            if blocks:
                push("assistant", blocks)
        elif role == "tool":
            push("user", [{"type": "tool_result", "tool_use_id": _tool_use_id(m.get("tool_call_id"), f"toolu_{n}_0"),
                           "content": text or "(empty result)"}])
    for i, msg in enumerate(out):
        if msg["role"] == "user":
            msg["content"].sort(key=lambda b: 0 if b["type"] == "tool_result" else 1)
    open_ids: list = []
    for i, msg in enumerate(out):
        if msg["role"] == "user":
            answered = {b["tool_use_id"] for b in msg["content"] if b["type"] == "tool_result"}
            for b in list(msg["content"]):
                if b["type"] == "tool_result" and b["tool_use_id"] not in open_ids:
                    msg["content"][msg["content"].index(b)] = {"type": "text", "text": str(b["content"])}
            missing = [t for t in open_ids if t not in answered]
            for t in reversed(missing):
                msg["content"].insert(0, {"type": "tool_result", "tool_use_id": t, "content": "(no result recorded)"})
            open_ids = []
        else:
            open_ids = [b["id"] for b in msg["content"] if b["type"] == "tool_use"]
    if open_ids:
        out.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": t, "content": "(no result recorded)"}
                                                for t in open_ids]})
    return system, out


def anthropic_build_request(messages: list, tools, spec, temperature: float, *, cache: bool | None = None,
                            fallback_tools=None, thinking_by_id: dict | None = None) -> dict:
    system, conv = anthropic_messages(messages, thinking_by_id)
    api_tools = _anthropic_tools(tools)
    has_tool_blocks = any(b["type"] in ("tool_use", "tool_result") for m in conv for b in m["content"])
    body: dict = {"model": spec.api_model_id, "messages": conv}
    if api_tools:
        body["tools"] = api_tools
    elif has_tool_blocks:
        body["tools"] = _anthropic_tools(fallback_tools)
        if body["tools"]:
            body["tool_choice"] = {"type": "none"}
        else:
            for m in conv:
                m["content"] = [{"type": "text", "text": json.dumps(b)} if b["type"] in ("tool_use", "tool_result") else b
                                for b in m["content"]]
    if system:
        body["system"] = system
    thinking = dict(spec.thinking) if spec.thinking else None
    max_tokens = MAX_ANSWER_TOKENS
    if thinking and thinking.get("type") == "adaptive":
        body["thinking"] = {"type": "adaptive"}
        if thinking.get("effort"):
            body["output_config"] = {"effort": thinking["effort"]}
        max_tokens += int(thinking.get("allowance_tokens") or 0)
        body["temperature"] = 1.0
    elif thinking:
        body["thinking"] = thinking
        max_tokens += int(thinking.get("budget_tokens") or 0)
        body["temperature"] = 1.0
    else:
        body["temperature"] = temperature
    body["max_tokens"] = max_tokens
    use_cache = bool(body.get("tools")) if cache is None else cache
    if use_cache:
        # Place ephemeral cache breakpoints on the system prompt and the latest tool_result turn.
        if system:
            system[-1]["cache_control"] = {"type": "ephemeral"}
        for m in reversed(conv):
            results = [b for b in m["content"] if b["type"] == "tool_result"]
            if results:
                results[-1]["cache_control"] = {"type": "ephemeral"}
                break
    return body


def anthropic_parse_response(data: dict, spec, params: dict) -> tuple:
    if not isinstance(data, dict) or data.get("type") == "error":
        err = (data or {}).get("error", data) if isinstance(data, dict) else data
        return None, [], {"error": f"Error communicating with Claude: {err}", "error_type": "api", "finish_reason": "",
                          "request_params": params}
    u = data.get("usage") or {}
    cache_read = int(u.get("cache_read_input_tokens") or 0)
    cache_write = int(u.get("cache_creation_input_tokens") or 0)
    out_tokens = int(u.get("output_tokens") or 0)
    details = u.get("output_tokens_details") if isinstance(u.get("output_tokens_details"), dict) else {}
    thinking_tokens = u.get("thinking_tokens", details.get("thinking_tokens"))
    thoughts = int(thinking_tokens) if isinstance(thinking_tokens, int) else 0
    stop = str(data.get("stop_reason") or "")
    usage = {
        "prompt_tokens": int(u.get("input_tokens") or 0) + cache_read + cache_write,
        "completion_tokens": max(0, out_tokens - thoughts),
        "thoughts_tokens": thoughts,
        "cache_read_tokens": cache_read, "cache_write_tokens": cache_write,
        "finish_reason": stop, "model_version": data.get("model") or spec.name, "response_id": data.get("id"),
        "request_params": params, "temperature": params.get("temperature"),
        "batch": bool(data.get("_batch")),
    }
    text_parts, tool_calls, thinking_blocks, thinking_chars = [], [], [], 0
    for b in data.get("content") or []:
        t = b.get("type")
        if t == "text":
            text_parts.append(b.get("text", ""))
        elif t == "thinking":
            thinking_blocks.append({"type": "thinking", "thinking": b.get("thinking", ""), "signature": b.get("signature", "")})
            thinking_chars += len(b.get("thinking", ""))
        elif t == "redacted_thinking":
            thinking_blocks.append({"type": "redacted_thinking", "data": b.get("data", "")})
        elif t == "tool_use":
            tool_calls.append({"id": b.get("id") or f"call_{b.get('name')}", "name": b.get("name"),
                               "arguments": b.get("input") if isinstance(b.get("input"), dict) else {}})
    if thinking_blocks:
        usage["thinking_chars"] = thinking_chars
        for tc in tool_calls:
            tc["thinking_blocks"] = thinking_blocks
    text = "".join(text_parts)
    has_text = bool(text.strip())
    if stop == "refusal":
        usage.update(error="Claude refused to answer (stop_reason refusal)", error_type="provider_refusal")
        return None, [], usage
    if stop == "model_context_window_exceeded":
        usage.update(error="the prompt plus the output do not fit the context window", error_type="context_overflow")
        return None, [], usage
    if stop == "max_tokens" and not has_text and not tool_calls:
        usage.update(error="Claude spent the output budget (thinking) before any text", error_type="truncated_no_output")
        return None, [], usage
    if stop == "max_tokens":
        tool_calls = []
    return (text if has_text else None), tool_calls, usage


def gemini_thinking_config(thinking: dict):
    if not thinking:
        return None
    fields = types.ThinkingConfig.model_fields
    if thinking.get("thinking_level") and "thinking_level" in fields:
        return types.ThinkingConfig(thinking_level=thinking["thinking_level"])
    budget = thinking.get("thinking_budget") or thinking.get("allowance_tokens")
    if budget and "thinking_budget" in fields:
        return types.ThinkingConfig(thinking_budget=int(budget))
    raise RuntimeError("the installed google-genai ThinkingConfig has neither thinking_level nor thinking_budget; "
                       "cannot apply the declared Gemini thinking setting")


def _empty_gemini_result(response, last_error: str = "", params: dict | None = None, *,
                         is_batch: bool | None = None) -> tuple[None, list, dict]:
    candidates = getattr(response, "candidates", None) if response else None
    finish = str(getattr(candidates[0], "finish_reason", "") or "").split(".")[-1].upper() if candidates else ""
    feedback = getattr(response, "prompt_feedback", None) if response else None
    block = str(getattr(feedback, "block_reason", "") or "").split(".")[-1].upper() if feedback else ""
    um = getattr(response, "usage_metadata", None) if response else None
    counts = {}
    if um is not None:
        for out_key, attr in (("prompt_tokens", "prompt_token_count"), ("completion_tokens", "candidates_token_count"),
                              ("thoughts_tokens", "thoughts_token_count")):
            v = getattr(um, attr, 0)
            counts[out_key] = v if isinstance(v, int) else 0
    extra = {**({"request_params": params} if params is not None else {}),
             **({"batch": is_batch} if is_batch is not None else {}),
             **counts}
    if finish == "MAX_TOKENS":
        return None, [], {"error": "Gemini spent the output budget (thinking tokens) before any text",
                          "error_type": "truncated_no_output", "finish_reason": finish, **extra}
    if finish in GEMINI_BLOCK_REASONS or (block and block != "BLOCK_REASON_UNSPECIFIED"):
        return None, [], {"error": f"Gemini blocked the answer ({finish or block})",
                          "error_type": "provider_refusal", "finish_reason": finish or block, **extra}
    err_msg = (f"Error: Empty response candidates from Gemini ({last_error})."
               if last_error else "Error: Empty response candidates from Gemini.")
    return None, [], {"error": err_msg, "error_type": "empty", "finish_reason": finish}


class ModelClientRouter:
    def __init__(self, model_name: str, *, anthropic_post=None):
        self.model_name = model_name or "local-model"
        self.spec = get_spec(self.model_name)
        self.provider = self.spec.provider
        self.is_gemini = self.provider == "gemini"
        self.is_claude = self.provider == "anthropic"
        self._anthropic_post = anthropic_post or _anthropic_http_post
        self._batch_post = None
        if self.is_claude and anthropic_post is None:
            from src.claude_batch import make_post, store_from_env
            env = store_from_env()
            if env is not None:
                self._batch_post = make_post(*env)
        self._anthropic_key = None
        self._anthropic_tools = None
        self._anthropic_thinking = {}

        self._gemini_batch_call = None
        if self.is_gemini:
            from src import gemini_batch
            env = gemini_batch.store_from_env()
            if env is not None:
                self._gemini_batch_call = gemini_batch.make_call(*env)
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                if self._gemini_batch_call is not None:
                    self.client = None
                else:
                    raise ValueError("GEMINI_API_KEY environment variable is required for Gemini models.")
            else:
                self.client = self._new_gemini_client(api_key)
        elif self.is_claude:
            self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            if not self._anthropic_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required for Claude models.")
            self.client = None
        else:
            base_url = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:12345/v1")
            self.client = OpenAI(
                base_url=base_url,
                api_key="lm-studio",
                max_retries=0,
                http_client=httpx.Client(timeout=httpx.Timeout(LOCAL_REQUEST_TIMEOUT_S, connect=30.0))
            )

    @staticmethod
    def _new_gemini_client(api_key: str):
        return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS))

    def generate_response(self, messages: list, tools: list = None, temperature: float = 0.1,
                          seed: int | None = None) -> tuple[str, list, dict]:
        if self.is_gemini:
            return self._call_gemini(messages, tools, temperature, seed=seed)
        if self.is_claude:
            return self._call_anthropic(messages, tools, temperature, seed=seed)
        return self._call_local_openai(messages, tools, temperature, seed=seed)

    def max_output_tokens(self) -> int:
        return MAX_ANSWER_TOKENS + self.spec.thinking_budget_tokens

    def request_params(self, temperature: float, seed=None) -> dict:
        return {"temperature": temperature, "sampling": dict(self.spec.sampling) or None,
                "max_output_tokens": "window" if self.spec.output_to_window else self.max_output_tokens(),
                "max_answer_tokens": MAX_ANSWER_TOKENS, "thinking": dict(self.spec.thinking) or None,
                "chat_template_kwargs": dict(self.spec.chat_template_kwargs) or None, "seed": seed}

    @staticmethod
    def _status_of(e: Exception):
        code = getattr(e, "status_code", None) or getattr(e, "code", None)
        resp = getattr(e, "response", None)
        if code is None and resp is not None:
            code = getattr(resp, "status_code", None)
        try:
            return int(code) if code is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _is_503(cls, e: Exception) -> bool:
        return cls._status_of(e) == 503 or "503" in str(e)

    @classmethod
    def _is_transient(cls, e: Exception) -> bool:
        status = cls._status_of(e)
        if status is not None:
            return status in (408, 409, 425, 429) or status >= 500
        text = f"{type(e).__name__} {e}".lower()
        return isinstance(e, (httpx.TransportError, ConnectionError, TimeoutError)) or any(
            k in text for k in ("connection", "timeout", "timed out", "temporarily", "unavailable", "overloaded",
                                "rate limit", "resource_exhausted", "503", "502", "504", "429"))

    def _call_local_openai(self, messages: list, tools: list, temperature: float,
                           seed: int | None = None) -> tuple[str, list, dict]:
        last_error = "unknown error"
        spec = self.spec
        send_tools = bool(tools) and spec.native_tools
        extra_body = {"cache_prompt": False, "slot_id": -1, **spec.sampling}
        if spec.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = dict(spec.chat_template_kwargs)
        params = self.request_params(temperature, seed)
        # Merge consecutive user messages so strict alternating-turn chat templates (Gemma, Mistral) do not fail.
        wire_messages = []
        for m in messages:
            if (wire_messages and m.get("role") == "user" and wire_messages[-1].get("role") == "user"
                    and isinstance(m.get("content"), str) and isinstance(wire_messages[-1].get("content"), str)):
                wire_messages[-1] = {**wire_messages[-1], "content": wire_messages[-1]["content"] + "\n\n" + m["content"]}
            else:
                wire_messages.append(m)
        for attempt in range(TRANSPORT_ATTEMPTS):
            try:
                kwargs = dict(model=self.model_name, messages=wire_messages, temperature=temperature, extra_body=extra_body)
                if not spec.output_to_window:
                    kwargs["max_tokens"] = self.max_output_tokens()
                if send_tools:
                    kwargs["tools"] = tools
                if seed is not None:
                    kwargs["seed"] = int(seed) + attempt
                response = self.client.chat.completions.create(**kwargs)

                if not response or not response.choices:
                    return None, [], {"error": "Error: Empty response from local model (empty choices list).",
                                      "error_type": "empty", "finish_reason": "", "request_params": params}

                ru = getattr(response, "usage", None)
                usage = {
                    "prompt_tokens": ru.prompt_tokens if ru else 0,
                    "completion_tokens": ru.completion_tokens if ru else 0,
                    "finish_reason": str(response.choices[0].finish_reason or ""),
                    "request_params": params,
                }
                served = getattr(response, "model", None)
                if isinstance(served, str):
                    usage["model_version"] = served
                details = getattr(ru, "completion_tokens_details", None) if ru else None
                rt = getattr(details, "reasoning_tokens", None)
                if isinstance(rt, int):
                    usage["reasoning_tokens"] = rt
                message = response.choices[0].message
                text_content = message.content
                reasoning = _message_reasoning(message)
                if isinstance(text_content, str):
                    text_content, inline = _split_inline_reasoning(text_content)
                    reasoning = reasoning or inline
                if reasoning:
                    usage["reasoning_chars"] = len(reasoning)
                    usage["reasoning_text"] = reasoning[:MAX_REASONING_TEXT_CHARS]
                tool_calls = []
                if message.tool_calls:
                    for tc in message.tool_calls:
                        args = tc.function.arguments
                        if not isinstance(args, str):
                            args = json.dumps(args) if args else "{}"
                        tool_calls.append({
                            "id": tc.id if tc.id else f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": args},
                        })
                text_content = str(text_content) if text_content is not None else ""
                if usage["finish_reason"].lower() == "content_filter" and not text_content.strip() and not tool_calls:
                    return None, [], {"error": "the provider's content filter blocked the answer",
                                      "error_type": "provider_refusal", "finish_reason": usage["finish_reason"],
                                      **{k: usage[k] for k in ("prompt_tokens", "completion_tokens")}}
                return text_content, tool_calls, usage

            except Exception as e:
                msg = str(e)
                last_error = f"{type(e).__name__}: {msg}"
                if "maximum context length" in msg.lower() or "context length" in msg.lower():
                    return None, [], {"error": f"Error communicating with local model: {msg}",
                                      "error_type": "context_overflow", "finish_reason": ""}
                is_read_timeout = isinstance(e, (httpx.ReadTimeout, TimeoutError)) or (
                    "timed out" in msg.lower() and not isinstance(e, httpx.ConnectTimeout)
                )
                if is_read_timeout:
                    if attempt < 1:
                        delay = _backoff(attempt)
                        print(f"  [Transport] {type(e).__name__} (attempt {attempt + 1}/2); retrying in {delay:.0f}s")
                        time.sleep(delay)
                        continue
                    return None, [], {"error": f"Error communicating with local model: {last_error}",
                                      "error_type": "timeout", "finish_reason": "", "request_params": params}
                if self._is_transient(e) and attempt < TRANSPORT_ATTEMPTS - 1:
                    delay = 10.0 if self._is_503(e) else _backoff(attempt)
                    print(f"  [Transport] {type(e).__name__} (attempt {attempt + 1}/{TRANSPORT_ATTEMPTS}); retrying in {delay:.0f}s")
                    time.sleep(delay)
                    continue
                return None, [], {"error": f"Error communicating with local model: {last_error}",
                                  "error_type": "boot" if self._is_503(e) else "api", "finish_reason": ""}

        return None, [], {"error": f"Error: transport retries exhausted ({last_error}).",
                          "error_type": "boot", "finish_reason": ""}

    def _call_anthropic(self, messages: list, tools: list, temperature: float,
                        seed: int | None = None) -> tuple[str, list, dict]:
        if tools:
            self._anthropic_tools = tools
        body = anthropic_build_request(messages, tools, self.spec, temperature, fallback_tools=self._anthropic_tools,
                                       thinking_by_id=self._anthropic_thinking)
        params = self.request_params(body["temperature"], seed)
        params["seed"] = None
        params["requested_temperature"] = temperature
        headers = {"x-api-key": self._anthropic_key, "anthropic-version": ANTHROPIC_VERSION,
                   "content-type": "application/json"}
        url = os.environ.get("ANTHROPIC_API_BASE", ANTHROPIC_BASE_URL).rstrip("/") + "/v1/messages"
        last_error = "unknown error"
        post = self._batch_post if (self._batch_post is not None and not tools) else self._anthropic_post
        for attempt in range(TRANSPORT_ATTEMPTS):
            try:
                data = post(url, headers, body, ANTHROPIC_REQUEST_TIMEOUT_S)
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if _is_context_error(e):
                    return None, [], {"error": f"Error communicating with Claude: {last_error}",
                                      "error_type": "context_overflow", "finish_reason": "", "request_params": params}
                if self._is_transient(e) and attempt < TRANSPORT_ATTEMPTS - 1:
                    delay = max(_backoff(attempt), min(BACKOFF_CAP_S, float(getattr(e, "retry_after", None) or 0)))
                    print(f"  [Transport] {type(e).__name__} (attempt {attempt + 1}/{TRANSPORT_ATTEMPTS}); retrying in {delay:.0f}s")
                    time.sleep(delay)
                    continue
                return None, [], {"error": f"Error communicating with Claude: {last_error}", "error_type": "api",
                                  "finish_reason": "", "request_params": params}
            text, calls, usage = anthropic_parse_response(data, self.spec, params)
            for tc in calls:
                blocks = tc.pop("thinking_blocks", None)
                if blocks:
                    self._anthropic_thinking[tc["id"]] = blocks
            return text, calls, usage
        return None, [], {"error": f"Error: transport retries exhausted ({last_error}).", "error_type": "api",
                          "finish_reason": "", "request_params": params}

    def _convert_schema(self, openai_schema: dict) -> dict:
        if not openai_schema:
            return None
        type_str = openai_schema.get("type", "string").upper()

        schema_dict = {"type": type_str}

        if "description" in openai_schema:
            schema_dict["description"] = openai_schema["description"]

        if type_str == "ARRAY" and "items" in openai_schema:
            schema_dict["items"] = self._convert_schema(openai_schema["items"])
        elif type_str == "OBJECT" and "properties" in openai_schema:
            props = {}
            for k, v in openai_schema["properties"].items():
                props[k] = self._convert_schema(v)
            schema_dict["properties"] = props
            if "required" in openai_schema:
                schema_dict["required"] = openai_schema["required"]

        return schema_dict

    def _call_gemini(self, messages: list, tools: list, temperature: float, seed: int | None = None) -> tuple[str, list, dict]:
        gemini_tools = []
        if tools:
            function_declarations = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool["function"]
                    func_decl = {"name": func["name"], "description": func.get("description", "")}
                    if "parameters" in func:
                        func_decl["parameters"] = self._convert_schema(func["parameters"])
                    function_declarations.append(func_decl)
            if function_declarations:
                gemini_tools = [{"function_declarations": function_declarations}]

        contents = []
        system_instruction = None
        for msg in messages:
            role = msg["role"]
            content = msg.get("content")

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))
            elif role == "assistant":
                parts = []
                if content:
                    parts.append(types.Part.from_text(text=content))
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        name = tc.get("name") or tc.get("function", {}).get("name")
                        args = tc.get("arguments", tc.get("function", {}).get("arguments", {}))
                        if isinstance(args, str):
                            args = json.loads(args or "{}")

                        part = types.Part.from_function_call(name=name, args=args)
                        if "id" in tc and isinstance(tc["id"], str):
                            part.function_call.id = tc["id"]
                        # Echo binary thought_signature back on toolCall parts so Gemini 3/4 thinking models accept multi-turn tool use.
                        if "thought_signature" in tc:
                            if isinstance(tc["thought_signature"], str):
                                part.thought_signature = bytes.fromhex(tc["thought_signature"])
                            else:
                                part.thought_signature = tc["thought_signature"]
                        parts.append(part)

                if parts:
                    contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                try:
                    resp_dict = json.loads(content)
                except Exception:
                    resp_dict = {"result": content}

                response_part = types.Part.from_function_response(name=msg["name"], response=resp_dict)
                prev = contents[-1] if contents else None
                if prev is not None and prev.role == "user" and prev.parts and all(
                        getattr(part, "function_response", None) for part in prev.parts):
                    prev.parts.append(response_part)
                else:
                    contents.append(types.Content(role="user", parts=[response_part]))

        config_kwargs = {"temperature": temperature, "max_output_tokens": self.max_output_tokens()}
        thinking_config = gemini_thinking_config(self.spec.thinking)
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config
        params = self.request_params(temperature, seed)
        if seed is not None:
            config_kwargs["seed"] = int(seed)
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_kwargs)

        response = None
        last_error = "unknown error"
        blocked_attempts = 0
        is_batch = False
        if self._gemini_batch_call is not None:
            from src.gemini_batch import BatchResultMissing
            try:
                response, is_batch = self._gemini_batch_call(
                    self.model_name, contents, config, getattr(self.client, "_api_client", None)
                )
            except BatchResultMissing as e:
                return None, [], {"error": f"Error communicating with Gemini model: {e}",
                                  "error_type": "api", "finish_reason": ""}
            if isinstance(response, dict) and response.get("_context_overflow"):
                return None, [], {"error": f"Error communicating with Gemini model: {response.get('error')}",
                                  "error_type": "context_overflow", "finish_reason": "", "request_params": params, "batch": True}
        else:
            for attempt in range(TRANSPORT_ATTEMPTS):
                try:
                    response = self.client.models.generate_content(model=self.model_name, contents=contents, config=config)
                except APIError as e:
                    last_error = f"{type(e).__name__}: {e}"
                    if _is_context_error(e):
                        return None, [], {"error": f"Error communicating with Gemini model: {last_error}",
                                          "error_type": "context_overflow", "finish_reason": "", "request_params": params}
                    if self._is_transient(e) and attempt < TRANSPORT_ATTEMPTS - 1:
                        time.sleep(_backoff(attempt) * 2)
                        continue
                    if not self._is_transient(e):
                        return None, [], {"error": f"Error communicating with Gemini model: {last_error}",
                                          "error_type": "api", "finish_reason": ""}
                    return None, [], {"error": f"Error communicating with Gemini model: {last_error}",
                                      "error_type": "api", "finish_reason": ""}
                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"
                    if attempt < TRANSPORT_ATTEMPTS - 1:
                        time.sleep(_backoff(attempt))
                        try:
                            api_key = os.environ.get("GEMINI_API_KEY")
                            if api_key:
                                self.client = self._new_gemini_client(api_key)
                        except Exception:
                            pass
                        continue
                    return None, [], {"error": f"Error communicating with Gemini model: {last_error}",
                                      "error_type": "api", "finish_reason": ""}

                candidates = getattr(response, "candidates", None) if response else None
                parts = None
                if candidates and getattr(candidates[0], "content", None) is not None:
                    parts = getattr(candidates[0].content, "parts", None)
                if parts:
                    break
                _, _, empty_usage = _empty_gemini_result(response, "", params)
                if empty_usage["error_type"] == "truncated_no_output":
                    return None, [], empty_usage
                if empty_usage["error_type"] == "provider_refusal":
                    blocked_attempts += 1
                    if blocked_attempts >= 2:
                        empty_usage.pop("request_params", None)
                        return None, [], empty_usage
                    continue
                last_error = "empty candidate list from Gemini"
                if attempt < TRANSPORT_ATTEMPTS - 1:
                    time.sleep(_backoff(attempt))
                    continue
                return None, [], empty_usage

        if response is None or not response.candidates or not response.candidates[0].content \
                or not response.candidates[0].content.parts:
            return _empty_gemini_result(response, last_error, params, is_batch=is_batch)

        um = getattr(response, "usage_metadata", None)
        usage = {
            "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0,
            "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0,
            "finish_reason": str(response.candidates[0].finish_reason or ""),
            "request_params": params,
        }
        thoughts = getattr(um, "thoughts_token_count", 0) if um is not None else 0
        usage["thoughts_tokens"] = thoughts if isinstance(thoughts, int) else 0
        served = getattr(response, "model_version", None)
        if isinstance(served, str):
            usage["model_version"] = served
        if is_batch:
            usage["batch"] = True

        text_content = ""
        tool_calls = []
        pending_thought_sig = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, "thought_signature") and part.thought_signature:
                ts_val = (part.thought_signature.hex()
                          if isinstance(part.thought_signature, bytes) else part.thought_signature)
                if tool_calls and "thought_signature" not in tool_calls[-1]:
                    tool_calls[-1]["thought_signature"] = ts_val
                else:
                    pending_thought_sig = ts_val

            if part.function_call:
                args = {}
                if part.function_call.args:
                    try:
                        args = dict(part.function_call.args)
                    except Exception:
                        try:
                            args = json.loads(json.dumps(part.function_call.args))
                        except Exception:
                            args = {}
                tool_call = {
                    "id": getattr(part.function_call, "id", f"call_{part.function_call.name}"),
                    "name": part.function_call.name,
                    "arguments": args,
                }
                ts = None
                if hasattr(part, "thought_signature") and part.thought_signature:
                    ts = (part.thought_signature.hex()
                          if isinstance(part.thought_signature, bytes) else part.thought_signature)
                elif pending_thought_sig:
                    ts = pending_thought_sig
                    pending_thought_sig = None
                if ts:
                    tool_call["thought_signature"] = ts
                tool_calls.append(tool_call)
            elif part.text:
                if getattr(part, "thought", None) is True:
                    continue
                text_content += part.text

        return text_content if text_content else None, tool_calls, usage


def get_serving_info(base_url: str | None = None, model_name: str | None = None, *, timeout: float = 5.0,
                     http_get=None, run_cmd=None) -> dict:
    base = (base_url or os.environ.get("VLLM_API_BASE", "http://127.0.0.1:12345/v1")).rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    info: dict = {"base_url": base, "vllm_version": None, "served_models": [], "max_model_len": None, "gpus": [],
                  "descriptor": None, "errors": [], "recorded_at_unix": time.time()}
    get = http_get or (lambda url, t: httpx.get(url, timeout=t))
    try:
        r = get(f"{root}/version", timeout)
        if r.status_code == 200:
            info["vllm_version"] = (r.json() or {}).get("version")
        else:
            info["errors"].append(f"/version HTTP {r.status_code}")
    except Exception as e:
        info["errors"].append(f"/version: {type(e).__name__}: {e}")
    served = None
    try:
        r = get(f"{base}/models", timeout)
        if r.status_code == 200:
            rows = [m for m in (r.json() or {}).get("data") or [] if m.get("id")]
            info["served_models"] = [str(m["id"]) for m in rows]
            pick = next((m for m in rows if model_name and str(m["id"]).split("/")[-1] == str(model_name).split("/")[-1]),
                        rows[0] if rows else None)
            if pick is not None:
                served = str(pick["id"])
                info["max_model_len"] = pick.get("max_model_len")
        else:
            info["errors"].append(f"/models HTTP {r.status_code}")
    except Exception as e:
        info["errors"].append(f"/models: {type(e).__name__}: {e}")
    try:
        run = run_cmd or (lambda argv: subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=True).stdout)
        out = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        for line in str(out).strip().splitlines():
            name, _, mem = line.partition(",")
            info["gpus"].append({"name": name.strip(), "memory_total": mem.strip()})
    except Exception as e:
        info["errors"].append(f"nvidia-smi: {type(e).__name__}: {e}")
    target = model_name if model_name not in (None, "", "local-model") else served
    if target:
        info["descriptor"] = descriptor(target)
        info["registered"] = is_registered(target)
    info["served_model"] = served
    return info
