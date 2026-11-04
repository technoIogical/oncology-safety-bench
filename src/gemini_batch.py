from __future__ import annotations

import base64
from enum import Enum
import hashlib
import inspect
import json
import os
import tempfile
import time
import typing
from typing import Any, Callable, Sequence

from google.genai import models, types
try:
    from google.genai._api_client import BaseApiClient
except ImportError:
    BaseApiClient = None
from pydantic import BaseModel

from src.claude_batch import BatchPending, BatchResultMissing, BatchStore

STORE_ENV = "GEMINI_BATCH_STORE"
MODE_ENV = "GEMINI_BATCH_MODE"
MODES = ("record", "replay")


def request_key(body: dict) -> str:
    serialized = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "model_dump"):
        return _to_json_serializable(obj.model_dump(mode="json", exclude_none=True, by_alias=True))
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    return obj


def gemini_build_request(model_name: str, contents: list, config, api_client=None) -> dict:
    if api_client is None and BaseApiClient is not None:
        try:
            api_client = BaseApiClient()
        except Exception:
            api_client = None

    params = types._GenerateContentParameters(model=model_name, contents=contents, config=config)
    try:
        mldev_req = models._GenerateContentParameters_to_mldev(api_client, params)
    except Exception:
        mldev_req = params.model_dump(mode="json", exclude_none=True, by_alias=True)

    if isinstance(mldev_req, dict):
        mldev_req.pop("_url", None)

    clean_req = _to_json_serializable(mldev_req)
    return {"model": model_name, "request": clean_req}


def _prune_unknown(data: Any, model_cls: Any) -> Any:
    if not isinstance(data, dict) or not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
        return data
    allowed = {}
    for fname, finfo in model_cls.model_fields.items():
        allowed[fname] = finfo.annotation
        if finfo.alias:
            allowed[finfo.alias] = finfo.annotation
    out = {}
    for k, v in data.items():
        if k not in allowed:
            continue
        ann = allowed[k]
        args = typing.get_args(ann)
        non_none = [a for a in args if a is not type(None)]
        target = non_none[0] if len(non_none) == 1 else ann
        if typing.get_origin(target) in (list, typing.Sequence) and isinstance(v, list):
            elem = typing.get_args(target)[0] if typing.get_args(target) else None
            out[k] = [_prune_unknown(x, elem) for x in v]
        else:
            out[k] = _prune_unknown(v, target)
    return out


def gemini_response_from_dict(raw_response_dict: dict, api_client=None) -> types.GenerateContentResponse | None:
    if raw_response_dict is None:
        return None
    if isinstance(raw_response_dict, types.GenerateContentResponse):
        return raw_response_dict

    try:
        sig = inspect.signature(models._GenerateContentResponse_from_mldev)
        if len(sig.parameters) > 0 and list(sig.parameters.keys())[0] == "api_client":
            snake = models._GenerateContentResponse_from_mldev(api_client, raw_response_dict)
        else:
            snake = models._GenerateContentResponse_from_mldev(raw_response_dict)
        return types.GenerateContentResponse.model_validate(_prune_unknown(snake, types.GenerateContentResponse))
    except Exception:
        return types.GenerateContentResponse.model_validate(_prune_unknown(raw_response_dict, types.GenerateContentResponse))


class GeminiCandidateBatchClient:
    MAX_REQUESTS = 100_000

    def __init__(self, api_key: str | None = None, client: Any | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.sleep = sleep
        self.clock = clock
        if client is not None:
            self.client = client
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if key:
                from google import genai
                self.client = genai.Client(api_key=key)
            else:
                self.client = None

    def submit(self, requests: Sequence[tuple[str, dict]], batch_size: int = 350) -> list[tuple[str, list[str]]]:
        if not requests or len(requests) > self.MAX_REQUESTS:
            raise ValueError(f"a batch needs 1..{self.MAX_REQUESTS} requests, got {len(requests)}")
        if self.client is None:
            raise ValueError("GEMINI_API_KEY is required to submit Gemini batches")

        grouped: dict[str, list[tuple[str, dict]]] = {}
        for key, body in requests:
            model = body.get("model", "gemini-4-argon")
            grouped.setdefault(model, []).append((key, body))

        out_batches = []
        for model, model_requests in grouped.items():
            for i in range(0, len(model_requests), batch_size):
                chunk = model_requests[i:i + batch_size]
                batch_id = self._submit_chunk(model, chunk)
                out_batches.append((batch_id, [k for k, _ in chunk]))

        return out_batches

    def _submit_chunk(self, model: str, chunk: list[tuple[str, dict]]) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
            for key, body in chunk:
                req_dict = body.get("request", body)
                fh.write(json.dumps({"key": key, "request": req_dict}, ensure_ascii=False) + "\n")
            tpath = fh.name

        try:
            up = self.client.files.upload(file=tpath, config=types.UploadFileConfig(mime_type="jsonl"))
        finally:
            try:
                os.remove(tpath)
            except OSError:
                pass

        bjob = None
        for attempt in range(30):
            try:
                bjob = self.client.batches.create(model=model, src=up.name)
                break
            except Exception as exc:
                if ("429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)) and attempt < 29:
                    self.sleep(30.0)
                    continue
                raise

        if not bjob or not getattr(bjob, "name", None):
            raise ValueError(f"Gemini batch submit did not return a batch id: {bjob}")
        return bjob.name

    def status(self, batch_id: str) -> dict:
        bjob = self.client.batches.get(name=batch_id)
        raw_state = getattr(bjob, "state", None) or ""
        if isinstance(raw_state, Enum):
            raw_state = raw_state.value
        raw_state_str = str(raw_state).upper()

        ended_states = {
            "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
            "JOB_STATE_PARTIALLY_SUCCEEDED", "SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED", "ENDED",
        }
        is_done = raw_state_str in ended_states

        dest = getattr(bjob, "dest", None)
        if hasattr(dest, "model_dump"):
            dest_dict = dest.model_dump(mode="json", exclude_none=True)
        elif isinstance(dest, dict):
            dest_dict = dest
        elif dest is not None:
            dest_dict = {
                "file_name": getattr(dest, "file_name", getattr(dest, "fileName", None)),
                "inlined_responses": getattr(dest, "inlined_responses", getattr(dest, "inlinedResponses", None)),
            }
        else:
            dest_dict = {}

        error_val = getattr(bjob, "error", None)
        if hasattr(error_val, "model_dump"):
            error_dict = error_val.model_dump(mode="json", exclude_none=True)
        elif isinstance(error_val, dict):
            error_dict = error_val
        elif error_val is not None:
            error_dict = {"message": str(error_val)}
        else:
            error_dict = None

        return {
            "id": batch_id,
            "name": getattr(bjob, "name", batch_id),
            "state": raw_state_str,
            "processing_status": "ended" if is_done else "in_progress",
            "dest": dest_dict,
            "error": error_dict,
        }

    def wait(self, batch_id: str, poll_s: float = 30.0, max_wait_s: float = 26 * 3600.0,
             on_poll: Callable[[dict], None] | None = None) -> dict:
        start = self.clock()
        while True:
            st = self.status(batch_id)
            if on_poll:
                on_poll(st)
            if st.get("processing_status") == "ended":
                return st
            if self.clock() - start > max_wait_s:
                raise TimeoutError(f"batch {batch_id} still {st.get('processing_status')!r} after {max_wait_s:.0f}s")
            self.sleep(poll_s)

    def results(self, batch_id: str, status_dict: dict | None = None) -> dict[str, dict]:
        st = status_dict or self.status(batch_id)
        dest = st.get("dest") or {}
        inlined = dest.get("inlined_responses") or dest.get("inlinedResponses")
        responses = None

        if inlined:
            responses = inlined
        else:
            fname = dest.get("file_name") or dest.get("fileName")
            if fname:
                raw_bytes = self.client.files.download(file=fname)
                raw_str = raw_bytes.decode("utf-8") if isinstance(raw_bytes, (bytes, bytearray)) else str(raw_bytes)
                responses = [json.loads(line) for line in raw_str.splitlines() if line.strip()]

        if responses is None:
            responses = []

        out: dict[str, dict] = {}
        for item in responses:
            cid = item.get("key") or (item.get("metadata") or {}).get("custom_id") or item.get("custom_id") or item.get("id")
            if not cid:
                continue
            out[cid] = item
        return out


def store_from_env() -> tuple[BatchStore, str] | None:
    root = os.environ.get(STORE_ENV)
    if not root:
        return None
    mode = os.environ.get(MODE_ENV, "replay")
    if mode not in MODES:
        raise ValueError(f"{MODE_ENV} must be one of {MODES}, got {mode!r}")
    return BatchStore(root), mode


def make_call(store: BatchStore, mode: str) -> Callable:
    def call(model_name: str, contents: list, config, api_client=None) -> tuple[types.GenerateContentResponse | dict, bool]:
        body = gemini_build_request(model_name, contents, config, api_client=api_client)
        key = request_key(body)
        if store.has_result(key):
            raw = store.load_result(key)
            if isinstance(raw, dict) and raw.get("_context_overflow"):
                return raw, True
            if isinstance(raw, dict) and (raw.get("candidates") or (raw.get("response") or {}).get("candidates")):
                return gemini_response_from_dict(raw, api_client=api_client), True
            store.forget_result(key)
        if mode == "record":
            store.save_request(key, body)
            raise BatchPending(key)
        raise BatchResultMissing(f"no gemini batch result for request {key[:12]} (record, submit and collect it first)")

    return call


def submit_pending(store: BatchStore, client: GeminiCandidateBatchClient, batch_size: int = 350,
                   max_requests: int | None = None) -> dict:
    todo = [(k, b) for k, b in store.pending() if k not in store.in_flight_keys()]
    if max_requests:
        todo = todo[:max_requests]
    if not todo:
        return {"submitted": 0, "batches": []}

    submitted_batches = client.submit(todo, batch_size=batch_size)
    for bid, keys in submitted_batches:
        store.add_batch(bid, keys)

    batch_ids = [bid for bid, _ in submitted_batches]
    return {
        "submitted": len(todo),
        "batch_ids": batch_ids,
        "batch_id": batch_ids[0] if len(batch_ids) == 1 else None,
        "batches": [{"id": bid, "count": len(keys)} for bid, keys in submitted_batches],
    }


def collect_results(store: BatchStore, client: GeminiCandidateBatchClient, poll_s: float = 30.0,
                    wait: bool = True) -> dict:
    n = errors = missing = 0
    for b in store.batches():
        if b.get("collected"):
            continue
        if wait:
            st = client.wait(b["id"], poll_s=poll_s)
        else:
            st = client.status(b["id"])
            if st.get("processing_status") != "ended":
                continue

        got = client.results(b["id"], st)
        for key in b["keys"]:
            if key in got:
                item = got[key]
                resp = item.get("response")
                err = item.get("error")
                if resp and resp.get("candidates") and not err:
                    store.save_result(key, resp)
                    n += 1
                elif err and any(m in str(err).lower() for m in ("exceeds the maximum", "too many tokens", "context length", "prompt is too long")):
                    store.save_result(key, {"_context_overflow": True, "error": err})
                    n += 1
                else:
                    errors += 1
            else:
                missing += 1
        store.mark_collected(b["id"])

    return {"collected": n, "errors": errors, "missing": missing}
