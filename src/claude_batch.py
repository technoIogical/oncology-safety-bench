from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Callable

STORE_ENV, MODE_ENV = "CLAUDE_BATCH_STORE", "CLAUDE_BATCH_MODE"
MODES = ("record", "replay")


class BatchPending(BaseException):

    def __init__(self, key: str):
        super().__init__(f"batch request {key[:12]} has no result yet")
        self.key = key


class BatchResultMissing(RuntimeError):
    pass


class BatchStore:
    def __init__(self, root: str):
        self.root = root
        self.req_dir, self.res_dir = os.path.join(root, "requests"), os.path.join(root, "results")
        os.makedirs(self.req_dir, exist_ok=True)
        os.makedirs(self.res_dir, exist_ok=True)

    def _path(self, kind_dir: str, key: str) -> str:
        return os.path.join(kind_dir, f"{key}.json")

    def has_result(self, key: str) -> bool:
        return os.path.exists(self._path(self.res_dir, key))

    def load_result(self, key: str) -> dict:
        with open(self._path(self.res_dir, key), encoding="utf-8") as f:
            return json.load(f)

    def save_request(self, key: str, body: dict) -> None:
        p = self._path(self.req_dir, key)
        if not os.path.exists(p):
            self._write(p, {"key": key, "body": body})

    def save_result(self, key: str, message: dict) -> None:
        self._write(self._path(self.res_dir, key), message)

    def forget_result(self, key: str) -> None:
        try:
            os.remove(self._path(self.res_dir, key))
        except OSError:
            pass

    def pending(self) -> list[tuple[str, dict]]:
        out = []
        for name in sorted(os.listdir(self.req_dir)):
            key = name[:-5]
            if name.endswith(".json") and not self.has_result(key):
                with open(os.path.join(self.req_dir, name), encoding="utf-8") as f:
                    out.append((key, json.load(f)["body"]))
        return out

    @staticmethod
    def _write(path: str, obj) -> None:
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @property
    def _batches_path(self) -> str:
        return os.path.join(self.root, "batches.json")

    def batches(self) -> list[dict]:
        try:
            with open(self._batches_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return []

    def add_batch(self, batch_id: str, keys: list[str]) -> None:
        self._write(self._batches_path, self.batches() + [{"id": batch_id, "keys": keys, "submitted_at": time.time()}])

    def in_flight_keys(self) -> set[str]:
        return {k for b in self.batches() if not b.get("collected") for k in b["keys"]}

    def mark_collected(self, batch_id: str) -> None:
        self._write(self._batches_path, [{**b, "collected": True} if b["id"] == batch_id else b for b in self.batches()])


class BatchClient:
    MAX_REQUESTS = 10_000

    def __init__(self, api_key: str, base: str = "https://api.anthropic.com", transport: Callable | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        from src.harness_judge import _httpx_transport
        self.key, self.base = api_key, base.rstrip("/")
        self.transport, self.sleep, self.clock = transport or _httpx_transport, sleep, clock

    def _call(self, method: str, url: str, body=None):
        headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        return self.transport(method, url, headers, body, 300.0)

    def submit(self, requests: list[tuple[str, dict]]) -> str:
        if not requests or len(requests) > self.MAX_REQUESTS:
            raise ValueError(f"a batch needs 1..{self.MAX_REQUESTS} requests, got {len(requests)}")
        data = self._call("POST", f"{self.base}/v1/messages/batches",
                          {"requests": [{"custom_id": k, "params": b} for k, b in requests]})
        return data["id"]

    def wait(self, batch_id: str, poll_s: float = 60.0, max_wait_s: float = 26 * 3600.0) -> dict:
        start = self.clock()
        while True:
            st = self._call("GET", f"{self.base}/v1/messages/batches/{batch_id}")
            if st.get("processing_status") == "ended":
                return st
            if self.clock() - start > max_wait_s:
                raise TimeoutError(f"batch {batch_id} still {st.get('processing_status')!r} after {max_wait_s:.0f}s")
            self.sleep(poll_s)

    def results(self, batch_id: str, status: dict | None = None) -> dict[str, dict]:
        st = status or self._call("GET", f"{self.base}/v1/messages/batches/{batch_id}")
        raw = self._call("GET", st.get("results_url") or f"{self.base}/v1/messages/batches/{batch_id}/results")
        out: dict[str, dict] = {}
        for line in (raw.splitlines() if isinstance(raw, str) else list(raw)):
            if isinstance(line, str):
                if not line.strip():
                    continue
                line = json.loads(line)
            res, cid = line.get("result") or {}, line.get("custom_id")
            if res.get("type") == "succeeded":
                out[cid] = res.get("message") or {}
            elif res.get("type") == "errored":
                out[cid] = {"type": "error", "error": res.get("error") or {"message": "batch request errored"}}
        return out


def request_key(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def store_from_env() -> tuple[BatchStore, str] | None:
    root = os.environ.get(STORE_ENV)
    if not root:
        return None
    mode = os.environ.get(MODE_ENV, "replay")
    if mode not in MODES:
        raise ValueError(f"{MODE_ENV} must be one of {MODES}, got {mode!r}")
    return BatchStore(root), mode


def make_post(store: BatchStore, mode: str) -> Callable:

    def post(url, headers, body, timeout):
        key = request_key(body)
        if store.has_result(key):
            return {**store.load_result(key), "_batch": True}
        if mode == "record":
            store.save_request(key, body)
            raise BatchPending(key)
        raise BatchResultMissing(f"no batch result for request {key[:12]} (record, submit and collect it first)")

    return post


def submit_pending(store: BatchStore, client: BatchClient, max_requests: int | None = None) -> dict:
    todo = [(k, b) for k, b in store.pending() if k not in store.in_flight_keys()]
    if max_requests:
        todo = todo[:max_requests]
    if not todo:
        return {"submitted": 0, "batch_id": None}
    batch_id = client.submit(todo)
    store.add_batch(batch_id, [k for k, _ in todo])
    return {"submitted": len(todo), "batch_id": batch_id}


def collect_results(store: BatchStore, client: BatchClient, poll_s: float = 60.0) -> dict:
    n = errors = missing = 0
    for b in store.batches():
        if b.get("collected"):
            continue
        st = client.wait(b["id"], poll_s=poll_s)
        got = client.results(b["id"], st)
        for key in b["keys"]:
            if key in got:
                store.save_result(key, got[key])
                n += 1
                errors += got[key].get("type") == "error"
            else:
                missing += 1
        store.mark_collected(b["id"])
    return {"collected": n, "errors": errors, "missing": missing}
