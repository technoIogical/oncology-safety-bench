"""Grades cached unjudged model generations from generation_cache/ into a judged result CSV."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import json
import os
from pathlib import Path
import re
import sys
import threading
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_batch as rb
from src.evaluator import prepare_judge_prompt
from src.harness_config import EVIDENCE_ARMS, JUDGE_MODEL
from src.harness_data import expand_stages, load_harness_dataset
from src.harness_judge import (ENFORCEMENT, AnthropicBatch, BatchTimeout, GeminiBatch,
                               JudgeCache, JudgeConfig, cache_key, judge_config_from_spec,
                               primary_judge_config, prompt_hash)


def model_of(run_dir: str) -> str:
    return re.split(r"_h\d", os.path.basename(os.path.normpath(run_dir)))[0]


def done_ids(path: str) -> set[tuple[str, str, str]]:
    """Returns completed (case_id, replicate, variant) tuples from an output CSV after archiving retryable rows."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    retry = [r for r in rows if r.get("infra_failure_type") not in ("none", "", None)]
    if retry:
        keep = [r for r in rows if r.get("infra_failure_type") in ("none", "", None)]
        archive = path[:-4] + ".retried.csv"
        new_archive = not os.path.exists(archive)
        with open(archive, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
            if new_archive:
                w.writeheader()
            w.writerows(retry)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
            w.writeheader()
            w.writerows(keep)
        os.replace(tmp, path)
        rows = keep
    rb.forget_row_keys(path)
    return {
        (r["case_id"], str(r.get("replicate") or 0), r.get("variant") or "standard")
        for r in rows
    }


def batch_prefetch_cached(todo_items: Sequence[dict], cfg: JudgeConfig, cache_dir: str,
                           out_dir: str, *, batch_size: int = 2000, poll_s: float = 60.0,
                           max_wait_s: float = 26 * 3600.0, wait: bool = True,
                           transport: Callable | None = None, sleep: Callable[[float], None] | None = None,
                           log: Callable[..., None] = print) -> dict:
    """Submits uncached judge prompts for cached generations via the Batches API and stores responses in cache_dir."""
    cache = JudgeCache(cache_dir, "readwrite")
    kwargs = {"transport": transport} if transport is not None else {}
    if sleep is not None:
        kwargs["sleep"] = sleep
    if cfg.provider == "anthropic":
        client = AnthropicBatch(cfg, **kwargs)
        enforcement = "anthropic_forced_tool"
    elif cfg.provider == "gemini":
        client = GeminiBatch(cfg, **kwargs)
        enforcement = "gemini_response_schema"
    else:
        raise ValueError(f"batch mode not supported for provider {cfg.provider!r}")

    state_path = os.path.join(out_dir, "batches.json")
    state = json.loads(Path(state_path).read_text(encoding="utf-8")) if os.path.exists(state_path) else {"batches": []}

    def save():
        tmp = state_path + ".tmp"
        Path(tmp).write_text(json.dumps(state, indent=1), encoding="utf-8")
        os.replace(tmp, state_path)

    prompts: dict[str, tuple[str, int]] = {}
    for it in todo_items:
        if not rb.classify_run(it["g"].get("final_recommendation", ""), it["metrics"]).judged:
            continue
        ckey = it["cache_key"]
        if ckey not in prompts:
            prompts[ckey] = (it["prompt"], 0)

    stats = {"requests": len(prompts), "cached": 0, "submitted": 0, "collected": 0, "failed": 0, "batches": 0}

    def collect(entry) -> bool:
        st = client.status(entry["id"]) if not wait else client.wait(
            entry["id"], poll_s, max_wait_s,
            on_poll=lambda s: log(f"[batch {entry['id']}] {s.get('processing_status')} {s.get('request_counts', '')}"))
        if st.get("processing_status") != "ended":
            return False
        for key, res in client.results(entry["id"], st).items():
            val = prompts.get(key)
            if val is None:
                continue
            p, rep = val
            if res.ok:
                cache.put(key, {
                    "provider": cfg.provider, "model": cfg.model, "thinking_budget": cfg.thinking_budget,
                    "temperature": cfg.temperature, "replicate": rep, "prompt_sha256": prompt_hash(p),
                    "response_text": res.text, "model_version": res.model_version,
                    "enforcement": enforcement,
                })
                stats["collected"] += 1
            else:
                stats["failed"] += 1
                log(f"[batch {entry['id']}] request {key[:12]} failed: {res.error}")
        entry["collected"] = True
        save()
        return True

    pending = [b for b in state["batches"] if not b.get("collected")]
    in_flight = {k for b in pending for k in b["keys"]}
    for b in pending:
        if not collect(b):
            return {"pending": True, "batch_id": b["id"], **stats}

    todo_keys = [k for k in prompts if cache.get(k) is None and k not in in_flight]
    stats["cached"] = len(prompts) - len(todo_keys)
    for i in range(0, len(todo_keys), batch_size):
        chunk = todo_keys[i:i + batch_size]
        bid = client.submit([(k, prompts[k][0]) for k in chunk])
        entry = {"id": bid, "keys": chunk, "collected": False}
        state["batches"].append(entry)
        save()
        stats["submitted"] += len(chunk)
        stats["batches"] += 1
        log(f"submitted batch {bid} with {len(chunk)} requests")
        if not collect(entry):
            return {"pending": True, "batch_id": bid, **stats}

    return stats


def main(argv=None, transport=None, sleep=None) -> int:
    ap = argparse.ArgumentParser(description=None)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8, help="concurrent judge workers (default: 8)")
    ap.add_argument("--batch", action="store_true", help="prefetch judge prompts via Batches API before grading")
    ap.add_argument("--no-wait", action="store_true", help="with --batch: submit, then exit 5 until batches end")
    ap.add_argument("--batch-size", type=int, default=2000, help="batch chunk size (default: 2000)")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--max-wait-hours", type=float, default=26.0)
    ap.add_argument("--judge-model", default=None, help="judge model spec (e.g. gemini-4-argon, claude:claude-haiku-5-5)")
    args = ap.parse_args(argv)
    rb.prompt_fingerprint = rb.harness_fingerprint(rb.dataset_hash(args.dataset))

    if args.judge_model:
        cfg = judge_config_from_spec(args.judge_model)
    else:
        cfg = primary_judge_config(JUDGE_MODEL)

    cache_dir = os.path.join(args.out, "judge_cache")
    cases: dict[str, dict] = {}
    for c in expand_stages(load_harness_dataset(args.dataset), "both"):
        cid = str(c.get("case_id"))
        cases[cid] = c
        cases[re.sub(r"[^A-Za-z0-9_.-]", "_", cid)] = c

    skipped, todo = [], []
    for run_dir in args.run_dirs:
        model = model_of(run_dir)
        for arm_dir in sorted(glob.glob(os.path.join(run_dir, "generation_cache", "*"))):
            stem = os.path.basename(arm_dir)
            raw_arm = stem[:-4] if stem.endswith("_run") else stem
            is_v2_check = raw_arm.endswith("_v2")
            dir_arm = raw_arm[:-3] if is_v2_check else raw_arm
            dir_arm = "agent" if dir_arm == "agentic" else dir_arm
            csv_arm = f"{dir_arm}_v2" if is_v2_check else dir_arm
            fieldnames = rb.fieldnames_for_arm(dir_arm)
            fname = stem + ".csv"
            out_csv = os.path.join(args.out, model, fname)
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            if not os.path.exists(out_csv):
                with open(out_csv, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n").writeheader()
            already = done_ids(out_csv)
            for path in sorted(glob.glob(os.path.join(arm_dir, "*.json"))):
                g = json.load(open(path, encoding="utf-8"))
                m = dict(g.get("metrics") or {})
                raw_case_id = re.sub(r"(__r\d+)?\.json$", "", os.path.basename(path))
                case = cases.get(raw_case_id)
                if case is None or g.get("case_hash") != case.get("case_hash", ""):
                    skipped.append((model, fname, raw_case_id, "case missing or case_hash differs"))
                    continue
                case_id = str(case["case_id"])
                replicate = int(m.get("replicate") or 0)
                variant = m.get("variant") or "standard"
                if (case_id, str(replicate), variant) in already:
                    continue
                arm = csv_arm if is_v2_check else (m.get("arm") or dir_arm)
                m["arm"] = arm
                axis2 = (arm in EVIDENCE_ARMS) or (dir_arm in EVIDENCE_ARMS)
                bundle_ids = rb._supported_or_bundle(m, case)
                prep = prepare_judge_prompt(case["vignette"], case["ground_truth_rubric"],
                                            g["final_recommendation"], axis2, bundle_ids)
                ckey = cache_key(cfg, prep.prompt, 0, schema_tag="JudgeVerdict")
                todo.append({
                    "model": model, "fname": fname, "out_csv": out_csv, "fieldnames": fieldnames,
                    "case_id": case_id, "case": case, "g": g, "metrics": m, "arm": arm, "axis2": axis2,
                    "replicate": replicate, "variant": variant, "seed": m.get("seed"),
                    "prompt": prep.prompt, "cache_key": ckey,
                })

    if args.batch and todo:
        b_stats = batch_prefetch_cached(
            todo, cfg, cache_dir, args.out, batch_size=args.batch_size,
            poll_s=args.poll_seconds, max_wait_s=args.max_wait_hours * 3600.0,
            wait=not args.no_wait, transport=transport, sleep=sleep, log=print)
        if b_stats.get("pending"):
            print(f"PENDING: batch {b_stats.get('batch_id')} is still processing; re-run to collect", file=sys.stderr)
            return 5
        print(f"batch: {b_stats}")

    mode = "replay" if args.batch else "readwrite"
    local = threading.local()

    def get_evaluator():
        if not hasattr(local, "evaluator"):
            local.evaluator = rb.ClinicalEvaluator(
                model_name=cfg.model, config=cfg, cache_dir=cache_dir,
                cache_mode=mode, replicate=0)
        return local.evaluator

    def judge_one(item: dict) -> dict:
        ev = get_evaluator()
        row = rb.async_evaluate_case(
            ev, item["case"]["vignette"], item["case"]["ground_truth_rubric"],
            item["g"]["final_recommendation"], item["metrics"], item["case"], item["model"],
            item["g"].get("transcript_path", ""), item["g"].get("tps", 0), item["out_csv"],
            item.get("fieldnames") or rb.fieldnames_for_arm(item.get("arm")), evaluate_axis_2=item["axis2"],
            variant=item["variant"], seed=item["seed"],
            replicate=item["replicate"])
        print(f"{item['model']} {item['fname']} {item['case_id']} (r{item['replicate']}): "
              f"{row.get('outcome_class')} reasoning={row.get('clinical_reasoning_score')} "
              f"utility={row.get('clinical_utility_score')} unsafe={row.get('unsafe_outcome')}", flush=True)
        return row

    n_rows = 0
    if args.concurrency > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(judge_one, item) for item in todo]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
                n_rows += 1
    else:
        for item in todo:
            judge_one(item)
            n_rows += 1

    print(f"judged {n_rows} answers into {args.out}; skipped {len(skipped)}: {skipped[:5]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
