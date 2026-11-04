from __future__ import annotations

import re

from src.evidence.items import EvidenceBundle, PubmedItem, TrialItem
from src.tool_backends import NO_RECORD

SYNONYMS = {"nsclc": "non-small cell lung cancer", "crc": "colorectal cancer",
            "gist": "gastrointestinal stromal tumor", "tnbc": "triple-negative breast cancer"}
STOP_WORDS = {"a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "when", "where", "how", "why",
              "who", "which", "this", "that", "these", "those", "then", "just", "so", "than", "such", "both",
              "through", "about", "for", "is", "of", "while", "during", "to", "in", "out", "on", "off", "over",
              "under", "again", "further", "once", "here", "there", "all", "any", "each", "few", "more", "most",
              "other", "some", "no", "nor", "not", "only", "own", "same", "too", "very", "s", "t", "can",
              "will", "don", "should", "now"}
_WORD = re.compile(r"\b\w+\b")


def _limit(max_results) -> int:
    try:
        return max(0, int(max_results)) if max_results is not None else 20
    except (TypeError, ValueError):
        return 20


def _flat(text) -> str:
    return " ".join(str(text or "").split())


def _age(x) -> str:
    if x is None:
        return "not stated"
    f = float(x)
    return str(int(f)) if f.is_integer() else f"{f:g}"


def normalize_condition(text) -> str:
    t = str(text or "").lower().strip()
    for key, val in SYNONYMS.items():
        t = re.sub(r"\b" + re.escape(key) + r"\b", val, t)
    return t


def search_items(bundle: EvidenceBundle, query, max_results=20) -> list[PubmedItem]:
    q = str(query or "").lower()
    words = set(_WORD.findall(q)) - STOP_WORDS
    scored = []
    for order, item in enumerate(i for i in bundle.items if isinstance(i, PubmedItem)):
        title, abstract = (item.title or "").lower(), (item.abstract or "").lower()
        tw, aw = set(_WORD.findall(title)), set(_WORD.findall(abstract))
        score = sum(3 * (w in tw) + (w in aw) for w in words)
        score += 5 * bool(q and q in title) + 5 * bool(q and q in abstract)
        if score > 0:
            scored.append((-score, order, item))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [item for _, _, item in scored[:_limit(max_results)]]


def pubmed_line(item: PubmedItem) -> str:
    abstract = _flat(item.abstract)
    if len(abstract) > 300:
        abstract = abstract[:297] + "..."
    return f"- [PMID {item.pmid}] {_flat(item.title)} ({item.year}, {_flat(item.journal)}) — {abstract}"


def format_pubmed(query, items: list[PubmedItem]) -> str:
    if not items:
        return f"{NO_RECORD} (query: '{query}')"
    return "\n".join([f"PubMed results for '{query}':"] + [pubmed_line(i) for i in items])


def find_trial_items(bundle: EvidenceBundle, condition, gene=None, variant=None) -> list[TrialItem]:
    cond = normalize_condition(condition)
    gv = f"{gene or ''} {variant or ''}".strip().lower()
    out = []
    for t in (i for i in bundle.items if isinstance(i, TrialItem)):
        title, elig = (t.title or "").lower(), (t.eligibility_text or "").lower()
        biom = (t.biomarker_text or "").lower()
        if not (cond and (cond in title or cond in elig)):
            continue
        if gv:
            g = str(gene or "").lower()
            ok = (gv in title or gv in elig or "any mutation" in elig or "any gene" in elig
                  or bool(g and (g in elig or g in biom or g in title)))
            if not ok:
                continue
        out.append(t)
    return out


def trial_line(t: TrialItem) -> str:
    ecog = f"ECOG <= {t.ecog_max}" + (f" (\"{t.ecog_quote}\")" if t.ecog_quote else "") \
        if t.ecog_max is not None else "ECOG not stated"
    biom = f"biomarker: {t.biomarker_text}" if t.biomarker_text else "biomarker: not stated"
    text = _flat(t.eligibility_text)
    if len(text) > 1000:
        text = text[:500] + " ...[TRUNCATED FOR LENGTH]... " + text[-500:]
    return (f"- [{t.nct_id}] {_flat(t.title)} (Phase: {t.phase}, Status: {t.status}) — Eligibility: min age {_age(t.min_age)}; "
            f"max age {_age(t.max_age)}; {ecog}; {biom} — {text}")


def format_trials(condition, gene, variant, items: list[TrialItem]) -> str:
    if not items:
        return NO_RECORD
    head = f"Clinical trials for {condition}" + (f", {gene}" if gene else "") + (f" {variant}" if variant else "") + ":"
    return "\n".join([head] + [trial_line(t) for t in items])
