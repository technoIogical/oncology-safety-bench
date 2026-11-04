from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−﹘﹣－"), "-")
_SUBS = (
    ("≤", "<="), ("≥", ">="), ("≦", "<="), ("≧", ">="),
    ("µ", "u"), ("μ", "u"), ("×", "x"),
    ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
)
_PMID = (
    re.compile(r"\bpmid\s*:?\s*(\d{1,9})\b"),
    re.compile(r"\bpubmed\s*id\s*:?\s*(\d{1,9})\b"),
    re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{1,9})"),
)
_NCT = (re.compile(r"\bnct[\s-]?(\d{8})\b"),)
_EID = (
    re.compile(r"\beid\s*:?\s*(\d{1,9})\b"),
    re.compile(r"civicdb\.org/(?:links/)?evidence(?:_items)?/(\d{1,9})(?!\d)"),
)


@dataclass(frozen=True)
class Identifiers:
    pmids: tuple[str, ...]
    ncts: tuple[str, ...]
    eids: tuple[str, ...]

    def item_ids(self) -> tuple[str, ...]:
        return (tuple(f"pmid:{p}" for p in self.pmids) + tuple(f"nct:{n}" for n in self.ncts)
                + tuple(f"civic:EID{e}" for e in self.eids))


def normalize_text(s: str | None) -> str:
    s = unicodedata.normalize("NFKC", s or "").translate(_DASHES)
    for a, b in _SUBS:
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _find(patterns, text: str, fmt) -> tuple[str, ...]:
    hits = sorted((m.start(), fmt(m.group(1))) for rx in patterns for m in rx.finditer(text))
    out: list[str] = []
    for _, value in hits:
        if value not in out:
            out.append(value)
    return tuple(out)


def _strip_leading_zeros(digits: str) -> str:
    return digits.lstrip("0") or "0"


def extract_identifiers(text: str) -> Identifiers:
    t = normalize_text(text)
    return Identifiers(
        pmids=_find(_PMID, t, _strip_leading_zeros),
        ncts=_find(_NCT, t, lambda d: f"NCT{d}"),
        eids=_find(_EID, t, _strip_leading_zeros),
    )
