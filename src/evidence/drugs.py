"""RxNorm drug-alias resolution, ATC/MeSH class lookup, NCI Thesaurus therapy synonyms, and product equivalence."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from src.evidence.identifiers import normalize_text
from src.evidence.items import InteractionType, Therapy

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = REPO_ROOT / "data" / "evidence_snapshots" / "tables"

Status = Literal["match", "no_match", "undetermined"]
_EPS = 1e-12
_FORM_WORDS = frozenset({
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps", "injection", "inj", "solution",
    "oral", "suspension", "cream", "ointment", "gel", "patch", "extended", "release", "er", "xr", "sr", "dr",
    "ec", "delayed", "film", "coated", "chewable", "powder", "syrup", "elixir", "drops", "spray", "inhaler",
    "vial", "hfa", "po", "iv", "im", "mg", "mcg", "ug", "ml", "units", "unit", "iu", "daily", "bid", "tid",
})
_ABBREV = {"hcl": "hydrochloride", "hbr": "hydrobromide"}
_DOSE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|ml|%|units?|iu)\b(?:/\S+)?")
_SEP = re.compile(r"[\s\-_/.]+")
SHORT_MAX = 3
TABLE_NAME = "therapy_aliases"
_SPLIT = re.compile(r"\s*(?:,|;|\+|/|&|\band\b)\s*", re.IGNORECASE)
_BASED = re.compile(
    r"^(?P<drug>[A-Za-z0-9][A-Za-z0-9 \-]*?)[\s\-]+(?:based|containing)\s+"
    r"(?:chemotherap(?:y|ies)|therap(?:y|ies)|regimens?|treatments?|combinations?|approach(?:es)?)$",
    re.IGNORECASE,
)
INTERCHANGEABLE: dict[str, frozenset[str]] = {
    "51499": frozenset({"51499", "72965"}),
    "72965": frozenset({"51499", "72965"}),
    "40048": frozenset({"40048", "39998"}),
    "39998": frozenset({"40048", "39998"}),
}


@dataclass(frozen=True)
class EvidenceParams:
    fuzzy_min: float = 0.92
    fuzzy_margin: float = 0.05


PARAMS = EvidenceParams()


@dataclass(frozen=True)
class Resolution:
    status: Status
    ingredients: tuple[str, ...] = ()
    matched_name: str | None = None
    method: Literal["exact", "fuzzy", "none"] = "none"
    score: float | None = None
    candidates: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class ClassResolution:
    status: Status
    class_id: str | None = None
    class_name: str | None = None
    source: str | None = None
    members: tuple[str, ...] = ()
    score: float | None = None


def normalize_drug_key(raw: str) -> str:
    s = (raw or "").replace("\u00ae", " ").replace("\u2122", " ").replace("\u00a9", " ")
    s = normalize_text(s).replace("\\", " ")
    s = re.sub(r"/\d{4,}/", " ", s)
    s = re.sub(r"\([^)]*\)", " ", s)
    s = _DOSE.sub(" ", s)
    tokens = [t for t in re.split(r"[^a-z0-9+\-/]+", s) if t]
    tokens = [_ABBREV.get(t, t) for t in tokens if t not in _FORM_WORDS and not re.fullmatch(r"\d+(?:\.\d+)?", t)]
    return " ".join(tokens)


def _sorted_set(ingredients: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(i) for i in ingredients}, key=int))


# Matches a normalized drug query by Levenshtein similarity >= fuzzy_min, returning 'undetermined' on close ties.
def _fuzzy_pick(query: str, keys: list[str], groups_of, params: EvidenceParams):
    hits = process.extract(query, keys, scorer=Levenshtein.normalized_similarity, limit=None,
                           score_cutoff=params.fuzzy_min - params.fuzzy_margin)
    best: dict = {}
    for key, score, _ in hits:
        for group in groups_of(key):
            if group not in best or score > best[group][0]:
                best[group] = (score, key)
    ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))
    if not ranked or ranked[0][1][0] < params.fuzzy_min:
        return "no_match", None, None, (ranked[0][1][0] if ranked else None), ()
    top_group, (top_score, top_key) = ranked[0]
    near = tuple(g for g, (s, _) in ranked if s > top_score - params.fuzzy_margin + _EPS)
    if len(near) > 1:
        return "undetermined", None, top_key, top_score, tuple(sorted(near))
    return "match", top_group, top_key, top_score, ()


def _class_key(name: str) -> str:
    words = normalize_text(name).split()
    return " ".join(w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words)


class AliasIndex:
    def __init__(self, entries: dict[str, Iterable[Iterable[str]]] | None = None,
                 names: dict[str, str] | None = None, params: EvidenceParams = PARAMS):
        self._entries: dict[str, set[tuple[str, ...]]] = {
            k: {_sorted_set(s) for s in v} for k, v in (entries or {}).items()}
        self._names: dict[str, str] = dict(names or {})
        self._params = params
        self._keys: list[str] | None = None

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> list[str]:
        if self._keys is None:
            self._keys = sorted(self._entries)
        return self._keys

    def add(self, name: str, ingredients: Iterable[str]) -> None:
        key = normalize_drug_key(name)
        ings = _sorted_set(ingredients)
        if key and ings:
            self._entries.setdefault(key, set()).add(ings)
            self._keys = None

    def has(self, name: str) -> bool:
        return normalize_drug_key(name) in self._entries

    def set_ingredient_name(self, rxcui: str, name: str) -> None:
        self._names[str(rxcui)] = name.lower()

    def ingredient_name(self, rxcui: str) -> str | None:
        return self._names.get(str(rxcui))

    def resolve(self, raw: str) -> Resolution:
        key = normalize_drug_key(raw)
        if not key:
            return Resolution("no_match")
        sets = self._entries.get(key)
        if sets:
            if len(sets) == 1:
                return Resolution("match", next(iter(sets)), key, "exact", 1.0)
            return Resolution("undetermined", (), key, "exact", 1.0, tuple(sorted(sets)))
        status, group, best_key, score, near = _fuzzy_pick(key, self.keys(), lambda k: self._entries[k], self._params)
        if status == "match":
            return Resolution("match", group, best_key, "fuzzy", score)
        if status == "undetermined":
            return Resolution("undetermined", (), best_key, "fuzzy", score, near)
        return Resolution("no_match", score=score)

    def therapy(self, names: Iterable[str], interaction_type: InteractionType) -> tuple[Therapy | None, tuple[str, ...]]:
        pairs: dict[str, str] = {}
        unresolved: list[str] = []
        for name in names:
            r = self.resolve(name)
            if r.status != "match":
                unresolved.append(name)
                continue
            for rx in r.ingredients:
                pairs[rx] = self.ingredient_name(rx) or (r.matched_name if len(r.ingredients) == 1 else rx)
        if unresolved or not pairs:
            return None, tuple(unresolved)
        kind = interaction_type
        if len(pairs) == 1:
            kind = "single"
        elif kind == "single":
            kind = "combination"
        return Therapy.of(pairs.items(), kind), ()

    def to_json(self) -> dict:
        return {"entries": {k: sorted(list(s) for s in self._entries[k]) for k in sorted(self._entries)},
                "names": dict(sorted(self._names.items(), key=lambda kv: int(kv[0])))}

    @classmethod
    def from_json(cls, d: dict, params: EvidenceParams = PARAMS) -> "AliasIndex":
        return cls(entries=d["entries"], names=d.get("names") or {}, params=params)


class ClassIndex:
    def __init__(self, classes: dict[str, dict] | None = None, params: EvidenceParams = PARAMS):
        self._classes: dict[str, dict] = {}
        self._by_key: dict[str, set[str]] = {}
        self._params = params
        for cid, c in (classes or {}).items():
            self.add(cid, c["name"], c["source"], c["members"])

    def add(self, class_id: str, name: str, source: Literal["ATC", "MESHPA"], members: Iterable[str]) -> None:
        self._classes[class_id] = {"name": name, "source": source, "members": list(_sorted_set(members))}
        self._by_key.setdefault(_class_key(name), set()).add(class_id)

    def _pick(self, class_ids: set[str]) -> str | None:
        atc = sorted(c for c in class_ids if self._classes[c]["source"] == "ATC")
        pool = atc or sorted(class_ids)
        return pool[0] if len(pool) == 1 else None

    def _result(self, class_id: str, score: float) -> ClassResolution:
        c = self._classes[class_id]
        return ClassResolution("match", class_id, c["name"], c["source"], tuple(c["members"]), score)

    def resolve(self, raw: str) -> ClassResolution:
        key = _class_key(raw)
        if not key:
            return ClassResolution("no_match")
        if key in self._by_key:
            picked = self._pick(self._by_key[key])
            return self._result(picked, 1.0) if picked else ClassResolution("undetermined", score=1.0)
        status, group, _, score, _ = _fuzzy_pick(
            key, sorted(self._by_key),
            lambda k: [p] if (p := self._pick(self._by_key[k])) else [f"?{k}"], self._params)
        if status == "match" and not group.startswith("?"):
            return self._result(group, score)
        if status == "no_match":
            return ClassResolution("no_match", score=score)
        return ClassResolution("undetermined", score=score)

    def to_json(self) -> dict:
        return {cid: self._classes[cid] for cid in sorted(self._classes)}

    @classmethod
    def from_json(cls, d: dict, params: EvidenceParams = PARAMS) -> "ClassIndex":
        return cls(classes=d, params=params)


def norm_key(text: str) -> str:
    stripped = _SEP.sub("", text)
    return stripped if len(stripped) <= SHORT_MAX else stripped.lower()


def with_interchangeable(ingredients: set[str] | frozenset[str]) -> set[str]:
    out = set(ingredients)
    for rx in list(ingredients):
        if rx in INTERCHANGEABLE:
            out |= INTERCHANGEABLE[rx]
    return out


@lru_cache(maxsize=1)
def load_therapy_table() -> dict:
    return json.loads((TABLES_DIR / f"{TABLE_NAME}.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _classified_ingredients() -> frozenset[str]:
    raw = json.loads((TABLES_DIR / "class_index.json").read_text(encoding="utf-8"))
    return frozenset(str(m) for c in ClassIndex.from_json(raw).to_json().values() for m in c["members"])


def lookup(table: dict, raw: str) -> dict | None:
    return table["abbreviations"].get(norm_key(raw))


def _table_hit(table: dict, raw: str) -> Resolution:
    entry = lookup(table, raw)
    if entry is None or entry["status"] not in ("unique", "nested"):
        return Resolution("no_match")
    return Resolution("match", tuple(sorted(entry["ingredients"], key=int)), entry["forms"][0], "exact", 1.0)


def _drug_based(alias: AliasIndex, name: str, drugs: frozenset[str]) -> Resolution:
    r = alias.resolve(name)
    return r if r.status == "match" and set(r.ingredients) <= drugs else Resolution("no_match")


def resolve_therapy_mention(alias: AliasIndex, raw: str, table: dict | None = None,
                            drug_ingredients: frozenset[str] | None = None) -> Resolution:
    table = load_therapy_table() if table is None else table
    r = _table_hit(table, raw)
    if r.status == "match":
        return r
    r = alias.resolve(raw)
    if r.status == "match":
        return r
    parts = [p for p in _SPLIT.split(raw.strip()) if p]
    if len(parts) > 1:
        listed = [resolve_therapy_mention(alias, p, table, drug_ingredients) for p in parts]
        if all(x.status == "match" for x in listed):
            ings = tuple(sorted({i for x in listed for i in x.ingredients}, key=int))
            return Resolution("match", ings, " + ".join(parts), "exact", 1.0)
    m = _BASED.match(raw.strip())
    if m:
        drugs = _classified_ingredients() if drug_ingredients is None else drug_ingredients
        based = _drug_based(alias, m.group("drug"), drugs)
        if based.status == "match":
            return based
    return r
