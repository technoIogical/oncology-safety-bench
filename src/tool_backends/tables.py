"""Shared in-memory loaders and caches for frozen reference tables (RxNorm, ATC, DDInter, CPIC, FDA labels)."""
from __future__ import annotations

import gzip
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from src.evidence.drugs import AliasIndex, ClassIndex, ClassResolution, Resolution
from src.evidence.items import EVIDENCE_ITEM_ADAPTER, CpicItem, DdiItem, EvidenceBundle, ddi_id

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = REPO_ROOT / "data" / "evidence_snapshots" / "tables"
DDINTER_DESCRIPTION_CACHE = REPO_ROOT / "data" / "ddinter_cache.json"
SEVERITY_ORDER = {"Unknown": 0, "Minor": 1, "Moderate": 2, "Major": 3}
_EMPTY = {"", "-", "not found", "n/a", "na", "none"}
_TABLES: GlobalTables | None = None
_TABLES_LOCK = threading.Lock()


class LexiconLike(Protocol):
    def resolve(self, raw: str) -> Resolution: ...
    def ingredient_name(self, rxcui: str) -> str | None: ...


class ClassIndexLike(Protocol):
    def resolve(self, raw: str) -> ClassResolution: ...


class DdiTableLike(Protocol):
    def lookup(self, rxcui_a: str, rxcui_b: str, descriptions=None) -> DdiItem | None: ...
    def interaction_count(self, rxcui: str) -> int: ...


@dataclass(frozen=True)
class DdiPair:
    id_a: str
    name_a: str
    id_b: str
    name_b: str
    severity: str
    levels: tuple[str, ...]
    source_files: tuple[str, ...]


class DescriptionCache:
    def __init__(self, path: Path = DDINTER_DESCRIPTION_CACHE):
        self._data = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}

    def get(self, name_a: str, name_b: str) -> str | None:
        entry = self._data.get("::".join(sorted([name_a.lower(), name_b.lower()])))
        text = (entry or {}).get("mechanism")
        return text if text and text.strip().lower() not in _EMPTY else None


def _num(ddinter_id: str) -> int:
    return int(ddinter_id.replace("DDInter", ""))


class DDInterTable:
    def __init__(self, pairs: list[DdiPair], name_to_rxcui: dict[str, str], unresolved_names: list[str]):
        self.pairs = list(pairs)
        self.name_to_rxcui = dict(name_to_rxcui)
        self.unresolved_names = sorted(set(unresolved_names))
        self._index: dict[str, list[tuple[str, DdiPair]]] = {}
        for p in self.pairs:
            ra, rb = self.name_to_rxcui.get(p.name_a), self.name_to_rxcui.get(p.name_b)
            if ra and rb and ra != rb:
                self._index.setdefault(ra, []).append((rb, p))
                self._index.setdefault(rb, []).append((ra, p))

    def _pair(self, rxcui_a: str, rxcui_b: str) -> DdiPair | None:
        hits = [p for partner, p in self._index.get(rxcui_a, []) if partner == rxcui_b]
        if not hits:
            return None
        return sorted(hits, key=lambda p: (-SEVERITY_ORDER[p.severity], _num(p.id_a), _num(p.id_b)))[0]

    def lookup(self, rxcui_a: str, rxcui_b: str, descriptions: DescriptionCache | None = None) -> DdiItem | None:
        p = self._pair(str(rxcui_a), str(rxcui_b))
        if p is None:
            return None
        return DdiItem(id=ddi_id(p.id_a, p.id_b), ingredient_a=self.name_to_rxcui[p.name_a],
                       ingredient_b=self.name_to_rxcui[p.name_b], name_a=p.name_a, name_b=p.name_b,
                       severity=p.severity, description=descriptions.get(p.name_a, p.name_b) if descriptions else None,
                       source_files=p.source_files)

    def interaction_count(self, rxcui: str) -> int:
        return len({partner for partner, _ in self._index.get(str(rxcui), [])})

    @classmethod
    def from_json(cls, d: dict) -> "DDInterTable":
        pairs = [DdiPair(a, na, b, nb, s, tuple(lv), tuple(fs)) for a, na, b, nb, s, lv, fs in d["pairs"]]
        return cls(pairs, d["name_to_rxcui"], d["unresolved_names"])


@dataclass(frozen=True)
class GlobalTables:
    lexicon: LexiconLike
    ddinter: DdiTableLike
    ddi_descriptions: Any
    label_entry: Callable[[str], dict | None]
    cpic_items: tuple[CpicItem, ...]


def _cpic_from_table(raw) -> tuple[CpicItem, ...]:
    rows = raw.get("items", []) if isinstance(raw, dict) else list(raw or [])
    items = (EVIDENCE_ITEM_ADAPTER.validate_python(r) for r in rows)
    return tuple(i for i in items if isinstance(i, CpicItem))


def parse_bundle(d: dict) -> EvidenceBundle:
    return EvidenceBundle.model_validate(d)


def load_table(name: str, tables_dir: Path = TABLES_DIR) -> Any:
    plain, gz = Path(tables_dir) / f"{name}.json", Path(tables_dir) / f"{name}.json.gz"
    if plain.exists():
        return json.loads(plain.read_text(encoding="utf-8"))
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as fh:
            return json.loads(fh.read())
    raise FileNotFoundError(f"evidence table {name} not found in {tables_dir}")


def load_alias_index(tables_dir: Path = TABLES_DIR) -> AliasIndex:
    return AliasIndex.from_json(load_table("alias_index", tables_dir))


def load_class_index(tables_dir: Path = TABLES_DIR) -> ClassIndex:
    return ClassIndex.from_json(load_table("class_index", tables_dir))


def load_ddinter(tables_dir: Path = TABLES_DIR) -> DDInterTable:
    return DDInterTable.from_json(load_table("ddinter_full", tables_dir))


def load_global_tables(tables_dir=None, descriptions_path=None) -> GlobalTables:
    d = Path(tables_dir) if tables_dir is not None else TABLES_DIR
    desc_path = Path(descriptions_path) if descriptions_path is not None else Path(DDINTER_DESCRIPTION_CACHE)
    if not desc_path.exists():
        raise FileNotFoundError(f"DDInter description cache missing: {desc_path} "
                                f"(interaction descriptions would silently read as empty)")
    labels = load_table("label_table", tables_dir=d)
    if not isinstance(labels, dict):
        raise ValueError(f"label_table in {d} is not a rxcui -> entry mapping")

    def label_entry(rxcui: str) -> dict | None:
        return labels.get(str(rxcui))

    return GlobalTables(lexicon=load_alias_index(tables_dir=d), ddinter=load_ddinter(tables_dir=d),
                        ddi_descriptions=DescriptionCache(desc_path), label_entry=label_entry,
                        cpic_items=_cpic_from_table(load_table("cpic_recommendations", tables_dir=d)))


def global_tables() -> GlobalTables:
    global _TABLES
    with _TABLES_LOCK:
        if _TABLES is None:
            _TABLES = load_global_tables()
        return _TABLES


def set_global_tables(t: GlobalTables | None) -> None:
    global _TABLES
    with _TABLES_LOCK:
        _TABLES = t
