"""Thread-local case context and callable tool entrypoints exposed to language models."""
from __future__ import annotations

import re
import threading
from typing import List, Optional

from src.tool_backends import NO_RECORD
from src.tool_backends import interactions as _interactions
from src.tool_backends import labels as _labels
from src.tool_backends import literature as _literature
from src.tool_backends import pgx as _pgx
from src.tool_backends import variants as _variants
from src.tool_backends.tables import global_tables, parse_bundle

UNGRADED_DDI_NOTE = _interactions.UNGRADED_DDI_NOTE
EVIDENCE_MODES = ("own", "donor", "empty")
_COMBO = re.compile(r"\s*(?:\+|/|,|;|\band\b)\s*", re.IGNORECASE)


class ThreadLocalDB(threading.local):
    def __init__(self):
        self.db = []

    def clear(self):
        self.db.clear()

    def extend(self, items):
        self.db.extend(items)

    def append(self, item):
        self.db.append(item)

    def __iter__(self):
        return iter(self.db)

    def __bool__(self):
        return bool(self.db)

    def __len__(self):
        return len(self.db)


BUNDLE_DB = ThreadLocalDB()
CASE_DB_CACHE = {}
_db_lock = threading.Lock()
thread_local = threading.local()


def _cached_bundle(key: str, raw) -> object:
    if key not in CASE_DB_CACHE:
        CASE_DB_CACHE[key] = parse_bundle(raw)
    return CASE_DB_CACHE[key]


def set_current_case_id(case_id: str, case_data: Optional[dict] = None, *, evidence_mode: str = "own",
                        evidence_case: Optional[dict] = None):
    if evidence_mode not in EVIDENCE_MODES:
        raise ValueError(f"unknown evidence_mode {evidence_mode!r}")
    thread_local.current_case_id = case_id
    bundle = None
    with _db_lock:
        if evidence_mode == "own":
            if case_id in CASE_DB_CACHE:
                bundle = CASE_DB_CACHE[case_id]
            elif case_data is not None and case_data.get("evidence_bundle") is not None:
                bundle = _cached_bundle(case_id, case_data["evidence_bundle"])
        elif evidence_mode == "donor":
            donor_id = (evidence_case or {}).get("case_id")
            if not donor_id or donor_id == case_id or (evidence_case or {}).get("evidence_bundle") is None:
                raise ValueError("the donor evidence case must be a different case with an evidence bundle")
            bundle = _cached_bundle(donor_id, evidence_case["evidence_bundle"])
        else:
            bundle = _cached_bundle("__empty__", {"case_id": "__empty__", "case_doid": None, "doid_match": None,
                                                  "items": []})
    BUNDLE_DB.clear()
    if bundle is not None:
        BUNDLE_DB.append(bundle)


def current_bundle():
    for bundle in BUNDLE_DB:
        return bundle
    raise LookupError("no case bound to this thread; call set_current_case_id(case_id, case) first")


def _name_of():
    return global_tables().lexicon.ingredient_name


def _split_combo(name) -> List[str]:
    return [p.strip() for p in _COMBO.split(str(name or "")) if p and p.strip()]


def annotate_variants(mutation_list: List[str]) -> str:
    anns = _variants.annotate(current_bundle(), [str(m) for m in (mutation_list or [])])
    return _variants.format_annotations(anns, _name_of())


def check_drug_interactions(med_list: List[str]) -> str:
    tables = global_tables()
    meds = [p for m in (med_list or []) for p in (_split_combo(m) or [str(m)])]
    report = _interactions.check_interactions(tables, meds)
    return _interactions.format_interactions(report, name_of=tables.lexicon.ingredient_name)


def get_drug_label(drug_name: str, sections=None) -> str:
    tables = global_tables()
    wanted = _labels.parse_sections(sections)
    parts = _split_combo(drug_name)
    if len(parts) > 1:
        return "\n\n".join(_labels.format_label(_labels.drug_label(tables, p), name_of=tables.lexicon.ingredient_name,
                                                 sections=wanted) for p in parts)
    return _labels.format_label(_labels.drug_label(tables, drug_name), name_of=tables.lexicon.ingredient_name, sections=wanted)


def get_pharmacogenomic_guidance(drug_name: str, gene: Optional[str] = None, phenotype: Optional[str] = None) -> str:
    tables = global_tables()
    return _pgx.format_pgx(_pgx.pgx_guidance(tables, drug_name, gene, phenotype),
                           name_of=tables.lexicon.ingredient_name)


def search_pubmed(query: str, max_results: int = 20) -> str:
    return _literature.format_pubmed(query, _literature.search_items(current_bundle(), query, max_results))


def find_clinical_trials(condition: str, gene: Optional[str] = None, variant: Optional[str] = None) -> str:
    items = _literature.find_trial_items(current_bundle(), condition, gene, variant)
    return _literature.format_trials(condition, gene, variant, items)


def surfaced_therapies(mutation_list: List[str]) -> List[str]:
    anns = _variants.annotate(current_bundle(), [str(m) for m in (mutation_list or [])])
    return _variants.surfaced_therapies(anns, _name_of())
