from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.evidence.items import DdiItem

DDI_NO_INTERACTION = "no interaction recorded in DDInter"
DDI_NOT_FOUND = "drug not found in DDInter"
UNGRADED_DDI_NOTE = ("DDInter lists this pair as interacting but has not graded its severity or documented a mechanism. "
                     "This is neither evidence of no interaction nor a graded contraindication; it warrants explicit caution "
                     "and verification with a pharmacist or a second interaction reference before co-administration.")
NEED_LIST = ("Pass every medication to check (current medications and proposed therapies) in one call; "
             "interactions are checked between the listed drugs.")


@dataclass(frozen=True)
class DrugLookup:
    raw: str
    status: Literal["found", "not_in_ddinter", "unresolved", "undetermined"]
    ingredients: tuple[str, ...] = ()
    candidates: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class InteractionReport:
    drugs: tuple[DrugLookup, ...]
    pairs: tuple[DdiItem, ...]


def resolve_drug(lexicon, raw: str) -> DrugLookup:
    res = lexicon.resolve(str(raw))
    if res.status == "match" and res.ingredients:
        return DrugLookup(raw=str(raw), status="found", ingredients=tuple(res.ingredients))
    if res.status == "undetermined":
        return DrugLookup(raw=str(raw), status="undetermined", candidates=tuple(res.candidates))
    return DrugLookup(raw=str(raw), status="unresolved")


def ambiguous_line(d: DrugLookup, name_of=None) -> str:
    cands = "; ".join(" + ".join((name_of(rx) if name_of else None) or rx for rx in c) for c in d.candidates)
    return f"{d.raw}: drug name is ambiguous (candidates: {cands}); use the generic name"


def check_interactions(tables, med_list) -> InteractionReport:
    drugs, seen = [], set()
    for raw in med_list or []:
        raw = str(raw).strip()
        if not raw or raw.casefold() in seen:
            continue
        seen.add(raw.casefold())
        d = resolve_drug(tables.lexicon, raw)
        if d.status == "found" and not any(tables.ddinter.interaction_count(rx) > 0 for rx in d.ingredients):
            d = DrugLookup(raw=d.raw, status="not_in_ddinter", ingredients=d.ingredients)
        drugs.append(d)
    found = [d for d in drugs if d.status == "found"]
    pairs, ids = [], set()
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            for a in found[i].ingredients:
                for b in found[j].ingredients:
                    if a == b:
                        continue
                    item = tables.ddinter.lookup(a, b, descriptions=tables.ddi_descriptions)
                    if item is not None and item.id not in ids:
                        ids.add(item.id)
                        pairs.append(item)
    return InteractionReport(drugs=tuple(drugs), pairs=tuple(pairs))


def format_interactions(report: InteractionReport, name_of=None) -> str:
    lines = [f"Checked {len(report.drugs)} medications against DDInter."]
    if report.pairs:
        lines.append("Interactions:")
        for p in report.pairs:
            sev = str(p.severity or "Unknown")
            desc = (p.description or "").strip() or "No description provided."
            if sev.lower() == "unknown":
                lines.append(f"- {p.name_a} -- {p.name_b}: Unknown (ungraded). {desc} [{p.id}] [Note: {UNGRADED_DDI_NOTE}]")
            else:
                lines.append(f"- {p.name_a} -- {p.name_b}: {sev}. {desc} [{p.id}]")
    in_pair = {rx for p in report.pairs for rx in (p.ingredient_a, p.ingredient_b)}
    for d in report.drugs:
        if d.status == "found" and not set(d.ingredients) & in_pair:
            lines.append(f"{d.raw}: {DDI_NO_INTERACTION} with the other listed drugs")
        elif d.status == "not_in_ddinter":
            lines.append(f"{d.raw}: {DDI_NOT_FOUND}")
        elif d.status == "unresolved":
            lines.append(f"{d.raw}: {DDI_NOT_FOUND} (name not recognized)")
        elif d.status == "undetermined":
            lines.append(ambiguous_line(d, name_of))
    if sum(1 for d in report.drugs if d.status == "found") < 2:
        lines.append(NEED_LIST)
    return "\n".join(lines)
