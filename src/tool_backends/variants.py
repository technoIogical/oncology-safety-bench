"""CIViC and ClinVar genomic variant and biomarker annotation backend."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal

from src.evidence.items import CivicItem, ClinvarItem, EvidenceBundle, LabelIndicationItem
from src.evidence.variants import canonical_variant
from src.tool_backends import NO_RECORD

UNRESOLVED_FMT = "could not resolve '{raw}'; use GENE PROTEIN_CHANGE"
_BIOMARKERS = {"TMB-H": "TMB-H", "TMB HIGH": "TMB-H", "TMB-HIGH": "TMB-H", "HIGH TMB": "TMB-H",
               "MSI-H": "MSI-H", "MSI HIGH": "MSI-H", "MSI-HIGH": "MSI-H", "HIGH MSI": "MSI-H",
               "DMMR": "dMMR", "MMR DEFICIENT": "dMMR"}
_CNA = {"DEEP DELETION": "Deletion", "HOMOZYGOUS DELETION": "Deletion", "LOSS": "Deletion",
        "DELETION": "Deletion", "AMPLIFICATION": "Amplification", "AMPLIFIED": "Amplification",
        "AMP": "Amplification", "GAIN": "Amplification"}
_GENE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]*$")
_GENE_DASH_VARIANT = re.compile(r"([A-Z][A-Z0-9]+)[-:]([A-Z]\d+(?:[A-Z*]|FS\*?\d*|DEL|INS)?)")
_PAIR = re.compile(r"\b([A-Z0-9]+)-([A-Z0-9]+)\b")
_RAW_PROTEIN = re.compile(r"^(?:P\.)?(?:[A-Z*]{1,4}\d+[A-Z*?]{1,4}|PROMOTER)$")


@dataclass(frozen=True)
class ParsedQuery:
    raw: str
    kind: Literal["variant", "fusion", "biomarker", "unresolved"]
    gene: str | None = None
    genes: tuple[str, ...] = ()
    canonical: str | None = None
    biomarker: str | None = None

    @property
    def display(self) -> str:
        return " ".join(self.raw.split())


@dataclass(frozen=True)
class AlterationAnnotation:
    query: ParsedQuery
    sensitivity: tuple[CivicItem, ...] = ()
    resistance: tuple[CivicItem, ...] = ()
    adverse: tuple[CivicItem, ...] = ()
    clinvar: tuple[ClinvarItem, ...] = ()
    indications: tuple[LabelIndicationItem, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.sensitivity or self.resistance or self.adverse or self.clinvar or self.indications)


def _unresolved(raw: str) -> ParsedQuery:
    return ParsedQuery(raw=raw, kind="unresolved")


def _genelike(tok: str) -> bool:
    return len(tok) >= 2 and any(c.isalpha() for c in tok)


def _is_gene_pair(token: str) -> bool:
    m = _PAIR.fullmatch(token)
    return bool(m) and _genelike(m.group(1)) and _genelike(m.group(2))


def _strip_gene(text: str | None, gene: str) -> str | None:
    if not text:
        return None
    t = text.strip()
    return t[len(gene) + 1:] if t.upper().startswith(gene.upper() + " ") else t


def _item_canonicals(item: CivicItem) -> set[str]:
    keys = set()
    for text in (item.variant_name, _strip_gene(item.matched_alteration, item.gene)):
        if text:
            key = variant_key(item.gene, _CNA.get(text.upper(), text))
            if key is not None:
                keys.add(key)
    return keys


def _civic_matches(item: CivicItem, q: ParsedQuery) -> bool:
    if q.kind == "fusion":
        text = f"{item.gene} {item.matched_alteration or ''} {item.variant_name or ''}".upper()
        tokens = set(re.split(r"[\s\-:/]+", text))
        return bool(item.is_fusion) and any(g in tokens for g in q.genes)
    return q.kind == "variant" and item.gene.upper() == q.gene and q.canonical in _item_canonicals(item)


def _clinvar_matches(item: ClinvarItem, q: ParsedQuery) -> bool:
    if q.kind != "variant" or not item.found or str(item.gene).upper() != q.gene:
        return False
    return variant_key(item.gene, item.variant) == q.canonical


def _dedupe_key(q: ParsedQuery):
    if q.kind == "unresolved":
        return ("unresolved", q.raw)
    return (q.kind, q.canonical or q.biomarker or frozenset(q.genes))


def _therapy_text(item: CivicItem) -> str:
    if item.therapy is not None:
        if item.therapy.interaction_type == "substitutes":
            return f"{' or '.join(item.therapy.names)} (alternatives)"
        text = " + ".join(item.therapy.names)
        return text if item.therapy.interaction_type == "single" else f"{text} ({item.therapy.interaction_type})"
    return ", ".join(item.unresolved_therapies) or "no therapy recorded"


def _profile_text(item: CivicItem, q: ParsedQuery | None) -> str:
    profile = (item.molecular_profile or f"{item.gene} {item.variant_name}").strip()
    text = f" | molecular profile: {profile}"
    if q is not None and q.kind == "variant" and variant_key(item.gene, item.variant_name) != q.canonical:
        text += f" (not specific to {q.display})"
    return text


def _civic_line(item: CivicItem, q: ParsedQuery | None = None) -> str:
    return (f"  - EID{item.eid} | Level {item.level} | {_therapy_text(item)} | {item.disease_name} "
            f"({item.match_type} disease match){_profile_text(item, q)}")


def _names(rxcuis, name_of) -> str:
    return " + ".join(name_of(rx) or f"RxCUI {rx}" for rx in rxcuis)


def variant_key(gene: str, text: str) -> str | None:
    cv = canonical_variant(gene, text)
    if cv is not None:
        return cv.canonical
    tok = (text or "").strip().upper()
    cv = canonical_variant(gene, tok.removeprefix("P.")) if tok and tok != (text or "").strip() else None
    if cv is not None:
        return cv.canonical
    return f"{gene.strip().upper()} raw:{tok.removeprefix('P.')}" if _RAW_PROTEIN.fullmatch(tok) else None


def parse_query(raw: str) -> ParsedQuery:
    raw = str(raw or "").strip()
    text = " ".join(raw.split())
    if " " not in text and "_" in text:
        text = text.replace("_", " ", 1)
    up = text.upper()
    if not text:
        return _unresolved(raw)
    if up in _BIOMARKERS:
        return ParsedQuery(raw=raw, kind="biomarker", biomarker=_BIOMARKERS[up])
    m = _GENE_DASH_VARIANT.fullmatch(up)
    if m:
        text = f"{m.group(1)} {m.group(2)}"
        up = text
    if "FUSION" in up or (" " not in text and _is_gene_pair(up)):
        genes = []
        for a, b in _PAIR.findall(up):
            if _genelike(a) and _genelike(b):
                genes += [a, b]
        first = up.split(" ", 1)[0]
        if not genes and _GENE.match(first) and first != "FUSION":
            genes = [first]
        genes = tuple(dict.fromkeys(g for g in genes if g != "FUSION"))
        return ParsedQuery(raw=raw, kind="fusion", genes=genes) if genes else _unresolved(raw)
    parts = text.split(" ", 1)
    if len(parts) != 2 or not _GENE.match(parts[0].upper()):
        return _unresolved(raw)
    gene, variant = parts[0].upper(), parts[1].strip()
    variant = _CNA.get(variant.upper(), variant)
    key = variant_key(gene, variant)
    if key is None:
        return _unresolved(raw)
    return ParsedQuery(raw=raw, kind="variant", gene=gene, genes=(gene,), canonical=key)


def evidence_group(item: CivicItem) -> Literal["sensitivity", "resistance", "adverse", "other"]:
    s = str(item.significance or "").upper()
    if "RESIST" in s or "REDUCED" in s:
        return "resistance"
    if "ADVERSE" in s:
        return "adverse"
    if "SENSITIV" in s or "RESPONSE" in s:
        return "sensitivity"
    return "other"


def annotate(bundle: EvidenceBundle, mutation_list: list[str]) -> list[AlterationAnnotation]:
    civic = [i for i in bundle.items if isinstance(i, CivicItem)]
    clinvar = [i for i in bundle.items if isinstance(i, ClinvarItem)]
    approved = [i for i in bundle.approved_indications() if isinstance(i, LabelIndicationItem)]
    out, seen = [], set()
    for raw in mutation_list or []:
        q = parse_query(raw)
        key = _dedupe_key(q)
        if key in seen:
            continue
        seen.add(key)
        if q.kind in ("unresolved",):
            out.append(AlterationAnnotation(query=q))
            continue
        if q.kind == "biomarker":
            inds = tuple(i for i in approved if i.biomarker is not None and i.biomarker.kind == q.biomarker)
            out.append(AlterationAnnotation(query=q, indications=inds))
            continue
        hits = [i for i in civic if _civic_matches(i, q)]
        out.append(AlterationAnnotation(
            query=q,
            sensitivity=tuple(i for i in hits if evidence_group(i) == "sensitivity"),
            resistance=tuple(i for i in hits if evidence_group(i) == "resistance"),
            adverse=tuple(i for i in hits if evidence_group(i) == "adverse"),
            clinvar=tuple(i for i in clinvar if _clinvar_matches(i, q))))
    return out


def format_annotations(annotations: list[AlterationAnnotation], name_of: Callable[[str], str | None]) -> str:
    lines = [f"Annotated {len(annotations)} alterations.", ""]
    for a in annotations:
        q = a.query
        if q.kind == "unresolved":
            lines.append(UNRESOLVED_FMT.format(raw=q.raw))
            continue
        if a.empty:
            lines.append(f"{q.display}: {NO_RECORD}")
            continue
        lines.append(f"{q.display}:")
        if q.kind == "biomarker":
            lines.append("  Approved biomarker indications (FDA label):")
            for i in a.indications:
                lines.append(f"  - {_names(i.drug_rxcuis, name_of)} | {i.disease_text} | setting: "
                             f"{str(i.setting).replace('_', ' ')} | {i.id}")
            continue
        for title, items in (("Sensitivity (CIViC)", a.sensitivity), ("Resistance (CIViC)", a.resistance)):
            if items:
                lines.append(f"  {title}:")
                lines.extend(_civic_line(i, q) for i in items)
            else:
                lines.append(f"  {title}: none recorded")
        if a.adverse:
            lines.append("  Adverse response (CIViC):")
            lines.extend(_civic_line(i, q) for i in a.adverse)
        if a.clinvar:
            for c in a.clinvar:
                lines.append(f"  ClinVar: {c.classification} ({c.stars} stars) [{c.accession}]")
        else:
            lines.append("  ClinVar: no record")
    return "\n".join(lines).rstrip()


def surfaced_therapies(annotations: list[AlterationAnnotation], name_of) -> list[str]:
    names: list[str] = []
    for a in annotations:
        for item in a.sensitivity:
            if item.therapy is not None and str(item.direction).upper() == "SUPPORTS":
                names.extend(item.therapy.names)
        for ind in a.indications:
            names.extend(n for n in (name_of(rx) for rx in ind.drug_rxcuis) if n)
    return list(dict.fromkeys(n.strip() for n in names if n and n.strip()))
