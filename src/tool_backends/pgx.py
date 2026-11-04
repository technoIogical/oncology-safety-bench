from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.evidence.items import CpicItem
from src.tool_backends.interactions import ambiguous_line, resolve_drug

NO_CPIC_FMT = "No CPIC guideline recorded for {drug}."
HINT = "Pass gene and phenotype for a phenotype-specific recommendation."
_ABBREV = {"pm": "poor metabolizer", "im": "intermediate metabolizer", "nm": "normal metabolizer",
           "rm": "rapid metabolizer", "um": "ultrarapid metabolizer", "em": "normal metabolizer"}


@dataclass(frozen=True)
class PgxGuidance:
    raw: str
    status: Literal["found", "none", "unresolved", "undetermined"]
    drug_name: str = ""
    pairs: tuple[tuple[str, str], ...] = ()
    recommendations: tuple[CpicItem, ...] = ()
    gene: str | None = None
    phenotype: str | None = None
    note: str | None = None
    candidates: tuple[tuple[str, ...], ...] = ()

    @property
    def has_guidance(self) -> bool:
        return bool(self.recommendations)


def _genes(item: CpicItem) -> list[str]:
    return [g.strip() for g in str(item.gene).split("+") if g.strip()]


def _phenotype_of(item: CpicItem, gene: str) -> str | None:
    if "+" not in str(item.gene):
        return item.phenotype
    for part in str(item.phenotype or "").split(";"):
        g, _, p = part.partition(":")
        if g.strip().casefold() == gene.casefold():
            return p.strip()
    return None


def normalize_phenotype(text) -> str:
    t = " ".join(str(text or "").casefold().replace("_", " ").split())
    return _ABBREV.get(t, t)


def pgx_guidance(tables, drug_name, gene=None, phenotype=None) -> PgxGuidance:
    raw = str(drug_name or "").strip()
    gene = str(gene).strip().upper() if gene else None
    phenotype = str(phenotype).strip() if phenotype else None
    d = resolve_drug(tables.lexicon, raw)
    if d.status == "undetermined":
        return PgxGuidance(raw=raw, status="undetermined", candidates=d.candidates)
    if d.status != "found":
        return PgxGuidance(raw=raw, status="unresolved", note=NO_CPIC_FMT.format(drug=raw))
    items = [i for i in tables.cpic_items if i.drug_rxcui in d.ingredients]
    if not items:
        return PgxGuidance(raw=raw, status="none", note=NO_CPIC_FMT.format(drug=raw))
    name = items[0].drug_name
    pairs = tuple(sorted({(g, str(i.cpic_level)) for i in items for g in _genes(i)}))
    base = dict(raw=raw, status="found", drug_name=name, pairs=pairs, gene=gene, phenotype=phenotype)
    if not gene:
        return PgxGuidance(**base, note=HINT)
    for_gene = [i for i in items if gene.casefold() in (g.casefold() for g in _genes(i))]
    if not for_gene:
        return PgxGuidance(**base, note=f"No CPIC recommendation for {name} and {gene}.")
    if not phenotype:
        return PgxGuidance(**base, recommendations=tuple(for_gene))
    want = normalize_phenotype(phenotype)
    hits = tuple(i for i in for_gene if normalize_phenotype(_phenotype_of(i, gene)) == want)
    if not hits:
        recorded = "; ".join(sorted({_phenotype_of(i, gene) or "?" for i in for_gene}))
        return PgxGuidance(**base, note=(f"No CPIC recommendation for {gene} phenotype '{phenotype}'; "
                                         f"recorded phenotypes: {recorded}"))
    return PgxGuidance(**base, recommendations=hits)


def format_pgx(result: PgxGuidance, name_of=None) -> str:
    if result.status == "undetermined":
        return ambiguous_line(result, name_of)
    if result.status in ("none", "unresolved"):
        return result.note
    lines = [f"CPIC guidance for {result.drug_name}:",
             "Gene-drug pairs: " + ", ".join(f"{g} (CPIC level {lvl})" for g, lvl in result.pairs)]
    if result.recommendations:
        title = f"{result.gene} {result.phenotype}" if result.phenotype else result.gene
        lines.append(f"Recommendations for {title}:")
        groups: dict[tuple, list[str]] = {}
        for r in result.recommendations:
            pheno = f" [{r.phenotype}]" if (not result.phenotype or "+" in str(r.gene)) else ""
            action = f" (action: {str(r.action).replace('_', ' ')})" if r.action else ""
            groups.setdefault((pheno, r.classification, r.recommendation_text, action), []).append(r.id)
        for (pheno, cls, text, action), ids in groups.items():
            lines.append(f"- [{', '.join(ids)}]{pheno} {cls}: {text}{action}")
    if result.note:
        lines.append(result.note)
    return "\n".join(lines)
