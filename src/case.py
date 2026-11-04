"""Patient case state, clinical workup, genomics, labs, and medication schemas."""
from __future__ import annotations

import re
from typing import Any, Generic, Iterable, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from src.evidence.items import Therapy
from src.evidence.variants import Alteration as EvidenceAlteration

T = TypeVar("T")
Sex = Literal["Male", "Female"]
AltClass = Literal["mutation", "cna", "fusion"]
MsiStatus = Literal["MSI-H", "MSS"]
CLASS_ORDER = {"mutation": 0, "cna": 1, "fusion": 2}
CNA_VARIANTS = ("Amplification", "Deletion")
_BARE_NUMBER = re.compile(r"(?:^|[+,])\s*\d+\s*(?=$|[+,])")
ADVANCED_SETTING = "advanced (unresectable or metastatic) disease"
SETTING_LINE = "US academic cancer centre; FDA-approved drugs available."


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Sourced(_Frozen, Generic[T]):
    value: T
    source: str
    imputed: bool = False


SourcedInt = Sourced[int]
SourcedStr = Sourced[str]
SourcedSex = Sourced[Sex]
SourcedOptStr = Sourced[str | None]
SourcedOptFloat = Sourced[float | None]
SourcedMsi = Sourced[MsiStatus | None]


class Alteration(_Frozen):
    gene: str
    variant: str
    klass: AltClass
    partner: str | None = None
    mutation_type: str | None = None
    vaf: float | None = None

    @model_validator(mode="after")
    def _check(self):
        if not self.gene or not self.variant:
            raise ValueError("gene and variant are required")
        if self.klass == "cna" and self.variant not in CNA_VARIANTS:
            raise ValueError(f"CNA variant must be one of {CNA_VARIANTS}")
        if self.klass == "fusion" and (self.variant != "Fusion" or not self.partner):
            raise ValueError("a fusion has variant 'Fusion' and a partner gene")
        if self.klass != "fusion" and self.partner is not None:
            raise ValueError("only fusions have a partner")
        return self

    def _ev(self, gene: str, partner: str | None) -> EvidenceAlteration:
        return EvidenceAlteration(gene=gene, variant=self.variant, klass=self.klass, partner=partner,
                                  mutation_type=self.mutation_type, vaf=self.vaf)

    @property
    def short(self) -> str:
        return self._ev(self.gene, self.partner).short

    @property
    def display(self) -> str:
        if self.klass == "fusion":
            return f"{self.gene}-{self.partner} fusion"
        if self.klass == "cna":
            return f"{self.gene} {'Amplification' if self.variant == 'Amplification' else 'Deep deletion'}"
        return f"{self.gene} {self.short}"

    @property
    def genes(self) -> tuple[str, ...]:
        return (self.gene, self.partner) if self.klass == "fusion" else (self.gene,)

    def evidence_alterations(self) -> tuple[EvidenceAlteration, ...]:
        if self.klass == "fusion" and self.partner != self.gene:
            return self._ev(self.gene, self.partner), self._ev(self.partner, self.gene)
        return (self._ev(self.gene, self.partner),)


def sort_alterations(alts: Iterable[Alteration]) -> tuple[Alteration, ...]:
    seen, out = set(), []
    for a in alts:
        key = (a.klass, a.gene, a.short, a.partner)
        if key not in seen:
            seen.add(key)
            out.append(a)
    return tuple(sorted(out, key=lambda a: (CLASS_ORDER[a.klass], a.gene, a.display)))


class Diagnosis(_Frozen):
    main_type: str
    detailed_type: SourcedStr
    oncotree_code: str | None = None
    setting: str = ADVANCED_SETTING


class TumorProfile(_Frozen):
    panel_id: str | None
    panel_display: str
    panel_source: Literal["sample", "wes_fallback", "unknown"]
    panel_genes: tuple[str, ...]
    alterations: tuple[Alteration, ...]
    off_panel: tuple[Alteration, ...] = ()
    tmb: SourcedOptFloat
    msi_status: SourcedMsi
    msi_score: float | None = None
    msi_method: Literal["categorical", "MANTIS", "MSIsensor"] | None = None
    receptors: tuple[tuple[str, str], ...] = ()
    receptor_source: str | None = None

    @field_validator("alterations", "off_panel")
    @classmethod
    def _sorted(cls, v):
        return sort_alterations(v)

    @model_validator(mode="after")
    def _on_panel(self):
        if self.panel_genes:
            panel = set(self.panel_genes)
            bad = [a.display for a in self.alterations if not panel & set(a.genes)]
            if bad:
                raise ValueError(f"reported alterations not on panel {self.panel_id}: {bad}")
        return self


class Demographics(_Frozen):
    age: SourcedInt
    sex: SourcedSex
    ecog: SourcedInt
    cpic_population: SourcedOptStr
    race_raw: str | None = None
    ethnicity_raw: str | None = None

    @model_validator(mode="after")
    def _ranges(self):
        if not 0 <= self.ecog.value <= 4:
            raise ValueError("ECOG must be 0-4")
        if not 18 <= self.age.value <= 110:
            raise ValueError("age must be an adult age in years")
        return self

    @property
    def female(self) -> bool:
        return self.sex.value == "Female"


class Ulns(_Frozen):
    sex: Sex
    cycle: str
    bili: float
    ast: float
    alt: float
    scr: float
    method: str = "survey-weighted (WTMEC2YR) 97.5th percentile, adults, sex-specific"


class Labs(_Frozen):
    weight_kg: float
    height_cm: float | None
    scr: float
    bun: float
    bili: float
    ast: float
    alt: float
    albumin: float
    hgb: float
    plt: float
    anc: float
    crcl: float
    egfr: float
    hepatic_category: Literal["normal", "mild", "moderate", "severe"]
    ulns: Ulns
    source: str
    nhanes_seqn: int
    nhanes_cycle: str
    match_rung: str
    imputed: bool = False


class Medication(_Frozen):
    name: str
    ingredients: tuple[str, ...]
    source: str

    @model_validator(mode="after")
    def _check(self):
        if not self.name.strip() or _BARE_NUMBER.search(self.name):
            raise ValueError(f"medication name must be a drug name, got {self.name!r}")
        if not self.ingredients:
            raise ValueError("medication needs at least one ingredient RxCUI")
        return self


class DroppedDrug(_Frozen):
    raw: str
    reason: Literal["unresolved", "undetermined", "non_drug", "no_name", "antineoplastic", "excluded_atc", "non_systemic", "inactive"]
    ingredient: str | None = None


class MedList(_Frozen):
    items: tuple[Medication, ...]
    source: Literal["faers", "nhanes_rx"]
    detail: str
    rung: str | None = None
    dropped: tuple[DroppedDrug, ...] = ()

    @field_validator("items")
    @classmethod
    def _sorted_unique(cls, v):
        seen, out = set(), []
        for m in v:
            if m.name.lower() not in seen:
                seen.add(m.name.lower())
                out.append(m)
        return tuple(sorted(out, key=lambda m: m.name.lower()))


class PgxResult(_Frozen):
    gene: str
    alleles: tuple[str, str]
    phenotype: str
    activity_score: str | None
    normal_phenotype: str
    population_used: str | None
    redraws: int = 0

    @property
    def is_normal(self) -> bool:
        return self.phenotype == self.normal_phenotype

    @property
    def is_indeterminate(self) -> bool:
        return self.phenotype.strip().lower().startswith("indeterminate")

    @property
    def display(self) -> str:
        return f"{self.gene} {self.phenotype}"


class PgxPanel(_Frozen):
    results: tuple[PgxResult, ...]
    population: str | None
    pooled_fallback_genes: tuple[str, ...] = ()
    unavailable: tuple[tuple[str, str], ...] = ()
    uninformative: tuple[str, ...] = ()
    withheld: tuple[PgxResult, ...] = ()

    @field_validator("results")
    @classmethod
    def _by_gene(cls, v):
        return tuple(sorted(v, key=lambda r: r.gene))

    @property
    def non_normal(self) -> tuple[PgxResult, ...]:
        return tuple(r for r in self.results if not r.is_normal and not r.is_indeterminate)

    @property
    def indeterminate(self) -> tuple[PgxResult, ...]:
        return tuple(r for r in self.results if r.is_indeterminate)

    @property
    def not_tested(self) -> tuple[str, ...]:
        return tuple(sorted({g for g, _ in self.unavailable} | set(self.uninformative)))


class PriorTherapy(_Frozen):
    therapy: Therapy | None
    evidence_id: str | None
    regimen: str | None

    @model_validator(mode="after")
    def _check(self):
        if (self.therapy is None) != (self.regimen is None):
            raise ValueError("therapy and regimen must both be set or both be None")
        if self.regimen is not None and (not self.regimen.strip() or _BARE_NUMBER.search(self.regimen)):
            raise ValueError(f"regimen must list drug names, got {self.regimen!r}")
        return self


class StageSpread(_Frozen):
    stage: str | None = None
    t: str | None = None
    n: str | None = None
    m: str | None = None
    stage_source: str = "not_reported"
    sites: tuple[str, ...] = ()
    sites_source: str = "not_reported"


class Pathology(_Frozen):
    histology: str | None = None
    histology_source: str = "not_reported"
    grade: str | None = None
    grade_source: str = "not_reported"
    her2_ihc: str | None = None
    mmr: Literal["intact", "deficient", "not reported"] = "not reported"
    mmr_source: str = "not_reported"
    pdl1: Literal["positive", "negative", "not tested"] = "not tested"
    pdl1_source: str = "not_tested"


class TreatmentHistory(_Frozen):
    months: float | None = None
    best_response: Literal["complete response", "partial response", "stable disease", "progressive disease"] | None = None
    reason_stopped: Literal["disease progression", "toxicity"] | None = None
    source: str = "not_reported"


class PriorToxicity(_Frozen):
    term: str | None = None
    grade: int | None = None
    drug: str | None = None
    evidence_id: str | None = None
    source: str = "not_reported"


class Condition(_Frozen):
    name: str
    source: str


class Comorbidities(_Frozen):
    items: tuple[Condition, ...] = ()
    reported: bool = False
    donor: str | None = None


class GateEvent(_Frozen):
    field: str
    value: str
    action: Literal["kept", "redrawn", "suppressed"]
    reason: str = ""
    evidence_id: str | None = None


class Workup(_Frozen):
    setting: str = SETTING_LINE
    stage: StageSpread = StageSpread()
    pathology: Pathology = Pathology()
    history: TreatmentHistory = TreatmentHistory()
    toxicity: PriorToxicity = PriorToxicity()
    comorbidities: Comorbidities = Comorbidities()
    gate: tuple[GateEvent, ...] = ()
    followup: bool = False


class CaseMeta(_Frozen):
    study_id: str | None
    sample_ids: tuple[str, ...]
    sample_types: tuple[str, ...] = ()
    stage: str | None = None
    patient_source: Literal["cbioportal", "existing"]
    notes: tuple[str, ...] = ()


class CaseState(_Frozen):
    case_id: str
    demographics: Demographics
    diagnosis: Diagnosis
    tumor: TumorProfile
    pgx: PgxPanel
    prior: PriorTherapy
    meds: MedList
    labs: Labs
    meta: CaseMeta
    workup: Workup | None = None


class TrapRecord(_Frozen):
    case_id: str
    label: str
    components: tuple[str, ...] = ()
    details: dict[str, Any] = {}
    evidence_ids: tuple[str, ...] = ()
