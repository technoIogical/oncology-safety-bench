from __future__ import annotations

import re
from typing import Annotated, Iterable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

InteractionType = Literal["single", "combination", "substitutes", "sequential"]
LLMCutoffParameter = Literal["CrCl", "eGFR", "bilirubin_xULN", "AST_xULN", "Child-Pugh", "other"]
HoldParameter = Literal["ANC", "platelets", "ALT_xULN", "AST_xULN"]
CutoffParameter = Literal["CrCl", "eGFR", "bilirubin_xULN", "AST_xULN", "Child-Pugh", "other", "ANC", "platelets", "ALT_xULN"]
Comparator = Literal["<", "<=", ">", ">=", "range"]
CutoffMethod = Literal["cockcroft_gault", "ckd_epi", "nci_odwg", "unspecified"]
LabelAction = Literal["contraindicated", "avoid", "not_recommended", "dose_reduce", "not_established", "monitor", "withhold"]
CpicAction = Literal["alternative", "avoid", "contraindicated", "not_recommended", "dose_reduce", "monitor",
                     "standard_dosing", "other"]
Setting = Literal["first_line", "previously_treated", "any_line", "adjuvant", "neoadjuvant", "maintenance", "other"]
Severity = Literal["Major", "Moderate", "Minor", "Unknown"]

BUNDLE_SCHEMA_VERSION = "1"
ID_PATTERNS: dict[str, re.Pattern] = {
    "civic": re.compile(r"^civic:EID\d+$"),
    "label_cutoff": re.compile(r"^fda:[^:]+:[^:]+:cutoff:(?:\d+|[0-9a-f]{12}(?:-\d+)?)$"),
    "label_indication": re.compile(r"^fda:[^:]+:[^:]+:ind:(?:\d+|[0-9a-f]{12}(?:-\d+)?)$"),
    "label_warning": re.compile(r"^fda:[^:]+:[^:]+:warn:(?:\d+|[0-9a-f]{12}(?:-\d+)?)$"),
    "ddi": re.compile(r"^ddinter:DDInter\d+-DDInter\d+$"),
    "cpic": re.compile(r"^cpic:rec:\d+$"),
    "clinvar": re.compile(r"^clinvar:(VCV\d+|absent:[^:]+:.+)$"),
    "pubmed": re.compile(r"^pmid:\d+$"),
    "trial": re.compile(r"^nct:NCT\d{8}$"),
    "guideline": re.compile(r"^soc:[A-Za-z0-9._-]+$"),
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Therapy(_Frozen):
    ingredients: tuple[str, ...]
    names: tuple[str, ...]
    interaction_type: InteractionType

    @model_validator(mode="after")
    def _check(self):
        if tuple(sorted(set(self.ingredients), key=int)) != self.ingredients:
            raise ValueError("ingredients must be unique RxCUIs sorted numerically")
        if len(self.names) != len(self.ingredients):
            raise ValueError("names must align with ingredients")
        if (self.interaction_type == "single") != (len(self.ingredients) == 1):
            raise ValueError("single therapies have exactly one ingredient; others at least two")
        return self

    @classmethod
    def of(cls, pairs: Iterable[tuple[str, str]], interaction_type: InteractionType) -> "Therapy":
        uniq = dict(sorted({str(r): n for r, n in pairs}.items(), key=lambda kv: int(kv[0])))
        return cls(ingredients=tuple(uniq), names=tuple(uniq.values()), interaction_type=interaction_type)

    @property
    def key(self) -> str:
        return f"{self.interaction_type}:{'+'.join(self.ingredients)}"


class Biomarker(_Frozen):
    kind: Literal["mutation", "amplification", "deletion", "fusion", "expression", "TMB-H", "MSI-H", "dMMR", "other"]
    gene: str | None = None
    alteration: str | None = None
    threshold: float | None = None
    threshold_unit: str | None = None
    genes: tuple[str, ...] = ()


class _Item(_Frozen):
    id: str

    @model_validator(mode="after")
    def _check_id(self):
        if not ID_PATTERNS[self.type].match(self.id):
            raise ValueError(f"id {self.id!r} does not match the {self.type} format")
        return self


class CivicItem(_Item):
    type: Literal["civic"] = "civic"
    eid: int
    gene: str
    variant_name: str
    molecular_profile: str
    level: Literal["A", "B", "C", "D", "E"]
    rating: int | None
    significance: str
    direction: str
    status: str
    disease_name: str | None
    doid: str | None
    match_type: Literal["exact", "ancestor"] | None
    therapy: Therapy | None
    unresolved_therapies: tuple[str, ...] = ()
    matched_alteration: str | None = None
    is_fusion: bool = False


class LabelCutoffItem(_Item):
    type: Literal["label_cutoff"] = "label_cutoff"
    drug_rxcui: str
    set_id: str
    version: str
    effective_time: str | None
    section: str
    parameter: CutoffParameter
    comparator: Comparator
    low: float | None
    high: float | None
    unit: str
    method: CutoffMethod
    action: LabelAction
    amount: str | None
    quote: str
    applies_to: Literal["dosing", "restart", "monitoring", "hold"] = "dosing"
    compound: bool = False
    child_pugh_classes: tuple[Literal["A", "B", "C"], ...] = ()
    applies_when: str | None = None
    or_branch: bool = False
    flags: tuple[str, ...] = ()
    route: Literal["oral", "parenteral"] | None = None


class LabelIndicationItem(_Item):
    type: Literal["label_indication"] = "label_indication"
    drug_rxcuis: tuple[str, ...]
    set_id: str
    version: str
    disease_text: str
    doid: str | None
    biomarker: Biomarker | None
    setting: Setting
    combination_partners: tuple[str, ...] = ()
    unresolved_partners: tuple[str, ...] = ()
    quote: str
    subsection: str | None = None
    supportive_care: bool = False


class LabelWarningItem(_Item):
    type: Literal["label_warning"] = "label_warning"
    drug_rxcui: str
    set_id: str
    version: str
    kind: Literal["contraindication", "boxed_warning", "limitation_of_use"]
    statement: str
    quote: str
    genes: tuple[str, ...] = ()
    polarity: Literal["mutant", "wild_type"] | None = None
    section: str | None = None


class DdiItem(_Item):
    type: Literal["ddi"] = "ddi"
    ingredient_a: str
    ingredient_b: str
    name_a: str
    name_b: str
    severity: Severity
    description: str | None
    source_files: tuple[str, ...]


class CpicCondition(_Frozen):
    gene: str
    lookup: str
    phenotype: str | None = None
    activity_score: str | None = None
    allele_status: str | None = None


class CpicItem(_Item):
    type: Literal["cpic"] = "cpic"
    recommendation_id: int
    drug_rxcui: str
    drug_name: str
    gene: str
    phenotype: str | None
    activity_score: str | None
    classification: str
    cpic_level: str | None
    action: CpicAction | None
    action_quote: str | None
    recommendation_text: str
    conditions: tuple[CpicCondition, ...] = ()
    population: str | None = None
    action_status: Literal["verified", "rule", "unclassified"] | None = None
    unclassified_reason: str | None = None
    secondary_actions: tuple[str, ...] = ()


class ClinvarItem(_Item):
    type: Literal["clinvar"] = "clinvar"
    found: bool
    gene: str
    variant: str
    accession: str | None = None
    title: str | None = None
    classification: str | None = None
    review_status: str | None = None
    stars: int | None = None
    conflicts: bool | None = None
    conditions: tuple[str, ...] = ()
    classification_type: Literal["germline"] | None = None
    oncogenicity: str | None = None
    oncogenicity_review_status: str | None = None
    somatic_clinical_impact: str | None = None
    somatic_clinical_impact_review_status: str | None = None
    civic_evidence: tuple[str, ...] = ()


class PubmedItem(_Item):
    type: Literal["pubmed"] = "pubmed"
    pmid: str
    title: str
    journal: str
    year: str | None
    abstract: str


class TrialItem(_Item):
    type: Literal["trial"] = "trial"
    nct_id: str
    title: str
    status: str
    phase: str
    eligibility_text: str
    min_age: float | None = None
    max_age: float | None = None
    ecog_max: int | None = None
    ecog_quote: str | None = None
    biomarker_text: str | None = None
    biomarker_quote: str | None = None


class GuidelineItem(_Item):
    type: Literal["guideline"] = "guideline"
    rule_id: str
    source: str
    line: Literal["1L", "2L"]
    statement: str
    options: tuple[str, ...] = ()
    option_rxcuis: tuple[tuple[str, ...], ...] = ()
    option_conditions: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    after_rxcuis: tuple[tuple[str, ...], ...] = ()
    rules_sha256: str
    flags: tuple[str, ...] = ("needs_clinician_signoff",)


EvidenceItem = Annotated[
    Union[CivicItem, LabelCutoffItem, LabelIndicationItem, LabelWarningItem, DdiItem, CpicItem, ClinvarItem,
          PubmedItem, TrialItem, GuidelineItem],
    Field(discriminator="type"),
]
EVIDENCE_ITEM_ADAPTER: TypeAdapter = TypeAdapter(EvidenceItem)


class EvidenceBundle(BaseModel):
    schema_version: str = BUNDLE_SCHEMA_VERSION
    case_id: str
    case_doid: str | None
    doid_match: dict | None
    items: list[EvidenceItem]
    approved_indication_ids: list[str] = Field(default_factory=list)
    extraction_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    snapshot_manifest: list[dict] = Field(default_factory=list)
    evidence_provenance: dict | None = None

    def get(self, item_id: str) -> EvidenceItem:
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(item_id)

    def by_type(self, item_type: str) -> list[EvidenceItem]:
        return [i for i in self.items if i.type == item_type]

    def ids(self) -> list[str]:
        return [i.id for i in self.items]

    def approved_indications(self) -> list[LabelIndicationItem]:
        return [self.get(i) for i in self.approved_indication_ids]


def _ddinter_num(ddinter_id: str) -> int:
    return int(ddinter_id.replace("DDInter", ""))


def ddi_id(ddinter_a: str, ddinter_b: str) -> str:
    a, b = sorted((ddinter_a, ddinter_b), key=_ddinter_num)
    return f"ddinter:{a}-{b}"
