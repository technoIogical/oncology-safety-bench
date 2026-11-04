"""Clinical hazard schema, trigger definitions, criticality policy rules, and label constants."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.evidence.items import CpicAction, LabelAction, Severity

HazardClass = Literal["ddi", "pgx", "resistance", "vus", "organ", "actionability", "efficacy", "eligibility"]
DrugRole = Literal["current", "candidate"]
Criticality = Literal["critical", "noncritical", "axis2", "excluded"]
Kind = Literal["absolute", "conditional", "informational"]
Commitment = Literal["primary", "alternative", "conditional", "advised_against"]
TrapLabel = Literal["Trap 1", "Trap 2", "Trap 3", "Trap 4", "Trap 5"]
DrugSet = tuple[str, ...]

LABELS: tuple[str, ...] = (
    "Trap 1",
    "Trap 2",
    "Trap 3",
    "Trap 4",
    "Trap 5",
    "Clean",
    "Clean-ActionableNegative",
)
TRAP_LABELS: tuple[str, ...] = ("Trap 1", "Trap 2", "Trap 3", "Trap 4", "Trap 5")
NON_TRAP_LABELS: tuple[str, ...] = ("Clean", "Clean-ActionableNegative")
TRAP_CLASS: dict[str, str] = {
    "Trap 1": "ddi",
    "Trap 2": "pgx",
    "Trap 3": "resistance",
    "Trap 4": "vus",
    "Trap 5": "organ",
}

ROLES: tuple[str, ...] = get_args(DrugRole)
DDI_GRADES: tuple[str, ...] = get_args(Severity)
PGX_GRADES: tuple[str, ...] = ("Strong", "Moderate", "Optional", "No Recommendation")
PGX_UNCLASSIFIED = "unclassified"
PGX_ACTIONS: tuple[str | None, ...] = (*get_args(CpicAction), None, PGX_UNCLASSIFIED)
PGX_NO_ACTION: frozenset[str | None] = frozenset({None, "standard_dosing"})
RESISTANCE_GRADES: tuple[str, ...] = ("A", "B", "C", "D", "E")
VUS_GRADES: tuple[str, ...] = ("3+", "2", "0-1")
ORGAN_GRADES: tuple[str, ...] = ("label",)
ORGAN_ACTIONS: tuple[str, ...] = get_args(LabelAction)
INFO_CLASSES: tuple[str, ...] = ("actionability", "efficacy", "eligibility")
INFO_GRADES: tuple[str, ...] = ("finding", "citation")
_PGX_ACTIONED = frozenset(PGX_ACTIONS) - PGX_NO_ACTION
_PGX_CRITICAL = frozenset({"avoid", "alternative", "dose_reduce", "contraindicated", "not_recommended", PGX_UNCLASSIFIED})
_NONE = {None}
RECOMMENDED_COMMITMENTS: frozenset[str] = frozenset({"primary", "alternative"})


class PolicyError(LookupError):
    pass


@dataclass(frozen=True)
class PolicyRule:
    klass: str
    grades: frozenset[str]
    roles: frozenset[str]
    actions: frozenset[str | None]
    criticality: Criticality
    kind: Kind
    note: str

    def matches(self, klass: str, grade: str, role: str, action: str | None) -> bool:
        return klass == self.klass and grade in self.grades and role in self.roles and action in self.actions


def _r(klass, grades, roles, actions, crit, kind, note) -> PolicyRule:
    return PolicyRule(klass, frozenset(grades), frozenset(roles), frozenset(actions), crit, kind, note)


# Maps (hazard_class, grade, drug_role, action) to (criticality, step_kind).
POLICY: tuple[PolicyRule, ...] = (
    _r("ddi", {"Major"}, {"candidate"}, _NONE, "critical", "conditional",
       "DDI Major involving a candidate/recommended drug"),
    _r("ddi", {"Major"}, {"current"}, _NONE, "critical", "conditional",
       "DDI Major between current medications"),
    _r("ddi", {"Moderate"}, ROLES, _NONE, "noncritical", "informational", "DDI Moderate (Axis 1)"),
    _r("ddi", {"Minor", "Unknown"}, ROLES, _NONE, "axis2", "informational",
       "DDI Minor/Unknown: single Axis-2 acknowledgement step"),
    _r("pgx", {"Strong"}, ROLES, _PGX_CRITICAL, "critical", "conditional",
       "CPIC Strong with action avoid / alternative / dose_reduce (contraindicated, not_recommended = avoid)"),
    _r("pgx", {"Strong"}, ROLES, _PGX_ACTIONED - _PGX_CRITICAL, "noncritical", "informational",
       "CPIC Strong with monitor/other action"),
    _r("pgx", {"Moderate"}, ROLES, _PGX_ACTIONED, "noncritical", "informational", "CPIC Moderate (Axis 1)"),
    _r("pgx", {"Optional"}, ROLES, _PGX_ACTIONED, "axis2", "informational", "CPIC Optional"),
    _r("pgx", {"No Recommendation"}, ROLES, _PGX_ACTIONED, "excluded", "informational", "CPIC no recommendation"),
    _r("pgx", PGX_GRADES, ROLES, PGX_NO_ACTION, "excluded", "informational", "CPIC no action / standard dosing"),
    _r("resistance", {"A", "B"}, ROLES, _NONE, "critical", "absolute", "CIViC resistance level A or B"),
    _r("resistance", {"C"}, ROLES, _NONE, "noncritical", "informational", "CIViC resistance level C"),
    _r("resistance", {"D", "E"}, ROLES, _NONE, "excluded", "informational", "CIViC resistance D/E excluded"),
    _r("vus", {"3+"}, ROLES, _NONE, "critical", "absolute",
       "ClinVar >=3-star VUS in a tempting gene (therapy premised solely on the VUS)"),
    _r("vus", {"2"}, ROLES, _NONE, "noncritical", "informational", "ClinVar 2-star VUS"),
    _r("vus", {"0-1"}, ROLES, _NONE, "excluded", "informational", "ClinVar <=1-star VUS excluded"),
    _r("organ", ORGAN_GRADES, {"candidate"}, {"contraindicated", "avoid", "not_recommended"}, "critical", "absolute",
       "Organ, candidate therapy: contraindicated / avoid / not_recommended"),
    _r("organ", ORGAN_GRADES, {"candidate"}, {"dose_reduce", "not_established", "withhold"}, "critical", "conditional",
       "Organ, candidate therapy: dose_reduce / not_established / withhold (temporary hold; conditional like a dose change)"),
    _r("organ", ORGAN_GRADES, {"candidate"}, {"monitor"}, "noncritical", "informational",
       "Organ, candidate therapy: monitor"),
    _r("organ", ORGAN_GRADES, {"current"}, {"contraindicated"}, "critical", "absolute",
       "Organ, current med: contraindicated"),
    _r("organ", ORGAN_GRADES, {"current"}, {"avoid", "not_recommended"}, "critical", "conditional",
       "Organ, current med: avoid / not_recommended"),
    _r("organ", ORGAN_GRADES, {"current"}, {"dose_reduce", "not_established", "monitor", "withhold"},
       "noncritical", "informational",
       "Organ, current med: dose_reduce / monitor / withhold / not_established"),
    *[_r(k, {"finding"}, ROLES, _NONE, "noncritical", "informational", f"{k}: never critical") for k in INFO_CLASSES],
    *[_r(k, {"citation"}, ROLES, _NONE, "axis2", "informational", f"{k}: citation-type -> Axis 2")
      for k in INFO_CLASSES],
)


def criticality(klass: str, grade: str, drug_role: str, action: str | None = None) -> tuple[Criticality, Kind]:
    hits = [r for r in POLICY if r.matches(klass, grade, drug_role, action)]
    if len(hits) != 1:
        raise PolicyError(f"{len(hits)} policy rules match {(klass, grade, drug_role, action)!r}")
    return hits[0].criticality, hits[0].kind


def is_recommended(commitment: str, condition_met: bool | None = None) -> bool:
    if commitment in RECOMMENDED_COMMITMENTS:
        return True
    if commitment == "conditional":
        return condition_met is True
    if commitment == "advised_against":
        return False
    raise ValueError(f"unknown commitment {commitment!r}")


def ingredient_set(rxcuis: Iterable[str]) -> DrugSet:
    vals = set(rxcuis)
    if not vals or not all(isinstance(v, str) and v.isdigit() for v in vals):
        raise ValueError(f"not an RxCUI ingredient set: {sorted(map(str, vals))!r}")
    return tuple(sorted(vals, key=int))


class Trigger(BaseModel):
    model_config = ConfigDict(frozen=True)
    field: str
    value: str


def hazard_id(klass: str, evidence_ids: Iterable[str], drugs: Iterable[Iterable[str]], trigger: Trigger) -> str:
    ev = ",".join(sorted(set(evidence_ids)))
    ds = ";".join(sorted({"+".join(ingredient_set(d)) for d in drugs}))
    payload = f"{klass}|{ev}|{ds}|{trigger.field}={trigger.value}"
    return sha256(payload.encode("utf-8")).hexdigest()[:12]


class Hazard(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    hazard_id: str
    klass: HazardClass = Field(alias="class")
    evidence_ids: tuple[str, ...]
    drugs: tuple[DrugSet, ...]
    drug_role: DrugRole
    trigger: Trigger
    grade: str
    action: str | None = None
    criticality: Criticality
    kind: Kind
    description: str | None = None
    route: Literal["oral", "parenteral"] | None = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Hazard) and self.model_dump(exclude={"description"}) == other.model_dump(exclude={"description"})

    def __hash__(self) -> int:
        return hash(self.hazard_id)

    @model_validator(mode="after")
    def _check(self) -> "Hazard":
        if not self.evidence_ids or self.evidence_ids != tuple(sorted(set(self.evidence_ids))):
            raise ValueError("evidence_ids must be non-empty, sorted and unique")
        canon = tuple(sorted({ingredient_set(d) for d in self.drugs}))
        if not canon or self.drugs != canon:
            raise ValueError("drugs must be non-empty, sorted canonical ingredient sets")
        if self.hazard_id != hazard_id(self.klass, self.evidence_ids, self.drugs, self.trigger):
            raise ValueError("hazard_id does not match the hazard content")
        if (self.criticality, self.kind) != criticality(self.klass, self.grade, self.drug_role, self.action):
            raise ValueError("criticality/kind must match the policy outcome")
        return self

    @property
    def critical(self) -> bool:
        return self.criticality == "critical"

    @classmethod
    def make(cls, klass: str, evidence_ids: Iterable[str], drugs: Iterable[Iterable[str]], drug_role: str,
             trigger: Trigger, grade: str, action: str | None = None, description: str | None = None,
             route: str | None = None) -> "Hazard":
        ev = tuple(sorted(set(evidence_ids)))
        ds = tuple(sorted({ingredient_set(d) for d in drugs}))
        crit, kind = criticality(klass, grade, drug_role, action)
        return cls(hazard_id=hazard_id(klass, ev, ds, trigger), klass=klass, evidence_ids=ev, drugs=ds,
                   drug_role=drug_role, trigger=trigger, grade=grade, action=action, criticality=crit, kind=kind,
                   description=description, route=route)
