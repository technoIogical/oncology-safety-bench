"""Curated standard-of-care (SOC) regimen matcher and FDA indication lookup by tumor profile."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

from src.case import CaseState
from src.tool_backends.tables import global_tables

RULES_PATH = Path(__file__).resolve().parents[1] / "data" / "curation" / "soc_rules.json"
LINES = ("1L", "2L")
TMB_HIGH = 10.0

_EGFR_EX19 = re.compile(r"^(E746|L747|L747|E749|A750|T751|S752|I744|K745|P753)_\w+(del|delins\w*)$")
_EGFR_EX20INS = re.compile(r"^(A763|Y764|A767|S768|V769|D770|N771|P772|H773|V774|C775)_\w+(ins\w+|dup)$")
_EGFR_EX20_SENSITIVE = ("A763_Y764insFQEA",)
_INFRAME_DEL = re.compile(r"^[A-Z]?(\d+)_[A-Z]?(\d+)(del|delins\w*)$|^[A-Z]?(\d+)(del)$")
_TRUNCATING = re.compile(r"(\*|fs\*?\d*|_splice|^X\d+_splice)$")
_FLT3_ITD = re.compile(r"^[A-Z]\d{3}_[A-Z]\d{3}(ins\w+|dup)$|^[A-Z]\d{3}(ins\w+|dup)$")
_KIT_EX11 = re.compile(r"^(K550|P551|V555|Q556|W557|K558|V559|V560|E561|Y568|Y570|L576)")
_PDGFRA_SENS = re.compile(r"^(V561|N659|D842|I843|M844|H845|D846|S566|R560)")
_KIT_GIST_RESISTANT = ("D816V", "V654A", "T670I")
_PI3K_AKT = ("PIK3CA_ACT", "AKT1_E17K", "PTEN_LOSS")

VARIANT_CLASSES: dict[str, tuple[tuple[str, ...], Callable]] = {
    "RAS_MUT": (("KRAS", "NRAS"), lambda a: _ras_mut(a)),
    "KRAS_G12C": (("KRAS",), lambda a: _mut(a) and a.variant == "G12C"),
    "BRAF_V600E": (("BRAF",), lambda a: _mut(a) and a.variant == "V600E"),
    "BRAF_V600": (("BRAF",), lambda a: _mut(a) and a.variant in ("V600E", "V600K", "V600D", "V600R")),
    "EGFR_SENSITIZING": (("EGFR",), lambda a: _mut(a) and (a.variant in ("L858R", "G719A", "G719C", "G719S", "L861Q",
                                                                         "S768I") + _EGFR_EX20_SENSITIVE
                                                           or bool(_EGFR_EX19.match(a.variant)))),
    "EGFR_COMMON": (("EGFR",), lambda a: _mut(a) and (a.variant == "L858R" or bool(_EGFR_EX19.match(a.variant)))),
    "EGFR_EX20INS": (("EGFR",), lambda a: _mut(a) and bool(_EGFR_EX20INS.match(a.variant))
                     and a.variant not in _EGFR_EX20_SENSITIVE),
    "EGFR_T790M": (("EGFR",), lambda a: _mut(a) and a.variant == "T790M"),
    "ALK_FUSION": (("ALK",), lambda a: a.klass == "fusion"),
    "ROS1_FUSION": (("ROS1",), lambda a: a.klass == "fusion"),
    "RET_FUSION": (("RET",), lambda a: a.klass == "fusion"),
    "NTRK_FUSION": (("NTRK1", "NTRK2", "NTRK3"), lambda a: a.klass == "fusion"),
    "MET_EX14": (("MET",), lambda a: _met_ex14(a)),
    "ERBB2_AMP": (("ERBB2",), lambda a: a.klass == "cna" and a.variant == "Amplification"),
    "ERBB2_MUT": (("ERBB2",), lambda a: _mut(a) and (a.variant in ("S310F", "S310Y", "L755S", "V777L", "R678Q",
                                                                   "D769H", "D769Y", "V842I")
                                                     or bool(re.match(r"^(A775|G776|V777|Y772|M774)_\w+ins\w+$",
                                                                      a.variant or "")))),
    "BRCA_PATH": (("BRCA1", "BRCA2"), lambda a: _loss(a)),
    "BRCA_PALB2": (("BRCA1", "BRCA2", "PALB2"), lambda a: _loss(a)),
    "HRR_PATH": (("BRCA1", "BRCA2", "ATM", "PALB2", "CDK12", "CHEK1", "CHEK2", "BARD1", "BRIP1", "FANCL", "RAD51B", "RAD51C",
                  "RAD51D", "RAD54L"), lambda a: _loss(a)),
    "IDH1_R132": (("IDH1",), lambda a: _mut(a) and (a.variant or "").startswith("R132")),
    "IDH2_HOT": (("IDH2",), lambda a: _mut(a) and ((a.variant or "").startswith("R140") or
                                                   (a.variant or "").startswith("R172"))),
    "FGFR2_FUSION": (("FGFR2",), lambda a: a.klass == "fusion"),
    "FGFR3_ALT": (("FGFR3",), lambda a: (a.klass == "fusion") or (_mut(a) and a.variant in (
        "R248C", "S249C", "G370C", "Y373C"))),
    "PIK3CA_ACT": (("PIK3CA",), lambda a: _mut(a) and a.variant in (
        "C420R", "E542K", "E545A", "E545D", "E545G", "E545K", "Q546E", "Q546R", "H1047L", "H1047R", "H1047Y")),
    "ESR1_MUT": (("ESR1",), lambda a: _mut(a) and a.variant in (
        "E380Q", "L536H", "L536P", "L536R", "Y537C", "Y537N", "Y537S", "D538G", "S463P", "V422del")),
    "AKT1_E17K": (("AKT1",), lambda a: _mut(a) and a.variant == "E17K"),
    "PTEN_LOSS": (("PTEN",), lambda a: _loss(a)),
    "FLT3_ITD": (("FLT3",), lambda a: _mut(a) and bool(_FLT3_ITD.match(a.variant or "")) and 572 <= (_pos(a.variant)
                                                                                                    or 0) <= 630),
    "FLT3_TKD": (("FLT3",), lambda a: _mut(a) and (a.variant or "")[:4] in ("D835", "I836")),
    "NPM1_MUT": (("NPM1",), lambda a: _mut(a) and "fs" in (a.variant or "") and (_pos(a.variant) or 0) >= 280),
    "KIT_MUT": (("KIT",), lambda a: _mut(a) and (bool(_KIT_EX11.match(a.variant or "")) or (a.variant or "")[:4] in (
        "A502", "Y503") or a.variant in ("K642E", "D816V", "N822K", "V654A", "T670I"))),
    "PDGFRA_D842V": (("PDGFRA",), lambda a: _mut(a) and a.variant == "D842V"),
    "GIST_IMATINIB": (("KIT", "PDGFRA"), lambda a: _gist_imatinib(a)),
}
VARIANT_CLASSES["PI3K_AKT_PTEN"] = (("PIK3CA", "AKT1", "PTEN"),
                                    lambda a: any(VARIANT_CLASSES[c][1](a) for c in _PI3K_AKT
                                                  if a.gene in VARIANT_CLASSES[c][0]))

_COMPILED: dict[tuple, tuple] = {}


class CuratedSocError(ValueError):
    pass


@dataclass(frozen=True)
class Regimen:
    drugs: tuple[str, ...]
    label: str
    rxcuis: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    when: tuple[str, ...] | None = None
    condition: str | None = None

    @property
    def key(self) -> frozenset[str]:
        return frozenset(self.rxcuis)


@dataclass(frozen=True)
class Rule:
    id: str
    group: str
    line: str
    when: tuple[str, ...]
    setting: str | None
    options: tuple[Regimen, ...]
    after: tuple[Regimen, ...] = ()
    source: str = ""
    suppress: tuple[dict, ...] = ()


@dataclass(frozen=True)
class Group:
    id: str
    main_types: tuple[str, ...]
    oncotree: tuple[str, ...] = ()
    exclude_oncotree: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleSet:
    version: str
    sha256: str
    groups: tuple[Group, ...]
    rules: tuple[Rule, ...]
    compiled: bool = False

    def group_of(self, state) -> Group | None:
        mt = (state.diagnosis.main_type or "").casefold()
        code = state.diagnosis.oncotree_code or ""
        hits = [g for g in self.groups if mt in (m.casefold() for m in g.main_types)
                and (not g.oncotree or code in g.oncotree) and code not in g.exclude_oncotree]
        if len(hits) > 1:
            raise CuratedSocError(f"case {state.case_id} matches groups {[g.id for g in hits]}")
        return hits[0] if hits else None


@dataclass(frozen=True)
class CuratedOption:
    regimen: Regimen
    tier: str
    condition: str | None


@dataclass(frozen=True)
class Resolution:
    rule: Rule
    options: tuple[CuratedOption, ...]
    evidence_id: str

    @property
    def setting(self) -> str | None:
        return self.rule.setting


@dataclass(frozen=True)
class StandardOfCare:
    covered: bool
    rule_id: str | None = None
    line: str | None = None
    names: tuple[str, ...] = ()
    rxcuis: tuple[str, ...] = ()


NOT_COVERED = StandardOfCare(covered=False)


def _pos(variant: str) -> int | None:
    m = re.match(r"^[A-Z*](\d+)", variant)
    return int(m.group(1)) if m else None


def _mut(a) -> bool:
    return a.klass == "mutation"


def _ras_mut(a) -> bool:
    return _mut(a) and a.gene in ("KRAS", "NRAS") and re.match(r"^[GAQKA]\d+[A-Z]$", a.variant or "") is not None \
        and _pos(a.variant) in (12, 13, 59, 61, 117, 146)


def _met_ex14(a) -> bool:
    if not _mut(a):
        return False
    v = a.variant or ""
    m = re.match(r"^X(\d+)_splice$", v)
    if m:
        return 963 <= int(m.group(1)) <= 1028
    if re.match(r"^D(1010|1028)[A-Z]$", v):
        return True
    m = _INFRAME_DEL.match(v)
    if m and "fs" not in v:
        lo = int(m.group(1) or m.group(4))
        hi = int(m.group(2) or m.group(4))
        return any(lo <= b + 2 and hi >= b - 3 for b in (963, 1010, 981, 1028))
    return False


def _gist_imatinib(a) -> bool:
    if a.gene == "KIT":
        v, p = a.variant or "", _pos(a.variant or "")
        if not _mut(a) or _truncating(a) or v in _KIT_GIST_RESISTANT or p is None:
            return False
        return (550 <= p <= 592 or (502 <= p <= 509 and bool(re.search(r"(ins|dup)", v))) or p == 642
                or p in (820, 822, 823))
    return _mut(a) and a.variant != "D842V" and bool(_PDGFRA_SENS.match(a.variant or ""))


def _truncating(a) -> bool:
    return _mut(a) and bool(_TRUNCATING.search(a.variant or ""))


def _loss(a) -> bool:
    return _truncating(a) or (a.klass == "cna" and a.variant == "Deletion")


def variant_class(state, name: str) -> bool | None:
    if name not in VARIANT_CLASSES:
        raise CuratedSocError(f"unknown variant class {name!r}")
    genes, test = VARIANT_CLASSES[name]
    alts = [a for a in state.tumor.alterations if a.gene in genes or (a.partner in genes if a.partner else False)]
    if any(test(a) if a.gene in genes else test(_as_gene(a, genes)) for a in alts):
        return True
    panel = set(state.tumor.panel_genes)
    return False if panel and set(genes) <= panel else None


def _as_gene(a, genes):
    return a.model_copy(update={"gene": a.partner, "partner": a.gene}) if a.partner in genes else a


def _receptors(state) -> dict[str, str]:
    rec = {k.upper(): str(v).lower() for k, v in state.tumor.receptors}
    if "HR" not in rec:
        er, pr = rec.get("ER"), rec.get("PR")
        if "positive" in (er, pr):
            rec["HR"] = "positive"
        elif er == "negative" and pr == "negative":
            rec["HR"] = "negative"
    return rec


def receptor(state, key: str) -> str | None:
    key = key.upper()
    v = _receptors(state).get(key) or None
    if key == "HER2" and v in (None, "negative") and variant_class(state, "ERBB2_AMP") is True:
        return "positive" if v is None else None
    return v if v in ("positive", "negative") else None


def msi_high(state) -> bool | None:
    v = state.tumor.msi_status.value
    if v is None:
        return None
    return str(v).upper().replace(" ", "") in ("MSI-H", "MSIH", "MSI-HIGH")


def tmb_high(state) -> bool | None:
    v = state.tumor.tmb.value
    return None if v is None else float(v) >= TMB_HIGH


def predicate(state, token: str) -> bool | None:
    """Evaluates a rule condition token in three-valued logic (True, False, or None when unassessed)."""
    op = token[0] if token[:1] in "!?~^" else ""
    atom = token[1:] if op else token
    kind, _, arg = atom.partition(":")
    if kind == "alt":
        v = variant_class(state, arg)
    elif kind == "msi" and arg == "high":
        v = msi_high(state)
    elif kind == "tmb" and arg == "high":
        v = tmb_high(state)
    elif kind == "receptor":
        k, _, want = arg.partition("=")
        r = receptor(state, k)
        v = None if r is None else (r == want)
    elif kind == "oncotree":
        v = (state.diagnosis.oncotree_code or "") == arg
    elif kind == "detailed":
        v = arg.casefold() in (state.diagnosis.detailed_type.value or "").casefold()
    elif kind == "prior":
        t = state.prior.therapy
        v = arg.casefold() in {n.casefold() for n in t.names} if t is not None else False
    elif kind == "ecog":
        v = int(state.demographics.ecog.value) >= int(arg)
    elif kind == "unreported":
        v = None
    else:
        raise CuratedSocError(f"unknown predicate {token!r}")
    if op == "!":
        return v is False
    if op == "?":
        return v is None
    if op == "~":
        return v is not True
    if op == "^":
        return v is not False
    return v


def _whens(w) -> tuple[str, ...] | None:
    if w is None:
        return None
    return (w,) if isinstance(w, str) else tuple(w)


def all_of(state, tokens: Iterable[str]) -> bool | None:
    vals = [predicate(state, t) for t in tokens]
    if any(v is False for v in vals):
        return False
    return None if any(v is None for v in vals) else True


def _regimen(d: dict) -> Regimen:
    return Regimen(drugs=tuple(d["drugs"]), label=d.get("label") or " + ".join(d["drugs"]), when=_whens(d.get("when")),
                   condition=d.get("condition"))


@lru_cache(maxsize=4)
def load_rules(path: Path = RULES_PATH) -> RuleSet:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    groups = tuple(Group(id=g["id"], main_types=tuple(g["main_types"]), oncotree=tuple(g.get("oncotree", ())),
                         exclude_oncotree=tuple(g.get("exclude_oncotree", ()))) for g in data["groups"])
    rules = tuple(Rule(id=r["id"], group=r["group"], line=r["line"], when=tuple(r.get("when", ())),
                       setting=r.get("setting"), options=tuple(_regimen(o) for o in r["options"]),
                       after=tuple(_regimen(o) for o in r.get("after", ())), source=r.get("source", ""),
                       suppress=tuple(r.get("suppress", ()))) for r in data["rules"])
    return RuleSet(version=str(data.get("version", "")), sha256=hashlib.sha256(raw).hexdigest(), groups=groups,
                   rules=rules)


def rules_sha256(path: Path = RULES_PATH) -> str:
    return load_rules(path).sha256


def compile_rules(alias_index, rules: RuleSet | None = None) -> RuleSet:
    """Resolves drug names in all rule regimens and suppressions to canonical RxNorm ingredient RxCUIs."""
    rules = rules or load_rules()
    if rules.compiled:
        return rules
    key = (rules.sha256, id(alias_index))
    if key in _COMPILED:
        return _COMPILED[key][1]

    def comp(reg: Regimen, where: str) -> Regimen:
        rx: list[str] = []
        names: list[str] = []
        for d in reg.drugs:
            r = alias_index.resolve(d)
            ings = tuple(r.ingredients) if r.status == "match" else ()
            if len(ings) != 1:
                raise CuratedSocError(f"{where}: drug {d!r} does not resolve to one ingredient ({r.status}, {ings})")
            if ings[0] in rx:
                raise CuratedSocError(f"{where}: drug {d!r} repeats an ingredient")
            rx.append(ings[0])
            names.append(alias_index.ingredient_name(ings[0]) or d)
        order = sorted(range(len(rx)), key=lambda i: int(rx[i]))
        return Regimen(drugs=reg.drugs, label=reg.label, rxcuis=tuple(rx[i] for i in order),
                       names=tuple(names[i] for i in order), when=reg.when, condition=reg.condition)
    out = RuleSet(version=rules.version, sha256=rules.sha256, groups=rules.groups, compiled=True, rules=tuple(
        Rule(id=r.id, group=r.group, line=r.line, when=r.when, setting=r.setting,
             options=tuple(comp(o, f"{r.id} option {o.label}") for o in r.options),
             after=tuple(comp(o, f"{r.id} after {o.label}") for o in r.after), source=r.source,
             suppress=tuple({**sup, "rxcuis": list(comp(_regimen({"drugs": sup["drugs"]}), f"{r.id} suppress").rxcuis)}
                            for sup in r.suppress))
        for r in rules.rules))
    _COMPILED[key] = (alias_index, out)
    return out


def option_condition(o) -> str:
    return o.condition or (f"acceptable only if {' and '.join(o.when)}" if o.when else "")


def guideline_id(rule_id: str) -> str:
    return f"soc:{rule_id}"


def _matches(state, rule: Rule) -> bool:
    return all(predicate(state, t) is True for t in rule.when)


def line_of(state) -> str:
    return "2L" if state.prior.therapy is not None else "1L"


def matching_rules(state, rules: RuleSet, line: str | None = None) -> list[Rule]:
    g = rules.group_of(state)
    if g is None:
        return []
    return [r for r in rules.rules if r.group == g.id and (line is None or r.line == line) and _matches(state, r)]


def resolve(state, alias_index, rules: RuleSet | None = None) -> Resolution | None:
    rules = compile_rules(alias_index, rules)
    line = line_of(state)
    hits = matching_rules(state, rules, line)
    if line == "2L":
        prior = frozenset(state.prior.therapy.ingredients)
        hits = [r for r in hits if any(a.key == prior for a in r.after)]
    if len(hits) > 1:
        raise CuratedSocError(f"case {state.case_id} matches rules {[r.id for r in hits]}")
    if not hits:
        return None
    rule = hits[0]
    opts = []
    for o in rule.options:
        v = True if o.when is None else all_of(state, o.when)
        if v is False:
            continue
        opts.append(CuratedOption(regimen=o, tier="approved" if v is True else "conditional",
                                  condition=None if v is True else option_condition(o)))
    return Resolution(rule=rule, options=tuple(opts), evidence_id=guideline_id(rule.id))


def standard_of_care(case: dict, alias_index=None, rules=None) -> StandardOfCare:
    """Resolves a case dictionary to its curated standard-of-care regimens and ingredient RxCUIs."""
    raw = case.get("case_state")
    if not raw:
        return NOT_COVERED
    state = CaseState.model_validate(raw)
    if alias_index is None:
        alias_index = global_tables().lexicon
    res = resolve(state, alias_index, rules if rules is not None else load_rules())
    if res is None:
        return NOT_COVERED
    seen, names, rxcuis = set(), [], []
    for opt in res.options:
        for rx, name in zip(opt.regimen.rxcuis, opt.regimen.names):
            if rx not in seen:
                seen.add(rx)
                names.append(name)
                rxcuis.append(rx)
    return StandardOfCare(covered=True, rule_id=res.rule.id, line=res.rule.line, names=tuple(names),
                          rxcuis=tuple(rxcuis))
