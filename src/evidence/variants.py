from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

Kind = Literal["missense", "nonsense", "synonymous", "frameshift", "inframe_del", "inframe_ins", "inframe_dup",
               "delins", "splice", "extension", "start_loss", "promoter", "amplification", "deletion", "fusion"]

AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
       "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
       "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V", "Ter": "*", "Xaa": "X"}
_AA3_ALT = "|".join(AA3)
_AA3_RE = re.compile(f"(?i)({_AA3_ALT})")
_STARTS_AA3 = re.compile(f"(?i)^(?:{_AA3_ALT}|\\*)\\d")
_SHAPE = re.compile(f"(?i)^(?:[A-Z*]|{_AA3_ALT})\\d+(?:[A-Z*=]|{_AA3_ALT}|fs|del|ins|dup|_|>|splice)")
_SUBST = re.compile(r"^([A-Z*])(\d+)([A-Z*=])$")
_FS = re.compile(r"^([A-Z*])(\d+)[A-Z*]?FS(?:\*\d*|\*\?)?$")
_SPLICE = re.compile(r"^[A-Z]?(\d+)_?SPLICE$")
_EXT = re.compile(r"^([A-Z*])(\d+)([A-Z*])EXT.*$")
_RANGE = re.compile(r"^([A-Z*])(\d+)_([A-Z*]?)(\d+)(DELINS|DEL|DUP|INS|>|=)([A-Z*]*)$")
_POINT = re.compile(r"^([A-Z*])(\d+)(DELINS|DEL|DUP|INS)([A-Z*]*)$")
_CDNA_TAIL = re.compile(r"\s*\(c\.[^)]*\)\s*$")
_MNV = re.compile(r"^([A-Z]{2,4})(\d+)([A-Z]{2,4})$")
_START_LOSS = re.compile(r"(?i)^(?:M|Met)1\?$")
_UPSTREAM = re.compile(r"(?i)^(?:c\.)?-(\d+)([ACGT])>([ACGT])$")
_AMP = {"amplification", "amp", "gain"}
_DEL = {"deletion", "deep deletion", "homozygous deletion", "homdel", "loss"}
_FUSION = {"fusion", "fusions", "rearrangement"}
_PROMOTER = {"promoter", "promoter mutation", "5'flank", "5' flank"}


def _strip(text: str) -> str:
    t = _CDNA_TAIL.sub("", (text or "").strip())
    t = re.sub(r"(?i)^\(?p\.", "", t).strip("() ")
    return t


def _prepare(text: str) -> str:
    t = _strip(text)
    if _STARTS_AA3.match(t):
        t = _AA3_RE.sub(lambda m: AA3[m.group(1).capitalize()], t)
    return t.upper()


def _alternatives(text: str) -> list[str]:
    out: list[str] = []
    for part in (p.strip() for p in (text or "").split(",")):
        m = re.match(r"^(.*\d)([A-Za-z*])/([A-Za-z*])$", part)
        out += [m.group(1) + m.group(2), m.group(1) + m.group(3)] if m else [part]
    return [p for p in out if p]


def _canonical_one(text: str) -> str | None:
    t = _strip(text)
    if _START_LOSS.match(t):
        return "M1?"
    m = _MNV.match(t)
    if m and len(m.group(1)) == len(m.group(3)) and not (m.group(1).capitalize() in AA3 and m.group(3).capitalize() in AA3):
        ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)
        return f"{ref[0]}{pos}_{ref[-1]}{pos + len(ref) - 1}delins{alt}"
    if not looks_like_protein(text):
        return None
    s = _prepare(text)
    m = _SUBST.match(s)
    if m:
        ref, pos, alt = m.groups()
        if alt == "X" and ref != "X":
            alt = "*"
        return f"{ref}{pos}=" if alt in ("=", ref) else f"{ref}{pos}{alt}"
    m = _FS.match(s)
    if m:
        return f"{m.group(1)}{m.group(2)}fs"
    m = _SPLICE.match(s)
    if m:
        return f"X{m.group(1)}_splice"
    m = _EXT.match(s)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}ext"
    m = _RANGE.match(s)
    if m:
        a, p1, b, p2, op, tail = m.groups()
        if op == "=":
            return f"{a}{p1}_{b}{p2}=" if not tail else None
        if op == ">":
            return f"{a}{p1}_{b}{p2}delins{tail}" if tail else None
        if op in ("DEL", "DUP"):
            return f"{a}{p1}_{b}{p2}{op.lower()}"
        return f"{a}{p1}_{b}{p2}{op.lower()}{tail}" if tail else None
    m = _POINT.match(s)
    if m:
        a, p1, op, tail = m.groups()
        if op in ("DEL", "DUP"):
            return f"{a}{p1}{op.lower()}"
        return f"{a}{p1}{op.lower()}{tail}" if tail else None
    return None


def _kind(short: str) -> Kind:
    if short == "M1?":
        return "start_loss"
    if short.endswith("_splice"):
        return "splice"
    if short.endswith("ext"):
        return "extension"
    if short.endswith("fs"):
        return "frameshift"
    if "delins" in short:
        return "delins"
    if short.endswith("del"):
        return "inframe_del"
    if short.endswith("dup"):
        return "inframe_dup"
    if "ins" in short:
        return "inframe_ins"
    if short.endswith("="):
        return "synonymous"
    if short.endswith("*"):
        return "nonsense"
    return "missense"


@dataclass(frozen=True)
class CanonicalVariant:
    gene: str
    short: str
    kind: Kind

    @property
    def canonical(self) -> str:
        if self.kind in ("amplification", "deletion", "fusion", "promoter"):
            return f"{self.gene} {self.short}"
        return f"{self.gene} p.{self.short}"


@dataclass(frozen=True)
class Alteration:
    gene: str
    variant: str
    klass: Literal["mutation", "cna", "fusion"]
    partner: str | None = None
    mutation_type: str | None = None
    vaf: float | None = None

    @property
    def short(self) -> str:
        if self.klass == "mutation":
            return canonical_protein(self.variant) or self.variant
        return self.variant

    @property
    def label(self) -> str:
        return f"{self.gene} {self.short}"


def looks_like_protein(text: str) -> bool:
    t = _strip(text)
    return bool(_SHAPE.match(t)) and "?" not in t


def canonical_proteins(text: str) -> list[str]:
    out = []
    for alt in _alternatives(text) or [text]:
        c = _canonical_one(alt)
        if c and c not in out:
            out.append(c)
    return out


@lru_cache(maxsize=None)
def canonical_protein(text: str) -> str | None:
    found = canonical_proteins(text)
    return found[0] if found else None


def canonical_variant(gene: str, text: str) -> CanonicalVariant | None:
    g = (gene or "").strip().upper()
    low = (text or "").strip().lower()
    if low in _AMP:
        return CanonicalVariant(g, "Amplification", "amplification")
    if low in _DEL:
        return CanonicalVariant(g, "Deletion", "deletion")
    if low in _FUSION:
        return CanonicalVariant(g, "Fusion", "fusion")
    if g == "TERT" and low in _PROMOTER:
        return CanonicalVariant(g, "Promoter", "promoter")
    m = _UPSTREAM.match((text or "").strip())
    if g == "TERT" and m:
        return CanonicalVariant(g, f"c.-{m.group(1)}{m.group(2).upper()}>{m.group(3).upper()}", "promoter")
    short = canonical_protein(text)
    return CanonicalVariant(g, short, _kind(short)) if short else None
