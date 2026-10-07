"""
Clinical comparator base — shared types, outcome constants, metadata, and helpers.

All types and constants required by the individual comparator functions live here.
lambda_function.py imports from this module so nothing in the comparators package
needs to import from the parent — no circular imports possible.
"""
from __future__ import annotations

import re

try:
    from shared.clinical_fact_extractor import _extract_statistical_identity
except ImportError:
    _extract_statistical_identity = None


# ─── Severity levels ──────────────────────────────────────────────────────────
# Single source of truth for severity strings used across the validation engine.
_SEV_NONE   = "NONE"
_SEV_LOW    = "LOW"
_SEV_MEDIUM = "MEDIUM"
_SEV_HIGH   = "HIGH"
_SEV_FATAL  = "FATAL"

# Ordering used by _ValidationResult.severity() for data-driven escalation.
_SEV_RANK: dict[str, int] = {
    _SEV_NONE: 0, _SEV_LOW: 1, _SEV_MEDIUM: 2, _SEV_HIGH: 3, _SEV_FATAL: 4,
}
_RANK_SEV: dict[int, str] = {v: k for k, v in _SEV_RANK.items()}


# ─── Comparison outcomes ──────────────────────────────────────────────────────
#   MATCH           — both sides agree on this dimension
#   RELATED         — same clinical space, non-identical (e.g. COMBINATION backbone)
#   SPECIALIZATION  — candidate is a specific instance of the CI's class
#                     (e.g. CI = BCMA antibody → candidate = Teclistamab)
#   GENERALIZATION  — candidate is more general than what CI asked for
#                     (e.g. CI = Teclistamab → candidate = BCMA antibody)
#   UNKNOWN         — insufficient data on one or both sides; no inference made
#   CONFLICT        — confirmed semantic mismatch
CMP_MATCH          = "MATCH"
CMP_RELATED        = "RELATED"
CMP_SPECIALIZATION = "SPECIALIZATION"
CMP_GENERALIZATION = "GENERALIZATION"
CMP_UNKNOWN        = "UNKNOWN"
CMP_CONFLICT       = "CONFLICT"


# ─── ComparisonResult ─────────────────────────────────────────────────────────

class ComparisonResult:
    """
    Return type of every clinical comparator function.

    slot     — which dimension was compared ("drug", "temporal", …)
    outcome  — CMP_MATCH | CMP_RELATED | CMP_SPECIALIZATION | CMP_GENERALIZATION
               | CMP_UNKNOWN | CMP_CONFLICT
    score    — additive penalty (≤ 0 for non-MATCH outcomes; 0.0 for MATCH/UNKNOWN)
    severity — _SEV_* constant; meaningful for CONFLICT/RELATED/SPECIALIZATION/GENERALIZATION
    evidence — free-form dict for explainability and debugging

    Adding a new comparator:
      1. Write a function in its own file that returns ComparisonResult.
      2. Import it in comparators/__init__.py and append it to _COMPARATORS.
      Nothing else changes.
    """
    __slots__ = ("slot", "outcome", "score", "severity", "evidence")

    def __init__(
        self,
        slot:     str,
        outcome:  str,
        score:    float = 0.0,
        severity: str   = _SEV_NONE,
        evidence: dict | None = None,
    ) -> None:
        self.slot     = slot
        self.outcome  = outcome
        self.score    = score
        self.severity = severity
        self.evidence = evidence or {}

    def __repr__(self) -> str:
        return (
            f"ComparisonResult(slot={self.slot!r}, outcome={self.outcome!r}, "
            f"score={self.score}, severity={self.severity!r})"
        )


# ─── ClinicalContext ──────────────────────────────────────────────────────────

class ClinicalContext:
    """
    All enriched fields needed to compare one clinical object against another.

    Introduced to keep comparator signatures stable as the enrichment pipeline
    grows.  Adding a new identity field (e.g. temporal_context) requires only
    updating the two builder functions below — no comparator signatures change.
    """
    __slots__ = (
        "facts", "treatment", "population", "endpoint",
        "temporal", "negated_slots", "modality", "entities", "identity_overlap",
        "statistical_identity",
    )

    def __init__(
        self,
        facts:                dict,
        treatment:            dict,
        population:           dict,
        endpoint:             dict,
        temporal:             dict,
        negated_slots:        list,
        modality:             str,
        entities:             list,
        identity_overlap:     dict,
        statistical_identity: dict,
    ) -> None:
        self.facts                 = facts
        self.treatment             = treatment
        self.population            = population
        self.endpoint              = endpoint
        self.temporal              = temporal
        self.negated_slots         = negated_slots
        self.modality              = modality
        self.entities              = entities
        self.identity_overlap      = identity_overlap
        self.statistical_identity  = statistical_identity


def _refresh_statistical_identity(text: str, stored: dict) -> dict:
    """Re-extract statistical facts from the object's own text at query time —
    stale/incomplete pre-indexed enrichment (older docs indexed before an
    extractor fix) must not silently suppress a real conflict. Freshly
    extracted values win on key collision; re-indexing is not required."""
    if not text or _extract_statistical_identity is None:
        return stored
    try:
        fresh = _extract_statistical_identity(text)
    except Exception:
        return stored
    return {**stored, **fresh} if fresh else stored


def _build_ci_context(ci: dict, entities: list) -> "ClinicalContext":
    """Build a ClinicalContext from req['ci'] (the CI dict from OpenSearch)."""
    return ClinicalContext(
        facts                = ci.get("effective_facts") or ci.get("facts") or {},
        treatment            = ci.get("treatment_identity") or {},
        population           = ci.get("population_identity") or {},
        endpoint             = ci.get("endpoint_identity") or {},
        temporal             = ci.get("temporal_context") or {},
        negated_slots        = ci.get("negated_slots") or [],
        modality             = (ci.get("modality") or "GENERAL").upper(),
        entities             = entities,
        identity_overlap     = {},   # N/A for the CI itself
        statistical_identity = _refresh_statistical_identity(
            ci.get("knownCI", ""), ci.get("statistical_identity") or {}
        ),
    )


def _build_cand_context(cand: dict) -> "ClinicalContext":
    """Build a ClinicalContext from a candidate dict (matched_object + identity_overlap)."""
    obj = cand.get("matched_object") or {}
    return ClinicalContext(
        facts                = obj.get("effective_facts") or obj.get("facts") or {},
        treatment            = obj.get("treatment_identity") or {},
        population           = obj.get("population_identity") or {},
        endpoint             = obj.get("endpoint_identity") or {},
        temporal             = obj.get("temporal_context") or {},
        negated_slots        = obj.get("negated_slots") or [],
        modality             = (obj.get("modality") or "GENERAL").upper(),
        entities             = obj.get("entities") or [],
        identity_overlap     = cand.get("identity_overlap") or {},
        statistical_identity = _refresh_statistical_identity(
            obj.get("text", ""), obj.get("statistical_identity") or {}
        ),
    )


# ─── Semantic comparator metadata ─────────────────────────────────────────────
# Single source of truth for weight AND per-issue severity for every comparator.
# Adding a new comparator: one entry here (and one new file + __init__ registration).

_CONFLICT_METADATA: dict[str, dict] = {
    "drug":        {"weight": -0.35, "severity": _SEV_HIGH},
    "endpoint":    {"weight": -0.28, "severity": _SEV_MEDIUM},
    "population":  {"weight": -0.20, "severity": _SEV_LOW},
    "phase":       {"weight": -0.18, "severity": _SEV_LOW},
    "biomarker":   {"weight": -0.15, "severity": _SEV_LOW},
    "study_arm":   {"weight": -0.15, "severity": _SEV_LOW},
    "modality":    {"weight": -0.22, "severity": _SEV_MEDIUM},
    "regimen":     {"weight": -0.18, "severity": _SEV_MEDIUM},
    "temporal":    {"weight": -0.12, "severity": _SEV_LOW},   # base; _TEMPORAL_FIELD_SEVERITY overrides per-field
    "negation":    {"weight": -0.28, "severity": _SEV_HIGH},
    "statistical": {"weight": -0.35, "severity": _SEV_HIGH},  # base; per-key overrides in statistical.py
}
_CONFLICT_WEIGHTS: dict[str, float] = {k: v["weight"] for k, v in _CONFLICT_METADATA.items()}

# Gradient drug-conflict penalty by DrugRelation label.
# The aggregator computes drug.relation via the identity graph and stores it on
# each candidate's identity_overlap.drug.relation field.
#
#   EXACT        → no penalty
#   COMBINATION  → one drug is part of the other's regimen (mild)
#   SAME_FAMILY  → same mechanism class, different compound → SPECIALIZATION
#   RELATED      → different class, overlapping clinical space → GENERALIZATION
#   DIFFERENT    → confirmed distinct drug → full CONFLICT penalty
_DRUG_CONTRA_WEIGHTS: dict[str, float] = {
    "EXACT":        0.00,
    "COMBINATION": -0.10,
    "SAME_FAMILY": -0.18,
    "RELATED":     -0.25,
    "DIFFERENT":   -0.35,
}

# Modality groups: modalities in the same group are compatible; cross-group = CONFLICT.
# GENERAL and unclassified never fire — absence of data ≠ contradiction.
_MODALITY_GROUP: dict[str, str] = {
    "OBJECTIVE":   "assertion",    # "What we aim to measure"
    "ENDPOINT":    "assertion",
    "EFFICACY":    "assertion",
    "PROCEDURE":   "operational",  # "What the patient must do / receive"
    "REQUIREMENT": "operational",
    "ELIGIBILITY": "operational",
    "DOSING":      "operational",
    "OBSERVATION": "evidence",     # "What was observed / resulted"
    "RESULT":      "evidence",
}

# Temporal field severity and weight — only the most-severe conflict per candidate is reported.
_TEMPORAL_FIELD_SEVERITY: dict[str, tuple[str, float]] = {
    "protocol_version": (_SEV_HIGH,   -0.25),
    "phase":            (_SEV_HIGH,   -0.22),
    "cycle":            (_SEV_MEDIUM, -0.15),
    "day":              (_SEV_LOW,    -0.08),
    "week":             (_SEV_LOW,    -0.08),
    "window":           (_SEV_LOW,    -0.05),
}


# ─── Endpoint family normalisation ────────────────────────────────────────────
# Prevents false mismatches when CI uses an abbreviation (ORR) and the candidate
# uses the long form ("overall response [PR or better]") or vice versa.

_ENDPOINT_FAMILY: dict[str, str] = {
    "orr":                              "overall_response",
    "overall response rate":            "overall_response",
    "overall response":                 "overall_response",
    "overall response (pr or better)": "overall_response",
    "overall response [pr or better]": "overall_response",
    "pr or better":                     "overall_response",
    "pfs":                              "progression_free_survival",
    "progression free survival":        "progression_free_survival",
    "progression-free survival":        "progression_free_survival",
    "os":                               "overall_survival",
    "overall survival":                 "overall_survival",
    "cr":                               "complete_response",
    "complete response":                "complete_response",
    "cr or better":                     "complete_response",
    "scr":                              "complete_response",
    "stringent complete response":      "complete_response",
    "vgpr":                             "vgpr_or_better",
    "vgpr or better":                   "vgpr_or_better",
    "very good partial response":       "vgpr_or_better",
    "mrd":                              "mrd",
    "minimal residual disease":         "mrd",
    "mrd negativity":                   "mrd",
    "mrd-negativity":                   "mrd",
    "mrd negative":                     "mrd",
    "dor":                              "duration_of_response",
    "duration of response":             "duration_of_response",
    "ttr":                              "time_to_response",
    "time to response":                 "time_to_response",
    "cbr":                              "clinical_benefit_rate",
    "clinical benefit rate":            "clinical_benefit_rate",
    "pfs2":                             "pfs2",
    "ttnt":                             "time_to_next_treatment",
    "time to next treatment":           "time_to_next_treatment",
}


def _ep_family(val: str) -> str:
    """Normalise an endpoint string to its clinical family for conflict comparison."""
    norm = val.lower().strip()
    if norm in _ENDPOINT_FAMILY:
        return _ENDPOINT_FAMILY[norm]
    # Word-boundary containment only — a plain substring check lets short
    # abbreviation keys like "cr" or "os" spuriously match inside unrelated
    # words ("criteria" contains "cr"), silently mis-tagging CIs with an
    # endpoint identity they don't have.
    for key, family in _ENDPOINT_FAMILY.items():
        if re.search(rf'\b{re.escape(key)}\b', norm) or re.search(rf'\b{re.escape(norm)}\b', key):
            return family
    return norm


# ─── Entity label sets ────────────────────────────────────────────────────────
# Module-level frozensets so each comparator file can import exactly what it needs.

_CMP_DRUG_LABELS        = frozenset({"MEDICATION", "TREATMENT_NAME", "BRAND_NAME"})
_CMP_ENDPOINT_LABELS    = frozenset({"CLINICAL_ENDPOINT", "QUESTIONNAIRE"})
_CMP_PHASE_LABELS       = frozenset({"PHASE"})
_CMP_POPULATION_LABELS  = frozenset({"STUDY_POPULATION", "PATIENT_POPULATION"})
_CMP_BIOMARKER_LABELS   = frozenset({"BIOMARKER", "GENETIC_VARIANT"})
_CMP_ARM_LABELS         = frozenset({"STUDY_ARM"})
# Comprehend Medical tags compound regimen names ("Tal-DP", "DPd") as
# TEST_TREATMENT_PROCEDURE / sub_type TREATMENT_NAME — include them for drug comparisons.
_CMP_TREAT_DRUG_SUBTYPES = frozenset({"TREATMENT_NAME", "DRUG", "GENERIC_NAME", "BRAND_NAME"})


# ─── Extraction helpers ───────────────────────────────────────────────────────

def _cmp_slot_vals(
    facts: dict, key: str, entities: list[dict], labels: frozenset,
) -> list[str]:
    """Normalised values from facts dict + entity list for a single comparison slot."""
    vals: list[str] = [v.lower().strip() for v in facts.get(key, []) if v]
    for e in entities:
        label    = e.get("label") or e.get("type", "")
        text     = (e.get("text") or "").lower().strip()
        sub_type = (e.get("sub_type") or "").upper()
        if not text:
            continue
        if label in labels:
            vals.append(text)
        elif (
            key == "drug"
            and label == "TEST_TREATMENT_PROCEDURE"
            and sub_type in _CMP_TREAT_DRUG_SUBTYPES
        ):
            vals.append(text)
    return [v for v in vals if v]


def _cmp_ep_vals(facts: dict, entities: list[dict]) -> list[str]:
    """Endpoint values from facts + entities, normalised to clinical family."""
    raw = _cmp_slot_vals(facts, "endpoint", entities, _CMP_ENDPOINT_LABELS)
    raw += [
        (e.get("text") or "").lower().strip()
        for e in entities
        if e.get("label") == "BIOMARKER" and e.get("text")
    ]
    return list({_ep_family(v) for v in raw if v})


def _cmp_has_conflict(a: list[str], b: list[str]) -> bool:
    """True when both sides are non-empty and no value overlaps (substring-normalised)."""
    if not a or not b:
        return False
    return not any(av in bv or bv in av for av in a for bv in b)
