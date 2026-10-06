"""
Search Pipeline — Stage 3: Aggregator
=======================================
Merges hit lists from all selected retrievers, deduplicates by chunk_id,
and computes a hybrid score that combines:

  Retriever scores (weighted by retriever reliability)
    vector   × 0.40
    literal  × 0.25
    bm25     × 0.20
    fact     × 0.15   ← new: structured facts.* field query
    ontology × 0.10
    ner      × 0.03
    regex    × 0.02

  Entity overlap bonus      + 0.10 × Jaccard(CI entities, candidate entities)
  Fact overlap bonus        + 0.08 × slot-weighted Jaccard(CI facts, candidate facts)
  Relation co-occurrence    + 0.05 × entity-type-pair overlap (drug+AE, drug+dose, …)
  Section alignment         × 1.00–1.25 multiplicative boost when candidate section
                              matches the expected section for the CI type

Input
------
{
    "search_id":        str,
    "document_id":      str,
    "ci":               dict,
    "classification":   dict,
    "retriever_results": list[{"retriever": str, "hits": list[Hit]}]
}

Appends
-------
"candidates": list[Candidate]

Candidate schema
----------------
{
    "chunk_id":      str,
    "page_start":    int,
    "page_end":      int,
    "sources":       list[str],
    "max_score":     float,       # best individual retriever score (kept for compatibility)
    "agg_score":     float,       # hybrid combined score
    "entity_overlap": float,      # Jaccard CI↔candidate entity overlap (0–1)
    "fact_overlap":  float,       # slot-weighted Jaccard CI↔candidate facts (0–1)
    "relation_score": float,      # entity-type-pair co-occurrence score (0–1)
    "section_boost":  float,      # section alignment multiplier applied
    "snippet":       str,
}
"""

from __future__ import annotations

import json
import logging
import re
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── #5 Hybrid ranking weights ─────────────────────────────────────────────────
# Weights represent how much we trust each retriever's score signal.
# Sum intentionally < 1.0 — entity and relation bonuses fill the rest.
_RETRIEVER_WEIGHTS: dict[str, float] = {
    "vector":   0.40,   # semantic understanding — most reliable
    "literal":  0.25,   # exact match — very high precision when it fires
    "bm25":     0.20,   # keyword overlap — reliable recall
    "fact":     0.15,   # structured facts.* field match — high precision slot signal
    "ontology": 0.10,   # synonym expansion — useful but less specific
    "numeric":  0.45,   # structured numeric/statistical query — primary signal for NUMERIC/STATISTICAL CIs
    "ner":      0.03,
    "regex":    0.02,
}
_ENTITY_WEIGHT   = 0.10   # additive: Jaccard entity text overlap
_FACT_WEIGHT     = 0.08   # additive: slot-weighted Jaccard fact overlap
_RELATION_WEIGHT = 0.05   # additive: entity-type-pair co-occurrence

# ── Retriever score normalisation ────────────────────────────────────────────
# BM25 and literal retrievers return raw OpenSearch TF-IDF scores (range 10–200+)
# while vector returns cosine similarity (0–1).  Without normalisation a single
# BM25 score of 49 × 0.20 = 9.8 dwarfs a perfect vector 0.99 × 0.40 = 0.40.
#
# Defaults were derived from corpus-level p90 measurements on this document set:
#   bm25: mean≈40, max≈92   literal: mean≈27, max≈195
#   fact: mean≈22, max≈84   ontology: mean≈5, max≈11
#
# When the index changes materially (different BM25 analyser, new embedding model,
# order-of-magnitude more studies) recompute p95 values and update the env vars
# RETRIEVER_CAP_BM25 / RETRIEVER_CAP_LITERAL / etc. — no code change needed.
# Use tests/compute_retriever_caps.py to sample the live index and print new values.
# Scores above the cap are clamped to 1.0 (saturated signal — not a hard rejection).
_RETRIEVER_SCORE_CAP: dict[str, float] = {
    "bm25":     float(os.environ.get("RETRIEVER_CAP_BM25",     "40.0")),
    "literal":  float(os.environ.get("RETRIEVER_CAP_LITERAL",  "30.0")),
    "fact":     float(os.environ.get("RETRIEVER_CAP_FACT",     "25.0")),
    "ontology": float(os.environ.get("RETRIEVER_CAP_ONTOLOGY", "10.0")),
    "vector":   float(os.environ.get("RETRIEVER_CAP_VECTOR",    "1.0")),
    "ner":      float(os.environ.get("RETRIEVER_CAP_NER",       "1.0")),
    "regex":    float(os.environ.get("RETRIEVER_CAP_REGEX",     "1.0")),
    "numeric":  float(os.environ.get("RETRIEVER_CAP_NUMERIC",  "20.0")),
}

# ── Semantic identity signals ─────────────────────────────────────────────────
# Identity bonus: reward direct drug/endpoint overlap in effective_facts.
# This lifts semantically correct candidates above pure keyword hits.
_IDENTITY_BONUS_WEIGHT  =  0.15   # additive; max +0.15 when drug or endpoint fully matches

# Zero-identity penalty: fires only when CI has extractable drug/endpoint AND
# the candidate has neither.  NOT a hard gate — a genuine match that happens to
# omit the drug due to heading inheritance can still win via identity_bonus.
_ZERO_IDENTITY_PENALTY  = -0.40

# Fully-unenriched penalty: candidate has no entities, facts, or effective_facts
# (NER stage failed).  Should rank below any enriched candidate.
_ZERO_ENRICHMENT_PENALTY = -0.25

# CI types that use the numeric retriever.  These types carry no drug/endpoint
# enrichment by design (the number IS the evidence) so structural penalties
# that fire when enrichment is absent must be suppressed for them.
_NUMERIC_CI_TYPES: frozenset[str] = frozenset({
    # Fine-grained subtypes (current)
    "NUMERIC_SAMPLE_SIZE", "CONFIDENCE_INTERVAL", "P_VALUE",
    "HAZARD_RATIO", "ODDS_RATIO", "NUMERIC_PERCENTAGE", "MEDIAN",
    # Coarse types (legacy — kept for any in-flight events)
    "NUMERIC", "STATISTICAL",
})

# ── Granularity bonus — break near-ties in favour of more specific evidence ──
# A sentence at 0.91 should beat a paragraph at 0.90 for a specific CI query.
# The bonus is small enough not to override a genuinely stronger paragraph hit.
_GRANULARITY_BONUS: dict[str, float] = {
    "sentence":  0.02,
    "paragraph": 0.01,
    "heading":   0.00,
}

# ── #3 Section alignment ──────────────────────────────────────────────────────
# CI type → set of section_category values expected to contain that evidence.
# Matched candidates receive a multiplicative boost; mismatched ones are not penalised.
_CI_TYPE_TO_SECTIONS: dict[str, set[str]] = {
    "PHARMACOKINETICS":  {"PHARMACOKINETICS", "PK", "PK_PD", "PHARMACOLOGY"},
    "SAFETY":            {"SAFETY", "ADVERSE_EVENTS", "TOLERABILITY", "RISK"},
    "EFFICACY":          {"OBJECTIVES", "ENDPOINTS", "EFFICACY", "RESULTS"},
    "DOSING":            {"DOSING", "DOSE_MODIFICATIONS", "DOSE_ESCALATION", "ADMINISTRATION", "BACKGROUND"},
    "POPULATION":        {"ELIGIBILITY", "INCLUSION_CRITERIA", "POPULATION", "DEMOGRAPHICS"},
    "PROTOCOL":          {"STUDY_DESIGN", "PROTOCOL", "METHODS", "DESIGN", "PROCEDURES"},
    "OBJECTIVE":         {"OBJECTIVES", "ENDPOINTS", "BACKGROUND"},
    "BIOMARKER":         {"BIOMARKERS", "TRANSLATIONAL", "CORRELATIVE"},
    "STUDY_DESIGN":      {"STUDY_DESIGN", "PROTOCOL", "METHODS", "OVERVIEW"},
    "PHARMACODYNAMICS":  {"PHARMACODYNAMICS", "PK_PD", "PHARMACOLOGY"},
    "MANUFACTURING":     {"CMC", "MANUFACTURING", "FORMULATION", "CHEMISTRY", "CHARACTERIZATION"},
}
_SECTION_BOOST_MATCH    = 1.25   # boost for exact section category match
_SECTION_BOOST_DEFAULT  = 1.00   # no boost when section unknown or CI type unrecognised

# ── Section drift penalties ───────────────────────────────────────────────────
# Applied as additive penalties (≤ 0) when a CI's intent and the candidate's
# heading path or section contradict at a structural level (e.g. Primary
# Objective CI matched against a Secondary Endpoints object).
# Distinct from _SECTION_BOOST which rewards correct sections — this fires
# ONLY on confirmed mismatch evidence, not on absence of section info.
# Cross-domain mismatches (Objective ↔ Eligibility, etc.) are included.
_SECTION_DRIFT_PENALTY_PRIMARY_SECONDARY = -0.25  # primary CI → secondary heading
_SECTION_DRIFT_PENALTY_DOMAIN_CROSS      = -0.20  # objective/efficacy CI → eligibility section
_SECTION_DRIFT_PENALTY_BACKGROUND        = -0.15  # obj/efficacy CI → background/appendix

# Sections that constitute a domain-cross for OBJECTIVE/EFFICACY CIs
_ELIGIBILITY_SECTIONS = frozenset({
    "ELIGIBILITY", "INCLUSION_CRITERIA", "EXCLUSION_CRITERIA",
    "STUDY_POPULATION",
})
_BACKGROUND_SECTIONS = frozenset({
    "BACKGROUND", "APPENDIX", "REFERENCES", "ADMINISTRATIVE", "ABBREVIATIONS",
})

# ── Study-context penalty ─────────────────────────────────────────────────────
# CIs asking about the current protocol should not retrieve paragraphs that
# describe cited/historical evidence about OTHER studies.
#
# Applied as a multiplicative penalty on the hybrid score before the candidate
# list is passed to the reranker — fewer CITED candidates reach reranking.
#
# CITED    = text describes a different external study's design/results
# HISTORICAL = general literature / preclinical background
# GENERAL  = no classification available → no penalty (conservative)
_CURRENT_PROTOCOL_CI_TYPES = frozenset({
    "OBJECTIVE", "EFFICACY", "DOSING", "POPULATION",
    "PROTOCOL", "STUDY_DESIGN", "PHARMACOKINETICS", "PHARMACODYNAMICS",
    "SAFETY", "ENDPOINT", "BIOMARKER",
})
_STUDY_CONTEXT_PENALTY: dict[str, float] = {
    "CITED":      0.70,   # paragraph about another study → large penalty
    "HISTORICAL": 0.82,   # preclinical / prior-art background → moderate penalty
    "CURRENT":    1.00,   # current-protocol text → no change
    "GENERAL":    1.00,   # unknown → no change (conservative)
}

# ── Fact slot boosts (mirrors fact_retriever._SLOT_BOOST) ───────────────────
# Higher weight = slot is more clinically specific = a match matters more.
_FACT_SLOT_BOOST: dict[str, float] = {
    "drug":               4.0,
    "study_id":           4.0,
    "study_arm":          3.0,
    "endpoint":           2.5,
    "adverse_event":      2.0,
    "biomarker":          2.0,
    "study_population":   1.5,
    "dose":               1.5,
    "phase":              1.2,
    "response_criterion": 1.0,
    "statistical_method": 1.0,
}

# ── #4 Relation proxy — entity-type pairs that imply a clinical relationship ──
# If both types co-occur in the CI AND in the candidate, that's a relation signal.
_RELATION_PAIRS: list[tuple[str, str]] = [
    ("MEDICATION",      "ADVERSE_EVENT"),      # drug causes/exacerbates AE
    ("MEDICATION",      "MEDICATION"),          # combination regimen
    ("MEDICATION",      "DOSAGE"),              # drug at a specific dose
    ("TREATMENT_NAME",  "ADVERSE_EVENT"),
    ("TREATMENT_NAME",  "TREATMENT_PHASE"),
    ("MEDICATION",      "TEST_NAME"),           # drug with a PK/PD test
    ("MEDICATION",      "PROCEDURE"),
]

# ── Retrieval quality penalties — single source of truth for every structural penalty ──
# Each entry maps a structural issue type to its additive weight (always ≤ 0).
# These are about **retrieval quality** (is this object likely to be useful?),
# NOT clinical correctness.  Clinical-value mismatches (drug / endpoint / phase)
# are surfaced as raw identity_overlap evidence for the reranker to interpret.
#
# Philosophy: aggregator = evidence fusion + structural quality filters;
#             reranker   = clinical reasoning + value-level contradiction scoring.
_RETRIEVAL_QUALITY_PENALTIES: dict[str, dict] = {
    "missing_relation":    {"weight": -0.12},   # CI has relations; candidate has none
    "stmt_type_mismatch":  {"weight": -0.10},   # candidate stmt_type incompatible with CI type
    "abbreviation_object": {"weight": -0.20},   # candidate is an abbreviation/definition list
    "missing_fact_slot":   {"weight": -0.06},   # per required slot CI has but candidate lacks
    "section_mismatch":    {"weight": -0.08},   # candidate section incompatible with CI category
}

# Endpoint family normalisation — mirrors the reranker's _ENDPOINT_FAMILY.
# Prevents false endpoint-mismatch penalties when the CI uses an abbreviation
# ("ORR") and the candidate uses the long form ("overall response [PR or better]").
_AGG_ENDPOINT_FAMILY: dict[str, str] = {
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


def _agg_ep_family(val: str) -> str:
    """Normalise an endpoint string to its clinical family (aggregator copy)."""
    norm = val.lower().strip()
    if norm in _AGG_ENDPOINT_FAMILY:
        return _AGG_ENDPOINT_FAMILY[norm]
    # Word-boundary containment only — a plain substring check lets short
    # abbreviation keys like "cr" or "os" spuriously match inside unrelated
    # words ("criteria" contains "cr"), silently mis-tagging CIs with an
    # endpoint identity they don't have.
    for key, family in _AGG_ENDPOINT_FAMILY.items():
        if re.search(rf'\b{re.escape(key)}\b', norm) or re.search(rf'\b{re.escape(norm)}\b', key):
            return family
    return norm

# Statement types that are structurally incompatible with a CI type.
# GENERAL / BACKGROUND = extractor found no clinical assertion — strong signal
# that the chunk is context, not evidence, for an OBJECTIVE/EFFICACY CI.
_INCOMPATIBLE_STMT_TYPES: dict[str, frozenset] = {
    "OBJECTIVE":        frozenset({"GENERAL", "BACKGROUND", "DEFINITION", "ABBREVIATION"}),
    "EFFICACY":         frozenset({"GENERAL", "BACKGROUND", "DEFINITION", "ABBREVIATION"}),
    "DOSING":           frozenset({"BACKGROUND", "DEFINITION", "ABBREVIATION"}),
    "SAFETY":           frozenset({"BACKGROUND", "DEFINITION", "ABBREVIATION"}),
    "PHARMACOKINETICS": frozenset({"BACKGROUND", "DEFINITION", "ABBREVIATION"}),
}

# Sections that should almost never hold evidence for these CI types.
_INCOMPATIBLE_SECTIONS: dict[str, frozenset] = {
    "OBJECTIVE":     frozenset({"PROCEDURES", "MANUFACTURING", "CMC", "CHEMISTRY",
                                "CHARACTERIZATION", "APPENDIX", "ADMINISTRATIVE"}),
    "EFFICACY":      frozenset({"PROCEDURES", "MANUFACTURING", "CMC", "APPENDIX",
                                "ADMINISTRATIVE"}),
    "DOSING":        frozenset({"MANUFACTURING", "CMC", "CHEMISTRY", "APPENDIX",
                                "ADMINISTRATIVE"}),
    "SAFETY":        frozenset({"MANUFACTURING", "CMC", "APPENDIX", "ADMINISTRATIVE"}),
    "MANUFACTURING": frozenset({"OBJECTIVES", "ENDPOINTS", "EFFICACY", "RESULTS"}),
}

# Required fact slots per CI type: penalise when CI *has* the slot but candidate lacks it.
# If the CI itself has no drug/endpoint extracted, no expectation is set.
_REQUIRED_FACT_SLOTS: dict[str, list] = {
    "OBJECTIVE":        ["drug", "endpoint"],
    "EFFICACY":         ["drug", "endpoint"],
    "DOSING":           ["drug", "dose"],
    "SAFETY":           ["drug", "adverse_event"],
    "PHARMACOKINETICS": ["drug", "endpoint"],
}

# Abbreviation-list detector: ≥3 occurrences of "ABBREV=definition" in one object
_ABBREV_RE = re.compile(r'\b[A-Z]{2,8}\s*=\s*\w')

# Drug identity graph (optional import — degrades gracefully to binary conflict if absent)
try:
    from shared.drug_identity import DrugRelation, DRUG_RELATION_WEIGHTS, best_drug_relation as _best_drug_relation
    _DRUG_IDENTITY_AVAILABLE = True
except ImportError:
    _DRUG_IDENTITY_AVAILABLE = False
    DrugRelation = None  # type: ignore
    DRUG_RELATION_WEIGHTS = {}  # type: ignore



# Entity label sets used by slot-value extraction (module-level so helpers can share them)
_DRUG_LABELS          = frozenset({"MEDICATION", "TREATMENT_NAME", "BRAND_NAME"})
_ENDPOINT_LABELS      = frozenset({"CLINICAL_ENDPOINT", "BIOMARKER", "QUESTIONNAIRE"})
_TREAT_DRUG_SUBTYPES  = frozenset({"TREATMENT_NAME", "DRUG", "GENERIC_NAME", "BRAND_NAME"})


# ── ValidationIssue — lightweight structured contradiction signal ─────────────
# Intentionally stripped down vs the reranker edition:
#   • No severity field   — the Aggregator ranks, it does not gate
#   • No severity()       — no FATAL/HIGH/MEDIUM/LOW escalation here
#   • No merge()          — single flat issue list per candidate
#   • No veto logic       — CE veto belongs in the reranker
class ValidationIssue:
    """One contradiction signal detected between CI and a candidate object."""
    __slots__ = ("type", "weight", "evidence")

    def __init__(
        self,
        type:     str,
        weight:   float           = 0.0,
        evidence: dict | None     = None,
    ) -> None:
        self.type     = type
        self.weight   = weight
        self.evidence = evidence if evidence is not None else {}

    def __repr__(self) -> str:
        return (
            f"ValidationIssue(type={self.type!r}, weight={self.weight}, "
            f"evidence={self.evidence!r})"
        )


class _ValidationResult:
    """Container for contradiction issues detected for one candidate."""
    __slots__ = ("issues",)

    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []


# ── Module-level slot-value helpers (shared by detect & test code) ────────────

def _slot_vals(
    entities: list[dict],
    labels:   frozenset,
    facts:    dict,
    key:      str,
) -> list[str]:
    """Extract normalised slot values from entity list + facts dict."""
    vals: list[str] = []
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
            and sub_type in _TREAT_DRUG_SUBTYPES
        ):
            vals.append(text)
    vals += [v.lower().strip() for v in facts.get(key, []) if v]
    return vals


def _ep_slot_vals(entities: list[dict], facts: dict) -> list[str]:
    """Extract endpoint values normalised to clinical family."""
    raw = _slot_vals(entities, _ENDPOINT_LABELS, facts, "endpoint")
    return [_agg_ep_family(v) for v in raw if v]


def _add_issue(
    result:          "_ValidationResult",
    issue_type:      str,
    evidence:        dict,
    weight_override: float | None = None,
) -> None:
    """
    Append one ValidationIssue.

    weight_override lets callers supply a custom penalty instead of the
    _RETRIEVAL_QUALITY_PENALTIES default.
    """
    weight = weight_override if weight_override is not None else _RETRIEVAL_QUALITY_PENALTIES[issue_type]["weight"]
    result.issues.append(ValidationIssue(
        type=issue_type,
        weight=weight,
        evidence=evidence,
    ))


_aws: dict = {}

def _get(service: str):
    if service not in _aws:
        import boto3
        _aws[service] = boto3.client(service)
    return _aws[service]


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Aggregator] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Aggregator] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Aggregator] done search_id=%s candidates=%d",
                search_id, len(result["candidates"]))
    return result


def _process(req: dict) -> dict:
    retriever_results = req.get("retriever_results", [])

    # CI context needed for entity-aware, fact-aware, and section-aware scoring
    ci             = req.get("ci", {})
    ci_entities    = ci.get("ner", {}).get("entities", [])
    ci_facts       = ci.get("effective_facts") or ci.get("facts", {})
    ci_relations   = ci.get("clinical_relations", [])
    ci_type        = req.get("classification", {}).get("ci_type", "")

    candidates  = _merge(retriever_results)
    _score_candidates(candidates, ci_entities, ci_facts, ci_relations, ci_type)

    return {
        **req,
        "candidates": candidates,
    }


# Retrievers whose hits carry their own precise {text,start,end} span in raw_text,
# keyed by the hit field that holds it. All feed the same "literal_matches" accumulator
# on the merged candidate — add new retrievers here as they gain exact-span extraction.
_EXACT_MATCH_FIELD_BY_RETRIEVER = {
    "literal": "literal_matches",
    "regex":   "regex_matches",
}


def _merge(retriever_results: list[dict]) -> list[dict]:
    """
    Cluster hits from all retrievers into deduplicated candidates.

    Deduplication order (most to least specific):
      1. Exact object_id match (same semantic object returned by multiple retrievers)
      2. Text similarity > 0.85 on the same page/type
         (near-duplicate object hits when a retriever does not expose object_id)
      3. Stable fallback identity built from object type + page + global position +
         normalized text; chunk_id is used only when no object-level identity is available.

    Geometry is intentionally excluded from candidate identity. It is canonical
    payload, not a retrieval/aggregation signal.
    """
    # Canonical candidates keyed only by semantic identity. Geometry/bbox is never
    # consulted for clustering.
    clusters: dict[str, dict] = {}

    for rr in retriever_results:
        retriever = rr.get("retriever", "unknown")
        for hit in rr.get("hits", []):
            matched_obj = hit.get("matched_object")
            key         = _cluster_key(hit, matched_obj, clusters)

            if key not in clusters:
                clusters[key] = {
                    "chunk_id":       hit["chunk_id"],
                    "page_start":     hit.get("page_start", 0),
                    "page_end":       hit.get("page_end",   0),
                    "sources":        [],
                    "max_score":      0.0,
                    "snippet":        hit.get("snippet", ""),
                    "matched_object": None,
                    "literal_matches": [],
                    "_best_score":    0.0,
                    "_per_scores":    {},   # retriever → best score from that retriever
                }
            entry = clusters[key]
            if retriever not in entry["sources"]:
                entry["sources"].append(retriever)

            # Track best score per retriever (for hybrid formula)
            score = hit.get("score", 0.0)
            if score > entry["_per_scores"].get(retriever, 0.0):
                entry["_per_scores"][retriever] = score

            # Propagate exact matched spans forward so context_expander/Stage 6.5 can
            # resolve the containing object without re-discovery. Not literal-only: any
            # retriever that finds a precise span in raw_text (literal, regex, future
            # additions) feeds the same accumulator via this field-name map.
            match_field = _EXACT_MATCH_FIELD_BY_RETRIEVER.get(retriever)
            if match_field and hit.get(match_field):
                existing_starts = {m["start"] for m in entry["literal_matches"]}
                for lm in hit[match_field]:
                    if lm["start"] not in existing_starts:
                        entry["literal_matches"].append(lm)
                        existing_starts.add(lm["start"])

            score = hit.get("score", 0.0)
            if score > entry["_per_scores"].get(retriever, 0.0):
                entry["_per_scores"][retriever] = score

            # matched_object: accept the first one we see (covers filter-only
            # retrievers like numeric/regex whose score is always 0.0);
            # only overwrite when a strictly higher-scored hit comes in.
            if matched_obj:
                if entry["matched_object"] is None or score > entry["_best_score"]:
                    entry["matched_object"] = matched_obj

            if score > entry["_best_score"]:
                entry["max_score"]   = score
                entry["_best_score"] = score

    candidates = list(clusters.values())
    for c in candidates:
        del c["_best_score"]
    # agg_score computed by _score_candidates (called by _process) so sorting happens there
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Identity overlap — raw evidence for the reranker to interpret
# ─────────────────────────────────────────────────────────────────────────────

def _compute_identity_overlap(ci_facts: dict, cand_facts: dict) -> dict:
    """
    Compute raw value-level overlap for drug, endpoint, and phase.

    Returns structured evidence dicts — NOT penalties.  The reranker consumes
    these to decide severity (e.g. drug_overlap=0.0 → FATAL contradiction).
    Using effective_facts on both sides ensures inherited heading/paragraph
    context (drug, endpoint) participates even when the sentence omits them.
    """

    def _olap(ci_vals: list[str], cand_vals: list[str]) -> dict:
        if not ci_vals or not cand_vals:
            return {
                "ci": ci_vals, "candidate": cand_vals,
                "matched": [], "overlap": 0.0,
            }
        ci_n   = {v.lower().strip() for v in ci_vals if v}
        cand_n = {v.lower().strip() for v in cand_vals if v}
        matched = [
            v for v in ci_n
            if any(v in cd or cd in v for cd in cand_n)
        ]
        overlap = len(matched) / max(len(ci_n), len(cand_n))
        return {
            "ci": sorted(ci_n), "candidate": sorted(cand_n),
            "matched": matched, "overlap": round(overlap, 3),
        }

    ci_drugs   = [v for v in ci_facts.get("drug",   []) if v]
    cand_drugs = [v for v in cand_facts.get("drug", []) if v]
    drug_ev    = _olap(ci_drugs, cand_drugs)

    # Drug relation via identity graph (degrades gracefully to heuristic)
    if _DRUG_IDENTITY_AVAILABLE and ci_drugs and cand_drugs:
        try:
            rel = _best_drug_relation(ci_drugs, cand_drugs)
            drug_ev["relation"] = rel.value
        except Exception:
            drug_ev["relation"] = "UNKNOWN"
    elif not ci_drugs or not cand_drugs:
        drug_ev["relation"] = "NONE"
    elif drug_ev["matched"]:
        drug_ev["relation"] = "EXACT"
    else:
        drug_ev["relation"] = "DIFFERENT"

    # Endpoint — normalise to family before overlap
    ci_eps   = [_agg_ep_family(v) for v in ci_facts.get("endpoint",   []) if v]
    cand_eps = [_agg_ep_family(v) for v in cand_facts.get("endpoint", []) if v]

    ci_phases   = [v.lower().strip() for v in ci_facts.get("phase",   []) if v]
    cand_phases = [v.lower().strip() for v in cand_facts.get("phase", []) if v]

    return {
        "drug":     drug_ev,
        "endpoint": _olap(ci_eps, cand_eps),
        "phase":    _olap(ci_phases, cand_phases),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid scoring  (#2 entity-aware  #3 section-aware  #4 relation  #5 hybrid)
# ─────────────────────────────────────────────────────────────────────────────

def _score_candidates(
    candidates:    list[dict],
    ci_entities:   list[dict],
    ci_facts:      dict,
    ci_relations:  list[dict],
    ci_type:       str,
) -> None:
    """Compute hybrid agg_score in-place for every candidate and sort descending."""
    for c in candidates:
        per_scores   = c.pop("_per_scores", {})
        matched_obj  = c.get("matched_object") or {}
        cand_entities = matched_obj.get("entities", [])

        # #5 Weighted retriever component
        # Normalise each retriever's raw score to [0, 1] using per-retriever caps
        # before applying weights.  Without this, BM25/literal TF-IDF scores
        # (range 10–200) completely overwhelm vector cosine scores (0–1).
        retriever_component = sum(
            _RETRIEVER_WEIGHTS.get(r, 0.02)
            * min(s / _RETRIEVER_SCORE_CAP.get(r, 10.0), 1.0)
            for r, s in per_scores.items()
        )

        # #2 Entity overlap bonus
        ent_overlap = _entity_overlap(ci_entities, cand_entities)

        # Fact overlap bonus — slot-aware comparison (drug+endpoint vs drug+AE).
        # effective_facts carries inherited heading/paragraph context so objects
        # that don't explicitly repeat the drug still score correctly.
        cand_facts     = matched_obj.get("facts", {})
        cand_facts_eff = matched_obj.get("effective_facts") or cand_facts
        fact_olap   = _fact_overlap(ci_facts, cand_facts_eff)

        # #4 Relation score — direct clinical_relations comparison when available,
        # entity-type-pair proxy as fallback for objects without stored relations.
        cand_relations = matched_obj.get("clinical_relations", [])
        rel_score = _relation_score(ci_entities, cand_entities, ci_relations, cand_relations)

        # #3 Section alignment multiplier
        sect_boost = _section_boost(ci_type, matched_obj.get("section_category"))

        # Study-context penalty — penalise CITED/HISTORICAL candidates for
        # current-protocol CI types (e.g. OBJECTIVE, EFFICACY, DOSING …).
        # Only applied when study_context was set during indexing (non-GENERAL).
        study_ctx    = matched_obj.get("study_context", "GENERAL")
        ctx_penalty  = _study_context_penalty(ci_type, study_ctx)

        # Granularity bonus — prefer sentence > paragraph > heading in near-ties
        gran_bonus = _GRANULARITY_BONUS.get(matched_obj.get("type", ""), 0.0)

        # Contradiction penalties — detect structural mismatches, then score.
        # Pass effective_facts as the candidate facts argument so that
        # missing_fact_slot only fires when the slot is absent even after
        # heading/paragraph context inheritance.
        # Structural contradiction penalties (no clinical-value mismatches —
        # those move to identity_overlap for the reranker to interpret).
        _contra_vr                   = _detect_contradictions(
            ci_type, ci_facts, ci_relations, matched_obj,
            cand_facts_eff, cand_relations
        )
        contradiction, contra_detail = _score_contradictions(_contra_vr)

        # Identity overlap — raw evidence (drug / endpoint / phase) for reranker
        identity_olap = _compute_identity_overlap(ci_facts, cand_facts_eff)

        # Semantic identity bonus — reward propagated drug/endpoint overlap above
        # raw word overlap; lifts semantically correct candidates above keyword hits.
        # Weighted combination (not max): candidates matching BOTH drug and endpoint
        # should rank above those matching only one.  Drug carries 70% of the signal
        # because it is more clinically discriminating in this corpus.
        _drug_olap     = identity_olap.get("drug",     {}).get("overlap", 0.0)
        _ep_olap       = identity_olap.get("endpoint", {}).get("overlap", 0.0)
        identity_bonus = _IDENTITY_BONUS_WEIGHT * (0.7 * _drug_olap + 0.3 * _ep_olap)

        # Zero-identity gating penalty — fires only when CI has extractable
        # drug/endpoint AND candidate has neither in its effective_facts.
        _ci_has_drug  = bool(ci_facts.get("drug"))
        _ci_has_ep    = bool(ci_facts.get("endpoint"))
        _cand_has_drug = bool(cand_facts_eff.get("drug"))
        _cand_has_ep   = bool(cand_facts_eff.get("endpoint"))
        zero_id_pen = (
            _ZERO_IDENTITY_PENALTY
            if (_ci_has_drug or _ci_has_ep) and not _cand_has_drug and not _cand_has_ep
            else 0.0
        )

        # Fully-unenriched penalty — NER stage failed; no entities, facts, or
        # effective_facts present in the candidate object.
        # Suppressed for NUMERIC/STATISTICAL CIs: a table cell or sentence
        # containing only "n = 8" or "p<0.0001" is correctly unenriched and
        # should not be penalised — the number is the evidence, not the entities.
        _is_numeric_ci = ci_type.upper() in _NUMERIC_CI_TYPES
        zero_enrich_pen = (
            _ZERO_ENRICHMENT_PENALTY
            if not _is_numeric_ci
            and not cand_facts_eff
            and not matched_obj.get("facts")
            and not matched_obj.get("entities")
            else 0.0
        )

        # Section drift penalty — keyword-level mismatch in heading path / section
        # (e.g. Primary Objective CI against a "Secondary Endpoints" object).
        sect_drift_pen = _section_drift_penalty(ci_type, matched_obj)

        raw = (retriever_component
               + _ENTITY_WEIGHT   * ent_overlap
               + _FACT_WEIGHT     * fact_olap
               + _RELATION_WEIGHT * rel_score
               + gran_bonus
               + contradiction          # ≤ 0
               + identity_bonus         # ≥ 0; rewards identity overlap
               + zero_id_pen            # ≤ 0; gates zero-identity candidates
               + zero_enrich_pen        # ≤ 0; penalises NER-failed objects
               + sect_drift_pen)        # ≤ 0; explicit section mismatch
        c["agg_score"]           = round(raw * sect_boost * ctx_penalty, 4)
        c["entity_overlap"]      = round(ent_overlap, 3)
        c["fact_overlap"]        = round(fact_olap, 3)
        c["relation_score"]      = round(rel_score, 3)
        c["contradiction"]       = round(contradiction, 3)
        c["section_boost"]       = round(sect_boost, 3)
        c["study_context"]       = study_ctx
        c["study_context_mult"]  = round(ctx_penalty, 3)
        c["identity_overlap"]    = identity_olap
        # Per-component breakdown — kept for diagnostic tooling (diagnose-ci flag in search_test)
        c["score_breakdown"] = {
            "vector":           round(per_scores.get("vector",   0.0), 4),
            "bm25":             round(per_scores.get("bm25",     0.0), 4),
            "literal":          round(per_scores.get("literal",  0.0), 4),
            "ontology":         round(per_scores.get("ontology", 0.0), 4),
            "fact_ret":         round(per_scores.get("fact",     0.0), 4),
            "numeric_ret":      round(per_scores.get("numeric",  0.0), 4),
            "ret_component":    round(retriever_component, 4),
            "entity_olap":      round(ent_overlap, 4),
            "fact_olap":        round(fact_olap, 4),
            "relation":         round(rel_score, 4),
            "gran":             round(gran_bonus, 4),
            "contradiction":    round(contradiction, 4),
            "contra_detail":    contra_detail,
            "identity_overlap": identity_olap,
            "identity_bonus":   round(identity_bonus, 4),
            "zero_id_pen":      round(zero_id_pen, 4),
            "zero_enrich_pen":  round(zero_enrich_pen, 4),
            "sect_drift_pen":   round(sect_drift_pen, 4),
            "raw":              round(raw, 4),
            "sect_mult":        round(sect_boost, 4),
            "ctx_mult":         round(ctx_penalty, 4),
        }

    candidates.sort(key=lambda x: x["agg_score"], reverse=True)


def _detect_contradictions(
    ci_type:        str,
    ci_facts:       dict,
    ci_relations:   list[dict],
    matched_obj:    dict,
    cand_facts:     dict,
    cand_relations: list[dict],
) -> "_ValidationResult":
    """
    Detect STRUCTURAL contradictions between the CI and one candidate.

    Each detected contradiction becomes a ValidationIssue carrying:
      • type     — what kind of structural issue it is
      • weight   — the additive penalty (always ≤ 0)
      • evidence — structured dict explaining the signal

    Clinical-value mismatches (drug / endpoint / phase) are NOT detected here.
    They are computed by _compute_identity_overlap() and surfaced as raw
    evidence on the candidate so the reranker can apply clinical reasoning.

    Returns a _ValidationResult whose .issues list is ready for _score_contradictions().
    """
    result   = _ValidationResult()
    ci_upper = (ci_type or "").upper()

    # 1. Missing required relation
    #    CI has drug→endpoint triplets; candidate has no clinical_relations at all
    #    → it makes no structured assertion, likely context or abbreviation text.
    if ci_relations and not cand_relations:
        _add_issue(result, "missing_relation",
                   {"expected_relations": len(ci_relations), "found_relations": 0})

    # 2. Statement type incompatibility
    #    GENERAL / BACKGROUND = extractor found no clinical claim in the text.
    #    For OBJECTIVE/EFFICACY CIs this is a strong signal the chunk is context.
    stmt           = (matched_obj.get("statement_type") or "").upper()
    incompat_stmts = _INCOMPATIBLE_STMT_TYPES.get(ci_upper, frozenset())
    if stmt and stmt in incompat_stmts:
        _add_issue(result, "stmt_type_mismatch",
                   {"expected": "ASSERTION", "found": stmt})

    # 3. Abbreviation / definition object
    #    Prefer the indexed object_subtype (computed at index time, reliable).
    #    Fall back to the run-time regex heuristic for objects indexed before
    #    this field was added (backward-compatible).
    if ci_upper in {"OBJECTIVE", "EFFICACY", "DOSING", "SAFETY", "PHARMACOKINETICS"}:
        obj_subtype = (matched_obj.get("object_subtype") or "").upper()
        if obj_subtype in {"ABBREVIATION_TABLE", "DEFINITION"}:
            _add_issue(result, "abbreviation_object",
                       {"object_subtype": obj_subtype})
        else:
            # Legacy heuristic for objects not yet re-indexed with object_subtype
            text      = matched_obj.get("text", "")
            n_abbrevs = len(_ABBREV_RE.findall(text))
            n_ents    = len(matched_obj.get("entities", []))
            n_words   = max(len(text.split()), 1)
            if n_abbrevs >= 3 or (n_ents / n_words > 0.30 and not cand_facts):
                _add_issue(result, "abbreviation_object",
                           {"pattern": "abbreviation_density", "abbreviations_found": n_abbrevs})

    # 4. Missing required fact slots — one issue per slot.
    #    Only fires when the CI itself has a non-empty value for the slot.
    for slot in _REQUIRED_FACT_SLOTS.get(ci_upper, []):
        if ci_facts.get(slot) and not cand_facts.get(slot):
            _add_issue(result, "missing_fact_slot", {"slot": slot})

    # 5. Incompatible section
    section = (matched_obj.get("section_category") or "").upper()
    if section and section in _INCOMPATIBLE_SECTIONS.get(ci_upper, frozenset()):
        expected_sections = sorted(_CI_TYPE_TO_SECTIONS.get(ci_upper, set()))
        _add_issue(result, "section_mismatch",
                   {"found_section": section, "expected_sections": expected_sections})

    # Note: drug / endpoint / phase value-overlap evidence is computed separately
    # by _compute_identity_overlap() — NOT penalised here.  The reranker interprets
    # those overlap scores as clinical contradiction evidence.

    return result


def _score_contradictions(vr: "_ValidationResult") -> tuple[float, list[dict]]:
    """
    Sum penalties from all ValidationIssue objects.

    Returns:
        penalty      — total negative additive adjustment (≤ 0.0)
        contra_detail — list of per-issue dicts for the score_breakdown
    """
    penalty = 0.0
    detail: list[dict] = []
    for issue in vr.issues:
        penalty += issue.weight
        detail.append({
            "type":     issue.type,
            "weight":   round(issue.weight, 4),
            "evidence": issue.evidence,
        })
    return round(penalty, 4), detail


def _facts_conflict(ci_vals: list, cand_vals: list) -> bool:
    """
    True when both sides have at least one value and no value from the CI
    side overlaps with any value on the candidate side (case-insensitive
    substring match).  Returns False when either list is empty — absence of
    a fact is handled by the missing_fact_slots rule, not here.
    """
    if not ci_vals or not cand_vals:
        return False
    ci_norm   = {v.lower().strip() for v in ci_vals   if v}
    cand_norm = {v.lower().strip() for v in cand_vals if v}
    return not any(
        ci_v in cd_v or cd_v in ci_v
        for ci_v in ci_norm
        for cd_v in cand_norm
    )


def _entity_overlap(ci_entities: list[dict], cand_entities: list[dict]) -> float:
    """
    Jaccard similarity over normalised entity texts.
    Uses the candidate entity's confidence as a soft weight:
    low-confidence entities still participate but count less.
    """
    if not ci_entities or not cand_entities:
        return 0.0
    ci_set   = {e["text"].lower() for e in ci_entities if e.get("text")}
    cand_set = {e["text"].lower() for e in cand_entities if e.get("text")}
    if not ci_set or not cand_set:
        return 0.0
    intersection = ci_set & cand_set
    union        = ci_set | cand_set
    return len(intersection) / len(union)


def _fact_overlap(ci_facts: dict, cand_facts: dict) -> float:
    """
    Slot-weighted Jaccard similarity over pre-computed facts dicts.

    Unlike _entity_overlap (which is a flat set comparison over entity text),
    this compares facts slot by slot.  A candidate that has the same drug but
    a *different* slot (adverse_event instead of endpoint) scores lower, which
    is exactly the signal we want to separate:

        CI:       {drug: teclistamab, endpoint: ORR}
        Chunk A:  {drug: teclistamab, endpoint: ORR}  → high overlap (1.0 weighted)
        Chunk B:  {drug: teclistamab, adverse_event: headache}  → partial (drug only)

    Score = sum_over_slots(weight[slot] * Jaccard(ci_values, cand_values))
            ─────────────────────────────────────────────────────────────────
            sum_over_slots(weight[slot] for all CI slots in _FACT_SLOT_BOOST)

    Returns 0.0 if either dict is empty (graceful — many objects won't have
    facts if they were indexed before enrich_object was added to the pipeline).
    """
    if not ci_facts or not cand_facts:
        return 0.0

    total_weight = 0.0
    score        = 0.0

    for slot, ci_values in ci_facts.items():
        weight = _FACT_SLOT_BOOST.get(slot)
        if weight is None or not ci_values:
            continue
        total_weight += weight
        cand_values = cand_facts.get(slot, [])
        if not cand_values:
            continue
        # Normalise endpoint values to their clinical family so that
        # "ORR" == "overall response" == "overall response (pr or better)".
        # All other slots use plain lowercase comparison.
        if slot == "endpoint":
            ci_set   = {_agg_ep_family(v) for v in ci_values if v}
            cand_set = {_agg_ep_family(v) for v in cand_values if v}
        else:
            ci_set   = {v.lower() for v in ci_values if v}
            cand_set = {v.lower() for v in cand_values if v}
        if ci_set and cand_set:
            jaccard = len(ci_set & cand_set) / len(ci_set | cand_set)
            score  += weight * jaccard

    return score / total_weight if total_weight > 0.0 else 0.0


def _relation_score(
    ci_entities:   list[dict],
    cand_entities: list[dict],
    ci_relations:  list[dict] | None = None,
    cand_relations:list[dict] | None = None,
) -> float:
    """
    Relation similarity between CI and candidate, returning 0–1.

    Strategy
    --------
    When both CI and candidate have stored `clinical_relations` (extracted by
    `enrich_object` at indexing time), use direct triplet comparison:

        {drug: Teclistamab, endpoint: ORR, relation: measured_by}  (CI)
        vs
        {drug: Teclistamab, endpoint: ORR, relation: measured_by}  (Chunk A)  → 1.0
        {drug: Teclistamab, endpoint: ORR, relation: associated_with}         → 0.8
        {drug: Teclistamab, adverse_event: Headache, relation: causes_ae}     → 0.3

    Falls back to the entity-type-pair proxy when stored relations are absent
    (older indexed objects, or objects where spaCy found no relations).
    """
    if ci_relations and cand_relations:
        return _direct_relation_score(ci_relations, cand_relations)
    return _relation_proxy_score(ci_entities, cand_entities)


def _direct_relation_score(ci_relations: list[dict], cand_relations: list[dict]) -> float:
    """
    Triplet-level comparison of extracted clinical relations.

    For each CI relation, finds the best-matching candidate relation anchored
    on the same drug (case-insensitive).  The match score is:

      drug + relation_type + other_entity  → 1.00  (full triplet match)
      drug + other_entity                  → 0.80  (same entities, different relation)
      drug + relation_type                 → 0.60  (same relation verb, different entity)
      drug only                            → 0.30  (anchor only)

    Each CI relation is weighted by its extraction confidence.
    Returns the confidence-weighted average over all CI relations.
    """
    # Build lookup: drug.lower() → list of candidate relations
    cand_by_drug: dict[str, list[dict]] = {}
    for rel in cand_relations:
        drug = rel.get("drug", "").lower()
        if drug:
            cand_by_drug.setdefault(drug, []).append(rel)

    total_weight = 0.0
    total_score  = 0.0

    for ci_rel in ci_relations:
        ci_drug    = ci_rel.get("drug", "").lower()
        ci_reltype = ci_rel.get("relation", "").lower()
        ci_conf    = float(ci_rel.get("confidence", 1.0))
        ci_slot, ci_val = _rel_other_slot(ci_rel)

        total_weight += ci_conf

        matches = cand_by_drug.get(ci_drug, [])
        if not matches:
            continue   # drug not found in candidate — contributes 0

        best = 0.0
        for c_rel in matches:
            c_reltype   = c_rel.get("relation", "").lower()
            rel_match   = bool(ci_reltype and c_reltype == ci_reltype)
            other_match = False
            if ci_slot and ci_val:
                other_match = c_rel.get(ci_slot, "").lower() == ci_val

            if rel_match and other_match:
                slot_score = 1.00
            elif other_match:
                slot_score = 0.80
            elif rel_match:
                slot_score = 0.60
            else:
                slot_score = 0.30

            if slot_score > best:
                best = slot_score

        total_score += ci_conf * best

    return total_score / total_weight if total_weight > 0.0 else 0.0


def _rel_other_slot(rel: dict) -> tuple[str, str]:
    """Return (slot_name, value.lower()) for the non-drug entity in a relation dict."""
    _SKIP = {"drug", "relation", "verb", "confidence"}
    for k, v in rel.items():
        if k not in _SKIP and isinstance(v, str) and v:
            return k, v.lower()
    return "", ""


def _relation_proxy_score(ci_entities: list[dict], cand_entities: list[dict]) -> float:
    """
    Fallback: entity-type-pair co-occurrence proxy.

    For each entity-type pair that co-occurs in the CI (e.g. MEDICATION+ADVERSE_EVENT),
    check whether the same pair also co-occurs in the candidate.  Used when stored
    clinical_relations are not available (pre-enrichment indexed objects).
    """
    if not ci_entities or not cand_entities:
        return 0.0

    def _by_type(entities: list[dict]) -> dict[str, set[str]]:
        d: dict[str, set[str]] = {}
        for e in entities:
            t = e.get("label") or e.get("sub_type") or "OTHER"
            if e.get("text"):
                d.setdefault(t, set()).add(e["text"].lower())
        return d

    ci_by_type   = _by_type(ci_entities)
    cand_by_type = _by_type(cand_entities)

    matched = 0
    checked = 0
    for type_a, type_b in _RELATION_PAIRS:
        if type_a in ci_by_type and type_b in ci_by_type:
            checked += 1
            if type_a in cand_by_type and type_b in cand_by_type:
                # At least one entity text overlaps for either type in the pair
                a_overlap = bool(ci_by_type[type_a] & cand_by_type[type_a])
                b_overlap = bool(ci_by_type[type_b] & cand_by_type[type_b])
                if a_overlap or b_overlap:
                    matched += 1

    return matched / checked if checked > 0 else 0.0


def _section_drift_penalty(ci_type: str, matched_obj: dict) -> float:
    """
    Additive penalty for confirmed section/intent drift — when the CI intent
    and the candidate heading path contradict at a structural level.

    Distinct from the section_boost multiplier (which rewards correct sections):
    this fires only on *positive evidence* of mismatch, never on absence.

    Examples of what fires:
      • CI type=OBJECTIVE, candidate heading contains "secondary" → -0.25
        (Primary Objective CI matched to a Secondary Endpoints object)
      • CI type=OBJECTIVE/EFFICACY, candidate section=ELIGIBILITY → -0.20
        (Objective CI matched to inclusion/exclusion criteria)
      • CI type=OBJECTIVE/EFFICACY, candidate section=BACKGROUND → -0.15
        (supplementary penalty on top of any _INCOMPATIBLE_SECTIONS penalty)

    Returns 0.0 when there is insufficient evidence to confirm drift.
    """
    if not ci_type or not matched_obj:
        return 0.0

    ci_upper     = ci_type.upper()
    heading_text = " ".join(matched_obj.get("heading_path") or []).lower()
    section_cat  = (matched_obj.get("section_category") or "").upper()

    # Primary/secondary objective mismatch — fires on heading keyword evidence
    if ci_upper in {"OBJECTIVE", "EFFICACY"} and "secondary" in heading_text:
        return _SECTION_DRIFT_PENALTY_PRIMARY_SECONDARY

    # Cross-domain: objective/efficacy CI against eligibility section
    if ci_upper in {"OBJECTIVE", "EFFICACY"} and section_cat in _ELIGIBILITY_SECTIONS:
        return _SECTION_DRIFT_PENALTY_DOMAIN_CROSS

    # Background/appendix penalty for assertion-type CIs
    if ci_upper in {"OBJECTIVE", "EFFICACY"} and section_cat in _BACKGROUND_SECTIONS:
        return _SECTION_DRIFT_PENALTY_BACKGROUND

    return 0.0


def _section_boost(ci_type: str | None, section_category: str | None) -> float:
    """
    Multiplicative boost for section alignment (#3).

    Returns _SECTION_BOOST_MATCH (1.25) when the candidate's section_category
    is in the expected set for this CI type, otherwise 1.0.
    """
    if not ci_type or not section_category:
        return _SECTION_BOOST_DEFAULT
    expected = _CI_TYPE_TO_SECTIONS.get(ci_type.upper(), set())
    if section_category.upper() in expected:
        return _SECTION_BOOST_MATCH
    return _SECTION_BOOST_DEFAULT


def _study_context_penalty(ci_type: str | None, study_context: str) -> float:
    """
    Multiplicative penalty when a candidate's study_context is CITED or HISTORICAL
    but the CI is asking about the current study's protocol.

    "MonumenTAL-1 demonstrated…" (study_context=CITED) should not compete
    with "The primary objective of Part 3 is to evaluate…" (study_context=CURRENT)
    for an OBJECTIVE CI.

    Returns 1.0 (no penalty) when:
      - study_context is GENERAL (field not yet populated → conservative)
      - CI type is not a current-protocol type (e.g. a PHRASE lookup)
    """
    if not ci_type or ci_type.upper() not in _CURRENT_PROTOCOL_CI_TYPES:
        return 1.0
    return _STUDY_CONTEXT_PENALTY.get(study_context, 1.0)


def _cluster_key(hit: dict, matched_obj: dict | None,
                 existing: dict[str, dict]) -> str:
    """Return a geometry-free, object-stable aggregation identity.

    Priority:
      1. exact semantic object_id
      2. same-page/type text similarity when object_id is absent
      3. deterministic object identity from type + page + global_position + text
      4. chunk_id only for genuine chunk-level hits with no usable object identity

    ``bbox``/rects/geometry are deliberately never inspected here.
    """
    if matched_obj:
        oid = str(matched_obj.get("object_id") or "").strip()
        if oid:
            return f"obj:{oid}"

    page = _hit_page(hit, matched_obj)
    object_type = str(
        (matched_obj or {}).get("type")
        or hit.get("retrieval_object_type")
        or hit.get("object_type")
        or ""
    ).strip().lower()
    snippet = _identity_text(hit, matched_obj)

    # If this is an object-level hit but the producer omitted object_id, first
    # merge only with a genuinely similar object on the same page/type.
    if snippet:
        for key, entry in existing.items():
            if _entry_page(entry) != page:
                continue
            existing_obj = entry.get("matched_object") or {}
            existing_type = str(existing_obj.get("type") or "").strip().lower()
            if object_type and existing_type and existing_type != object_type:
                continue
            existing_text = _identity_text_from_entry(entry)
            if existing_text and _text_sim(snippet, existing_text) > 0.85:
                return key

    if matched_obj:
        global_position = matched_obj.get("global_position")
        if global_position is not None:
            return f"fallback:{object_type}:{page}:pos:{global_position}"
        return f"fallback:{object_type}:{page}:text:{_normalize_identity_text(snippet)}"

    # A retriever hit without an object is genuinely chunk-level. Keep chunk_id
    # as the last-resort identity rather than merging unrelated hits from the same page.
    chunk_id = str(hit.get("chunk_id") or hit.get("id") or "").strip()
    if chunk_id:
        return f"chunk:{chunk_id}"

    return f"fallback:{object_type}:{page}:text:{_normalize_identity_text(snippet)}"


def _normalize_identity_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())[:1000]


def _identity_text(hit: dict, matched_obj: dict | None) -> str:
    return _normalize_identity_text(
        (matched_obj or {}).get("text")
        or hit.get("snippet")
        or hit.get("text")
        or ""
    )


def _identity_text_from_entry(entry: dict) -> str:
    obj = entry.get("matched_object") or {}
    return _normalize_identity_text(
        obj.get("text")
        or entry.get("snippet")
        or ""
    )


def _hit_page(hit: dict, matched_obj: dict | None) -> int:
    value = (matched_obj or {}).get("page")
    if value is None:
        value = hit.get("page_start")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _entry_page(entry: dict) -> int:
    obj = entry.get("matched_object") or {}
    value = obj.get("page")
    if value is None:
        value = entry.get("page_start")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _text_sim(a: str, b: str) -> float:
    """Fast Jaccard token similarity — no external deps."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)
