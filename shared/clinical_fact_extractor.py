"""
clinical_fact_extractor.py
──────────────────────────
Deterministic clinical fact extractor.  Zero API calls, zero LLM inference.

Responsibility split (strict):

  Apryse extraction metadata   →  section_category, heading_path
  GLiNER entities               →  clinical slot values (drug, endpoint, dose …)
  spaCy dep parser              →  which verb connects which entity pair
  Clinical Mapper (this module) →  canonical typed relation from entity-pair + verb

spaCy does NOT classify PRIMARY_OBJECTIVE vs BACKGROUND.  That comes from
section_category and heading_path, which are more reliable for protocol docs.

Produces four fields per object:

  study_context     "CURRENT" | "CITED" | "HISTORICAL" | "GENERAL"
                    Derived from section_category + heading_path first.
                    Text-level fallback only (no dep parse needed).

  statement_type    "PRIMARY_OBJECTIVE" | "SECONDARY_OBJECTIVE" | "EXPLORATORY" |
                    "SAFETY" | "EFFICACY" | "PHARMACOKINETICS" | "PHARMACODYNAMICS" |
                    "DOSING" | "POPULATION" | "PROTOCOL" | "BACKGROUND" | "GENERAL"
                    Derived from heading_path → section_category → text patterns.

  facts             Flat dict keyed by clinical slot.
                    Keys: action, drug, endpoint, dose, study_id, study_arm,
                          phase, population, adverse_event, biomarker,
                          statistical_method, response_criterion
                    Values: list[str] of entity texts from GLiNER.

  clinical_relations  list[typed slot dicts]
                    spaCy finds the connecting verb; GLiNER fills the slots.
                    Each relation has only the slots that are present, plus:
                      "relation" : canonical relation type  (e.g. "measured_by")
                      "verb"     : dep-parse verb lemma     (e.g. "evaluate")

Example
-------
Text:  "The primary objective of Part 3 is to evaluate the efficacy of
        teclistamab at the RP2D as measured by ORR."

GLiNER entities: teclistamab→MEDICATION, RP2D→DOSAGE, ORR→CLINICAL_ENDPOINT, Part 3→STUDY_ARM

Output:
  study_context      : "CURRENT"
  statement_type     : "PRIMARY_OBJECTIVE"
  facts              : {drug: [teclistamab], endpoint: [ORR], dose: [RP2D], study_arm: [Part 3]}
  clinical_relations : [
    {drug: "teclistamab", endpoint: "ORR",  relation: "measured_by",  verb: "evaluate"},
    {drug: "teclistamab", dose:     "RP2D", relation: "evaluated_at", verb: "evaluate"},
  ]
"""

from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── spaCy lazy model ──────────────────────────────────────────────────────────
_nlp = None
SPACY_MODEL = "en_core_sci_md"


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            # Only dep parser + tagger; NER disabled (GLiNER already ran).
            # Keep lemmatizer — it's a fast lookup table and is needed for
            # verb lemmatisation (token.lemma_ is empty without it).
            _nlp = spacy.load(SPACY_MODEL, disable=["ner"])
            logger.info("[FactExtractor] loaded %s", SPACY_MODEL)
        except Exception as exc:
            logger.warning("[FactExtractor] spaCy load failed (%s) — heuristic-only mode", exc)
    return _nlp


# ─────────────────────────────────────────────────────────────────────────────
# study_context: section_category + heading_path are primary signals.
# Text patterns are secondary; spaCy verb tense is tertiary.
# ─────────────────────────────────────────────────────────────────────────────

_CITED_SECTIONS   = frozenset({"BACKGROUND", "RATIONALE", "APPENDIX", "REFERENCE", "ADMINISTRATIVE"})
_CURRENT_SECTIONS = frozenset({"OBJECTIVES", "ENDPOINTS", "DESIGN", "SYNOPSIS",
                                "TREATMENT", "PROCEDURES", "ELIGIBILITY", "STATISTICS", "DOSING"})

_THIS_STUDY_RE = re.compile(
    r"\b(this\s+(?:study|trial|protocol|part)"
    r"|the\s+(?:study|trial)"
    r"|we\s+(?:will|aim|seek|intend|plan)"
    r"|the\s+(?:primary|secondary|exploratory)\s+(?:objective|endpoint))\b",
    re.I,
)
_HISTORICAL_RE = re.compile(
    r"\b(previous(?:ly)?|prior\s+(?:study|studies|data)"
    r"|preclinical|in\s+(?:vitro|vivo)"
    r"|published\s+(?:data|study|studies)|earlier\s+(?:study|data))\b",
    re.I,
)
# Linguistic citation patterns — text-level CITED signal (FIX-10).
# Used as a secondary check inside _classify_context_metadata when section_cat
# is ambiguous. Matches verb phrases that indicate a prior-study citation rather
# than a current-protocol assertion.
_CITED_LINGUISTIC_RE = re.compile(
    r"\b("
    r"based\s+on\s+(?:phase|data|results?|findings?|the\s+\w+\s+data)"
    r"|(?:data|results?|findings?)\s+(?:from|of)\s+(?:phase|study|studies|the|a)"
    r"|(?:observed|reported|demonstrated|shown|noted)\s+in\s+(?:phase|prior|previous)"
    r"|(?:as\s+)?(?:previously|previously\s+reported|reported\s+previously)"
    r"|in\s+(?:the\s+)?(?:phase\s+[0-9]|prior|previous)\s+(?:study|data|cohort|arm|part)"
    r")\b",
    re.I,
)

# POS tags for verb tense (used only as tertiary signal)
_PAST_TAGS    = frozenset({"VBD", "VBN"})
_MODAL_TAGS   = frozenset({"MD"})
_PRESENT_TAGS = frozenset({"VBZ", "VBP"})

# ─────────────────────────────────────────────────────────────────────────────
# statement_type: heading_path → section_category → text patterns
# ─────────────────────────────────────────────────────────────────────────────

_HEADING_STMT_RULES: list[tuple[str, re.Pattern]] = [
    ("PRIMARY_OBJECTIVE",   re.compile(r"\bprimary\s+objective\b",   re.I)),
    ("SECONDARY_OBJECTIVE", re.compile(r"\bsecondary\s+objective\b", re.I)),
    ("EXPLORATORY",         re.compile(r"\bexploratory\b|\btertiary\b", re.I)),
    ("SAFETY",              re.compile(r"\bsafety\b|\badverse\s+event", re.I)),
    ("PHARMACOKINETICS",    re.compile(r"\bpharmacokinetic|\bPK\b",  re.I)),
    ("PHARMACODYNAMICS",    re.compile(r"\bpharmacodynamic|\bPD\b",  re.I)),
]

_SECTION_TO_STMT: dict[str, str] = {
    "OBJECTIVES": "GENERAL_OBJECTIVE", "ENDPOINTS": "GENERAL_OBJECTIVE",
    "SAFETY": "SAFETY",
    "PK": "PHARMACOKINETICS", "PHARMACOKINETICS": "PHARMACOKINETICS",
    "PHARMACODYNAMICS": "PHARMACODYNAMICS", "PK_PD": "PHARMACOKINETICS",
    "DOSING": "DOSING", "TREATMENT": "DOSING",
    "EFFICACY": "EFFICACY", "RESULTS": "EFFICACY",
    "ELIGIBILITY": "POPULATION",
    "BACKGROUND": "BACKGROUND", "RATIONALE": "BACKGROUND",
    "DESIGN": "PROTOCOL", "SYNOPSIS": "PROTOCOL",
    "STATISTICS": "PROTOCOL", "PROCEDURES": "PROTOCOL",
    "BIOMARKER": "EFFICACY",
}

# ─────────────────────────────────────────────────────────────────────────────
# facts: GLiNER label → slot name
# ─────────────────────────────────────────────────────────────────────────────

_LABEL_TO_SLOT: dict[str, str] = {
    "MEDICATION":         "drug",
    "CLINICAL_ENDPOINT":  "endpoint",
    "DOSAGE":             "dose",
    "STUDY_ARM":          "study_arm",
    "PROTOCOL_ID":        "study_id",
    "STUDY_POPULATION":   "population",
    "ADVERSE_EVENT":      "adverse_event",
    "BIOMARKER":          "biomarker",
    "STATISTICAL_METHOD": "statistical_method",
    "TREATMENT_PHASE":    "phase",
    "CLINICAL_RESPONSE":  "response_criterion",
    # Fixed by NER post-processing but mapped here too for safety:
    "TREATMENT_NAME":     "treatment_regimen",  # regimen combos (DPd, Tal-DP, R-CHOP)
    "QUESTIONNAIRE":      "assessment",          # PRO instruments (HRQoL, EQ-5D, MRU)
}

# Sample-size statistical notation: "n = 8", "N=8", "(n=8)".
# "n" / "N" here is a statistical variable, NOT a clinical population entity.
# Used in enrich_object() to suppress the spurious population fact and
# back-fill facts["sample_size"] instead.
_SAMPLE_N_RE = re.compile(r'\b[Nn]\s*=\s*\d', re.ASCII)

# Category names (lowercased) that unambiguously indicate a sample-size CI.
_SAMPLE_SIZE_CATEGORY_NAMES: frozenset[str] = frozenset({"sample size", "sample_size"})

# FIX-3: Closed ontology of valid IMWG response criteria.
# Only CLINICAL_RESPONSE entities whose canonical form appears in this set
# are promoted to facts["response_criterion"].  GLiNER guesses such as
# "plateau", "trend", "improvement", "increase" are filtered out.
_IMWG_RESPONSE_CRITERIA: frozenset[str] = frozenset({
    # Full-text canonicals (from clinical_dict.py)
    "sCR", "CR", "VGPR", "PR", "MR", "SD", "PD", "MRD-Negative",
    # Case variants / raw text that GLiNER or spaCy may emit
    "scr", "cr", "vgpr", "pr", "mr", "sd", "pd",
    "mrd-negative", "mrd negative",
    "stringent complete response", "complete response",
    "partial response", "very good partial response",
    "minor response", "stable disease", "progressive disease",
})

# Dose-finding concepts that GLiNER occasionally mislabels as BIOMARKER.
# RP2D, MTD, MAD, MFD, OBD, PAD are dosing anchors — they belong in
# facts["dose"] so the regimen comparator (not the biomarker comparator)
# handles them.  Checked post-slot-fill in enrich_object().
_DOSE_CONCEPT_RE = re.compile(
    r'\b(RP2D|MTD|MAD|MFD|OBD|PAD'
    r'|recommended\s+phase\s*2\s+dose'
    r'|maximum\s+tolerated\s+dose'
    r'|maximum\s+administered\s+dose'
    r'|maximum\s+feasible\s+dose)\b',
    re.I,
)

# Generic/meta phrases GLiNER sometimes mislabels CLINICAL_ENDPOINT that are
# NOT actual named endpoints (ORR, PFS, OS, DOR, …) — they describe structure
# ("clinical criteria", "primary endpoint") rather than a measurable outcome.
# Left unfiltered, these give a CI a spurious endpoint identity that then
# fires the aggregator's zero-identity gate against every real candidate.
_GENERIC_ENDPOINT_PHRASES: frozenset[str] = frozenset({
    "clinical criteria", "criteria", "endpoint", "endpoints",
    "objective", "objectives", "primary endpoint", "secondary endpoint",
    "exploratory endpoint", "response criteria", "eligibility criteria",
    "inclusion criteria", "exclusion criteria",
})

# Clinical endpoint abbreviations that GLiNER occasionally mislabels as BIOMARKER.
# ORR, PFS, OS, DOR etc. are endpoints — they belong in facts["endpoint"] so
# the endpoint comparator handles them.  Checked post-slot-fill in enrich_object().
# Stored all-uppercase; membership tests use value.strip().upper().
_ENDPOINT_ABBREVS_UPPER: frozenset[str] = frozenset({
    "PFS", "OS", "ORR", "DOR", "PFS2", "EFS", "DFS", "RFS", "TTP",
    "TTR", "TTNT", "MFS", "TFS", "DCR", "CBR", "BICR", "IRC",
    "AUC", "CMAX", "TMAX", "T½", "CL", "MRD",
})

# ─────────────────────────────────────────────────────────────────────────────
# clinical_relations: typed slot-filled relations
#
# spaCy's job: find which verb connects which entity pair.
# GLiNER's job: supply the entity types (already done upstream).
# Clinical Mapper (below): convert (entity_type_a, verb, entity_type_b)
#                          into a canonical relation name + filled slots.
#
# Entity-type priority determines canonical ordering (lower = higher priority).
# The pair lookup is always done in canonical order; the mapper resolves
# which entity fills which slot (e.g. drug vs endpoint).
# ─────────────────────────────────────────────────────────────────────────────

_SLOT_PRIORITY: dict[str, int] = {
    "MEDICATION": 0, "TREATMENT_NAME": 0,
    "CLINICAL_ENDPOINT": 1,
    "DOSAGE": 2,
    "ADVERSE_EVENT": 3,
    "STUDY_POPULATION": 4,
    "STUDY_ARM": 5,
    "PROTOCOL_ID": 6,
    "CLINICAL_RESPONSE": 7,
    "BIOMARKER": 8,
    "TREATMENT_PHASE": 9,
    "QUESTIONNAIRE": 10,
}

# Each verb maps to a semantic intent category.
# The relation map uses these categories — smaller and more maintainable than per-verb patterns.
_VERB_INTENT: dict[str, str] = {
    # MEASURE — design-intent verbs
    "evaluate":     "MEASURE",
    "assess":       "MEASURE",
    "measure":      "MEASURE",
    "quantify":     "MEASURE",
    "determine":    "MEASURE",
    "estimate":     "MEASURE",
    "characterize": "MEASURE",
    "validate":     "MEASURE",
    "examine":      "MEASURE",
    "investigate":  "MEASURE",
    "explore":      "MEASURE",
    "identify":     "MEASURE",
    "describe":     "MEASURE",
    "define":       "MEASURE",
    "monitor":      "MEASURE",
    "establish":    "MEASURE",
    # COMPARE — head-to-head evaluation
    "compare":      "COMPARE",
    # OBSERVE — cited-evidence verbs
    "demonstrate":  "OBSERVE",
    "show":         "OBSERVE",
    "report":       "OBSERVE",
    "find":         "OBSERVE",
    "reveal":       "OBSERVE",
    "indicate":     "OBSERVE",
    "suggest":      "OBSERVE",
    "support":      "OBSERVE",
    "observe":      "OBSERVE",
    "confirm":      "OBSERVE",
    # ADMINISTER — dosing / administration
    "administer":   "ADMINISTER",
    "receive":      "ADMINISTER",
    "give":         "ADMINISTER",
    "dose":         "ADMINISTER",
    "treat":        "ADMINISTER",
    # COMBINE — combination therapy
    "combine":      "COMBINE",
    "add":          "COMBINE",
}

# Derived set of all recognised clinical verbs (used for fast membership tests).
_CLINICAL_VERBS = frozenset(_VERB_INTENT)

# Relation map: (canonical_type_a, canonical_type_b) → {intent_category → relation_name}
# Keys are always in canonical pair order (lower _SLOT_PRIORITY value first).
# "_default" catches any intent not explicitly listed.
_RELATION_MAP: dict[tuple[str, str], dict[str, str]] = {
    ("MEDICATION", "CLINICAL_ENDPOINT"): {
        "MEASURE":    "measured_by",
        "COMPARE":    "compared_against",
        "OBSERVE":    "demonstrated",
        "_default":   "associated_with",
    },
    ("MEDICATION", "DOSAGE"): {
        "ADMINISTER": "administered_at",
        "MEASURE":    "evaluated_at",
        "_default":   "at_dose",
    },
    ("MEDICATION", "ADVERSE_EVENT"): {
        "OBSERVE":    "causes_ae",
        "MEASURE":    "monitored_for",
        "_default":   "related_to_ae",
    },
    ("MEDICATION", "STUDY_POPULATION"): {
        "MEASURE":    "evaluated_in",
        "ADMINISTER": "used_in",
        "_default":   "used_in",
    },
    ("MEDICATION", "MEDICATION"): {
        "COMBINE":    "combined_with",
        "COMPARE":    "compared_with",
        "_default":   "co_administered",
    },
    ("MEDICATION", "BIOMARKER"): {
        "MEASURE":    "evaluated_by_biomarker",
        "_default":   "biomarker_associated",
    },
    ("CLINICAL_ENDPOINT", "CLINICAL_RESPONSE"): {
        "_default":   "uses_criterion",
    },
    ("MEDICATION", "STUDY_ARM"): {          # canonical: MEDICATION(0) < STUDY_ARM(5)
        "ADMINISTER": "arm_receives",
        "_default":   "arm_drug",
    },
    ("MEDICATION", "TREATMENT_PHASE"): {
        "_default":   "in_phase",
    },
    ("MEDICATION", "PROTOCOL_ID"): {
        "OBSERVE":    "study_drug",
        "MEASURE":    "study_drug",
        "_default":   "study_drug",
    },
}


# Allowed entity-type pairs — derived from _RELATION_MAP keys.
# Used as an early whitelist before any dep-tree computation.
_VALID_PAIRS: frozenset = frozenset(_RELATION_MAP.keys())


def _canonical_pair(type_a: str, type_b: str) -> tuple[str, str]:
    """Return entity-type pair in canonical order (lower priority number first)."""
    pa = _SLOT_PRIORITY.get(type_a, 99)
    pb = _SLOT_PRIORITY.get(type_b, 99)
    return (type_a, type_b) if pa <= pb else (type_b, type_a)


def _lookup_relation(type_a: str, type_b: str, verb: str) -> Optional[str]:
    """Return canonical relation name for this entity-type pair + verb intent, or None."""
    key      = _canonical_pair(type_a, type_b)
    verb_map = _RELATION_MAP.get(key)
    if verb_map is None:
        return None
    intent = _VERB_INTENT.get(verb, verb)    # normalise verb → intent category
    return verb_map.get(intent, verb_map.get("_default"))


# ─────────────────────────────────────────────────────────────────────────────
# Clinical signature — canonical fingerprint for de-duplication and matching
# ─────────────────────────────────────────────────────────────────────────────

def _build_clinical_signature(
    statement_type:      str,
    study_context:       str,
    modality:            str,
    treatment_identity:  dict,
    endpoint_identity:   dict,
    population_identity: dict,
    study_hierarchy:     dict,
) -> dict:
    """
    Build a compact, canonical fingerprint from the highest-quality identity fields.

    Priority: Knowledge Layer identity objects > raw facts > entity labels.
    All values are lowercased for stable comparison.  Empty fields are omitted.

    Used by the verifier and diagnostic tooling to quickly identify whether two
    objects describe the same clinical assertion (same drug + endpoint + population
    + modality + context).
    """
    sig: dict = {}

    # Structural intent
    if statement_type and statement_type != "GENERAL":
        sig["intent"] = statement_type.upper()
    if study_context and study_context != "GENERAL":
        sig["context"] = study_context.upper()
    if modality and modality != "GENERAL":
        sig["modality"] = modality.upper()

    # Drug / regimen (treatment_identity is highest fidelity)
    drug = (
        treatment_identity.get("primary_drug")
        or (treatment_identity.get("companion_drugs") or [None])[0]
    )
    if drug:
        sig["drug"] = drug.lower().strip()
    if treatment_identity.get("regimen"):
        sig["regimen"] = treatment_identity["regimen"].lower().strip()
    if treatment_identity.get("line_of_therapy"):
        sig["line_of_therapy"] = treatment_identity["line_of_therapy"].lower().strip()

    # Endpoint
    ep = endpoint_identity.get("endpoint")
    if ep:
        sig["endpoint"] = ep.lower().strip()
    if endpoint_identity.get("type"):
        sig["endpoint_type"] = endpoint_identity["type"].upper()

    # Population
    disease = population_identity.get("disease")
    if disease:
        sig["population"] = disease.lower().strip()
    prior = population_identity.get("prior_lines")
    if prior is not None:
        sig["prior_lines"] = prior

    # Study phase from hierarchy
    phase = study_hierarchy.get("phase")
    if phase:
        sig["phase"] = phase.lower().strip()

    return sig


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def enrich_object(
    text:             str,
    entities:         list[dict],
    section_category: str  = "",
    heading_path:     list = None,
) -> dict:
    """
    Return enrichment dict for one semantic object (paragraph, sentence, heading).

    Always safe — gracefully degrades to metadata-only enrichment if spaCy is
    unavailable; never raises.

    Parameters
    ----------
    text             : raw text of the object
    entities         : GLiNER/dict entities already assigned to this object
                       ({text, label, object_start, object_end, …})
    section_category : canonical section category from extraction
    heading_path     : breadcrumb list from most-general to most-specific
    """
    heading_path = heading_path or []
    # Normalize: objects store heading_path as a " > "-joined string at index time;
    # convert to list so _classify_statement and _extract_study_hierarchy work correctly.
    if isinstance(heading_path, str):
        heading_path = [h.strip() for h in heading_path.split(" > ") if h.strip()]
    cat = (section_category or "").upper()

    # 1. Entity facts — GLiNER labels → typed slots (no model needed)
    facts: dict[str, list[str]] = {}
    for ent in entities:
        slot = _LABEL_TO_SLOT.get(ent.get("label", ""))
        if slot and ent.get("text"):
            # FIX-1: Route-of-administration tokens (SC, IV, PO, IM) carry
            # sub_type="ROUTE" from the clinical dictionary.  They must not
            # pollute facts["drug"], companion_drugs, or the drug comparator.
            if ent.get("label") == "MEDICATION" and ent.get("sub_type") == "ROUTE":
                continue
            # FIX-4: Generic/meta CLINICAL_ENDPOINT phrases ("clinical criteria",
            # "primary endpoint") are structural, not a named endpoint — block them
            # so they don't give the CI a fake endpoint identity.
            if ent.get("label") == "CLINICAL_ENDPOINT" and ent["text"].strip().lower() in _GENERIC_ENDPOINT_PHRASES:
                continue
            # FIX-3: CLINICAL_RESPONSE entities are only valid response_criterion
            # facts when their canonical form is a recognised IMWG criterion.
            # GLiNER guesses ("plateau", "trend", "improvement") are blocked here.
            if ent.get("label") == "CLINICAL_RESPONSE":
                canon = (ent.get("canonical") or ent.get("text", "")).strip()
                if canon not in _IMWG_RESPONSE_CRITERIA:
                    continue
            facts.setdefault(slot, [])
            if ent["text"] not in facts[slot]:
                facts[slot].append(ent["text"])

    # 1b. Reclassify dose-finding terms mislabelled BIOMARKER by NER.
    # RP2D, MTD etc. are dosing anchors and belong in facts["dose"] so the
    # regimen comparator handles them rather than the biomarker comparator.
    _bm = facts.get("biomarker")
    if _bm:
        _to_dose = [v for v in _bm if _DOSE_CONCEPT_RE.search(v)]
        if _to_dose:
            _remaining_bm = [v for v in _bm if v not in _to_dose]
            if _remaining_bm:
                facts["biomarker"] = _remaining_bm
            else:
                del facts["biomarker"]
            _dose_list = facts.setdefault("dose", [])
            for _v in _to_dose:
                if _v not in _dose_list:
                    _dose_list.append(_v)

    # 1b2. Reclassify endpoint abbreviations mislabelled BIOMARKER by NER.
    # ORR, PFS, OS, DOR etc. are clinical endpoints, not biomarkers.
    # Move them from facts["biomarker"] to facts["endpoint"] so the endpoint
    # comparator handles them and they no longer trigger spurious biomarker
    # conflicts.  Runs after step 1b so dose-finding terms are already stripped.
    _bm = facts.get("biomarker")
    if _bm:
        _to_ep = [v for v in _bm if v.strip().upper() in _ENDPOINT_ABBREVS_UPPER]
        if _to_ep:
            _remaining_bm = [v for v in _bm if v not in _to_ep]
            if _remaining_bm:
                facts["biomarker"] = _remaining_bm
            else:
                del facts["biomarker"]
            _ep_list = facts.setdefault("endpoint", [])
            for _v in _to_ep:
                if _v not in _ep_list:
                    _ep_list.append(_v)

    # 1c. Sample-size statistical notation — "n = X" / "N = X".
    # "n" extracted by NER as STUDY_POPULATION is statistical notation, not a
    # clinical population.  Triggered by:
    #   • section_category (CI category) == "Sample Size"  (from JSON metadata)
    #   • OR text contains a bare n = <digits> pattern (universal signal)
    # When triggered: strip bare "n"/"N" from facts["population"] and
    # back-fill facts["sample_size"] from the parsed statistical_identity.
    _is_sample_size_ctx = (
        (section_category or "").lower() in _SAMPLE_SIZE_CATEGORY_NAMES
        or bool(_SAMPLE_N_RE.search(text or ""))
    )
    if _is_sample_size_ctx and "population" in facts:
        _clean_pop = [v for v in facts["population"] if v.lower() not in {"n"}]
        if _clean_pop:
            facts["population"] = _clean_pop
        else:
            del facts["population"]

    # 2. statement_type — heading_path is primary, section_category is fallback
    statement_type = _classify_statement(heading_path, text, cat)

    # 3. study_context — section_category is primary, text patterns secondary,
    #    spaCy verb tense tertiary (only adds value for ambiguous "OTHER" sections)
    study_context    = _classify_context_metadata(cat, text)
    clinical_relations: list[dict] = []

    if study_context == "GENERAL":
        # Ambiguous section — use spaCy to refine context AND extract relations
        nlp = _get_nlp()
        if nlp is not None and text and 10 <= len(text) <= 2_000:
            try:
                doc = nlp(text)
                study_context      = _refine_context_spacy(doc, entities, cat)
                clinical_relations = _extract_clinical_relations(doc, entities)
                # action verbs into facts["action"]
                verbs = _extract_action_verbs(doc)
                if verbs:
                    facts["action"] = verbs
            except Exception as exc:
                logger.debug("[FactExtractor] spaCy error: %s", exc)
    else:
        # Context already known from metadata — still extract relations if possible
        nlp = _get_nlp()
        if nlp is not None and text and 10 <= len(text) <= 2_000:
            try:
                doc = nlp(text)
                clinical_relations = _extract_clinical_relations(doc, entities)
                verbs = _extract_action_verbs(doc)
                if verbs:
                    facts["action"] = verbs
            except Exception as exc:
                logger.debug("[FactExtractor] spaCy error: %s", exc)

    # Compute statistical_identity before the return so we can back-fill
    # facts["sample_size"] when a sample-size context was detected (step 1c).
    si = _extract_statistical_identity(text)
    if _is_sample_size_ctx and si.get("type") == "sample_size" and "sample_size" in si:
        _n_str  = str(si["sample_size"])
        _ss_lst = facts.setdefault("sample_size", [])
        if _n_str not in _ss_lst:
            _ss_lst.append(_n_str)

    return {
        "study_context":        study_context,
        "statement_type":       statement_type,
        "modality":             _classify_modality(text, statement_type),
        "object_subtype":       _classify_object_subtype(text, entities, facts),
        "facts":                facts,
        "negated_slots":        _detect_negated_slots(text, entities),
        "treatment_identity":   _extract_treatment_identity(text, entities, facts),
        "endpoint_identity":    _extract_endpoint_identity(text, entities, facts),
        "population_identity":  _extract_population_identity(text, entities, facts),
        "temporal_context":     _extract_temporal_context(text),
        "statistical_identity": si,
        "clinical_relations":   clinical_relations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# object_subtype: semantic kind of object beyond statement_type
# ─────────────────────────────────────────────────────────────────────────────
# Assigned once at index time; aggregator uses it to pre-filter structural
# false positives (e.g. abbreviation tables returned for OBJECTIVE CIs).

_ABBREV_DEF_RE = re.compile(r'\b[A-Z]{2,8}\s*[=:]\s*\w')
_DEFINITION_RE = re.compile(
    r'\b(?:is defined as|means|refers to|is used to denote|is an abbreviation for)\b',
    re.I,
)
_ELIGIBILITY_RE = re.compile(
    r'\b(?:eligible|eligibility|inclusion|exclusion|must have|must not|criterion|criteria)\b',
    re.I,
)
_SCHEDULE_RE = re.compile(
    r'\b(?:visit|cycle|day\s+\d|week\s+\d|timepoint|schedule)\b',
    re.I,
)


def _classify_object_subtype(
    text: str,
    entities: list[dict],
    facts: dict,
) -> str:
    """
    Classify the semantic kind of this object beyond statement_type.

    Returns one of:
        ABBREVIATION_TABLE  — ≥3 “ABBREV=definition” patterns in the text
        DEFINITION          — single term definition sentence
        ELIGIBILITY_CRITERION — inclusion/exclusion criterion
        SCHEDULE_MATRIX     — visit/cycle/day schedule text
        GENERAL             — default

    Called once per object at index time.  The aggregator reads the stored
    value at search time to reliably detect glossary-type objects without
    re-running regex on every candidate.
    """
    n_abbrev = len(_ABBREV_DEF_RE.findall(text))
    if n_abbrev >= 3:
        return "ABBREVIATION_TABLE"
    if _DEFINITION_RE.search(text):
        return "DEFINITION"
    if _ELIGIBILITY_RE.search(text):
        return "ELIGIBILITY_CRITERION"
    if _SCHEDULE_RE.search(text):
        return "SCHEDULE_MATRIX"
    return "GENERAL"


# ─────────────────────────────────────────────────────────────────────────────
# Negation detection  (negated_slots)
# ─────────────────────────────────────────────────────────────────────────────
# Proximity-window approach: if a negation cue appears in the NEGATION_WINDOW
# characters immediately before an entity, that slot is flagged as negated.
# Used by the reranker to suppress missing_fact_slot penalties on CIs that
# explicitly exclude a slot (e.g. "must not have received prior BCMA therapy").

_NEGATION_WINDOW = 80   # characters to look back before each entity

_NEGATION_CUE_RE = re.compile(
    r'\b(?:no|not|never|without|exclud(?:e|ed|ing|es)|'
    r'must\s+not|should\s+not|cannot|can\s+not|'
    r'ineligible|contraindicated|'
    r'prior\s+(?:to|therapy|treatment|exposure|line)|'
    r'treatment[-\s]na[i\xef]ve|na[i\xef]ve\s+(?:to|patient)|'
    r'free\s+(?:from|of)|absence\s+of|refractory\s+to)\b',
    re.I,
)


def _detect_negated_slots(text: str, entities: list[dict]) -> list[str]:
    """
    Return slot names whose entity appears within a negation window.

    Checks the NEGATION_WINDOW characters immediately before each entity
    for a negation cue.  The returned list is stored as ``negated_slots``
    on the indexed object and used by the reranker to avoid false
    contradictions on eligibility and exclusion criteria.
    """
    negated: list[str] = []
    for ent in entities:
        slot = _LABEL_TO_SLOT.get(ent.get("label", ""))
        if not slot or slot in negated:
            continue
        start = ent.get("object_start", ent.get("start", 0))
        window_start = max(0, start - _NEGATION_WINDOW)
        preceding = text[window_start:start]
        if _NEGATION_CUE_RE.search(preceding):
            negated.append(slot)
    return negated


# ─────────────────────────────────────────────────────────────────────────────
# Treatment Identity  (treatment_identity)
# ─────────────────────────────────────────────────────────────────────────────
# Structured combination-therapy and line-of-therapy extraction.
# Deterministic pattern lookup — no inference, no ML.
#
# Covers the common shorthand regimen abbreviations used in haematology /
# oncology protocols.  Expand _KNOWN_REGIMENS as new protocols are added.

_KNOWN_REGIMENS: list[tuple] = [
    # Bispecific + combination (myeloma)
    (re.compile(r'\bTal[-\s]?DP\b',    re.I), "Tal-DP"),
    (re.compile(r'\bTecVRd\b',         re.I), "TecVRd"),
    (re.compile(r'\bIsa[-\s]?Kd\b',    re.I), "Isa-Kd"),
    (re.compile(r'\bIsa[-\s]?Rd\b',    re.I), "Isa-Rd"),
    (re.compile(r'\bDara[-\s]?VRd\b',  re.I), "Dara-VRd"),
    (re.compile(r'\bDara[-\s]?Rd\b',   re.I), "Dara-Rd"),
    (re.compile(r'\bDVd\b',            re.I), "DVd"),
    (re.compile(r'\bDPd\b',            re.I), "DPd"),
    (re.compile(r'\bDRd\b',            re.I), "DRd"),
    # PI + IMiD (myeloma)
    (re.compile(r'\bKRd\b',            re.I), "KRd"),
    (re.compile(r'\bVRd\b',            re.I), "VRd"),
    (re.compile(r'\bVCd\b',            re.I), "VCd"),
    (re.compile(r'\bVMP\b',            re.I), "VMP"),
    (re.compile(r'\bMPT\b',            re.I), "MPT"),
    (re.compile(r'\bPd\b',             re.I), "Pd"),
    (re.compile(r'\bKd\b',             re.I), "Kd"),
    (re.compile(r'\bRd\b',             re.I), "Rd"),
    # Lymphoma / solid tumour
    (re.compile(r'\bR[-\s]?CHOP\b',    re.I), "R-CHOP"),
    (re.compile(r'\bR[-\s]?CVP\b',     re.I), "R-CVP"),
    (re.compile(r'\bR[-\s]?EPOCH\b',   re.I), "R-EPOCH"),
    (re.compile(r'\bCHOP\b',           re.I), "CHOP"),
    (re.compile(r'\bmonotherapy\b',    re.I), "monotherapy"),
]

# Line-of-therapy normalisation: match → canonical label
_LOT_NORMALIZER: list[tuple] = [
    (re.compile(
        r'\b(1L|first[-\s]?line|1st[-\s]?line|frontline|front[-\s]?line|'
        r'newly\s+diagnosed|treatment[-\s]?na[i\xef]ve|na[i\xef]ve)\b', re.I), "1L"),
    (re.compile(
        r'\b(2L|second[-\s]?line|2nd[-\s]?line)\b', re.I), "2L"),
    (re.compile(
        r'\b(3L|third[-\s]?line|3rd[-\s]?line)\b', re.I), "3L"),
    (re.compile(
        r'\bRRMM\b|\bR[/\s]R(?:\s+MM)?\b|'
        r'relapsed[\s\-]and[\s\-]refractory|relapsed[/]refractory', re.I), "R/R"),
    (re.compile(
        r'[\u2265>=]\s*(\d+)\s+(?:prior\s+)?lines?', re.I), None),   # dynamic: "≥3L"
]

_THERAPY_PHASE_RE = re.compile(
    r'\b(induction|maintenance|consolidation|bridg(?:e|ing)|salvage|conditioning)\b',
    re.I,
)


def _extract_treatment_identity(
    text:     str,
    entities: list[dict],
    facts:    dict,
) -> dict:
    """
    Structured treatment identity: regimen, drug decomposition, line, phase.

    Returns {primary_drug, companion_drugs, regimen, line_of_therapy, therapy_phase}.
    All values may be None / [].
    """
    drugs           = list(facts.get("drug", []))
    regimen_facts   = list(facts.get("treatment_regimen", []))

    # Prefer a regimen fact (TREATMENT_NAME entity); fall back to text pattern.
    regimen: Optional[str] = regimen_facts[0] if regimen_facts else None
    if regimen is None:
        for pat, name in _KNOWN_REGIMENS:
            if pat.search(text):
                regimen = name
                break

    primary_drug    = drugs[0] if drugs else None
    companion_drugs = drugs[1:] if len(drugs) > 1 else []

    # Line of therapy
    line_of_therapy: Optional[str] = None
    for pat, normalized in _LOT_NORMALIZER:
        m = pat.search(text)
        if m:
            if normalized is None:
                # Dynamic case: "≥3L" / "≥ 3 prior lines"
                try:
                    line_of_therapy = f">={m.group(1)}L"
                except IndexError:
                    line_of_therapy = m.group(0).strip()
            else:
                line_of_therapy = normalized
            break

    # Therapy phase
    therapy_phase: Optional[str] = None
    phase_m = _THERAPY_PHASE_RE.search(text)
    if phase_m:
        therapy_phase = phase_m.group(1).capitalize()

    # RP2D / Recommended Phase 2 Dose — dose-finding concept, not a biomarker.
    # Detect it here so it's available as treatment_identity.recommended_phase2_dose
    # and comparators can use it for regimen/dose matching.
    rp2d_m = re.search(r'\bRP2D(?:s)?\b', text, re.I)
    recommended_phase2_dose: Optional[str] = rp2d_m.group(0) if rp2d_m else None

    return {
        "primary_drug":            primary_drug,
        "companion_drugs":         companion_drugs,
        "regimen":                 regimen,
        "line_of_therapy":         line_of_therapy,
        "therapy_phase":           therapy_phase,
        "recommended_phase2_dose": recommended_phase2_dose,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Context  (temporal_context)
# ─────────────────────────────────────────────────────────────────────────────
# Extracts cycle / day / week anchors and named timepoints.
# Protocols express timing in many forms: "Cycle 1 Day 1", "C1D1",
# "within 28 days", "at baseline", "prior to first dose".

_CYCLE_VISIT_RE = re.compile(
    r'\b(?:'
    r'C(?:ycle)?\s*(\d+)\s*D(?:ay)?\s*(\d+)'   # C1D1 / Cycle 1 Day 1
    r'|[Cc]ycle\s+(\d+)'                         # Cycle 3
    r'|[Dd]ay\s+(\d+)'                           # Day 15
    r'|[Ww]eek\s+(\d+)'                          # Week 12
    r')\b',
)

_WINDOW_RE = re.compile(
    r'\b(?:within|no\s+more\s+than|no\s+later\s+than)\s+(\d+)\s+(days?|weeks?|months?|hours?)\b',
    re.I,
)

_TEMPORAL_ANCHOR_RE = re.compile(
    r'\b(before|after|during|within|prior\s+to|following|upon|'
    r'at\s+(?:baseline|screening|randomization)|from\s+(?:baseline|start))\b',
    re.I,
)

_TIMEPOINT_RE = re.compile(
    r'\b(baseline|screening|randomization|'
    r'end\s+of\s+(?:treatment|study|induction|consolidation)|'
    r'(?:first|last)\s+dose|best\s+response|'
    r'progression|follow[-\s]up)\b',
    re.I,
)


def _extract_temporal_context(text: str) -> dict:
    """
    Extract temporal markers: cycle, day, week anchors and named timepoints.

    Returns {anchor, cycle, day, week, window, timepoint}.
    All values may be None.
    """
    anchor: Optional[str] = None
    anchor_m = _TEMPORAL_ANCHOR_RE.search(text)
    if anchor_m:
        anchor = anchor_m.group(1).lower()

    cycle: Optional[str] = None
    day:   Optional[str] = None
    week:  Optional[str] = None

    cv_m = _CYCLE_VISIT_RE.search(text)
    if cv_m:
        g = cv_m.groups()
        if g[0] and g[1]:    # C1D1 / Cycle 1 Day 1
            cycle = f"C{g[0]}"
            day   = f"D{g[1]}"
        elif g[2]:           # Cycle N
            cycle = f"C{g[2]}"
        elif g[3]:           # Day N
            day   = f"D{g[3]}"
        elif g[4]:           # Week N
            week  = f"W{g[4]}"

    window: Optional[str] = None
    win_m = _WINDOW_RE.search(text)
    if win_m:
        window = f"{win_m.group(1)} {win_m.group(2)}"

    timepoint: Optional[str] = None
    tp_m = _TIMEPOINT_RE.search(text)
    if tp_m:
        timepoint = tp_m.group(1).lower()

    return {
        "anchor":    anchor,
        "cycle":     cycle,
        "day":       day,
        "week":      week,
        "window":    window,
        "timepoint": timepoint,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Identity  (statistical_identity)
# ─────────────────────────────────────────────────────────────────────────────
# Deterministic extraction of numeric/statistical values from clinical text.
# Stored on every indexed object so the numeric retriever can pre-filter on
# exact numeric tokens instead of using semantic vector search.
#
# Examples:
#   "n = 8"              → {sample_size: 8}
#   "95% CI: 27–48 days" → {confidence_level: 95, lower_ci: 27, upper_ci: 48, ci_unit: "days"}
#   "p<0.0001"           → {p_value: 0.0001}
#   "73%"                → {percentage: 73.0}
#   "HR = 0.65"          → {hazard_ratio: 0.65}

_SAMPLE_SIZE_STAT_RE = re.compile(
    r'\b[nN]\s*=\s*(\d+)'
    r'|(\d+)\s+(?:subjects?|patients?|participants?|individuals?|volunteers?)\b',
    re.I,
)

_CI_BOUNDS_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*%\s+CI\s*[:\s]\s*'
    r'(\d+(?:\.\d+)?)\s*[\u2013\u2014-]\s*(\d+(?:\.\d+)?)'
    r'(?:\s+(\w+))?',
    re.I,
)

_P_VALUE_STAT_RE = re.compile(
    r'\bp\s*[<>=\u2264\u2265]\s*(0\.\d+)\b',
    re.I,
)

_HR_STAT_RE = re.compile(
    r'\bHR\s*=\s*([\d.]+)\b'
    r'|\bhazard\s+ratio\s*=?\s*([\d.]+)\b',
    re.I,
)

_OR_STAT_RE = re.compile(
    r'\bOR\s*=\s*([\d.]+)\b'
    r'|\bodds\s+ratio\s*=?\s*([\d.]+)\b',
    re.I,
)

_MEDIAN_STAT_RE = re.compile(
    r'\bmedian\s+(?:of\s+)?(\d+(?:\.\d+)?)\s+(\w+)\b'
    r'|\bmedian\s*[=:]\s*(\d+(?:\.\d+)?)\s*(\w+)?\b',
    re.I,
)

_PERCENTAGE_STAT_RE = re.compile(r'(\d+(?:\.\d+)?)\s*%')

_STAT_TIME_UNITS = frozenset({
    "day", "days", "week", "weeks", "month", "months",
    "year", "years", "hour", "hours",
})


def _extract_statistical_identity(text: str) -> dict:
    """
    Extract structured numeric/statistical values from text.

    Returns a dict containing only the keys that were found (empty dict when
    nothing is recognisable as a statistical value).

    Called from enrich_object() so that both document objects and CI objects
    carry statistical_identity in the index for structured numeric lookup.
    """
    result: dict = {}

    # 1. Confidence interval bounds (highest priority — most specific)
    m = _CI_BOUNDS_RE.search(text)
    if m:
        try:
            result["confidence_level"] = float(m.group(1))
            result["lower_ci"]         = float(m.group(2))
            result["upper_ci"]         = float(m.group(3))
        except (ValueError, TypeError):
            pass
        unit = (m.group(4) or "").lower().strip()
        if unit in _STAT_TIME_UNITS:
            result["ci_unit"] = unit

    # 2. Sample size (n = X or X subjects/patients)
    m = _SAMPLE_SIZE_STAT_RE.search(text)
    if m:
        raw = m.group(1) or m.group(2)
        try:
            result["sample_size"] = int(raw)
        except (ValueError, TypeError):
            pass

    # 3. p-value
    m = _P_VALUE_STAT_RE.search(text)
    if m:
        try:
            result["p_value"] = float(m.group(1))
        except (ValueError, TypeError):
            pass

    # 4. Hazard ratio
    m = _HR_STAT_RE.search(text)
    if m:
        raw = m.group(1) or m.group(2)
        try:
            result["hazard_ratio"] = float(raw)
        except (ValueError, TypeError):
            pass

    # 5. Odds ratio
    m = _OR_STAT_RE.search(text)
    if m:
        raw = m.group(1) or m.group(2)
        try:
            result["odds_ratio"] = float(raw)
        except (ValueError, TypeError):
            pass

    # 6. Median with time unit
    m = _MEDIAN_STAT_RE.search(text)
    if m:
        raw  = m.group(1) or m.group(3)
        unit = (m.group(2) or m.group(4) or "").lower().strip()
        try:
            result["median"] = float(raw)
        except (ValueError, TypeError):
            pass
        if unit in _STAT_TIME_UNITS:
            result["median_unit"] = unit

    # 7. Standalone percentage — skip only when CI bounds already consumed the
    # "X% CI" portion of the match (that % belongs to confidence_level, not a
    # separate percentage fact). p-value never contains a "%" in its own
    # regex, so it cannot collide here and must not suppress percentage
    # extraction — a sentence can state both a p-value and an unrelated
    # percentage (e.g. "90% power ... p = 0.05") and both are real facts.
    if "lower_ci" not in result:
        pct_matches = _PERCENTAGE_STAT_RE.findall(text)
        if pct_matches:
            try:
                result["percentage"] = float(pct_matches[0])
            except (ValueError, TypeError):
                pass

    # 8. Type discriminant — must be last so all value keys are set first.
    # Priority: confidence_interval > hazard_ratio > odds_ratio > median
    #           > p_value > sample_size > percentage
    # This single field lets the numeric retriever use a filter query
    # (statistical_identity.type = X) instead of just numeric token matching,
    # eliminating cases where e.g. "dose level 8" matches a CI of "n = 8".
    if "lower_ci" in result and "upper_ci" in result:
        result["type"] = "confidence_interval"
    elif "hazard_ratio" in result:
        result["type"] = "hazard_ratio"
    elif "odds_ratio" in result:
        result["type"] = "odds_ratio"
    elif "median" in result:
        result["type"] = "median"
    elif "p_value" in result:
        result["type"] = "p_value"
    elif "sample_size" in result:
        result["type"] = "sample_size"
    elif "percentage" in result:
        result["type"] = "percentage"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Modality  (modality)
# ─────────────────────────────────────────────────────────────────────────────
# Orthogonal to statement_type:
#   statement_type = WHAT the statement is about  (primary objective, safety…)
#   modality       = HOW the statement is asserted (requirement, eligibility…)
#
# The distinction matters for reranking: a REQUIREMENT CI should match a
# document sentence with REQUIREMENT modality, not an OBSERVATION.

_MODALITY_RULES: list[tuple] = [
    ("REQUIREMENT",    re.compile(
        r'\b(must|shall|(?:is|are)\s+required|mandatory|'
        r'(?:is|are)\s+not\s+permitted|prohibited|'
        r'needs?\s+to|will\s+be\s+required)\b', re.I)),
    ("ELIGIBILITY",    re.compile(
        r'\b(inclusion\s+criterion|exclusion\s+criterion|'
        r'criteria\s+for\s+(?:inclusion|exclusion)|'
        r'must\s+have|must\s+not\s+have|should\s+have|should\s+not\s+have|'
        r'(?:is|are)\s+eligible|eligibility\s+criteria)\b', re.I)),
    ("OBSERVATION",    re.compile(
        r'\b(demonstrated|showed?|has\s+shown|reported|found|revealed|'
        r'indicated|confirmed|observed|was\s+(?:shown|found|demonstrated))\b', re.I)),
    ("RECOMMENDATION", re.compile(
        r'\b((?:it\s+is\s+)?recommended|should\s+(?:be\s+)?consider(?:ed)?|'
        r'suggested|may\s+be\s+(?:considered|appropriate)|'
        r'it\s+is\s+advised)\b', re.I)),
    ("OBJECTIVE",      re.compile(
        r'\b(?:primary|secondary|exploratory)\s+(?:objective|endpoint)\b|'
        r'\bto\s+(?:evaluate|assess|determine|characterize|measure|compare|'
        r'investigate|establish|identify|estimate|describe)\b', re.I)),
]

_STMT_TO_MODALITY: dict[str, str] = {
    "PRIMARY_OBJECTIVE":    "OBJECTIVE",
    "SECONDARY_OBJECTIVE":  "OBJECTIVE",
    "EXPLORATORY":          "OBJECTIVE",
    "GENERAL_OBJECTIVE":    "OBJECTIVE",
    "SAFETY":               "OBJECTIVE",
    "EFFICACY":             "OBJECTIVE",
    "PHARMACOKINETICS":     "OBJECTIVE",
    "PHARMACODYNAMICS":     "OBJECTIVE",
    "DOSING":               "REQUIREMENT",
    "PROTOCOL":             "REQUIREMENT",
    "POPULATION":           "ELIGIBILITY",
    "BACKGROUND":           "OBSERVATION",
}


def _classify_modality(text: str, statement_type: str) -> str:
    """
    Classify the assertional modality of the statement.

    Returns one of: OBJECTIVE | REQUIREMENT | ELIGIBILITY | OBSERVATION |
                    RECOMMENDATION | GENERAL

    Rules are tried in priority order; the first match wins.
    Falls back to the statement_type → modality mapping, then "GENERAL".
    """
    for modality, pat in _MODALITY_RULES:
        if pat.search(text):
            return modality
    return _STMT_TO_MODALITY.get(statement_type, "GENERAL")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint Identity  (endpoint_identity)
# ─────────────────────────────────────────────────────────────────────────────
# Complements facts["endpoint"] with structured sub-dimensions.
# "ORR at 12 months by IMWG, investigator-assessed" differs clinically from
# "ORR by IMWG"  — these fields make that distinction retrievable.

_ENDPOINT_TYPE_MAP: list[tuple] = [
    ("EFFICACY",     re.compile(
        r'\b(orr|overall\s+response(?:\s+rate)?|response\s+rate|'
        r'pfs|progression[-\s]free\s+survival|'
        r'dor|duration\s+of\s+response|'
        r'os|overall\s+survival|'
        r'(?:scr|sCR|stringent\s+complete|very\s+good\s+partial|partial|complete|minimal)\s+response|'
        r'mrd(?:\s+negativity)?|minimal\s+residual|'
        r'cbr|clinical\s+benefit\s+rate|dcr|disease\s+control\s+rate|'
        r'ttnt|time\s+to\s+next\s+treatment|pfs2)\b', re.I)),
    ("SAFETY",       re.compile(
        r'\b(adverse\s+event|ae|toxicity|tolerability|safety|'
        r'serious\s+adverse|grade\s+[34]|dlt|dose[-\s]limiting|'
        r'crs|cytokine\s+release|icans|neurotoxicity|irr|infusion[-\s]related)\b', re.I)),
    ("DOSE_FINDING", re.compile(
        r'\b(rp2d|recommended\s+phase\s+2\s+dose|'
        r'mtd|maximum\s+tolerated|dose\s+(?:escalation|expansion)|'
        r'safety\s+and\s+tolerability|pharmacologically\s+active\s+dose)\b', re.I)),
    ("PK",           re.compile(
        r'\b(pk|pharmacokinetic|auc|cmax|half[-\s]life|clearance|'
        r'bioavailability|vd|volume\s+of\s+distribution|plasma\s+concentration)\b', re.I)),
    ("BIOMARKER",    re.compile(
        r'\b(biomarker|ctdna|circulating\s+tumor\s+dna|'
        r'bcma\s+expression|cd38\s+expression|minimal\s+residual\s+disease)\b', re.I)),
    ("PRO",          re.compile(
        r'\b(hrqol|health[-\s]related\s+quality\s+of\s+life|'
        r'patient[-\s]reported|\bpro\b|eq[-\s]5d|eortc|facit|hru|mru)\b', re.I)),
]

_CRITERIA_RE = re.compile(
    r'\b(IMWG|International\s+Myeloma\s+Working\s+Group|'
    r'RECIST\s*[\d.]*|CTCAE\s*[\d.]*|IWG|IWCLL|Lugano|'
    r'Response\s+Evaluation\s+Criteria\s+in\s+Solid\s+Tumors)\b',
    re.I,
)

_THRESHOLD_RE = re.compile(
    r'\b(?:at\s+least\s+a?\s*)?'
    r'(?:stringent\s+complete|complete|very\s+good\s+partial|partial|minimal)\s+'
    r'response\s*(?:\[(?:sCR|CR|VGPR|PR|MR)\])?(?:\s+or\s+better)?'
    r'|MRD[-\s]negativ(?:e|ity)'
    r'|(?:\u2265|>=)\s*\d+%\s*(?:reduction)?'
    r'|\bPR\s+or\s+better\b'
    r'|\bCR\s+or\s+better\b'
    r'|\bVGPR\s+or\s+better\b',
    re.I,
)

_ASSESSOR_RE = re.compile(
    r'\b(investigator[-\s](?:assessed|reviewed|reported)|'
    r'irc|independent\s+review(?:\s+committee)?|'
    r'blinded\s+(?:central|independent|irc)|'
    r'central\s+(?:lab(?:oratory)?|review)|'
    r'sponsor[-\s]independent)\b',
    re.I,
)

_ENDPOINT_TIME_RE = re.compile(
    r'\b(?:at\s+(?:\d+\s+(?:months?|weeks?|years?)|(?:week|month)\s+\d+|randomization)|'
    r'\d+[-\s]month\s+(?:landmark|rate|estimate)|'
    r'landmark\s+(?:at\s+)?\d+|within\s+\d+\s+(?:months?|weeks?))\b',
    re.I,
)


def _extract_endpoint_identity(
    text:     str,
    entities: list[dict],
    facts:    dict,
) -> dict:
    """
    Structured endpoint identity: type, assessment criteria, threshold, assessor, timepoint.

    Returns {endpoint, type, criteria, threshold, assessor, timepoint}.
    Complements facts["endpoint"] with structured sub-dimensions that distinguish
    e.g. ORR by IMWG investigator-assessed from ORR by IRC.
    """
    endpoints = facts.get("endpoint", [])
    endpoint: Optional[str] = endpoints[0] if endpoints else None

    endpoint_type = "GENERAL"
    for et, pat in _ENDPOINT_TYPE_MAP:
        if pat.search(text):
            endpoint_type = et
            break

    criteria: Optional[str] = None
    crit_m = _CRITERIA_RE.search(text)
    if crit_m:
        raw = crit_m.group(1).upper().strip()
        if "INTERNATIONAL MYELOMA" in raw or "INTERNATIONAL MYELOMA WORKING" in raw:
            criteria = "IMWG"
        elif "RESPONSE EVALUATION" in raw:
            criteria = "RECIST"
        else:
            criteria = raw

    threshold: Optional[str] = None
    thresh_m = _THRESHOLD_RE.search(text)
    if thresh_m:
        threshold = thresh_m.group(0).strip()

    assessor: Optional[str] = None
    assessor_m = _ASSESSOR_RE.search(text)
    if assessor_m:
        assessor = assessor_m.group(1).lower()

    timepoint: Optional[str] = None
    time_m = _ENDPOINT_TIME_RE.search(text)
    if time_m:
        timepoint = time_m.group(0).lower()

    return {
        "endpoint":  endpoint,
        "type":      endpoint_type,
        "criteria":  criteria,
        "threshold": threshold,
        "assessor":  assessor,
        "timepoint": timepoint,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Population Identity  (population_identity)
# ─────────────────────────────────────────────────────────────────────────────
# Decomposes the flat population string into clinically actionable sub-fields:
# RRMM patients who received ≥3 prior lines and are triple-class refractory
# is meaningfully different from newly diagnosed MM patients.

_DISEASE_ABBREV_RE = re.compile(
    r'\b(RRMM|NDMM|SMM|R/R\s+MM|NHL|DLBCL|CLL|AML|MDS|MCL|FL)\b',
)

_DISEASE_FULL_RE = re.compile(
    r'\b((?:relapsed(?:\s+(?:and|or|and/or))?\s*(?:/\s*)?refractory\s+)?multiple\s+myeloma|'
    r'newly\s+diagnosed\s+multiple\s+myeloma|'
    r'smoldering\s+multiple\s+myeloma|'
    r'diffuse\s+large\s+b[-\s]cell\s+lymphoma|'
    r'chronic\s+lymphocytic\s+leukemia|'
    r'mantle\s+cell\s+lymphoma|'
    r'follicular\s+lymphoma)\b',
    re.I,
)

_POP_PRIOR_LINES_RE = re.compile(
    r'(?:received|had|with)\s+(?:at\s+least\s+)?'
    r'([1-9]|one|two|three|four|five)\s+(?:or\s+more\s+)?'
    r'(?:prior\s+)?lines?\s+of\s+(?:anti-?myeloma\s+)?therapy'
    r'|[\u2265>=]\s*(\d+)\s+(?:prior\s+)?(?:lines?\s+of\s+)?'
    r'(?:anti-?myeloma\s+)?(?:therapy|treatment|regimens?)\b',
    re.I,
)

_ECOG_RE = re.compile(
    r'\bECOG\s+(?:PS|performance\s+status|score)\s*(?:of\s*)?'
    r'([0-4](?:\s*[-\u2013]\s*[0-4])?|\u2264\s*[0-4]|\u2265\s*[0-4])\b',
    re.I,
)

_REFRACTORINESS_RE = re.compile(
    r'\b(triple[-\s]class\s+(?:exposed|refractory)|'
    r'penta[-\s]refractory|'
    r'lenalidomide[-\s]refractory|'
    r'pi[-\s]refractory|proteasome\s+inhibitor[-\s]refractory|'
    r'anti[-\s]cd38[-\s]refractory|'
    r'daratumumab[-\s]refractory|'
    r'bcma[-\s]refractory|'
    r'double[-\s]class\s+refractory)\b',
    re.I,
)

_POP_WORDS_TO_NUMS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}


def _extract_population_identity(
    text:     str,
    entities: list[dict],
    facts:    dict,
) -> dict:
    """
    Structured population identity: disease, relapse/refractory status, prior lines, ECOG.

    Returns {disease, relapsed, refractory, prior_lines, ecog_status, refractoriness}.
    """
    # Disease abbreviation (preferred) or long-form normalisation
    disease: Optional[str] = None
    abbrev_m = _DISEASE_ABBREV_RE.search(text)
    if abbrev_m:
        disease = abbrev_m.group(1)
    else:
        full_m = _DISEASE_FULL_RE.search(text)
        if full_m:
            raw = full_m.group(1).lower()
            if "relapsed" in raw and "refractory" in raw:
                disease = "RRMM"
            elif "newly diagnosed" in raw:
                disease = "NDMM"
            elif "smoldering" in raw:
                disease = "SMM"
            elif "multiple myeloma" in raw:
                disease = "MM"
            else:
                disease = raw[:40]

    # IMWG / International Myeloma Working Group → infer MM when no explicit
    # disease string was found.  IMWG criteria are myeloma-specific by definition.
    if disease is None and re.search(
        r'\b(imwg|international\s+myeloma\s+working\s+group)\b', text, re.I
    ):
        disease = "MM"

    tl = text.lower()
    relapsed   = bool(re.search(r'\brelapsed\b',   tl))
    refractory = bool(re.search(r'\brefractory\b', tl))

    # Prior lines of therapy
    prior_lines: Optional[str] = None
    lot_m = _POP_PRIOR_LINES_RE.search(text)
    if lot_m:
        raw_num = lot_m.group(1) or lot_m.group(2)
        if raw_num:
            num = _POP_WORDS_TO_NUMS.get(raw_num.lower(), raw_num)
            prior_lines = f"\u2265{num}"

    # ECOG performance status
    ecog_status: Optional[str] = None
    ecog_m = _ECOG_RE.search(text)
    if ecog_m:
        ecog_status = ecog_m.group(1).strip()

    # Refractoriness profile (triple-class, penta, lenalidomide-refractory, etc.)
    refractoriness: Optional[str] = None
    ref_m = _REFRACTORINESS_RE.search(text)
    if ref_m:
        refractoriness = ref_m.group(1).lower()

    return {
        "disease":        disease,
        "relapsed":       relapsed,
        "refractory":     refractory,
        "prior_lines":    prior_lines,
        "ecog_status":    ecog_status,
        "refractoriness": refractoriness,
    }


# ─────────────────────────────────────────────────────────────────────────────
# statement_type  (metadata-only — no model)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_statement(heading_path: list, text: str, section_cat: str) -> str:
    for heading in reversed(heading_path):   # most-specific heading first
        for stmt_type, pat in _HEADING_STMT_RULES:
            if pat.search(heading):
                return stmt_type
    tl = text.lower()
    if "primary objective" in tl or "primary endpoint" in tl:
        return "PRIMARY_OBJECTIVE"
    if "secondary objective" in tl or "secondary endpoint" in tl:
        return "SECONDARY_OBJECTIVE"
    if "exploratory" in tl and ("objective" in tl or "endpoint" in tl):
        return "EXPLORATORY"
    return _SECTION_TO_STMT.get(section_cat, "GENERAL")


# ─────────────────────────────────────────────────────────────────────────────
# study_context  (metadata primary — spaCy only for "OTHER" sections)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_context_metadata(section_cat: str, text: str) -> str:
    """Section_category + text patterns only — no spaCy."""
    if section_cat in _CITED_SECTIONS:
        return "CITED"
    if section_cat in _CURRENT_SECTIONS:
        return "CURRENT"
    if _THIS_STUDY_RE.search(text):
        return "CURRENT"
    # FIX-10: Linguistic citation patterns beat HISTORICAL so that
    # "Based on Phase 1 SC data..." is correctly CITED rather than GENERAL.
    # Checked before HISTORICAL to avoid masking genuine historical mentions.
    if _CITED_LINGUISTIC_RE.search(text):
        return "CITED"
    if _HISTORICAL_RE.search(text):
        return "HISTORICAL"
    return "GENERAL"     # ambiguous — spaCy will refine if available


def _refine_context_spacy(doc, entities: list[dict], section_cat: str) -> str:
    """
    Tertiary signal: verb tense.  Only called when section_cat is "OTHER"/unknown.
    Checks if the ROOT verb is past-tense + cited verb → CITED,
    or modal/present + intent verb → CURRENT.
    """
    has_external_id = any(e.get("label") == "PROTOCOL_ID" for e in entities)
    for sent in doc.sents:
        root_tok = _find_root(sent)
        if root_tok is None or root_tok.pos_ != "VERB":
            continue
        lemma = (root_tok.lemma_ or root_tok.text).lower()
        tag   = root_tok.tag_
        if tag in _PAST_TAGS and lemma in {
            "demonstrate", "show", "report", "find", "reveal", "indicate", "confirm"
        }:
            return "CITED"
        if tag in _PAST_TAGS and has_external_id:
            return "CITED"
        if (tag in _MODAL_TAGS or tag in _PRESENT_TAGS) and lemma in {
            "evaluate", "assess", "determine", "characterize", "compare",
            "measure", "investigate", "explore", "identify",
        }:
            return "CURRENT"
    return "GENERAL"


# ─────────────────────────────────────────────────────────────────────────────
# Action verb extraction  (facts["action"])
# ─────────────────────────────────────────────────────────────────────────────

def _extract_action_verbs(doc) -> list[str]:
    """Collect lemmas of clinical ROOT/xcomp verbs."""
    verbs: list[str] = []
    for token in doc:
        if token.pos_ != "VERB" or token.dep_ not in {"ROOT", "xcomp"}:
            continue
        lemma = (token.lemma_ or token.text).lower()
        if lemma in _CLINICAL_VERBS and lemma not in verbs:
            verbs.append(lemma)
    return verbs


# ─────────────────────────────────────────────────────────────────────────────
# Clinical relation extraction  (spaCy dep + GLiNER entities → typed slots)
#
# spaCy's contribution is narrowly defined:
#   1. Find ALL clinical verbs in each sentence (ROOT + xcomp children)
#   2. Group entities by sentence — those in the same sentence are candidates
#   3. Filter pairs by dependency distance (> MAX_DEP_DISTANCE → skip)
#   4. For every remaining pair + verb, call _lookup_relation()
#   5. If a known relation exists, emit a slot-filled dict
# ─────────────────────────────────────────────────────────────────────────────

MAX_DEP_DISTANCE     = 5   # entity-to-entity dep-tree distance cutoff
MAX_ENTITY_VERB_DIST = 4   # each entity must be within this many hops of its verb


def _head_token_for_entity(doc, entity: dict):
    """
    Find the syntactic head token for a GLiNER entity (using char offsets).
    Returns the token whose .head falls outside the entity span.
    """
    start = entity.get("object_start", -1)
    if start < 0:
        return None
    end = entity.get("object_end", start + len(entity.get("text", "")))
    span_tokens = [t for t in doc if start <= t.idx < end]
    if not span_tokens:
        return None
    for tok in span_tokens:
        if not (start <= tok.head.idx < end):
            return tok
    return span_tokens[0]


def _dep_distance(tok_a, tok_b) -> int:
    """Shortest path length between two tokens in the (undirected) dependency tree."""
    if tok_a == tok_b:
        return 0
    visited: set = {tok_a}
    queue:   list = [(tok_a, 0)]
    while queue:
        cur, dist = queue.pop(0)
        if cur.head is not cur:               # not ROOT
            nxt = cur.head
            if nxt == tok_b:
                return dist + 1
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, dist + 1))
        for child in cur.children:
            if child == tok_b:
                return dist + 1
            if child not in visited:
                visited.add(child)
                queue.append((child, dist + 1))
    return 999


def _relation_confidence(dep_dist: int, verb_dist: int, verb_dep: str) -> float:
    """
    Deterministic confidence score for a candidate relation (no ML).

    Signals:
      dep_dist  — entity-to-entity dependency distance  (shorter = better)
      verb_dist — max(entity_a→verb, entity_b→verb) distance  (shorter = better)
      verb_dep  — dep type of the governing verb token (ROOT best, conj weakest)
    """
    dist_score = max(0.0, 1.0 - (dep_dist - 1) / MAX_DEP_DISTANCE)
    verb_score = max(0.0, 1.0 - (verb_dist - 1) / MAX_ENTITY_VERB_DIST)
    type_score = {"ROOT": 1.0, "xcomp": 0.9, "advcl": 0.8,
                  "ccomp": 0.8, "conj":  0.7}.get(verb_dep, 0.75)
    return round(0.40 * dist_score + 0.35 * verb_score + 0.25 * type_score, 3)


def _extract_clinical_relations(doc, entities: list[dict]) -> list[dict]:
    """
    Produce typed clinical slot-filled relations.

    Filter order (fastest checks first):
      1. Valid-pair whitelist (_VALID_PAIRS) — no dep parse needed
      2. Entity-to-entity dep distance ≤ MAX_DEP_DISTANCE
      3. Each entity within MAX_ENTITY_VERB_DIST hops of the governing verb
      4. Relation lookup + slot-fill + deterministic confidence score

    Output format (only present slots are included):
      {
        "drug":       "teclistamab",
        "endpoint":   "ORR",
        "relation":   "measured_by",
        "verb":       "evaluate",
        "confidence": 0.847,
      }
    """
    relations: list[dict] = []
    seen: set[tuple] = set()

    for sent in doc.sents:
        verb_entries = _all_clinical_verbs(sent)   # [(lemma, token), ...]
        if not verb_entries:
            continue

        s_start = sent.start_char
        s_end   = sent.end_char
        sent_ents = [
            e for e in entities
            if s_start <= e.get("object_start", -1) < s_end
            and e.get("label") in _SLOT_PRIORITY
        ]
        if len(sent_ents) < 2:
            continue

        head_tokens = {id(e): _head_token_for_entity(doc, e) for e in sent_ents}

        for verb, verb_tok in verb_entries:
            for i, ent_a in enumerate(sent_ents):
                for ent_b in sent_ents[i + 1:]:
                    type_a = ent_a.get("label", "")
                    type_b = ent_b.get("label", "")

                    # 1. Valid-pair whitelist — skip impossible combinations before any dep parse
                    if _canonical_pair(type_a, type_b) not in _VALID_PAIRS:
                        continue

                    # 1b. FIX-3 (relations): mirror the facts["response_criterion"]
                    # IMWG whitelist here so non-IMWG CLINICAL_RESPONSE entities
                    # (e.g. "efficacy", "plateau", "trend") never appear in
                    # clinical_relations either.
                    if type_a == "CLINICAL_RESPONSE":
                        _canon_a = (ent_a.get("canonical") or ent_a.get("text", "")).strip()
                        if _canon_a not in _IMWG_RESPONSE_CRITERIA:
                            continue
                    if type_b == "CLINICAL_RESPONSE":
                        _canon_b = (ent_b.get("canonical") or ent_b.get("text", "")).strip()
                        if _canon_b not in _IMWG_RESPONSE_CRITERIA:
                            continue

                    tok_a = head_tokens.get(id(ent_a))
                    tok_b = head_tokens.get(id(ent_b))

                    # 2. Entity-to-entity distance (neutral fallback when token not found)
                    dep_dist = MAX_DEP_DISTANCE
                    if tok_a is not None and tok_b is not None:
                        dep_dist = _dep_distance(tok_a, tok_b)
                        if dep_dist > MAX_DEP_DISTANCE:
                            continue

                    # 3. Entity-to-verb distance (each entity must be near the governing verb)
                    verb_dist_a = MAX_ENTITY_VERB_DIST if tok_a is None else _dep_distance(tok_a, verb_tok)
                    verb_dist_b = MAX_ENTITY_VERB_DIST if tok_b is None else _dep_distance(tok_b, verb_tok)
                    max_verb_dist = max(verb_dist_a, verb_dist_b)
                    if max_verb_dist > MAX_ENTITY_VERB_DIST:
                        continue

                    text_a = ent_a.get("text", "")
                    text_b = ent_b.get("text", "")

                    rel_name = _lookup_relation(type_a, type_b, verb)
                    if rel_name is None:
                        continue

                    dedup_key = (text_a, rel_name, text_b)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    slot_dict = _build_slot_dict(type_a, text_a, type_b, text_b)
                    slot_dict["relation"]   = rel_name
                    slot_dict["verb"]       = verb
                    slot_dict["confidence"] = _relation_confidence(
                        dep_dist, max_verb_dist, verb_tok.dep_
                    )
                    relations.append(slot_dict)

    return relations


def _all_clinical_verbs(sent) -> list[tuple]:
    """
    Return (lemma, token) pairs for ALL clinical verbs in a sentence.

    Handles constructions like:
      "evaluate efficacy and assess safety"  → [("evaluate", tok), ("assess", tok)]
      "is to evaluate"                       → [("evaluate", tok)]
    """
    verbs: list[tuple] = []
    seen:  set[str]    = set()

    root_tok = _find_root(sent)
    if root_tok is None:
        return verbs

    # Gather ROOT + xcomp / advcl / ccomp VERB tokens.
    # advcl covers purpose clauses: "combined ... to evaluate"
    # xcomp covers control verbs:  "is to evaluate"
    candidates = [root_tok] + [
        t for t in sent
        if t.dep_ in {"xcomp", "advcl", "ccomp"} and t.pos_ == "VERB"
    ]

    # Second pass: conj children of already-gathered verbs.
    # Handles "evaluate ORR and assess safety" where "assess" is conj of "evaluate".
    for tok in list(candidates):
        for child in tok.children:
            if child.dep_ == "conj" and child.pos_ == "VERB" and child not in candidates:
                candidates.append(child)

    for tok in candidates:
        lemma = (tok.lemma_ or tok.text).lower()
        # Resolve copula: "is to evaluate" → skip "is", pick its xcomp child
        if lemma in {"be", "is", "are", "was", "were"} and tok.pos_ in {"VERB", "AUX"}:
            for child in tok.children:
                if child.dep_ == "xcomp" and child.pos_ == "VERB":
                    cl = (child.lemma_ or child.text).lower()
                    if cl in _CLINICAL_VERBS and cl not in seen:
                        seen.add(cl)
                        verbs.append((cl, child))
            continue
        if lemma in _CLINICAL_VERBS and lemma not in seen:
            seen.add(lemma)
            verbs.append((lemma, tok))

    return verbs


def _build_slot_dict(type_a: str, text_a: str, type_b: str, text_b: str) -> dict:
    """
    Map entity types to clinical slot names and fill them.
    Handles both orders — the canonical pair ordering determines which is first.
    """
    slot_a = _LABEL_TO_SLOT.get(type_a)
    slot_b = _LABEL_TO_SLOT.get(type_b)
    result: dict[str, str] = {}
    if slot_a:
        result[slot_a] = text_a
    if slot_b and slot_b != slot_a:
        result[slot_b] = text_b
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _find_root(sent) -> Optional[object]:
    for token in sent:
        if token.dep_ == "ROOT":
            return token
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Effective Facts  (heading → paragraph → sentence inheritance)
# Phase 2 — StudyHierarchy   (Study → Part → Arm → Cohort → Phase)
# Phase 3 — ClinicalIdentity (unified comparable object)
# ─────────────────────────────────────────────────────────────────────────────
#
# New fields added to every enriched object after propagate_effective_facts():
#
#   own_facts         dict  facts extracted from this object's own text only
#   effective_facts   dict  own_facts ∪ inherited ancestor context
#                           (own_facts wins on conflict; used by scoring)
#   study_hierarchy   dict  {study_id, part, arm, cohort, phase}
#   clinical_identity dict  {drug, endpoint, population, study_id, part,
#                            arm, cohort, phase, section}
#
# Why two-tier (not indexing-only):
#   A sentence "Overall response rate..." has own_facts = {endpoint: [ORR]}.
#   The drug lives in the paragraph above or the section heading.
#   effective_facts = {drug: [teclistamab], endpoint: [ORR]} — complete picture.
#   own_facts is preserved so human review can see what was explicit vs inherited.

_PART_RE   = re.compile(r'\bpart\s+(\d+|[ivxlc]+)\b',     re.I)
_COHORT_RE = re.compile(r'\bcohort\s+(\d+|[A-Z]\w*)\b',   re.I)
_ARM_RE    = re.compile(r'\barm\s+([A-Z]\b|\w+)',           re.I)
_PHASE_RE  = re.compile(r'\bphase\s+(\d[ab]?|[IVXivx]+)\b', re.I)


def _extract_study_hierarchy(
    heading_path:    list,
    effective_facts: dict,
    text:            str = "",
) -> dict:
    """
    Extract a structured study hierarchy from heading breadcrumbs + effective facts.

    Returns {study_id, part, arm, cohort, phase}.  All values may be None.
    Heading text is a richer signal for part/cohort than entity extraction because
    GLiNER treats "Part 3" as a STUDY_ARM, conflating arm and part.
    """
    # Normalize: heading_path may arrive as a " > "-joined string (index-stored form).
    if isinstance(heading_path, str):
        heading_path = [h.strip() for h in heading_path.split(" > ") if h.strip()]
    combined = " ".join(heading_path) + " " + text

    # study_id — prefer structured fact, regex as fallback
    study_ids = effective_facts.get("study_id", [])
    study_id  = study_ids[0] if study_ids else None

    # part — heading text is the primary signal ("Part 3", "Part II", etc.)
    part_m = _PART_RE.search(combined)
    part   = part_m.group(1).upper() if part_m else None

    # arm — prefer structured fact; fall back to heading regex
    arm_vals = effective_facts.get("study_arm", [])
    arm      = arm_vals[0] if arm_vals else None
    if arm is None:
        arm_m = _ARM_RE.search(" ".join(heading_path))
        arm   = arm_m.group(1) if arm_m else None

    # cohort
    cohort_m = _COHORT_RE.search(combined)
    cohort   = cohort_m.group(1) if cohort_m else None

    # phase — prefer structured fact; fall back to heading regex
    phase_vals = effective_facts.get("phase", [])
    phase      = phase_vals[0] if phase_vals else None
    if phase is None:
        phase_m = _PHASE_RE.search(combined)
        phase   = phase_m.group(1) if phase_m else None

    return {
        "study_id": study_id,
        "part":     part,
        "arm":      arm,
        "cohort":   cohort,
        "phase":    phase,
    }


def _build_clinical_identity(
    effective_facts:    dict,
    study_hierarchy:    dict,
    section_category:   str            = "",
    treatment_identity: dict | None    = None,
    endpoint_identity:  dict | None    = None,
) -> dict:
    """
    Unified clinical identity for one indexed object.

    Combines all clinically significant signals into a single comparable object.
    When the reranker eventually moves to identity-level matching (Phase 3),
    this is the object it will compare CI→candidate against.

    Fields:
        drug             — from effective_facts["drug"]            (up to 2 values)
        primary_drug     — canonical single drug (treatment_identity.primary_drug
                           → effective_facts.drug[0])
        endpoint         — from effective_facts["endpoint"]        (up to 2 values)
        primary_endpoint — canonical single endpoint (endpoint_identity.endpoint
                           → effective_facts.endpoint[0])
        population       — from effective_facts["population"]      (up to 2 values)
        study_id         — from study_hierarchy
        part             — study part (Part 1/2/3 etc.)
        arm              — treatment arm
        cohort           — expansion cohort label
        phase            — clinical phase
        section          — section_category (intent proxy)
    """
    ti    = treatment_identity or {}
    ei    = endpoint_identity  or {}
    drugs    = (effective_facts.get("drug") or [])[:2]
    eps      = (effective_facts.get("endpoint") or [])[:2]
    diseases = (effective_facts.get("disease") or [])[:1]
    return {
        "drug":             drugs,
        "primary_drug":     ti.get("primary_drug") or (drugs[0] if drugs else None),
        "endpoint":         eps,
        "primary_endpoint": ei.get("endpoint")    or (eps[0]   if eps   else None),
        "population": (effective_facts.get("study_population") or
                       effective_facts.get("population")       or [])[:2],
        "disease":    diseases[0] if diseases else None,
        "study_id":   study_hierarchy.get("study_id"),
        "part":       study_hierarchy.get("part"),
        "arm":        study_hierarchy.get("arm"),
        "cohort":     study_hierarchy.get("cohort"),
        "phase":      study_hierarchy.get("phase"),
        "section":    section_category,
    }


def propagate_effective_facts(objects: list) -> None:
    """
    Second-pass enrichment: compute effective_facts, study_hierarchy, and
    clinical_identity for every object in a chunk.

    Must be called AFTER enrich_object() has populated facts on all objects.
    Mutates objects in-place.

    Inheritance model (document order within the chunk):

      heading     — partial-reset: inherits slots the heading doesn't redefine,
                    then lets own_facts override.  (Sub-headings don't lose drug
                    context established by a parent heading.)
      paragraph   — inherits context, then contributes its own facts back up.
      list        — same as paragraph.
      table        — same as paragraph; clinical protocols love tables.
      table_row   — inherits context; does NOT update context (too granular).
      sentence    — inherits context; does NOT update context.

    own_facts       = GLiNER facts from this object's own text (unchanged).
    effective_facts = context ∪ own_facts  (own_facts wins on slot conflict).
    inherited_slots = list of slots in effective_facts that came from context,
                      not from this object's own text.  Used by downstream
                      validators to apply a confidence discount (0.75 vs 1.0).
    """
    # Process in document order
    sorted_objs = sorted(objects, key=lambda o: float(o.get("position") or 0))

    # Running context: accumulated facts from headings and paragraphs above.
    context: dict[str, list] = {}
    context_source: dict[str, str] = {}   # slot → object_type that last wrote it

    # Object types that actively contribute their own facts back to the context
    # (so that subsequent siblings/children can inherit them).
    _CONTEXT_CONTRIBUTORS = frozenset({"paragraph", "list", "table"})

    for obj in sorted_objs:
        own      = {k: list(v) for k, v in (obj.get("facts") or {}).items() if v}
        obj_type = (obj.get("type") or "paragraph").lower()

        if obj_type == "heading":
            # Partial reset: carry over slots the heading doesn't explicitly define,
            # then let own_facts override.
            new_ctx: dict[str, list] = {
                slot: vals for slot, vals in context.items() if slot not in own
            }
            new_ctx.update(own)
            context = new_ctx
            for slot in own:                    # heading is the source for its own slots
                context_source[slot] = "heading"
            eff = dict(context)
        else:
            # Build effective facts: context as defaults, own as override
            eff = {slot: list(vals) for slot, vals in context.items() if vals}
            for slot, vals in own.items():
                eff[slot] = vals  # own_facts wins

            # Paragraphs, lists, and tables contribute back to the running context
            if obj_type in _CONTEXT_CONTRIBUTORS:
                for slot, vals in own.items():
                    if vals:
                        context[slot] = list(vals)
                        context_source[slot] = obj_type   # "paragraph" | "list" | "table"

        # Slots that came from context (not own text) → lower confidence downstream
        inherited = [s for s in eff if s not in own]
        # slot_provenance: maps each effective_facts slot to its source level.
        #   "explicit"   = in this object's own text
        #   "heading"    = inherited from an ancestor heading
        #   "paragraph"  = inherited from a preceding paragraph / list / table
        #   "context"    = inherited from an unclassified ancestor (edge case)
        slot_provenance = {
            slot: "explicit" if slot in own else context_source.get(slot, "context")
            for slot in eff
        }

        obj["own_facts"]       = own
        # Promote population_identity.disease into effective_facts.
        # Disease is commonly inferred from text patterns (e.g. RRMM, NDMM,
        # IMWG → MM) and does not appear as a typed NER entity slot, so it
        # never flows through the entity→slot pipeline automatically.
        _pd = (obj.get("population_identity") or {}).get("disease")
        if _pd and "disease" not in eff:
            eff["disease"] = [_pd]
        # ── effective_facts is always set BEFORE identity builders so that even
        # if an identity builder throws, this object still has effective_facts.
        obj["effective_facts"] = eff
        obj["inherited_slots"] = inherited
        obj["slot_provenance"] = slot_provenance

        # ── Graduate inherited slots back into identity objects ─────────────
        # treatment_identity / endpoint_identity / population_identity are set
        # by enrich_object() using only the object's OWN text.  When a slot's
        # value came entirely from heading context (e.g. the paragraph doesn't
        # mention a drug, but the heading does), the identity object is empty
        # even though effective_facts is populated.  This makes comparators
        # that read from treatment_identity (e.g. regimen) return UNKNOWN.
        #
        # Table-driven: (identity_obj_key, identity_field, ef_slot, "first"|"rest")
        #   "first" → take eff[slot][0]
        #   "rest"  → take eff[slot][1:]  (only set when ≥2 values present)
        # own value always wins — entry is only applied when identity_field is absent/empty.
        # To add a new graduated field: add one row to the tuple below.
        for _id_key, _id_field, _ef_slot, _mode in (
            ("treatment_identity",  "primary_drug",    "drug",      "first"),
            ("treatment_identity",  "companion_drugs", "drug",      "rest"),
            ("endpoint_identity",   "endpoint",        "endpoint",  "first"),
            ("population_identity", "disease",         "disease",   "first"),
        ):
            _ef_vals = eff.get(_ef_slot)
            if not _ef_vals:
                continue
            _ident = dict(obj.get(_id_key) or {})
            if _ident.get(_id_field):
                continue                  # own value present — own wins
            if _mode == "first":
                _ident[_id_field] = _ef_vals[0]
            else:                         # "rest"
                if len(_ef_vals) < 2:
                    continue
                _ident[_id_field] = _ef_vals[1:]
            obj[_id_key] = _ident

        try:
            study_hier = _extract_study_hierarchy(
                obj.get("heading_path") or [], eff, obj.get("text", "")
            )
            obj["study_hierarchy"]    = study_hier
            obj["clinical_identity"]  = _build_clinical_identity(
                eff, study_hier, obj.get("section_category", ""),
                obj.get("treatment_identity"),
                obj.get("endpoint_identity"),
            )
            obj["clinical_signature"] = _build_clinical_signature(
                statement_type      = obj.get("statement_type", "GENERAL"),
                study_context       = obj.get("study_context", "GENERAL"),
                modality            = obj.get("modality", "GENERAL"),
                treatment_identity  = obj.get("treatment_identity") or {},
                endpoint_identity   = obj.get("endpoint_identity") or {},
                population_identity = obj.get("population_identity") or {},
                study_hierarchy     = study_hier,
            )
        except Exception as _id_exc:
            logger.warning(
                "[PropagateEF] identity builder failed for %s: %s — "
                "effective_facts preserved, identity fields set to empty",
                obj.get("object_id", "?"), _id_exc,
            )
            obj.setdefault("study_hierarchy",    {"study_id": None, "part": None, "arm": None, "cohort": None, "phase": None})
            obj.setdefault("clinical_identity",  {})
            obj.setdefault("clinical_signature", {})

