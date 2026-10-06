"""
Search Pipeline — Stage 3.5: Evidence Grounder
================================================
Decides WHICH semantic object(s) a retrieval hit actually refers to. This is the ONLY
stage where candidate cardinality may change — Context Expander (S4) consumes its output
1:1 and only adds context.

  retrieval hit ──► ground ──► 0..N grounded objects ──► Context Expander ──► 0..N contexts

Grounding contract:
  object-level hit (matched_object already set)  → itself                      [direct_object]
  chunk hit + literal/regex match spans          → EVERY distinct semantic      [literal_span /
                                                    object genuinely containing  regex_span /
                                                    a span (most specific        *_span_covered]
                                                    container wins per span;
                                                    multiple matches in the same
                                                    object merge into one
                                                    candidate with multiple
                                                    evidence_matches)
  chunk hit, no usable span                      → top-K objects of THAT chunk  [local_vector /
                                                    ranked against the CI        local_lexical]
  nothing reliable                               → unresolved (one unanchored candidate,
                                                    never a blind fan-out of the whole chunk)

Evidence multiplicity is not candidate multiplicity: 20 literal matches in one chunk may
collapse to as few as 1 object (all inside the same sentence) or as many as 20 (one per
object) — the number is determined by evidence, never by how many objects the chunk holds.

Input:  aggregated search request (must have "candidates", "ci", "document_id", "tenant")
Appends: "grounded_candidates": list[dict]  (each = original candidate fields +
         "matched_object" + "grounding": {"method", "confidence", "evidence_matches": [...]})
         "grounding_stats": {"retrieval_candidates", "grounded_candidates", "grounding_methods"}

Grounded candidate also carries "_chunk_pool": list[dict] | None — the full set of semantic
objects fetched for its chunk (None for direct hits, which never need one). Context Expander
reuses this instead of re-fetching, and slices it to a ±window for context building.
"""

from __future__ import annotations

import logging
import math
import operator
import os
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENSEARCH_ENDPOINT    = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
AWS_REGION             = os.environ.get("AWS_REGION", "us-east-1")

# Pool pagination page size (and running cap) for a single chunk's semantic objects —
# must be the FULL chunk, not a truncated prefix, or a late literal match never grounds.
CHUNK_OBJECTS_PAGE_SIZE = int(os.environ.get("CHUNK_OBJECTS_PAGE_SIZE", "400"))
CHUNK_OBJECTS_MAX       = int(os.environ.get("CHUNK_OBJECTS_MAX", "4000"))

# A span matching more than this many distinct objects (after hierarchy dedup) is not a
# reliable per-span anchor and that one span is skipped (other spans in the same chunk
# candidate are still tried). Set high — recall preferred over cost, every genuine match
# is sent to S6 llm_verify to judge rather than discarded here.
GROUNDING_SPAN_AMBIGUITY_LIMIT = int(os.environ.get("GROUNDING_SPAN_AMBIGUITY_LIMIT", "1000"))
# Safety net on TOTAL distinct objects a single chunk candidate may ground to via spans —
# evidence-driven, so this should rarely bind; it only guards against pathological chunks.
GROUNDING_MAX_SPAN_OBJECTS = int(os.environ.get("GROUNDING_MAX_SPAN_OBJECTS", "1000"))
# Fully-covered-object fallback (span longer than any single object) ignores objects shorter
# than this many characters.
GROUNDING_COVERED_MIN_CHARS = int(os.environ.get("GROUNDING_COVERED_MIN_CHARS", "20"))
# No-span fallback: how many locally-ranked objects a chunk hit may ground to.
GROUNDING_MAX_LOCAL_OBJECTS = int(os.environ.get("GROUNDING_MAX_LOCAL_OBJECTS", "3"))
GROUNDING_SCORE_MARGIN      = float(os.environ.get("GROUNDING_SCORE_MARGIN", "0.10"))
GROUNDING_MIN_VECTOR_SCORE  = float(os.environ.get("GROUNDING_MIN_VECTOR_SCORE", "0.20"))
GROUNDING_MIN_LEXICAL_SCORE = float(os.environ.get("GROUNDING_MIN_LEXICAL_SCORE", "0.30"))

_LITERAL_SPAN_CONFIDENCE  = 0.95
_COVERED_SPAN_CONFIDENCE  = 0.80
_LOCAL_SCORE_TYPES = {"sentence", "paragraph", "table_row", "list_item"}

# Specificity when several objects contain the same span: most specific first. A paragraph
# that contains the span is only context once one of its sentences is the candidate.
_GROUND_TYPE_RANK = {
    "sentence":     0,
    "table_cell":   1,
    "list_item":    2,
    "paragraph":    3,
    "table_row":    4,
    "table_header": 5,
    "heading":      6,
}


def _get_os():
    from shared.opensearch_client import get_opensearch_client
    return get_opensearch_client()


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Evidence Grounder] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Evidence Grounder] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Evidence Grounder] done search_id=%s grounded=%d",
                search_id, len(result["grounded_candidates"]))
    return result


def _process(req: dict) -> dict:
    candidates  = req.get("candidates", [])
    document_id = req.get("document_id") or ""
    tenant      = req.get("tenant") or {}
    project_id  = req.get("project_id")
    tenant_id   = tenant.get("tenant_id")
    ci          = req.get("ci") or {}

    if not candidates:
        return {**req, "grounded_candidates": [],
                "grounding_stats": {"retrieval_candidates": 0, "grounded_candidates": 0,
                                    "grounding_methods": {}}}

    chunk_ids_needed = {
        c.get("chunk_id", "") for c in candidates
        if not c.get("matched_object") and c.get("chunk_id")
    }
    pools = _fetch_chunk_pools(document_id, chunk_ids_needed, tenant_id, project_id)

    grounded: list[dict] = []
    methods: Counter = Counter()
    for c in candidates:
        direct_obj = c.get("matched_object")
        if direct_obj:
            grounded.append({
                **c, "matched_object": direct_obj, "_chunk_pool": None,
                "grounding": {"method": "direct_object", "confidence": 1.0, "evidence_matches": []},
            })
            methods["direct_object"] += 1
            continue

        pool = pools.get(c.get("chunk_id", ""), [])
        picks, reason = _ground_chunk_candidate(c, ci, pool)
        if not picks:
            grounded.append({
                **c, "matched_object": None, "_chunk_pool": pool,
                "grounding": {"method": "unresolved", "confidence": 0.0, "reason": reason, "evidence_matches": []},
            })
            methods[f"unresolved:{reason}"] += 1
            continue
        for obj, g in picks:
            grounded.append({**c, "matched_object": obj, "_chunk_pool": pool, "grounding": g})
            methods[g["method"]] += 1

    stats = {
        "retrieval_candidates": len(candidates),
        "grounded_candidates":  len(grounded),
        "grounding_methods":    dict(methods),
    }
    logger.info("[Evidence Grounder] search_id=%s  grounding=%s", req.get("search_id"), stats)
    return {**req, "grounded_candidates": grounded, "grounding_stats": stats}


# ── Chunk pool fetch (full, paginated — never a truncated prefix) ────────────────────────

def _fetch_chunk_pools(
    document_id: str,
    chunk_ids:   set[str],
    tenant_id:   str | None,
    project_id:  str | None,
) -> dict[str, list[dict]]:
    if not chunk_ids:
        return {}
    chunk_ids = sorted(chunk_ids)
    body: list[dict] = []
    for cid in chunk_ids:
        body.append({})
        body.append({
            "size": CHUNK_OBJECTS_PAGE_SIZE,
            "query": {"bool": {"filter": [
                {"term": {"parent_chunk_id": cid}},
                *([{"term": {"document_id": document_id}}] if document_id else []),
                *([{"term": {"tenant_id": tenant_id}}] if tenant_id else []),
                *([{"term": {"project_id": project_id}}] if project_id else []),
            ]}},
            "sort": [{"global_position": "asc"}],
        })
    pools: dict[str, list[dict]] = {}
    try:
        resp = _get_os().msearch(body=body, index=SEMANTIC_OBJECTS_INDEX)
        for cid, r in zip(chunk_ids, resp.get("responses", [])):
            pools[cid] = [h["_source"] for h in r.get("hits", {}).get("hits", [])]
    except Exception as exc:
        logger.warning("[Evidence Grounder] chunk pool msearch failed: %s", exc)
        return {cid: [] for cid in chunk_ids}

    # Page past CHUNK_OBJECTS_PAGE_SIZE for any chunk that hit it exactly, so a literal
    # match late in a long multi-page chunk still has a complete pool to ground against.
    pending = {cid for cid in chunk_ids
               if len(pools.get(cid, [])) == CHUNK_OBJECTS_PAGE_SIZE and len(pools[cid]) < CHUNK_OBJECTS_MAX}
    while pending:
        body = []
        keys = []
        for cid in pending:
            last_pos = pools[cid][-1].get("global_position") if pools[cid] else None
            if last_pos is None:
                continue
            keys.append(cid)
            body.append({})
            body.append({
                "size": CHUNK_OBJECTS_PAGE_SIZE,
                "query": {"bool": {"filter": [
                    {"term":  {"parent_chunk_id": cid}},
                    {"range": {"global_position": {"gt": last_pos}}},
                    *([{"term": {"document_id": document_id}}] if document_id else []),
                    *([{"term": {"tenant_id": tenant_id}}] if tenant_id else []),
                    *([{"term": {"project_id": project_id}}] if project_id else []),
                ]}},
                "sort": [{"global_position": "asc"}],
            })
        if not keys:
            break
        try:
            resp = _get_os().msearch(body=body, index=SEMANTIC_OBJECTS_INDEX)
            responses = resp.get("responses", [])
        except Exception as exc:
            logger.warning("[Evidence Grounder] chunk pool pagination failed: %s", exc)
            break
        next_pending: set[str] = set()
        for cid, r in zip(keys, responses):
            more = [h["_source"] for h in r.get("hits", {}).get("hits", [])]
            pools[cid].extend(more)
            if len(more) == CHUNK_OBJECTS_PAGE_SIZE and len(pools[cid]) < CHUNK_OBJECTS_MAX:
                next_pending.add(cid)
        pending = next_pending
    return pools


# ── Grounding ─────────────────────────────────────────────────────────────────────────

_DEDUP_NORMALIZE_RE = re.compile(r"[^\w]+")

def _normalize_for_dedup(text: str) -> str:
    return _DEDUP_NORMALIZE_RE.sub(" ", text.lower()).strip()


def _parent_object_id(obj: dict) -> str:
    return re.sub(r"_s\d+$", "", obj.get("object_id") or "")


def _hierarchy_related(a: dict, b: dict) -> bool:
    """True when one object is literally the other's container (sentence↔own paragraph).
    Table cell/row pairs are NOT collapsed here — same table position but distinct object
    types are kept as separate candidates and left to S6 llm_verify to judge independently."""
    ida, idb = a.get("object_id"), b.get("object_id")
    if not ida or not idb or ida == idb:
        return False
    return _parent_object_id(a) == idb or _parent_object_id(b) == ida


def _objects_for_span(span_n: str, source: str, norm_pool: list[tuple[dict, str]]) -> tuple[list[dict], str, float] | None:
    """Most-specific object(s) containing one normalized span. None if unresolvable/ambiguous."""
    hits = [(o, t) for o, t in norm_pool if span_n in t]
    method, confidence = f"{source}_span", _LITERAL_SPAN_CONFIDENCE
    if not hits and len(span_n) >= GROUNDING_COVERED_MIN_CHARS:
        hits = [(o, t) for o, t in norm_pool if len(t) >= GROUNDING_COVERED_MIN_CHARS and t in span_n]
        method, confidence = f"{source}_span_covered", _COVERED_SPAN_CONFIDENCE
    if not hits:
        return None
    hits.sort(key=lambda ot: (_GROUND_TYPE_RANK.get(ot[0].get("type"), 7), ot[0].get("global_position") or 0))
    specific: list[dict] = []
    for o, _ in hits:
        if not any(_hierarchy_related(o, s) for s in specific):
            specific.append(o)
    if len(specific) > GROUNDING_SPAN_AMBIGUITY_LIMIT:
        return None
    return specific, method, confidence


def _ground_by_spans(matches: list[dict], pool: list[dict]) -> list[tuple[dict, dict]]:
    """Every distinct literal/regex span → the object(s) it genuinely grounds to. Spans that
    land in the same object merge into ONE candidate carrying every supporting match_id —
    evidence multiplicity is tracked, but it does not multiply candidates on its own."""
    if not matches:
        return []
    norm_pool = [(o, _normalize_for_dedup(o.get("text") or "")) for o in pool]
    norm_pool = [(o, t) for o, t in norm_pool if t]
    if not norm_pool:
        return []

    by_object: dict[str, dict] = {}   # object_id -> {"obj", "method", "confidence", "evidence_matches"}
    # Longest spans first so a short sub-phrase doesn't needlessly widen an already-specific pick.
    ordered = sorted(matches, key=lambda m: -len(m.get("text") or ""))
    for m in ordered:
        span_n = _normalize_for_dedup(m.get("text") or "")
        if not span_n:
            continue
        match_id = m.get("match_id") or f"{m.get('source', 'literal')}_{m.get('start')}_{m.get('end')}"
        resolved = _objects_for_span(span_n, m.get("source") or "literal", norm_pool)
        if not resolved:
            continue
        objs, method, confidence = resolved
        for o in objs:
            oid = o.get("object_id")
            entry = by_object.get(oid)
            if entry is None:
                by_object[oid] = {
                    "obj": o, "method": method, "confidence": confidence,
                    "evidence_matches": [match_id],
                }
            else:
                entry["evidence_matches"].append(match_id)
                entry["confidence"] = max(entry["confidence"], confidence)

    picks = sorted(by_object.values(), key=lambda e: e["obj"].get("global_position") or 0)
    if len(picks) > GROUNDING_MAX_SPAN_OBJECTS:
        logger.info("[Evidence Grounder] span objects truncated %d -> %d", len(picks), GROUNDING_MAX_SPAN_OBJECTS)
        picks = picks[:GROUNDING_MAX_SPAN_OBJECTS]
    return [
        (e["obj"], {"method": e["method"], "confidence": e["confidence"], "evidence_matches": e["evidence_matches"]})
        for e in picks
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(map(operator.mul, a, a)))
    nb = math.sqrt(sum(map(operator.mul, b, b)))
    return sum(map(operator.mul, a, b)) / (na * nb) if na and nb else 0.0


def _ground_by_local_score(ci: dict, pool: list[dict]) -> list[tuple[dict, dict]]:
    """No exact span: rank this chunk's own objects against the CI and keep the top few."""
    sentence_parents = {_parent_object_id(o) for o in pool if o.get("type") == "sentence"}
    eligible = [
        o for o in pool
        if o.get("type") in _LOCAL_SCORE_TYPES
        and (o.get("text") or "").strip()
        and not (o.get("type") == "paragraph" and o.get("object_id") in sentence_parents)
    ]
    if not eligible:
        return []

    ci_vec = (ci.get("embedding") or {}).get("dense_vector") or []
    use_vector = bool(ci_vec) and 2 * sum(1 for o in eligible if o.get("dense_vector")) >= len(eligible)
    if use_vector:
        method, floor = "local_vector", GROUNDING_MIN_VECTOR_SCORE
        scored = [(_cosine(ci_vec, o["dense_vector"]), o) for o in eligible if o.get("dense_vector")]
    else:
        method, floor = "local_lexical", GROUNDING_MIN_LEXICAL_SCORE
        ci_text = " ".join((ci.get("normalization") or {}).get("tokens") or []) or ci.get("knownCI") or ""
        ci_tokens = {t for t in _normalize_for_dedup(ci_text).split() if len(t) >= 3}
        if not ci_tokens:
            return []
        scored = [
            (len(ci_tokens & set(_normalize_for_dedup(o["text"]).split())) / len(ci_tokens), o)
            for o in eligible
        ]

    scored.sort(key=lambda s: (-s[0], s[1].get("global_position") or 0))
    if not scored or scored[0][0] < floor:
        return []
    best = scored[0][0]
    return [
        (o, {"method": method, "confidence": round(s, 4), "evidence_matches": []})
        for s, o in scored[:GROUNDING_MAX_LOCAL_OBJECTS]
        if s >= floor and s >= best - GROUNDING_SCORE_MARGIN
    ]


def _ground_chunk_candidate(candidate: dict, ci: dict, pool: list[dict]) -> tuple[list[tuple[dict, dict]], str | None]:
    """Ground a chunk-level candidate. Returns (picks, unresolved_reason)."""
    if not any((o.get("text") or "").strip() for o in pool):
        return [], "no_semantic_objects_for_chunk"
    picks = _ground_by_spans(candidate.get("literal_matches") or [], pool)
    if not picks:
        picks = _ground_by_local_score(ci, pool)
    return (picks, None) if picks else ([], "no_reliable_anchor")
