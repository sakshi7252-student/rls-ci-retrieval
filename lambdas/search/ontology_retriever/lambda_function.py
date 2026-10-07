"""
Search Pipeline — Stage 2d: Ontology Retriever
================================================
Searches using the CI's curated synonyms and abbreviation expansions.
Best for:  CLINICAL_ROLE ("PI" → "Principal Investigator" → "Lead Investigator").

Input:  classified search request  (ci must have "ontology")
Output: { "retriever": "ontology", "hits": list[Hit] }
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENSEARCH_ENDPOINT    = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_INDEX       = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
AWS_REGION             = os.environ.get("AWS_REGION", "us-east-1")
TOP_K                  = int(os.environ.get("RETRIEVER_TOP_K", "10"))
TIE_BUFFER             = int(os.environ.get("RETRIEVER_TIE_BUFFER", "15"))


def _adaptive_k(page_count: int, base_k: int = 10) -> int:
    if page_count <= 0:    return base_k
    if page_count < 500:   return base_k
    if page_count < 3_000: return max(base_k, 25)
    if page_count < 10_000: return max(base_k, 50)
    return max(base_k, 75)


def _with_ties(sorted_hits: list[dict], k: int) -> list[dict]:
    if len(sorted_hits) <= k:
        return sorted_hits
    cutoff = sorted_hits[k - 1]["score"]
    result = sorted_hits[:k]
    for h in sorted_hits[k:]:
        if h["score"] == cutoff:
            result.append(h)
        else:
            break
    return result

from shared.opensearch_client import get_opensearch_client

def _get_os():
    return get_opensearch_client()


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Ontology Retriever] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Ontology Retriever] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Ontology Retriever] done search_id=%s hits=%d", search_id, len(result["hits"]))
    return result


def _process(req: dict) -> dict:
    ontology    = req["ci"].get("ontology", {})
    document_id = req.get("document_id")
    tenant = req.get("tenant")
    project_id = req.get("project_id")
    tenant_id = tenant.get("tenant_id")

    # Use canonical identity fields — not raw ontology expansions.
    # Canonical names are already resolved; searching for them in normalized_text
    # gives high-precision matches without the noise of raw abbreviations.
    ci_obj      = req.get("ci", {})
    search_terms: list[str] = []

    def _collect(identity: dict) -> None:
        for v in identity.values():
            if isinstance(v, list):
                search_terms.extend(str(x).lower() for x in v if x)
            elif isinstance(v, str) and v:
                search_terms.append(v.lower())

    _collect(ci_obj.get("clinical_identity",   {}))
    _collect(ci_obj.get("treatment_identity",  {}))
    _collect(ci_obj.get("endpoint_identity",   {}))
    _collect(ci_obj.get("population_identity", {}))

    # Deduplicate preserving order
    seen: set[str] = set()
    search_terms = [t for t in search_terms if t not in seen and not seen.add(t)]

    if not search_terms:
        return {"retriever": "ontology", "hits": []}

    page_count = int(req.get("document_page_count", 0))
    k          = _adaptive_k(page_count, TOP_K)

    # Object hits first (precision + geometry), chunk hits fill remaining recall —
    # same split as bm25_retriever, since this is keyword/synonym matching too and
    # has no structural need for raw chunk text (unlike literal/regex/ner).
    obj_hits   = _ontology_search_objects(search_terms, document_id, tenant_id=tenant_id, project_id=project_id, k=k)
    chunk_hits = _ontology_search_chunks(search_terms, document_id, tenant_id=tenant_id, project_id=project_id, k=k)

    seen_chunks: set[str] = set()
    hits: list[dict] = []
    for h in obj_hits:
        hits.append(h)
        seen_chunks.add(h["chunk_id"])
    for h in chunk_hits:
        if h["chunk_id"] not in seen_chunks:
            hits.append(h)
            seen_chunks.add(h["chunk_id"])
    hits.sort(key=lambda x: x["score"], reverse=True)

    return {
        "retriever": "ontology",
        "hits":      _with_ties(hits, k),
    }


def _ontology_search_objects(terms: list[str], document_id: str | None, tenant_id: str | None = None, project_id: str | None = None, k: int = TOP_K) -> list[dict]:
    """Ontology/synonym match against semantic-objects — gives a direct matched_object + geometry."""
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []
    if tenant_id:
        filter_clause.append({"term": {"tenant_id": tenant_id}})
    if project_id:
        filter_clause.append({"term": {"project_id": project_id}})

    should_clauses = [
        {"match": {"text": {"query": term, "boost": 1.0}}}
        for term in terms
    ]

    body = {
        "size": k + TIE_BUFFER,
        "query": {
            "bool": {
                "filter": filter_clause,
                "should": should_clauses,
                "minimum_should_match": 1,
            }
        },
        "_source": [
            "object_id", "parent_chunk_id", "document_id",
            "position", "global_position", "document_position", "type", "text", "page", "bbox", "geometry",
            "list_id", "list_level", "list_label", "list_number_format",
            "table_id", "row_index", "row_start", "col_start", "row_span", "col_span",
            "entities",
            "section_category", "heading_path", "semantic_path", "section_confidence",
            "prev_sentence_text", "next_sentence_text", "paragraph_text",
            "facts", "own_facts", "effective_facts", "inherited_slots", "slot_provenance",
            "study_context", "statement_type", "object_subtype", "modality",
            "clinical_relations",
            "clinical_identity", "treatment_identity", "endpoint_identity",
            "population_identity", "temporal_context",
            "study_hierarchy", "negated_slots", "clinical_signature",
            "statistical_identity",
        ],
    }

    try:
        resp = _get_os().search(index=SEMANTIC_OBJECTS_INDEX, body=body)
    except Exception as exc:
        logger.warning("[Ontology Retriever] semantic-objects search failed: %s", exc)
        return []

    return _parse_object_hits(resp)


def _ontology_search_chunks(terms: list[str], document_id: str | None, tenant_id: str | None = None, project_id: str | None = None, k: int = TOP_K) -> list[dict]:
    """Ontology/synonym fallback against document-chunks for broad recall."""
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []
    if tenant_id:
        filter_clause.append({"term": {"tenant_id": tenant_id}})
    if project_id:
        filter_clause.append({"term": {"project_id": project_id}})

    should_clauses = [
        {"match": {"normalized_text": {"query": term, "boost": 1.0}}}
        for term in terms
    ]

    body = {
        "size": k + TIE_BUFFER,
        "query": {
            "bool": {
                "filter": filter_clause,
                "should": should_clauses,
                "minimum_should_match": 1,
            }
        },
        "_source": ["chunk_id", "document_id", "page_start", "page_end", "raw_text"],
    }

    try:
        resp = _get_os().search(index=OPENSEARCH_INDEX, body=body)
    except Exception as exc:
        logger.warning("[Ontology Retriever] document-chunks search failed: %s", exc)
        return []

    return _parse_chunk_hits(resp, terms)


def _parse_object_hits(resp: dict) -> list[dict]:
    hits = []
    for h in resp.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        hits.append({
            "chunk_id":       src.get("parent_chunk_id", ""),
            "score":          round(h.get("_score", 0.0), 4),
            "page_start":     src.get("page", 0),
            "page_end":       src.get("page", 0),
            "snippet":        src.get("text", "")[:200],
            "matched_object": {
                "object_id":          src["object_id"],
                "parent_chunk_id":    src["parent_chunk_id"],
                "document_id":        src["document_id"],
                "position":           src.get("position"),
                "global_position":    src.get("global_position"),
                "document_position":  src.get("document_position"),
                "type":               src["type"],
                "list_id":            src.get("list_id"),
                "list_level":         src.get("list_level"),
                "list_label":         src.get("list_label"),
                "list_number_format": src.get("list_number_format"),
                "table_id":           src.get("table_id", src.get("table_key")),
                "row_index":          src.get("row_index", src.get("row_start")),
                "row_start":          src.get("row_start"),
                "col_start":          src.get("col_start"),
                "row_span":           src.get("row_span"),
                "col_span":           src.get("col_span"),
                "text":               src["text"],
                "page":               src.get("page"),
                "bbox":               src.get("bbox", []),
                "geometry":           src.get("geometry") or {},
                "entities":           src.get("entities", []),
                "section_category":   src.get("section_category"),
                "heading_path":       src.get("heading_path"),
                "semantic_path":      src.get("semantic_path"),
                "section_confidence": src.get("section_confidence"),
                "prev_sentence_text": src.get("prev_sentence_text"),
                "next_sentence_text": src.get("next_sentence_text"),
                "paragraph_text":     src.get("paragraph_text"),
                "facts":               src.get("facts", {}),
                "own_facts":           src.get("own_facts", {}),
                "study_context":       src.get("study_context", "GENERAL"),
                "statement_type":      src.get("statement_type"),
                "clinical_relations":  src.get("clinical_relations", []),
                "effective_facts":     src.get("effective_facts", {}),
                "inherited_slots":     src.get("inherited_slots", []),
                "slot_provenance":     src.get("slot_provenance", {}),
                "clinical_identity":   src.get("clinical_identity", {}),
                "treatment_identity":  src.get("treatment_identity", {}),
                "endpoint_identity":   src.get("endpoint_identity", {}),
                "population_identity": src.get("population_identity", {}),
                "temporal_context":    src.get("temporal_context", {}),
                "modality":            src.get("modality", "GENERAL"),
                "object_subtype":      src.get("object_subtype", "GENERAL"),
                "study_hierarchy":     src.get("study_hierarchy", {}),
                "negated_slots":       src.get("negated_slots", []),
                "clinical_signature":  src.get("clinical_signature", {}),
                "statistical_identity": src.get("statistical_identity", {}),
            },
        })
    return hits


def _parse_chunk_hits(resp: dict, terms: list[str]) -> list[dict]:
    hits = []
    for h in resp.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        hits.append({
            "chunk_id":   src.get("chunk_id", h["_id"]),
            "score":      round(h.get("_score", 0.0), 4),
            "page_start": src.get("page_start", 0),
            "page_end":   src.get("page_end",   0),
            "snippet":    src.get("raw_text", "")[:200],
            "retrieval_evidence": {"retriever": "ontology", "search_terms": terms},
        })
    return hits
