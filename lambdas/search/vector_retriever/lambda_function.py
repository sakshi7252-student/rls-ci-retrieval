"""
Search Pipeline — Stage 2c: Vector Retriever
=============================================
Cosine similarity search using the CI's dense embedding.
Best for:  CLINICAL_ROLE, PHRASE, semantic variants.

Strategy: fetch all document chunks, compute cosine similarity in Python.
In production: use knn_vector mapping + OpenSearch k-NN plugin for efficiency.

Input:  classified search request  (ci must have "embedding.dense_vector")
Output: { "retriever": "vector", "hits": list[Hit] }
"""

from __future__ import annotations

import logging
import math
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
# Score-decay threshold: keep all hits with cosine >= top_cosine * ratio.
# 0.0 = disabled → adaptive-k + tie-inclusion.
VECTOR_SCORE_RATIO     = float(os.environ.get("VECTOR_SCORE_RATIO", "0.0"))
VECTOR_MAX_HITS        = int(os.environ.get("VECTOR_MAX_HITS", "100"))
FETCH_SIZE             = int(os.environ.get("VECTOR_FETCH_SIZE", "100"))
OPENSEARCH_MAXSIZE  = int(os.environ.get("OPENSEARCH_MAXSIZE", "256"))
VECTOR_SEARCH_WORKERS = int(os.environ.get("VECTOR_SEARCH_WORKERS", "1"))
# Comma-separated object types to exclude from semantic-objects vector search.
# Useful for ablation: VECTOR_EXCLUDE_TYPES=sentence  → Variant B (no sentence vectors)
#                      VECTOR_EXCLUDE_TYPES=sentence,heading → Variant C
# Empty (default) = no exclusion = current behaviour.
_raw_exclude = os.environ.get("VECTOR_EXCLUDE_TYPES", "")
VECTOR_EXCLUDE_TYPES: list[str] = [
    t.strip() for t in _raw_exclude.split(",") if t.strip()
]
# Extra candidates pulled from the on_disk-quantized (32x) index before
# rescoring against full-precision vectors, to offset quantization recall loss.
VECTOR_OVERSAMPLE_FACTOR = float(os.environ.get("VECTOR_OVERSAMPLE_FACTOR", "5.0"))

from shared.opensearch_client import get_opensearch_client

def _get_os():
    return get_opensearch_client()


# ─── Adaptive retrieval depth ─────────────────────────────────────────────────

def _adaptive_k(page_count: int, base_k: int = 10) -> int:
    """Scale retrieval depth with document size.

    Small documents (<500 pages) keep base TOP_K.  Large clinical documents
    (CSRs, dossiers) scale up so evidence beyond rank-10 is reachable.
    Set document_page_count=0 (or omit) to fall back to base_k.
    """
    if page_count <= 0:
        return base_k
    if page_count < 500:
        return base_k
    if page_count < 3_000:
        return max(base_k, 25)
    if page_count < 10_000:
        return max(base_k, 50)
    return max(base_k, 75)


def _with_ties(sorted_hits: list[dict], k: int) -> list[dict]:
    """Return top-k hits plus any additional hits that tie the score at rank k."""
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


def _score_decay_filter(sorted_hits: list[dict], ratio: float, max_hits: int) -> list[dict]:
    """Keep all hits within `ratio` of the top score, up to max_hits."""
    if not sorted_hits:
        return sorted_hits
    threshold = sorted_hits[0]["score"] * ratio
    return [h for h in sorted_hits if h["score"] >= threshold][:max_hits]


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Vector Retriever] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Vector Retriever] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Vector Retriever] done search_id=%s hits=%d", search_id, len(result["hits"]))
    return result


def _process(req: dict) -> dict:
    ci_embedding = req["ci"].get("embedding", {}).get("dense_vector", [])
    document_id  = req.get("document_id")
    tenant = req.get("tenant")
    project_id = req.get("project_id")
    tenant_id = tenant.get("tenant_id")

    if not ci_embedding:
        logger.warning("[Vector Retriever] no embedding on CI — returning empty")
        return {"retriever": "vector", "hits": []}

    page_count = int(req.get("document_page_count", 0))
    k          = _adaptive_k(page_count, TOP_K)

    # Lanes 1-3 are independent — run concurrently to eliminate serial latency
    from concurrent.futures import ThreadPoolExecutor as _TPE
    from contextvars import copy_context as _copy_context
    import time as _time
    # copy_context() per submit (fresh copy each time) propagates the caller's
    # [tenant=/document=/search=] ContextVars — a single reused Context can't be entered
    # by more than one thread at once and raises "cannot enter context".
    with _TPE(max_workers=VECTOR_SEARCH_WORKERS) as _pool:
        _ts_obj  = _time.perf_counter()
        _f_obj   = _pool.submit(_copy_context().run, _vector_search_objects,         ci_embedding, document_id, tenant_id=tenant_id, project_id=project_id, k=k)
        _ts_head = _time.perf_counter()
        _f_head  = _pool.submit(_copy_context().run, _vector_search_objects_heading, ci_embedding, document_id, tenant_id=tenant_id, project_id=project_id, k=k)
        _ts_chunk = _time.perf_counter()
        _f_chunk = _pool.submit(_copy_context().run, _vector_search_chunks,          ci_embedding, document_id, tenant_id=tenant_id, project_id=project_id, k=k)
        obj_hits   = _f_obj.result();   _te_obj   = _time.perf_counter()
        head_hits  = _f_head.result();  _te_head  = _time.perf_counter()
        chunk_hits = _f_chunk.result(); _te_chunk = _time.perf_counter()
    _sub_timings = {
        "vector_body":    round(_te_obj   - _ts_obj,   3),
        "vector_heading": round(_te_head  - _ts_head,  3),
        "vector_chunk":   round(_te_chunk - _ts_chunk, 3),
    }

    # Merge: body hits first (most precise), then heading hits, then chunk fallback
    seen_chunks: set[str] = set()
    hits: list[dict] = []
    for h in obj_hits:
        hits.append(h)
        seen_chunks.add(h["chunk_id"])
    for h in head_hits:
        if h["chunk_id"] not in seen_chunks:
            hits.append(h)
            seen_chunks.add(h["chunk_id"])
    for h in chunk_hits:
        if h["chunk_id"] not in seen_chunks:
            hits.append(h)
            seen_chunks.add(h["chunk_id"])

    hits.sort(key=lambda x: x["score"], reverse=True)

    ratio = VECTOR_SCORE_RATIO
    if ratio > 0.0 and hits:
        hits = _score_decay_filter(hits, ratio, VECTOR_MAX_HITS)
    else:
        hits = _with_ties(hits, k)

    return {
        "retriever":    "vector",
        "hits":         hits,
        "_sub_timings": _sub_timings,
    }


def _vector_search_objects(ci_embedding: list[float], document_id: str | None, tenant_id: str | None = None, project_id: str | None = None, k: int = TOP_K) -> list[dict]:
    """
    Search semantic-objects index by body dense_vector.
    Returns hits with full matched_object metadata.
    """
    filter_clause   = [{"term": {"document_id": document_id}}] if document_id else []
    if tenant_id:
        filter_clause.append({"term": {"tenant_id": tenant_id}})
    if project_id:
        filter_clause.append({"term": {"project_id": project_id}})
    
    must_not_clause = ([{"terms": {"type": VECTOR_EXCLUDE_TYPES}}]
                       if VECTOR_EXCLUDE_TYPES else [])

    body = {
        "size": k + TIE_BUFFER,
        "query": {
            "knn": {
                "dense_vector": {
                    "vector": ci_embedding,
                    "k":      k + TIE_BUFFER,
                    "rescore": {"oversample_factor": VECTOR_OVERSAMPLE_FACTOR},
                    **({
                        "filter": {
                            "bool": {
                                "filter": filter_clause,
                                **({"must_not": must_not_clause} if must_not_clause else {}),
                            }
                        }
                    } if filter_clause or must_not_clause else {}),
                }
            }
        },
        "_source": [
            "object_id", "parent_chunk_id", "document_id",
            "position", "global_position", "type", "text", "page", "bbox", "geometry",
    "list_id", "list_level", "list_label", "list_number_format",
    "table_id", "row_index", "row_start", "col_start", "row_span", "col_span",
            "section_category", "heading_path", "semantic_path",
            "section_confidence", "document_position",
            "chunk_idx", "parent_chunk_idx", "prev_chunk_idx", "next_chunk_idx",
            "effective_facts", "clinical_identity",
            "treatment_identity", "endpoint_identity", "population_identity",
            "modality", "study_context", "statement_type",
        ],
    }

    try:
        resp = _get_os().search(index=SEMANTIC_OBJECTS_INDEX, body=body)
    except Exception as exc:
        logger.warning("[Vector Retriever] semantic-objects knn failed: %s", exc)
        return []

    return [
        {
            "chunk_id":       h["_source"]["parent_chunk_id"],
            "score":          round(h["_score"], 4),
            "page_start":     h["_source"].get("page", 0),
            "page_end":       h["_source"].get("page", 0),
            "snippet":        h["_source"].get("text", "")[:200],
            "retrieved_type": h["_source"].get("type", "unknown"),
            "matched_object": _build_matched_object(h["_source"]),
        }
        for h in resp.get("hits", {}).get("hits", [])
        if h.get("_source", {}).get("parent_chunk_id")
    ]


def _build_matched_object(s: dict) -> dict:
    """Build the full matched_object dict from a semantic-objects _source."""
    return {
        "object_id":          s["object_id"],
        "parent_chunk_id":    s["parent_chunk_id"],
        "document_id":        s["document_id"],
        "position":           s.get("position"),
        "global_position":    s.get("global_position"),
        "type":               s["type"],
        "text":               s["text"],
        "page":               s.get("page"),
        "bbox":               s.get("bbox", []),
        # Canonical geometry from extraction/chunk construction.
        # Retriever only carries it forward; it does not infer or rename it.
        "geometry": s.get("geometry") or {},
        "entities":           s.get("entities", []),
        # Section / semantic metadata
        "section_category":   s.get("section_category"),
        "heading_path":       s.get("heading_path"),
        "semantic_path":      s.get("semantic_path"),
        "section_confidence": s.get("section_confidence"),
        "document_position":  s.get("document_position"),
        # Sentence neighbour context
        "prev_sentence_text": s.get("prev_sentence_text"),
        "next_sentence_text": s.get("next_sentence_text"),
        "paragraph_text":     s.get("paragraph_text"),
        # Chunk adjacency
        "chunk_idx":          s.get("chunk_idx"),
        "parent_chunk_idx":   s.get("parent_chunk_idx"),
        "prev_chunk_idx":     s.get("prev_chunk_idx"),
        "next_chunk_idx":     s.get("next_chunk_idx"),
        # Semantic layer: canonical identity fields (no raw entities / neighbor text)
        "effective_facts":     s.get("effective_facts", {}),
        "clinical_identity":   s.get("clinical_identity", {}),
        "treatment_identity":  s.get("treatment_identity", {}),
        "endpoint_identity":   s.get("endpoint_identity", {}),
        "population_identity": s.get("population_identity", {}),
        "modality":            s.get("modality", "GENERAL"),
        "study_context":       s.get("study_context", "GENERAL"),
        "statement_type":      s.get("statement_type"),
    }


def _vector_search_objects_heading(ci_embedding: list[float], document_id: str | None, tenant_id: str | None = None, project_id: str | None = None, k: int = TOP_K) -> list[dict]:
    """
    Search semantic-objects index by heading_dense_vector.

    Retrieves objects whose *heading* is semantically close to the CI query,
    even when the object body text doesn’t directly mention the topic.
    Scores are slightly penalised (0.90×) so body-vector hits take priority
    when both lanes return the same chunk.
    """
    if not ci_embedding:
        return []
    filter_clause   = [{"term": {"document_id": document_id}}] if document_id else []
    if tenant_id:
        filter_clause.append({"term": {"tenant_id": tenant_id}})
    if project_id:
        filter_clause.append({"term": {"project_id": project_id}})
    must_not_clause = ([{"terms": {"type": VECTOR_EXCLUDE_TYPES}}]
                       if VECTOR_EXCLUDE_TYPES else [])
    body = {
        "size": k + TIE_BUFFER,
        "query": {
            "knn": {
                "heading_dense_vector": {
                    "vector": ci_embedding,
                    "k":      k + TIE_BUFFER,
                    "rescore": {"oversample_factor": VECTOR_OVERSAMPLE_FACTOR},
                    **({"filter": {"bool": {
                        "filter": filter_clause,
                        **({"must_not": must_not_clause} if must_not_clause else {}),
                    }}} if filter_clause or must_not_clause else {}),
                }
            }
        },
        "_source": [
            "object_id", "parent_chunk_id", "document_id",
            "position", "global_position", "type", "text", "page", "bbox", "geometry",
    "list_id", "list_level", "list_label", "list_number_format",
    "table_id", "row_index", "row_start", "col_start", "row_span", "col_span",
            "section_category", "heading_path", "semantic_path",
            "section_confidence", "document_position",
            "chunk_idx", "parent_chunk_idx", "prev_chunk_idx", "next_chunk_idx",
            "effective_facts", "clinical_identity",
            "treatment_identity", "endpoint_identity", "population_identity",
            "modality", "study_context", "statement_type",
        ],
    }
    try:
        resp = _get_os().search(index=SEMANTIC_OBJECTS_INDEX, body=body)
    except Exception as exc:
        logger.warning("[Vector Retriever] heading knn failed: %s", exc)
        return []

    return [
        {
            "chunk_id":       h["_source"]["parent_chunk_id"],
            "score":          round(h["_score"] * 0.90, 4),
            "page_start":     h["_source"].get("page", 0),
            "page_end":       h["_source"].get("page", 0),
            "snippet":        h["_source"].get("text", "")[:200],
            "retrieved_type": h["_source"].get("type", "unknown"),
            "matched_object": _build_matched_object(h["_source"]),
        }
        for h in resp.get("hits", {}).get("hits", [])
        if h.get("_source", {}).get("parent_chunk_id")
    ]


def _vector_search_chunks(ci_embedding: list[float], document_id: str | None, tenant_id: str | None = None, project_id: str | None = None, k: int = TOP_K) -> list[dict]:
    """KNN search on document-chunks dense_vector (chunk-level fallback)."""
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []
    if tenant_id:
        filter_clause.append({"term": {"tenant_id": tenant_id}})
    if project_id:
        filter_clause.append({"term": {"project_id": project_id}})

    body = {
        "size": k + TIE_BUFFER,
        "query": {
            "knn": {
                "dense_vector": {
                    "vector": ci_embedding,
                    "k":      k + TIE_BUFFER,
                    **({
                        "filter": {"bool": {"filter": filter_clause}}
                    } if filter_clause else {}),
                }
            }
        },
        "_source": ["chunk_id", "document_id", "page_start", "page_end", "raw_text"],
    }

    try:
        resp = _get_os().search(index=OPENSEARCH_INDEX, body=body)
    except Exception as exc:
        logger.warning("[Vector Retriever] document-chunks knn failed: %s", exc)
        return []

    return [
        {
            "chunk_id":       h["_source"]["chunk_id"],
            "score":          round(h["_score"], 4),
            "page_start":     h["_source"].get("page_start", 0),
            "page_end":       h["_source"].get("page_end",   0),
            "snippet":        h["_source"].get("raw_text", "")[:200],
            "retrieved_type": "chunk",
        }
        for h in resp.get("hits", {}).get("hits", [])
        if h.get("_source", {}).get("chunk_id")
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(y * y for y in b))
    denom = na * nb
    return dot / denom if denom else 0.0
