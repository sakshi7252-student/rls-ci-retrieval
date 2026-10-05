"""
Search Pipeline — Stage 4: Context Expander
=============================================
For each candidate chunk, fetches the chunk's text from OpenSearch and
retrieves neighbouring chunks (±1 page range) to provide surrounding context.

Input:  aggregated search request  (must have "candidates")
Appends: "expanded_candidates": list[ExpandedCandidate]

ExpandedCandidate schema
-------------------------
{
    "chunk_id":      str,
    "page_start":    int,
    "page_end":      int,
    "sources":       list[str],
    "agg_score":     float,
    "context": {
        "prev_text":    str,   # last 500 chars of preceding chunk (if any)
        "current_text": str,   # full raw_text of this chunk
        "next_text":    str,   # first 500 chars of following chunk (if any)
    }
}
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENSEARCH_ENDPOINT    = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_INDEX       = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
AWS_REGION             = os.environ.get("AWS_REGION", "us-east-1")
CONTEXT_CHARS          = int(os.environ.get("CONTEXT_CHARS", "500"))
CONTEXT_WINDOW         = int(os.environ.get("CONTEXT_WINDOW", "3"))   # objects before+after match
OPENSEARCH_MAXSIZE  = int(os.environ.get("OPENSEARCH_MAXSIZE", "256"))
# Maximum semantic objects/characters included in table-aware verifier context.
# The matched table object is never replaced; this only augments its context.
TABLE_CONTEXT_MAX_OBJECTS = int(os.environ.get("TABLE_CONTEXT_MAX_OBJECTS", "200"))
TABLE_CONTEXT_MAX_CHARS = int(os.environ.get("TABLE_CONTEXT_MAX_CHARS", "16000"))
CONTEXT_EXPANDER_WORKERS = int(os.environ.get("CONTEXT_EXPANDER_WORKERS", "1"))

def _get_os():
    from shared.opensearch_client import get_opensearch_client
    return get_opensearch_client()


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Context Expander] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Context Expander] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Context Expander] done search_id=%s expanded=%d",
                search_id, len(result["expanded_candidates"]))
    
    return result


def _process(req: dict) -> dict:
    candidates  = req.get("candidates", [])
    document_id = req.get("document_id") or ""
    tenant = req.get("tenant")
    project_id = req.get("project_id")
    tenant_id = tenant.get("tenant_id")

    if not candidates:
        return {**req, "expanded_candidates": []}

    # ── Phase 1: collect all lookup keys ──────────────────────────────────────
    primary_ids:  list[str]                          = []
    idx_needed:   set[int]                           = set()
    ctx_keys:     list[tuple[str, int | None]]       = []
    _ctx_key_set: set[tuple[str, int | None]]        = set()
    page_lookups: set[tuple[str, int]]               = set()
    table_ids: set[str]                              = set()

    for c in candidates:
        cid      = c.get("chunk_id", "")
        obj_meta = c.get("matched_object") or {}
        primary_ids.append(cid)
        for key in ("prev_chunk_idx", "next_chunk_idx", "parent_chunk_idx"):
            idx = obj_meta.get(key)
            if idx is not None:
                idx_needed.add(idx)
        # Table hits need table-aware context. table_id is the canonical relationship
        # carried by the indexed object; do not rediscover the table from text.
        table_id = obj_meta.get("table_id")
        if table_id and obj_meta.get("type") in {
            "table_header", "table_row", "table_cell", "list_item"
        }:
            table_ids.add(str(table_id))

        ctx_key = (cid, obj_meta.get("global_position"))
        if ctx_key not in _ctx_key_set:
            ctx_keys.append(ctx_key)
            _ctx_key_set.add(ctx_key)
        # Collect page-range neighbor keys for candidates without chunk_idx adjacency
        if obj_meta.get("prev_chunk_idx") is None:
            page_lookups.add(("page_end",   c.get("page_start", 0) - 1))
        if obj_meta.get("next_chunk_idx") is None:
            page_lookups.add(("page_start", c.get("page_end",   0) + 1))

    # ── Phase 2: run all 4 fetches concurrently (they're independent) ──────────
    from concurrent.futures import ThreadPoolExecutor as _TPE
    deduped_ids = list(dict.fromkeys(primary_ids))
    with _TPE(max_workers=CONTEXT_EXPANDER_WORKERS) as _pool:
        _f_chunk = _pool.submit(_mget_chunks, deduped_ids)
        _f_idx   = _pool.submit(_msearch_by_idx, document_id, list(idx_needed), tenant_id=tenant_id, project_id=project_id)
        _f_ctx   = _pool.submit(_fetch_context_objects_merged, document_id, ctx_keys, tenant_id=tenant_id, project_id=project_id)
        _f_page  = _pool.submit(_msearch_neighbors_by_page, document_id, list(page_lookups), tenant_id=tenant_id, project_id=project_id)
        _f_table = _pool.submit(_fetch_table_context, document_id, sorted(table_ids), tenant_id=tenant_id, project_id=project_id)
        chunk_cache: dict[str, dict]         = _f_chunk.result()
        idx_cache:   dict[int, str]          = _f_idx.result()
        ctx_cache:   dict[tuple, list[dict]] = _f_ctx.result()
        page_cache:  dict[tuple, str]        = _f_page.result()
        table_cache: dict[str, list[dict]]     = _f_table.result()
    # Second mget pass for any IDs the first mget missed
    missed = [cid for cid in deduped_ids if cid not in chunk_cache]
    if missed:
        chunk_cache.update(_mget_chunks(missed))

    logger.info(
        "[Context Expander] search_id=%s  candidates=%d  idx_lookups=%d"
        "  ctx_queries=%d  chunk_cache_hits=%d/%d",
        req.get("search_id"), len(candidates), len(idx_needed),
        len(ctx_keys), len(chunk_cache), len(primary_ids),
    )

    # ── Phase 3: expand each candidate from caches ────────────────────────────
    expanded = [_expand(c, document_id, chunk_cache, idx_cache, ctx_cache, page_cache, table_cache) for c in candidates]

    if expanded:
        avg_chars = sum(e.get("current_text_chars", 0) for e in expanded) / len(expanded)
        logger.info(
            "[Context Expander] search_id=%s  avg_context_chars=%.0f  max_context_chars=%d",
            req.get("search_id"), avg_chars,
            max(e.get("current_text_chars", 0) for e in expanded),
        )

    return {**req, "expanded_candidates": expanded}


_OBJECT_TYPE_PRIORITY = {
    "sentence": 4,
    "paragraph": 3,
    "heading": 2,
    "table_header": 2,
    "table_row": 1,
    "list_item": 1,
}

# ── Heading normalizer ────────────────────────────────────────────────────────

_SECTION_NUM_RE = re.compile(r'^[\d\.]+\s+')

def _normalize_heading(heading: str) -> str:
    """Strip leading section numbers from each breadcrumb segment.

    Example: "5.1 Primary Objective > 5.1.2 ORR" → "Primary Objective > ORR"
    Passes through headings that have no numeric prefix unchanged.
    """
    parts   = heading.split(" > ")
    cleaned = [_SECTION_NUM_RE.sub("", seg.strip()) for seg in parts if seg.strip()]
    return " > ".join(cleaned)


# ── Context object sorter ─────────────────────────────────────────────────────

_CTX_TYPE_ORDER = {
    "heading": 1,
    "table_header": 2,
    "paragraph": 3,
    "sentence": 4,
    "table_row": 5,
    "table_cell": 6,
    "list_item": 5,
}

def _sort_context_objects(
    objs:       list[dict],
    matched_id: str | None,
    center_pos: int | None,
) -> list[dict]:
    """Sort context objects for verifier relevance.

    Order:
      1. The matched object itself (always first)
      2. Heading objects — nearest first
      3. Paragraph objects — nearest first
      4. Sentence objects — nearest first
      5. Table / list objects — nearest first
      6. Other types — nearest first
    """
    cp = center_pos or 0
    def _key(o: dict) -> tuple:
        is_match = 0 if o.get("object_id") == matched_id else 1
        tord     = _CTX_TYPE_ORDER.get(o.get("type", ""), 5)
        dist     = abs((o.get("global_position") or 0) - cp)
        return (is_match, tord, dist)
    return sorted(objs, key=_key)


def _expand(
    candidate:   dict,
    document_id: str | None,
    chunk_cache: dict[str, dict],
    idx_cache:   dict[int, str],
    ctx_cache:   dict[tuple, list[dict]],
    page_cache:  dict[tuple, str],
    table_cache: dict[str, list[dict]],
) -> dict:
    chunk_id   = candidate["chunk_id"]
    page_start = candidate.get("page_start", 0)
    page_end   = candidate.get("page_end",   0)

    chunk_doc    = chunk_cache.get(chunk_id, {})
    current_text = chunk_doc.get("raw_text", "")

    # If candidate came from semantic-objects index it already has the matched object
    matched_obj = candidate.get("matched_object")   # set by retriever for object-level hits
    # Capture origin before context_expander potentially assigns matched_obj from chunk context
    origin_is_direct = matched_obj is not None

    # context_strategy describes what actually ended up in current_text.
    # Set here to "chunk_fallback"; updated once we know the matched object type.
    context_strategy = "chunk_fallback"

    # Table-aware context: the matched table object remains the matched object,
    # while the verifier receives the complete table structure around it.
    #
    # IMPORTANT: list_item is also table-aware. Nested list items inside a
    # table_cell carry the same canonical table_id/cell_id relationship as
    # table rows/cells, so a list_item hit must expand to its table context too.
    # Do not treat it as an ordinary document-level list hit.
    table_context_objects: list[dict] = []
    if matched_obj and matched_obj.get("type") in {
        "table_header", "table_row", "table_cell", "list_item"
    }:
        table_id = matched_obj.get("table_id")
        if table_id:
            table_context_objects = table_cache.get(str(table_id), [])
            if table_context_objects:
                current_text = _format_table_context(table_context_objects, matched_obj)
                context_strategy = "table_full"

    # For sentence-level hits, build hierarchical context:
    #   normalized heading → parent paragraph (only when multi-sentence) → 3-sentence window.
    # Preserves the semantic hierarchy (drug / endpoint / disease inherited from heading)
    # while keeping the verifier's context tight and precise.
    if matched_obj and matched_obj.get("type") == "sentence":
        raw_heading = (matched_obj.get("heading_path") or matched_obj.get("semantic_path") or "").strip()
        heading     = _normalize_heading(raw_heading) if raw_heading else ""
        para_text   = (matched_obj.get("paragraph_text") or "").strip()
        sent_text   = (matched_obj.get("text") or "").strip()
        prev_s      = (matched_obj.get("prev_sentence_text") or "").strip()
        next_s      = (matched_obj.get("next_sentence_text") or "").strip()
        sent_window = " ".join(p for p in [prev_s, sent_text, next_s] if p)

        ctx_parts: list[str] = []
        if heading:
            ctx_parts.append(heading)
        # Include parent paragraph only when it contains more than this one sentence
        # (para_text == sent_text means single-sentence paragraph — would be redundant)
        if para_text and para_text != sent_text:
            ctx_parts.append(para_text)
        if sent_window:
            ctx_parts.append(sent_window)

        if ctx_parts:
            current_text     = "\n\n".join(ctx_parts)
            context_strategy = "sentence_hierarchical"

    # Extract adjacency indices set by the section chunker
    obj_meta         = matched_obj or {}
    chunk_idx        = obj_meta.get("chunk_idx")
    prev_chunk_idx   = obj_meta.get("prev_chunk_idx")
    next_chunk_idx   = obj_meta.get("next_chunk_idx")
    parent_chunk_idx = obj_meta.get("parent_chunk_idx")

    # Prefer exact adjacency-index lookup (from msearch cache) over page-range heuristic
    if prev_chunk_idx is not None:
        prev_text = idx_cache.get(prev_chunk_idx, "")
    else:
        prev_text = page_cache.get(("page_end", page_start - 1), "")

    if next_chunk_idx is not None:
        next_text = idx_cache.get(next_chunk_idx, "")
    else:
        next_text = page_cache.get(("page_start", page_end + 1), "")

    parent_text = idx_cache.get(parent_chunk_idx, "") if parent_chunk_idx is not None else ""

    # Context objects from msearch cache (deduped by chunk_id + center_pos)
    center_pos      = obj_meta.get("global_position")
    context_objects = ctx_cache.get((chunk_id, center_pos), [])
    if table_context_objects:
        # Keep table objects together and deduplicate by object_id.
        by_id = {o.get("object_id"): o for o in context_objects if o.get("object_id")}
        for o in table_context_objects:
            oid = o.get("object_id")
            if oid:
                by_id[oid] = o
        context_objects = list(by_id.values())

    # Track why this object was selected — useful for debugging retrieval decisions:
    #   retriever_direct  — retriever set matched_object directly from semantic-objects
    #   literal_match     — via_chunk: chosen because text contained a literal CI match
    #   highest_priority  — via_chunk: chosen by type-priority (sentence > paragraph > …)
    #   chunk_only        — via_chunk: no context objects found at all
    selection_reason    = "retriever_direct" if origin_is_direct else None
    literal_match_count = 0      # set below when selection_reason == "literal_match"

    if matched_obj is None and context_objects:
        lit_texts = [lm["text"].lower()
                     for lm in candidate.get("literal_matches", [])
                     if lm.get("text")]
        if lit_texts:
            # Prefer the object whose text contains the literal match.
            # Among objects that contain the match, prefer the most specific
            # type (sentence > paragraph) so the UI highlights the tightest span.
            # Fall back to type-priority ordering when no object contains the match.
            def _lit_key(o: dict) -> tuple:
                obj_lower = (o.get("text") or "").lower()
                return (int(any(lt in obj_lower for lt in lit_texts)),
                        _OBJECT_TYPE_PRIORITY.get(o.get("type", ""), 0))
            matched_obj         = max(context_objects, key=_lit_key)
            selection_reason    = "literal_match"
            # How many context objects contained the literal — low count = high confidence
            literal_match_count = sum(
                1 for o in context_objects
                if any(lt in (o.get("text") or "").lower() for lt in lit_texts)
            )
        else:
            matched_obj      = max(
                context_objects,
                key=lambda o: _OBJECT_TYPE_PRIORITY.get(o.get("type", ""), 0),
            )
            selection_reason = "highest_priority"
    elif matched_obj is None:
        selection_reason = "chunk_only"   # no context objects available

    # Finalize context_strategy now that matched_obj is settled.
    # "sentence_hierarchical" is already set above; all other types take the object type name.
    if context_strategy == "chunk_fallback" and matched_obj is not None:
        context_strategy = matched_obj.get("type", "unknown")

    # Sort context_objects by verifier relevance:
    # matched object first, then headings, paragraphs, sentences, tables — each nearest first.
    matched_id      = (matched_obj or {}).get("object_id")
    context_objects = _sort_context_objects(context_objects, matched_id, center_pos)

    # Distance from the retrieval center to the matched object's global_position.
    # distance=0 → exact indexed position; distance>0 → pulled from surrounding window.
    matched_pos      = (matched_obj or {}).get("global_position")
    matched_distance = (
        abs(matched_pos - center_pos)
        if matched_pos is not None and center_pos is not None
        else None
    )

    # Character length of current_text sent to the verifier.
    # Tracks whether hierarchical context (heading + para + window) is growing too large.
    current_text_chars = len(current_text)

    # distance_ratio = matched_distance / CONTEXT_WINDOW
    # Normalises distance against the configured window size so analytics remain
    # comparable if CONTEXT_WINDOW changes (e.g. 3 → 5).
    #   0.0 = exact hit   1.0 = edge of window   >1.0 = outside window (shouldn't happen)
    distance_ratio = (
        round(matched_distance / CONTEXT_WINDOW, 3)
        if matched_distance is not None and CONTEXT_WINDOW > 0
        else None
    )

    context_quality = {
        "parent":    bool(parent_text),
        "prev":      bool(prev_text),
        "next":      bool(next_text),
        "n_objects": len(context_objects),
    }
    # Granular origin: "{direct|via_chunk}_{object_type}"
    #   direct    — retriever found this object in semantic-objects index directly
    #   via_chunk — BM25 chunk fallback; context_expander assigned best matching object
    obj_type = (matched_obj or {}).get("type") or "unknown"
    prefix   = "direct" if origin_is_direct else "via_chunk"
    retrieval_origin = f"{prefix}_{obj_type}"

    return {
        **candidate,
        "matched_object":     matched_obj,
        "retrieval_origin":   retrieval_origin,
        "selection_reason":   selection_reason,
        "literal_match_count": literal_match_count,
        "context_strategy":   context_strategy,
        "matched_distance":   matched_distance,
        "distance_ratio":     distance_ratio,
        "current_text_chars": current_text_chars,
        "context_objects":    context_objects,
        "context_quality":    context_quality,
        "context": {
            "parent_text":  parent_text[:CONTEXT_CHARS]  if parent_text else "",
            "prev_text":    prev_text[-CONTEXT_CHARS:]   if prev_text  else "",
            "current_text": current_text,
            "next_text":    next_text[:CONTEXT_CHARS]    if next_text  else "",
        },
    }


def _format_table_context(table_objects: list[dict], matched_obj: dict) -> str:
    """Render canonical indexed table objects into verifier context.

    The matched object is always first. Header rows are preserved, followed by
    table rows in row order. Cells are included only when a row/header has no
    usable text, preventing cell-level duplication in normal tables.
    """
    objs = sorted(
        table_objects,
        key=lambda o: (
            0 if o.get("object_id") == matched_obj.get("object_id") else 1,
            0 if o.get("type") == "table_header" else 1,
            o.get("row_index") if isinstance(o.get("row_index"), int) else 10**9,
            o.get("global_position") if isinstance(o.get("global_position"), int) else 10**9,
        ),
    )
    lines: list[str] = []
    seen: set[str] = set()
    for o in objs[:TABLE_CONTEXT_MAX_OBJECTS]:
        oid = str(o.get("object_id") or "")
        text = " ".join(str(o.get("text") or "").split())
        typ = o.get("type") or "table_row"
        if not text or oid in seen:
            continue
        seen.add(oid)
        if typ == "table_header":
            prefix = "HEADER"
        elif typ == "table_row":
            prefix = "ROW"
        elif typ == "table_cell":
            prefix = "CELL"
        elif typ == "list_item":
            # Explicitly preserve nested list semantics in verifier context.
            # The list item is not promoted to a row/cell; it remains a list item
            # associated with its parent table/cell.
            prefix = "LIST_ITEM"
        else:
            prefix = typ.upper()
        lines.append(f"[{prefix}] {text}")
        if sum(len(x) + 1 for x in lines) >= TABLE_CONTEXT_MAX_CHARS:
            break
    return "\n".join(lines)


def _fetch_table_context(document_id: str, table_ids: list[str], tenant_id: str | None = None, project_id: str | None = None) -> dict[str, list[dict]]:
    """Fetch complete table context for matched table objects in one msearch.

    Uses the canonical table_id relationship; no text matching or geometry work
    is performed here. Only semantic object fields needed by the verifier are
    returned.
    """
    if not document_id or not table_ids:
        return {}
    body: list[dict] = []
    source_fields = [
        "object_id", "document_id", "parent_chunk_id", "global_position",
        "type", "text",
        # Canonical table relationships.
        "table_id", "table_role", "row_index",
        "cell_id", "row_start", "col_start", "row_span", "col_span",
        # Canonical list relationships for list_item objects inside cells.
        "list_id", "list_level", "list_label", "list_number_format",
        "heading_path", "semantic_path",
    ]
    for table_id in table_ids:
        body.append({})
        body.append({
            "size": TABLE_CONTEXT_MAX_OBJECTS,
            "query": {"bool": {"filter": [
                {"term": {"document_id": document_id}},
                {"term": {"table_id": table_id}},
                *([{"term": {"tenant_id": tenant_id}}] if tenant_id else []),
                *([{"term": {"project_id": project_id}}] if project_id else []),

            ]}},
            "_source": source_fields,
            "sort": [
                {"row_index": {"order": "asc", "missing": "_last"}},
                {"global_position": "asc"},
            ],
        })
    try:
        resp = _get_os().msearch(body=body, index=SEMANTIC_OBJECTS_INDEX)
        result: dict[str, list[dict]] = {}
        responses = resp.get("responses", [])
        for i, table_id in enumerate(table_ids):
            if i >= len(responses):
                result[table_id] = []
                continue
            result[table_id] = [h.get("_source", {}) for h in responses[i].get("hits", {}).get("hits", [])]
        logger.info("[Context Expander] table_context tables=%d objects=%d", len(table_ids), sum(len(v) for v in result.values()))
        return result
    except Exception as exc:
        logger.warning("[Context Expander] table context msearch failed: %s", exc)
        return {table_id: [] for table_id in table_ids}


def _mget_chunks(chunk_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch primary chunk docs via mget (one round-trip)."""
    if not chunk_ids:
        return {}
    try:
        resp = _get_os().mget(index=OPENSEARCH_INDEX, body={"ids": chunk_ids})
        return {
            doc["_id"]: doc["_source"]
            for doc in resp.get("docs", [])
            if doc.get("found")
        }
    except Exception as exc:
        logger.warning("[Context Expander] mget failed, will fall back per-doc: %s", exc)
        return {}


def _msearch_neighbors_by_page(
    document_id: str,
    page_keys:   list[tuple[str, int]],  # ("page_end", val) or ("page_start", val)
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> dict[tuple, str]:
    """Batch-fetch neighbor raw_text for candidates that lack chunk_idx adjacency info."""
    if not page_keys or not document_id:
        return {}
    body: list[dict] = []
    for field, val in page_keys:
        body.append({})
        body.append({
            "size": 1,
            "query": {"bool": {"filter": [
                {"term": {"document_id": document_id}},
                {"term": {field: val}},
                *([{"term": {"tenant_id": tenant_id}}] if tenant_id else []),
                *([{"term": {"project_id": project_id}}] if project_id else []),
            ]}},
            "_source": ["raw_text"],
        })
    try:
        resp = _get_os().msearch(body=body, index=OPENSEARCH_INDEX)
        return {
            page_keys[i]: (r.get("hits", {}).get("hits") or [{}])[0].get("_source", {}).get("raw_text", "")
            for i, r in enumerate(resp.get("responses", []))
        }
    except Exception as exc:
        logger.warning("[Context Expander] page neighbor msearch failed: %s", exc)
        return {}


def _msearch_by_idx(document_id: str, idx_list: list[int], tenant_id: str | None = None, project_id: str | None = None) -> dict[int, str]:
    """Batch-fetch raw_text for prev/next/parent chunks by chunk_idx via msearch."""
    if not idx_list or not document_id:
        return {}
    body: list[dict] = []
    for idx in idx_list:
        body.append({})
        body.append({
            "size": 1,
            "query": {"bool": {"filter": [
                {"term": {"document_id": document_id}},
                {"term": {"chunk_idx": idx}},
                *([{"term": {"tenant_id": tenant_id}}] if tenant_id else []),
                *([{"term": {"project_id": project_id}}] if project_id else []),
            ]}},
            "_source": ["raw_text"],
        })
    try:
        resp = _get_os().msearch(body=body, index=OPENSEARCH_INDEX)
        result: dict[int, str] = {}
        for i, r in enumerate(resp.get("responses", [])):
            hits = r.get("hits", {}).get("hits", [])
            if hits:
                result[idx_list[i]] = hits[0]["_source"].get("raw_text", "")
        return result
    except Exception as exc:
        logger.warning("[Context Expander] msearch by idx failed: %s", exc)
        return {}


def _fetch_context_objects_merged(
    document_id: str,
    ctx_keys:    list[tuple[str, int | None]],
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> dict[tuple, list[dict]]:
    """Fetch context objects for all candidates in ONE OpenSearch terms query.

    Instead of N range sub-queries, expand every center_pos by ±CONTEXT_WINDOW,
    deduplicate all resulting positions, then issue a single terms(global_position)
    query.  For 190 dispersed candidates × 7 positions each = ~1,288 unique
    positions vs 190 individual searches.
    """
    if not ctx_keys:
        return {}

    result: dict[tuple, list[dict]] = {}
    with_pos    = [(key, key[1]) for key in ctx_keys if key[1] is not None]
    without_pos = [key for key in ctx_keys if key[1] is None]

    # ── Chunk-only fallbacks (no global_position) — still need per-chunk queries ──
    if without_pos:
        body: list[dict] = []
        for key in without_pos:
            body.append({})
            body.append({
                "size": 100,
                "query": {"bool": {"filter": [{"term": {"parent_chunk_id": key[0]}}]}},
                "sort": [{"global_position": "asc"}],
            })
        try:
            resp = _get_os().msearch(body=body, index=SEMANTIC_OBJECTS_INDEX)
            for i, r in enumerate(resp.get("responses", [])):
                result[without_pos[i]] = [h["_source"] for h in r.get("hits", {}).get("hits", [])]
        except Exception as exc:
            logger.warning("[Context Expander] chunk-only msearch failed: %s", exc)
            for key in without_pos:
                result[key] = []

    if not with_pos:
        return result

    # ── Expand all center positions to full ±CONTEXT_WINDOW neighbourhoods ────
    needed: set[int] = set()
    for _, center_pos in with_pos:
        for offset in range(-CONTEXT_WINDOW, CONTEXT_WINDOW + 1):
            needed.add(center_pos + offset)

    logger.info(
        "[Context Expander] ctx_keys=%d  needed_positions=%d  os_queries=1",
        len(with_pos), len(needed),
    )

    # ── ONE terms query — all needed positions in a single request ────────────
    import time as _time
    pos_to_objs: dict[int, list[dict]] = {}
    try:
        fetch_size = min(len(needed) * 6, 10000)
        _t_os = _time.perf_counter()
        resp = _get_os().search(
            index=SEMANTIC_OBJECTS_INDEX,
            body={
                "size": fetch_size,
                "query": {"bool": {"filter": [
                    {"term":  {"document_id": document_id}},
                    {"terms": {"global_position": sorted(needed)}},
                    *([{"term": {"tenant_id": tenant_id}}] if tenant_id else []),
                    *([{"term": {"project_id": project_id}}] if project_id else []),
                ]}},
                "sort": [{"global_position": "asc"}],
            },
        )
        os_elapsed_ms = round((_time.perf_counter() - _t_os) * 1000)
        hits = resp.get("hits", {}).get("hits", [])
        logger.info(
            "[Context Expander] needed_positions=%d  returned_objects=%d  fetch_size=%d  os_ms=%d",
            len(needed), len(hits), fetch_size, os_elapsed_ms,
        )
        for h in hits:
            src  = h["_source"]
            gpos = src.get("global_position")
            if gpos is not None:
                pos_to_objs.setdefault(gpos, []).append(src)
    except Exception as exc:
        logger.warning("[Context Expander] terms context fetch failed: %s", exc)

    # ── Resolve per-key neighborhood from the local position dict ─────────────
    for key, center_pos in with_pos:
        neighborhood = []
        for offset in range(-CONTEXT_WINDOW, CONTEXT_WINDOW + 1):
            neighborhood.extend(pos_to_objs.get(center_pos + offset, []))
        result[key] = sorted(neighborhood, key=lambda o: o.get("global_position", 0))

    return result
