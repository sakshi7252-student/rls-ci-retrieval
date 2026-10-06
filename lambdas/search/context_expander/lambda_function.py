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
        # Primary fields — what the verifier should read.
        "matched_evidence": str,  # the SINGLE matched span only (never paragraph+window merged)
        "previous_text":    str,  # immediate previous sentence/object (or prev chunk, if unanchored)
        "next_text":        str,  # immediate next sentence/object (or next chunk, if unanchored)
        "parent_paragraph": str,  # full parent paragraph — sentence hits only, context only
        "section_heading":  str,  # normalized section heading path — interpretive only
        "table_context":    str,  # full table text (context only) when matched_obj is a table_* type
        "list_context":     str,  # full list text (context only) when matched_obj is a list/list_item
        "context_type":     str,  # same value as top-level context_strategy
        # Legacy aliases kept for existing consumers (merger/reranker/worker read these directly).
        "current_text":     str,  # == matched_evidence for anchored hits; whole chunk if unanchored
        "heading_context":  str,  # == section_heading
        "prev_text":        str,  # == previous_text
        "parent_text":      str,  # == parent_paragraph
        "neighbor_kind":    str,  # "object": prev/next are the matched object's own neighbours
                                  # "chunk":  prev/next are the adjacent chunks (unanchored candidates)
    },
    "anchored":          bool,  # True for every normally-resolved candidate (direct hit or a real,
                                # fanned-out semantic object). False only for "unresolved" candidates.
    "unanchored_reason": str|None,  # set only when resolution_status == "unresolved"
    "resolution_status": str,       # "resolved" | "unresolved" (chunk had zero indexed semantic objects)
    "resolution_reason": str|None,  # e.g. "no_semantic_objects_for_chunk"
    "context_complete":  bool,      # False when a table_*/list* object's full context couldn't be built
    "context_warning":   str|None,  # e.g. "table_object_missing_table_id", "table_fetch_empty"
}

Context Expander invariants:
  1. Every "resolved" candidate has matched_object != None.
  2. Context selection depends ONLY on matched_object.type, never on retrieval origin
     (direct hit vs chunk fan-out) — see _expand_object_context()'s dispatch.
  3. table_* objects always attempt full table context via table_id; context_complete=False
     (never a silent fallback) if table_id is missing or the table fetch is empty.
  4. list_item (and, if ever indexed, "list") objects always attempt full list context via
     list_id, with the same non-silent completeness rule as tables.
  5. sentence objects get sentence+parent-paragraph+neighbour-sentence context.
  6. Zero semantic objects for a chunk is an indexing gap, not ambiguity — stays
     resolution_status="unresolved", never a fabricated matched_object.
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
# Page size for table/list container fetches (table_id/list_id) — pagination below walks
# past this for containers with more objects than the page, so "full table"/"full list"
# holds regardless of container size. Not a hard cap.
CONTAINER_CONTEXT_PAGE_SIZE = int(os.environ.get("CONTAINER_CONTEXT_PAGE_SIZE", "200"))
# Hard cap on total objects paginated for a single table/list container. Formatting already
# truncates at TABLE_CONTEXT_MAX_CHARS, so a table/list beyond this size was always going to
# be cut off in the output anyway — this just stops paying the OpenSearch + memory cost of
# fetching objects that never make it into the formatted context.
CONTAINER_CONTEXT_MAX_OBJECTS = int(os.environ.get("CONTAINER_CONTEXT_MAX_OBJECTS", "200"))
TABLE_CONTEXT_MAX_CHARS = int(os.environ.get("TABLE_CONTEXT_MAX_CHARS", "16000"))
CONTEXT_EXPANDER_WORKERS = int(os.environ.get("CONTEXT_EXPANDER_WORKERS", "1"))
# Objects fetched per chunk for chunk-level (via_chunk) candidates. The old hard-coded 100 cut off
# long multi-page chunks, so a literal match late in the chunk was never found.
CHUNK_OBJECTS_MAX = int(os.environ.get("CHUNK_OBJECTS_MAX", "400"))
# Whole-chunk text cap (unanchored candidates) so the payload and verifier prompt stay bounded.
CHUNK_TEXT_MAX_CHARS = int(os.environ.get("CHUNK_TEXT_MAX_CHARS", "8000"))
# prev/next for an anchored object = this many same-type neighbouring objects each side.
NEIGHBOR_OBJECTS = int(os.environ.get("NEIGHBOR_OBJECTS", "2"))
# Global per-CI cap on expanded candidates. Direct candidates are always kept in full; this
# only trims chunk-fanout (which can blow up to tens of thousands of objects for table-heavy
# chunks) down to the highest-priority objects BEFORE the expensive context/table-fetch work.
MAX_EXPANDED_CANDIDATES = int(os.environ.get("MAX_EXPANDED_CANDIDATES", "500"))
# 1 = also attach adjacent-CHUNK text to anchored objects (old behaviour; causes cross-chunk bleed).
CHUNK_NEIGHBORS_FOR_ANCHORED = os.environ.get("CHUNK_NEIGHBORS_FOR_ANCHORED", "0") == "1"

def _get_os():
    from shared.opensearch_client import get_opensearch_client
    return get_opensearch_client()


# Cross-CI cache for table/list container fetches — every CI in a batch shares the same
# document_id, and CIs frequently hit the same tables (e.g. a demographics/sample-size
# table matched by several numeric CIs), so without this each CI independently re-fetches
# and re-holds a full copy of the same tens-of-thousands-of-objects table in memory.
# Scoped to the current document_id only — reset on a new document so a warm container
# reused across documents never serves stale data.
_CONTAINER_CACHE_DOC_ID: str | None = None
_TABLE_CACHE: dict[str, list[dict]] = {}
_LIST_CACHE: dict[str, list[dict]] = {}
# Rendered-text cache: fan-out means many expanded candidates can share the same
# table_id (often the same row_index too — a table_row object + several table_cell
# objects from one row all land in the same chunk). Without this, _build_table_context
# reformats and duplicates the ENTIRE table string once per fanned-out object — for a
# 189-table document this is the dominant memory cost, not the container fetch itself.
# Keyed by (table_id, matched_row_index) since that's the only thing that changes the
# rendered output (matched row is bubbled to the front).
_TABLE_TEXT_CACHE: dict[tuple[str, int | None], str] = {}


def _reset_container_cache_if_new_document(document_id: str) -> None:
    global _CONTAINER_CACHE_DOC_ID
    if document_id != _CONTAINER_CACHE_DOC_ID:
        _TABLE_CACHE.clear()
        _LIST_CACHE.clear()
        _TABLE_TEXT_CACHE.clear()
        _CONTAINER_CACHE_DOC_ID = document_id


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

    _reset_container_cache_if_new_document(document_id)

    # ── Phase 1: collect all lookup keys ──────────────────────────────────────
    primary_ids:  list[str]                          = []
    idx_needed:   set[int]                           = set()
    ctx_keys:     list[tuple[str, int | None]]       = []
    _ctx_key_set: set[tuple[str, int | None]]        = set()
    page_lookups: set[tuple[str, int]]               = set()
    table_ids: set[str]                              = set()
    list_ids: set[str]                                = set()

    for c in candidates:
        cid      = c.get("chunk_id", "")
        obj_meta = c.get("matched_object") or {}
        primary_ids.append(cid)
        for key in ("prev_chunk_idx", "next_chunk_idx", "parent_chunk_idx"):
            idx = obj_meta.get(key)
            if idx is not None:
                idx_needed.add(idx)
        # Table/list hits need their full container context. table_id/list_id are the
        # canonical relationships carried by the indexed object; never rediscover from text.
        table_id = obj_meta.get("table_id")
        if table_id and obj_meta.get("type") in _TABLE_TYPES:
            table_ids.add(str(table_id))
        list_id = obj_meta.get("list_id")
        if list_id and obj_meta.get("type") in _LIST_TYPES:
            list_ids.add(str(list_id))

        ctx_key = (cid, obj_meta.get("global_position"))
        if ctx_key not in _ctx_key_set:
            ctx_keys.append(ctx_key)
            _ctx_key_set.add(ctx_key)
        # Collect page-range neighbor keys for candidates without chunk_idx adjacency
        if obj_meta.get("prev_chunk_idx") is None:
            page_lookups.add(("page_end",   c.get("page_start", 0) - 1))
        if obj_meta.get("next_chunk_idx") is None:
            page_lookups.add(("page_start", c.get("page_end",   0) + 1))

    # ── Phase 2: run all 5 fetches concurrently (they're independent) ──────────
    from concurrent.futures import ThreadPoolExecutor as _TPE
    deduped_ids = list(dict.fromkeys(primary_ids))
    with _TPE(max_workers=CONTEXT_EXPANDER_WORKERS) as _pool:
        _f_chunk = _pool.submit(_mget_chunks, deduped_ids)
        _f_idx   = _pool.submit(_msearch_by_idx, document_id, list(idx_needed), tenant_id=tenant_id, project_id=project_id)
        _f_ctx   = _pool.submit(_fetch_context_objects_merged, document_id, ctx_keys, tenant_id=tenant_id, project_id=project_id)
        _f_page  = _pool.submit(_msearch_neighbors_by_page, document_id, list(page_lookups), tenant_id=tenant_id, project_id=project_id)
        _f_table = _pool.submit(_fetch_table_context, document_id, sorted(table_ids), tenant_id=tenant_id, project_id=project_id)
        _f_list  = _pool.submit(_fetch_list_context, document_id, sorted(list_ids), tenant_id=tenant_id, project_id=project_id)
        chunk_cache: dict[str, dict]         = _f_chunk.result()
        idx_cache:   dict[int, str]          = _f_idx.result()
        ctx_cache:   dict[tuple, list[dict]] = _f_ctx.result()
        page_cache:  dict[tuple, str]        = _f_page.result()
        table_cache: dict[str, list[dict]]     = _f_table.result()
        list_cache:  dict[str, list[dict]]     = _f_list.result()
    # Second mget pass for any IDs the first mget missed
    missed = [cid for cid in deduped_ids if cid not in chunk_cache]
    if missed:
        chunk_cache.update(_mget_chunks(missed))

    # Exact chunk adjacency (chunk_idx ±1) for candidates that need adjacent-chunk text:
    # chunk-level candidates (no matched object) and, optionally, anchored ones.
    extra_idx: set[int] = set()
    for c in candidates:
        if c.get("matched_object") and not CHUNK_NEIGHBORS_FOR_ANCHORED:
            continue
        self_idx = (chunk_cache.get(c.get("chunk_id", "")) or {}).get("chunk_idx")
        if isinstance(self_idx, int):
            for k in (self_idx - 1, self_idx + 1):
                if k >= 0 and k not in idx_cache:
                    extra_idx.add(k)
    if extra_idx:
        idx_cache.update(_msearch_by_idx(document_id, sorted(extra_idx),
                                         tenant_id=tenant_id, project_id=project_id))

    logger.info(
        "[Context Expander] search_id=%s  candidates=%d  idx_lookups=%d"
        "  ctx_queries=%d  chunk_cache_hits=%d/%d",
        req.get("search_id"), len(candidates), len(idx_needed),
        len(ctx_keys), len(chunk_cache), len(primary_ids),
    )

    # ── Phase 3: resolve every candidate to one or more real semantic objects ──────────
    # Resolution is separate from context building: a direct hit resolves to its own
    # matched_object; a chunk-level candidate (matched_object=None) resolves to EVERY real
    # semantic object indexed under that chunk (fan-out), not a single guessed object. The
    # only candidate that can leave this phase without a matched_object is one whose chunk
    # genuinely has zero indexed semantic objects — an indexing gap, not ambiguity to resolve.
    direct_pending: list[tuple[dict, dict | None, list[dict], str]] = []
    fanout_scored: list[tuple[tuple, dict, dict, list[dict]]] = []  # (sort_key, candidate, obj, pool)
    for c in candidates:
        direct_obj = c.get("matched_object")
        if direct_obj:
            pool = ctx_cache.get((c.get("chunk_id", ""), direct_obj.get("global_position")), [])
            direct_pending.append((c, direct_obj, pool, "direct"))
            continue

        pool = ctx_cache.get((c.get("chunk_id", ""), None), [])
        objs = _dedupe_fanout_objects(pool)
        if not objs:
            direct_pending.append((c, None, pool, "unresolved"))
            continue
        agg_score = c.get("agg_score") or 0.0
        for o in objs:
            # Retrieval relevance (originating chunk's own score) is the primary signal;
            # object type only breaks ties within equally-relevant chunks — otherwise every
            # heading from a weak chunk would outrank a sentence from a strong one.
            type_rank = _FANOUT_TYPE_PRIORITY.get(o.get("type"), 7)
            sort_key = (-agg_score, type_rank, c.get("chunk_id", ""), o.get("global_position") or 0)
            fanout_scored.append((sort_key, c, o, pool))

    # Global per-CI cap: direct candidates are never trimmed; fan-out objects are ranked by
    # type priority then by their originating chunk's own retrieval score, and truncated HERE —
    # before _build_expanded_candidate/table-fetch — so the cap actually saves the work, not
    # just the output size.
    fanout_budget = max(MAX_EXPANDED_CANDIDATES - len(direct_pending), 0)
    if len(fanout_scored) > fanout_budget:
        fanout_scored.sort(key=lambda t: t[0])
        logger.info(
            "[Context Expander] search_id=%s  fanout_cap=%d  fanout_total=%d  dropped=%d",
            req.get("search_id"), MAX_EXPANDED_CANDIDATES, len(fanout_scored),
            len(fanout_scored) - fanout_budget,
        )
        fanout_scored = fanout_scored[:fanout_budget]

    pending: list[tuple[dict, dict | None, list[dict], str]] = list(direct_pending)
    extra_table_ids: set[str] = set()
    extra_list_ids: set[str] = set()
    for _, c, o, pool in fanout_scored:
        otype = o.get("type")
        tid = o.get("table_id")
        if tid and otype in _TABLE_TYPES and str(tid) not in table_cache:
            extra_table_ids.add(str(tid))
        lid = o.get("list_id")
        if lid and otype in _LIST_TYPES and str(lid) not in list_cache:
            extra_list_ids.add(str(lid))
        pending.append((c, o, pool, "fanout"))

    # Fanned-out/direct objects can reference tables/lists Phase 1 never saw (it only scans
    # the candidate's own matched_object table_id/list_id, which chunk-level candidates don't
    # have yet) — fetch those now, same document-wide query as the primary path, so every
    # row/cell/list item gets the complete cross-chunk container instead of just whatever
    # happens to be in this chunk's objects.
    if extra_table_ids:
        table_cache.update(_fetch_table_context(document_id, sorted(extra_table_ids),
                                                 tenant_id=tenant_id, project_id=project_id))
    if extra_list_ids:
        list_cache.update(_fetch_list_context(document_id, sorted(extra_list_ids),
                                               tenant_id=tenant_id, project_id=project_id))

    expanded: list[dict] = []
    for c, o, pool, origin in pending:
        if o is None:
            expanded.append(_build_unresolved_candidate(c, chunk_cache, idx_cache, page_cache))
        else:
            expanded.append(_build_expanded_candidate(
                c, o, pool, table_cache, list_cache, chunk_cache, idx_cache, page_cache, origin,
            ))

    if expanded:
        avg_chars = sum(e.get("current_text_chars", 0) for e in expanded) / len(expanded)
        logger.info(
            "[Context Expander] search_id=%s  avg_context_chars=%.0f  max_context_chars=%d",
            req.get("search_id"), avg_chars,
            max(e.get("current_text_chars", 0) for e in expanded),
        )

    return {**req, "expanded_candidates": expanded}


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


_DEDUP_NORMALIZE_RE = re.compile(r"[^\w]+")

def _normalize_for_dedup(text: str) -> str:
    """Collapse whitespace/punctuation/case so near-identical spans (e.g. a sentence
    window that's just the paragraph's own tail, re-rendered with different tab/period
    formatting) are recognized as duplicates rather than appended again verbatim."""
    return _DEDUP_NORMALIZE_RE.sub(" ", text.lower()).strip()


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


_TABLE_TYPES = {"table_header", "table_row", "table_cell"}
_LIST_TYPES  = {"list_item", "list"}

# Fan-out prioritization when over MAX_EXPANDED_CANDIDATES: prose-like objects first (most
# likely to carry unique evidence), table_cell last (its text is mostly already covered by
# its own table_row — see _format_table_context's row-preferred rendering).
_FANOUT_TYPE_PRIORITY = {
    "heading":      0,
    "sentence":     1,
    "paragraph":    2,
    "table_row":    3,
    "table_header": 4,
    "list_item":    5,
    "table_cell":   6,
}



def _local_neighbors(
    context_objects: list[dict],
    matched_obj:     dict | None,
    n:               int,
) -> tuple[str, str]:
    """Text immediately before/after the matched object, taken from ITS OWN neighbours
    (same object type, by global_position) — not from adjacent chunks, which can be
    thousands of characters away and carry unrelated facts."""
    if not matched_obj or not context_objects:
        return "", ""
    mtype = matched_obj.get("type")
    mid   = matched_obj.get("object_id")
    mpos  = matched_obj.get("global_position")
    if mpos is None:
        return "", ""
    same = [
        o for o in context_objects
        if o.get("type") == mtype and o.get("object_id") != mid
        and isinstance(o.get("global_position"), int)
        and str(o.get("text") or "").strip()
    ]
    before = sorted((o for o in same if o["global_position"] < mpos),
                    key=lambda o: o["global_position"])[-n:]
    after  = sorted((o for o in same if o["global_position"] > mpos),
                    key=lambda o: o["global_position"])[:n]

    def _join(objs: list[dict]) -> str:
        return " ".join(" ".join(str(o["text"]).split()) for o in objs)

    return _join(before), _join(after)


def _dedupe_fanout_objects(pool: list[dict]) -> list[dict]:
    """Every real semantic object with text becomes its own candidate — correctness over
    candidate-count for now; paragraph/sentence overlap can be optimized later once fan-out
    volume and verifier duplicate-hit behavior are measured."""
    return [o for o in pool if (o.get("text") or "").strip()]


def _object_heading(obj: dict) -> str:
    raw = (obj.get("heading_path") or obj.get("semantic_path") or "").strip()
    return _normalize_heading(raw) if raw else ""


_EMPTY_CONTEXT = {"table_text": "", "list_text": "", "prev_text": "", "next_text": "", "parent_paragraph": ""}


def _build_table_context(obj: dict, context_objects: list[dict], table_cache: dict[str, list[dict]]) -> dict:
    """table_* objects ALWAYS attempt full table context via table_id — never silently
    fall back to bare row/cell text; a missing table_id or empty fetch is surfaced via
    context_complete=False so it's measurable, not hidden."""
    table_id = obj.get("table_id")
    table_text, warning = "", None
    if not table_id:
        warning = "table_object_missing_table_id"
    else:
        matched_row_index = obj.get("row_index") if isinstance(obj.get("row_index"), int) else None
        cache_key = (str(table_id), matched_row_index)
        table_text = _TABLE_TEXT_CACHE.get(cache_key)
        if table_text is None:
            table_objs = table_cache.get(str(table_id)) or [
                x for x in context_objects if x.get("table_id") == table_id
            ]
            if table_objs:
                table_text = _format_table_context(table_objs, obj)
                _TABLE_TEXT_CACHE[cache_key] = table_text
            else:
                table_text = ""
                warning = "table_fetch_empty"
    return {
        **_EMPTY_CONTEXT,
        "context_strategy":  "table_full" if table_text else "table_incomplete",
        "current_text":      (obj.get("text") or "").strip(),
        "table_text":        table_text,
        "heading":           _object_heading(obj),
        "context_complete":  table_text != "",
        "context_warning":   warning,
    }


def _build_list_context(obj: dict, context_objects: list[dict], list_cache: dict[str, list[dict]]) -> dict:
    """list_item (and "list", if ever indexed) objects ALWAYS attempt full list context via
    list_id — same non-silent completeness rule as tables."""
    list_id = obj.get("list_id")
    list_text, warning = "", None
    if not list_id:
        warning = "list_object_missing_list_id"
    else:
        list_objs = list_cache.get(str(list_id)) or [
            x for x in context_objects if x.get("list_id") == list_id
        ]
        if list_objs:
            list_text = _format_list_context(list_objs, obj)
        else:
            warning = "list_fetch_empty"
    return {
        **_EMPTY_CONTEXT,
        "context_strategy":  "list_full" if list_text else "list_incomplete",
        "current_text":      (obj.get("text") or "").strip(),
        "list_text":         list_text,
        "heading":           _object_heading(obj),
        "context_complete":  list_text != "",
        "context_warning":   warning,
    }


def _build_sentence_context(obj: dict, context_objects: list[dict]) -> dict:
    """current_text (== matched_evidence) is the matched sentence ALONE. Its parent
    paragraph and immediate neighbour sentences are exposed as separate context-only
    fields so the verifier can't mistake surrounding text for the matched span itself.

    A paragraph made of exactly one sentence has no sibling sentences to borrow
    prev/next text from, and parent_paragraph is correctly suppressed (it's identical to
    current_text) — without a fallback this leaves the candidate with ZERO context,
    letting the verifier judge it on the bare sentence alone. Fall back to the same
    paragraph-level neighbour lookup the parent paragraph object itself would get."""
    text      = (obj.get("text") or "").strip()
    para_text = (obj.get("paragraph_text") or "").strip()
    prev_text = (obj.get("prev_sentence_text") or "").strip()
    next_text = (obj.get("next_sentence_text") or "").strip()
    is_whole_paragraph = bool(para_text) and para_text == text
    if is_whole_paragraph and not prev_text and not next_text:
        parent_id  = re.sub(r"_s\d+$", "", obj.get("object_id") or "")
        parent_obj = next((o for o in context_objects if o.get("object_id") == parent_id), None)
        if parent_obj:
            prev_text, next_text = _local_neighbors(context_objects, parent_obj, NEIGHBOR_OBJECTS)
    return {
        **_EMPTY_CONTEXT,
        "context_strategy":  "sentence_hierarchical",
        "current_text":      text,
        "prev_text":         prev_text,
        "next_text":         next_text,
        "parent_paragraph":  para_text if para_text and para_text != text else "",
        "heading":           _object_heading(obj),
        "context_complete":  True,
        "context_warning":   None,
    }


def _build_generic_object_context(obj: dict, context_objects: list[dict]) -> dict:
    """paragraph, heading, or any other indexed type — nearest same-type neighbours."""
    prev_text, next_text = _local_neighbors(context_objects, obj, NEIGHBOR_OBJECTS)
    return {
        **_EMPTY_CONTEXT,
        "context_strategy":  obj.get("type") or "unknown",
        "current_text":      (obj.get("text") or obj.get("paragraph_text") or "").strip(),
        "prev_text":         prev_text,
        "next_text":         next_text,
        "heading":           _object_heading(obj),
        "context_complete":  True,
        "context_warning":   None,
    }


def _expand_object_context(
    obj:             dict,
    context_objects: list[dict],
    table_cache:     dict[str, list[dict]],
    list_cache:      dict[str, list[dict]],
) -> dict:
    """Type dispatch ONLY — by the time this runs, `obj` is already a real, resolved
    semantic object (direct hit or fanned-out); no object-selection logic lives here.
    Dispatch depends solely on matched_object.type, never on retrieval origin."""
    otype = obj.get("type") or "unknown"
    if otype in _TABLE_TYPES:
        return _build_table_context(obj, context_objects, table_cache)
    if otype in _LIST_TYPES:
        return _build_list_context(obj, context_objects, list_cache)
    if otype == "sentence":
        return _build_sentence_context(obj, context_objects)
    return _build_generic_object_context(obj, context_objects)


def _chunk_adjacent_text(
    candidate:   dict,
    obj:         dict | None,
    chunk_cache: dict[str, dict],
    idx_cache:   dict[int, str],
    page_cache:  dict[tuple, str],
) -> tuple[str, str]:
    """Adjacent-CHUNK text (not object-level) — exact chunk_idx adjacency when known,
    page-range lookup as a last resort. Only used for CHUNK_NEIGHBORS_FOR_ANCHORED and the
    unresolved (no-semantic-objects) fallback; normal candidates use the matched object's
    own neighbours instead (see _expand_object_context)."""
    chunk_id   = candidate.get("chunk_id", "")
    page_start = candidate.get("page_start", 0)
    page_end   = candidate.get("page_end", 0)
    chunk_doc  = chunk_cache.get(chunk_id, {})
    self_idx   = chunk_doc.get("chunk_idx")

    obj_meta = obj or {}
    prev_idx = obj_meta.get("prev_chunk_idx")
    next_idx = obj_meta.get("next_chunk_idx")
    if prev_idx is None and isinstance(self_idx, int):
        prev_idx = self_idx - 1
    if next_idx is None and isinstance(self_idx, int):
        next_idx = self_idx + 1

    prev_text = idx_cache.get(prev_idx, "") if prev_idx is not None else page_cache.get(("page_end", page_start - 1), "")
    next_text = idx_cache.get(next_idx, "") if next_idx is not None else page_cache.get(("page_start", page_end + 1), "")
    return prev_text, next_text


def _build_expanded_candidate(
    candidate:       dict,
    obj:             dict,
    context_objects: list[dict],
    table_cache:     dict[str, list[dict]],
    list_cache:      dict[str, list[dict]],
    chunk_cache:     dict[str, dict],
    idx_cache:       dict[int, str],
    page_cache:      dict[tuple, str],
    origin:          str,   # "direct" | "fanout"
) -> dict:
    built = _expand_object_context(obj, context_objects, table_cache, list_cache)

    prev_text, next_text, neighbor_kind = built["prev_text"], built["next_text"], "object"
    # Object-level neighbours are the default; adjacent-CHUNK text is unrelated to the
    # matched object and was the historical source of cross-chunk bleed, so it's opt-in only.
    if not prev_text and not next_text and CHUNK_NEIGHBORS_FOR_ANCHORED:
        prev_text, next_text = _chunk_adjacent_text(candidate, obj, chunk_cache, idx_cache, page_cache)
        neighbor_kind = "chunk" if (prev_text or next_text) else "object"

    matched_pos = obj.get("global_position")
    sorted_objects = _sort_context_objects(context_objects, obj.get("object_id"), matched_pos)

    retrieval_origin = (
        f"direct_{obj.get('type') or 'unknown'}" if origin == "direct"
        else f"via_chunk_fanout_{obj.get('type') or 'unknown'}"
    )
    selection_reason = "retriever_direct" if origin == "direct" else "chunk_fanout"

    # The candidate's page_start/page_end/match_page start out as the CONTAINING CHUNK's
    # page range (set by the retriever, e.g. literal_retriever's hit.page_start/page_end).
    # A chunk spans multiple pages, so a fanned-out object partway through it (e.g. a
    # sentence on page 26 of a page 24-26 chunk) would otherwise stay mislabeled under the
    # chunk's first page — resolving the real object means we know its true page, so use it.
    obj_page = obj.get("page")
    page_start = obj_page if isinstance(obj_page, int) else candidate.get("page_start", 0)
    page_end   = obj_page if isinstance(obj_page, int) else candidate.get("page_end", 0)

    return {
        **candidate,
        "matched_object":     obj,
        "page_start":         page_start,
        "page_end":           page_end,
        "match_page":         obj_page if isinstance(obj_page, int) else candidate.get("match_page"),
        "retrieval_origin":   retrieval_origin,
        "selection_reason":   selection_reason,
        "resolution_status":  "resolved",
        "resolution_reason":  None,
        "anchored":           True,
        "unanchored_reason":  None,
        "context_strategy":   built["context_strategy"],
        "context_complete":   built["context_complete"],
        "context_warning":    built["context_warning"],
        "matched_distance":   0,
        "distance_ratio":     0.0,
        "current_text_chars": len(built["current_text"]),
        "context_objects":    sorted_objects,
        "context_quality": {
            "parent":    bool(built["parent_paragraph"]),
            "prev":      bool(prev_text),
            "next":      bool(next_text),
            "n_objects": len(context_objects),
        },
        "context": {
            "matched_evidence": built["current_text"],
            "previous_text":    prev_text[-CONTEXT_CHARS:] if prev_text else "",
            "next_text":        next_text[:CONTEXT_CHARS] if next_text else "",
            "parent_paragraph": built["parent_paragraph"][:CONTEXT_CHARS] if built["parent_paragraph"] else "",
            "section_heading":  built["heading"],
            "table_context":    built["table_text"],
            "list_context":     built["list_text"],
            "context_type":     built["context_strategy"],
            "current_text":     built["current_text"],
            "heading_context":  built["heading"],
            "prev_text":        prev_text[-CONTEXT_CHARS:] if prev_text else "",
            "parent_text":      built["parent_paragraph"][:CONTEXT_CHARS] if built["parent_paragraph"] else "",
            "neighbor_kind":    neighbor_kind,
        },
    }


def _build_unresolved_candidate(
    candidate:   dict,
    chunk_cache: dict[str, dict],
    idx_cache:   dict[int, str],
    page_cache:  dict[tuple, str],
) -> dict:
    """Last resort only: the chunk has genuinely zero indexed semantic objects (an
    indexing gap, not ambiguity). Never fabricate a matched_object — surface the whole
    chunk text so a verifier *could* still judge it, but mark it unresolved so it is never
    mistaken for a normal, confidently-anchored hit."""
    chunk_id   = candidate.get("chunk_id", "")
    chunk_text = chunk_cache.get(chunk_id, {}).get("raw_text", "")
    if len(chunk_text) > CHUNK_TEXT_MAX_CHARS:
        chunk_text = chunk_text[:CHUNK_TEXT_MAX_CHARS]
    prev_text, next_text = _chunk_adjacent_text(candidate, None, chunk_cache, idx_cache, page_cache)

    return {
        **candidate,
        "matched_object":     None,
        "retrieval_origin":   "via_chunk_unresolved",
        "selection_reason":   "unresolved_no_objects",
        "resolution_status":  "unresolved",
        "resolution_reason":  "no_semantic_objects_for_chunk",
        "anchored":           False,
        "unanchored_reason":  "no_semantic_objects_for_chunk",
        "context_strategy":   "chunk_fallback",
        "context_complete":   False,
        "context_warning":    "no_semantic_objects_for_chunk",
        "matched_distance":   None,
        "distance_ratio":     None,
        "current_text_chars": len(chunk_text),
        "context_objects":    [],
        "context_quality": {
            "parent":    False,
            "prev":      bool(prev_text),
            "next":      bool(next_text),
            "n_objects": 0,
        },
        "context": {
            "matched_evidence": "",
            "previous_text":    prev_text[-CONTEXT_CHARS:] if prev_text else "",
            "next_text":        next_text[:CONTEXT_CHARS] if next_text else "",
            "parent_paragraph": "",
            "section_heading":  "",
            "table_context":    "",
            "list_context":     "",
            "context_type":     "chunk_fallback",
            "current_text":     chunk_text,
            "heading_context":  "",
            "prev_text":        prev_text[-CONTEXT_CHARS:] if prev_text else "",
            "parent_text":      "",
            "neighbor_kind":    "chunk",
        },
    }


def _format_table_context(table_objects: list[dict], matched_obj: dict) -> str:
    """Render canonical indexed table objects into a true row/column grid for the verifier.

    A "table_row" object's own text is already the pipe-joined grid line for that row, so it
    is preferred whenever present. "table_cell" objects are only rendered standalone for rows
    that have no table_row object indexed (grouped by row_index, ordered by col_start) — this
    avoids showing both the individual cells AND the row that already joins them, which looks
    like repeated evidence to the verifier. The matched object's own row is always first.
    """
    matched_row_index = matched_obj.get("row_index") if isinstance(matched_obj.get("row_index"), int) else None

    headers: list[dict] = []
    rows_by_index: dict[int, dict] = {}
    cells_by_index: dict[int, list[dict]] = {}
    list_items: list[dict] = []
    other: list[dict] = []

    for o in table_objects:
        typ = o.get("type") or "table_row"
        text = " ".join(str(o.get("text") or "").split())
        if not text:
            continue
        ridx = o.get("row_index") if isinstance(o.get("row_index"), int) else None
        if typ == "table_header":
            headers.append(o)
        elif typ == "table_row":
            if ridx is not None and ridx not in rows_by_index:
                rows_by_index[ridx] = o
        elif typ == "table_cell":
            if ridx is not None:
                cells_by_index.setdefault(ridx, []).append(o)
            else:
                other.append(o)
        elif typ == "list_item":
            list_items.append(o)
        else:
            other.append(o)

    # Rows with no indexed table_row object: synthesize one grid line from their cells.
    synthesized_rows: dict[int, str] = {}
    for ridx, cells in cells_by_index.items():
        if ridx in rows_by_index:
            continue
        ordered = sorted(cells, key=lambda o: o.get("col_start") if isinstance(o.get("col_start"), int) else 10**9)
        texts = [" ".join(str(o.get("text") or "").split()) for o in ordered]
        texts = [t for t in texts if t]
        if texts:
            synthesized_rows[ridx] = " | ".join(texts)

    all_row_indices = sorted(set(rows_by_index) | set(synthesized_rows))
    all_row_indices.sort(key=lambda r: (0 if r == matched_row_index else 1, r))

    lines: list[str] = []
    seen_text: set[str] = set()

    def _emit(prefix: str, text: str) -> bool:
        norm = _normalize_for_dedup(text)
        if norm and norm in seen_text:
            return False
        if norm:
            seen_text.add(norm)
        lines.append(f"[{prefix}] {text}")
        return True

    for o in headers[:CONTAINER_CONTEXT_PAGE_SIZE]:
        _emit("HEADER", " ".join(str(o.get("text") or "").split()))

    for ridx in all_row_indices[:CONTAINER_CONTEXT_PAGE_SIZE]:
        text = rows_by_index[ridx]["text"] if ridx in rows_by_index else synthesized_rows[ridx]
        text = " ".join(str(text).split())
        _emit("ROW", text)
        if sum(len(x) + 1 for x in lines) >= TABLE_CONTEXT_MAX_CHARS:
            return "\n".join(lines)

    # Objects with no row_index (loose cells/other fragments) — kept so nothing is dropped.
    for o in other[:CONTAINER_CONTEXT_PAGE_SIZE]:
        _emit("CELL", " ".join(str(o.get("text") or "").split()))
        if sum(len(x) + 1 for x in lines) >= TABLE_CONTEXT_MAX_CHARS:
            return "\n".join(lines)

    for o in list_items[:CONTAINER_CONTEXT_PAGE_SIZE]:
        _emit("LIST_ITEM", " ".join(str(o.get("text") or "").split()))
        if sum(len(x) + 1 for x in lines) >= TABLE_CONTEXT_MAX_CHARS:
            break

    return "\n".join(lines)


def _format_list_context(list_objects: list[dict], matched_obj: dict) -> str:
    """Render all items of a list in document order; the matched item is marked, not
    reordered, since a list's own ordering (e.g. inclusion criteria 1-5) is itself
    semantically meaningful to the verifier."""
    matched_id = matched_obj.get("object_id")
    ordered = sorted(
        (o for o in list_objects if (o.get("text") or "").strip()),
        key=lambda o: o.get("global_position") if isinstance(o.get("global_position"), int) else 0,
    )
    lines: list[str] = []
    seen_text: set[str] = set()
    for o in ordered:
        text = " ".join(str(o.get("text") or "").split())
        norm = _normalize_for_dedup(text)
        if norm and norm in seen_text:
            continue
        if norm:
            seen_text.add(norm)
        marker = "→" if o.get("object_id") == matched_id else "-"
        lines.append(f"{marker} {text}")
    return "\n".join(lines)


def _fetch_table_context(document_id: str, table_ids: list[str], tenant_id: str | None = None, project_id: str | None = None) -> dict[str, list[dict]]:
    """Fetch complete table context for matched table objects in one msearch.

    Uses the canonical table_id relationship; no text matching or geometry work
    is performed here. Only semantic object fields needed by the verifier are
    returned.
    """
    if not document_id or not table_ids:
        return {}

    result: dict[str, list[dict]] = {}
    to_fetch = []
    for table_id in table_ids:
        cached = _TABLE_CACHE.get(table_id)
        if cached is not None:
            result[table_id] = cached
        else:
            to_fetch.append(table_id)
    if not to_fetch:
        return result

    body: list[dict] = []
    # Only fields _format_table_context/_paginate_container_objects actually read —
    # row_index/col_start for grid ordering, global_position as the pagination cursor.
    source_fields = ["global_position", "type", "text", "row_index", "col_start"]
    for table_id in to_fetch:
        body.append({})
        body.append({
            "size": CONTAINER_CONTEXT_PAGE_SIZE,
            "query": {"bool": {"filter": [
                {"term": {"document_id": document_id}},
                {"term": {"table_id": table_id}},
                *([{"term": {"tenant_id": tenant_id}}] if tenant_id else []),
                *([{"term": {"project_id": project_id}}] if project_id else []),

            ]}},
            "_source": source_fields,
            # Sort by global_position only (not row_index) — pagination below cursors on
            # global_position, and _format_table_context regroups by row_index itself, so
            # fetch order carries no rendering meaning.
            "sort": [{"global_position": "asc"}],
        })
    try:
        resp = _get_os().msearch(body=body, index=SEMANTIC_OBJECTS_INDEX)
        fetched: dict[str, list[dict]] = {}
        responses = resp.get("responses", [])
        for i, table_id in enumerate(to_fetch):
            if i >= len(responses):
                fetched[table_id] = []
                continue
            fetched[table_id] = [h.get("_source", {}) for h in responses[i].get("hits", {}).get("hits", [])]
        # CONTAINER_CONTEXT_PAGE_SIZE is a page size, not a hard cap — page past it so "full
        # table" holds even for tables with more objects than the limit.
        fetched = _paginate_container_objects(fetched, document_id, "table_id", CONTAINER_CONTEXT_PAGE_SIZE, tenant_id, project_id)
        logger.info("[Context Expander] table_context tables=%d objects=%d (cached=%d)", len(to_fetch), sum(len(v) for v in fetched.values()), len(table_ids) - len(to_fetch))
        _TABLE_CACHE.update(fetched)
        result.update(fetched)
        return result
    except Exception as exc:
        logger.warning("[Context Expander] table context msearch failed: %s", exc)
        for table_id in to_fetch:
            result[table_id] = []
        return result


def _fetch_list_context(document_id: str, list_ids: list[str], tenant_id: str | None = None, project_id: str | None = None) -> dict[str, list[dict]]:
    """Fetch complete list context for matched list_item objects in one msearch.

    Mirrors _fetch_table_context: uses the canonical list_id relationship, no text
    matching or geometry work.
    """
    if not document_id or not list_ids:
        return {}

    result: dict[str, list[dict]] = {}
    to_fetch = []
    for list_id in list_ids:
        cached = _LIST_CACHE.get(list_id)
        if cached is not None:
            result[list_id] = cached
        else:
            to_fetch.append(list_id)
    if not to_fetch:
        return result

    body: list[dict] = []
    # Only fields _format_list_context/_paginate_container_objects actually read.
    source_fields = ["global_position", "type", "text"]
    for list_id in to_fetch:
        body.append({})
        body.append({
            "size": CONTAINER_CONTEXT_PAGE_SIZE,
            "query": {"bool": {"filter": [
                {"term": {"document_id": document_id}},
                {"term": {"list_id": list_id}},
                *([{"term": {"tenant_id": tenant_id}}] if tenant_id else []),
                *([{"term": {"project_id": project_id}}] if project_id else []),
            ]}},
            "_source": source_fields,
            "sort": [{"global_position": "asc"}],
        })
    try:
        resp = _get_os().msearch(body=body, index=SEMANTIC_OBJECTS_INDEX)
        fetched: dict[str, list[dict]] = {}
        responses = resp.get("responses", [])
        for i, list_id in enumerate(to_fetch):
            if i >= len(responses):
                fetched[list_id] = []
                continue
            fetched[list_id] = [h.get("_source", {}) for h in responses[i].get("hits", {}).get("hits", [])]
        # CONTAINER_CONTEXT_PAGE_SIZE is a page size here too — page past it so "full list"
        # holds even for lists with more objects than the limit.
        fetched = _paginate_container_objects(fetched, document_id, "list_id", CONTAINER_CONTEXT_PAGE_SIZE, tenant_id, project_id)
        logger.info("[Context Expander] list_context lists=%d objects=%d (cached=%d)", len(to_fetch), sum(len(v) for v in fetched.values()), len(list_ids) - len(to_fetch))
        _LIST_CACHE.update(fetched)
        result.update(fetched)
        return result
    except Exception as exc:
        logger.warning("[Context Expander] list context msearch failed: %s", exc)
        for list_id in to_fetch:
            result[list_id] = []
        return result


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


def _fetch_objects_page(
    filter_field:   str,
    filter_value:   str,
    document_id:    str,
    after_position: int,
    tenant_id: str | None = None,
    project_id: str | None = None,
    size:       int = CHUNK_OBJECTS_MAX,
) -> list[dict]:
    """Next page of semantic objects matching `filter_field`, strictly after `after_position`.

    Shared by chunk/table/list pagination — all three page by global_position behind a
    single-term container filter (parent_chunk_id / table_id / list_id respectively).
    """
    try:
        resp = _get_os().search(
            index=SEMANTIC_OBJECTS_INDEX,
            body={
                "size": size,
                "query": {"bool": {"filter": [
                    {"term":  {filter_field: filter_value}},
                    {"range": {"global_position": {"gt": after_position}}},
                    *([{"term": {"document_id": document_id}}] if document_id else []),
                    *([{"term": {"tenant_id": tenant_id}}] if tenant_id else []),
                    *([{"term": {"project_id": project_id}}] if project_id else []),
                ]}},
                "sort": [{"global_position": "asc"}],
            },
        )
        return [h["_source"] for h in resp.get("hits", {}).get("hits", [])]
    except Exception as exc:
        logger.warning("[Context Expander] %s pagination failed %s=%s: %s", filter_field, filter_field, filter_value, exc)
        return []


def _paginate_container_objects(
    initial:     dict[str, list[dict]],
    document_id: str,
    filter_field: str,
    page_size:   int,
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> dict[str, list[dict]]:
    """Page past `page_size` for every container (table/list) whose initial fetch hit the
    cap exactly — so "full table"/"full list" holds even for containers with >page_size
    objects, instead of silently truncating. Stops at CONTAINER_CONTEXT_MAX_OBJECTS total
    per container; formatting truncates well before that anyway, so this only avoids
    fetching/holding objects that would never reach the formatted output.

    Batches ALL containers still needing another page into a single msearch per round
    (round-robin), instead of one sequential _search per container per round — with ~190
    tables needing a 2nd/3rd page on a large document, the old per-container loop meant
    100+ serial network round-trips (the actual S4 wall-time driver, separate from the
    memory issue)."""
    pending = {
        key for key, objs in initial.items()
        if len(objs) == page_size and len(objs) < CONTAINER_CONTEXT_MAX_OBJECTS
    }
    while pending:
        body: list[dict] = []
        keys: list[str] = []
        for key in pending:
            last_pos = initial[key][-1].get("global_position") if initial[key] else None
            if last_pos is None:
                continue
            keys.append(key)
            body.append({})
            body.append({
                "size": page_size,
                "query": {"bool": {"filter": [
                    {"term":  {filter_field: key}},
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
            logger.warning("[Context Expander] %s batched pagination failed: %s", filter_field, exc)
            break

        next_pending: set[str] = set()
        for key, r in zip(keys, responses):
            more = [h["_source"] for h in r.get("hits", {}).get("hits", [])]
            initial[key].extend(more)
            if len(more) == page_size and len(initial[key]) < CONTAINER_CONTEXT_MAX_OBJECTS:
                next_pending.add(key)
        pending = next_pending

    for key, objs in initial.items():
        if len(objs) >= CONTAINER_CONTEXT_MAX_OBJECTS:
            logger.warning(
                "[Context Expander] %s=%s hit CONTAINER_CONTEXT_MAX_OBJECTS=%d — truncating fetch",
                filter_field, key, CONTAINER_CONTEXT_MAX_OBJECTS,
            )
    return initial


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
                "size": CHUNK_OBJECTS_MAX,
                "query": {"bool": {"filter": [
                    {"term": {"parent_chunk_id": key[0]}},
                    *([{"term": {"document_id": document_id}}] if document_id else []),
                    *([{"term": {"tenant_id": tenant_id}}] if tenant_id else []),
                    *([{"term": {"project_id": project_id}}] if project_id else []),
                ]}},
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

        # CHUNK_OBJECTS_MAX is a page size, not a hard cap — page through with search_after-
        # style range queries so a chunk with more objects than the limit isn't silently
        # truncated (fan-out's "every real object becomes a candidate" invariant depends on this).
        by_chunk_id = {key[0]: result.get(key, []) for key in without_pos}
        by_chunk_id = _paginate_container_objects(by_chunk_id, document_id, "parent_chunk_id", CHUNK_OBJECTS_MAX, tenant_id, project_id)
        for key in without_pos:
            result[key] = by_chunk_id[key[0]]

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
