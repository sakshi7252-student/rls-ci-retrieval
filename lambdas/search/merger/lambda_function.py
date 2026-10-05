"""Terminal merger.

This stage filters accepted candidates and passes the canonical upstream
geometry object through unchanged. It never calculates, flattens, merges,
or reconstructs PDF geometry.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Merger only filters accepted candidates and passes canonical upstream geometry through.
PAGE_GAP_THRESHOLD = int(os.environ.get("MERGE_PAGE_GAP", "1"))

def _merge_group(ci_id, ci_text: str, group: list[dict]) -> dict:
    """Build a final hit by passing the indexed candidate through unchanged.

    Geometry is never reconstructed, flattened, merged, or inferred here.
    The only authoritative geometry is candidate.matched_object.geometry,
    produced upstream during extraction/chunk construction.
    """
    ordered = sorted(
        group,
        key=lambda c: (
            c.get("page_start", 0),
            c.get("position_in_doc", 0),
            c.get("chunk_id", ""),
        ),
    )
    primary = ordered[0]
    matched_obj = primary.get("matched_object") or primary.get("indexed_object") or {}
    geometry = matched_obj.get("geometry") or primary.get("geometry") or {}

    canonical_text = matched_obj.get("text") or primary.get("text", "") or ""
    context_text = (primary.get("context") or {}).get("current_text") or ""

    chunk_ids = [c.get("chunk_id") for c in ordered if c.get("chunk_id")]

    return {
        "ci_id": ci_id,
        "ci_text": ci_text,
        "page_start": min(c.get("page_start", 0) for c in ordered),
        "page_end": max(c.get("page_end", 0) for c in ordered),
        "text": canonical_text,
        "context_text": context_text,
        "retrieval_heading_path": (
            matched_obj.get("heading_path")
            or matched_obj.get("semantic_path")
            or primary.get("retrieval_heading_path")
            or ""
        ),
        "match_span": primary.get("match_span") or canonical_text,
        "context_sentence": primary.get("context_sentence") or canonical_text,
        "match_page": primary.get("match_page") or primary.get("page_start", 0),
        "retrieval_object_type": matched_obj.get("type") or primary.get("retrieval_object_type"),
        "retrieval_object_id": matched_obj.get("object_id") or primary.get("retrieval_object_id"),
        "geometry": geometry,
        "highlight_mode": primary.get("highlight_mode", "span"),
        "text_search_pages": primary.get("text_search_pages") or [],
        "matched_object": matched_obj,
        "sources": sorted({s for c in ordered for s in c.get("sources", [])}),
        "verdict": "YES" if any(c.get("verdict") == "YES" for c in ordered) else "MAYBE",
        "confidence": round(max(c.get("confidence", 0.0) for c in ordered), 3),
        "chunk_ids": chunk_ids,
        "retrieval_origin": primary.get("retrieval_origin", "direct_unknown"),
        "selection_reason": primary.get("selection_reason"),
        "literal_match_count": primary.get("literal_match_count"),
        "context_strategy": primary.get("context_strategy"),
        "matched_distance": primary.get("matched_distance"),
        "distance_ratio": primary.get("distance_ratio"),
        "current_text_chars": primary.get("current_text_chars"),
        "supporting_sentences": primary.get("supporting_sentences", []),
        "highlight_type": primary.get("highlight_type", "sentence"),
        "primary_support_index": primary.get("primary_support_index", 0),
    }


def _process(req: dict) -> dict:
    ci = req.get("ci", {})
    ci_id = ci.get("id", ci.get("ci_id", ""))
    ci_text = ci.get("knownCI", "")
    verified = req.get("verified_candidates", [])
    accepted = [c for c in verified if c.get("verdict") in ("YES", "MAYBE")]
    final_hits = _build_individual_hits(ci_id, ci_text, accepted)
    return {**req, "final_hits": final_hits}


def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Merger] start search_id=%s", search_id)
    result = _process(event)
    logger.info("[Merger] done search_id=%s final_hits=%d",
                search_id, len(result["final_hits"]))
    return result


def _build_individual_hits(ci_id, ci_text: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    ordered = sorted(
        candidates,
        key=lambda c: (
            c.get("page_start", 0),
            c.get("position_in_doc", 0),
            c.get("chunk_id", ""),
        ),
    )
    return [_merge_group(ci_id, ci_text, [candidate]) for candidate in ordered]
