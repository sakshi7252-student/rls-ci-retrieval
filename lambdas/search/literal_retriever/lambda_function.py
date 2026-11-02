"""
Search Pipeline — Stage 2a: Literal Retriever
===============================================
Finds chunks via exact phrase match and fuzzy match on ``raw_text``.
Best for:  PERSON names, verbatim CI text.

Input:  classified search request  (must have "classification")
Output: { "retriever": "literal", "hits": list[Hit] }

Hit schema
----------
{ "chunk_id": str, "score": float, "page_start": int, "page_end": int, "snippet": str }
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_INDEX    = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
AWS_REGION          = os.environ.get("AWS_REGION", "us-east-1")
TOP_K               = int(os.environ.get("RETRIEVER_TOP_K", "10"))
# Literal retriever returns ALL exact matches (up to LITERAL_MAX).
# Exact/phrase hits are always evidence — never cap on a hard k.
LITERAL_MAX         = int(os.environ.get("LITERAL_MAX", "200"))

from shared.opensearch_client import get_opensearch_client

def _get_os():
    return get_opensearch_client()


# Fold typographic quote variants to ASCII so substring matching agrees with
# OpenSearch's analyzer (which treats curly/straight apostrophes the same).
# 1:1 char mapping — never changes string length, so match offsets stay valid.
_QUOTE_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u2032": "'", "\u00b4": "'",
    "\u201c": '"', "\u201d": '"',
})


def _fold_quotes(text: str) -> str:
    return text.translate(_QUOTE_FOLD)


# Same idea for dash variants — a CI authored with a plain hyphen should
# still match raw_text rendered with a typographic en/em-dash or minus sign.
# Also 1:1, offsets stay valid.
_DASH_FOLD = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-",
})


def _fold_dashes(text: str) -> str:
    return text.translate(_DASH_FOLD)


# str.lower() is NOT always length-preserving (e.g. "İ".lower() == "i̇", 2
# chars) — used on raw_folded this would desync raw_lower from raw_index_map
# (sized off raw_folded) and later index_map[b_start] lookups raise
# IndexError. Lower char-by-char and keep only the first resulting char so
# length always matches the 1:1 index map built upstream.
def _lower_preserve_len(text: str) -> str:
    return ''.join((ch.lower() or ch)[:1] for ch in text)


# Unicode allows the same visible character to be encoded as one precomposed
# codepoint (NFC, e.g. "é") or a base letter plus a combining mark (NFD, e.g.
# "e" + U+0301). Two textually-identical strings can differ this way, which
# would silently defeat substring matching. Unlike quote/dash folding this is
# NOT length-preserving (NFD forms are longer), so we can't just re-use
# raw_text offsets directly — instead build an explicit map from each
# character in the normalized string back to its origin offset in raw_text,
# so every strategy below can keep returning correct spans into the ORIGINAL
# raw_text no matter how normalization reshuffled character counts.
def _build_normalized_map(text: str) -> tuple[str, list[int]]:
    normalized = unicodedata.normalize('NFC', text)
    if normalized == text:
        return normalized, list(range(len(text)))
    index_map = [0] * len(normalized)
    matcher = difflib.SequenceMatcher(None, text, normalized, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(j2 - j1):
                index_map[j1 + k] = i1 + k
        else:
            for k in range(j1, j2):
                index_map[k] = i1
    return normalized, index_map


# Rather than special-casing every possible decoration ci_text can carry
# (numbering, bullets, quotes, trailing punctuation, section labels, name
# initials, ...), find the longest contiguous span shared between ci_text and
# raw_text and use that — this naturally absorbs ANY prefix/suffix mismatch
# and also surfaces genuine PARTIAL matches when only part of ci_text is
# actually present in raw_text.
_MIN_PARTIAL_CHARS = 3      # floor so we never report a trivial common word
_MIN_PARTIAL_RATIO = 0.5    # span must cover at least half of the (sub)phrase


def _longest_common_span(needle_lower: str, haystack_lower: str):
    matcher = difflib.SequenceMatcher(None, needle_lower, haystack_lower, autojunk=False)
    return matcher.find_longest_match(0, len(needle_lower), 0, len(haystack_lower))


# Spacing/wrapping punctuation varies a lot between how a CI is authored and
# how the source document renders it ("n=62" vs "n = 62" vs "(N = 62)") even
# though the content is identical. Compare with whitespace and wrapping
# punctuation removed so this never has to fall through to the weaker
# fuzzy/partial strategies just because of formatting, then map the found
# span back to real raw_text offsets (including whatever original formatting
# raw_text had there).
def _is_ignorable_format_char(ch: str) -> bool:
    return ch.isspace() or ch in '()[]{}'


def _find_ignoring_whitespace(
    needle_lower: str,
    haystack_lower: str,
    haystack_index_map: list[int] | None = None,
) -> tuple[int, int] | None:
    if haystack_index_map is None:
        haystack_index_map = list(range(len(haystack_lower)))
    compact_chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(haystack_lower):
        if not _is_ignorable_format_char(ch):
            compact_chars.append(ch)
            index_map.append(haystack_index_map[i])
    compact_haystack = ''.join(compact_chars)
    compact_needle = ''.join(ch for ch in needle_lower if not _is_ignorable_format_char(ch))
    if not compact_needle:
        return None
    idx = compact_haystack.find(compact_needle)
    if idx < 0:
        return None
    start = index_map[idx]
    end = index_map[idx + len(compact_needle) - 1] + 1
    return start, end


# Every strategy locates a match as a (start, size) pair in raw_lower's own
# index space; this maps that back to real offsets in the original raw_text
# via the index map produced by _build_normalized_map.
def _map_span(index_map: list[int], b_start: int, size: int) -> tuple[int, int]:
    start = index_map[b_start]
    end = index_map[b_start + size - 1] + 1
    return start, end


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Literal Retriever] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Literal Retriever] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Literal Retriever] done search_id=%s hits=%d", search_id, len(result["hits"]))
    return result


def _process(req: dict) -> dict:
    ci_text     = req["ci"].get("knownCI", "")
    norm_text   = req["ci"].get("normalization", {}).get("normalized_text", ci_text)
    document_id = req.get("document_id")
    tenant = req.get("tenant")
    project_id = req.get("project_id")
    tenant_id = tenant.get("tenant_id")

    hits = _literal_search(ci_text, norm_text, document_id, tenant_id=tenant_id, project_id=project_id)

    return {
        "retriever": "literal",
        "hits":      hits,
    }


def _extract_literal_matches(ci_text: str, raw_text: str) -> list[dict]:
    """
    Find where ci_text (or its significant sub-phrases) appear in raw_text,
    exactly or partially.

    Returns [{"text": str, "start": int, "end": int}, ...] sorted by position.
    Passed forward so Stage 6.5 can surface the exact matched term instead of
    re-discovering it through the scorer registry.

    Strategy 1 — whole phrase, exact substring match.
    Strategy 2 — whole phrase, longest common substring (handles decoration
                  mismatches and genuine partial matches).
    Strategy 3 — sub-phrases: split on commas/semicolons/newlines, search each
                  phrase ≥ 5 chars independently (exact, then fuzzy).
    """
    matches: list[dict] = []
    # Guard against None/non-str/blank inputs (upstream callers, malformed CI
    # records) — without this, ci_text="" spuriously "matches" every hit at
    # position 0, and None/non-str crashes .strip()/.translate() below.
    if not isinstance(ci_text, str) or not isinstance(raw_text, str):
        return matches
    if not ci_text.strip() or not raw_text:
        return matches

    # Normalize Unicode form (NFD -> NFC) before comparing, so an accented
    # character encoded differently in ci_text vs raw_text still matches.
    # This can change raw_text's effective length, so raw_index_map maps
    # every position below back to the real raw_text offset it came from.
    raw_normalized, raw_index_map = _build_normalized_map(raw_text)

    # Fold quote/dash variants before comparing so "Alzheimer's" (straight)
    # still matches "Alzheimer's" (curly), and a hyphen still matches an
    # en-/em-dash — OpenSearch's match_phrase already treats these as
    # equivalent, so the extractor must too or it silently returns no
    # literal_matches for a hit it just found. Both folds are 1:1, so
    # raw_index_map stays valid.
    raw_folded = _fold_dashes(_fold_quotes(raw_normalized))
    raw_lower  = _lower_preserve_len(raw_folded)

    ci_s = _fold_dashes(_fold_quotes(unicodedata.normalize('NFC', ci_text.strip())))
    ci_lower = ci_s.lower()

    # Strategy 1 — whole phrase, exact substring
    idx  = raw_lower.find(ci_lower)
    if idx >= 0:
        start, end = _map_span(raw_index_map, idx, len(ci_lower))
        return [{"text": raw_text[start:end], "start": start, "end": end, "match_id": f"literal_{start}_{end}"}]

    # Strategy 1b — whole phrase, exact match but ignoring whitespace
    # differences ("n=62" vs "n = 62") and wrapping punctuation.
    span_ws = _find_ignoring_whitespace(ci_lower, raw_lower, raw_index_map)
    if span_ws is not None:
        start, end = span_ws
        return [{"text": raw_text[start:end], "start": start, "end": end, "match_id": f"literal_{start}_{end}"}]

    # Strategy 2 — whole phrase, longest common substring. Absorbs any kind of
    # decoration mismatch (markers, quotes, labels, punctuation, initials...)
    # without needing to enumerate what the decoration looks like, and also
    # surfaces a genuine partial match when only part of ci_text is present.
    span = _longest_common_span(ci_lower, raw_lower)
    min_len = max(_MIN_PARTIAL_CHARS, int(len(ci_lower) * _MIN_PARTIAL_RATIO))
    if span.size >= min_len:
        start, end = _map_span(raw_index_map, span.b, span.size)
        matches.append({"text": raw_text[start:end], "start": start, "end": end, "match_id": f"literal_{start}_{end}"})

    # Strategy 3 — significant sub-phrases, exact then fuzzy. Only meaningful
    # when ci_text actually splits into more than one clause — a single-clause
    # ci_text would just re-run Strategy 2 against the identical whole string.
    sub_phrases = [p.strip() for p in re.split(r'[,;\n]+', ci_s) if len(p.strip()) >= 5]
    if len(sub_phrases) > 1:
        for phrase in sub_phrases:
            phrase_lower = phrase.lower()
            idx = raw_lower.find(phrase_lower)
            if idx >= 0:
                start, end = _map_span(raw_index_map, idx, len(phrase_lower))
                matches.append({"text": raw_text[start:end], "start": start, "end": end, "match_id": f"literal_{start}_{end}"})
                continue
            span_ws = _find_ignoring_whitespace(phrase_lower, raw_lower, raw_index_map)
            if span_ws is not None:
                start, end = span_ws
                matches.append({"text": raw_text[start:end], "start": start, "end": end, "match_id": f"literal_{start}_{end}"})
                continue
            span = _longest_common_span(phrase_lower, raw_lower)
            min_len = max(_MIN_PARTIAL_CHARS, int(len(phrase_lower) * _MIN_PARTIAL_RATIO))
            if span.size >= min_len:
                start, end = _map_span(raw_index_map, span.b, span.size)
                matches.append({"text": raw_text[start:end], "start": start, "end": end, "match_id": f"literal_{start}_{end}"})

    # Deduplicate overlapping spans, keep leftmost
    seen: set[int] = set()
    unique: list[dict] = []
    for m in sorted(matches, key=lambda x: x["start"]):
        if m["start"] not in seen:
            seen.add(m["start"])
            unique.append(m)
    return unique


def _literal_search(ci_text: str, norm_text: str, document_id: str | None, tenant_id: str | None = None, project_id: str | None = None) -> list[dict]:
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []
    if tenant_id:
        filter_clause.append({"term": {"tenant_id": tenant_id}})
    if project_id:
        filter_clause.append({"term": {"project_id": project_id}})

    body = {
        "size": LITERAL_MAX,
        "query": {
            "bool": {
                "filter": filter_clause,
                "should": [
                    # Exact phrase match on raw text (highest weight)
                    {
                        "match_phrase": {
                            "raw_text": {
                                "query": ci_text,
                                "boost": 3.0,
                            }
                        }
                    },
                    # Phrase on normalized text
                    {
                        "match_phrase": {
                            "normalized_text": {
                                "query": norm_text,
                                "boost": 2.0,
                            }
                        }
                    },
                    # Fuzzy match for OCR errors / typos (short CI texts only —
                    # long texts expand to thousands of term variations and hit
                    # OpenSearch's 1024 maxClauseCount limit).
                    *([{
                        "match": {
                            "raw_text": {
                                "query":     ci_text,
                                "fuzziness": "AUTO",
                                "boost":     1.0,
                            }
                        }
                    }] if len(ci_text) <= 50 else []),
                ],
                "minimum_should_match": 1,
            }
        },
        "_source": ["chunk_id", "document_id", "page_start", "page_end", "raw_text"],
    }

    resp = _get_os().search(index=OPENSEARCH_INDEX, body=body)
    return _parse_hits(resp, ci_text)


def _parse_hits(resp: dict, ci_text: str = "") -> list[dict]:
    hits = []
    for h in resp.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        raw = src.get("raw_text", "")
        literal_matches = _extract_literal_matches(ci_text, raw) if ci_text else []
        for lm in literal_matches:
            lm["source"] = "literal"
        hits.append({
            "chunk_id":        src.get("chunk_id", h["_id"]),
            "score":           round(h.get("_score", 0.0), 4),
            "page_start":      src.get("page_start", 0),
            "page_end":        src.get("page_end",   0),
            "snippet":         raw[:200],
            "literal_matches": literal_matches,
        })
    return hits
