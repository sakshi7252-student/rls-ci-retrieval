"""
Search Pipeline — Stage 2e: Regex Retriever
=============================================
Applies the CI's compiled regex patterns against raw_text.
Best for:  IDENTIFIER (NCT numbers, protocol IDs, patient IDs, dates).

Strategy: fetch candidate chunks via BM25 (broad), then apply Python regex
for precision.  OpenSearch regexp is limited; Python re gives full control.

Input:  classified search request  (ci must have "ontology.regex_patterns")
Output: { "retriever": "regex", "hits": list[Hit] }  # hits include regex_matches: [{text,start,end}]
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_INDEX    = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
OPENSEARCH_MAXSIZE  = int(os.environ.get("OPENSEARCH_MAXSIZE", "256"))  # Connection pool size
AWS_REGION          = os.environ.get("AWS_REGION", "us-east-1")
TOP_K               = int(os.environ.get("RETRIEVER_TOP_K", "10"))
FETCH_SIZE          = int(os.environ.get("REGEX_FETCH_SIZE", "200"))

from shared.opensearch_client import get_opensearch_client

def _get_os():
    return get_opensearch_client()


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Regex Retriever] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Regex Retriever] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Regex Retriever] done search_id=%s hits=%d", search_id, len(result["hits"]))
    return result


def _process(req: dict) -> dict:
    patterns    = req["ci"].get("ontology", {}).get("regex_patterns", [])
    ci_text     = req["ci"].get("knownCI", "")
    document_id = req.get("document_id")
    tenant = req.get("tenant")
    project_id = req.get("project_id")
    tenant_id = tenant.get("tenant_id")

    if not patterns:
        # Fallback: build patterns from raw CI text
        patterns = [re.escape(ci_text)]

    # Compile once — skip invalid patterns
    compiled: list[re.Pattern] = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat, re.IGNORECASE))
        except re.error:
            logger.warning("[Regex Retriever] invalid pattern skipped: %s", pat)

    if not compiled:
        return {"retriever": "regex", "hits": []}

    hits = _regex_search(compiled, document_id, tenant_id=tenant_id, project_id=project_id)

    return {
        "retriever": "regex",
        "hits":      hits,
    }


def _regex_search(
    patterns: list[re.Pattern],
    document_id: str | None,
    tenant_id: str | None = None,
    project_id: str | None = None
) -> list[dict]:
    """Fetch all chunks and apply Python regex to raw_text."""
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []
    if tenant_id:
        filter_clause.append({"term": {"tenant_id": tenant_id}})
    if project_id:
        filter_clause.append({"term": {"project_id": project_id}})

    body = {
        "size": FETCH_SIZE,
        "query": {
            "bool": {
                "filter": filter_clause,
                "must":   [{"match_all": {}}],
            }
        },
        "_source": ["chunk_id", "document_id", "page_start", "page_end", "raw_text"],
    }

    resp = _get_os().search(index=OPENSEARCH_INDEX, body=body)
    hits: list[dict] = []

    for h in resp.get("hits", {}).get("hits", []):
        src      = h.get("_source", {})
        raw_text = src.get("raw_text", "")

        # Count how many patterns match; use match count as score proxy.
        # Collect every match's exact span (not just the first) so context_expander
        # can resolve the containing object the same way it does for literal hits.
        match_count   = 0
        first_snippet = ""
        seen_starts: set[int] = set()
        regex_matches: list[dict] = []
        for pat in patterns:
            for m in pat.finditer(raw_text):
                match_count += 1
                if not first_snippet:
                    start = max(0, m.start() - 80)
                    end   = min(len(raw_text), m.end() + 80)
                    first_snippet = raw_text[start:end]
                if m.start() not in seen_starts:
                    seen_starts.add(m.start())
                    regex_matches.append({"text": m.group(0), "start": m.start(), "end": m.end(), "source": "regex"})

        if match_count > 0:
            regex_matches.sort(key=lambda rm: rm["start"])
            hits.append({
                "chunk_id":      src.get("chunk_id", h["_id"]),
                "score":         float(match_count),
                "page_start":    src.get("page_start", 0),
                "page_end":      src.get("page_end",   0),
                "snippet":       first_snippet[:200],
                "regex_matches": regex_matches,
            })

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:TOP_K]
