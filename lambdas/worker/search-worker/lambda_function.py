"""
Search Worker Lambda
====================
Processes a batch of pre-enriched CIs against a document using the
stage-parallel pipeline (mirrors search_new.py, no file I/O).

Input
-----
{
    "search_id":        str,
    "batch_idx":        int,
    "cis":              list[dict],   # enriched CIs (from ci-objects index)
    "document_id":      str,
    "tenant":           dict,
    "project_id":        str,
    "document_context": dict,
    "skip_rerank":      bool,
    "skip_verify":      bool,
    "workers":          int           # per-stage thread count (default: len(cis))
}

Output
------
{
    "search_id":  str,
    "batch_idx":  int,
    "results":    list[dict],         # per-CI dict with final_hits + timings
    "stage_wall": dict[str, float]
}

Env vars
--------
  OPENSEARCH_ENDPOINT     — host only (no https://)
  OPENSEARCH_INDEX        — default: document-chunks
  SEMANTIC_OBJECTS_INDEX  — default: semantic-objects
  OPENSEARCH_CI_INDEX     — default: ci-objects
  AWS_REGION
  VERIFIER_MODEL          — Bedrock model ID for LLM verifier + evidence classifier
"""

from __future__ import annotations

import ctypes
import gc
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from datetime import datetime
from pathlib import Path
from typing import Any


# ── Logging Context Variables (thread-safe for concurrent execution) ───────
_ctx_tenant = ContextVar("tenant", default="-")
_ctx_document_id = ContextVar("document_id", default="-")
_ctx_search_id = ContextVar("search_id", default="-")
_ctx_batch_idx = ContextVar("batch_idx", default="-")
_ctx_ci_id = ContextVar("ci_id", default="-")


class SearchContextFilter(logging.Filter):
    """Logging filter that injects search context into all log records.
    
    Automatically adds [tenant=...] [document=...] [search=...] [batch=...] [ci=...]
    prefix to every log from any module, without requiring manual propagation.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        tenant = _ctx_tenant.get()
        document_id = _ctx_document_id.get()
        search_id = _ctx_search_id.get()
        batch_idx = _ctx_batch_idx.get()
        ci_id = _ctx_ci_id.get()
        
        prefix = f"[tenant={tenant}] [document={document_id}] [search={search_id}]"
        if batch_idx != "-":
            prefix += f" [batch={batch_idx}]"
        if ci_id != "-":
            prefix += f" [ci={ci_id}]"
        
        record.msg = f"{prefix} {record.msg}"
        return True


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configure root logger so all dynamically loaded modules inherit the context filter
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Add context filter to all existing handlers (preserve Lambda's CloudWatch handler)
context_filter = SearchContextFilter()
for handler in root_logger.handlers:
    handler.addFilter(context_filter)

# If no handlers exist, create one (e.g., local testing)
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
    )
    handler.setFormatter(formatter)
    handler.addFilter(context_filter)
    root_logger.addHandler(handler)

_task_root_env = os.environ.get("LAMBDA_TASK_ROOT")
if _task_root_env and (Path(_task_root_env) / "lambdas").exists():
    ROOT = Path(_task_root_env)
else:
    ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── LLM cost estimation constants (Anthropic Haiku-4.5 base rates) ──────────
_HAIKU_INPUT_PRICE_PER_TOKEN  = 0.80 / 1_000_000   # USD per token
_HAIKU_OUTPUT_PRICE_PER_TOKEN = 4.00 / 1_000_000   # USD per token
# llm_verifier: full structured prompt with CI text, doc profile, excerpt, identity dims
# Output includes verdict + reason (~15 words) + confidence + 7-field identity block
_EST_INPUT_TOKENS_PER_CAND    = 418   # measured ~408 from real runs
_EST_OUTPUT_TOKENS_PER_CAND   = 85    # was 30 — identity block (same_drug/study/…) adds ~55 tokens
# _classify_evidence: CI text + match_span (400 chars) + 9-label classification list
# Output is short JSON: evidence_type + confidence + reason (~15 words)
_EST_EC_INPUT_TOKENS_PER_HIT  = 400   # was 300 — measured ~394 (long label list)
_EST_EC_OUTPUT_TOKENS_PER_HIT = 40    # was 60  — measured ~41 (shorter response)


# ── Env config ─────────────────────────────────────────────────────────────────
OPENSEARCH_ENDPOINT    = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_INDEX       = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
OPENSEARCH_CI_INDEX    = os.environ.get("OPENSEARCH_CI_INDEX", "ci-objects")
OPENSEARCH_TIMEOUT     = int(os.environ.get("OPENSEARCH_TIMEOUT", "30"))
OPENSEARCH_MAXSIZE     = int(os.environ.get("OPENSEARCH_MAXSIZE", "256"))  # Connection pool size (RETRIEVER_WORKERS × max concurrent CIs + buffer)
AWS_REGION             = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_REGION         = os.environ.get("BEDROCK_REGION", AWS_REGION)
VERIFIER_MODEL         = os.environ.get("VERIFIER_MODEL",
                                         "eu.anthropic.claude-haiku-4-5-20251001-v1:0")
EMBEDDING_MODEL        = os.environ.get("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
# Concurrency control (prevent overwhelming OpenSearch with nested thread pools)
SEARCH_CI_WORKERS      = int(os.environ.get("SEARCH_CI_WORKERS", "5"))      # CIs per Worker
RETRIEVER_WORKERS      = int(os.environ.get("RETRIEVER_WORKERS", "4"))
SEARCH_FLOW_DEBUG      = os.environ.get("SEARCH_FLOW_DEBUG", "true").lower() == "true"      # Retrievers per CI
SEARCH_RESULTS_DEBUG_BUCKET = os.environ.get("SEARCH_RESULTS_DEBUG_BUCKET", "rls-file-bucket-eu")
RESULTS_DEBUG_PREFIX   = os.environ.get("RESULTS_DEBUG_PREFIX", "search-results")
# Leaves headroom under the hard 6,291,556 byte Lambda sync-invoke response cap.
WORKER_RESPONSE_INLINE_LIMIT_BYTES = int(os.environ.get("WORKER_RESPONSE_INLINE_LIMIT_BYTES", "5000000"))
# Identifies this execution environment (set once at import/cold-start) so
# CloudWatch logs can tell a reused warm container from a fresh one.
EXECUTION_ENV_ID = uuid.uuid4().hex
COLD_START_TS = datetime.now().isoformat()
# ── Lazy singletons ────────────────────────────────────────────────────────────
_loaded: dict[str, types.ModuleType] = {}
_loaded_fresh: dict[tuple[str, int], types.ModuleType] = {}  # per-thread cache for _load_fresh
_evidence_classifier = None  # Lazy load for evidence classification


def _get_evidence_classifier():
    """Lazy-load evidence classification module."""
    global _evidence_classifier
    if _evidence_classifier is None:
        _evidence_classifier = _load("search/evidence_classification", "evidence_classifier")
    return _evidence_classifier


def _get_os():
    from shared.opensearch_client import get_opensearch_client
    return get_opensearch_client()


def _load(rel_path: str, alias: str) -> types.ModuleType:
    if alias in _loaded:
        return _loaded[alias]
    lf_path = ROOT / "lambdas" / rel_path / "lambda_function.py"
    spec    = importlib.util.spec_from_file_location(alias, lf_path)
    mod     = importlib.util.module_from_spec(spec)
    lf_dir  = str(lf_path.parent)
    if lf_dir not in sys.path:
        sys.path.insert(0, lf_dir)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    _loaded[alias] = mod
    return mod


def _load_fresh(rel_path: str) -> types.ModuleType:
    """One exec'd module per thread, cached — re-execing per CI leaked a boto3
    client + module globals on every call, compounding across warm invocations."""
    cache_key = (rel_path, threading.get_ident())
    if cache_key in _loaded_fresh:
        return _loaded_fresh[cache_key]
    lf_path = ROOT / "lambdas" / rel_path / "lambda_function.py"
    alias   = f"_fresh_{rel_path.replace('/', '_')}_{threading.get_ident()}"
    spec    = importlib.util.spec_from_file_location(alias, lf_path)
    mod     = importlib.util.module_from_spec(spec)
    lf_dir  = str(lf_path.parent)
    if lf_dir not in sys.path:
        sys.path.insert(0, lf_dir)
    spec.loader.exec_module(mod)
    _loaded_fresh[cache_key] = mod
    return mod


def _inject_os(mod: types.ModuleType) -> None:
    mod.OPENSEARCH_ENDPOINT    = OPENSEARCH_ENDPOINT
    mod.OPENSEARCH_INDEX       = OPENSEARCH_INDEX
    mod.SEMANTIC_OBJECTS_INDEX = SEMANTIC_OBJECTS_INDEX
    mod.AWS_REGION             = AWS_REGION
    if hasattr(mod, "_os_client"):
        mod._os_client = _get_os()


# ── Retriever map ──────────────────────────────────────────────────────────────
RETRIEVER_MAP: dict[str, str] = {
    "literal":  "search/literal_retriever",
    "bm25":     "search/bm25_retriever",
    "vector":   "search/vector_retriever",
    "ner":      "search/ner_retriever",
    "fact":     "search/fact_retriever",
    "ontology": "search/ontology_retriever",
    "regex":    "search/regex_retriever",
    "numeric":  "search/numeric_retriever",
}

# ── Evidence helpers ───────────────────────────────────────────────────────────
_EVIDENCE_RANK: dict[str, int] = {
    "DIRECT": 0, "SUPPORTING": 1,
    "RELATED_OBJECTIVE": 2, "RELATED_PROTOCOL": 2,
    "RELATED_DOSE": 3, "RELATED_POPULATION": 3,
    "RELATED_SAFETY": 3, "RELATED_EFFICACY": 3,
    "RELATED_DEFINITION": 4,
    "SAME_STUDY": 1, "SAME_PROTOCOL": 2, "SAME_OBJECTIVE": 2,
    "SAME_ENDPOINT": 3, "SAME_POPULATION": 3, "SAME_MECHANISM": 3,
    "BACKGROUND": 4, "UNRELATED": 9,
}

_CONF_THRESHOLD = 0.2

# Disagreement router (S6 post-verify): pre-LLM retrieval strength (raw agg_score, not
# _candidate_confidence() — that formula floors at ~0.5 regardless of how weak agg_score
# is) vs the LLM verdict. Agreement is left untouched; only the two disagreement
# quadrants are routed to MAYBE so neither signal is ever trusted alone:
#   strong retrieval + LLM NO  -> MAYBE (surfaces a possible false negative for review)
#   weak retrieval   + LLM YES -> MAYBE (catches unsupported LLM accepts, e.g. agg_score
#                                  near zero with a confident YES)
# Named by threshold role, not as a confidence/probability — agg_score is an unbounded
# hybrid retrieval score, not a 0-1 probability.
_DISAGREEMENT_HIGH_RETRIEVAL_THRESHOLD = float(os.environ.get("DISAGREEMENT_HIGH_RETRIEVAL_THRESHOLD", "0.5"))
_DISAGREEMENT_LOW_RETRIEVAL_THRESHOLD  = float(os.environ.get("DISAGREEMENT_LOW_RETRIEVAL_THRESHOLD",  "0.05"))


def _is_related(ev: str) -> bool:
    return ev.startswith("SAME_") or ev.startswith("RELATED_") or ev == "BACKGROUND"


def _candidate_confidence(c: dict) -> float:
    return round(
        0.5 * min(max(c.get("agg_score", 0.0) * 2.0, 0.0), 1.0)
        + 0.3 * max(0.0, 1.0 + c.get("zero_id_pen",    0.0) / 0.4)
        + 0.2 * max(0.0, 1.0 + c.get("zero_enrich_pen", 0.0) / 0.25),
        3,
    )


def _calibrate_evidence(hit: dict, ec: dict) -> dict:
    mq     = hit.get("highlight_score", 1.0)
    method = hit.get("match_method", "")
    ev_t   = ec.get("evidence_type", "RELATED_EFFICACY")
    ev_c   = ec.get("evidence_confidence", 0.5)
    result = dict(ec)
    if method in ("text_fallback", "text_fallback_skipped") and mq < 0.15 and ev_t.startswith("RELATED_"):
        result["evidence_type"]   = "SUPPORTING"
        result["evidence_reason"] = result.get("evidence_reason", "") + f" [downgraded: mq={mq:.3f}]"
    if method == "text_fallback_skipped":
        qf = 0.25
    elif mq > 0:
        qf = mq / (mq + 0.05)
    else:
        qf = 0.10
    result["evidence_confidence"] = round(ev_c * qf, 3)
    return result


# ── Stage functions ────────────────────────────────────────────────────────────

def _s1_classify(req: dict) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        return req
    t0 = time.perf_counter()
    ci = req.get("ci", {})
    logger.info(
        "[SearchFlow] S1_START search_id=%s ci_idx=%s ci_id=%s ci_type_before=%s "
        "text=%r embedding_dim=%s",
        req.get("search_id"), req.get("_ci_idx"), ci.get("id"),
        ci.get("category") or ci.get("type"),
        ci.get("knownCI", "")[:300],
        len((ci.get("embedding") or {}).get("dense_vector", []) or []),
    )
    mod = _load("search/classifier", "search_classifier")
    req = mod._process(req)
    req["_st"]["classifier"] = round(time.perf_counter() - t0, 3)
    cls = req.get("classification") or {}
    logger.info(
        "[SearchFlow] S1_DONE search_id=%s ci_id=%s elapsed_s=%.3f ci_type=%s "
        "strategies=%s reason=%r",
        req.get("search_id"), ci.get("id"), req["_st"]["classifier"],
        cls.get("ci_type"), cls.get("strategies", []), cls.get("reason"),
    )
    return req


def _s2_retrieve(req: dict) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        return req
    classification = req.get("classification") or {}
    strategies = classification.get("strategies", list(RETRIEVER_MAP.keys()))
    logger.info(
        "[SearchFlow] S2_START search_id=%s ci_id=%s ci_type=%s strategies=%s "
        "retriever_workers=%s",
        req.get("search_id"), req.get("ci", {}).get("id"),
        classification.get("ci_type"), strategies, RETRIEVER_WORKERS,
    )
    valid = [(s, RETRIEVER_MAP[s]) for s in strategies if s in RETRIEVER_MAP]
    if not valid:
        req["retriever_results"]       = []
        req["_st"]["retrievers"]       = {}
        req["_st"]["retrievers_total"] = 0.0
        return req

    def _run_one(s_path: tuple[str, str]) -> tuple[str, dict, float]:
        strategy, path = s_path
        mod = _load(path, f"search_{strategy}")
        _inject_os(mod)
        t0 = time.perf_counter()
        logger.info(
            "[SearchFlow] RETRIEVER_START search_id=%s ci_id=%s ci_type=%s "
            "strategy=%s module=%s",
            req.get("search_id"), req.get("ci", {}).get("id"),
            (req.get("classification") or {}).get("ci_type"),
            strategy, path,
        )
        try:
            result = mod._process(req)
        except Exception as exc:
            logger.error(
                "[SearchFlow] RETRIEVER_ERROR search_id=%s ci_id=%s strategy=%s "
                "module=%s elapsed_s=%.3f error_type=%s error=%s",
                req.get("search_id"), req.get("ci", {}).get("id"),
                strategy, path, time.perf_counter()-t0,
                type(exc).__name__, exc, exc_info=True,
            )
            raise
        elapsed = round(time.perf_counter() - t0, 3)
        logger.info(
            "[SearchFlow] RETRIEVER_DONE search_id=%s ci_id=%s strategy=%s "
            "elapsed_s=%.3f hits=%d keys=%s",
            req.get("search_id"), req.get("ci", {}).get("id"),
            strategy, elapsed, len(result.get("hits", [])),
            sorted(result.keys()),
        )
        return strategy, result, elapsed

    # Propagate ContextVar to nested retriever threads so [ci=...] appears in their logs
    with ThreadPoolExecutor(max_workers=min(RETRIEVER_WORKERS, len(valid))) as pool:
        futures = [
            pool.submit(copy_context().run, _run_one, s_path)
            for s_path in valid
        ]
        raw = [f.result() for f in futures]

    timings: dict[str, float] = {}
    retriever_results = []
    for strategy, result, elapsed in raw:
        timings[strategy] = elapsed
        # Merge vector sub-timings (body/heading/chunk) directly into the timings dict
        for k_sub, v_sub in (result.pop("_sub_timings", None) or {}).items():
            timings[k_sub] = v_sub
        retriever_results.append(result)
    req["retriever_results"]       = retriever_results
    req["_st"]["retrievers"]       = timings
    req["_st"]["retrievers_total"] = round(sum(timings.values()), 3)
    logger.info(
        "[SearchFlow] S2_DONE search_id=%s ci_id=%s retrievers=%s "
        "retriever_timings=%s total_reported_s=%.3f",
        req.get("search_id"), req.get("ci", {}).get("id"),
        [r.get("retriever") for r in retriever_results],
        timings, req["_st"]["retrievers_total"],
    )
    return req


def _s3_aggregate(req: dict) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        return req
    t0  = time.perf_counter()
    mod = _load("search/aggregator", "search_aggregator")
    req = mod._process(req)
    req["_st"]["aggregator"] = round(time.perf_counter() - t0, 3)
    if not req.get("candidates"):
        req["_early_exit"] = True
        req.setdefault("final_hits", [])
        req.setdefault("verified_candidates", [])
    return req


def _s3_5_ground_evidence(req: dict) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        return req
    t0  = time.perf_counter()
    mod = _load("search/evidence_grounder", "search_evidence_grounder")
    _inject_os(mod)
    req = mod._process(req)
    req["_st"]["evidence_grounder"] = round(time.perf_counter() - t0, 3)
    return req


def _s4_context_expand(req: dict) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        return req
    t0  = time.perf_counter()
    mod = _load("search/context_expander", "search_expander")
    _inject_os(mod)
    req = mod._process(req)
    req["_st"]["context_expander"] = round(time.perf_counter() - t0, 3)
    return req


def _dedupe_by_object_id(candidates: list[dict]) -> list[dict]:
    """Merge candidates that resolve to the same semantic object.

    S3 (aggregator) clusters purely on `matched_object` as seen at retrieval
    time, so a literal/regex hit with no `matched_object` yet gets its own
    `chunk:<chunk_id>` cluster. S4 (context_expander) later resolves that
    hit's containing object from `literal_matches`, which can make it equal
    to an object-level bm25/vector candidate that S3 already clustered
    separately under `obj:<object_id>`. Re-group here, now that every
    candidate's object identity is known, and combine their sources/proof.
    """
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for c in candidates:
        key = (c.get("matched_object") or {}).get("object_id") or c.get("chunk_id") or c.get("id") or ""
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(c)

    merged: list[dict] = []
    for key in order:
        group = groups[key]
        if len(group) == 1:
            merged.append(group[0])
            continue
        # Prefer the candidate carrying literal proof as the base representation.
        base = max(group, key=lambda c: (bool(c.get("literal_matches")), c.get("agg_score", 0)))
        sources = sorted({s for c in group for s in (c.get("sources") or [])})
        seen_starts, literal_matches = set(), []
        for c in group:
            for lm in c.get("literal_matches") or []:
                start = lm.get("start")
                if start in seen_starts:
                    continue
                seen_starts.add(start)
                literal_matches.append(lm)
        merged.append({
            **base,
            "sources": sources,
            "literal_matches": literal_matches,
            "agg_score": max(c.get("agg_score", 0) for c in group),
        })
    return merged


def _run_comparators_only(req: dict) -> None:
    """Populate score_breakdown.contra_detail without loading the CE model.

    _detect_semantic_conflicts/_score_semantic_conflicts are pure Python
    (drug/endpoint/statistical/etc. comparators) — no sentence-transformers
    import happens unless _get_ce_model() is called, which we never do here.
    """
    reranker = _load("search/reranker", "search_reranker")
    ci = req.get("ci") or {}
    ci_entities = ci.get("ner", {}).get("entities", [])
    ci_ctx = reranker._build_ci_context(ci, ci_entities)
    for c in req.get("ranked_candidates", []):
        if not c.get("matched_object"):
            continue
        cand_ctx = reranker._build_cand_context(c)
        vr = reranker._detect_semantic_conflicts(ci_ctx, cand_ctx)
        _, contra_detail = reranker._score_semantic_conflicts(vr)
        c["score_breakdown"] = {**c.get("score_breakdown", {}), "contra_detail": contra_detail}


def _s5_rerank(req: dict, skip_rerank: bool = False) -> dict:
    """Reranking stage — the CE model is always skipped for cold start reduction,
    but the deterministic comparator pipeline (drug/endpoint/statistical/etc.
    conflict detection) still runs since it costs no model load, just Python."""
    if req.get("_failed") or req.get("_early_exit"):
        return req
    t0 = time.perf_counter()
    # Skip CE model entirely — set uniform score above MIN_RERANK_SCORE (3.0)
    # so all candidates reach Claude regardless of comparator findings below.
    expanded = req.get("expanded_candidates", [])
    req["ranked_candidates"] = [
        {**c, "cross_encoder_score": 10.0}
        for c in expanded
    ]
    req["ranked_candidates"] = _dedupe_by_object_id(req["ranked_candidates"])
    _run_comparators_only(req)
    req["_st"]["reranker"] = round(time.perf_counter() - t0, 3)

    # 5.6 Confidence gate
    # Hard invariant: no semantic object -> no LLM call, no final_hit. An unresolved
    # candidate (Evidence Grounder couldn't anchor it to a real semantic object) has no
    # matched_object and therefore no geometry to highlight in the UI — sending its raw
    # chunk text to the LLM only risks producing a YES/MAYBE verdict with nothing
    # displayable behind it. Gate it out here, before it ever reaches S6.
    passed, gated = [], []
    for c in req.get("ranked_candidates", []):
        if not c.get("matched_object"):
            gated.append({**c, "verdict": "SKIP", "reason": "unresolved_no_semantic_object", "llm_verified": False})
        elif _candidate_confidence(c) >= _CONF_THRESHOLD:
            passed.append(c)
        else:
            gated.append({**c, "verdict": "SKIP", "reason": "candidate_confidence_gate", "llm_verified": False})
    req["ranked_candidates"] = passed
    req.setdefault("skipped_hits", []).extend(gated)
    return req


def _s6_llm_verify(req: dict, skip_verify: bool = False) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        req["_st"].update({
            "n_candidates_to_verifier": 0,
            "per_verifier_call_s": {},
            "actual_verifier_tokens": {"input": 0, "output": 0},
            "llm_verifier": 0.0,
        })
        return req
    req["_st"]["n_candidates_to_verifier"] = len(req.get("ranked_candidates", []))
    t0 = time.perf_counter()
    if skip_verify:
        req["verified_candidates"] = [
            # llm_verified=False: no real verdict was produced, so the disagreement
            # router below must not treat this as an LLM judgement to agree/disagree with.
            {**c, "verdict": "SKIP", "reason": "skipped", "confidence": 0.5, "llm_verified": False}
            for c in req.get("ranked_candidates", [])
        ]
        req["_st"]["per_verifier_call_s"]    = {}
        req["_st"]["actual_verifier_tokens"] = {"input": 0, "output": 0}
    else:
        verifier    = _load_fresh("search/llm_verifier")
        _inject_os(verifier)
        call_times: list[float] = []
        call_tokens: list[dict] = []
        # module is now cached per-thread (see _load_fresh) — remember the true
        # original once so repeated monkey-patching doesn't nest wrappers.
        # Wrap _invoke (the real Bedrock call), not _verify — _process routes
        # through _verify_batch/_verify_chunk/_verify_one, all of which call
        # _invoke directly; _verify itself is dead code on this path and was
        # never actually hit, which silently zeroed out every token/timing stat.
        if not hasattr(verifier, "_invoke_orig"):
            verifier._invoke_orig = verifier._invoke
        orig = verifier._invoke_orig

        def _timed(*a, **kw):
            _t = time.perf_counter()
            _r = orig(*a, **kw)
            call_times.append(round(time.perf_counter() - _t, 3))
            _, in_tok, out_tok = _r
            call_tokens.append({"input": in_tok, "output": out_tok})
            return _r

        verifier._invoke = _timed
        req = verifier._process(req)
        for c in req["verified_candidates"]:
            c["llm_verified"] = True
        req["_st"]["per_verifier_call_s"]    = {i + 1: t for i, t in enumerate(call_times)}
        req["_st"]["actual_verifier_tokens"] = {
            "input":  sum(t["input"]  for t in call_tokens),
            "output": sum(t["output"] for t in call_tokens),
        }
    req["_st"]["llm_verifier"] = round(time.perf_counter() - t0, 3)
    # S5-gated candidates (unresolved_no_semantic_object / candidate_confidence_gate) never
    # reach the verifier above, but without this they vanish from the debug json entirely —
    # rejoin them here so _clean_result's rejected_hits picks them up.
    req["verified_candidates"].extend(req.pop("skipped_hits", []))
    _downgrade_statistical_conflicts(req.get("verified_candidates", []))
    _apply_disagreement_router(req.get("verified_candidates", []))
    return req


# HIGH/MEDIUM-severity structural conflicts (e.g. statistical.py's percentage/
# p_value/sample_size comparator) are objective, code-computed ground truth —
# the LLM free-reasons from text and can render YES even when it has itself
# narrated the mismatch (confirmed in production: CI220, 95%-vs-90% power,
# verdict=YES/confidence=0.92 with verifier_reason explicitly stating "power
# differs"). Downgrade rather than hard-veto to NO: a wrong comparator slot
# match (extraction mis-mapping two different facts) should surface the hit
# for review, not silently delete a true positive.
_STRUCTURAL_VETO_SEVERITIES = ("HIGH", "MEDIUM")


def _downgrade_statistical_conflicts(candidates: list) -> None:
    for c in candidates:
        if c.get("verdict") != "YES":
            continue
        contra_detail = (c.get("score_breakdown") or {}).get("contra_detail")
        if not isinstance(contra_detail, dict):
            continue   # [] when no conflicts were found — nothing to check
        conflict = contra_detail.get("statistical")
        if not conflict or conflict.get("severity") not in _STRUCTURAL_VETO_SEVERITIES:
            continue
        c["verdict"] = "MAYBE"
        c["structural_conflict"] = conflict
        # Kept separate from `reason` — that field is the LLM's verbatim verdict
        # explanation shown in the UI and must not be mutated/appended to.
        c["verdict_override_reason"] = (
            f"downgraded: structured comparator found statistical conflict "
            f"{conflict.get('evidence', {}).get('conflict')}"
        )
    return


def _apply_disagreement_router(candidates: list) -> None:
    """Route pre-LLM-vs-LLM disagreements to MAYBE instead of trusting either signal alone.

    Only candidates with an explicit llm_verified=True flag are considered — S5-gated
    candidates never got a real LLM verdict, so a low agg_score there isn't a
    "disagreement", it's the deterministic pre-LLM gate working as designed. Checking
    the explicit flag (set right after verifier._process/skip_verify, see _s6_llm_verify)
    rather than inferring it from the `reason` string keeps this safe if a later stage
    rewrites `reason` for an unrelated purpose.
    """
    for c in candidates:
        if not c.get("llm_verified"):
            continue
        agg_score = c.get("agg_score")
        if agg_score is None:
            continue
        verdict = c.get("verdict")
        # Kept separate from `reason` — that field is the LLM's verbatim verdict
        # explanation shown in the UI and must not be mutated/appended to.
        if verdict == "NO" and agg_score >= _DISAGREEMENT_HIGH_RETRIEVAL_THRESHOLD:
            c["verdict"] = "MAYBE"
            c["disagreement_route"] = "strong_retrieval_llm_no"
            c["verdict_override_reason"] = (
                f"resurfaced: strong pre-LLM retrieval (agg_score={agg_score}) but LLM said NO"
            )
        elif verdict == "YES" and agg_score <= _DISAGREEMENT_LOW_RETRIEVAL_THRESHOLD:
            c["verdict"] = "MAYBE"
            c["disagreement_route"] = "weak_retrieval_llm_yes"
            c["verdict_override_reason"] = (
                f"downgraded: LLM said YES but pre-LLM retrieval is weak (agg_score={agg_score})"
            )
    return


def _s7_highlight_extract(req: dict) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        req["_st"]["highlight_extractor"] = 0.0
        return req
    t0  = time.perf_counter()
    mod = _load("search/highlight_extractor", "search_highlight_extractor")
    req = mod._process(req)
    req["_st"]["highlight_extractor"] = round(time.perf_counter() - t0, 3)
    return req


def _s8_merge(req: dict) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        req["_st"]["merger"] = 0.0
        req.setdefault("final_hits", [])
        return req
    t0  = time.perf_counter()
    mod = _load("search/merger", "search_merger")
    req = mod._process(req)
    req["_st"]["merger"] = round(time.perf_counter() - t0, 3)
    return req


def _s9_evidence_classify(req: dict, skip_verify: bool = False) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        req["_st"].update({
            "evidence_classification": 0.0,
            "per_ec_call_s":           {},
            "n_ec_calls":              0,
            "actual_ec_tokens":        {"input": 0, "output": 0},
        })
        return req
    t0        = time.perf_counter()
    ec_times: list[float]         = []
    ec_tokens: dict[str, int]     = {"input": 0, "output": 0}
    hits      = req.get("final_hits", [])
    ci_text   = req["ci"].get("knownCI", "")
    doc_ctx   = req.get("document_context", {})

    if hits and not skip_verify:
        to_classify = [h for h in hits if h.get("verdict") in ("YES", "MAYBE")]
        if to_classify:
            _t_batch = time.perf_counter()
            classifier = _get_evidence_classifier()
            classified = classifier._classify_evidence_batch(ci_text, to_classify, doc_ctx)
            ec_times.append(round(time.perf_counter() - _t_batch, 3))
            for hit, ec in zip(to_classify, classified):
                tok = ec.pop("_ec_tokens", {"input": 0, "output": 0})
                ec_tokens["input"]  += tok["input"]
                ec_tokens["output"] += tok["output"]
                ec_clean = {k: ec[k] for k in ("evidence_type", "evidence_confidence", "evidence_reason") if k in ec}
                ec_clean = _calibrate_evidence(hit, ec_clean)
                hit.update(ec_clean)
                if _is_related(ec_clean["evidence_type"]):
                    hit["verdict"] = "MAYBE"
                elif ec_clean["evidence_type"] == "UNRELATED":
                    # Deliberately allowed to override a disagreement-router MAYBE
                    # (see _apply_disagreement_router): this is a third, independent
                    # evidence signal, not a re-litigation of the same LLM verdict —
                    # a hard NO should require weak retrieval + LLM NO + no supporting
                    # downstream evidence, and UNRELATED here is exactly that absence.
                    hit["verdict"] = "NO"
        hits.sort(key=lambda h: (
            _EVIDENCE_RANK.get(h.get("evidence_type", "BACKGROUND"), 4),
            -h.get("evidence_confidence", 0.0),
            -h.get("cross_encoder_score", 0.0),
            -h.get("highlight_score", 0.0),
        ))
        req["final_hits"] = hits

    req["_st"].update({
        "evidence_classification": round(time.perf_counter() - t0, 3),
        "per_ec_call_s":           {i + 1: t for i, t in enumerate(ec_times)},
        "n_ec_calls":              len(ec_times),
        "actual_ec_tokens":        ec_tokens,
    })
    return req


# ── Pipeline ───────────────────────────────────────────────────────────────────

def _safe_stage_wrapper(stage_key: str, stage_fn, req: dict) -> dict:
    """Wrap stage functions to catch exceptions and mark CI as failed.
    
    Sets CI context for all logs during this CI's stage processing.
    """
    if req.get("_failed") or req.get("_early_exit"):
        return req
    
    ci_id = req["ci"].get("id")
    token = _ctx_ci_id.set(ci_id)
    
    stage_t0 = time.perf_counter()
    logger.info(
        "[SearchFlow] STAGE_START stage=%s search_id=%s ci_idx=%s ci_id=%s "
        "ci_type=%s strategies=%s",
        stage_key, req.get("search_id"), req.get("_ci_idx"),
        ci_id, (req.get("classification") or {}).get("ci_type"),
        (req.get("classification") or {}).get("strategies", []),
    )
    try:
        result = stage_fn(req)
        logger.info(
            "[SearchFlow] STAGE_DONE stage=%s search_id=%s ci_id=%s elapsed_s=%.3f "
            "failed=%s early_exit=%s candidates=%d expanded=%d final_hits=%d",
            stage_key, req.get("search_id"), ci_id,
            time.perf_counter()-stage_t0,
            result.get("_failed"), result.get("_early_exit"),
            len(result.get("candidates", []) or []),
            len(result.get("expanded_candidates", []) or []),
            len(result.get("final_hits", []) or []),
        )
        return result
    except Exception as exc:
        logger.error(
            "[SearchFlow] STAGE_ERROR stage=%s search_id=%s ci_id=%s elapsed_s=%.3f "
            "error_type=%s error=%s",
            stage_key, req.get("search_id"), ci_id,
            time.perf_counter()-stage_t0, type(exc).__name__, exc, exc_info=True,
        )
        req["_failed"] = True
        req["_failure"] = {
            "stage": stage_key,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return req
    finally:
        _ctx_ci_id.reset(token)  # Properly restore previous context


def _run_pipeline(all_reqs: list[dict], skip_rerank: bool, skip_verify: bool,
                  n_workers: int, tenant: dict) -> tuple[list[dict], dict[str, float]]:
    """Run stage-parallel pipeline, return (all_reqs, stage_wall)."""
    # Reranker uses max_workers=1 to serialise CrossEncoder.predict() calls.
    STAGES = [
        ("S1:classify",          lambda r: _s1_classify(r),                         n_workers),
        ("S2:retrieve",          lambda r: _s2_retrieve(r),                         n_workers),
        ("S3:aggregate",         lambda r: _s3_aggregate(r),                        n_workers),
        ("S3.5:ground_evidence", lambda r: _s3_5_ground_evidence(r),                n_workers),
        ("S4:context_expand",    lambda r: _s4_context_expand(r),                   n_workers),
        ("S5:rerank",            lambda r: _s5_rerank(r, skip_rerank),         1),
        ("S6:llm_verify",        lambda r: _s6_llm_verify(r, skip_verify),     n_workers),
        ("S7:highlight",         lambda r: _s7_highlight_extract(r),                n_workers),
        ("S8:merge",             lambda r: _s8_merge(r),                            n_workers),
        ("S9:evidence_classify", lambda r: _s9_evidence_classify(r, skip_verify), n_workers),
    ]

    stage_wall: dict[str, float] = {}
    for stage_key, stage_fn, stage_workers in STAGES:
        active = sum(1 for r in all_reqs if not r.get("_failed") and not r.get("_early_exit"))
        if active == 0:
            stage_wall[stage_key] = 0.0
            continue
        t_stage = time.perf_counter()
        # copy_context() propagates the [tenant=/document=/search=] ContextVars (set on the
        # main thread in _handle_invocation) into these pool worker threads — plain pool.map
        # leaves them on default "-" since new threads don't inherit the caller's context.
        with ThreadPoolExecutor(max_workers=stage_workers) as pool:
            futures = [
                pool.submit(copy_context().run, _safe_stage_wrapper, stage_key, stage_fn, r)
                for r in all_reqs
            ]
            all_reqs = [f.result() for f in futures]
        stage_wall[stage_key] = round(time.perf_counter() - t_stage, 3)
        logger.info(
            "[SearchFlow] STAGE_WALL stage=%s wall_s=%.3f active=%d "
            "failed=%d early_exit=%d",
            stage_key, stage_wall[stage_key], active,
            sum(1 for r in all_reqs if r.get("_failed")),
            sum(1 for r in all_reqs if r.get("_early_exit")),
        )
    
    logger.info("[SearchWorker] stage wall summary: %s", stage_wall)
    return all_reqs, stage_wall



# ─────────────────────────────────────────────────────────────────────────────
# Result serialization
# ─────────────────────────────────────────────────────────────────────────────

def _save_results_debug_s3(all_results: list[dict], event, wall_time: float = 0.0,
                           n_cis_total: int = 0, n_completed: int = 0, 
                           n_failed: int = 0, ci_failures: list[dict] = None) -> str:
    """Write a clean, human-readable results JSON — strips large vectors/context."""
    if ci_failures is None:
        ci_failures = []
    
    batch_idx = event.get('batch_idx')
    search_id = event.get('search_id')
    document_id = event.get('document_id')
    tenant = event.get("tenant", {})
    tenant_name = tenant.get("tenant_name", "-")

    debug_json = {
        "run": {
            "timestamp":   datetime.now().isoformat(),
            "document_id": document_id,
            "opensearch":  OPENSEARCH_ENDPOINT,
            "skip_rerank": event.get("skip_rerank", False),
            "skip_verify": event.get("skip_verify", False),
        },
        "concurrency": {
            "ci_workers":      SEARCH_CI_WORKERS,
            "retriever_workers": RETRIEVER_WORKERS,
            "os_maxsize":      OPENSEARCH_MAXSIZE,
            "theoretical_max_concurrent_retrievers": SEARCH_CI_WORKERS * RETRIEVER_WORKERS,
        },
        "batch": {
            "batch_idx":      int(batch_idx),
            "expected_cis":   n_cis_total,
            "completed_cis":  n_completed,
            "failed_cis":     n_failed,
            "ci_failures":    ci_failures if ci_failures else [],
        },
        "summary": {
            "cis_searched":     len(all_results),
            "total_final_hits": sum(len(r.get("final_hits", [])) for r in all_results),
            "direct_hits":    sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if h.get("evidence_type") == "DIRECT"
            ),
            "same_study_hits": sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if h.get("evidence_type") == "SAME_STUDY"
            ),
            "related_hits":   sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if (h.get("evidence_type") or "").startswith("SAME_")
                or h.get("evidence_type") == "BACKGROUND"
            ),
            "related_breakdown": {
                sub: sum(
                    1 for r in all_results for h in r.get("final_hits", [])
                    if h.get("evidence_type") == f"SAME_{sub}"
                )
                for sub in (
                    "PROTOCOL", "OBJECTIVE", "ENDPOINT",
                    "POPULATION", "MECHANISM"
                )
            },
            "background_hits": sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if h.get("evidence_type") == "BACKGROUND"
            ),
            "no_hits":        sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if h.get("verdict") in ("NO", "SKIP")
            ),
            "total_rejected": sum(
                1 for r in all_results
                for v in r.get("verified_candidates", [])
                if v.get("verdict") == "NO"
            ),
            "total_skipped": sum(
                1 for r in all_results
                for v in r.get("verified_candidates", [])
                if v.get("verdict") == "SKIP"
            ),
            "object_type_stats": _object_type_stats(all_results),
        },
        "results": [_clean_result(r, debug=True) for r in all_results],
    }

    # ── Timing + cost summary ─────────────────────────────────────────────────
    # Stage functions populate req["_st"], not req["timings"] — that key is never set.
    all_t = [r.get("_st", {}) for r in all_results]
    n     = len(all_results) or 1

    def _agg_t(key: str) -> dict:
        vals = [t.get(key, 0.0) for t in all_t]
        return {"total_s": round(sum(vals), 3), "avg_s": round(sum(vals) / n, 3)}

    retriever_keys: list[str] = []
    for _tr in all_t:
        for _k in _tr.get("retrievers", {}):
            if _k not in retriever_keys:
                retriever_keys.append(_k)

    debug_json["timing_summary"] = {
        "classifier":              _agg_t("classifier"),
        "retrievers_total":        _agg_t("retrievers_total"),
        "retrievers": {
            rk: {
                "total_s": round(sum(t.get("retrievers", {}).get(rk, 0) for t in all_t), 3),
                "avg_s":   round(sum(t.get("retrievers", {}).get(rk, 0) for t in all_t) / n, 3),
            }
            for rk in retriever_keys
        },
        "aggregator":              _agg_t("aggregator"),
        "evidence_grounder":       _agg_t("evidence_grounder"),
        "context_expander":        _agg_t("context_expander"),
        "reranker":                _agg_t("reranker"),
        "llm_verifier":            _agg_t("llm_verifier"),
        "highlight_extractor":     _agg_t("highlight_extractor"),
        "merger":                  _agg_t("merger"),
        "evidence_classification": _agg_t("evidence_classification"),
        # total_cpu_s = sum of all per-CI pipeline times (CIs run in parallel,
        # so this is >  wall-clock time when workers > 1)
        "total": {
            "wall_clock_s":  round(wall_time, 3),
        },
        "_note": "per-stage total_s = cumulative CPU time across all parallel workers",
    }

    n_to_verifier  = sum(t.get("n_candidates_to_verifier", 0) for t in all_t)
    n_actual_calls = sum(len(t.get("per_verifier_call_s", {})) for t in all_t)

    v_actual_in  = sum(t.get("actual_verifier_tokens", {}).get("input",  0) for t in all_t)
    v_actual_out = sum(t.get("actual_verifier_tokens", {}).get("output", 0) for t in all_t)
    if v_actual_in > 0:
        v_in, v_out, v_label = v_actual_in, v_actual_out, "actual"
    else:
        v_in, v_out, v_label = (n_actual_calls * _EST_INPUT_TOKENS_PER_CAND,
                                 n_actual_calls * _EST_OUTPUT_TOKENS_PER_CAND, "est.")
    v_cost = v_in * _HAIKU_INPUT_PRICE_PER_TOKEN + v_out * _HAIKU_OUTPUT_PRICE_PER_TOKEN

    n_ec_calls = sum(t.get("n_ec_calls", 0) for t in all_t)
    ec_actual_in  = sum(t.get("actual_ec_tokens", {}).get("input",  0) for t in all_t)
    ec_actual_out = sum(t.get("actual_ec_tokens", {}).get("output", 0) for t in all_t)
    if ec_actual_in > 0:
        ec_in, ec_out, ec_label = ec_actual_in, ec_actual_out, "actual"
    else:
        ec_in, ec_out, ec_label = (n_ec_calls * _EST_EC_INPUT_TOKENS_PER_HIT,
                                    n_ec_calls * _EST_EC_OUTPUT_TOKENS_PER_HIT, "est.")
    ec_cost = ec_in * _HAIKU_INPUT_PRICE_PER_TOKEN + ec_out * _HAIKU_OUTPUT_PRICE_PER_TOKEN

    debug_json["cost_estimate"] = {
        "model": VERIFIER_MODEL,
        "llm_verifier": {
            "candidates_passed_to_verifier": n_to_verifier,
            "actual_bedrock_calls":           n_actual_calls,
            "input_tokens":                  v_in,
            "output_tokens":                 v_out,
            "total_tokens":                  v_in + v_out,
            "token_source":                  v_label,
            "est_cost_usd":                  round(v_cost, 4),
        },
        "evidence_classification": {
            "bedrock_calls":   n_ec_calls,
            "input_tokens":    ec_in,
            "output_tokens":   ec_out,
            "total_tokens":    ec_in + ec_out,
            "token_source":    ec_label,
            "est_cost_usd":    round(ec_cost, 4),
        },
        "combined_est_cost_usd": round(v_cost + ec_cost, 4),
    }

    s3_url = _upload_debug_json_to_s3(debug_json, search_id, batch_idx, document_id, tenant_name)
    return s3_url



def _object_type_stats(all_results: list[dict]) -> dict:
    """Per-object-type breakdown: how many candidates were retrieved, passed, rejected, skipped."""
    from collections import defaultdict
    retrieved: dict[str, int] = defaultdict(int)
    final_yes: dict[str, int] = defaultdict(int)
    rejected:  dict[str, int] = defaultdict(int)
    skipped:   dict[str, int] = defaultdict(int)

    for r in all_results:
        for v in r.get("verified_candidates", []):
            obj_type = (v.get("matched_object") or {}).get("type") or "chunk"
            retrieved[obj_type] += 1
            verdict = v.get("verdict", "")
            if verdict in ("YES", "MAYBE"):
                final_yes[obj_type] += 1
            elif verdict == "NO":
                rejected[obj_type] += 1
            elif verdict == "SKIP":
                skipped[obj_type] += 1

    all_types = sorted(set(list(retrieved.keys())))
    return {
        t: {
            "retrieved": retrieved[t],
            "final_yes": final_yes[t],
            "rejected":  rejected[t],
            "skipped":   skipped[t],
        }
        for t in all_types
    }


def _base_object_id(object_id: str | None) -> str:
    """Normalize retriever object-id variants like *_s0/*_s1 to a stable base id."""
    import re as _re
    oid = str(object_id or "")
    return _re.sub(r"_s\d+$", "", oid)


def _hit_with_provenance(hit: dict, debug: bool = False) -> dict:
    """Add retrieval provenance fields to a final hit; strip the raw matched_object."""
    obj            = hit.get("matched_object") or {}
    obj_type       = obj.get("type") or "unknown"
    sources        = hit.get("sources", [])

    # context_expander sets retrieval_origin as "{direct|via_chunk}_{object_type}":
    #   direct_sentence   → retriever returned a sentence object from the semantic-objects index
    #   direct_paragraph  → retriever returned a paragraph object directly
    #   via_chunk_sentence → retriever returned a CHUNK; context_expander extracted a sentence
    # We use this to determine what the retriever ACTUALLY retrieved, not what was expanded to.
    ce_origin = hit.get("retrieval_origin", "")  # context_expander's value (will be overwritten)
    if ce_origin.startswith("via_chunk_"):
        # retriever found a chunk; context_expander chose best object within it
        retrieved_unit = "chunk"
    elif ce_origin.startswith("direct_"):
        # retriever found this object type directly in the index
        retrieved_unit = ce_origin[len("direct_"):]
    else:
        # no context_expander info → use retrieved_type (set by vector_retriever) or matched_object.type
        retrieved_unit = hit.get("retrieved_type") or obj_type

    # retrieval_origin: analytics format "sources/retrieved_unit"
    # e.g. "vector/chunk", "bm25/sentence", "vector/paragraph"
    origin_str = ("+".join(sorted(sources)) if sources else "unknown") + "/" + retrieved_unit
    object_id = obj.get("object_id")
    parent_chunk_id = obj.get("parent_chunk_id")
    retrieval_chunk_id = parent_chunk_id or hit.get("chunk_id")
    extra = {
        "retrieval_object_type": obj_type,
        # retrieved_type: the unit the retriever actually fetched from the index.
        # "chunk" = vector_search_chunks fallback; context_expander assigned the object.
        # "sentence"/"paragraph"/etc = object fetched directly from semantic-objects index.
        "retrieved_type":        retrieved_unit,
        # expansion_origin: context_expander's raw value preserved for debugging.
        # Format: "direct_{type}" or "via_chunk_{type}".
        "expansion_origin":      ce_origin or None,
        "retrieval_object_id":   object_id,
        "retrieval_object_id_base": _base_object_id(object_id),
        "retrieval_parent_chunk_id": parent_chunk_id,
        "retrieval_chunk_id": retrieval_chunk_id,
        "retrieval_section":     obj.get("section_category") or obj.get("section"),
        "retrieval_origin":      origin_str,
        "selection_reason":      hit.get("selection_reason"),
        "literal_match_count":   hit.get("literal_match_count"),
        "context_strategy":      hit.get("context_strategy"),
        "matched_distance":      hit.get("matched_distance"),
        "distance_ratio":        hit.get("distance_ratio"),
        "current_text_chars":    hit.get("current_text_chars"),
        "agg_score":             hit.get("agg_score"),
        "score_breakdown":       hit.get("score_breakdown"),
        "agg_score_breakdown":   hit.get("agg_score_breakdown"),
        "llm_verified":          hit.get("llm_verified"),
        "disagreement_route":    hit.get("disagreement_route"),
        "verdict_override_reason": hit.get("verdict_override_reason"),
        # Full detail in debug mode (S3 debug.json); trimmed for the orchestrator's
        # inline response since the UI only reads geometry/ids (see createHighlight.ts).
        "indexed_object":        _indexed_object(hit) if debug else _indexed_object_minimal(hit),
    }
    if debug:
        extra["retrieval_heading_path"] = obj.get("heading_path")
    # Remove embedding vectors and other unnecessary large fields
    vectors_to_exclude = {
        "matched_object",  # Already handled separately
        "dense_vector",
        "embedding",
        "sparse_vector",
        "vector",
        "dense_embedding",
        "context",  # Context expanded separately in indexed_object
    }
    if not debug:
        vectors_to_exclude |= {"retrieval_heading_path", "context_sentence"}
    base = {k: v for k, v in hit.items() if k not in vectors_to_exclude}
    return {**base, **extra}


def _indexed_object_minimal(v: dict) -> dict | None:
    """Geometry + identity only - all the UI reads from indexed_object (see createHighlight.ts)."""
    obj = v.get("matched_object")
    if not obj:
        return None
    return {
        "object_id":       obj.get("object_id"),
        "parent_chunk_id": obj.get("parent_chunk_id"),
        "geometry":        obj.get("geometry") or {},
        "type" :           obj.get("type"),
        "bbox":            obj.get("bbox"),
    }


def _indexed_object(v: dict) -> dict | None:
    """
    Return the full indexed data for a candidate — everything stored in OpenSearch
    for this semantic object, minus large vector fields.
    Included in every hit/rejected/skipped entry so reviewers can audit
    exactly what was indexed (entities, facts, clinical_relations, etc.).
    """
    obj = v.get("matched_object")
    if not obj:
        return None
    ctx_text = (v.get("context") or {}).get("current_text", "")
    return {
        # Identity
        "object_id":          obj.get("object_id"),
        "parent_chunk_id":    obj.get("parent_chunk_id"),
        "type":               obj.get("type"),
        # Location
        "page":               obj.get("page"),
        "bbox":               obj.get("bbox"),
        # Canonical geometry contract from extraction/indexing. This includes
        # page-local paragraph geometry, contributing native spans, and the
        # exact per-page sentence text used by PDF text-search mode.
        "geometry":           obj.get("geometry") or {},
        "list_id":            obj.get("list_id"),
        "list_level":         obj.get("list_level"),
        "list_label":         obj.get("list_label"),
        "list_number_format": obj.get("list_number_format"),
        "table_id":           obj.get("table_id", obj.get("table_key")),            "row_index":          obj.get("row_index", obj.get("row_start")),
        "row_start":          obj.get("row_start"),
        "col_start":          obj.get("col_start"),
        "row_span":           obj.get("row_span"),
        "col_span":           obj.get("col_span"),
        "position":           obj.get("position"),
        "global_position":    obj.get("global_position"),
        "document_position":  obj.get("document_position"),
        # Text content
        "text":               obj.get("text"),
        "paragraph_text":     obj.get("paragraph_text"),
        "prev_sentence_text": obj.get("prev_sentence_text"),
        "next_sentence_text": obj.get("next_sentence_text"),
        "context_chunk_text": ctx_text or None,
        # Section / heading
        "section_category":   obj.get("section_category"),
        "heading_path":       obj.get("heading_path"),
        "semantic_path":      obj.get("semantic_path"),
        "section_confidence": obj.get("section_confidence"),
        # NER
        "entities":           obj.get("entities", []),
        # Clinical fact extraction
        "facts":              obj.get("facts", {}),
        "statement_type":     obj.get("statement_type"),
        "study_context":      obj.get("study_context"),
        "clinical_relations": obj.get("clinical_relations", []),
    }


def _ci_metadata(ci: dict) -> dict:
    """
    Return all enriched CI fields suitable for the result JSON.
    Dense and sparse embedding vectors are intentionally excluded (large + irrelevant to reviewers).
    """
    emb  = ci.get("embedding", {})
    norm = ci.get("normalization", {})
    ner  = ci.get("ner", {})
    onto = ci.get("ontology", {})
    cls  = ci.get("classification", {})

    return {
        # ── Identity ─────────────────────────────────────────────────────────
        "id":                   ci.get("id"),
        "text":                 ci.get("knownCI", ""),
        "category":             ci.get("category") or ci.get("type"),
        # ── Normalization ─────────────────────────────────────────────────────
        "normalized_text":      norm.get("normalized_text", ""),
        "tokens":               norm.get("tokens", []),
        "abbreviations":        norm.get("abbreviations_found", {}),
        # ── NER ───────────────────────────────────────────────────────────────
        "entities":             ner.get("entities", []),
        "ner_model":            ner.get("model"),
        # ── Ontology ──────────────────────────────────────────────────────────
        "ontology_expansions":  onto.get("expansions", []),
        "ontology_synonyms":    onto.get("synonyms", {}),
        "regex_patterns":       onto.get("regex_patterns", []),
        # ── Embedding metadata (no vectors) ───────────────────────────────────
        "embedding_model":      emb.get("model"),
        "embedding_dimensions": emb.get("dimensions"),
        # ── Classification ────────────────────────────────────────────────────
        "ci_type":              cls.get("ci_type"),
        "strategies":           cls.get("strategies", []),
        "classification_reason": cls.get("reason"),
        # ── Clinical facts ────────────────────────────────────────────────────
        "facts":                ci.get("facts", {}),
        "own_facts":            ci.get("own_facts", {}),
        "effective_facts":      ci.get("effective_facts", {}),
        "inherited_slots":      ci.get("inherited_slots", []),
        "slot_provenance":      ci.get("slot_provenance", {}),
        "study_hierarchy":      ci.get("study_hierarchy", {}),
        "clinical_identity":    ci.get("clinical_identity", {}),
        "study_context":        ci.get("study_context"),
        "statement_type":       ci.get("statement_type"),
        "modality":             ci.get("modality"),
        "negated_slots":        ci.get("negated_slots", []),
        "treatment_identity":   ci.get("treatment_identity", {}),
        "endpoint_identity":    ci.get("endpoint_identity", {}),
        "population_identity":  ci.get("population_identity", {}),
        "temporal_context":     ci.get("temporal_context", {}),
        "clinical_relations":   ci.get("clinical_relations", []),
        # ── Source metadata (passthrough from raw CI file) ────────────────────
        "justification_text":   ci.get("justificationText"),
        "assets":               ci.get("assets", []),
    }


def _full_candidate_record(v: dict) -> dict:
    """
    Comprehensive per-candidate record saved for every verified_candidate
    (verdict = YES / MAYBE / NO / SKIP).

    Captures all pipeline stages: aggregator scores, reranker score,
    verifier verdict + reason, highlight extraction, and the full indexed
    object (prev_sentence_text, text, next_sentence_text, paragraph_text,
    context_chunk_text, heading_path, section, NER, facts, relations).
    """
    sb  = v.get("score_breakdown") or {}
    obj = v.get("matched_object") or {}
    ctx_text = (v.get("context") or {}).get("current_text", "")
    object_id = obj.get("object_id")
    parent_chunk_id = obj.get("parent_chunk_id")
    retrieval_chunk_id = parent_chunk_id or v.get("chunk_id")
    return {
        # ── Identity ─────────────────────────────────────────────────────
        "chunk_id":             v.get("chunk_id", ""),
        "page_start":           v.get("page_start"),
        "page_end":             v.get("page_end"),
        "sources":              v.get("sources", []),
        "retriever":            v.get("retriever", ""),
        "retrieval_origin":     v.get("retrieval_origin", "direct_unknown"),
        "selection_reason":     v.get("selection_reason"),
        "literal_match_count":  v.get("literal_match_count"),
        "context_strategy":     v.get("context_strategy"),
        "matched_distance":     v.get("matched_distance"),
        "distance_ratio":       v.get("distance_ratio"),
        "current_text_chars":   v.get("current_text_chars"),
        # ── Verifier outcome ─────────────────────────────────────────────
        "verdict":              v.get("verdict"),
        "confidence":           v.get("confidence"),
        "verifier_reason":      v.get("reason", ""),
        "verifier_identity":    v.get("identity", {}),
        "verifier_supporting_sentences":  v.get("supporting_sentences", []),
        "verifier_highlight_type":        v.get("highlight_type", "sentence"),
        "verifier_primary_support_index": v.get("primary_support_index", 0),
        "verifier_tokens":      v.get("_tokens"),   # actual input/output token counts from Bedrock
        # ── Disagreement router (pre-LLM agg_score vs LLM verdict) ────────
        "llm_verified":           v.get("llm_verified"),
        "disagreement_route":     v.get("disagreement_route"),
        "structural_conflict":    v.get("structural_conflict"),
        "verdict_override_reason": v.get("verdict_override_reason"),
        # ── Reranker scores ──────────────────────────────────────────────
        "cross_encoder_score":  v.get("cross_encoder_score") or sb.get("ce"),
        "agg_score":            v.get("agg_score"),
        "score_breakdown":      sb,
        "agg_score_breakdown":  v.get("agg_score_breakdown"),
        # ── Highlight / match (populated for YES/MAYBE) ──────────────────
        "match_span":           v.get("match_span", ""),
        "context_sentence":     v.get("context_sentence", ""),
        "highlight_score":      v.get("highlight_score"),
        "match_method":         v.get("match_method", ""),
        "highlight_mode":       v.get("highlight_mode", "span"),
        "text_search_pages":    v.get("text_search_pages", []),
        "match_page":           v.get("match_page"),
        # Canonical geometry — copied from the indexed semantic object.
        "geometry":             obj.get("geometry") or {},
        # ── Retrieval provenance ─────────────────────────────────────────
        "retrieval_object_type":  obj.get("type"),
        "retrieval_object_id":    object_id,
        "retrieval_object_id_base": _base_object_id(object_id),
        "retrieval_parent_chunk_id": parent_chunk_id,
        "retrieval_chunk_id":     retrieval_chunk_id,
        "retrieval_heading_path": obj.get("heading_path"),
        "retrieval_section":      obj.get("section_category") or obj.get("section"),
        # ── Full indexed object ──────────────────────────────────────────
        "indexed_object": {
            # Identity
            "object_id":          obj.get("object_id"),
            "parent_chunk_id":    obj.get("parent_chunk_id"),
            "type":               obj.get("type"),
            "list_id":            obj.get("list_id"),
            "list_level":         obj.get("list_level"),
            "list_label":         obj.get("list_label"),
            "list_number_format": obj.get("list_number_format"),
            "table_id":           obj.get("table_id", obj.get("table_key")),
            "row_index":          obj.get("row_index", obj.get("row_start")),
            "row_start":          obj.get("row_start"),
            "col_start":          obj.get("col_start"),
            "row_span":           obj.get("row_span"),
            "col_span":           obj.get("col_span"),
            # Location
            "page":               obj.get("page"),
            "bbox":               obj.get("bbox"),
            "geometry":            obj.get("geometry") or {},
            "position":           obj.get("position"),
            "global_position":    obj.get("global_position"),
            "document_position":  obj.get("document_position"),
            # ── Text context (all layers) ─────────────────────────────────
            "prev_sentence_text": obj.get("prev_sentence_text", ""),
            "text":               obj.get("text", ""),
            "next_sentence_text": obj.get("next_sentence_text", ""),
            "paragraph_text":     obj.get("paragraph_text", ""),
            "context_chunk_text": ctx_text or obj.get("context_chunk_text", ""),
            # ── Document structure ────────────────────────────────────────
            "section_category":   obj.get("section_category"),
            "heading_path":       obj.get("heading_path"),
            "semantic_path":      obj.get("semantic_path"),
            "section_confidence": obj.get("section_confidence"),
            # ── NER ───────────────────────────────────────────────────────
            "entities":           obj.get("entities", []),
            # ── Clinical facts ────────────────────────────────────────────
            "facts":              obj.get("facts", {}),
            "effective_facts":    obj.get("effective_facts"),
            "statement_type":     obj.get("statement_type"),
            "study_context":      obj.get("study_context"),
            "clinical_relations": obj.get("clinical_relations", []),
        },
    }


def _clean_result(result: dict,debug: bool = False) -> dict:
    """Strip embedding vectors, raw context blobs — keep only what matters."""
    ci = result.get("ci", {})

    # Candidates the verifier rejected (verdict=NO). Included here so evaluators
    # can review what was retrieved-but-rejected and label false negatives.
    def _provenance(v: dict) -> dict:
        obj = (v.get("matched_object") or {})
        object_id = obj.get("object_id")
        parent_chunk_id = obj.get("parent_chunk_id")
        retrieval_chunk_id = parent_chunk_id or v.get("chunk_id")
        return {
            "retrieval_object_type": obj.get("type"),
            "retrieval_object_id":   object_id,
            "retrieval_object_id_base": _base_object_id(object_id),
            "retrieval_parent_chunk_id": parent_chunk_id,
            "retrieval_chunk_id":   retrieval_chunk_id,
            "retrieval_heading_path": obj.get("heading_path"),
            "retrieval_section":     obj.get("section_category") or obj.get("section"),
            "retrieval_origin":      v.get("retrieval_origin", "direct_unknown"),
            "selection_reason":      v.get("selection_reason"),
            "literal_match_count":   v.get("literal_match_count"),
            "context_strategy":      v.get("context_strategy"),
            "matched_distance":      v.get("matched_distance"),
            "distance_ratio":        v.get("distance_ratio"),
            "current_text_chars":    v.get("current_text_chars"),
        }

    rejected_hits = [
        {
            "chunk_id":      v.get("chunk_id", ""),
            "page_start":    v.get("page_start"),
            "page_end":      v.get("page_end"),
            "text":          (v.get("context", {}).get("current_text", "") or v.get("text", ""))[:500],
            "verdict":       v.get("verdict"),
            "confidence":    v.get("confidence"),
            "reason":        v.get("reason", ""),
            "retriever":     v.get("retriever", ""),
            "sources":       v.get("sources", []),
            "agg_score":     v.get("agg_score"),
            "score_breakdown": v.get("score_breakdown"),
            "agg_score_breakdown": v.get("agg_score_breakdown"),
            "llm_verified":  v.get("llm_verified"),
            "disagreement_route": v.get("disagreement_route"),
            "verdict_override_reason": v.get("verdict_override_reason"),
            "indexed_object": _indexed_object(v),
            **_provenance(v),
        }
        for v in result.get("verified_candidates", [])
        if v.get("verdict") == "NO"
    ]

    # Candidates that never reached Claude (verdict=SKIP: cross_encoder_score below threshold).
    # Included so evaluators can see what the reranker filtered out before LLM verification.
    skipped_hits = [
        {
            "chunk_id":            v.get("chunk_id", ""),
            "page_start":          v.get("page_start"),
            "page_end":            v.get("page_end"),
            "text":                (v.get("context", {}).get("current_text", "") or v.get("text", ""))[:500],
            "verdict":             v.get("verdict"),
            "cross_encoder_score": v.get("cross_encoder_score"),
            "agg_score":           v.get("agg_score"),
            "reason":              v.get("reason", ""),
            "sources":             v.get("sources", []),
            "score_breakdown":     v.get("score_breakdown"),
            "agg_score_breakdown": v.get("agg_score_breakdown"),
            "llm_verified":        v.get("llm_verified"),
            "disagreement_route":  v.get("disagreement_route"),
            "verdict_override_reason": v.get("verdict_override_reason"),
            "indexed_object":      _indexed_object(v),
            **_provenance(v),
        }
        for v in result.get("verified_candidates", [])
        if v.get("verdict") == "SKIP"
    ]


    return {
        "search_id":        result.get("search_id"),
        "ci_id":            ci.get("id"),
        "ci_text":          ci.get("knownCI", ""),
        "ci_type":          result.get("classification", {}).get("ci_type"),
        "strategies":       result.get("classification", {}).get("strategies", []),
        # Full enriched CI metadata (everything except dense/sparse vectors)
        "ci":               _ci_metadata(ci),
        "object_type_stats": _object_type_stats([result]),
        "candidates_found": len(result.get("candidates", [])),
        # ── Per-candidate detail (all verdicts) ───────────────────────────────
        # One entry per verified_candidate — gives full candidate-level
        # granularity for the CSV exporter and for manual review.
        # Replaces the separate rejected_hits / skipped_hits split for
        # downstream tools; both are kept below for backward compatibility.
        "final_hits":       [_hit_with_provenance(h,debug) for h in result.get("final_hits", [])],
        **({"candidates": [_full_candidate_record(v) for v in result.get("verified_candidates", [])],
            "rejected_hits":    rejected_hits,
            "skipped_hits":     skipped_hits,
            "ce_histogram":     result.get("ce_histogram"),
        } if debug else {}),

        "timings":          result.get("_st", {}),
        "highlight_mode":   result.get("highlight_mode", "span"),
    }


def _strip_vectors(obj: dict | list | str | int | float | bool | None) -> dict | list | str | int | float | bool | None:
    """Recursively remove all vector/embedding fields from an object.
    
    COMPREHENSIVE: Catches dense_vector, sparse_vector, embeddings in indexed_object,
    entities, facts, and anywhere else they might hide to reduce payload size.
    
    Vector fields are large (1000+ dimensions) and irrelevant for the Orchestrator.
    Removes:
    - dense_vector, sparse_vector
    - embedding, vector, _embedding, _vector  
    - dense_embedding, sparse_embedding
    - dense, sparse (when they're vectors)
    - Any *_vector or *_embedding fields
    """
    # Aggressive list of vector field patterns
    VECTOR_KEYWORDS = {
        "dense_vector", "sparse_vector", "dense", "sparse",
        "embedding", "vector", "_embedding", "_vector",
        "dense_embedding", "sparse_embedding",
        "_dense", "_sparse", "embeddings", "vectors",
        "dense_embeddings", "sparse_embeddings"
    }
    
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # per_verifier_call_s / per_ec_call_s use int keys (call index) — not strippable by name.
            if not isinstance(k, str):
                result[k] = _strip_vectors(v)
                continue
            k_lower = k.lower()
            
            # Skip if key matches vector patterns (case-insensitive)
            if (k_lower in VECTOR_KEYWORDS or 
                k.endswith("_vector") or k.endswith("_embedding") or
                k_lower.endswith("_vec") or k_lower.endswith("_emb") or
                # Catch vector-like fields even if slightly different
                k_lower.startswith("dense") or k_lower.startswith("sparse")):
                continue
            
            # Recursively strip from values
            result[k] = _strip_vectors(v)
        return result
    elif isinstance(obj, list):
        return [_strip_vectors(item) for item in obj]
    else:
        # Primitives are returned as-is
        return obj


def _build_result(req: dict) -> dict:
    """Return full detailed results with ALL fields (entities, facts, candidates, indexed_object).
    
    Vectors are stripped aggressively to keep payload under 6MB Lambda limit while
    preserving all detailed clinical data needed for review and analysis.
    """
    # Return the complete cleaned result with all details
    result = _clean_result(req)
    
    # Aggressively strip vectors from entire result to reduce payload size
    result = _strip_vectors(result)
    
    return result


def _upload_debug_json_to_s3(debug_json: dict, search_id: str, batch_idx: int, document_id: str, tenant_name: str) -> str:
    """Upload detailed debug JSON to S3 and return the S3 URL."""
    try:
        import boto3
        s3 = boto3.client("s3", region_name=AWS_REGION)
        bucket = SEARCH_RESULTS_DEBUG_BUCKET
        
        from datetime import datetime as _dt
        s3_key = f"{RESULTS_DEBUG_PREFIX}/{tenant_name}/{search_id}/{document_id}/batch/{batch_idx}/debug.json"
        
        s3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(debug_json, indent=2, ensure_ascii=False),
            ContentType="application/json",
        )
        s3_url = f"s3://{bucket}/{s3_key}"
        logger.info("[S3] debug results uploaded to %s", s3_url)
        return s3_url
    except Exception as exc:
        logger.warning("[S3] failed to upload debug JSON: %s", exc)
        return ""


def _upload_results_to_s3(results: list, search_id: str, batch_idx: int, document_id: str, tenant_name: str) -> str:
    """Upload the full per-CI results list to S3 and return the S3 URL.

    Used as an overflow path when the inline Lambda response would exceed the
    6MB sync-invoke limit - the orchestrator downloads and merges this instead.
    """
    import boto3
    s3 = boto3.client("s3", region_name=AWS_REGION)
    bucket = SEARCH_RESULTS_DEBUG_BUCKET
    s3_key = f"{RESULTS_DEBUG_PREFIX}/{tenant_name}/{search_id}/{document_id}/batch/{batch_idx}/results.json"
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(results, default=str, ensure_ascii=False).encode(),
        ContentType="application/json",
    )
    s3_url = f"s3://{bucket}/{s3_key}"
    logger.info("[S3] overflow results uploaded to %s", s3_url)
    return s3_url


# ── Lambda handler ─────────────────────────────────────────────────────────────

# Delay before the self-destruct thread kills the process — must be long enough
# for the Lambda runtime to finish flushing our `return` value back over the
# Runtime API before the process dies (killing it synchronously pre-return
# would drop the response, not just the container).


def _current_rss_mb() -> float:
    """Current (not peak) resident memory, read from /proc so it reflects
    what's actually held right now rather than the invocation's high-water mark."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def _cleanup_memory() -> None:
    """Release everything that could carry over into the next invocation on a
    reused warm container: per-module caches, the per-thread fresh-module
    cache, and any heap pages freed by gc that glibc would otherwise keep."""
    global _evidence_classifier
    _loaded.clear()
    _loaded_fresh.clear()
    _evidence_classifier = None
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _handle_invocation(event: dict, context: Any) -> dict:
    search_id   = event.get("search_id", str(uuid.uuid4()))
    batch_idx   = event.get("batch_idx", 0)
    enriched_cis = event.get("cis", [])
    document_id  = event.get("document_id", "")
    tenant  = event.get("tenant", {})
    project_id   = event.get("project_id", "")
    doc_context  = event.get("document_context", {})
    skip_rerank  = bool(event.get("skip_rerank", False))
    skip_verify  = bool(event.get("skip_verify",  False))
    
    n_workers    = SEARCH_CI_WORKERS
    tenant_name   = tenant.get("tenant_name", "default")

    # ── Set context variables for all logs (automatically injected by SearchContextFilter) ────────────────────────────────
    _ctx_tenant.set(tenant_name)
    _ctx_document_id.set(document_id)
    _ctx_search_id.set(search_id)
    _ctx_batch_idx.set(str(batch_idx))
    
    # Test log to verify this version is deployed
    logger.info("🔥 SEARCH WORKER VERSION 2026-08-20 — context vars active")
    logger.info(
        "[SearchFlow] INVOCATION_START search_id=%s batch_idx=%s document_id=%s "
        "cis=%d ci_workers=%s retriever_workers=%s theoretical_max_concurrent_retrievers=%s",
        search_id, batch_idx, document_id, len(enriched_cis),
        SEARCH_CI_WORKERS, RETRIEVER_WORKERS,
        SEARCH_CI_WORKERS * RETRIEVER_WORKERS,
    )
    for i, ci in enumerate(enriched_cis):
        logger.info(
            "[SearchFlow] CI_INPUT search_id=%s ci_idx=%d ci_id=%s text=%r category=%s "
            "strategies_preclass=%s",
            search_id, i, ci.get("id"), ci.get("knownCI", "")[:300],
            ci.get("category") or ci.get("type"),
            (ci.get("classification") or {}).get("strategies", [])
            if isinstance(ci.get("classification"), dict) else [],
        )

    all_reqs = [
        {
            "search_id":        f"{search_id}-{i}",
            "document_id":      document_id,
            "ci":               ci,
            "document_context": doc_context,
            "_st":              {},
            "_failed":          False,
            "_early_exit":      False,
            "_ci_idx":          i,
            "tenant":            tenant,
            "project_id":        project_id,
        }
        for i, ci in enumerate(enriched_cis)
    ]
    # enriched_cis's ci dicts (with embeddings) now live inside all_reqs too —
    # drop this extra reference so it doesn't outlive all_reqs's own cleanup.
    n_cis_total = len(enriched_cis)
    del enriched_cis

    t_total = time.perf_counter()
    all_reqs, stage_wall = _run_pipeline(all_reqs, skip_rerank, skip_verify, n_workers, tenant)
    wall_time = round(time.perf_counter() - t_total, 3)

    # ── Track CI-level success/failure ─────────────────────────────────────────
    failed_cis = [r for r in all_reqs if r.get("_failed")]
    completed_cis = [r for r in all_reqs if not r.get("_failed")]
    
    # Extract detailed failure info for each failed CI
    ci_failures = []
    for req in failed_cis:
        ci_failures.append({
            "ci_id": req["ci"].get("id"),
            "ci_text": req["ci"].get("knownCI", ""),
            "error_type": req.get("_failure", {}).get("error_type", "unknown"),
            "error": req.get("_failure", {}).get("error", ""),
            "stage": req.get("_failure", {}).get("stage", ""),
        })
    
    # NOW save debug JSON with complete failure information
    s3_url = _save_results_debug_s3(all_reqs, event, wall_time, n_cis_total,
                                     len(completed_cis), len(failed_cis), ci_failures)
    
    # ── Simple return to orchestrator ──────────────────────────────────────────
    results = [_build_result(r) for r in completed_cis]  # Only return completed CIs
    total_hits = sum(len(r.get("final_hits", [])) for r in results)

    # all_reqs/completed_cis/failed_cis carry every stage's full intermediate
    # data (candidates, expanded_candidates, matched_object, context) for every
    # CI — nothing downstream needs them once `results` is built, so capture
    # their counts and drop the references now rather than waiting for
    # end-of-invocation cleanup.
    n_completed, n_failed = len(completed_cis), len(failed_cis)
    del all_reqs, completed_cis, failed_cis
    gc.collect()

    # Build response
    response = {
        "document_id":   document_id,
        "search_id":     search_id,
        "batch_idx":     batch_idx,
        "n_cis":         n_cis_total,
        "completed_cis": n_completed,
        "failed_cis":    n_failed,
        "ci_failures":   ci_failures,  # NEW: detailed failure info
        "results":       results,
        "stage_wall":    stage_wall,
        "wall_time":     wall_time,
        "debug_s3_url":  s3_url,
    }
    
    # CRITICAL: Strip ALL vectors from entire response before returning
    # This is the final safety net to prevent 6MB Lambda response limit
    import json
    pre_strip_size = len(json.dumps(response, default=str).encode())
    
    response = _strip_vectors(response)
    # response now owns its own (stripped) copy of what `results` held —
    # drop the pre-strip reference instead of letting it ride out the function.
    del results
    
    post_strip_size = len(json.dumps(response, default=str).encode())
    reduction_pct = round(100 * (1 - post_strip_size / max(pre_strip_size, 1)), 1)
    
    logger.info(
        "[SearchWorker] done wall=%.1fs "
        "cis_total=%d completed=%d failed=%d hits=%d "
        "payload_before_strip=%d bytes payload_after_strip=%d bytes reduction=%.1f%%",
        wall_time,
        n_cis_total, n_completed, n_failed, total_hits,
        pre_strip_size, post_strip_size, reduction_pct
    )

    # Stripping alone doesn't bound total size (many CIs x many final_hits with
    # full indexed_object text/entities/facts can still exceed the 6MB sync-invoke
    # cap) - only offload to S3 when actually needed, so the common case stays inline.
    if post_strip_size > WORKER_RESPONSE_INLINE_LIMIT_BYTES:
        results_s3_url = _upload_results_to_s3(
            response["results"], search_id, batch_idx, document_id, tenant_name
        )
        logger.warning(
            "[SearchWorker] response %d bytes exceeds inline limit %d bytes - "
            "offloaded results to %s",
            post_strip_size, WORKER_RESPONSE_INLINE_LIMIT_BYTES, results_s3_url,
        )
        response["results"] = []
        response["results_offloaded"] = True
        response["results_s3_url"] = results_s3_url

    return response


def handler(event: dict, context: Any) -> dict:
    """Thin wrapper: run the invocation, then explicitly release module-level
    caches so a reused warm container starts the next invocation clean. No
    forced os._exit() - we rely on this cleanup plus RSS telemetry instead."""
    logger.info(
        "[Runtime] env_id=%s pid=%s cold_start=%s",
        EXECUTION_ENV_ID, os.getpid(), COLD_START_TS,
    )
    logger.info("[Memory] START pid=%s rss_mb=%.1f", os.getpid(), _current_rss_mb())
    try:
        return _handle_invocation(event, context)
    finally:
        logger.info("[Memory] BEFORE_CLEANUP rss_mb=%.1f", _current_rss_mb())
        _cleanup_memory()
        logger.info("[Memory] AFTER_CLEANUP rss_mb=%.1f", _current_rss_mb())
