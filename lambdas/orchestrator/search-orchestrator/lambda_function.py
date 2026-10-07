"""
Search Orchestrator Lambda
===========================
Splits a list of CIs into fixed-size batches, fans out to Search Worker
Lambdas (direct synchronous invoke via ThreadPoolExecutor), collects all
results and returns the merged response.

Input
-----
{
    "search_id":        str (optional, generated if absent),
    "cimAnnotationJobId": int | str,
    "cis":              list[dict],   # optional legacy/manual input; ignored when cimAnnotationJobId is present
    "document_id":      str,
    "tenant":           dict,
    "project_id":        str,
    "document_context": dict (optional, derived from document_assets.json if absent),
    "batch_size":       int  (default: 50),
    "skip_rerank":      bool (default: false),
    "skip_verify":      bool (default: false),
    "bucketName":        str (optional, S3 bucket for full result JSON of backend service),
    "s3ResultPath":     str (optional, if provided, full result JSON is written to S3),
    "pageCount":        int (optional, for future use, not currently used)
}

Output
------
{
    "search_id":      str,
    "document_id":    str,
    "status":         str (COMPLETED | PARTIAL | FAILED),
    "expected_cis":   int,
    "completed_cis":  int,
    "failed_cis":     int,
    "batch_summary":  list[dict],      # per-batch status with CI failures
    "results":        list[dict],      # per-CI result with final_hits
    "stage_wall":     dict[str, float],  # max wall-clock per stage
    "wall_time":      float
}

Env vars
--------
  WORKER_LAMBDA_ARN       — ARN of the Search Worker Lambda (required)
                             e.g., arn:aws:lambda:eu-west-1:064051750322:function:rls-ci-retrieval-search-worker
  OPENSEARCH_ENDPOINT     — host only, no https:// (required)
                             e.g., search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com
  OPENSEARCH_CI_INDEX     — CI objects index (default: ci-objects)
  OPENSEARCH_MAXSIZE      — Connection pool size (default: 128)
  EMBEDDING_MODEL         — Bedrock embedding model for CI lookup fallback
  OPEN_SEARCH_REGION              — AWS region (required)
  CI_LOOKUP_WORKERS       — Concurrent threads for CI enrichment (default: 10)
  MAX_WORKERS             — Concurrent Worker Lambda invocations (default: 3, prevents OpenSearch 429s)
  DOCUMENT_ASSETS_PATH    — Local path to document_assets.json (optional)
  
  Connection pool: OPENSEARCH_MAXSIZE (default 128) provides safety margin for concurrent
  CI lookups (up to CI_LOOKUP_WORKERS × 2 threads per invocation)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar, copy_context
from pathlib import Path
from typing import Any

from shared.server_notify import (
    get_cim_annotation_job_cis,
    notify_cim_annotation_job_status,
)
from shared.id_resolver import get_global_ci_id
from shared.utility import getTenantFromEvent

# ── Logging Context Variables (thread-safe for concurrent batch execution) ───────
_ctx_tenant = ContextVar("tenant", default="-")
_ctx_document_id = ContextVar("document_id", default="-")
_ctx_search_id = ContextVar("search_id", default="-")
_ctx_batch_idx = ContextVar("batch_idx", default="-")


class SearchContextFilter(logging.Filter):
    """Logging filter that injects search context into all log records.
    
    Automatically adds [tenant=...] [document=...] [search=...] [batch=...]
    prefix to every log from any module, without requiring manual propagation.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        tenant = _ctx_tenant.get()
        document_id = _ctx_document_id.get()
        search_id = _ctx_search_id.get()
        batch_idx = _ctx_batch_idx.get()
        
        prefix = f"[tenant={tenant}] [document={document_id}] [search={search_id}]"
        if batch_idx != "-":
            prefix += f" [batch={batch_idx}]"
        
        record.msg = f"{prefix} {record.msg}"
        return True


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure root logger so all dynamically loaded modules inherit the context filter
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Add context filter to all existing handlers (preserve Lambda's CloudWatch handler)
context_filter = SearchContextFilter()
for handler in root_logger.handlers:
    handler.addFilter(context_filter)

# If no handlers exist, create one (e.g., local testing)
if not root_logger.handlers:
    import sys
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(levelname)s]\t%(asctime)s\t%(name)s\t%(message)s'
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

# ── Env config ─────────────────────────────────────────────────────────────────
WORKER_LAMBDA_ARN    = os.environ.get("WORKER_LAMBDA_ARN", "")
OPENSEARCH_ENDPOINT  = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_CI_INDEX  = os.environ.get("OPENSEARCH_CI_INDEX", "ci-objects")
OPENSEARCH_TIMEOUT   = int(os.environ.get("OPENSEARCH_TIMEOUT", "30"))
OPENSEARCH_MAXSIZE   = int(os.environ.get("OPENSEARCH_MAXSIZE", "256"))  # Connection pool size
AWS_REGION              = os.environ.get("AWS_REGION", "us-east-1")
OPEN_SEARCH_REGION           = os.environ.get("OPEN_SEARCH_REGION", AWS_REGION)
EMBEDDING_MODEL      = os.environ.get("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
DOCUMENT_ASSETS_PATH = os.environ.get(
    "DOCUMENT_ASSETS_PATH",
    str(ROOT / "localfiles" / "assets" / "document_assets.json"),
)

# Max CIs processed in parallel within a single worker Lambda.
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "3"))
CI_LOOKUP_WORKERS = int(os.environ.get("CI_LOOKUP_WORKERS", "10"))

_DEFAULT_BATCH_SIZE = 50

# ── Lazy singletons ────────────────────────────────────────────────────────────
_aws: dict = {}
_os_client = None


def _get(service: str, region: str | None = None):
    key = f"{service}:{region or ''}"
    if key not in _aws:
        import boto3
        from botocore.config import Config
        
        # For Lambda invocations, set read timeout above the worker Lambda's own
        # timeout (800s) so the orchestrator never gives up client-side before the
        # worker itself would have finished/timed out.
        config = Config(
            read_timeout=850,  # must stay > search-worker Lambda TIMEOUT (800s)
            connect_timeout=10,
            retries={'max_attempts': 1}  # Don't retry on timeout
        ) if service == "lambda" else None
        
        _aws[key] = boto3.client(service, region_name=region, config=config)
    return _aws[key]


def _get_os():
    global _os_client
    if _os_client is None:
        import boto3
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth
        
        # Custom connection class that configures HTTPAdapter pool sizes
        class PooledRequestsHttpConnection(RequestsHttpConnection):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                from requests.adapters import HTTPAdapter
                adapter = HTTPAdapter(
                    pool_connections=OPENSEARCH_MAXSIZE,
                    pool_maxsize=OPENSEARCH_MAXSIZE,
                )
                self.session.mount("https://", adapter)
                self.session.mount("http://", adapter)
        
        frozen  = boto3.Session().get_credentials().get_frozen_credentials()
        awsauth = AWS4Auth(frozen.access_key, frozen.secret_key, OPEN_SEARCH_REGION, "es",
                          session_token=frozen.token)
        _os_client = OpenSearch(
            hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
            http_auth=awsauth, use_ssl=True, verify_certs=True,
            connection_class=PooledRequestsHttpConnection,
            timeout=OPENSEARCH_TIMEOUT,
            max_retries=2,
            retry_on_timeout=True,
            maxsize=OPENSEARCH_MAXSIZE,  # Connection pool size
        )
    return _os_client


# ── Document context ───────────────────────────────────────────────────────────

def _load_document_context(document_id: str) -> dict:
    path = Path(DOCUMENT_ASSETS_PATH)
    if not path.exists():
        return {}
    try:
        with path.open() as fh:
            return json.load(fh).get(document_id, {})
    except Exception as exc:
        logger.warning("[Orchestrator] document_context load failed: %s", exc)
        return {}


# ── CI enrichment lookup ───────────────────────────────────────────────────────

def _lookup_ci(raw_ci: dict, tenant: dict) -> dict | None:
    """Fetch the tenant-scoped enriched CI from the ci-objects index."""
    ci_id = raw_ci.get("id") or raw_ci.get("ci_id")
    if ci_id is None:
        return None

    # CI OpenSearch IDs are tenant-scoped: {tenant_id}__ci_{ci_id}.
    # The backend response already carries tenant_id, but fall back to the
    # request tenant when necessary.  CI is not project-scoped.
    tenant_id = raw_ci.get("tenant_id") or tenant.get("tenant_id") or tenant.get("id")
    global_ci_id = get_global_ci_id(raw_ci, tenant_id=tenant_id)

    try:
        from shared.opensearch_enrichment import ENRICHMENT_DEFAULTS
        resp = _get_os().get(
            index=OPENSEARCH_CI_INDEX,
            id=str(global_ci_id),
            ignore=[404],
        )
        if not resp.get("found"):
            logger.warning(
                "[Orchestrator] CI %s not found in ci-objects _id=%s",
                ci_id, global_ci_id,
            )
            return None

        doc = resp["_source"]
        enrichment_fields = {
            k: doc.get(k, default)
            for k, default in ENRICHMENT_DEFAULTS.items()
        }
        return {
            **raw_ci,
            **enrichment_fields,
            "entities": doc.get("entities", []),
            "knownCI": doc.get("known_ci", raw_ci.get("knownCI", "")),
            "normalization": {
                "normalized_text": doc.get("normalized_text", ""),
                "tokens": doc.get("tokens", []),
                "abbreviations_found": {},
            },
            "ner": {
                "entities": doc.get("entities", []),
                "model": doc.get("ner_model", "gliner"),
            },
            "ontology": {
                "expansions": doc.get("ontology_expansions", []),
                "synonyms": doc.get("ontology_synonyms", {}),
                "regex_patterns": doc.get("regex_patterns", []),
            },
            "embedding": {
                "dense_vector": doc.get("dense_vector", []),
                "sparse_vector": doc.get("sparse_vector", {}),
                "model": doc.get("embedding_model", EMBEDDING_MODEL),
                "dimensions": len(doc.get("dense_vector", [])),
            },
        }
    except Exception as exc:
        logger.warning("[Orchestrator] lookup failed ci_id=%s: %s", ci_id, exc)
        return None


def _load_cis_parallel(
    raw_cis: list[dict], tenant: dict, n_workers: int
) -> list[dict]:
    """Bulk-fetch enriched CIs from ci-objects in parallel."""
    # copy_context() per submit (fresh copy each time) propagates the caller's
    # [tenant=/document=/search=] ContextVars — a single reused Context can't be entered
    # by more than one thread at once and raises "cannot enter context".
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(copy_context().run, _lookup_ci, ci, tenant) for ci in raw_cis]
        results = [f.result() for f in futures]
    enriched = [r for r in results if r is not None]
    logger.info("[Orchestrator] enriched %d/%d CIs", len(enriched), len(raw_cis))
    return enriched


# ── Worker invocation ──────────────────────────────────────────────────────────

def _invoke_worker(batch_payload: dict) -> dict:
    """Synchronously invoke the Search Worker Lambda and return its response."""
    if not WORKER_LAMBDA_ARN:
        raise RuntimeError("WORKER_LAMBDA_ARN env var is not set")
    # Extract function name from ARN (format: arn:aws:lambda:region:account:function:name)
    function_name = WORKER_LAMBDA_ARN.split(":")[-1]
    batch_started_at = time.perf_counter()
    batch_idx = batch_payload.get("batch_idx", "-")
    cis = batch_payload.get("cis", [])
    logger.info(
        "[Orchestrator] worker batch START batch=%s cis=%d "
        "search_id=%s",
        batch_idx,
        len(cis) if isinstance(cis, list) else 0,
        batch_payload.get("search_id", "-"),
    )
    resp = _get("lambda").invoke(
        FunctionName   = function_name,
        InvocationType = "RequestResponse",
        Payload        = json.dumps(batch_payload).encode(),
    )
    raw      = resp["Payload"].read()
    result   = json.loads(raw)
    # Lambda wraps unhandled errors in {"errorMessage": ..., "errorType": ...}
    if "errorMessage" in result:
        logger.error(
            "[Orchestrator] worker batch FAILED batch=%s elapsed=%.3fs "
            "errorType=%s error=%s",
            batch_idx,
            time.perf_counter() - batch_started_at,
            result.get("errorType"),
            result.get("errorMessage"),
        )
        raise RuntimeError(
            f"Worker batch_idx={batch_payload['batch_idx']} failed: "
            f"{result.get('errorType')}: {result.get('errorMessage')}"
        )
    if result.get("results_offloaded"):
        result["results"] = _download_results_from_s3(result.get("results_s3_url", ""))
    return result


def _download_results_from_s3(s3_url: str) -> list:
    """Fetch results offloaded by the worker when its inline response would exceed the 6MB sync-invoke limit."""
    if not s3_url.startswith("s3://"):
        logger.error("[Orchestrator] invalid results_s3_url=%r", s3_url)
        return []
    bucket, _, key = s3_url[len("s3://"):].partition("/")
    body = _get("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    results = json.loads(body)
    logger.info("[Orchestrator] downloaded offloaded results from %s (%d bytes, %d results)", s3_url, len(body), len(results))
    return results


# ── Result merge ───────────────────────────────────────────────────────────────

def _clean_response_object(value):
    """Recursively remove ALL vector/embedding data from response objects.

    The full CI/document metadata remains available to the UI, but embedding
    vectors are never serialized into the terminal S3 response.  Vectors can
    appear anywhere (top level, nested ``embedding`` objects, inside indexed_object,
    entities, facts, etc.), so this must be deeply recursive and aggressive.
    
    Removes:
    - dense_vector, sparse_vector
    - embedding, vector, _embedding, _vector  
    - dense_embedding, sparse_embedding
    - Any *_vector or *_embedding fields
    """
    # Aggressive list of vector field patterns (same as Worker's _strip_vectors)
    VECTOR_KEYWORDS = {
        "dense_vector", "sparse_vector", "dense", "sparse",
        "embedding", "vector", "_embedding", "_vector",
        "dense_embedding", "sparse_embedding",
        "_dense", "_sparse", "embeddings", "vectors",
        "dense_embeddings", "sparse_embeddings"
    }

    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            # Skip if key matches vector patterns (case-insensitive)
            if key.lower() in VECTOR_KEYWORDS or key.endswith("_vector") or key.endswith("_embedding"):
                continue
            # Recursively clean values
            cleaned[key] = _clean_response_object(item)
        return cleaned

    if isinstance(value, list):
        return [_clean_response_object(item) for item in value]

    return value


def _clean_ci_vectors(ci: dict) -> dict:
    """Return the complete UI CI object without embedding/vector payloads."""
    return _clean_response_object(ci)


# ── Handler ────────────────────────────────────────────────────────────────────

def _process_payload(event: dict, context: Any = None) -> dict:
    search_id = event.get("search_id") or str(uuid.uuid4())
    document_id = str(event.get("document_id", ""))
    tenant = getTenantFromEvent(event)
    project_id = str(event.get("project_id", ""))
    cim_job_id = event.get("cimAnnotationJobId")
    bucket_name = event.get("bucketName")
    s3_result_path = event.get("s3ResultPath")
    batch_size = int(event.get("batch_size", _DEFAULT_BATCH_SIZE))
    skip_rerank = bool(event.get("skip_rerank", False))
    skip_verify = bool(event.get("skip_verify", False))

    _ctx_tenant.set(tenant["tenant_name"])
    _ctx_document_id.set(document_id)
    _ctx_search_id.set(search_id)

    # CIM annotation job lifecycle:
    #   start -> ON_GOING
    #   successful completion -> COMPLETED
    #   any orchestrator failure -> FAILED
    #
    # These callbacks are deliberately non-fatal. Search processing must not
    # fail merely because the backend notification endpoint is temporarily down.
    if cim_job_id is not None:
        notify_cim_annotation_job_status(
            job_id=cim_job_id,
            tenant_schema=tenant["tenant_schema"],
            status="ON_GOING",
        )

    try:
        logger.info(
            "[Orchestrator] start cis_source=%s batch_size=%d",
            "cim_annotation_job" if cim_job_id is not None else "event",
            batch_size,
        )

        if cim_job_id is not None:
            # Backend is the source of truth for the CI set belonging to this
            # CIM annotation job. The potentially large CI list never travels
            # through the SQS message.
            raw_cis = get_cim_annotation_job_cis(
                cim_job_id,
                tenant_schema=tenant["tenant_schema"],
            )
        else:
            raw_cis = event.get("cis", [])

        logger.info(
            "[Orchestrator] loaded CIs count=%d job_id=%s",
            len(raw_cis),
            cim_job_id,
        )

        t0 = time.perf_counter()

        doc_context = event.get("document_context") or _load_document_context(document_id)

        raw_ci_map = {ci.get("id"): ci for ci in raw_cis}
        n_lookup_workers = min(CI_LOOKUP_WORKERS, len(raw_cis)) if raw_cis else 1
        enriched = _load_cis_parallel(
            raw_cis,
            tenant,
            n_lookup_workers,
        )

        if not enriched:
            logger.warning(
                "[Orchestrator] no enriched CIs found — returning empty result"
            )

            response = {
                "search_id": search_id,
                "document_id": document_id,
                "status": "COMPLETED",
                "expected_cis": 0,
                "completed_cis": 0,
                "failed_cis": 0,
                "n_cis": 0,
                "n_batches": 0,
                "results": [],
                "stage_wall": {},
                "wall_time": round(time.perf_counter() - t0, 3),
            }

            if cim_job_id is not None:
                notify_cim_annotation_job_status(
                    job_id=cim_job_id,
                    tenant_schema=tenant["tenant_schema"],
                    status="COMPLETED",
                )

            return response

        batches = [
            enriched[i:i + batch_size]
            for i in range(0, len(enriched), batch_size)
        ]
        n_batches = len(batches)

        logger.info(
            "[Orchestrator] %d CIs → %d batches of ≤%d",
            len(enriched),
            n_batches,
            batch_size,
        )

        payloads = [
            {
                "search_id": search_id,
                "batch_idx": idx,
                "cis": batch,
                "document_id": document_id,
                "tenant": tenant,
                "project_id": project_id,
                "document_context": doc_context,
                "skip_rerank": skip_rerank,
                "skip_verify": skip_verify,
            }
            for idx, batch in enumerate(batches)
        ]

        n_invoke_workers = int(min(MAX_WORKERS, n_batches))
        logger.info(
            "[Orchestrator] parallelism: %d batches max",
            n_invoke_workers,
        )

        all_results: list[dict] = [{}] * n_batches
        batch_times: dict[int, float] = {}
        failed_batches: set[int] = set()
        errors: list[dict] = []

        # copy_context() per submit (fresh copy each time) propagates the caller's
        # [tenant=/document=/search=] ContextVars — a single reused Context can't be entered
        # by more than one thread at once and raises "cannot enter context".
        with ThreadPoolExecutor(max_workers=n_invoke_workers) as pool:
            futures = {
                pool.submit(copy_context().run, _invoke_worker, payload): payload["batch_idx"]
                for payload in payloads
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    batch_response = future.result()
                    all_results[idx] = batch_response
                    batch_times[idx] = batch_response.get("wall_time", 0.0)

                    logger.info(
                        "[Orchestrator] batch %d/%d done (%.1fs)",
                        idx + 1,
                        n_batches,
                        batch_response.get("wall_time", 0.0),
                    )
                except Exception as exc:
                    logger.error(
                        "[Orchestrator] batch %d failed: %s",
                        idx,
                        exc,
                    )
                    failed_batches.add(idx)
                    batch_times[idx] = 0.0
                    errors.append({
                        "batch_idx": idx,
                        "error": str(exc),
                    })

        flat_results: list[dict] = []
        total_worker_completed = 0
        total_worker_failed = 0
        batch_summary: list[dict] = []

        for batch_idx, batch_resp in enumerate(all_results):
            batch_results = batch_resp.get("results", [])

            for result in batch_results:
                ci_id = result.get("ci_id") or result.get("id")
                if ci_id and ci_id in raw_ci_map:
                    # Keep the complete CI available to the UI, but strip
                    # retrieval-only embeddings/vectors before S3 serialization.
                    clean_ci = _clean_ci_vectors(raw_ci_map[ci_id])

                    # The UI expects the CI directly on each final hit.
                    # Also keep it at the per-CI result level so consumers
                    # that render the CI independently do not need to recover
                    # it from a hit.
                    result["ci"] = clean_ci

                    for final_hit in result.get("final_hits", []):
                        if isinstance(final_hit, dict):
                            # Keep the complete UI CI on every final hit.
                            final_hit["ci"] = clean_ci

                            # matched_object is useful to the UI (text, geometry,
                            # object/page metadata), but it may contain the large
                            # retrieval-time embedding vectors.  Strip only those
                            # vector fields from matched_object; do not remove the
                            # matched object itself.
                            if "matched_object" in final_hit:
                                final_hit["matched_object"] = _clean_response_object(
                                    final_hit["matched_object"]
                                )

                            # Some worker paths use indexed_object instead of
                            # matched_object. Apply the same terminal-response
                            # cleanup there as well.
                            if "indexed_object" in final_hit:
                                final_hit["indexed_object"] = _clean_response_object(
                                    final_hit["indexed_object"]
                                )

                flat_results.append(result)

            batch_completed = batch_resp.get("completed_cis", 0)
            batch_failed = batch_resp.get("failed_cis", 0)
            total_worker_completed += batch_completed
            total_worker_failed += batch_failed

            expected_batch_cis = (
                len(batches[batch_idx])
                if batch_idx < len(batches)
                else 0
            )

            batch_status = (
                "FAILED"
                if batch_idx in failed_batches
                else (
                    "PARTIAL"
                    if batch_failed > 0
                    else "COMPLETED"
                )
            )

            batch_summary.append({
                "batch_idx": batch_idx,
                "status": batch_status,
                "expected_cis": expected_batch_cis,
                "completed_cis": batch_completed,
                "failed_cis": batch_failed,
                "execution_time_ms": round(
                    batch_times.get(batch_idx, 0.0) * 1000,
                    1,
                ),
                "ci_failures": batch_resp.get("ci_failures", []),
            })

        wall_time = round(time.perf_counter() - t0, 3)
        total_hits = sum(
            len(result.get("final_hits", []))
            for result in flat_results
        )

        total_cis_at_orchestrator = len(enriched)

        if failed_batches and len(failed_batches) == n_batches:
            status = "FAILED"
        elif failed_batches or total_worker_failed > 0:
            status = "PARTIAL"
        else:
            status = "COMPLETED"

        logger.info(
            "[Orchestrator] done wall=%.1fs status=%s "
            "cis_expected=%d cis_completed=%d cis_failed_in_worker=%d "
            "failed_batches=%d",
            wall_time,
            status,
            total_cis_at_orchestrator,
            total_worker_completed,
            total_worker_failed,
            len(failed_batches),
        )

        failed_batches_with_times = []

        for batch_idx in sorted(failed_batches):
            failed_batches_with_times.append({
                "batch_idx": batch_idx,
                "execution_time_ms": round(
                    batch_times.get(batch_idx, 0.0) * 1000,
                    1,
                ),
                "error": next(
                    (
                        error.get("error")
                        for error in errors
                        if error.get("batch_idx") == batch_idx
                    ),
                    None,
                ),
            })

        response = {
            "search_id": search_id,
            "document_id": document_id,
            "status": status,
            "expected_cis": total_cis_at_orchestrator,
            "completed_cis": total_worker_completed,
            "failed_cis": total_worker_failed,
            "n_cis": total_cis_at_orchestrator,
            "n_batches": n_batches,
            "failed_batches": (
                failed_batches_with_times
                if failed_batches
                else []
            ),
            "batch_summary": batch_summary,
            "results": flat_results,
            "wall_time": wall_time,
        }

        if errors:
            response["errors"] = errors

        # CRITICAL: Strip ALL vectors from entire response to prevent 6MB Lambda limit
        # This is the final safety net before serializing for S3 or Lambda response
        response = _clean_response_object(response)

        if bucket_name and s3_result_path:
            s3_key = s3_result_path

            body = json.dumps(
                response,
                default=str,
            ).encode()

            _get("s3").put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=body,
                ContentType="application/json",
            )

            logger.info(
                "[Orchestrator] results written s3://%s/%s (%d bytes)",
                bucket_name,
                s3_key,
                len(body),
            )

            summary = {
                "search_id": search_id,
                "document_id": document_id,
                "status": status,
                "expected_cis": total_cis_at_orchestrator,
                "completed_cis": total_worker_completed,
                "failed_cis": total_worker_failed,
                "total_hits": total_hits,
                "wall_time": wall_time,
                "batch_summary": batch_summary,
                "failed_batches": (
                    failed_batches_with_times
                    if failed_batches
                    else []
                ),
                "s3_bucket": bucket_name,
                "s3_key": s3_key,
            }

            if errors:
                summary["errors"] = errors

            if cim_job_id is not None:
                notify_cim_annotation_job_status(
                    job_id=cim_job_id,
                    tenant_schema=tenant["tenant_schema"],
                    status="COMPLETED" if status == "COMPLETED" else "FAILED",
                )

            return summary

        if cim_job_id is not None:
            notify_cim_annotation_job_status(
                job_id=cim_job_id,
                tenant_schema=tenant["tenant_schema"],
                status="COMPLETED" if status == "COMPLETED" else "FAILED",
            )

        return response

    except Exception as exc:
        logger.exception(
            "[Orchestrator] failed search_id=%s document_id=%s "
            "cimAnnotationJobId=%s error=%s",
            search_id,
            document_id,
            cim_job_id,
            exc,
        )

        if cim_job_id is not None:
            notify_cim_annotation_job_status(
                job_id=cim_job_id,
                tenant_schema=tenant["tenant_schema"],
                status="FAILED",
            )

        raise


# ── AWS SQS entry point ────────────────────────────────────────────────────────

def _decode_sqs_record(record: dict) -> dict:
    """Decode one SQS record into the application payload."""
    if not isinstance(record, dict):
        raise ValueError("SQS record must be a JSON object")

    body = record.get("body")
    if body is None:
        raise ValueError("SQS record is missing body")

    if isinstance(body, str):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"SQS message body is not valid JSON: {exc}") from exc
    elif isinstance(body, dict):
        payload = body
    else:
        raise ValueError("SQS message body must be a JSON object or JSON string")

    if not isinstance(payload, dict):
        raise ValueError("CI/Search payload must be a JSON object")

    return payload


def handler(event: dict, context: Any) -> dict:
    """
    Lambda entry point.

    Supports:
      1. AWS SQS event: {"Records": [{"body": "..."}]}
      2. Direct application payload for local/manual invocation.

    The SQS event source mapping may deliver multiple messages in one
    invocation. Each message is processed independently.
    """
    if not isinstance(event, dict):
        raise ValueError("Lambda event must be a JSON object")

    records = event.get("Records")

    # Direct/manual invocation remains supported.
    if not records:
        return _process_payload(event, context)

    if not isinstance(records, list):
        raise ValueError("Lambda SQS Records must be a list")

    batch_received_at = time.perf_counter()
    logger.info(
        "[Orchestrator] SQS invocation START records=%d",
        len(records),
    )

    results = []
    failures = []

    for index, record in enumerate(records):
        message_id = record.get("messageId", f"record-{index}")
        message_started_at = time.perf_counter()

        try:
            payload = _decode_sqs_record(record)

            logger.info(
                "[Orchestrator] SQS message START "
                "message=%d/%d message_id=%s search_id=%s "
                "cimAnnotationJobId=%s document_id=%s",
                index + 1,
                len(records),
                message_id,
                payload.get("search_id", "-"),
                payload.get("cimAnnotationJobId", "-"),
                payload.get("document_id", "-"),
            )

            result = _process_payload(payload, context)

            message_elapsed = time.perf_counter() - message_started_at
            logger.info(
                "[Orchestrator] SQS message COMPLETE "
                "message=%d/%d message_id=%s status=%s "
                "expected_cis=%s completed_cis=%s failed_cis=%s "
                "batches=%s elapsed=%.3fs",
                index + 1,
                len(records),
                message_id,
                result.get("status"),
                result.get("expected_cis"),
                result.get("completed_cis"),
                result.get("failed_cis"),
                result.get("n_batches"),
                message_elapsed,
            )

            results.append({
                "message_id": message_id,
                "success": True,
                "result": result,
            })

        except Exception as exc:
            message_elapsed = time.perf_counter() - message_started_at
            logger.exception(
                "[Orchestrator] SQS message FAILED "
                "message=%d/%d message_id=%s elapsed=%.3fs error=%s",
                index + 1,
                len(records),
                message_id,
                message_elapsed,
                exc,
            )

            failures.append({
                "itemIdentifier": message_id,
            })

            results.append({
                "message_id": message_id,
                "success": False,
                "error": str(exc),
            })

    invocation_elapsed = time.perf_counter() - batch_received_at

    logger.info(
        "[Orchestrator] SQS invocation COMPLETE "
        "records=%d succeeded=%d failed=%d elapsed=%.3fs",
        len(records),
        len(results) - len(failures),
        len(failures),
        invocation_elapsed,
    )

    # This is intentionally returned in the AWS partial-batch format.
    # It only has an effect when the event source mapping has
    # ReportBatchItemFailures enabled.
    return {
        "batchItemFailures": failures,
        "results": results,
    }
