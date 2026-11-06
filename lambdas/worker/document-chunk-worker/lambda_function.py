from __future__ import annotations

import importlib.util
import json
import logging
import os
import resource
import sys
import time
import types
import uuid
from pathlib import Path
from typing import Any

from shared import embedding_batch
from shared.server_notify import (
    notify_document_failed_chunk,
    notify_document_indexed_chunk,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# In Lambda containers, code lives under /var/task. Use that root when present.
_task_root_env = os.environ.get("LAMBDA_TASK_ROOT")
if _task_root_env and (Path(_task_root_env) / "lambdas").exists():
    ROOT = Path(_task_root_env)
else:
    # Local fallback when running from repository checkout.
    ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_loaded: dict[str, types.ModuleType] = {}
_aws: dict[str, Any] = {}

# Selects which embedding lambda module to load; default keeps Titan behaviour unchanged.
_EMBEDDING_MODULE: tuple[str, str] = {
    "cohere": ("embedding_cohere", "embedding_cohere"),
}.get(os.environ.get("EMBEDDING_PROVIDER", "titan"), ("embedding", "embedding"))

# "sync" (default) keeps today's per-chunk InvokeModel behaviour unchanged.
# "batch" submits ONE Bedrock batch-inference job per SQS invocation, covering
# every chunk this Lambda received in that invocation — whatever the batch
# size turned out to be. A separate embedding-batch-collector Lambda polls the
# job and finishes indexing + the indexed/failed-chunk callbacks once it completes.
EMBEDDING_MODE = os.environ.get("EMBEDDING_MODE", "sync")
EMBEDDING_BATCH_PAYLOAD_BUCKET = os.environ.get("EMBEDDING_BATCH_PAYLOAD_BUCKET", "")
EMBEDDING_BATCH_PAYLOAD_PREFIX = os.environ.get("EMBEDDING_BATCH_PAYLOAD_PREFIX", "extractions/embedding-batch")
EMBEDDING_BATCH_ROLE_ARN = os.environ.get("EMBEDDING_BATCH_ROLE_ARN", "")


def _get(service: str):
    if service not in _aws:
        import boto3

        _aws[service] = boto3.client(service)
    return _aws[service]


def _load(rel_path: str, alias: str) -> types.ModuleType:
    if alias in _loaded:
        return _loaded[alias]

    lf_path = ROOT / "lambdas" / rel_path / "lambda_function.py"
    spec = importlib.util.spec_from_file_location(alias, lf_path)
    mod = importlib.util.module_from_spec(spec)
    lf_dir = str(lf_path.parent)
    if lf_dir not in sys.path:
        sys.path.insert(0, lf_dir)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    _loaded[alias] = mod
    return mod


def _decode_payload(record_or_event: dict) -> dict:
    payload = record_or_event
    if "body" in record_or_event:
        payload = json.loads(record_or_event["body"])

    # SNS-wrapped SQS payload support
    if isinstance(payload, dict) and isinstance(payload.get("Message"), str):
        try:
            payload = json.loads(payload["Message"])
        except Exception:
            pass

    # Optional wrapper format: {"chunk": {...}}
    if isinstance(payload, dict) and isinstance(payload.get("chunk"), dict):
        payload = payload["chunk"]

    # Optional large-payload indirection format
    # {"chunk_id": "...", "s3_payload": {"bucket": "...", "key": "..."}}
    s3_payload = payload.get("s3_payload") if isinstance(payload, dict) else None
    if isinstance(s3_payload, dict):
        bucket = s3_payload["bucket"]
        key = s3_payload["key"]
        obj = _get("s3").get_object(Bucket=bucket, Key=key)
        payload = json.loads(obj["Body"].read())

    if not isinstance(payload, dict):
        raise ValueError("Message payload must be a JSON object")

    payload.setdefault("source_type", "document")
    return payload


# Requeue if less than this many ms remain in the Lambda invocation.
_MIN_REMAINING_MS = 60_000


def _run_chunk(chunk: dict, context: Any = None) -> dict:
    normalize = _load("normalize", "normalize")
    ner = _load("ner", "ner")
    ontology = _load("ontology", "ontology")
    embedding = _load(*_EMBEDDING_MODULE)
    idx = _load("index", "idx")

    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    def _timed(label: str, fn, *args, **kwargs):
        t = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[label] = round(time.perf_counter() - t, 3)
        return result
    chunk = _timed("normalize", normalize._process_document, chunk)
    chunk = _timed("ner",       ner._process_document,       chunk)
    chunk = _timed("ontology",  ontology._process_document,  chunk)
    chunk = _timed("embedding", embedding._process_document, chunk, lambda_ctx=context)
    _timed("index",     idx._process_document,       chunk)

    elapsed = round(time.perf_counter() - t_total, 3)
    extraction = chunk.get("extraction", {})
    obj_list   = extraction.get("objects", [])
    objects    = len(obj_list)
    sentences  = sum(
        sum(1 for s in o.get("display_spans", []) if s.get("type") == "sentence")
        for o in obj_list
    )
    embeddings = sum(1 for o in obj_list if o.get("embedding"))
    embed_api_calls = (
        1  # chunk text
        + (1 if chunk.get("heading_embedding_text") else 0)
        + embeddings  # one per object
        + sum(
            1 for o in obj_list
            for s in o.get("display_spans", [])
            if s.get("type") == "sentence" and s.get("embedding")
        )
    )
    pages = chunk.get("page_end", 0) - chunk.get("page_start", 0) + 1

    # Peak RSS in MB (Linux: ru_maxrss is in KB; macOS: bytes — normalise to MB).
    _rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import platform
    rss_mb = round(_rss_raw / 1024 if platform.system() != "Darwin" else _rss_raw / 1_048_576, 1)

    # Warn on unusually long chunks so pathological sections can be identified.
    _SLOW_CHUNK_THRESHOLD_S = 300
    log_fn = logger.warning if elapsed > _SLOW_CHUNK_THRESHOLD_S else logger.info
    log_fn(
        "[ChunkWorker] stats chunk_id=%s pages=%d objects=%d sentences=%d "
        "embeddings=%d embed_api_calls=%d rss_mb=%.1f "
        "normalize=%.3fs ner=%.3fs ontology=%.3fs embedding=%.3fs index=%.3fs total=%.3fs%s",
        chunk.get("chunk_id"), pages, objects, sentences,
        embeddings, embed_api_calls, rss_mb,
        timings.get("normalize", 0), timings.get("ner", 0),
        timings.get("ontology", 0), timings.get("embedding", 0),
        timings.get("index", 0), elapsed,
        " [SLOW]" if elapsed > _SLOW_CHUNK_THRESHOLD_S else "",
    )

    return {
        "ok": True,
        "document_id": chunk.get("document_id"),
        "chunk_id": chunk.get("chunk_id"),
        "pages": pages,
        "objects": objects,
        "sentences": sentences,
        "embeddings": embeddings,
        "embed_api_calls": embed_api_calls,
        "elapsed_s": elapsed,
        "timings": timings,
    }


def _prepare_chunk_for_batch(chunk: dict, context: Any = None) -> dict:
    """normalize -> ner -> ontology only; embedding/index are deferred to the
    batch-inference collector once the submitted Bedrock job completes."""
    normalize = _load("normalize", "normalize")
    ner = _load("ner", "ner")
    ontology = _load("ontology", "ontology")

    t0 = time.perf_counter()
    chunk = normalize._process_document(chunk)
    chunk = ner._process_document(chunk)
    chunk = ontology._process_document(chunk)
    logger.info(
        "[ChunkWorker] prepared chunk_id=%s elapsed_s=%.3f",
        chunk.get("chunk_id"), time.perf_counter() - t0,
    )
    return chunk


def _submit_embedding_batch(chunks: list[dict], context: Any = None) -> dict | None:
    if not EMBEDDING_BATCH_PAYLOAD_BUCKET:
        raise ValueError("EMBEDDING_BATCH_PAYLOAD_BUCKET is required when EMBEDDING_MODE=batch")
    if not EMBEDDING_BATCH_ROLE_ARN:
        raise ValueError("EMBEDDING_BATCH_ROLE_ARN is required when EMBEDDING_MODE=batch")

    embedding = _load(*_EMBEDDING_MODULE)
    job_submission_id = getattr(context, "aws_request_id", "") or uuid.uuid4().hex

    return embedding_batch.submit_batch_job(
        bedrock_client=_get("bedrock"),
        s3_client=_get("s3"),
        payload_bucket=EMBEDDING_BATCH_PAYLOAD_BUCKET,
        payload_prefix=EMBEDDING_BATCH_PAYLOAD_PREFIX,
        role_arn=EMBEDDING_BATCH_ROLE_ARN,
        model_id=embedding.EMBEDDING_MODEL,
        job_submission_id=job_submission_id,
        chunks=chunks,
        embedding_mod=embedding,
    )


def _get_attempt_id(chunk: dict) -> str | None:
    value = chunk.get("attemptId") or chunk.get("attempt_id")
    return str(value) if value is not None else None


def _is_final_sqs_attempt(record: dict) -> bool:
    """
    Return True only when this delivery is the final retry before SQS moves
    the message to the DLQ.

    Configure MAX_RECEIVE_COUNT to match the SQS redrive policy. If it is
    unavailable, default to 5 and log the decision.
    """
    attributes = record.get("attributes") or {}

    try:
        receive_count = int(attributes.get("ApproximateReceiveCount", "1"))
    except (TypeError, ValueError):
        receive_count = 1

    try:
        max_receive_count = int(os.environ.get("SQS_MAX_RECEIVE_COUNT", "5"))
    except (TypeError, ValueError):
        max_receive_count = 5

    if max_receive_count < 1:
        max_receive_count = 1

    return receive_count >= max_receive_count


def _is_callback_error_retryable(status_code: int | None, exc: Exception) -> bool:
    """
    Classify HTTP response codes as retryable or terminal.

    Retryable (infrastructure/transient):
    - No response (connection refused, timeout, reset, DNS failure)
    - 5xx errors (backend crash, deployment, DB unavailable, pool exhausted, 503, 504)

    Terminal (application/permanent):
    - 4xx errors (invalid data, auth, validation)
    """
    if status_code is None:
        # No HTTP response — network/connection issue
        # (ECONNREFUSED, ETIMEDOUT, ECONNRESET, socket timeout, DNS failure, etc)
        return True

    if 500 <= status_code < 600:
        # 5xx — server error (deployment, crash, DB temporarily unavailable)
        # Idempotent operation, safe to retry.
        return True

    if 400 <= status_code < 500:
        # 4xx — client error (bad request, validation, auth)
        # Retrying won't help; would just waste retries.
        return False

    # Other status codes (3xx, 2xx) are not errors.
    return False


def _retry_callback_with_backoff(
    attempt_id: str,
    chunk_id: str,
    tenant_schema: str,
    max_attempts: int = 4,
) -> None:
    """
    Retry callback with exponential backoff (1s, 2s, 4s, 8s).

    Handles transient failures (503, timeout, connection refused, etc).
    Propagates terminal failures (400, 404, 422, etc) immediately.

    Args:
        attempt_id: File indexing attempt ID.
        chunk_id: Chunk identifier.
        tenant_schema: Database schema.
        max_attempts: Retry budget (default 4 = 1 initial + 3 retries).

    Raises:
        Exception: If all attempts fail, or a terminal error occurs.
    """
    last_exc = None

    for attempt in range(max_attempts):
        try:
            notify_document_indexed_chunk(
                attempt_id=attempt_id,
                chunk_id=chunk_id,
                tenant_schema=tenant_schema,
            )
            # Success
            return

        except Exception as exc:
            last_exc = exc

            # Extract HTTP status code if available
            status_code = getattr(exc, "response", {}).get("status_code") if hasattr(exc, "response") else None

            # Check if retryable
            if not _is_callback_error_retryable(status_code, exc):
                # Terminal error — don't retry
                logger.error(
                    "[ChunkWorker] callback failed (terminal) attempt=%d status=%s "
                    "attempt_id=%s chunk_id=%s error=%s",
                    attempt + 1,
                    status_code,
                    attempt_id,
                    chunk_id,
                    exc,
                )
                raise

            # Transient error — backoff and retry
            if attempt < max_attempts - 1:
                backoff_s = 2 ** attempt  # 1s, 2s, 4s, 8s
                logger.warning(
                    "[ChunkWorker] callback failed (retryable) attempt=%d status=%s "
                    "backoff_s=%d attempt_id=%s chunk_id=%s error=%s",
                    attempt + 1,
                    status_code,
                    backoff_s,
                    attempt_id,
                    chunk_id,
                    exc,
                )
                time.sleep(backoff_s)
            else:
                # Final attempt exhausted
                logger.error(
                    "[ChunkWorker] callback failed (exhausted retries) status=%s "
                    "attempt_id=%s chunk_id=%s error=%s",
                    status_code,
                    attempt_id,
                    chunk_id,
                    exc,
                )

    # All retries exhausted — propagate last exception to SQS
    if last_exc:
        raise last_exc


def _notify_indexed(chunk: dict, result: dict) -> None:
    attempt_id = _get_attempt_id(chunk)
    tenant_schema = chunk.get("tenant_schema")

    # Validation failures: data is missing, so callback cannot be made.
    # This is NOT a transient failure — chunk is missing required fields.
    # We skip the callback but do NOT fail the Lambda (data quality issue, not network).
    if not attempt_id:
        logger.warning(
            "[ChunkWorker] indexed callback skipped: missing attemptId chunk_id=%s",
            result.get("chunk_id"),
        )
        return

    chunk_id = str(result.get("chunk_id") or chunk.get("chunk_id") or "")

    if not chunk_id:
        logger.warning(
            "[ChunkWorker] indexed callback skipped: missing chunk_id attempt_id=%s",
            attempt_id,
        )
        return

    # Retry callback with exponential backoff for transient failures (503, timeout, etc).
    # Terminal failures (400, 404, 422) propagate immediately without retry.
    # After application-level retries exhausted, exception propagates to handler.
    # Handler adds message to SQS batchItemFailures, enabling SQS retry layer.
    _retry_callback_with_backoff(
        attempt_id=attempt_id,
        chunk_id=chunk_id,
        tenant_schema=tenant_schema,
    )


def _notify_failed(chunk: dict, exc: Exception) -> None:
    attempt_id = _get_attempt_id(chunk)
    tenant_schema = chunk.get("tenant_schema")
    chunk_id = str(chunk.get("chunk_id") or "")

    if not attempt_id:
        logger.warning(
            "[ChunkWorker] failed callback skipped: missing attemptId chunk_id=%s",
            chunk_id,
        )
        return

    if not tenant_schema:
        logger.warning(
            "[ChunkWorker] failed callback skipped: missing tenant attempt_id=%s chunk_id=%s",
            attempt_id,
            chunk_id,
        )
        return

    notify_document_failed_chunk(
        attempt_id=attempt_id,
        chunk_id=chunk_id,
        error=str(exc),
        tenant_schema=tenant_schema,
    )


def _handle_batch_sqs_invoke(event: dict, context: Any) -> dict:
    """EMBEDDING_MODE=batch: run normalize/ner/ontology for every record this
    invocation received, then submit ONE Bedrock batch-inference job covering
    the whole invocation's batch (whatever size SQS handed us). Embedding,
    indexing, and the indexed/failed-chunk callbacks happen later in
    embedding-batch-collector once that job completes."""
    failures: list[dict[str, str]] = []
    processed = 0
    prepared: list[tuple[str, dict]] = []

    for record in event.get("Records", []):
        message_id = record.get("messageId", "")

        if context is not None:
            remaining_ms = context.get_remaining_time_in_millis()
            if remaining_ms < _MIN_REMAINING_MS:
                logger.warning(
                    "[ChunkWorker] only %d ms remaining — requeueing message_id=%s",
                    remaining_ms, message_id,
                )
                if message_id:
                    failures.append({"itemIdentifier": message_id})
                continue

        chunk = None
        try:
            chunk = _decode_payload(record)
            sent_ts_str = record.get("attributes", {}).get("SentTimestamp")
            if sent_ts_str:
                chunk["_queue_wait_ms"] = int(time.time() * 1000) - int(sent_ts_str)
            if context is not None:
                chunk["_memory_limit_mb"] = int(getattr(context, "memory_limit_in_mb", 0) or 0)
            chunk = _prepare_chunk_for_batch(chunk, context)
            prepared.append((message_id, chunk))
            processed += 1
        except Exception as exc:
            chunk_id = chunk.get("chunk_id", "?") if chunk else "?"
            logger.exception(
                "[ChunkWorker] prepare failed message_id=%s chunk_id=%s error=%s",
                message_id, chunk_id, exc,
            )
            if chunk is not None and _is_final_sqs_attempt(record):
                _notify_failed(chunk, exc)
            if message_id:
                failures.append({"itemIdentifier": message_id})

    if prepared:
        try:
            _submit_embedding_batch([c for _, c in prepared], context)
        except embedding_batch.InsufficientRecordsError as exc:
            # Too few embeddable records for a Bedrock batch job (e.g. a
            # trailing partial batch at the end of a document). Rather than
            # looping on the same failure until DLQ, embed these synchronously.
            logger.warning(
                "[ChunkWorker] %s for %d chunks — falling back to synchronous embedding",
                exc, len(prepared),
            )
            embedding = _load(*_EMBEDDING_MODULE)
            idx = _load("index", "idx")
            for message_id, chunk in prepared:
                try:
                    chunk = embedding._process_document(chunk, lambda_ctx=context)
                    idx._process_document(chunk)
                    _notify_indexed(chunk, {"chunk_id": chunk.get("chunk_id")})
                except Exception as chunk_exc:
                    logger.exception(
                        "[ChunkWorker] synchronous embedding fallback failed message_id=%s chunk_id=%s error=%s",
                        message_id, chunk.get("chunk_id", "?"), chunk_exc,
                    )
                    _notify_failed(chunk, chunk_exc)
                    if message_id:
                        failures.append({"itemIdentifier": message_id})
                    processed -= 1
        except Exception as exc:
            # Submission is all-or-nothing for this invocation's batch: if the
            # Bedrock job never got submitted, none of these chunks were handed
            # off anywhere, so every one of them must go back on the queue.
            logger.exception(
                "[ChunkWorker] batch embedding submission failed for %d chunks: %s",
                len(prepared), exc,
            )
            for message_id, _chunk in prepared:
                if message_id:
                    failures.append({"itemIdentifier": message_id})
            processed -= len(prepared)

    return {
        "processed": processed,
        "failed": len(failures),
        "batchItemFailures": failures,
    }


def handler(event: dict, context: Any) -> dict:
    # Direct invoke for smoke testing
    if "Records" not in event:
        result = _run_chunk(_decode_payload(event), context)
        logger.info(
            "[ChunkWorker] done chunk_id=%s elapsed_s=%s",
            result.get("chunk_id"),
            result.get("elapsed_s"),
        )
        return result

    # SQS batch invoke
    if EMBEDDING_MODE == "batch":
        return _handle_batch_sqs_invoke(event, context)

    failures: list[dict[str, str]] = []
    processed = 0

    for record in event.get("Records", []):
        message_id = record.get("messageId", "")

        # Safety: if Lambda is nearly out of time, requeue rather than timeout.
        if context is not None:
            remaining_ms = context.get_remaining_time_in_millis()
            if remaining_ms < _MIN_REMAINING_MS:
                logger.warning(
                    "[ChunkWorker] only %d ms remaining — requeueing message_id=%s",
                    remaining_ms, message_id,
                )
                if message_id:
                    failures.append({"itemIdentifier": message_id})
                continue

        chunk = None
        try:
            chunk = _decode_payload(record)
            # Stamp queue wait time and memory limit for observability
            sent_ts_str = record.get("attributes", {}).get("SentTimestamp")
            if sent_ts_str:
                chunk["_queue_wait_ms"] = int(time.time() * 1000) - int(sent_ts_str)
            if context is not None:
                chunk["_memory_limit_mb"] = int(getattr(context, "memory_limit_in_mb", 0) or 0)
            result = _run_chunk(chunk, context)

            # Notify server that chunk was indexed.
            # CRITICAL: If this callback fails (network error, backend down, etc),
            # the exception MUST propagate to the outer except block so that:
            # 1. message_id is added to batchItemFailures
            # 2. SQS does NOT acknowledge this message
            # 3. SQS retries after visibility timeout expires
            # This ensures the database update eventually happens when backend recovers.
            # We do NOT swallow callback exceptions here.
            _notify_indexed(chunk, result)

            processed += 1
            logger.info(
                "[ChunkWorker] done chunk_id=%s elapsed_s=%s",
                result.get("chunk_id"),
                result.get("elapsed_s"),
            )
        except Exception as exc:
            chunk_id = chunk.get("chunk_id", "?") if chunk else "?"
            logger.exception(
                "[ChunkWorker] failed message_id=%s chunk_id=%s error=%s",
                message_id, chunk_id, exc,
            )

            # Do not increment failedChunks for transient attempts. The same
            # SQS message can be retried multiple times. Count it only on the
            # final configured receive attempt, immediately before DLQ.
            if chunk is not None and _is_final_sqs_attempt(record):
                _notify_failed(chunk, exc)

            # CRITICAL: Add to batchItemFailures so SQS does NOT acknowledge this message.
            # Message will become visible again after visibility timeout.
            # This applies to ALL failures:
            # - indexing failures (OpenSearch unavailable)
            # - callback failures (backend unavailable during deployment)
            # - data quality issues (missing chunk_id, etc)
            if message_id:
                failures.append({"itemIdentifier": message_id})

    return {
        "processed": processed,
        "failed": len(failures),
        "batchItemFailures": failures,
    }
