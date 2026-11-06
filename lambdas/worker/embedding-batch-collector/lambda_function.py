"""
Embedding-batch collector Lambda.

Primary trigger: an EventBridge rule matching Bedrock's "Batch Inference Job
State Change" event (source=aws.bedrock) — fires near-instantly when a job
submitted by document-chunk-worker (EMBEDDING_MODE=batch) completes/fails, so
this handler processes exactly that one job without polling.

Secondary trigger: a coarse scheduled invocation (e.g. rate(15 minutes)) with
no matching event detail — scans every pending job under
s3://<bucket>/embedding-batch/jobs/*.json and polls GetModelInvocationJob.
Bedrock emits events on a best-effort basis, so this is a safety net for any
dropped event, not the primary mechanism.

For a job that is Completed/Failed/Stopped:
  - Completed  -> download the Bedrock batch output, re-hydrate each pending
                  chunk from S3, attach embeddings, run idx._process_document
                  unchanged, notify_document_indexed_chunk per chunk_id, then
                  clean up the job's S3 objects.
  - Failed/Stopped -> notify_document_failed_chunk for every chunk_id in the
                  job, then clean up.
  - anything else (Submitted/Validating/InProgress/Scheduled) -> leave it, try
                  again next run.

This intentionally mirrors document-chunk-worker's dynamic module loading
(ROOT/_load) so embedding._generate_sparse_embedding/EMBEDDING_MODEL and
idx._process_document stay the single source of truth — nothing here
reimplements that logic.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

from shared import embedding_batch
from shared.server_notify import (
    notify_document_failed_chunk,
    notify_document_indexed_chunk,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_task_root_env = os.environ.get("LAMBDA_TASK_ROOT")
if _task_root_env and (Path(_task_root_env) / "lambdas").exists():
    ROOT = Path(_task_root_env)
else:
    ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_loaded: dict[str, types.ModuleType] = {}
_aws: dict[str, Any] = {}

_EMBEDDING_MODULE: tuple[str, str] = {
    "cohere": ("embedding_cohere", "embedding_cohere"),
}.get(os.environ.get("EMBEDDING_PROVIDER", "titan"), ("embedding", "embedding"))

EMBEDDING_BATCH_PAYLOAD_BUCKET = os.environ.get("EMBEDDING_BATCH_PAYLOAD_BUCKET", "")

# Terminal Bedrock batch-inference job statuses.
_COMPLETED_STATUSES = {"Completed"}
_FAILED_STATUSES = {"Failed", "Stopped", "PartiallyCompleted"}


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


def _cleanup_job(s3_client: Any, job: dict) -> None:
    bucket = job["payload_bucket"]
    # Derive the job's data prefix from a key we know was written under it,
    # rather than assuming JOBS_PREFIX's sibling folder name — stays correct
    # even if EMBEDDING_BATCH_PAYLOAD_PREFIX is customized.
    job_data_prefix = job["manifest_key"].rsplit("/", 1)[0] + "/"

    paginator = s3_client.get_paginator("list_objects_v2")
    keys_to_delete = [f"{embedding_batch.JOBS_PREFIX}/{job['job_submission_id']}.json"]
    for page in paginator.paginate(Bucket=bucket, Prefix=job_data_prefix):
        keys_to_delete.extend(obj["Key"] for obj in page.get("Contents", []))

    for i in range(0, len(keys_to_delete), 1000):
        batch = keys_to_delete[i:i + 1000]
        s3_client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in batch]})


def _handle_completed_job(job: dict) -> None:
    s3_client = _get("s3")
    bucket = job["payload_bucket"]
    embedding = _load(*_EMBEDDING_MODULE)
    idx = _load("index", "idx")

    mapping = json.loads(s3_client.get_object(Bucket=bucket, Key=job["mapping_key"])["Body"].read())

    output_key = embedding_batch.find_output_key(s3_client, bucket, job["output_prefix"])
    if not output_key:
        logger.warning(
            "[EmbeddingBatchCollector] job_arn=%s marked Completed but no output file found yet "
            "under %s — will retry next run",
            job["job_arn"], job["output_prefix"],
        )
        return

    vectors = embedding_batch.parse_output(s3_client, bucket, output_key)

    descriptors_by_chunk: dict[str, list[tuple[str, dict]]] = {}
    for record_id, desc in mapping.items():
        descriptors_by_chunk.setdefault(desc["chunk_id"], []).append((record_id, desc))

    indexed = 0
    failed = 0
    for chunk_id in job["chunk_ids"]:
        chunk_key = f"{job['chunks_prefix']}/{chunk_id}.json"
        try:
            chunk = json.loads(s3_client.get_object(Bucket=bucket, Key=chunk_key)["Body"].read())
            descriptors = descriptors_by_chunk.get(chunk_id, [])
            chunk = embedding_batch.apply_embeddings(chunk, descriptors, vectors, embedding)
            idx._process_document(chunk)

            attempt_id = chunk.get("attemptId") or chunk.get("attempt_id")
            tenant_schema = chunk.get("tenant_schema")
            if attempt_id and tenant_schema:
                notify_document_indexed_chunk(
                    attempt_id=str(attempt_id), chunk_id=chunk_id, tenant_schema=tenant_schema,
                )
            else:
                logger.warning(
                    "[EmbeddingBatchCollector] indexed chunk_id=%s but missing attempt_id/tenant_schema "
                    "— skipping indexed-chunk callback",
                    chunk_id,
                )
            indexed += 1
        except Exception as exc:
            logger.exception(
                "[EmbeddingBatchCollector] failed to finish chunk_id=%s job_arn=%s error=%s",
                chunk_id, job["job_arn"], exc,
            )
            try:
                chunk = json.loads(s3_client.get_object(Bucket=bucket, Key=chunk_key)["Body"].read())
                attempt_id = chunk.get("attemptId") or chunk.get("attempt_id")
                tenant_schema = chunk.get("tenant_schema")
                if attempt_id and tenant_schema:
                    notify_document_failed_chunk(
                        attempt_id=str(attempt_id), chunk_id=chunk_id, error=str(exc),
                        tenant_schema=tenant_schema,
                    )
            except Exception:
                logger.exception(
                    "[EmbeddingBatchCollector] could not even load chunk_id=%s to report failure",
                    chunk_id,
                )
            failed += 1

    logger.info(
        "[EmbeddingBatchCollector] job_arn=%s completed indexed=%d failed=%d",
        job["job_arn"], indexed, failed,
    )
    _cleanup_job(s3_client, job)


def _handle_failed_job(job: dict, status: str) -> None:
    s3_client = _get("s3")
    bucket = job["payload_bucket"]
    for chunk_id in job["chunk_ids"]:
        try:
            chunk = json.loads(
                s3_client.get_object(Bucket=bucket, Key=f"{job['chunks_prefix']}/{chunk_id}.json")["Body"].read()
            )
            attempt_id = chunk.get("attemptId") or chunk.get("attempt_id")
            tenant_schema = chunk.get("tenant_schema")
            if attempt_id and tenant_schema:
                notify_document_failed_chunk(
                    attempt_id=str(attempt_id), chunk_id=chunk_id,
                    error=f"Bedrock batch-inference job {status}", tenant_schema=tenant_schema,
                )
        except Exception:
            logger.exception(
                "[EmbeddingBatchCollector] could not report failure for chunk_id=%s job_arn=%s",
                chunk_id, job["job_arn"],
            )

    logger.error(
        "[EmbeddingBatchCollector] job_arn=%s status=%s — %d chunks marked failed",
        job["job_arn"], status, len(job["chunk_ids"]),
    )
    _cleanup_job(s3_client, job)


def handler(event: dict, context: Any) -> dict:
    bucket = EMBEDDING_BATCH_PAYLOAD_BUCKET
    if not bucket:
        raise ValueError("EMBEDDING_BATCH_PAYLOAD_BUCKET is required")

    s3_client = _get("s3")

    # Primary path: EventBridge "Batch Inference Job State Change" event —
    # https://docs.aws.amazon.com/bedrock/latest/userguide/monitoring-eventbridge-how-it-works.html
    # {"source": "aws.bedrock", "detail-type": "Batch Inference Job State Change",
    #  "detail": {"batchJobArn": "...", "status": "Completed", ...}}
    # Bedrock emits these on a best-effort basis, so a coarse scheduled
    # invocation (no matching detail) still falls through to the full scan
    # below as a safety net for any dropped event.
    detail = event.get("detail") if isinstance(event, dict) else None
    if event.get("source") == "aws.bedrock" and detail and detail.get("batchJobArn"):
        job_arn = detail["batchJobArn"]
        status = detail.get("status", "")
        job = embedding_batch.find_job_by_arn(s3_client, bucket, job_arn)
        if not job:
            logger.warning(
                "[EmbeddingBatchCollector] event for job_arn=%s status=%s has no matching job record "
                "(already cleaned up, or submitted before this Lambda existed)",
                job_arn, status,
            )
            return {"handled": False, "job_arn": job_arn, "status": status}

        if status in _COMPLETED_STATUSES:
            _handle_completed_job(job)
        elif status in _FAILED_STATUSES:
            _handle_failed_job(job, status)
        else:
            logger.info(
                "[EmbeddingBatchCollector] job_arn=%s status=%s is non-terminal, ignoring", job_arn, status,
            )
        return {"handled": True, "job_arn": job_arn, "status": status}

    # Fallback path: scheduled/manual invocation — scan every pending job and
    # poll its real status, to catch anything the event-driven path missed.
    bedrock = _get("bedrock")
    jobs = embedding_batch.list_pending_jobs(s3_client, bucket)

    completed = 0
    failed = 0
    still_running = 0

    for job in jobs:
        status_resp = bedrock.get_model_invocation_job(jobIdentifier=job["job_arn"])
        status = status_resp.get("status", "")

        if status in _COMPLETED_STATUSES:
            _handle_completed_job(job)
            completed += 1
        elif status in _FAILED_STATUSES:
            _handle_failed_job(job, status)
            failed += 1
        else:
            still_running += 1

    logger.info(
        "[EmbeddingBatchCollector] pending_jobs=%d completed=%d failed=%d still_running=%d",
        len(jobs), completed, failed, still_running,
    )
    return {"pending_jobs": len(jobs), "completed": completed, "failed": failed, "still_running": still_running}
