"""
Bedrock batch-inference helpers for the document embedding pipeline.

Submit side (document-chunk-worker, EMBEDDING_MODE=batch):
  build_manifest(chunks, embedding_mod) -> (manifest_jsonl_bytes, mapping: dict)
  submit_batch_job(bedrock, s3, ...) -> job record dict

Collect side (embedding-batch-collector):
  list_pending_jobs(s3, bucket) -> list[job record]
  find_output_key(s3, bucket, output_prefix) -> str | None
  parse_output(s3, bucket, output_key) -> dict[recordId, embedding_vector]
  apply_embeddings(chunk, descriptors, outputs, embedding_mod) -> chunk (mutated)

recordId format is a short synthetic sequence number ("r00000001") — NOT the
chunk_id — so Bedrock's recordId length/charset limits are never a concern.
The mapping.json file (recordId -> descriptor) is the only thing that ties a
result back to a chunk/object/sentence.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

JOBS_PREFIX = "embedding-batch/jobs"

# Bedrock rejects batch-inference jobs with fewer than this many records
# (e.g. "contains less records (1) than the required minimum of: 100").
MIN_BATCH_RECORDS = 100


class InsufficientRecordsError(Exception):
    """Raised when a candidate batch has fewer embeddable records than Bedrock's
    minimum, so the job was never submitted — caller should fall back to
    synchronous (on-demand) embedding for these chunks instead of retrying."""

    def __init__(self, record_count: int):
        self.record_count = record_count
        super().__init__(
            f"only {record_count} embeddable records, below Bedrock's minimum of {MIN_BATCH_RECORDS}"
        )


def _embeddable_texts(chunk: dict, embedding_mod: Any) -> list[dict]:
    """Every text in this chunk that still needs a dense embedding, with a
    descriptor describing where the resulting vector belongs."""
    chunk_id = chunk.get("chunk_id", "")
    items: list[dict] = []

    text = chunk.get("normalization", {}).get("normalized_text", "")
    items.append({"chunk_id": chunk_id, "kind": "chunk", "text": text})

    heading_text = chunk.get("heading_embedding_text", "")
    if heading_text:
        items.append({"chunk_id": chunk_id, "kind": "heading", "text": heading_text})

    objects = chunk.get("extraction", {}).get("objects", [])
    for obj_idx, obj in enumerate(objects):
        if obj.get("text") and not obj.get("embedding"):
            items.append({
                "chunk_id": chunk_id, "kind": "obj", "obj_idx": obj_idx,
                "text": embedding_mod._object_embedding_text(obj),
            })
        for span_idx, span in enumerate(obj.get("display_spans", [])):
            if span.get("type") == "sentence" and span.get("text") and not span.get("embedding"):
                items.append({
                    "chunk_id": chunk_id, "kind": "span", "obj_idx": obj_idx, "span_idx": span_idx,
                    "text": embedding_mod._sentence_embedding_text(obj, span),
                })

    return items


def build_manifest(chunks: list[dict], embedding_mod: Any) -> tuple[bytes, dict[str, dict]]:
    """Build the Bedrock batch-inference input JSONL plus the recordId->descriptor map.

    Returns (manifest_jsonl_bytes, mapping) where mapping[recordId] = descriptor
    (without the "text" key — only chunk_id/kind/obj_idx/span_idx).
    """
    lines: list[str] = []
    mapping: dict[str, dict] = {}
    seq = 0

    for chunk in chunks:
        for item in _embeddable_texts(chunk, embedding_mod):
            record_id = f"r{seq:08d}"
            seq += 1
            text = item.pop("text")
            mapping[record_id] = item
            lines.append(json.dumps({
                "recordId": record_id,
                "modelInput": {"inputText": text[:18_000]},
            }, separators=(",", ":")))

    manifest_bytes = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    return manifest_bytes, mapping


def submit_batch_job(
    *,
    bedrock_client: Any,
    s3_client: Any,
    payload_bucket: str,
    payload_prefix: str,
    role_arn: str,
    model_id: str,
    job_submission_id: str,
    chunks: list[dict],
    embedding_mod: Any,
) -> dict | None:
    """Writes manifest/mapping/per-chunk payloads to S3 and submits one Bedrock
    batch-inference job covering every embeddable text across `chunks`.

    Returns the job record dict (also written to S3 under JOBS_PREFIX) or None
    if `chunks` had nothing left to embed.

    Raises InsufficientRecordsError (before any S3 write or Bedrock call) if
    there are embeddable records but fewer than Bedrock's MIN_BATCH_RECORDS.
    """
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    manifest_bytes, mapping = build_manifest(chunks, embedding_mod)
    timings["build_manifest"] = round(time.perf_counter() - t0, 3)
    if not mapping:
        logger.info("[EmbeddingBatch] nothing to embed for job_submission_id=%s (skipped)", job_submission_id)
        return None
    if len(mapping) < MIN_BATCH_RECORDS:
        raise InsufficientRecordsError(len(mapping))

    base = f"{payload_prefix}/{job_submission_id}"
    manifest_key = f"{base}/manifest.jsonl"
    mapping_key = f"{base}/mapping.json"
    chunks_prefix = f"{base}/chunks"
    output_prefix = f"{base}/output/"

    t0 = time.perf_counter()
    s3_client.put_object(Bucket=payload_bucket, Key=manifest_key, Body=manifest_bytes)
    s3_client.put_object(
        Bucket=payload_bucket, Key=mapping_key,
        Body=json.dumps(mapping, separators=(",", ":")).encode("utf-8"),
    )

    chunk_ids: list[str] = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        chunk_ids.append(chunk_id)
        s3_client.put_object(
            Bucket=payload_bucket, Key=f"{chunks_prefix}/{chunk_id}.json",
            Body=json.dumps(chunk, separators=(",", ":")).encode("utf-8"),
        )
    timings["s3_upload"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    job_name = f"doc-embed-{job_submission_id}"[:63]
    response = bedrock_client.create_model_invocation_job(
        jobName=job_name,
        roleArn=role_arn,
        modelId=model_id,
        inputDataConfig={"s3InputDataConfig": {"s3Uri": f"s3://{payload_bucket}/{manifest_key}"}},
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{payload_bucket}/{output_prefix}"}},
    )
    timings["create_job"] = round(time.perf_counter() - t0, 3)
    job_arn = response["jobArn"]

    job_record = {
        "job_submission_id": job_submission_id,
        "job_arn": job_arn,
        "payload_bucket": payload_bucket,
        "manifest_key": manifest_key,
        "mapping_key": mapping_key,
        "chunks_prefix": chunks_prefix,
        "output_prefix": output_prefix,
        "chunk_ids": chunk_ids,
        "record_count": len(mapping),
        "created_at": time.time(),
    }
    s3_client.put_object(
        Bucket=payload_bucket, Key=f"{JOBS_PREFIX}/{job_submission_id}.json",
        Body=json.dumps(job_record, separators=(",", ":")).encode("utf-8"),
    )

    logger.info(
        "[EmbeddingBatch] submitted job_arn=%s chunks=%d records=%d job_submission_id=%s "
        "build_manifest=%.3fs s3_upload=%.3fs create_job=%.3fs",
        job_arn, len(chunk_ids), len(mapping), job_submission_id,
        timings["build_manifest"], timings["s3_upload"], timings["create_job"],
    )
    return job_record


def list_pending_jobs(s3_client: Any, bucket: str) -> list[dict]:
    jobs: list[dict] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{JOBS_PREFIX}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
            jobs.append(json.loads(body))
    return jobs


def find_job_by_arn(s3_client: Any, bucket: str, job_arn: str) -> dict | None:
    """Used by the EventBridge-triggered path to resolve one job record
    without scanning on every invocation's worth of unrelated jobs."""
    for job in list_pending_jobs(s3_client, bucket):
        if job.get("job_arn") == job_arn:
            return job
    return None


def find_output_key(s3_client: Any, bucket: str, output_prefix: str) -> str | None:
    """Bedrock writes <manifest_basename>.jsonl.out under a job-id subfolder of
    output_prefix. Find it by suffix rather than assuming the exact path."""
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=output_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".jsonl.out"):
                return obj["Key"]
    return None


def parse_output(s3_client: Any, bucket: str, output_key: str) -> dict[str, list[float]]:
    body = s3_client.get_object(Bucket=bucket, Key=output_key)["Body"].read().decode("utf-8")
    vectors: dict[str, list[float]] = {}
    for line in body.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        model_output = record.get("modelOutput") or {}
        embedding = model_output.get("embedding")
        if embedding is not None:
            vectors[record["recordId"]] = embedding
    return vectors


def apply_embeddings(
    chunk: dict,
    descriptors: list[tuple[str, dict]],
    vectors: dict[str, list[float]],
    embedding_mod: Any,
) -> dict:
    """Mutates `chunk` in place: attaches the dense vectors produced by the
    batch job, then fills in the sparse vector/model/dimensions exactly as
    `embedding._process_document` does for the synchronous path."""
    dense_vector: list[float] = []
    heading_dense_vector: list[float] = []

    objects = chunk.get("extraction", {}).get("objects", [])
    for record_id, desc in descriptors:
        vector = vectors.get(record_id)
        if vector is None:
            logger.warning(
                "[EmbeddingBatch] missing output for recordId=%s chunk_id=%s kind=%s",
                record_id, desc.get("chunk_id"), desc.get("kind"),
            )
            continue

        kind = desc["kind"]
        if kind == "chunk":
            dense_vector = vector
        elif kind == "heading":
            heading_dense_vector = vector
        elif kind == "obj":
            objects[desc["obj_idx"]]["embedding"] = vector
        elif kind == "span":
            objects[desc["obj_idx"]]["display_spans"][desc["span_idx"]]["embedding"] = vector

    tokens = chunk.get("normalization", {}).get("tokens", [])
    sparse_vector = embedding_mod._generate_sparse_embedding(tokens)

    chunk["embedding"] = {
        "dense_vector": dense_vector,
        "heading_dense_vector": heading_dense_vector,
        "sparse_vector": sparse_vector,
        "model": embedding_mod.EMBEDDING_MODEL,
        "dimensions": len(dense_vector),
    }
    return chunk
