"""
Alert Lambda: stuck / permanently-failed document (chunk) indexing.

The cron job (rls-backend cron/document-index/recovery.ts, every 30 min)
handles three cases itself:
  1. CLAIMED with no taskArn recorded, stuck > 10 min -> reclaimed.
  2. FAILED / DISPATCH_FAILED / DISPATCH_PARTIAL with attemptCount < 3 -> retried.
  3. CLAIMED/PROCESSING WITH a recorded taskArn -> checked against live ECS
     task status (DescribeTasksCommand), never against elapsed time.

Case 3 is the real gap: if ECS keeps reporting the task as running (hung
task, DescribeTasks misreporting, etc.) the file stays PROCESSING forever
with **no time-based safety net at all** in the cron itself. This Lambda
adds that missing safety net, plus the same two watchdog alerts as the
other two resources:
  1. MISSED_STUCK: CLAIMED (no taskArn) well past the cron's own 10-minute
     window (3x margin), OR PROCESSING for an implausibly long time (6h)
     regardless of what ECS currently reports.
  2. EXHAUSTED_RETRIES: attemptCount has hit the cap (3) and status is
     terminal-failed, so the recovery cron's own WHERE clause permanently
     excludes it.
  3. NEVER_STARTED: extraction is done but indexingStatus is still PENDING/
     NULL well past the backfill cron's 1-minute cadence, meaning the
     initial dispatch never happened for this file.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.scan_runner import run_alert_scan

logger = logging.getLogger()
logger.setLevel(logging.INFO)

RESOURCE = "document-indexing"

# attemptCount is derived (COUNT of file_indexing_attempts rows per file),
# same as db.files.getFilesForDocumentIndexQuery's latestAttempt lateral join.
MISSED_STUCK_SQL = """
    SELECT
        f.id AS "fileId",
        f."indexingStatus",
        ia."status" AS "attemptStatus",
        ia."lastActivityAt",
        ia."meta"->>'taskArn' AS "taskArn",
        (SELECT COUNT(*) FROM {schema}.file_indexing_attempts c WHERE c."fileId" = f.id) AS "attemptCount"
    FROM {schema}.files f
    LEFT JOIN LATERAL (
        SELECT "status", "startedAt", "lastActivityAt", "meta"
        FROM {schema}.file_indexing_attempts ia
        WHERE ia."fileId" = f.id
        ORDER BY ia."startedAt" DESC NULLS LAST, ia.id DESC
        LIMIT 1
    ) ia ON true
    WHERE f.deleted = false
      AND (
        (
          f."indexingStatus" = 'CLAIMED'
          AND ia."meta"->>'taskArn' IS NULL
          AND COALESCE(ia."lastActivityAt", ia."startedAt") < NOW() - INTERVAL '30 minutes'
        )
        OR (
          f."indexingStatus" IN ('CLAIMED', 'PROCESSING')
          AND ia."meta"->>'taskArn' IS NOT NULL
          AND ia."lastActivityAt" < NOW() - INTERVAL '6 hours'
        )
      )
"""

EXHAUSTED_SQL = """
    SELECT
        f.id AS "fileId",
        f."indexingStatus",
        (SELECT COUNT(*) FROM {schema}.file_indexing_attempts c WHERE c."fileId" = f.id) AS "attemptCount"
    FROM {schema}.files f
    WHERE f.deleted = false
      AND f."indexingStatus" IN ('FAILED', 'DISPATCH_FAILED', 'DISPATCH_PARTIAL')
      AND (SELECT COUNT(*) FROM {schema}.file_indexing_attempts c WHERE c."fileId" = f.id) >= 3
"""

# document-indexing-backfill (every 1 min) dispatches files eligible per
# db.files.getFilesForDocumentIndexQuery({backfill:true}): extraction done
# (extractionStatus='TEXT_EXTRACTED' + fullTablesKey present) and
# indexingStatus still PENDING/NULL. Still PENDING well past that window
# means backfill itself missed it.
NEVER_STARTED_SQL = """
    SELECT f.id AS "fileId", f."indexingStatus", f."extractionUpdatedAt"
    FROM {schema}.files f
    WHERE f.deleted = false
      AND f.name ILIKE '%.pdf'
      AND f."extractionStatus" = 'TEXT_EXTRACTED'
      AND f."extractionMeta"->>'fullTablesKey' IS NOT NULL
      AND COALESCE(f."indexingStatus", 'PENDING') = 'PENDING'
      AND f."extractionUpdatedAt" < NOW() - INTERVAL '15 minutes'
"""


def lambda_handler(event, context):
    return run_alert_scan(
        event,
        resource=RESOURCE,
        required_table="file_indexing_attempts",
        queries={
            "MISSED_STUCK": MISSED_STUCK_SQL,
            "EXHAUSTED_RETRIES": EXHAUSTED_SQL,
            "NEVER_STARTED": NEVER_STARTED_SQL,
        },
        require_cim_access=True,
    )
