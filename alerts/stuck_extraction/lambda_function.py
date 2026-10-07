"""
Alert Lambda: stuck / permanently-failed PDF text extraction.

The cron jobs (rls-backend cron/extraction/{retry,stuck}.ts) already retry
and recover extraction failures on their own schedule:
  - extraction-retry (every 10 min): retries TEXT_EXTRACTION_FAILED files
    with extractionFailedAttempt < 3.
  - extraction-stuck (hourly): re-dispatches EXTRACTING files stuck > 1h,
    PENDING/QUEUED files stuck > 30 min, provided extractionFailedAttempt < 3.

This Lambda is a watchdog, not a recovery path. It catches three things those
crons cannot catch themselves:
  1. MISSED_STUCK: a file is still stuck well past the cron's own window
     (3x margin), meaning the cron missed it (crashed, deploy gap, DB down).
  2. EXHAUSTED_RETRIES: extractionFailedAttempt has hit the cap (3), so the
     retry cron's own WHERE clause permanently excludes it — a silent
     terminal failure unless someone is told.
  3. NEVER_STARTED: extractionStatus is still NULL well past the backfill
     cron's 1-minute cadence, meaning even the initial dispatch never
     happened for this file.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.scan_runner import run_alert_scan

logger = logging.getLogger()
logger.setLevel(logging.INFO)

RESOURCE = "file-extraction"

# 3x the cron's own stuck windows (EXTRACTING >1h, PENDING/QUEUED >30min).
MISSED_STUCK_SQL = """
    SELECT id AS "fileId", "extractionStatus", "extractionUpdatedAt", "extractionFailedAttempt"
    FROM {schema}.files
    WHERE deleted = false
      AND name ILIKE '%.pdf'
      AND (
        ("extractionStatus" = 'EXTRACTING' AND "extractionUpdatedAt" < NOW() - INTERVAL '3 hours')
        OR ("extractionStatus" IN ('PENDING', 'QUEUED') AND "extractionUpdatedAt" < NOW() - INTERVAL '90 minutes')
      )
      AND COALESCE("extractionFailedAttempt", 0) < 3
"""

EXHAUSTED_SQL = """
    SELECT id AS "fileId", "extractionStatus", "extractionUpdatedAt", "extractionFailedAttempt"
    FROM {schema}.files
    WHERE deleted = false
      AND name ILIKE '%.pdf'
      AND "extractionStatus" = 'TEXT_EXTRACTION_FAILED'
      AND COALESCE("extractionFailedAttempt", 0) >= 3
"""

# extraction-backfill (every 1 min) dispatches extractionStatus IS NULL files
# (db.files.getFilesWithoutExtraction). A file still NULL well past that
# window means backfill itself missed it — nothing else will ever pick it up.
NEVER_STARTED_SQL = """
    SELECT id AS "fileId", "createdAt"
    FROM {schema}.files
    WHERE deleted = false
      AND name ILIKE '%.pdf'
      AND "extractionStatus" IS NULL
      AND "createdAt" < NOW() - INTERVAL '15 minutes'
"""


def lambda_handler(event, context):
    return run_alert_scan(
        event,
        resource=RESOURCE,
        required_table="files",
        queries={
            "MISSED_STUCK": MISSED_STUCK_SQL,
            "EXHAUSTED_RETRIES": EXHAUSTED_SQL,
            "NEVER_STARTED": NEVER_STARTED_SQL,
        },
    )
