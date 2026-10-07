"""
Alert Lambda: stuck / permanently-failed CI OpenSearch indexing.

The cron job (rls-backend cron/ci-index/recovery.ts, every 2 min) already
retries FAILED rows with attemptCount < 3 and re-dispatches QUEUED/PROCESSING
rows whose lastAttemptAt is older than its 20-minute stuck window.

This Lambda is a watchdog, not a recovery path:
  1. MISSED_STUCK: still QUEUED/PROCESSING well past the cron's own 20-minute
     window (3x margin) — the cron missed it.
  2. EXHAUSTED_RETRIES: attemptCount has hit the cap (3) and status is FAILED,
     so the recovery cron's own WHERE clause permanently excludes it.
  3. NEVER_STARTED: no ci_index_state row exists at all well past the
     backfill cron's 1-minute cadence, meaning the initial dispatch never
     happened for this CI.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.scan_runner import run_alert_scan

logger = logging.getLogger()
logger.setLevel(logging.INFO)

RESOURCE = "ci-indexing"

# 3x the cron's own 20-minute stuck window.
MISSED_STUCK_SQL = """
    SELECT "ciId", "indexName", "indexStatus", "attemptCount", "lastAttemptAt"
    FROM {schema}.ci_index_state
    WHERE "indexStatus" IN ('QUEUED', 'PROCESSING')
      AND "lastAttemptAt" < NOW() - INTERVAL '60 minutes'
      AND "attemptCount" < 3
"""

EXHAUSTED_SQL = """
    SELECT "ciId", "indexName", "indexStatus", "attemptCount", "lastAttemptAt", "lastError"
    FROM {schema}.ci_index_state
    WHERE "indexStatus" = 'FAILED'
      AND "attemptCount" >= 3
"""

# ci-indexing-backfill (every 1 min) dispatches CIs with NO ci_index_state
# row at all (NOT EXISTS, see ciIndexingQueryBuilder({backFill:true}) in
# lib/db-client/ci.ts). A CI still missing a row well past that window means
# backfill itself missed it -- there's no row for the recovery cron to retry.
NEVER_STARTED_SQL = """
    SELECT ci.id AS "ciId", ci."createdAt"
    FROM {schema}.ci ci
    WHERE NOT EXISTS (
        SELECT 1 FROM {schema}.ci_index_state cis
        WHERE cis."ciId" = ci.id AND cis."indexName" = 'ci-objects'
      )
      AND ci."createdAt" < NOW() - INTERVAL '15 minutes'
"""


def lambda_handler(event, context):
    return run_alert_scan(
        event,
        resource=RESOURCE,
        required_table="ci_index_state",
        queries={
            "MISSED_STUCK": MISSED_STUCK_SQL,
            "EXHAUSTED_RETRIES": EXHAUSTED_SQL,
            "NEVER_STARTED": NEVER_STARTED_SQL,
        },
    )
