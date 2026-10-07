"""
Alert Lambda: SQS queue capacity watchdog (document-chunk-worker + ci-worker
input queues).

"Queue Capacity Alerts: Set up alerts for when the queue remains full for
extended periods. Early warning for high traffic, allowing strategic action
on performance/scalability."

This is SQS, not OpenSearch. Polls CloudWatch AWS/SQS metrics directly for
the two real queues feeding the worker Lambdas (no native CloudWatch Alarms
-- detection logic lives here):
  - rls-ci-retrieval-document-chunk-worker-queue (deployed per-region, see
    DOCUMENT_CHUNK_QUEUE_REGIONS -- matches deploy-chunk-worker.yml's matrix)
  - rls-ci-retrieval-ci-chunk-worker-queue (single region)

Checks per queue:
  1. QUEUE_DEPTH_SUSTAINED: ApproximateNumberOfMessagesVisible average
     stayed above QUEUE_DEPTH_THRESHOLD for every period in the last
     QUEUE_SUSTAINED_MINUTES minutes -- the queue has remained full for an
     extended period, not just a momentary spike.
  2. OLDEST_MESSAGE_AGE_SUSTAINED: ApproximateAgeOfOldestMessage stayed
     above OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS for every period in the same
     window -- messages are sitting unprocessed, i.e. consumers can't keep
     up with produced volume.
"""

import os
import json
import time
import logging
import boto3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.sns_alert import send_sns_alert

logger = logging.getLogger()
logger.setLevel(logging.INFO)

RESOURCE = "sqs-queue-capacity"

DOCUMENT_CHUNK_QUEUE_NAME = os.environ.get("DOCUMENT_CHUNK_QUEUE_NAME", "rls-ci-retrieval-document-chunk-worker-queue")
DOCUMENT_CHUNK_QUEUE_REGIONS = json.loads(os.environ.get("DOCUMENT_CHUNK_QUEUE_REGIONS", '["eu-west-1"]'))
CI_CHUNK_QUEUE_NAME = os.environ.get("CI_CHUNK_QUEUE_NAME", "rls-ci-retrieval-ci-chunk-worker-queue")
CI_CHUNK_QUEUE_REGION = os.environ.get("CI_CHUNK_QUEUE_REGION", "eu-west-1")

QUEUE_DEPTH_THRESHOLD = float(os.environ.get("QUEUE_DEPTH_THRESHOLD", "1000"))
QUEUE_SUSTAINED_MINUTES = int(os.environ.get("QUEUE_SUSTAINED_MINUTES", "15"))
OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS = float(os.environ.get("OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS", "1800"))

_cloudwatch_clients: dict[str, "boto3.client"] = {}


def _cloudwatch(region: str):
    if region not in _cloudwatch_clients:
        _cloudwatch_clients[region] = boto3.client("cloudwatch", region_name=region)
    return _cloudwatch_clients[region]


def _get_datapoints(region: str, queue_name: str, metric_name: str, statistic: str, period_seconds: int, lookback_seconds: int):
    """Returns datapoint values ordered oldest-to-newest for the lookback window."""
    now = int(time.time())
    response = _cloudwatch(region).get_metric_statistics(
        Namespace="AWS/SQS",
        MetricName=metric_name,
        Dimensions=[{"Name": "QueueName", "Value": queue_name}],
        StartTime=now - lookback_seconds,
        EndTime=now,
        Period=period_seconds,
        Statistics=[statistic],
    )
    datapoints = sorted(response.get("Datapoints", []), key=lambda d: d["Timestamp"])
    return [d[statistic] for d in datapoints]


def _sustained_breach(values: list[float], threshold: float, min_periods: int) -> bool:
    """All expected periods present and every one of them breaches threshold."""
    if len(values) < min_periods:
        return False
    return all(v > threshold for v in values[-min_periods:])


def lambda_handler(event, context):
    start_time = time.monotonic()

    try:
        totals: dict[str, int] = {}

        def alert(alert_type: str, queue_name: str, region: str, row: dict):
            row["schema"] = f"{queue_name}@{region}"
            send_sns_alert([row], alert_type, RESOURCE)
            totals[alert_type] = totals.get(alert_type, 0) + 1
            logger.info(f"{alert_type} fired for {queue_name}@{region}: {row}")

        min_periods = max(1, QUEUE_SUSTAINED_MINUTES * 60 // 300)

        queues = [(DOCUMENT_CHUNK_QUEUE_NAME, region) for region in DOCUMENT_CHUNK_QUEUE_REGIONS]
        queues.append((CI_CHUNK_QUEUE_NAME, CI_CHUNK_QUEUE_REGION))

        for queue_name, region in queues:
            depth_start = time.monotonic()
            depths = _get_datapoints(region, queue_name, "ApproximateNumberOfMessagesVisible", "Average", 300, QUEUE_SUSTAINED_MINUTES * 60)
            logger.info(f"{queue_name}@{region} depth check completed in {time.monotonic() - depth_start:.2f}s ({len(depths)} datapoints)")
            if _sustained_breach(depths, QUEUE_DEPTH_THRESHOLD, min_periods):
                alert("QUEUE_DEPTH_SUSTAINED", queue_name, region, {
                    "queue": queue_name,
                    "region": region,
                    "threshold": QUEUE_DEPTH_THRESHOLD,
                    "sustained_minutes": QUEUE_SUSTAINED_MINUTES,
                    "recent_values": depths[-min_periods:],
                })

            age_start = time.monotonic()
            ages = _get_datapoints(region, queue_name, "ApproximateAgeOfOldestMessage", "Maximum", 300, QUEUE_SUSTAINED_MINUTES * 60)
            logger.info(f"{queue_name}@{region} oldest-message-age check completed in {time.monotonic() - age_start:.2f}s ({len(ages)} datapoints)")
            if _sustained_breach(ages, OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS, min_periods):
                alert("OLDEST_MESSAGE_AGE_SUSTAINED", queue_name, region, {
                    "queue": queue_name,
                    "region": region,
                    "threshold_seconds": OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS,
                    "sustained_minutes": QUEUE_SUSTAINED_MINUTES,
                    "recent_values": ages[-min_periods:],
                })

        execution_time = time.monotonic() - start_time
        logger.info(f"{RESOURCE} scan completed in {execution_time:.2f}s, totals={totals}")

        return {
            "statusCode": 200,
            "resource": RESOURCE,
            "queues_checked": len(queues),
            "execution_time_seconds": execution_time,
            **{f"{alert_type.lower()}_found": count for alert_type, count in totals.items()},
        }
    except Exception as e:
        logger.error(f"{RESOURCE} scan failed: {str(e)}")
        return {"statusCode": 500, "resource": RESOURCE, "error": str(e)}
