"""
Alert Lambda: SQS queue capacity + worker health watchdog (document-chunk-worker
+ ci-worker pipelines: queue -> DLQ -> Lambda consumer).

"Queue Capacity Alerts: Set up alerts for when the queue remains full for
extended periods. Early warning for high traffic, allowing strategic action
on performance/scalability."

Covers the real queues/workers:
  - rls-ci-retrieval-document-chunk-worker-queue (deployed per-region, see
    DOCUMENT_CHUNK_QUEUE_REGIONS -- matches deploy-chunk-worker.yml's matrix)
    -> DLQ rls-ci-retrieval-document-chunk-worker-dlq
    -> worker Lambda DOCUMENT_CHUNK_WORKER_FUNCTION_NAME (per-region)
  - rls-ci-retrieval-ci-worker-queue (single region)
    -> DLQ rls-ci-retrieval-ci-worker-dlq
    -> worker Lambda CI_WORKER_FUNCTION_NAME

Checks per queue (all via CloudWatch GetMetricStatistics, no native Alarms):
  1. QUEUE_DEPTH_SUSTAINED: ApproximateNumberOfMessagesVisible average
     stayed above QUEUE_DEPTH_THRESHOLD for every period in the last
     QUEUE_SUSTAINED_MINUTES minutes.
  2. QUEUE_DEPTH_RISING_FAST: depth grew by GROWTH_RATIO_THRESHOLD across
     the same window even if it hasn't crossed the absolute threshold yet --
     catches a backlog forming before it becomes a sustained breach.
  3. OLDEST_MESSAGE_AGE_SUSTAINED: ApproximateAgeOfOldestMessage stayed
     above OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS for every period in the
     window.
  4. OLDEST_MESSAGE_AGE_RISING_FAST: age grew by GROWTH_RATIO_THRESHOLD
     across the window.
  5. IN_FLIGHT_MESSAGES_HIGH: ApproximateNumberOfMessagesNotVisible average
     sustained above NOT_VISIBLE_THRESHOLD -- messages received but not yet
     deleted (slow/stuck processing), distinct from a large visible backlog.
  6. IN_FLIGHT_NEAR_QUOTA: NotVisible approaching IN_FLIGHT_QUOTA (SQS's
     per-queue in-flight message limit) -- can stop consumers from
     receiving any more work.
  7. CONSUMER_STOPPED: messages were sent (NumberOfMessagesSent sum > 0)
     but none were received in the same window -- consumer polling has
     stopped while the producer is still active.
  8. MESSAGES_RECEIVED_NOT_DELETED: messages were received but none were
     deleted in the same window -- the worker is picking messages up but
     failing to complete them (crash/timeout/downstream failure).
  9. DLQ_MESSAGES_PRESENT: the DLQ has >0 visible messages -- permanently
     failed messages (exhausted maxReceiveCount retries).
  10. DLQ_MESSAGE_AGE_HIGH: oldest DLQ message age above
      DLQ_AGE_THRESHOLD_SECONDS -- shows how long failures have piled up.
  11. DLQ_NOT_CONFIGURED: the queue has no RedrivePolicy at all -- DLQ
      identity is resolved from the real RedrivePolicy, not a naming-
      convention guess, so a missing/misconfigured DLQ itself alerts
      instead of silently checking the wrong (or a nonexistent) queue.
      DLQ existence is also verified directly (GetMetricStatistics on a
      missing queue returns zero datapoints, not an error, which would
      otherwise look identical to "DLQ is empty").
  12. VISIBILITY_TIMEOUT_UNSAFE: queue VisibilityTimeout is not at least
      VISIBILITY_TIMEOUT_MARGIN_SECONDS greater than the worker Lambda's
      configured timeout -- without that margin a slow invocation can have
      its message redelivered to a second concurrent invocation.
  13. LAMBDA_ERRORS_DETECTED / LAMBDA_THROTTLING_DETECTED: worker Lambda
      Errors/Throttles sum > 0 in the window -- direct cause of backlog/
      failed processing, not visible from queue metrics alone.
  14. LAMBDA_DURATION_NEAR_TIMEOUT: worker Lambda max Duration above
      LAMBDA_DURATION_WARNING_RATIO of its configured timeout.
  15. LAMBDA_CONCURRENCY_NEAR_LIMIT: worker Lambda max ConcurrentExecutions
      above LAMBDA_CONCURRENCY_WARNING_RATIO of the account's unreserved
      concurrency limit -- the worker can't scale further.
  16. QUEUE_CONFIG_CHECK_FAILED: the queue or worker Lambda couldn't be
      resolved at all (e.g. queue deleted/renamed) -- a monitoring system
      must not silently skip a resource it can no longer find.
  17. WATCHDOG_EXECUTION_FAILED: the watchdog itself threw before
      completing its checks -- best-effort SNS alert sent from the
      top-level exception handler so a watchdog crash isn't silent.
  18. MESSAGE_COMPLETION_RATE_LOW: deleted/received ratio for the window
      fell below MESSAGE_COMPLETION_RATE_WARNING (P2) or
      MESSAGE_COMPLETION_RATE_CRITICAL (P1) -- catches partial processing
      failure that #8 misses (e.g. 7,000/10,000 received messages deleted).
  19. CONSUMER_THROUGHPUT_LOW: received/sent ratio for the window fell
      below CONSUMER_THROUGHPUT_WARNING_RATIO -- the consumer is active but
      falling behind the producer, short of the CONSUMER_STOPPED extreme.
      Warning-only (P2): producer/consumer rates aren't strictly 1:1 per window.
  20. LAMBDA_ITERATOR_AGE_HIGH: worker Lambda's IteratorAge (SQS event
      source mapping lag) max exceeded ITERATOR_AGE_THRESHOLD_SECONDS --
      direct consumer-lag signal independent of queue depth.
  21. EVENT_SOURCE_MAPPING_UNHEALTHY: the SQS->Lambda event source mapping
      for this queue/worker is missing or not State=Enabled -- queue and
      Lambda can both look healthy while messages simply never get pulled.
  22. QUEUE_CONFIG_DRIFT: MessageRetentionPeriod / ReceiveMessageWaitTimeSeconds
      / DelaySeconds differ from the EXPECTED_* env var when one is set --
      catches accidental infra changes before they become incidents.
  23. MONITORING_DATA_MISSING: fewer CloudWatch datapoints returned than
      QUEUE_SUSTAINED_MINUTES implies -- "no data" is surfaced explicitly
      instead of silently being treated as "no breach".

NOTE on QUEUE_DEPTH_RISING_FAST / OLDEST_MESSAGE_AGE_RISING_FAST: growth is
only flagged when the ratio AND the absolute increase both clear their
thresholds (QUEUE_DEPTH_MIN_ABSOLUTE_GROWTH / OLDEST_AGE_MIN_ABSOLUTE_GROWTH_SECONDS)
-- a bare ratio check fires just as loudly on 1->2 as on 1000->2000.

NOT covered: baseline-aware "producer stopped" detection (Sent/Received/
Deleted all == 0 while traffic is normally expected) -- this needs an
expected-traffic/schedule baseline this lambda doesn't have; flagged as a
future improvement rather than guessed at here.

Requires these IAM actions on ROLE_ARN beyond the original
cloudwatch:GetMetricStatistics: sqs:GetQueueUrl, sqs:GetQueueAttributes,
lambda:GetFunctionConfiguration, lambda:GetAccountSettings,
lambda:ListEventSourceMappings (for check 21, EVENT_SOURCE_MAPPING_UNHEALTHY).
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
DOCUMENT_CHUNK_WORKER_FUNCTION_NAME = os.environ.get("DOCUMENT_CHUNK_WORKER_FUNCTION_NAME", "rls-ci-retrieval-document-chunk-worker")
# NOTE: the real queue is "...-ci-worker-queue" (no "chunk"); keep the env var
# name for backward compatibility but fix the default to match the real resource.
CI_CHUNK_QUEUE_NAME = os.environ.get("CI_CHUNK_QUEUE_NAME", "rls-ci-retrieval-ci-worker-queue")
CI_CHUNK_QUEUE_REGION = os.environ.get("CI_CHUNK_QUEUE_REGION", "eu-west-1")
CI_WORKER_FUNCTION_NAME = os.environ.get("CI_WORKER_FUNCTION_NAME", "rls-ci-retrieval-ci-worker")

QUEUE_DEPTH_THRESHOLD = float(os.environ.get("QUEUE_DEPTH_THRESHOLD", "1000"))
QUEUE_SUSTAINED_MINUTES = int(os.environ.get("QUEUE_SUSTAINED_MINUTES", "15"))
OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS = float(os.environ.get("OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS", "1800"))
GROWTH_RATIO_THRESHOLD = float(os.environ.get("GROWTH_RATIO_THRESHOLD", "2.0"))

NOT_VISIBLE_THRESHOLD = float(os.environ.get("NOT_VISIBLE_THRESHOLD", "5000"))
IN_FLIGHT_QUOTA = float(os.environ.get("IN_FLIGHT_QUOTA", "120000"))  # SQS standard-queue in-flight limit
IN_FLIGHT_QUOTA_WARNING_RATIO = float(os.environ.get("IN_FLIGHT_QUOTA_WARNING_RATIO", "0.8"))

DLQ_AGE_THRESHOLD_SECONDS = float(os.environ.get("DLQ_AGE_THRESHOLD_SECONDS", "1800"))

VISIBILITY_TIMEOUT_MARGIN_SECONDS = float(os.environ.get("VISIBILITY_TIMEOUT_MARGIN_SECONDS", "60"))

LAMBDA_ERROR_THRESHOLD = float(os.environ.get("LAMBDA_ERROR_THRESHOLD", "0"))
LAMBDA_THROTTLE_THRESHOLD = float(os.environ.get("LAMBDA_THROTTLE_THRESHOLD", "0"))
LAMBDA_DURATION_WARNING_RATIO = float(os.environ.get("LAMBDA_DURATION_WARNING_RATIO", "0.8"))
LAMBDA_CONCURRENCY_WARNING_RATIO = float(os.environ.get("LAMBDA_CONCURRENCY_WARNING_RATIO", "0.8"))

MESSAGE_COMPLETION_RATE_WARNING = float(os.environ.get("MESSAGE_COMPLETION_RATE_WARNING", "0.90"))
MESSAGE_COMPLETION_RATE_CRITICAL = float(os.environ.get("MESSAGE_COMPLETION_RATE_CRITICAL", "0.75"))
CONSUMER_THROUGHPUT_WARNING_RATIO = float(os.environ.get("CONSUMER_THROUGHPUT_WARNING_RATIO", "0.5"))

ITERATOR_AGE_THRESHOLD_SECONDS = float(os.environ.get("ITERATOR_AGE_THRESHOLD_SECONDS", "300"))

# Absolute-increase guard for _rising_fast() -- a bare ratio (e.g. 1 -> 2) is noise.
QUEUE_DEPTH_MIN_ABSOLUTE_GROWTH = float(os.environ.get("QUEUE_DEPTH_MIN_ABSOLUTE_GROWTH", "500"))
OLDEST_AGE_MIN_ABSOLUTE_GROWTH_SECONDS = float(os.environ.get("OLDEST_AGE_MIN_ABSOLUTE_GROWTH_SECONDS", "300"))

# Optional queue-config-drift baselines -- unset (default) skips the check entirely,
# since there's no single correct value across every queue without per-queue config.
_EXPECTED_MESSAGE_RETENTION_SECONDS = os.environ.get("EXPECTED_MESSAGE_RETENTION_SECONDS")
_EXPECTED_RECEIVE_WAIT_TIME_SECONDS = os.environ.get("EXPECTED_RECEIVE_WAIT_TIME_SECONDS")
_EXPECTED_DELAY_SECONDS = os.environ.get("EXPECTED_DELAY_SECONDS")
EXPECTED_MESSAGE_RETENTION_SECONDS = float(_EXPECTED_MESSAGE_RETENTION_SECONDS) if _EXPECTED_MESSAGE_RETENTION_SECONDS else None
EXPECTED_RECEIVE_WAIT_TIME_SECONDS = float(_EXPECTED_RECEIVE_WAIT_TIME_SECONDS) if _EXPECTED_RECEIVE_WAIT_TIME_SECONDS else None
EXPECTED_DELAY_SECONDS = float(_EXPECTED_DELAY_SECONDS) if _EXPECTED_DELAY_SECONDS else None

_cloudwatch_clients: dict[str, "boto3.client"] = {}
_sqs_clients: dict[str, "boto3.client"] = {}
_lambda_clients: dict[str, "boto3.client"] = {}
_account_concurrency_limits: dict[str, float] = {}


def _cloudwatch(region: str):
    if region not in _cloudwatch_clients:
        _cloudwatch_clients[region] = boto3.client("cloudwatch", region_name=region)
    return _cloudwatch_clients[region]


def _sqs(region: str):
    if region not in _sqs_clients:
        _sqs_clients[region] = boto3.client("sqs", region_name=region)
    return _sqs_clients[region]


def _lambda(region: str):
    if region not in _lambda_clients:
        _lambda_clients[region] = boto3.client("lambda", region_name=region)
    return _lambda_clients[region]


def _dlq_name_from_arn(dlq_arn: str) -> str:
    return dlq_arn.rsplit(":", 1)[-1]


def _dlq_name_guess(queue_name: str) -> str:
    """Naming-convention fallback, only used when RedrivePolicy can't be read."""
    if queue_name.endswith("-queue"):
        return queue_name[: -len("-queue")] + "-dlq"
    return queue_name + "-dlq"


def _get_datapoints(region: str, namespace: str, dimensions: list[dict], metric_name: str, statistic: str, period_seconds: int, lookback_seconds: int):
    """Returns datapoint values ordered oldest-to-newest for the lookback window."""
    now = int(time.time())
    response = _cloudwatch(region).get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=dimensions,
        StartTime=now - lookback_seconds,
        EndTime=now,
        Period=period_seconds,
        Statistics=[statistic],
    )
    datapoints = sorted(response.get("Datapoints", []), key=lambda d: d["Timestamp"])
    return [d[statistic] for d in datapoints]


def _sqs_datapoints(region: str, queue_name: str, metric_name: str, statistic: str, period_seconds: int, lookback_seconds: int):
    return _get_datapoints(region, "AWS/SQS", [{"Name": "QueueName", "Value": queue_name}], metric_name, statistic, period_seconds, lookback_seconds)


def _lambda_datapoints(region: str, function_name: str, metric_name: str, statistic: str, period_seconds: int, lookback_seconds: int):
    return _get_datapoints(region, "AWS/Lambda", [{"Name": "FunctionName", "Value": function_name}], metric_name, statistic, period_seconds, lookback_seconds)


def _sustained_breach(values: list[float], threshold: float, min_periods: int) -> bool:
    """All expected periods present and every one of them breaches threshold."""
    if len(values) < min_periods:
        return False
    return all(v > threshold for v in values[-min_periods:])


def _rising_fast(values: list[float], min_absolute_growth: float) -> bool:
    """
    First value is positive and the last value grew by GROWTH_RATIO_THRESHOLD or
    more AND the absolute increase clears min_absolute_growth -- ratio alone
    fires just as loudly on 1->2 as on 1000->2000.
    """
    if len(values) < 2 or values[0] <= 0:
        return False
    grew_enough_absolute = (values[-1] - values[0]) >= min_absolute_growth
    grew_enough_ratio = (values[-1] / values[0]) >= GROWTH_RATIO_THRESHOLD
    return grew_enough_ratio and grew_enough_absolute


def _account_concurrency_limit(region: str) -> float:
    if region not in _account_concurrency_limits:
        settings = _lambda(region).get_account_settings()
        _account_concurrency_limits[region] = float(settings["AccountLimit"]["UnreservedConcurrentExecutions"])
    return _account_concurrency_limits[region]


def _event_source_mapping_for_queue(region: str, function_name: str, queue_arn: str) -> dict | None:
    """Returns the event source mapping entry whose EventSourceArn is this queue, or None."""
    paginator = _lambda(region).get_paginator("list_event_source_mappings")
    for page in paginator.paginate(FunctionName=function_name, EventSourceArn=queue_arn):
        for mapping in page.get("EventSourceMappings", []):
            if mapping.get("EventSourceArn") == queue_arn:
                return mapping
    return None


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
        lookback_seconds = QUEUE_SUSTAINED_MINUTES * 60

        queues = [
            {"queue_name": DOCUMENT_CHUNK_QUEUE_NAME, "region": region, "worker_function_name": DOCUMENT_CHUNK_WORKER_FUNCTION_NAME}
            for region in DOCUMENT_CHUNK_QUEUE_REGIONS
        ]
        queues.append({"queue_name": CI_CHUNK_QUEUE_NAME, "region": CI_CHUNK_QUEUE_REGION, "worker_function_name": CI_WORKER_FUNCTION_NAME})

        for q in queues:
            queue_name, region, worker_function_name = q["queue_name"], q["region"], q["worker_function_name"]

            # 1 + 2. Depth: sustained breach + rising-fast.
            depth_start = time.monotonic()
            depths = _sqs_datapoints(region, queue_name, "ApproximateNumberOfMessagesVisible", "Average", 300, lookback_seconds)
            logger.info(f"{queue_name}@{region} depth check completed in {time.monotonic() - depth_start:.2f}s ({len(depths)} datapoints)")
            if len(depths) < min_periods:
                alert("MONITORING_DATA_MISSING", queue_name, region, {
                    "queue": queue_name, "region": region, "metric": "ApproximateNumberOfMessagesVisible",
                    "expected_datapoints": min_periods, "actual_datapoints": len(depths), "severity": "P2",
                })
            if _sustained_breach(depths, QUEUE_DEPTH_THRESHOLD, min_periods):
                alert("QUEUE_DEPTH_SUSTAINED", queue_name, region, {
                    "queue": queue_name, "region": region, "threshold": QUEUE_DEPTH_THRESHOLD,
                    "sustained_minutes": QUEUE_SUSTAINED_MINUTES, "recent_values": depths[-min_periods:], "severity": "P1",
                })
            if _rising_fast(depths, QUEUE_DEPTH_MIN_ABSOLUTE_GROWTH):
                alert("QUEUE_DEPTH_RISING_FAST", queue_name, region, {
                    "queue": queue_name, "region": region, "growth_ratio_threshold": GROWTH_RATIO_THRESHOLD,
                    "min_absolute_growth": QUEUE_DEPTH_MIN_ABSOLUTE_GROWTH, "recent_values": depths, "severity": "P1",
                })

            # 3 + 4. Oldest message age: sustained breach + rising-fast.
            age_start = time.monotonic()
            ages = _sqs_datapoints(region, queue_name, "ApproximateAgeOfOldestMessage", "Maximum", 300, lookback_seconds)
            logger.info(f"{queue_name}@{region} oldest-message-age check completed in {time.monotonic() - age_start:.2f}s ({len(ages)} datapoints)")
            if len(ages) < min_periods:
                alert("MONITORING_DATA_MISSING", queue_name, region, {
                    "queue": queue_name, "region": region, "metric": "ApproximateAgeOfOldestMessage",
                    "expected_datapoints": min_periods, "actual_datapoints": len(ages), "severity": "P2",
                })
            if _sustained_breach(ages, OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS, min_periods):
                alert("OLDEST_MESSAGE_AGE_SUSTAINED", queue_name, region, {
                    "queue": queue_name, "region": region, "threshold_seconds": OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS,
                    "sustained_minutes": QUEUE_SUSTAINED_MINUTES, "recent_values": ages[-min_periods:], "severity": "P1",
                })
            if _rising_fast(ages, OLDEST_AGE_MIN_ABSOLUTE_GROWTH_SECONDS):
                alert("OLDEST_MESSAGE_AGE_RISING_FAST", queue_name, region, {
                    "queue": queue_name, "region": region, "growth_ratio_threshold": GROWTH_RATIO_THRESHOLD,
                    "min_absolute_growth_seconds": OLDEST_AGE_MIN_ABSOLUTE_GROWTH_SECONDS, "recent_values": ages, "severity": "P1",
                })


            # 5 + 6. In-flight (NotVisible): sustained breach + near quota.
            not_visible = _sqs_datapoints(region, queue_name, "ApproximateNumberOfMessagesNotVisible", "Average", 300, lookback_seconds)
            if _sustained_breach(not_visible, NOT_VISIBLE_THRESHOLD, min_periods):
                alert("IN_FLIGHT_MESSAGES_HIGH", queue_name, region, {
                    "queue": queue_name, "region": region, "threshold": NOT_VISIBLE_THRESHOLD,
                    "recent_values": not_visible[-min_periods:], "severity": "P2",
                })
            if not_visible and not_visible[-1] >= IN_FLIGHT_QUOTA * IN_FLIGHT_QUOTA_WARNING_RATIO:
                alert("IN_FLIGHT_NEAR_QUOTA", queue_name, region, {
                    "queue": queue_name, "region": region, "value": not_visible[-1],
                    "quota": IN_FLIGHT_QUOTA, "warning_ratio": IN_FLIGHT_QUOTA_WARNING_RATIO, "severity": "P0",
                })

            # 7 + 8 + 18 + 19. Flow: sent/received/deleted relationship, plus
            # completion-rate and throughput ratios for partial-failure/lag
            # cases that a bare "== 0" check misses.
            sent = sum(_sqs_datapoints(region, queue_name, "NumberOfMessagesSent", "Sum", 300, lookback_seconds))
            received = sum(_sqs_datapoints(region, queue_name, "NumberOfMessagesReceived", "Sum", 300, lookback_seconds))
            deleted = sum(_sqs_datapoints(region, queue_name, "NumberOfMessagesDeleted", "Sum", 300, lookback_seconds))
            if sent > 0 and received == 0:
                alert("CONSUMER_STOPPED", queue_name, region, {
                    "queue": queue_name, "region": region, "sent": sent, "received": received,
                    "sustained_minutes": QUEUE_SUSTAINED_MINUTES, "severity": "P0",
                })
            elif received > 0 and deleted == 0:
                alert("MESSAGES_RECEIVED_NOT_DELETED", queue_name, region, {
                    "queue": queue_name, "region": region, "received": received, "deleted": deleted,
                    "sustained_minutes": QUEUE_SUSTAINED_MINUTES, "severity": "P1",
                })
            elif received > 0:
                completion_rate = deleted / received
                if completion_rate < MESSAGE_COMPLETION_RATE_CRITICAL:
                    alert("MESSAGE_COMPLETION_RATE_LOW", queue_name, region, {
                        "queue": queue_name, "region": region, "received": received, "deleted": deleted,
                        "completion_rate": round(completion_rate, 3), "threshold": MESSAGE_COMPLETION_RATE_CRITICAL,
                        "sustained_minutes": QUEUE_SUSTAINED_MINUTES, "severity": "P1",
                    })
                elif completion_rate < MESSAGE_COMPLETION_RATE_WARNING:
                    alert("MESSAGE_COMPLETION_RATE_LOW", queue_name, region, {
                        "queue": queue_name, "region": region, "received": received, "deleted": deleted,
                        "completion_rate": round(completion_rate, 3), "threshold": MESSAGE_COMPLETION_RATE_WARNING,
                        "sustained_minutes": QUEUE_SUSTAINED_MINUTES, "severity": "P2",
                    })
            # Warning-only: producer/consumer rates aren't strictly 1:1 per window,
            # so this is a lag signal, not an outage signal (CONSUMER_STOPPED above).
            if sent > 0 and received > 0:
                throughput_ratio = received / sent
                if throughput_ratio < CONSUMER_THROUGHPUT_WARNING_RATIO:
                    alert("CONSUMER_THROUGHPUT_LOW", queue_name, region, {
                        "queue": queue_name, "region": region, "sent": sent, "received": received,
                        "throughput_ratio": round(throughput_ratio, 3), "threshold": CONSUMER_THROUGHPUT_WARNING_RATIO,
                        "sustained_minutes": QUEUE_SUSTAINED_MINUTES, "severity": "P2",
                    })


            # 9 + 10. DLQ -- resolved from the queue's actual RedrivePolicy, not
            # a naming-convention guess, so a misconfigured/missing DLQ itself
            # raises an alert instead of silently checking the wrong queue.
            dlq_name = None
            try:
                queue_url = _sqs(region).get_queue_url(QueueName=queue_name)["QueueUrl"]
                redrive_policy_raw = _sqs(region).get_queue_attributes(
                    QueueUrl=queue_url, AttributeNames=["RedrivePolicy"],
                )["Attributes"].get("RedrivePolicy")
                if redrive_policy_raw:
                    dlq_name = _dlq_name_from_arn(json.loads(redrive_policy_raw)["deadLetterTargetArn"])
                else:
                    alert("DLQ_NOT_CONFIGURED", queue_name, region, {
                        "queue": queue_name, "region": region, "severity": "P1",
                    })
                    dlq_name = _dlq_name_guess(queue_name)
            except Exception as exc:
                alert("QUEUE_CONFIG_CHECK_FAILED", queue_name, region, {
                    "queue": queue_name, "region": region, "worker_function": worker_function_name,
                    "error": str(exc), "severity": "P0",
                })
                dlq_name = _dlq_name_guess(queue_name)

            # GetMetricStatistics on a nonexistent queue returns zero datapoints,
            # not an error -- confirm the DLQ actually exists before trusting
            # "no datapoints" to mean "DLQ is empty".
            try:
                _sqs(region).get_queue_url(QueueName=dlq_name)
            except Exception as exc:
                alert("QUEUE_CONFIG_CHECK_FAILED", dlq_name, region, {
                    "queue": dlq_name, "source_queue": queue_name, "region": region,
                    "error": str(exc), "severity": "P0",
                })
                dlq_name = None

            if dlq_name:
                dlq_depth = _sqs_datapoints(region, dlq_name, "ApproximateNumberOfMessagesVisible", "Maximum", 300, 300)
                if dlq_depth and dlq_depth[-1] > 0:
                    alert("DLQ_MESSAGES_PRESENT", dlq_name, region, {
                        "queue": dlq_name, "source_queue": queue_name, "region": region,
                        "value": dlq_depth[-1], "severity": "P0",
                    })
                dlq_age = _sqs_datapoints(region, dlq_name, "ApproximateAgeOfOldestMessage", "Maximum", 300, 300)
                if dlq_age and dlq_age[-1] > DLQ_AGE_THRESHOLD_SECONDS:
                    alert("DLQ_MESSAGE_AGE_HIGH", dlq_name, region, {
                        "queue": dlq_name, "source_queue": queue_name, "region": region,
                        "value": dlq_age[-1], "threshold_seconds": DLQ_AGE_THRESHOLD_SECONDS, "severity": "P1",
                    })

            # 12, 13, 14, 15, 16, 20, 21, 22. Queue config + worker Lambda health.
            try:
                queue_url = _sqs(region).get_queue_url(QueueName=queue_name)["QueueUrl"]
                queue_attrs = _sqs(region).get_queue_attributes(
                    QueueUrl=queue_url,
                    AttributeNames=[
                        "QueueArn", "VisibilityTimeout", "MessageRetentionPeriod",
                        "ReceiveMessageWaitTimeSeconds", "DelaySeconds",
                    ],
                )["Attributes"]
                queue_arn = queue_attrs["QueueArn"]
                visibility_timeout = float(queue_attrs["VisibilityTimeout"])
                worker_config = _lambda(region).get_function_configuration(FunctionName=worker_function_name)
                worker_timeout = float(worker_config["Timeout"])

                if visibility_timeout < worker_timeout + VISIBILITY_TIMEOUT_MARGIN_SECONDS:
                    alert("VISIBILITY_TIMEOUT_UNSAFE", queue_name, region, {
                        "queue": queue_name, "region": region, "visibility_timeout": visibility_timeout,
                        "worker_function": worker_function_name, "worker_timeout": worker_timeout,
                        "required_margin_seconds": VISIBILITY_TIMEOUT_MARGIN_SECONDS, "severity": "P0",
                    })

                # Config drift -- only checked when an EXPECTED_* baseline is configured.
                _drift_checks = (
                    ("MessageRetentionPeriod", EXPECTED_MESSAGE_RETENTION_SECONDS),
                    ("ReceiveMessageWaitTimeSeconds", EXPECTED_RECEIVE_WAIT_TIME_SECONDS),
                    ("DelaySeconds", EXPECTED_DELAY_SECONDS),
                )
                for attr_name, expected_value in _drift_checks:
                    if expected_value is None:
                        continue
                    actual_value = float(queue_attrs[attr_name])
                    if actual_value != expected_value:
                        alert("QUEUE_CONFIG_DRIFT", queue_name, region, {
                            "queue": queue_name, "region": region, "attribute": attr_name,
                            "expected": expected_value, "actual": actual_value, "severity": "P2",
                        })

                # Event source mapping health -- queue and Lambda can both look
                # healthy while the SQS->Lambda trigger itself is missing/disabled.
                mapping = _event_source_mapping_for_queue(region, worker_function_name, queue_arn)
                if mapping is None:
                    alert("EVENT_SOURCE_MAPPING_UNHEALTHY", queue_name, region, {
                        "queue": queue_name, "region": region, "worker_function": worker_function_name,
                        "reason": "no_mapping_found", "severity": "P0",
                    })
                elif mapping.get("State") != "Enabled":
                    alert("EVENT_SOURCE_MAPPING_UNHEALTHY", queue_name, region, {
                        "queue": queue_name, "region": region, "worker_function": worker_function_name,
                        "reason": "not_enabled", "state": mapping.get("State"),
                        "batch_size": mapping.get("BatchSize"),
                        "maximum_batching_window_seconds": mapping.get("MaximumBatchingWindowInSeconds"),
                        "severity": "P0",
                    })

                # IteratorAge -- direct consumer-lag signal independent of queue depth.
                iterator_ages_ms = _lambda_datapoints(region, worker_function_name, "IteratorAge", "Maximum", 300, lookback_seconds)
                if iterator_ages_ms and max(iterator_ages_ms) >= ITERATOR_AGE_THRESHOLD_SECONDS * 1000:
                    alert("LAMBDA_ITERATOR_AGE_HIGH", queue_name, region, {
                        "worker_function": worker_function_name, "region": region,
                        "max_iterator_age_ms": max(iterator_ages_ms),
                        "threshold_seconds": ITERATOR_AGE_THRESHOLD_SECONDS, "severity": "P1",
                    })

                errors = sum(_lambda_datapoints(region, worker_function_name, "Errors", "Sum", 300, lookback_seconds))
                if errors > LAMBDA_ERROR_THRESHOLD:
                    alert("LAMBDA_ERRORS_DETECTED", queue_name, region, {
                        "worker_function": worker_function_name, "region": region, "errors": errors,
                        "sustained_minutes": QUEUE_SUSTAINED_MINUTES, "severity": "P0",
                    })

                throttles = sum(_lambda_datapoints(region, worker_function_name, "Throttles", "Sum", 300, lookback_seconds))
                if throttles > LAMBDA_THROTTLE_THRESHOLD:
                    alert("LAMBDA_THROTTLING_DETECTED", queue_name, region, {
                        "worker_function": worker_function_name, "region": region, "throttles": throttles,
                        "sustained_minutes": QUEUE_SUSTAINED_MINUTES, "severity": "P0",
                    })

                durations = _lambda_datapoints(region, worker_function_name, "Duration", "Maximum", 300, lookback_seconds)
                duration_warning_ms = worker_timeout * 1000 * LAMBDA_DURATION_WARNING_RATIO
                if durations and max(durations) >= duration_warning_ms:
                    alert("LAMBDA_DURATION_NEAR_TIMEOUT", queue_name, region, {
                        "worker_function": worker_function_name, "region": region, "max_duration_ms": max(durations),
                        "worker_timeout_ms": worker_timeout * 1000, "warning_ratio": LAMBDA_DURATION_WARNING_RATIO, "severity": "P1",
                    })

                concurrency = _lambda_datapoints(region, worker_function_name, "ConcurrentExecutions", "Maximum", 300, lookback_seconds)
                account_limit = _account_concurrency_limit(region)
                if concurrency and max(concurrency) >= account_limit * LAMBDA_CONCURRENCY_WARNING_RATIO:
                    alert("LAMBDA_CONCURRENCY_NEAR_LIMIT", queue_name, region, {
                        "worker_function": worker_function_name, "region": region, "max_concurrency": max(concurrency),
                        "account_limit": account_limit, "warning_ratio": LAMBDA_CONCURRENCY_WARNING_RATIO, "severity": "P1",
                    })
            except Exception as exc:
                alert("QUEUE_CONFIG_CHECK_FAILED", queue_name, region, {
                    "queue": queue_name, "region": region, "worker_function": worker_function_name,
                    "error": str(exc), "severity": "P0",
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
        try:
            send_sns_alert([{"schema": RESOURCE, "error": str(e), "severity": "P0"}], "WATCHDOG_EXECUTION_FAILED", RESOURCE)
        except Exception as sns_exc:
            logger.error(f"Failed to send WATCHDOG_EXECUTION_FAILED alert: {str(sns_exc)}")
        return {"statusCode": 500, "resource": RESOURCE, "error": str(e)}
