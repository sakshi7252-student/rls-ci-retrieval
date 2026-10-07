#!/usr/bin/env bash
# Deploy the rls-ci-retrieval-alert-queue-capacity Lambda (zip-based, pure Python).
# Watches SQS + DLQ + worker Lambda health for the document-chunk-worker and
# ci-worker pipelines. Needs cloudwatch:GetMetricStatistics (cross-region, one
# client per entry in DOCUMENT_CHUNK_QUEUE_REGIONS, AWS/SQS + AWS/Lambda
# namespaces), plus sqs:GetQueueUrl, sqs:GetQueueAttributes,
# lambda:GetFunctionConfiguration, lambda:GetAccountSettings,
# lambda:ListEventSourceMappings on ROLE_ARN for the visibility-timeout/
# config-drift/event-source-mapping-health checks.
#
# Also creates two native CloudWatch Alarms protecting the watchdog Lambda
# itself (the one case where a native Alarm beats this lambda's own
# GetMetricStatistics checks, since a crashed/un-invoked watchdog can't
# alert on itself): errors on the watchdog function, and a missing-
# invocation alarm (TreatMissingData=breaching) in case the hourly
# EventBridge rule itself stops firing.
#
# Required env vars:
#   ROLE_ARN, SNS_TOPIC_ARN
#
# Optional overrides:
#   FUNCTION_NAME, AWS_REGION, TIMEOUT, MEMORY_SIZE, ENVIRONMENT,
#   DOCUMENT_CHUNK_QUEUE_NAME, DOCUMENT_CHUNK_QUEUE_REGIONS (JSON list),
#   DOCUMENT_CHUNK_WORKER_FUNCTION_NAME,
#   CI_CHUNK_QUEUE_NAME, CI_CHUNK_QUEUE_REGION, CI_WORKER_FUNCTION_NAME,
#   QUEUE_DEPTH_THRESHOLD, QUEUE_SUSTAINED_MINUTES,
#   OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS, GROWTH_RATIO_THRESHOLD,
#   NOT_VISIBLE_THRESHOLD, IN_FLIGHT_QUOTA, IN_FLIGHT_QUOTA_WARNING_RATIO,
#   DLQ_AGE_THRESHOLD_SECONDS, VISIBILITY_TIMEOUT_MARGIN_SECONDS,
#   LAMBDA_ERROR_THRESHOLD, LAMBDA_THROTTLE_THRESHOLD,
#   LAMBDA_DURATION_WARNING_RATIO, LAMBDA_CONCURRENCY_WARNING_RATIO,
#   MESSAGE_COMPLETION_RATE_WARNING, MESSAGE_COMPLETION_RATE_CRITICAL,
#   CONSUMER_THROUGHPUT_WARNING_RATIO, ITERATOR_AGE_THRESHOLD_SECONDS,
#   QUEUE_DEPTH_MIN_ABSOLUTE_GROWTH, OLDEST_AGE_MIN_ABSOLUTE_GROWTH_SECONDS,
#   EXPECTED_MESSAGE_RETENTION_SECONDS, EXPECTED_RECEIVE_WAIT_TIME_SECONDS,
#   EXPECTED_DELAY_SECONDS (config-drift baselines, unset = check skipped),
#   WATCHDOG_ALARM_SNS_TOPIC_ARN (defaults to SNS_TOPIC_ARN)
#
# Example:
#   ROLE_ARN=arn:aws:iam::<acct>:role/... SNS_TOPIC_ARN=arn:aws:sns:... \
#   DOCUMENT_CHUNK_QUEUE_REGIONS='["eu-west-1","us-east-1"]' \
#   tools/deploy_alert_queue_capacity.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ALERT_DIR="alerts/queue_capacity"

AWS_REGION="${AWS_REGION:-eu-west-1}"
FUNCTION_NAME="${FUNCTION_NAME:-rls-ci-retrieval-alert-queue-capacity}"
RUNTIME="${RUNTIME:-python3.12}"
HANDLER="${HANDLER:-lambda_function.lambda_handler}"
ROLE_ARN="${ROLE_ARN:-}"
TIMEOUT="${TIMEOUT:-120}"
MEMORY_SIZE="${MEMORY_SIZE:-256}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-}"
DOCUMENT_CHUNK_QUEUE_NAME="${DOCUMENT_CHUNK_QUEUE_NAME:-rls-ci-retrieval-document-chunk-worker-queue}"
DOCUMENT_CHUNK_QUEUE_REGIONS="${DOCUMENT_CHUNK_QUEUE_REGIONS:-[\"eu-west-1\"]}"
DOCUMENT_CHUNK_WORKER_FUNCTION_NAME="${DOCUMENT_CHUNK_WORKER_FUNCTION_NAME:-rls-ci-retrieval-document-chunk-worker}"
# Real queue is "...-ci-worker-queue" (no "chunk") -- fixed default to match.
CI_CHUNK_QUEUE_NAME="${CI_CHUNK_QUEUE_NAME:-rls-ci-retrieval-ci-worker-queue}"
CI_CHUNK_QUEUE_REGION="${CI_CHUNK_QUEUE_REGION:-eu-west-1}"
CI_WORKER_FUNCTION_NAME="${CI_WORKER_FUNCTION_NAME:-rls-ci-retrieval-ci-worker}"
QUEUE_DEPTH_THRESHOLD="${QUEUE_DEPTH_THRESHOLD:-1000}"
QUEUE_SUSTAINED_MINUTES="${QUEUE_SUSTAINED_MINUTES:-15}"
OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS="${OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS:-1800}"
GROWTH_RATIO_THRESHOLD="${GROWTH_RATIO_THRESHOLD:-2.0}"
NOT_VISIBLE_THRESHOLD="${NOT_VISIBLE_THRESHOLD:-5000}"
IN_FLIGHT_QUOTA="${IN_FLIGHT_QUOTA:-120000}"
IN_FLIGHT_QUOTA_WARNING_RATIO="${IN_FLIGHT_QUOTA_WARNING_RATIO:-0.8}"
DLQ_AGE_THRESHOLD_SECONDS="${DLQ_AGE_THRESHOLD_SECONDS:-1800}"
VISIBILITY_TIMEOUT_MARGIN_SECONDS="${VISIBILITY_TIMEOUT_MARGIN_SECONDS:-60}"
LAMBDA_ERROR_THRESHOLD="${LAMBDA_ERROR_THRESHOLD:-0}"
LAMBDA_THROTTLE_THRESHOLD="${LAMBDA_THROTTLE_THRESHOLD:-0}"
LAMBDA_DURATION_WARNING_RATIO="${LAMBDA_DURATION_WARNING_RATIO:-0.8}"
LAMBDA_CONCURRENCY_WARNING_RATIO="${LAMBDA_CONCURRENCY_WARNING_RATIO:-0.8}"
MESSAGE_COMPLETION_RATE_WARNING="${MESSAGE_COMPLETION_RATE_WARNING:-0.90}"
MESSAGE_COMPLETION_RATE_CRITICAL="${MESSAGE_COMPLETION_RATE_CRITICAL:-0.75}"
CONSUMER_THROUGHPUT_WARNING_RATIO="${CONSUMER_THROUGHPUT_WARNING_RATIO:-0.5}"
ITERATOR_AGE_THRESHOLD_SECONDS="${ITERATOR_AGE_THRESHOLD_SECONDS:-300}"
QUEUE_DEPTH_MIN_ABSOLUTE_GROWTH="${QUEUE_DEPTH_MIN_ABSOLUTE_GROWTH:-500}"
OLDEST_AGE_MIN_ABSOLUTE_GROWTH_SECONDS="${OLDEST_AGE_MIN_ABSOLUTE_GROWTH_SECONDS:-300}"
# Config-drift baselines -- empty string = check skipped (lambda_function.py
# treats an unset/empty env var as None).
EXPECTED_MESSAGE_RETENTION_SECONDS="${EXPECTED_MESSAGE_RETENTION_SECONDS:-}"
EXPECTED_RECEIVE_WAIT_TIME_SECONDS="${EXPECTED_RECEIVE_WAIT_TIME_SECONDS:-}"
EXPECTED_DELAY_SECONDS="${EXPECTED_DELAY_SECONDS:-}"
WATCHDOG_ALARM_SNS_TOPIC_ARN="${WATCHDOG_ALARM_SNS_TOPIC_ARN:-$SNS_TOPIC_ARN}"

for var in ROLE_ARN SNS_TOPIC_ARN; do
  if [[ -z "${!var}" ]]; then
    echo "ERROR: $var is required"
    exit 1
  fi
done

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "Packaging source..."
cp -r "$ROOT_DIR/alerts/shared" "$BUILD_DIR/shared"
cp "$ROOT_DIR/$ALERT_DIR"/*.py "$BUILD_DIR/"

ZIP_PATH="$BUILD_DIR/../$(basename "$FUNCTION_NAME").zip"
(cd "$BUILD_DIR" && zip -q -r "$ZIP_PATH" .)

# DOCUMENT_CHUNK_QUEUE_REGIONS is a JSON array -- Lambda env vars are flat
# strings, so pass it through as-is; lambda_function.py does json.loads() on it.
# Read straight from the shell environment (all of these are already exported
# as plain vars above) rather than positional argv -- simpler to extend as new
# thresholds get added.
export DOCUMENT_CHUNK_QUEUE_NAME DOCUMENT_CHUNK_QUEUE_REGIONS DOCUMENT_CHUNK_WORKER_FUNCTION_NAME \
  CI_CHUNK_QUEUE_NAME CI_CHUNK_QUEUE_REGION CI_WORKER_FUNCTION_NAME SNS_TOPIC_ARN ENVIRONMENT \
  QUEUE_DEPTH_THRESHOLD QUEUE_SUSTAINED_MINUTES OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS GROWTH_RATIO_THRESHOLD \
  NOT_VISIBLE_THRESHOLD IN_FLIGHT_QUOTA IN_FLIGHT_QUOTA_WARNING_RATIO DLQ_AGE_THRESHOLD_SECONDS \
  VISIBILITY_TIMEOUT_MARGIN_SECONDS LAMBDA_ERROR_THRESHOLD LAMBDA_THROTTLE_THRESHOLD \
  LAMBDA_DURATION_WARNING_RATIO LAMBDA_CONCURRENCY_WARNING_RATIO \
  MESSAGE_COMPLETION_RATE_WARNING MESSAGE_COMPLETION_RATE_CRITICAL CONSUMER_THROUGHPUT_WARNING_RATIO \
  ITERATOR_AGE_THRESHOLD_SECONDS QUEUE_DEPTH_MIN_ABSOLUTE_GROWTH OLDEST_AGE_MIN_ABSOLUTE_GROWTH_SECONDS \
  EXPECTED_MESSAGE_RETENTION_SECONDS EXPECTED_RECEIVE_WAIT_TIME_SECONDS EXPECTED_DELAY_SECONDS
ENV_VARS=$(python3 <<'PYEOF'
import json, os

_PASSTHROUGH_VARS = [
    "DOCUMENT_CHUNK_QUEUE_NAME", "DOCUMENT_CHUNK_QUEUE_REGIONS", "DOCUMENT_CHUNK_WORKER_FUNCTION_NAME",
    "CI_CHUNK_QUEUE_NAME", "CI_CHUNK_QUEUE_REGION", "CI_WORKER_FUNCTION_NAME", "SNS_TOPIC_ARN", "ENVIRONMENT",
    "QUEUE_DEPTH_THRESHOLD", "QUEUE_SUSTAINED_MINUTES", "OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS", "GROWTH_RATIO_THRESHOLD",
    "NOT_VISIBLE_THRESHOLD", "IN_FLIGHT_QUOTA", "IN_FLIGHT_QUOTA_WARNING_RATIO", "DLQ_AGE_THRESHOLD_SECONDS",
    "VISIBILITY_TIMEOUT_MARGIN_SECONDS", "LAMBDA_ERROR_THRESHOLD", "LAMBDA_THROTTLE_THRESHOLD",
    "LAMBDA_DURATION_WARNING_RATIO", "LAMBDA_CONCURRENCY_WARNING_RATIO",
    "MESSAGE_COMPLETION_RATE_WARNING", "MESSAGE_COMPLETION_RATE_CRITICAL", "CONSUMER_THROUGHPUT_WARNING_RATIO",
    "ITERATOR_AGE_THRESHOLD_SECONDS", "QUEUE_DEPTH_MIN_ABSOLUTE_GROWTH", "OLDEST_AGE_MIN_ABSOLUTE_GROWTH_SECONDS",
]
# Config-drift baselines are optional -- only pass through when non-empty so
# lambda_function.py's `os.environ.get(...)` sees them as truly unset, not "".
_OPTIONAL_VARS = [
    "EXPECTED_MESSAGE_RETENTION_SECONDS", "EXPECTED_RECEIVE_WAIT_TIME_SECONDS", "EXPECTED_DELAY_SECONDS",
]

variables = {name: os.environ[name] for name in _PASSTHROUGH_VARS}
variables.update({name: os.environ[name] for name in _OPTIONAL_VARS if os.environ.get(name)})
print(json.dumps({"Variables": variables}))
PYEOF
)

wait_for_lambda_update() {
  aws lambda wait function-updated-v2 --function-name "$FUNCTION_NAME" --region "$AWS_REGION"
}

if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "Updating existing function $FUNCTION_NAME..."
  aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --zip-file "fileb://$ZIP_PATH" \
    --region "$AWS_REGION" >/dev/null
  wait_for_lambda_update
  aws lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --runtime "$RUNTIME" \
    --handler "$HANDLER" \
    --timeout "$TIMEOUT" \
    --memory-size "$MEMORY_SIZE" \
    --environment "$ENV_VARS" \
    --region "$AWS_REGION" >/dev/null
  wait_for_lambda_update
else
  echo "Creating new function $FUNCTION_NAME..."
  aws lambda create-function \
    --function-name "$FUNCTION_NAME" \
    --runtime "$RUNTIME" \
    --handler "$HANDLER" \
    --role "$ROLE_ARN" \
    --timeout "$TIMEOUT" \
    --memory-size "$MEMORY_SIZE" \
    --environment "$ENV_VARS" \
    --zip-file "fileb://$ZIP_PATH" \
    --region "$AWS_REGION" >/dev/null
  wait_for_lambda_update
fi

# Hourly invocation via EventBridge, same pattern as the existing NLP_Error_Notification/Hourly rule.
SCHEDULE_RULE_NAME="${SCHEDULE_RULE_NAME:-${FUNCTION_NAME}-hourly}"
SCHEDULE_EXPRESSION="${SCHEDULE_EXPRESSION:-rate(1 hour)}"
FUNCTION_ARN="$(aws lambda get-function --function-name "$FUNCTION_NAME" --region "$AWS_REGION" --query 'Configuration.FunctionArn' --output text)"
RULE_ARN="$(aws events put-rule \
  --name "$SCHEDULE_RULE_NAME" \
  --schedule-expression "$SCHEDULE_EXPRESSION" \
  --state ENABLED \
  --region "$AWS_REGION" \
  --query 'RuleArn' --output text)"
ADD_PERM_OUTPUT=$(aws lambda add-permission \
  --function-name "$FUNCTION_NAME" \
  --statement-id "$SCHEDULE_RULE_NAME" \
  --action "lambda:InvokeFunction" \
  --principal events.amazonaws.com \
  --source-arn "$RULE_ARN" \
  --region "$AWS_REGION" 2>&1) || {
    [[ "$ADD_PERM_OUTPUT" == *"ResourceConflictException"* ]] || { echo "$ADD_PERM_OUTPUT" >&2; exit 1; }
  }
aws events put-targets \
  --rule "$SCHEDULE_RULE_NAME" \
  --targets "Id=1,Arn=$FUNCTION_ARN" \
  --region "$AWS_REGION" >/dev/null

# This watchdog monitors other pipelines via its own hourly GetMetricStatistics
# checks -- it can't alert on itself if it crashes or stops being invoked, so
# this is the one place a native CloudWatch Alarm earns its keep.
echo "Creating native CloudWatch Alarms for the watchdog itself..."
aws cloudwatch put-metric-alarm \
  --alarm-name "${FUNCTION_NAME}-errors" \
  --namespace "AWS/Lambda" \
  --metric-name "Errors" \
  --dimensions "Name=FunctionName,Value=$FUNCTION_NAME" \
  --statistic Sum \
  --period 3600 \
  --evaluation-periods 1 \
  --threshold 0 \
  --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "$WATCHDOG_ALARM_SNS_TOPIC_ARN" \
  --region "$AWS_REGION" >/dev/null
aws cloudwatch put-metric-alarm \
  --alarm-name "${FUNCTION_NAME}-not-invoked" \
  --namespace "AWS/Lambda" \
  --metric-name "Invocations" \
  --dimensions "Name=FunctionName,Value=$FUNCTION_NAME" \
  --statistic Sum \
  --period 3600 \
  --evaluation-periods 2 \
  --threshold 1 \
  --comparison-operator LessThanThreshold \
  --treat-missing-data breaching \
  --alarm-actions "$WATCHDOG_ALARM_SNS_TOPIC_ARN" \
  --region "$AWS_REGION" >/dev/null

echo "Deployed $FUNCTION_NAME to $AWS_REGION"
