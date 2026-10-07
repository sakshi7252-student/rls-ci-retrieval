#!/usr/bin/env bash
# Deploy the rls-ci-retrieval-alert-opensearch-health Lambda (zip-based, pure Python).
# Unlike the other 3 alert lambdas this one has no DB dependency -- it reads
# CloudWatch metrics (needs cloudwatch:GetMetricStatistics + sts:GetCallerIdentity
# on ROLE_ARN) and publishes to SNS. If OPENSEARCH_ENDPOINT is set it also makes
# direct OpenSearch REST calls (needs es:ESHttpGet on the domain's ARN) for the
# per-index checks (CRITICAL_INDICES).
#
# Required env vars:
#   ROLE_ARN, DOMAIN_NAME, SNS_TOPIC_ARN
#
# Optional overrides (all have in-code defaults -- see lambda_function.py):
#   FUNCTION_NAME, AWS_REGION, TIMEOUT, MEMORY_SIZE, ENVIRONMENT,
#   YELLOW_SUSTAINED_MINUTES, JVM_MEMORY_PRESSURE_THRESHOLD, JVM_SUSTAINED_MINUTES,
#   STORAGE_WARNING_GIB, STORAGE_CRITICAL_GIB, STORAGE_EMERGENCY_GIB,
#   EXPECTED_NODE_COUNT, CPU_THRESHOLD, CPU_SUSTAINED_MINUTES,
#   NATIVE_MEMORY_PRESSURE_THRESHOLD, NATIVE_MEMORY_SUSTAINED_MINUTES,
#   THREADPOOL_WRITE_QUEUE_THRESHOLD, THREADPOOL_SEARCH_QUEUE_THRESHOLD,
#   OLD_GEN_JVM_THRESHOLD, OLD_GEN_JVM_SUSTAINED_MINUTES,
#   EBS_LATENCY_THRESHOLD_MS, EBS_IOPS_THRESHOLD, BURST_BALANCE_THRESHOLD,
#   SEARCH_LATENCY_THRESHOLD_MS, SEARCH_LATENCY_SUSTAINED_MINUTES,
#   HTTP_4XX_SPIKE_THRESHOLD, OPENSEARCH_ENDPOINT, OPEN_SEARCH_REGION,
#   CRITICAL_INDICES (comma-separated; check #15/per-index checks are
#   skipped entirely if OPENSEARCH_ENDPOINT is unset).
#
# Example:
#   ROLE_ARN=arn:aws:iam::<acct>:role/... DOMAIN_NAME=rls-dev \
#   SNS_TOPIC_ARN=arn:aws:sns:... tools/deploy_alert_opensearch_health.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ALERT_DIR="alerts/opensearch_health"

AWS_REGION="${AWS_REGION:-eu-west-1}"
FUNCTION_NAME="${FUNCTION_NAME:-rls-ci-retrieval-alert-opensearch-health}"
RUNTIME="${RUNTIME:-python3.12}"
HANDLER="${HANDLER:-lambda_function.lambda_handler}"
ROLE_ARN="${ROLE_ARN:-}"
TIMEOUT="${TIMEOUT:-120}"
MEMORY_SIZE="${MEMORY_SIZE:-256}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
DOMAIN_NAME="${DOMAIN_NAME:-}"
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-}"
YELLOW_SUSTAINED_MINUTES="${YELLOW_SUSTAINED_MINUTES:-3}"
JVM_MEMORY_PRESSURE_THRESHOLD="${JVM_MEMORY_PRESSURE_THRESHOLD:-92}"
JVM_SUSTAINED_MINUTES="${JVM_SUSTAINED_MINUTES:-3}"
STORAGE_WARNING_GIB="${STORAGE_WARNING_GIB:-50}"
STORAGE_CRITICAL_GIB="${STORAGE_CRITICAL_GIB:-20}"
STORAGE_EMERGENCY_GIB="${STORAGE_EMERGENCY_GIB:-5}"
EXPECTED_NODE_COUNT="${EXPECTED_NODE_COUNT:-}"
CPU_THRESHOLD="${CPU_THRESHOLD:-90}"
CPU_SUSTAINED_MINUTES="${CPU_SUSTAINED_MINUTES:-5}"
NATIVE_MEMORY_PRESSURE_THRESHOLD="${NATIVE_MEMORY_PRESSURE_THRESHOLD:-90}"
NATIVE_MEMORY_SUSTAINED_MINUTES="${NATIVE_MEMORY_SUSTAINED_MINUTES:-3}"
THREADPOOL_WRITE_QUEUE_THRESHOLD="${THREADPOOL_WRITE_QUEUE_THRESHOLD:-100}"
THREADPOOL_SEARCH_QUEUE_THRESHOLD="${THREADPOOL_SEARCH_QUEUE_THRESHOLD:-100}"
OLD_GEN_JVM_THRESHOLD="${OLD_GEN_JVM_THRESHOLD:-80}"
OLD_GEN_JVM_SUSTAINED_MINUTES="${OLD_GEN_JVM_SUSTAINED_MINUTES:-3}"
EBS_LATENCY_THRESHOLD_MS="${EBS_LATENCY_THRESHOLD_MS:-50}"
EBS_IOPS_THRESHOLD="${EBS_IOPS_THRESHOLD:-0}"
BURST_BALANCE_THRESHOLD="${BURST_BALANCE_THRESHOLD:-0}"
SEARCH_LATENCY_THRESHOLD_MS="${SEARCH_LATENCY_THRESHOLD_MS:-2000}"
SEARCH_LATENCY_SUSTAINED_MINUTES="${SEARCH_LATENCY_SUSTAINED_MINUTES:-3}"
HTTP_4XX_SPIKE_THRESHOLD="${HTTP_4XX_SPIKE_THRESHOLD:-50}"
OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT:-}"
OPEN_SEARCH_REGION="${OPEN_SEARCH_REGION:-$AWS_REGION}"
CRITICAL_INDICES="${CRITICAL_INDICES:-document-chunks,semantic-objects-v3,ci-objects,qc-document-terms}"

for var in ROLE_ARN DOMAIN_NAME SNS_TOPIC_ARN; do
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

# JSON (not CLI shorthand) -- CRITICAL_INDICES/comma-separated values break the
# Variables={K=V,K=V} shorthand parser, which can't distinguish a value-comma
# from a key-separator-comma.
ENV_JSON_PATH="$BUILD_DIR/../env.json"
ENV_JSON_PATH="$ENV_JSON_PATH" \
DOMAIN_NAME="$DOMAIN_NAME" SNS_TOPIC_ARN="$SNS_TOPIC_ARN" ENVIRONMENT="$ENVIRONMENT" \
YELLOW_SUSTAINED_MINUTES="$YELLOW_SUSTAINED_MINUTES" JVM_MEMORY_PRESSURE_THRESHOLD="$JVM_MEMORY_PRESSURE_THRESHOLD" JVM_SUSTAINED_MINUTES="$JVM_SUSTAINED_MINUTES" \
STORAGE_WARNING_GIB="$STORAGE_WARNING_GIB" STORAGE_CRITICAL_GIB="$STORAGE_CRITICAL_GIB" STORAGE_EMERGENCY_GIB="$STORAGE_EMERGENCY_GIB" \
EXPECTED_NODE_COUNT="$EXPECTED_NODE_COUNT" CPU_THRESHOLD="$CPU_THRESHOLD" CPU_SUSTAINED_MINUTES="$CPU_SUSTAINED_MINUTES" \
NATIVE_MEMORY_PRESSURE_THRESHOLD="$NATIVE_MEMORY_PRESSURE_THRESHOLD" NATIVE_MEMORY_SUSTAINED_MINUTES="$NATIVE_MEMORY_SUSTAINED_MINUTES" \
THREADPOOL_WRITE_QUEUE_THRESHOLD="$THREADPOOL_WRITE_QUEUE_THRESHOLD" THREADPOOL_SEARCH_QUEUE_THRESHOLD="$THREADPOOL_SEARCH_QUEUE_THRESHOLD" \
OLD_GEN_JVM_THRESHOLD="$OLD_GEN_JVM_THRESHOLD" OLD_GEN_JVM_SUSTAINED_MINUTES="$OLD_GEN_JVM_SUSTAINED_MINUTES" \
EBS_LATENCY_THRESHOLD_MS="$EBS_LATENCY_THRESHOLD_MS" EBS_IOPS_THRESHOLD="$EBS_IOPS_THRESHOLD" BURST_BALANCE_THRESHOLD="$BURST_BALANCE_THRESHOLD" \
SEARCH_LATENCY_THRESHOLD_MS="$SEARCH_LATENCY_THRESHOLD_MS" SEARCH_LATENCY_SUSTAINED_MINUTES="$SEARCH_LATENCY_SUSTAINED_MINUTES" HTTP_4XX_SPIKE_THRESHOLD="$HTTP_4XX_SPIKE_THRESHOLD" \
OPENSEARCH_ENDPOINT="$OPENSEARCH_ENDPOINT" OPEN_SEARCH_REGION="$OPEN_SEARCH_REGION" CRITICAL_INDICES="$CRITICAL_INDICES" \
python3 -c '
import json, os
keys = [
    "DOMAIN_NAME", "SNS_TOPIC_ARN", "ENVIRONMENT",
    "YELLOW_SUSTAINED_MINUTES", "JVM_MEMORY_PRESSURE_THRESHOLD", "JVM_SUSTAINED_MINUTES",
    "STORAGE_WARNING_GIB", "STORAGE_CRITICAL_GIB", "STORAGE_EMERGENCY_GIB",
    "EXPECTED_NODE_COUNT", "CPU_THRESHOLD", "CPU_SUSTAINED_MINUTES",
    "NATIVE_MEMORY_PRESSURE_THRESHOLD", "NATIVE_MEMORY_SUSTAINED_MINUTES",
    "THREADPOOL_WRITE_QUEUE_THRESHOLD", "THREADPOOL_SEARCH_QUEUE_THRESHOLD",
    "OLD_GEN_JVM_THRESHOLD", "OLD_GEN_JVM_SUSTAINED_MINUTES",
    "EBS_LATENCY_THRESHOLD_MS", "EBS_IOPS_THRESHOLD", "BURST_BALANCE_THRESHOLD",
    "SEARCH_LATENCY_THRESHOLD_MS", "SEARCH_LATENCY_SUSTAINED_MINUTES", "HTTP_4XX_SPIKE_THRESHOLD",
    "OPENSEARCH_ENDPOINT", "OPEN_SEARCH_REGION", "CRITICAL_INDICES",
]
variables = {k: os.environ.get(k, "") for k in keys}
with open(os.environ["ENV_JSON_PATH"], "w") as f:
    json.dump({"Variables": variables}, f)
'

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
    --environment "file://$ENV_JSON_PATH" \
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
    --environment "file://$ENV_JSON_PATH" \
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

echo "Deployed $FUNCTION_NAME to $AWS_REGION"
