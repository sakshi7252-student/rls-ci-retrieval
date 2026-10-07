#!/usr/bin/env bash
# Deploy the rls-ci-retrieval-alert-queue-capacity Lambda (zip-based, pure Python).
# Watches SQS (not OpenSearch) -- the document-chunk-worker and ci-worker input
# queues. Needs cloudwatch:GetMetricStatistics (cross-region, one client per
# entry in DOCUMENT_CHUNK_QUEUE_REGIONS) and publishes to SNS. No SQS IAM
# permissions required since CloudWatch metrics are read by QueueName, not
# by resolving/reading the queue itself.
#
# Required env vars:
#   ROLE_ARN, SNS_TOPIC_ARN
#
# Optional overrides:
#   FUNCTION_NAME, AWS_REGION, TIMEOUT, MEMORY_SIZE, ENVIRONMENT,
#   DOCUMENT_CHUNK_QUEUE_NAME, DOCUMENT_CHUNK_QUEUE_REGIONS (JSON list),
#   CI_CHUNK_QUEUE_NAME, CI_CHUNK_QUEUE_REGION,
#   QUEUE_DEPTH_THRESHOLD, QUEUE_SUSTAINED_MINUTES,
#   OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS
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
CI_CHUNK_QUEUE_NAME="${CI_CHUNK_QUEUE_NAME:-rls-ci-retrieval-ci-chunk-worker-queue}"
CI_CHUNK_QUEUE_REGION="${CI_CHUNK_QUEUE_REGION:-eu-west-1}"
QUEUE_DEPTH_THRESHOLD="${QUEUE_DEPTH_THRESHOLD:-1000}"
QUEUE_SUSTAINED_MINUTES="${QUEUE_SUSTAINED_MINUTES:-15}"
OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS="${OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS:-1800}"

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
ENV_VARS=$(python3 - "$DOCUMENT_CHUNK_QUEUE_NAME" "$DOCUMENT_CHUNK_QUEUE_REGIONS" "$CI_CHUNK_QUEUE_NAME" "$CI_CHUNK_QUEUE_REGION" "$SNS_TOPIC_ARN" "$ENVIRONMENT" "$QUEUE_DEPTH_THRESHOLD" "$QUEUE_SUSTAINED_MINUTES" "$OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS" <<'PYEOF'
import json, sys
(doc_queue, doc_regions, ci_queue, ci_region, sns_arn, environment,
 depth_threshold, sustained_minutes, age_threshold) = sys.argv[1:10]
variables = {
    "DOCUMENT_CHUNK_QUEUE_NAME": doc_queue,
    "DOCUMENT_CHUNK_QUEUE_REGIONS": doc_regions,
    "CI_CHUNK_QUEUE_NAME": ci_queue,
    "CI_CHUNK_QUEUE_REGION": ci_region,
    "SNS_TOPIC_ARN": sns_arn,
    "ENVIRONMENT": environment,
    "QUEUE_DEPTH_THRESHOLD": depth_threshold,
    "QUEUE_SUSTAINED_MINUTES": sustained_minutes,
    "OLDEST_MESSAGE_AGE_THRESHOLD_SECONDS": age_threshold,
}
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

echo "Deployed $FUNCTION_NAME to $AWS_REGION"
