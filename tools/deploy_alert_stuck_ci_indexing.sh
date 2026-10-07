#!/usr/bin/env bash
# Deploy the rls-ci-retrieval-alert-stuck-ci-indexing Lambda (zip-based, pure Python).
#
# Required env vars:
#   ROLE_ARN, DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, SNS_TOPIC_ARN
#
# Optional overrides:
#   FUNCTION_NAME, AWS_REGION, TIMEOUT, MEMORY_SIZE, ENVIRONMENT
#
# Example:
#   ROLE_ARN=arn:aws:iam::<acct>:role/... DB_HOST=... DB_NAME=... DB_USER=... \
#   DB_PASSWORD=... SNS_TOPIC_ARN=arn:aws:sns:... \
#   tools/deploy_alert_stuck_ci_indexing.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ALERT_DIR="alerts/stuck_ci_indexing"

AWS_REGION="${AWS_REGION:-eu-west-1}"
FUNCTION_NAME="${FUNCTION_NAME:-rls-ci-retrieval-alert-stuck-ci-indexing}"
RUNTIME="${RUNTIME:-python3.12}"
HANDLER="${HANDLER:-lambda_function.lambda_handler}"
ROLE_ARN="${ROLE_ARN:-}"
TIMEOUT="${TIMEOUT:-300}"
MEMORY_SIZE="${MEMORY_SIZE:-256}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
DB_HOST="${DB_HOST:-}"
DB_NAME="${DB_NAME:-}"
DB_USER="${DB_USER:-}"
DB_PASSWORD="${DB_PASSWORD:-}"
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-}"

for var in ROLE_ARN DB_HOST DB_NAME DB_USER DB_PASSWORD SNS_TOPIC_ARN; do
  if [[ -z "${!var}" ]]; then
    echo "ERROR: $var is required"
    exit 1
  fi
done

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "Installing dependencies..."
pip install --quiet --platform manylinux2014_x86_64 --target "$BUILD_DIR" \
  --implementation cp --python-version 3.12 --only-binary=:all: \
  -r "$ROOT_DIR/alerts/requirements.txt"

echo "Packaging source..."
cp -r "$ROOT_DIR/alerts/shared" "$BUILD_DIR/shared"
cp "$ROOT_DIR/$ALERT_DIR"/*.py "$BUILD_DIR/"

ZIP_PATH="$BUILD_DIR/../$(basename "$FUNCTION_NAME").zip"
(cd "$BUILD_DIR" && zip -q -r "$ZIP_PATH" .)

ENV_VARS="Variables={DB_HOST=$DB_HOST,DB_NAME=$DB_NAME,DB_USER=$DB_USER,DB_PASSWORD=$DB_PASSWORD,SNS_TOPIC_ARN=$SNS_TOPIC_ARN,ENVIRONMENT=$ENVIRONMENT}"

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
