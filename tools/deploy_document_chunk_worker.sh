#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

AWS_REGION="${AWS_REGION:-eu-west-1}"
FUNCTION_NAME="${FUNCTION_NAME:-rls-ci-retrieval-document-chunk-worker}"
ECR_REPO="${ECR_REPO:-rls-ci-retrieval-document-chunk-worker}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"
LATEST_TAG="${LATEST_TAG:-latest}"
ROLE_ARN="${ROLE_ARN:-}"
TIMEOUT="${TIMEOUT:-900}"
MEMORY_SIZE="${MEMORY_SIZE:-10240}"
EPHEMERAL_STORAGE="${EPHEMERAL_STORAGE:-10240}"   # MB — GLiNER model copy needs ~1.5 GB /tmp

CALLBACK_URL="${CALLBACK_URL:-}"
OPEN_SEARCH_REGION="${OPEN_SEARCH_REGION:-${AWS_REGION}}"
SECRETS_MANAGER_REGION="${SECRETS_MANAGER_REGION:-${AWS_REGION}}"
OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT:-search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com}"
OPENSEARCH_INDEX="${OPENSEARCH_INDEX:-document-chunks}"
SEMANTIC_OBJECTS_INDEX="${SEMANTIC_OBJECTS_INDEX:-semantic-objects}"
OPENSEARCH_CI_INDEX="${OPENSEARCH_CI_INDEX:-ci-objects}"
NER_MODEL="${NER_MODEL:-gliner}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-amazon.titan-embed-text-v2:0}"
EMBEDDING_MAX_WORKERS="${EMBEDDING_MAX_WORKERS:-2}"
EMBEDDING_SKIP_TYPES="${EMBEDDING_SKIP_TYPES:-}"
HF_TOKEN="${HF_TOKEN:-}"
EMBEDDING_MODE="${EMBEDDING_MODE:-sync}"
EMBEDDING_BATCH_PAYLOAD_BUCKET="${EMBEDDING_BATCH_PAYLOAD_BUCKET:-}"
EMBEDDING_BATCH_PAYLOAD_PREFIX="${EMBEDDING_BATCH_PAYLOAD_PREFIX:-extractions/embedding-batch}"
EMBEDDING_BATCH_ROLE_ARN="${EMBEDDING_BATCH_ROLE_ARN:-}"

if [[ -z "$ROLE_ARN" ]]; then
  echo "ERROR: ROLE_ARN is required"
  echo "Example: ROLE_ARN=arn:aws:iam::<acct>:role/rls-ci-worker-role tools/deploy_chunk_worker.sh"
  exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
IMAGE_URI="${ECR_URI}:${IMAGE_TAG}"
LATEST_IMAGE_URI="${ECR_URI}:${LATEST_TAG}"

aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" >/dev/null

aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

wait_for_lambda_update() {
  aws lambda wait function-updated-v2 \
    --function-name "$FUNCTION_NAME" \
    --region "$AWS_REGION"
}

retry_lambda_update() {
  local attempt=1
  local max_attempts=6
  while true; do
    if "$@"; then
      return 0
    fi
    local exit_code=$?
    if [[ "$attempt" -ge "$max_attempts" ]]; then
      echo "ERROR: command failed after ${attempt} attempts: $*"
      return "$exit_code"
    fi
    echo "Lambda is busy (attempt ${attempt}/${max_attempts}). Waiting and retrying..."
    wait_for_lambda_update || true
    attempt=$((attempt + 1))
  done
}

echo "Building and pushing image: ${IMAGE_URI}"
if ! docker buildx inspect >/dev/null 2>&1; then
  docker buildx create --use >/dev/null
fi

CACHE_IMAGE_URI="${ECR_URI}:buildcache"

docker buildx build \
  --progress=plain \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --push \
  --cache-from "type=registry,ref=${CACHE_IMAGE_URI}" \
  --cache-to   "type=registry,ref=${CACHE_IMAGE_URI},mode=max,image-manifest=true,oci-mediatypes=true" \
  -f "$ROOT_DIR/lambdas/worker/document-chunk-worker/Dockerfile" \
  -t "$IMAGE_URI" \
  -t "$LATEST_IMAGE_URI" \
  "$ROOT_DIR"

if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  retry_lambda_update aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --image-uri "$IMAGE_URI" \
    --region "$AWS_REGION" >/dev/null
else
  aws lambda create-function \
    --function-name "$FUNCTION_NAME" \
    --package-type Image \
    --code ImageUri="$IMAGE_URI" \
    --role "$ROLE_ARN" \
    --timeout "$TIMEOUT" \
    --memory-size "$MEMORY_SIZE" \
    --ephemeral-storage "{\"Size\": ${EPHEMERAL_STORAGE}}" \
    --region "$AWS_REGION" >/dev/null
fi

wait_for_lambda_update

retry_lambda_update aws lambda update-function-configuration \
  --function-name "$FUNCTION_NAME" \
  --timeout "$TIMEOUT" \
  --memory-size "$MEMORY_SIZE" \
  --ephemeral-storage "{\"Size\": ${EPHEMERAL_STORAGE}}" \
  --environment "Variables={CALLBACK_URL=${CALLBACK_URL},OPENSEARCH_ENDPOINT=${OPENSEARCH_ENDPOINT},OPEN_SEARCH_REGION=${OPEN_SEARCH_REGION},SECRETS_MANAGER_REGION=${SECRETS_MANAGER_REGION},OPENSEARCH_INDEX=${OPENSEARCH_INDEX},SEMANTIC_OBJECTS_INDEX=${SEMANTIC_OBJECTS_INDEX},OPENSEARCH_CI_INDEX=${OPENSEARCH_CI_INDEX},NER_MODEL=${NER_MODEL},EMBEDDING_MODEL=${EMBEDDING_MODEL},EMBEDDING_MAX_WORKERS=${EMBEDDING_MAX_WORKERS},EMBEDDING_SKIP_TYPES=${EMBEDDING_SKIP_TYPES},HF_HOME=/tmp/hf_cache,EMBEDDING_MODE=${EMBEDDING_MODE},EMBEDDING_BATCH_PAYLOAD_BUCKET=${EMBEDDING_BATCH_PAYLOAD_BUCKET},EMBEDDING_BATCH_PAYLOAD_PREFIX=${EMBEDDING_BATCH_PAYLOAD_PREFIX},EMBEDDING_BATCH_ROLE_ARN=${EMBEDDING_BATCH_ROLE_ARN}${HF_TOKEN:+,HF_TOKEN=${HF_TOKEN}}}" \
  --region "$AWS_REGION" >/dev/null

wait_for_lambda_update

echo "Deployed ${FUNCTION_NAME}"
echo "Image: ${IMAGE_URI}"

echo
if [[ "$EMBEDDING_MODE" == "batch" ]]; then
  echo "Next: connect SQS queue to Lambda (EMBEDDING_MODE=batch — size per invocation, tune 300-500)"
  echo "aws lambda create-event-source-mapping \\
  --function-name ${FUNCTION_NAME} \\
  --event-source-arn <SQS_QUEUE_ARN> \\
  --batch-size 300 \\
  --maximum-batching-window-in-seconds 30 \\
  --function-response-types ReportBatchItemFailures \\
  --scaling-config MaximumConcurrency=50 \\
  --region ${AWS_REGION}"
else
  echo "Next: connect SQS queue to Lambda"
  echo "aws lambda create-event-source-mapping \\
  --function-name ${FUNCTION_NAME} \\
  --event-source-arn <SQS_QUEUE_ARN> \\
  --batch-size 1 \\
  --function-response-types ReportBatchItemFailures \\
  --scaling-config MaximumConcurrency=50 \\
  --region ${AWS_REGION}"
fi
