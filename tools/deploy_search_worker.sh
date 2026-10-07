#!/usr/bin/env bash
# Deploy the rls-search-worker Lambda (search pipeline worker).
#
# Required env vars:
#   ROLE_ARN   — IAM role ARN for the Lambda function
#
# Optional overrides:
#   FUNCTION_NAME, ECR_REPO, AWS_REGION, TIMEOUT, MEMORY_SIZE,
#   CALLBACK_URL, OPEN_SEARCH_REGION (defaults to AWS_REGION),
#   SECRETS_MANAGER_REGION (defaults to AWS_REGION),
#   OPENSEARCH_ENDPOINT, OPENSEARCH_INDEX, SEMANTIC_OBJECTS_INDEX,
#   OPENSEARCH_CI_INDEX, VERIFIER_MODEL, RERANKER_CE_MODEL
#
# Example:
#   ROLE_ARN=arn:aws:iam::064051750322:role/service-role/rls-llm-cim-annotation-role-wgwvt366 \
#   tools/deploy_search_worker.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

AWS_REGION="${AWS_REGION:-eu-west-1}"
FUNCTION_NAME="${FUNCTION_NAME:-rls-ci-retrieval-search-worker}"
ECR_REPO="${ECR_REPO:-rls-ci-retrieval-search-worker}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"
LATEST_TAG="${LATEST_TAG:-latest}"
ROLE_ARN="${ROLE_ARN:-}"
# 800s: near AWS Lambda's 900s hard max, leaving margin below the orchestrator's
# own 900s timeout for CI lookups / response formatting around the worker invoke.
TIMEOUT="${TIMEOUT:-800}"
MEMORY_SIZE="${MEMORY_SIZE:-10240}"

CALLBACK_URL="${CALLBACK_URL:-}"
OPEN_SEARCH_REGION="${OPEN_SEARCH_REGION:-${AWS_REGION}}"
SECRETS_MANAGER_REGION="${SECRETS_MANAGER_REGION:-${AWS_REGION}}"
OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT:-search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com}"
OPENSEARCH_INDEX="${OPENSEARCH_INDEX:-document-chunks}"
OPENSEARCH_MAXSIZE="${OPENSEARCH_MAXSIZE:-256}"
SEMANTIC_OBJECTS_INDEX="${SEMANTIC_OBJECTS_INDEX:-semantic-objects}"
OPENSEARCH_CI_INDEX="${OPENSEARCH_CI_INDEX:-ci-objects}"
OPENSEARCH_TIMEOUT="${OPENSEARCH_TIMEOUT:-30}"
RETRIEVER_WORKERS="${RETRIEVER_WORKERS:-4}"
SEARCH_CI_WORKERS="${SEARCH_CI_WORKERS:-5}"
VERIFIER_MODEL="${VERIFIER_MODEL:-eu.anthropic.claude-haiku-4-5-20251001-v1:0}"
RERANKER_CE_MODEL="${RERANKER_CE_MODEL:-BAAI/bge-reranker-base}"
SEARCH_RESULTS_DEBUG_BUCKET="${SEARCH_RESULTS_DEBUG_BUCKET:-rls-file-bucket-eu}"
RESULTS_DEBUG_PREFIX="${RESULTS_DEBUG_PREFIX:-rls-ci-retrieval-search-results/debug}"
MAX_VERIFY_WORKERS="${MAX_VERIFY_WORKERS:-1}"

if [[ -z "$ROLE_ARN" ]]; then
  echo "ERROR: ROLE_ARN is required"
  echo "Example: ROLE_ARN=arn:aws:iam::<acct>:role/... tools/deploy_search_worker.sh"
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
  -f "$ROOT_DIR/lambdas/worker/search-worker/Dockerfile" \
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
    --region "$AWS_REGION" >/dev/null
fi

wait_for_lambda_update

retry_lambda_update aws lambda update-function-configuration \
  --function-name "$FUNCTION_NAME" \
  --timeout "$TIMEOUT" \
  --memory-size "$MEMORY_SIZE" \
  --environment "Variables={CALLBACK_URL=${CALLBACK_URL},OPENSEARCH_ENDPOINT=${OPENSEARCH_ENDPOINT},OPEN_SEARCH_REGION=${OPEN_SEARCH_REGION},SECRETS_MANAGER_REGION=${SECRETS_MANAGER_REGION},OPENSEARCH_INDEX=${OPENSEARCH_INDEX},OPENSEARCH_MAXSIZE=${OPENSEARCH_MAXSIZE},SEMANTIC_OBJECTS_INDEX=${SEMANTIC_OBJECTS_INDEX},OPENSEARCH_CI_INDEX=${OPENSEARCH_CI_INDEX},OPENSEARCH_TIMEOUT=${OPENSEARCH_TIMEOUT},RETRIEVER_WORKERS=${RETRIEVER_WORKERS},SEARCH_CI_WORKERS=${SEARCH_CI_WORKERS},VERIFIER_MODEL=${VERIFIER_MODEL},RERANKER_CE_MODEL=${RERANKER_CE_MODEL},SEARCH_RESULTS_DEBUG_BUCKET=${SEARCH_RESULTS_DEBUG_BUCKET},RESULTS_DEBUG_PREFIX=${RESULTS_DEBUG_PREFIX},MAX_VERIFY_WORKERS=${MAX_VERIFY_WORKERS},HF_HOME=/tmp/hf_cache}" \
  --region "$AWS_REGION" >/dev/null

wait_for_lambda_update

FUNCTION_ARN="$(aws lambda get-function-configuration \
  --function-name "$FUNCTION_NAME" --region "$AWS_REGION" \
  --query FunctionArn --output text)"

echo ""
echo "Deployed ${FUNCTION_NAME}"
echo "Image:   ${IMAGE_URI}"
echo "ARN:     ${FUNCTION_ARN}"
echo ""
echo "Pass this ARN to the search orchestrator:"
echo "  WORKER_LAMBDA_ARN=${FUNCTION_ARN} tools/deploy_search_orchestrator.sh"
