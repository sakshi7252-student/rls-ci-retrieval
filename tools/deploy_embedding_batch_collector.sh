#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

AWS_REGION="${AWS_REGION:-eu-west-1}"
FUNCTION_NAME="${FUNCTION_NAME:-rls-ci-retrieval-embedding-batch-collector}"
ECR_REPO="${ECR_REPO:-rls-ci-retrieval-embedding-batch-collector}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"
LATEST_TAG="${LATEST_TAG:-latest}"
ROLE_ARN="${ROLE_ARN:-}"
TIMEOUT="${TIMEOUT:-300}"
MEMORY_SIZE="${MEMORY_SIZE:-1024}"

CALLBACK_URL="${CALLBACK_URL:-}"
OPEN_SEARCH_REGION="${OPEN_SEARCH_REGION:-${AWS_REGION}}"
SECRETS_MANAGER_REGION="${SECRETS_MANAGER_REGION:-${AWS_REGION}}"
OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT:-search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com}"
OPENSEARCH_INDEX="${OPENSEARCH_INDEX:-document-chunks}"
SEMANTIC_OBJECTS_INDEX="${SEMANTIC_OBJECTS_INDEX:-semantic-objects}"
OPENSEARCH_CI_INDEX="${OPENSEARCH_CI_INDEX:-ci-objects}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-amazon.titan-embed-text-v2:0}"
EMBEDDING_BATCH_PAYLOAD_BUCKET="${EMBEDDING_BATCH_PAYLOAD_BUCKET:-}"

if [[ -z "$ROLE_ARN" ]]; then
  echo "ERROR: ROLE_ARN is required (needs bedrock:GetModelInvocationJob, s3 read/write/delete on"
  echo "EMBEDDING_BATCH_PAYLOAD_BUCKET, and OpenSearch access — same role class as document-chunk-worker)"
  exit 1
fi

if [[ -z "$EMBEDDING_BATCH_PAYLOAD_BUCKET" ]]; then
  echo "ERROR: EMBEDDING_BATCH_PAYLOAD_BUCKET is required"
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

echo "Building and pushing image: ${IMAGE_URI}"
if ! docker buildx inspect >/dev/null 2>&1; then
  docker buildx create --use >/dev/null
fi

docker buildx build \
  --progress=plain \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --push \
  -f "$ROOT_DIR/lambdas/worker/embedding-batch-collector/Dockerfile" \
  -t "$IMAGE_URI" \
  -t "$LATEST_IMAGE_URI" \
  "$ROOT_DIR"

if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --image-uri "$IMAGE_URI" \
    --region "$AWS_REGION" >/dev/null
  aws lambda wait function-updated-v2 --function-name "$FUNCTION_NAME" --region "$AWS_REGION"
else
  aws lambda create-function \
    --function-name "$FUNCTION_NAME" \
    --package-type Image \
    --code ImageUri="$IMAGE_URI" \
    --role "$ROLE_ARN" \
    --timeout "$TIMEOUT" \
    --memory-size "$MEMORY_SIZE" \
    --region "$AWS_REGION" >/dev/null
  aws lambda wait function-active-v2 --function-name "$FUNCTION_NAME" --region "$AWS_REGION"
fi

aws lambda update-function-configuration \
  --function-name "$FUNCTION_NAME" \
  --timeout "$TIMEOUT" \
  --memory-size "$MEMORY_SIZE" \
  --environment "Variables={CALLBACK_URL=${CALLBACK_URL},OPENSEARCH_ENDPOINT=${OPENSEARCH_ENDPOINT},OPEN_SEARCH_REGION=${OPEN_SEARCH_REGION},SECRETS_MANAGER_REGION=${SECRETS_MANAGER_REGION},OPENSEARCH_INDEX=${OPENSEARCH_INDEX},SEMANTIC_OBJECTS_INDEX=${SEMANTIC_OBJECTS_INDEX},OPENSEARCH_CI_INDEX=${OPENSEARCH_CI_INDEX},EMBEDDING_MODEL=${EMBEDDING_MODEL},EMBEDDING_BATCH_PAYLOAD_BUCKET=${EMBEDDING_BATCH_PAYLOAD_BUCKET}}" \
  --region "$AWS_REGION" >/dev/null
aws lambda wait function-updated-v2 --function-name "$FUNCTION_NAME" --region "$AWS_REGION"

echo "Deployed ${FUNCTION_NAME}"
echo "Image: ${IMAGE_URI}"

echo
echo "Next: wire EventBridge so Bedrock batch-inference job completions trigger this Lambda,"
echo "then grant it bedrock:GetModelInvocationJob + s3 read/write/delete on the bucket:"
echo
echo "# Primary: event-pattern rule (fires near-instantly on job state change — best-effort delivery)"
echo "aws events put-rule --name ${FUNCTION_NAME}-event --region ${AWS_REGION} \\"
echo "  --event-pattern '{\"source\":[\"aws.bedrock\"],\"detail-type\":[\"Batch Inference Job State Change\"]}'"
echo "aws lambda add-permission --function-name ${FUNCTION_NAME} --statement-id ${FUNCTION_NAME}-event \\"
echo "  --action lambda:InvokeFunction --principal events.amazonaws.com \\"
echo "  --source-arn arn:aws:events:${AWS_REGION}:${ACCOUNT_ID}:rule/${FUNCTION_NAME}-event --region ${AWS_REGION}"
echo "aws events put-targets --rule ${FUNCTION_NAME}-event --region ${AWS_REGION} \\"
echo "  --targets \"Id\"=\"1\",\"Arn\"=\"arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}\""
echo
echo "# Safety net: coarse schedule to catch any best-effort event Bedrock drops"
echo "aws events put-rule --name ${FUNCTION_NAME}-schedule --schedule-expression 'rate(15 minutes)' --region ${AWS_REGION}"
echo "aws lambda add-permission --function-name ${FUNCTION_NAME} --statement-id ${FUNCTION_NAME}-schedule \\"
echo "  --action lambda:InvokeFunction --principal events.amazonaws.com \\"
echo "  --source-arn arn:aws:events:${AWS_REGION}:${ACCOUNT_ID}:rule/${FUNCTION_NAME}-schedule --region ${AWS_REGION}"
echo "aws events put-targets --rule ${FUNCTION_NAME}-schedule --region ${AWS_REGION} \\"
echo "  --targets \"Id\"=\"1\",\"Arn\"=\"arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}\""
