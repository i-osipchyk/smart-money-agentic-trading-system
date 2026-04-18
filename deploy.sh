#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# deploy.sh — Build, push, and update all trading-signals Lambda functions.
#
# Builds a fresh Docker image, pushes it to the shared ECR repo, then updates
# every Lambda function whose name starts with "trading-signals-" to use the
# new image. This covers all deployed strategy/symbol combinations at once.
#
# Required env vars:
#   AWS_ACCOUNT_ID  — 12-digit AWS account ID
#   AWS_REGION      — e.g. us-east-1
#
# Usage:
#   AWS_ACCOUNT_ID=123456789012 AWS_REGION=us-east-1 ./deploy.sh
# ---------------------------------------------------------------------------

: "${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID is required}"
: "${AWS_REGION:?AWS_REGION is required}"

ECR_REPO="trading-signals"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
IMAGE_TAG="latest"
FULL_URI="${ECR_URI}:${IMAGE_TAG}"
LAMBDA_PREFIX="trading-signals-"

echo "==> Authenticating Docker to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "==> Building Docker image (linux/amd64)..."
docker build \
  --platform linux/amd64 \
  --tag "${FULL_URI}" \
  .

echo "==> Pushing image to ECR..."
docker push "${FULL_URI}"

echo "==> Discovering Lambda functions matching '${LAMBDA_PREFIX}*'..."
FUNCTION_NAMES=$(
  aws lambda list-functions \
    --region "${AWS_REGION}" \
    --query "Functions[?starts_with(FunctionName, '${LAMBDA_PREFIX}')].FunctionName" \
    --output text
)

if [ -z "${FUNCTION_NAMES}" ]; then
  echo "No Lambda functions found matching '${LAMBDA_PREFIX}*'. Nothing to update."
  echo "Run setup_aws.sh first to create a Lambda function."
  exit 0
fi

echo "Found functions: ${FUNCTION_NAMES}"

for FUNCTION_NAME in ${FUNCTION_NAMES}; do
  echo "==> Updating ${FUNCTION_NAME}..."
  aws lambda update-function-code \
    --region "${AWS_REGION}" \
    --function-name "${FUNCTION_NAME}" \
    --image-uri "${FULL_URI}" \
    --no-cli-pager > /dev/null

  echo "    Waiting for update to complete..."
  aws lambda wait function-updated \
    --region "${AWS_REGION}" \
    --function-name "${FUNCTION_NAME}"

  echo "    ${FUNCTION_NAME} updated."
done

echo ""
echo "Deploy complete. All functions updated to ${FULL_URI}"
