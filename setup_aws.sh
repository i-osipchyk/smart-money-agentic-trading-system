#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# setup_aws.sh — One-time AWS infrastructure setup for a single Lambda.
#
# Creates (all idempotent — safe to re-run):
#   - Shared ECR repository "trading-signals"
#   - Shared IAM execution role "trading-signals-lambda-role"
#   - Lambda function (container image)
#   - EventBridge scheduled rule triggering Lambda on LTF candle interval
#
# To add a new strategy or symbol, create a new stack-*.env file and run
# this script again with different STRATEGY_SLUG / SYMBOL_SLUG values.
# The ECR repo and IAM role are shared and will be reused automatically.
#
# Required env vars:
#   AWS_ACCOUNT_ID        — 12-digit AWS account ID
#   AWS_REGION            — e.g. us-east-1
#   STRATEGY_SLUG         — kebab-case strategy name, e.g. htf-fvg-ltf-bos
#   STRATEGY              — snake_case registry key, e.g. htf_fvg_ltf_bos
#   SYMBOL_SLUG           — short symbol identifier, e.g. btc
#   SYMBOL                — ccxt perpetual futures symbol, e.g. BTC/USDT:USDT
#   HTF_TIMEFRAME         — e.g. 1h
#   LTF_TIMEFRAME         — e.g. 15m
#   LTF_INTERVAL_CRON     — EventBridge cron expression, e.g. cron(0/15 * * * ? *)
#   TELEGRAM_BOT_TOKEN    — Telegram bot token from BotFather
#   TELEGRAM_CHAT_ID      — Telegram chat or user ID
#
# Optional env vars (have defaults):
#   HTF_LIMIT             — default 72
#   LTF_LIMIT             — default 16
#   FVG_OFFSET_SPINUNITS  — default 10
#   MODE                  — default prompt
#   ANTHROPIC_API_KEY     — required only when MODE=agent
#
# Usage:
#   source stack-htf-fvg-ltf-bos-btc.env && ./setup_aws.sh
# ---------------------------------------------------------------------------

: "${AWS_ACCOUNT_ID:?}"
: "${AWS_REGION:?}"
: "${STRATEGY_SLUG:?}"
: "${STRATEGY:?}"
: "${SYMBOL_SLUG:?}"
: "${SYMBOL:?}"
: "${HTF_TIMEFRAME:?}"
: "${LTF_TIMEFRAME:?}"
: "${LTF_INTERVAL_CRON:?}"
: "${TELEGRAM_BOT_TOKEN:?}"
: "${TELEGRAM_CHAT_ID:?}"

HTF_LIMIT="${HTF_LIMIT:-72}"
LTF_LIMIT="${LTF_LIMIT:-16}"
FVG_OFFSET_SPINUNITS="${FVG_OFFSET_SPINUNITS:-10}"
MODE="${MODE:-prompt}"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

ECR_REPO="trading-signals"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
LAMBDA_NAME="trading-signals-${STRATEGY_SLUG}-${SYMBOL_SLUG}"
ROLE_NAME="trading-signals-lambda-role"
ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"
RULE_NAME="trading-signals-${STRATEGY_SLUG}-${SYMBOL_SLUG}-schedule"

echo "============================================================"
echo " Setting up: ${LAMBDA_NAME}"
echo " ECR repo  : ${ECR_REPO}"
echo " IAM role  : ${ROLE_NAME}"
echo " Schedule  : ${LTF_INTERVAL_CRON}"
echo "============================================================"

# ---- 1. Shared ECR repository -------------------------------------------
echo ""
echo "==> [1/6] Creating ECR repository (shared)..."
aws ecr create-repository \
  --region "${AWS_REGION}" \
  --repository-name "${ECR_REPO}" \
  --image-scanning-configuration scanOnPush=true \
  --no-cli-pager 2>&1 \
  | grep -v "RepositoryAlreadyExistsException" || true
echo "    ECR repo ready: ${ECR_URI}"

# ---- 2. Shared IAM role -------------------------------------------------
echo ""
echo "==> [2/6] Creating IAM role (shared)..."

TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

aws iam create-role \
  --role-name "${ROLE_NAME}" \
  --assume-role-policy-document "${TRUST_POLICY}" \
  --no-cli-pager 2>&1 \
  | grep -v "EntityAlreadyExists" || true

aws iam attach-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" \
  2>&1 | grep -v "already attached" || true

ECR_PULL_POLICY='{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability"
      ],
      "Resource": "arn:aws:ecr:'"${AWS_REGION}"':'"${AWS_ACCOUNT_ID}"':repository/'"${ECR_REPO}"'"
    },
    {
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    }
  ]
}'

aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "trading-signals-ecr-pull" \
  --policy-document "${ECR_PULL_POLICY}"

echo "    IAM role ready: ${ROLE_ARN}"
echo "    Waiting 10s for IAM propagation..."
sleep 10

# ---- 3. Build & push initial image --------------------------------------
echo ""
echo "==> [3/6] Building and pushing Docker image..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build --platform linux/amd64 --tag "${ECR_URI}:latest" .
docker push "${ECR_URI}:latest"
echo "    Image pushed: ${ECR_URI}:latest"

# ---- 4. Lambda function -------------------------------------------------
echo ""
echo "==> [4/6] Creating Lambda function: ${LAMBDA_NAME}..."

# Use a temp JSON file — shorthand format breaks on '/' and ':' in SYMBOL values
ENV_JSON_FILE=$(mktemp)
trap 'rm -f "${ENV_JSON_FILE}"' EXIT

cat > "${ENV_JSON_FILE}" <<EOF
{
  "Variables": {
    "STRATEGY":             "${STRATEGY}",
    "SYMBOL":               "${SYMBOL}",
    "HTF_TIMEFRAME":        "${HTF_TIMEFRAME}",
    "LTF_TIMEFRAME":        "${LTF_TIMEFRAME}",
    "HTF_LIMIT":            "${HTF_LIMIT}",
    "LTF_LIMIT":            "${LTF_LIMIT}",
    "FVG_OFFSET_SPINUNITS": "${FVG_OFFSET_SPINUNITS}",
    "MODE":                 "${MODE}",
    "TELEGRAM_BOT_TOKEN":   "${TELEGRAM_BOT_TOKEN}",
    "TELEGRAM_CHAT_ID":     "${TELEGRAM_CHAT_ID}",
    "ANTHROPIC_API_KEY":    "${ANTHROPIC_API_KEY}"
  }
}
EOF

# Check if the function already exists
EXISTING=$(aws lambda get-function \
  --region "${AWS_REGION}" \
  --function-name "${LAMBDA_NAME}" \
  --query "Configuration.FunctionName" \
  --output text 2>/dev/null || echo "")

if [ -n "${EXISTING}" ]; then
  echo "    Function already exists — skipping creation."
else
  aws lambda create-function \
    --region "${AWS_REGION}" \
    --function-name "${LAMBDA_NAME}" \
    --package-type Image \
    --code "ImageUri=${ECR_URI}:latest" \
    --role "${ROLE_ARN}" \
    --timeout 120 \
    --memory-size 512 \
    --environment "file://${ENV_JSON_FILE}" \
    --no-cli-pager

  echo "    Waiting for Lambda to become active..."
  aws lambda wait function-active \
    --region "${AWS_REGION}" \
    --function-name "${LAMBDA_NAME}"
fi
echo "    Lambda ready: ${LAMBDA_NAME}"

# ---- 5. EventBridge rule ------------------------------------------------
echo ""
echo "==> [5/6] Creating EventBridge rule: ${RULE_NAME}..."

RULE_ARN=$(aws events put-rule \
  --region "${AWS_REGION}" \
  --name "${RULE_NAME}" \
  --schedule-expression "${LTF_INTERVAL_CRON}" \
  --state ENABLED \
  --description "Trigger ${LAMBDA_NAME} on LTF candle close" \
  --query RuleArn \
  --output text)
echo "    Rule ARN: ${RULE_ARN}"

# ---- 6. Wire rule → Lambda ----------------------------------------------
echo ""
echo "==> [6/6] Connecting EventBridge rule to Lambda..."

aws lambda add-permission \
  --region "${AWS_REGION}" \
  --function-name "${LAMBDA_NAME}" \
  --statement-id "eventbridge-${RULE_NAME}" \
  --action "lambda:InvokeFunction" \
  --principal "events.amazonaws.com" \
  --source-arn "${RULE_ARN}" \
  --no-cli-pager 2>&1 \
  | grep -v "ResourceConflictException" || true

aws events put-targets \
  --region "${AWS_REGION}" \
  --rule "${RULE_NAME}" \
  --targets "Id=1,Arn=arn:aws:lambda:${AWS_REGION}:${AWS_ACCOUNT_ID}:function:${LAMBDA_NAME}" \
  --no-cli-pager

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " Lambda  : ${LAMBDA_NAME}"
echo " Image   : ${ECR_URI}:latest"
echo " Schedule: ${LTF_INTERVAL_CRON}"
echo ""
echo " Test with:"
echo "   aws lambda invoke \\"
echo "     --function-name ${LAMBDA_NAME} \\"
echo "     --region ${AWS_REGION} \\"
echo "     /tmp/out.json && cat /tmp/out.json"
echo ""
echo " To deploy a code update (updates ALL trading-signals-* functions):"
echo "   AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID} AWS_REGION=${AWS_REGION} ./deploy.sh"
echo "============================================================"
