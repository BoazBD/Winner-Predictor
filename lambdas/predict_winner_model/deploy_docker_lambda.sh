#!/bin/bash
# Build + deploy the winner_model prediction lambda (Athena -> models -> Firestore), scheduled every 3 hours.
set -e
cd "$(dirname "$0")"
AWS_REGION=${AWS_REGION:-"il-central-1"}
ECR_REPOSITORY_NAME="predict-winner-model"
LAMBDA_FUNCTION_NAME="predict_winner_model"
LAMBDA_ROLE_NAME="winner-lambda-role"
THRESHOLD=${THRESHOLD:-"-0.03"}
RULE_NAME="$LAMBDA_FUNCTION_NAME-hourly"   # historical name; the schedule is every 3 hours
ENV="Variables={THRESHOLD=$THRESHOLD,ATHENA_DATABASE=winner-db,CUSTOM_AWS_REGION=$AWS_REGION,ATHENA_OUTPUT=s3://winner-athena-output/winner_model/,GOOGLE_APPLICATION_CREDENTIALS=service-account.json}"

./sync_model_files.sh
aws ecr describe-repositories --region $AWS_REGION --repository-names $ECR_REPOSITORY_NAME &>/dev/null || \
  aws ecr create-repository --repository-name $ECR_REPOSITORY_NAME --region $AWS_REGION >/dev/null
URI=$(aws ecr describe-repositories --region $AWS_REGION --repository-names $ECR_REPOSITORY_NAME --query 'repositories[0].repositoryUri' --output text)
# keep only the 2 newest images so ECR storage stays near zero
aws ecr put-lifecycle-policy --region $AWS_REGION --repository-name $ECR_REPOSITORY_NAME --lifecycle-policy-text \
  '{"rules":[{"rulePriority":1,"description":"keep last 2","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":2},"action":{"type":"expire"}}]}' >/dev/null
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $URI
docker buildx build --platform=linux/amd64 --provenance=false -t $URI:latest --push .
ROLE_ARN=$(aws iam get-role --role-name $LAMBDA_ROLE_NAME --query 'Role.Arn' --output text)
if ! aws lambda get-function --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION &>/dev/null; then
  aws lambda create-function --function-name $LAMBDA_FUNCTION_NAME --package-type Image --code ImageUri=$URI:latest \
    --role $ROLE_ARN --timeout 420 --memory-size 2048 --ephemeral-storage Size=1024 --environment "$ENV" --region $AWS_REGION >/dev/null
else
  aws lambda update-function-code --function-name $LAMBDA_FUNCTION_NAME --image-uri $URI:latest --region $AWS_REGION >/dev/null
  aws lambda wait function-updated --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION
  aws lambda update-function-configuration --function-name $LAMBDA_FUNCTION_NAME --timeout 420 --memory-size 2048 \
    --environment "$ENV" --region $AWS_REGION >/dev/null
fi
aws lambda wait function-active-v2 --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION
# every 3 hours, 5 minutes past (scraper writes at :00/:15/:30/:45)
aws events put-rule --name $RULE_NAME --schedule-expression "cron(5 0/3 * * ? *)" --state ENABLED --region $AWS_REGION >/dev/null
aws lambda add-permission --function-name $LAMBDA_FUNCTION_NAME --statement-id "AllowHourlyRule" --action "lambda:InvokeFunction" \
  --principal events.amazonaws.com --source-arn $(aws events describe-rule --name $RULE_NAME --region $AWS_REGION --query Arn --output text) \
  --region $AWS_REGION &>/dev/null || true
aws events put-targets --rule $RULE_NAME --region $AWS_REGION \
  --targets "Id"="1","Arn"="$(aws lambda get-function --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION --query Configuration.FunctionArn --output text)" >/dev/null
echo "Deployed $LAMBDA_FUNCTION_NAME, scheduled by $RULE_NAME (every 3 hours at :05 UTC)"

# sweep_v1 (sweep.py): same image, its own handler, every 5 minutes. Paper only.
SWEEP_FN="sweep_catch"; SWEEP_RULE="sweep_catch-5min"
SWEEP_ENV="Variables={GOOGLE_APPLICATION_CREDENTIALS=service-account.json,SWEEP_SHARES=500}"
if ! aws lambda get-function --function-name $SWEEP_FN --region $AWS_REGION &>/dev/null; then
  aws lambda create-function --function-name $SWEEP_FN --package-type Image --code ImageUri=$URI:latest \
    --image-config 'Command=["sweep.lambda_handler"]' --role $ROLE_ARN --timeout 240 --memory-size 1024 \
    --environment "$SWEEP_ENV" --region $AWS_REGION >/dev/null
else
  aws lambda update-function-code --function-name $SWEEP_FN --image-uri $URI:latest --region $AWS_REGION >/dev/null
  aws lambda wait function-updated --function-name $SWEEP_FN --region $AWS_REGION
  aws lambda update-function-configuration --function-name $SWEEP_FN --image-config 'Command=["sweep.lambda_handler"]' \
    --timeout 240 --memory-size 1024 --environment "$SWEEP_ENV" --region $AWS_REGION >/dev/null
fi
aws lambda wait function-active-v2 --function-name $SWEEP_FN --region $AWS_REGION
aws events put-rule --name $SWEEP_RULE --schedule-expression "rate(5 minutes)" --state ENABLED --region $AWS_REGION >/dev/null
aws lambda add-permission --function-name $SWEEP_FN --statement-id "AllowSweepRule" --action "lambda:InvokeFunction" \
  --principal events.amazonaws.com --source-arn $(aws events describe-rule --name $SWEEP_RULE --region $AWS_REGION --query Arn --output text) \
  --region $AWS_REGION &>/dev/null || true
aws events put-targets --rule $SWEEP_RULE --region $AWS_REGION \
  --targets "Id"="1","Arn"="$(aws lambda get-function --function-name $SWEEP_FN --region $AWS_REGION --query Configuration.FunctionArn --output text)" >/dev/null
echo "Deployed $SWEEP_FN, scheduled by $SWEEP_RULE (every 5 minutes)"
