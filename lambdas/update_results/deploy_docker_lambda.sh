#!/bin/bash
set -e

# Configuration
AWS_REGION=${AWS_REGION:-"il-central-1"}
ECR_REPOSITORY_NAME="update-results-lambda"
LAMBDA_FUNCTION_NAME="update_game_results"
LAMBDA_ROLE_NAME="winner-lambda-role"  # Use existing role
ATHENA_DATABASE=${ATHENA_DATABASE:-"winner-db"}
ALL_PREDICTIONS_TABLE=${ALL_PREDICTIONS_TABLE:-"all-predicted-games"}
PROFITABLE_GAMES_TABLE=${PROFITABLE_GAMES_TABLE:-"profitable-games"}
RESULTS_TABLE=${RESULTS_TABLE:-"results"}
ATHENA_OUTPUT_BUCKET=${ATHENA_OUTPUT_BUCKET:-"winner-athena-output"}

echo "Starting deployment process..."

# 1. Create ECR repository if it doesn't exist
echo "Checking if ECR repository exists..."
if ! aws ecr describe-repositories --region $AWS_REGION --repository-names $ECR_REPOSITORY_NAME &> /dev/null; then
    echo "Creating ECR repository: $ECR_REPOSITORY_NAME"
    aws ecr create-repository --repository-name $ECR_REPOSITORY_NAME --region $AWS_REGION
else
    echo "ECR repository already exists."
fi

# 2. Get ECR repository URI
ECR_REPOSITORY_URI=$(aws ecr describe-repositories --region $AWS_REGION --repository-names $ECR_REPOSITORY_NAME --query 'repositories[0].repositoryUri' --output text)
echo "ECR Repository URI: $ECR_REPOSITORY_URI"

# 3. Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPOSITORY_URI

# 4. Build Docker image
echo "Building Docker image..."
docker buildx build --platform=linux/amd64 -t $ECR_REPOSITORY_NAME:latest .

# 5. Tag Docker image
echo "Tagging Docker image..."
docker tag $ECR_REPOSITORY_NAME:latest $ECR_REPOSITORY_URI:latest

# 6. Push Docker image to ECR
echo "Pushing Docker image to ECR..."
docker push $ECR_REPOSITORY_URI:latest

# 7. Get the existing role ARN
echo "Using existing IAM role: $LAMBDA_ROLE_NAME"
ROLE_ARN=$(aws iam get-role --role-name $LAMBDA_ROLE_NAME --query 'Role.Arn' --output text)

# 8. Create or update Lambda function
echo "Creating/updating Lambda function..."
if ! aws lambda get-function --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION &> /dev/null; then
    echo "Creating Lambda function: $LAMBDA_FUNCTION_NAME"
    aws lambda create-function \
        --function-name $LAMBDA_FUNCTION_NAME \
        --package-type Image \
        --code ImageUri=$ECR_REPOSITORY_URI:latest \
        --role $ROLE_ARN \
        --timeout 300 \
        --memory-size 1024 \
        --environment "Variables={CUSTOM_AWS_REGION=$AWS_REGION,ATHENA_DATABASE=$ATHENA_DATABASE,ALL_PREDICTIONS_TABLE=$ALL_PREDICTIONS_TABLE,PROFITABLE_GAMES_TABLE=$PROFITABLE_GAMES_TABLE,RESULTS_TABLE=$RESULTS_TABLE,ATHENA_OUTPUT_BUCKET=$ATHENA_OUTPUT_BUCKET}" \
        --region $AWS_REGION
else
    echo "Updating Lambda function: $LAMBDA_FUNCTION_NAME"
    aws lambda update-function-code \
        --function-name $LAMBDA_FUNCTION_NAME \
        --image-uri $ECR_REPOSITORY_URI:latest \
        --region $AWS_REGION

    # Update configuration
    aws lambda update-function-configuration \
        --function-name $LAMBDA_FUNCTION_NAME \
        --timeout 300 \
        --memory-size 1024 \
        --environment "Variables={CUSTOM_AWS_REGION=$AWS_REGION,ATHENA_DATABASE=$ATHENA_DATABASE,ALL_PREDICTIONS_TABLE=$ALL_PREDICTIONS_TABLE,PROFITABLE_GAMES_TABLE=$PROFITABLE_GAMES_TABLE,RESULTS_TABLE=$RESULTS_TABLE,ATHENA_OUTPUT_BUCKET=$ATHENA_OUTPUT_BUCKET}" \
        --region $AWS_REGION
fi

# 9. Set up CloudWatch Events Rule
echo "Setting up CloudWatch Events Rule..."
RULE_NAME="$LAMBDA_FUNCTION_NAME-schedule"

# Create or update the rule
aws events put-rule \
    --name $RULE_NAME \
    --schedule-expression "rate(12 hours)" \
    --state ENABLED \
    --region $AWS_REGION

# Add permission to Lambda
# Attempt to remove old permission first, ignore errors if it doesn't exist
aws lambda remove-permission \
    --function-name $LAMBDA_FUNCTION_NAME \
    --statement-id "AllowExecutionFromCloudWatch" \
    --region $AWS_REGION 2>/dev/null || true

aws lambda add-permission \
    --function-name $LAMBDA_FUNCTION_NAME \
    --statement-id "AllowExecutionFromCloudWatch" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn $(aws events describe-rule --name $RULE_NAME --region $AWS_REGION --query 'Arn' --output text) \
    --region $AWS_REGION

# Set the Lambda function as the target
aws events put-targets \
    --rule $RULE_NAME \
    --targets "Id"="1","Arn"="$(aws lambda get-function --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION --query 'Configuration.FunctionArn' --output text)" \
    --region $AWS_REGION

echo "Deployment completed successfully!"
echo "Lambda function: $LAMBDA_FUNCTION_NAME"
echo "CloudWatch Rule: $RULE_NAME (runs every 12 hours)" 