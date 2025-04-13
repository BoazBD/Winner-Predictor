#!/bin/bash
set -e

# Configuration
AWS_REGION=${AWS_REGION:-"il-central-1"}
ECR_REPOSITORY_NAME="fetch-results-from-winner-lambda"
LAMBDA_FUNCTION_NAME="fetch_results_from_winner"
LAMBDA_ROLE_NAME="get-winner-results-lambda-role"  # Changed role name
ATHENA_DATABASE=${ATHENA_DATABASE:-"winner-db"}


echo "Starting deployment process..."
echo "Using PROXY_URL: $PROXY_URL" 

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

# 7. Get or create IAM role
echo "Checking if IAM role exists: $LAMBDA_ROLE_NAME"
if ! aws iam get-role --role-name $LAMBDA_ROLE_NAME &> /dev/null; then
    echo "Creating IAM role: $LAMBDA_ROLE_NAME"
    
    # Create trust policy
    cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
    
    # Create the role
    aws iam create-role \
        --role-name $LAMBDA_ROLE_NAME \
        --assume-role-policy-document file://trust-policy.json
    
    # Attach basic Lambda execution policy
    aws iam attach-role-policy \
        --role-name $LAMBDA_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    
    # Attach S3 access policy
    aws iam attach-role-policy \
        --role-name $LAMBDA_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
    
    # Attach Athena access policy
    aws iam attach-role-policy \
        --role-name $LAMBDA_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonAthenaFullAccess
    
    # Allow time for role to propagate
    echo "Waiting for IAM role to propagate..."
    sleep 10
fi

# Get the role ARN
ROLE_ARN=$(aws iam get-role --role-name $LAMBDA_ROLE_NAME --query 'Role.Arn' --output text)
echo "Using IAM role ARN: $ROLE_ARN"

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
        --environment "Variables={ENV=prod,PROXY_URL=$PROXY_URL,CUSTOM_AWS_REGION=$AWS_REGION,ATHENA_DATABASE=$ATHENA_DATABASE}" \
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
        --environment "Variables={ENV=prod,PROXY_URL=$PROXY_URL,CUSTOM_AWS_REGION=$AWS_REGION,ATHENA_DATABASE=$ATHENA_DATABASE}" \
        --region $AWS_REGION
fi

# 9. Set up CloudWatch Events Rule for scheduled execution
echo "Setting up CloudWatch Events Rule..."
RULE_NAME="$LAMBDA_FUNCTION_NAME-schedule"

# Create or update the rule to run twice daily
aws events put-rule \
    --name $RULE_NAME \
    --schedule-expression "rate(12 hours)" \
    --state ENABLED \
    --region $AWS_REGION

# Add permission to Lambda
echo "Adding CloudWatch Events permission to Lambda..."
aws lambda add-permission \
    --function-name $LAMBDA_FUNCTION_NAME \
    --statement-id "AllowExecutionFromCloudWatch" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn $(aws events describe-rule --name $RULE_NAME --region $AWS_REGION --query 'Arn' --output text) \
    --region $AWS_REGION \
    || echo "Permission may already exist, continuing..."

# Set the Lambda function as the target
aws events put-targets \
    --rule $RULE_NAME \
    --targets "Id"="1","Arn"="$(aws lambda get-function --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION --query 'Configuration.FunctionArn' --output text)" \
    --region $AWS_REGION

# Clean up
rm -f trust-policy.json

echo "Deployment completed successfully!"
echo "Lambda function: $LAMBDA_FUNCTION_NAME"
echo "CloudWatch Rule: $RULE_NAME (runs every 12 hours)" 