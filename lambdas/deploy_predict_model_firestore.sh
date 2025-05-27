#!/bin/bash

# Exit on error
set -e

echo "Starting deployment process..."

# Set variables
LAMBDA_NAME="predict-model-firestore"
LAMBDA_DIR="predict_model_firestore"
REGION="il-central-1"
RUNTIME="python3.9"
HANDLER="lambda_function.lambda_handler"
MEMORY_SIZE=1024
TIMEOUT=900  # 15 minutes

# Create a temporary directory for the deployment package
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Copy the Lambda function code to the temporary directory
cp -r "$LAMBDA_DIR"/* "$TEMP_DIR/"

# Install dependencies
echo "Installing dependencies..."
pip install -r "$LAMBDA_DIR/requirements.txt" -t "$TEMP_DIR/"

# Create the deployment package
echo "Creating deployment package..."
cd "$TEMP_DIR"
zip -r ../deployment.zip .
cd - > /dev/null

# Check if the Lambda function exists
if aws lambda get-function --function-name "$LAMBDA_NAME" --region "$REGION" 2>/dev/null; then
    echo "Updating existing Lambda function..."
    aws lambda update-function-code \
        --function-name "$LAMBDA_NAME" \
        --zip-file fileb://deployment.zip \
        --region "$REGION"
else
    echo "Creating new Lambda function..."
    aws lambda create-function \
        --function-name "$LAMBDA_NAME" \
        --runtime "$RUNTIME" \
        --handler "$HANDLER" \
        --memory-size "$MEMORY_SIZE" \
        --timeout "$TIMEOUT" \
        --role "arn:aws:iam::105144480348:role/lambda-predict-model-role" \
        --zip-file fileb://deployment.zip \
        --region "$REGION"
fi

# Clean up
echo "Cleaning up..."
rm -rf "$TEMP_DIR" deployment.zip

echo "Deployment completed successfully!" 