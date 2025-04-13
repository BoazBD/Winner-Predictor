# Deploying the Profitable Games Prediction Lambda

This guide explains how to deploy the Profitable Games Prediction Lambda function to AWS using Docker.

## Prerequisites

- AWS CLI installed and configured with appropriate credentials
- Docker installed and running
- Proper IAM permissions to:
  - Create/update IAM roles
  - Create/update Lambda functions
  - Create/update CloudWatch Events rules
  - Create/update ECR repositories
  - Create/update DynamoDB tables
  - Upload to S3 buckets

## Deployment Steps

### 1. Upload the Trained Model to S3

First, upload the trained model to S3:

```bash
# Set environment variables if needed
export AWS_REGION=il-central-1
export S3_BUCKET=winner-site-data
export MODEL_TYPE=lstm
export EPOCHS=100
export MAX_SEQ=12

# Make the script executable
chmod +x upload_model_to_s3.sh

# Run the upload script
./upload_model_to_s3.sh
```

### 2. Deploy the Lambda Function

After uploading the model, deploy the Lambda function:

```bash
# Set environment variables if needed
export AWS_REGION=il-central-1
export S3_BUCKET=winner-site-data
export MODEL_TYPE=lstm
export EPOCHS=100
export MAX_SEQ=12
export THRESHOLD=0.02
export ATHENA_DATABASE=winner-db
export DYNAMODB_TABLE=profitable-games

# Make the script executable
chmod +x deploy_docker_lambda.sh

# Run the deployment script
./deploy_docker_lambda.sh
```

## Deployment Process

The deployment process performs the following steps:

1. Creates an ECR repository if it doesn't exist
2. Builds a Docker image containing the Lambda function code
3. Pushes the Docker image to ECR
4. Creates or updates the Lambda function to use the Docker image
5. Sets up appropriate IAM roles and permissions
6. Creates a CloudWatch Events rule to trigger the Lambda function every 2 hours
7. Creates the DynamoDB table if it doesn't exist

## Monitoring

After deployment, you can monitor the Lambda function's execution:

1. Go to the AWS Lambda console
2. Select the `update_profitable_games` function
3. Click on the "Monitor" tab to see CloudWatch metrics
4. View logs in CloudWatch Logs by clicking on "View logs in CloudWatch"

## Troubleshooting

If you encounter issues:

1. **Docker build fails**: Check Docker installation and permissions
2. **Push to ECR fails**: Verify AWS credentials and permissions
3. **Lambda creation fails**: Check IAM roles and permissions
4. **Lambda execution fails**: Check CloudWatch Logs for error messages
5. **Model not found**: Ensure the model is correctly uploaded to S3 with the expected path

## Updating the Lambda Function

To update the Lambda function code:

1. Make changes to `lambda_function.py`
2. Run the `deploy_docker_lambda.sh` script again

## Environment Variables

You can customize the Lambda function's behavior by setting these environment variables:

- `AWS_REGION`: AWS region (default: 'il-central-1')
- `ATHENA_DATABASE`: Athena database name (default: 'winner-db')
- `DYNAMODB_TABLE`: DynamoDB table name (default: 'profitable-games')
- `S3_BUCKET`: S3 bucket name (default: 'winner-site-data')
- `MODEL_TYPE`: Model type (default: 'lstm')
- `EPOCHS`: Number of epochs used in model (default: 100)
- `MAX_SEQ`: Maximum sequence length (default: 12)
- `THRESHOLD`: Threshold for profitable predictions (default: 0.02) 