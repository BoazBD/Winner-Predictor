# Profitable Games Prediction Lambda

This Lambda function runs independently from the website to automatically fetch game data from AWS Athena, make predictions using a machine learning model, and save profitable predictions to AWS DynamoDB.

## Overview

The Lambda function performs the following steps:
1. Fetches game data from the past 7 days from the Athena `api_odds` table
2. Downloads the prediction model from S3
3. Processes the game data to prepare it for prediction
4. Predicts game outcomes and identifies profitable predictions
5. Saves these profitable predictions to a DynamoDB table

## Requirements

- Python 3.9+
- AWS CLI configured with appropriate permissions
- The following Python packages:
  - tensorflow
  - pandas
  - boto3
  - awswrangler

## Local Testing

You can run the tests locally without connecting to AWS:

```bash
python test_lambda_update_games.py
```

This will run unit tests using mocked AWS services and then run a local test simulation.

## Deployment

To deploy the Lambda function to AWS, run:

```bash
# Set model parameters
export MODEL_TYPE=lstm
export EPOCHS=100
export MAX_SEQ_LENGTH=12

# Deploy to AWS
./deploy_lambda.sh
```

The deploy script performs the following actions:
1. Creates a deployment package with all necessary code and dependencies
2. Creates an S3 bucket if it doesn't exist
3. Uploads the deployment package and model to S3
4. Creates or updates the IAM role with necessary permissions
5. Creates or updates the Lambda function
6. Sets up a CloudWatch Events rule to trigger the function every 2 hours

## AWS Resources Created

- **Lambda Function**: `update_profitable_games`
- **IAM Role**: `update_profitable_games_role`
- **S3 Bucket**: `winner-site-data` (if it doesn't exist)
- **DynamoDB Table**: `profitable-games`
- **CloudWatch Rule**: Triggers the Lambda function every 2 hours

## Environment Variables

The Lambda function uses the following environment variables:

- `AWS_REGION`: AWS region (default: 'il-central-1')
- `ATHENA_DATABASE`: Athena database name (default: 'winner-db')
- `DYNAMODB_TABLE`: DynamoDB table name (default: 'profitable-games')
- `S3_BUCKET`: S3 bucket name (default: 'winner-site-data')
- `MODEL_TYPE`: Model type (default: 'lstm')
- `EPOCHS`: Number of epochs used in model (default: 100)
- `MAX_SEQ`: Maximum sequence length (default: 12)
- `THRESHOLD`: Threshold for profitable predictions (default: 0.02)

## DynamoDB Schema

The DynamoDB table uses the following schema:

- **Partition Key**: `id` (string) - Unique game ID
- **Sort Key**: `prediction` (string) - Prediction type (Home Win, Draw, Away Win)

## Monitoring

You can monitor the Lambda function's execution through CloudWatch Logs. Each execution will log information about:
- Games fetched from Athena
- Prediction results
- DynamoDB updates

## Troubleshooting

Common issues:
1. **Lambda timeout**: If the function times out, increase the timeout in the Lambda configuration
2. **Memory issues**: Increase the allocated memory if the function runs out of memory
3. **Missing model**: Ensure the model file exists in S3 at `models/${MODEL_TYPE}_${EPOCHS}_${MAX_SEQ}_v1.h5`
4. **Permissions**: Check that the IAM role has all necessary permissions 