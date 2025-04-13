# Update Game Results Lambda

This Lambda function updates profitable game predictions with their actual results.

## How It Works

1. The function fetches game results for the past 10 days from the AWS Athena database.
2. It identifies games in the DynamoDB table that have a `result_id` but don't yet have a result.
3. For each pending game, it checks if there's a matching result in the fetched results.
4. If a match is found, it determines whether the prediction was correct by comparing the prediction with the actual result.
5. The function then updates both the `profitable-games` and `all-predicted-games` DynamoDB tables with the result information.

## Result Calculation Logic

The function uses the following logic to determine if a prediction was correct:

1. Gets the actual scores for the teams from the results table.
2. Applies any handicaps/constraints in the team names (e.g., "Barcelona (+1.5)").
3. Determines the actual outcome (home win, draw, or away win) based on the final scores.
4. Checks if the predicted outcome matches the actual outcome.

## Deployment

To deploy the Lambda function:

1. Make sure you have the AWS CLI installed and configured.
2. Run the deployment script:

```bash
cd lambda_update_results
./deploy_docker_lambda.sh
```

The script will:
- Create an ECR repository if it doesn't exist
- Build and push the Docker image
- Create or update the IAM role with necessary permissions
- Create or update the Lambda function
- Set up a CloudWatch Events Rule to run the function every 12 hours

## Environment Variables

- `AWS_REGION`: AWS region (default: "il-central-1")
- `ATHENA_DATABASE`: Athena database name (default: "winner-db")
- `RESULTS_TABLE_NAME`: Name of the table containing results (default: "results")
- `ALL_PREDICTIONS_TABLE`: DynamoDB table for all predictions (default: "all-predicted-games")
- `PROFITABLE_GAMES_TABLE`: DynamoDB table for profitable predictions (default: "profitable-games")

## Running Locally

To run the function locally for testing:

```python
python lambda_function.py
``` 