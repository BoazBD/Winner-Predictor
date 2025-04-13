# Winner Results Lambda

This AWS Lambda function fetches sports results from Winner.co.il and stores them in an S3 bucket formatted as a partitioned Parquet dataset.

## Structure

- `lambda_function.py` - The main Lambda handler and business logic
- `hash_generator.py` - Utility for generating unique IDs for results
- `winner_config.py` - Configuration including URLs and sport type mappings
- `requirements.txt` - Dependencies for the Lambda function

## Deployment

1. Install dependencies to a local package directory:
   ```
   pip install -r requirements.txt -t ./package
   ```

2. Copy the Python files to the package directory:
   ```
   cp *.py ./package/
   ```

3. Create a ZIP file for deployment:
   ```
   cd package
   zip -r ../deployment.zip .
   ```

4. Deploy to AWS Lambda using the AWS CLI:
   ```
   aws lambda update-function-code --function-name get-winner-results --zip-file fileb://deployment.zip
   ```

## Environment Variables

- `ENV` - Environment ("prod" or "local")
- `PROXY_URL` - URL of the proxy service (used in production)

## Execution

The Lambda function is designed to be triggered by an EventBridge (CloudWatch Events) rule, typically on a daily schedule.

When executed, it:
1. Retrieves the latest date from the existing dataset
2. Fetches new results from Winner.co.il starting from the day before the latest date
3. Processes the results and saves them to S3 as a partitioned Parquet dataset 