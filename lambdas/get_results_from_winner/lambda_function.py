import sys
import logging
import os
from datetime import datetime
from random import randint

import awswrangler as wr
import boto3
import pandas as pd
import requests
import json
from dotenv import load_dotenv

# Add the root directory to sys.path for local testing
sys.path.append("./")

# Import local modules
from hash_generator import HashGenerator
from winner_config import RESULTS_URL, SID_MAP

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables will be set in the Lambda configuration
ENV = os.environ.get("ENV", "prod")
PROXY_URL = os.environ.get("PROXY_URL", "")

# Validate environment variables
if ENV != "local" and not PROXY_URL:
    logger.error("PROXY_URL environment variable is not set. This is required in non-local environments.")

# Log important configuration
logger.info(f"Environment: {ENV}")
logger.info(f"RESULTS_URL: {RESULTS_URL}")
logger.info(f"PROXY_URL: {PROXY_URL}")

# Set up AWS session with the correct region
boto3.setup_default_session(region_name="il-central-1")


def get_max_date_from_athena() -> str:
    max_previous_date = wr.athena.read_sql_query(
        "SELECT MAX(date) AS max_date FROM results", database="winner-db"
    )["max_date"][0]
    return max_previous_date


def request_results_from_api(start_date: str, end_date: str) -> dict:
    headers = {
        "accept": "application/json",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json",
        "deviceid": f"5e7f72{randint(0, 10)}6a5ff149d4a1{randint(0, 10)}6e3d75b{randint(0, 10)}5a04",
        "origin": "https://www.winner.co.il",
        "referer": f"https://www.winner.co.il/%D7%AA%D7%9{randint(0, 10)}%D7%A6%D7%90%D7%9{randint(0, 10)}%D7%AA/%D7%95%D7%9{randint(0, 10)}%D7%99%D7%A0%D7%A8?date=1{randint(0, 10)}-01-2024",
        "requestid": f"53{randint(0, 10)}baa33f{randint(0, 10)}da45d4a11eff2fe45f{randint(0, 10)}53",
        "sec-ch-ua": '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "useragentdata": '{"devicemodel":"Macintosh","deviceos":"mac os","deviceosversion":"10.15.7","appversion":"1.7.0","apptype":"mobileweb","originId":"3","isAccessibility":false}',
        "x-csrf-token": "null",
    }
    data = {
        "startDate": f"{start_date}T00:00:00+02:00",
        "endDate": f"{end_date}T00:00:00+02:00",
        "sports": [],
        "leagues": [],
    }

    try:
        logger.info(f"Environment: {ENV}")
        
        if ENV == "local":
            # Make direct request in local environment
            response = requests.post(RESULTS_URL, headers=headers, json=data)
            response.raise_for_status()
            return response
        else:
            # Use proxy server in production environment
            logger.info(f"Sending request to proxy: {PROXY_URL}")
            # Simplified proxy format that matches main.py
            response = requests.post(
                PROXY_URL, 
                json={
                    "url": RESULTS_URL, 
                    "headers": headers,
                    "data": data
                }, 
                timeout=10
            )
            response.raise_for_status()
            
            # The proxy returns the raw content; we need to wrap it in a response-like object
            class ProxyResponse:
                def __init__(self, content):
                    self.content = content
                    
                def json(self):
                    return json.loads(self.content)
            
            return ProxyResponse(response.content)
    except Exception as e:
        logger.error(f"An error occurred during the API request: {str(e)}")
        raise


def process_response(response: dict) -> pd.DataFrame:
    results = []
    try:
        for event in response.json()["results"]["events"]:
            if "noScoreLabel" not in event:
                result = {
                    "type": SID_MAP.get(int(event["sportid"])),
                    "date": event["date"],
                    "time": event["time"],
                    "league": event["league"],
                    "teama": event["teamA"],
                    "teamb": event["teamB"],
                    "scorea": event["scoreA"],
                    "scoreb": event["scoreB"],
                    "id": HashGenerator().create_hash(
                        SID_MAP.get(int(event["sportid"])),
                        event["date"],
                        event["league"],
                        event["teamA"],
                        event["teamB"],
                    ),
                }
                results.append(result)
    except Exception as e:
        logger.error(f"Error processing response: {e}")
        logger.error(f"Response data: {response.json() if hasattr(response, 'json') else response}")
    return pd.DataFrame(results)


def fetch_and_combine_results(start_date: str, end_date: str) -> pd.DataFrame:
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    all_results = pd.DataFrame()

    while current_date <= end_date:
        next_date = min(current_date + pd.Timedelta(days=13), end_date)
        logger.info(f"Fetching results from {current_date} to {next_date}")
        response = request_results_from_api(
            current_date.strftime("%Y-%m-%d"), next_date.strftime("%Y-%m-%d")
        )
        if response:
            results_df = process_response(response)
            all_results = pd.concat([all_results, results_df], ignore_index=True)
        current_date = next_date + pd.Timedelta(days=1)

    return all_results


def save_results_to_s3(results_df: pd.DataFrame):
    try:
        wr.s3.to_parquet(
            results_df,
            path="s3://boaz-winner-api/results",
            dataset=True,
            database="winner-db",
            table="results",
            partition_cols=["date", "type"],
            mode="overwrite_partitions",
        )
        logger.info(f"Successfully loaded to S3 {results_df.shape[0]} results")
    except Exception as e:
        logger.error("An error occurred while saving to S3:", str(e))


def get_results():
    """Main function to fetch and save results"""
    logger.info(f"Starting get_results function in {ENV} environment")
    try:
        max_previous_date = get_max_date_from_athena()
        start_date = (pd.to_datetime(max_previous_date) - pd.Timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        end_date = (datetime.today()).strftime("%Y-%m-%d")
        logger.info(f"Fetching results from {start_date} to {end_date}")
        results_df = fetch_and_combine_results(start_date, end_date)
        
        if not results_df.empty:
            save_results_to_s3(results_df)
            logger.info(f"Successfully processed {results_df.shape[0]} results")
            return {
                "statusCode": 200,
                "body": json.dumps({"message": f"Successfully processed {results_df.shape[0]} results"})
            }
        else:
            logger.warning("No results found to save")
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "No results found to save"})
            }
    except Exception as e:
        logger.error(f"Error in get_results Lambda function: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


def lambda_handler(event, context):
    """AWS Lambda handler function"""
    logger.info("Lambda function invoked")
    logger.info(f"Event: {event}")
    
    # Call the main function
    return get_results() 