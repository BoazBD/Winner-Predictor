import sys

sys.path.append("./")

import logging
from datetime import datetime
from random import randint

import awswrangler as wr
import boto3
import pandas as pd
import requests

from api_request.api_scraper import SID_MAP, create_hash

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
boto3.setup_default_session(region_name="il-central-1")


def get_max_date_from_athena() -> str:
    max_previous_date = wr.athena.read_sql_query(
        "SELECT MAX(date) AS max_date FROM results", database="winner-db"
    )["max_date"][0]
    return max_previous_date


def request_results_from_api() -> dict:
    max_previous_date = get_max_date_from_athena()
    start_date = (pd.to_datetime(max_previous_date) - pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    end_date = datetime.today().strftime("%Y-%m-%d")

    URL = "https://www.winner.co.il/api/v2/publicapi/GetResults"
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
        response = requests.post(URL, headers=headers, json=data)
    except Exception as e:
        logger.error("An error occurred during the API request:", str(e))
    return response


def process_response(response: dict) -> pd.DataFrame:
    results = []
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
                "id": create_hash(
                    SID_MAP.get(int(event["sportid"])),
                    event["date"],
                    event["league"],
                    event["teamA"],
                    event["teamB"],
                ),
            }
            results.append(result)
    return pd.DataFrame(results)


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


def main():
    response = request_results_from_api()
    results_df = process_response(response)
    save_results_to_s3(results_df)


if __name__ == "__main__":
    main()
