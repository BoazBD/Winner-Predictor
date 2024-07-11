import datetime
import json
import logging
import os
from typing import Any

import awswrangler as wr
import boto3
import pandas as pd
import pytz
import requests

from api_request.hash import HashGenerator
from api_request.winner_config import (
    API_URL,
    HASH_CHECKSUM_URL,
    RECORDING_BETS,
    SID_MAP,
    headers,
)
from Bet import Bet

# Configuration
ENV = os.environ.get("ENV", "local")
PROXY_URL = os.environ.get("PROXY_URL")
AWS_REGION = os.environ.get("AWS_REGION", "il-central-1")

boto3.setup_default_session(region_name=AWS_REGION)

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def remove_bidirectional_control_chars(s: str) -> str:
    """Remove bidirectional control characters from a string."""
    bidi_chars = [
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    ]
    return "".join(c for c in s if c not in bidi_chars)


def fetch_lineChecksum(url: str) -> str:
    """Fetch the lineChecksum from the API using the proxy server."""
    try:
        response = requests.post(
            PROXY_URL,
            json={"url": url, "headers": headers},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["lineChecksum"]
    except requests.RequestException as e:
        logger.error("Failed to fetch lineChecksum from the API: %s", e)
        raise


def fetch_data(url: str, lineChecksum: str = "") -> dict:
    """Fetch data from the API."""
    # complete_url = f"{url}?lineChecksum={lineChecksum}"
    try:
        response = requests.post(
            PROXY_URL, json={"url": url, "headers": headers}, timeout=10
        )
        return response.json()
    except requests.RequestException as e:
        logger.error("Failed to fetch data from the API: %s", e)
        raise


def process_data(data: dict[str, Any]) -> pd.DataFrame:
    """Process data containing all the odds of all games, and return a list of bets.
    market is a segment of the data containing the odds of a single game.
    """
    markets = data["markets"]
    bet_list = []
    bet_list = [
        create_bet(market)
        for market in markets
        if market["mp"] in RECORDING_BETS
        and len(market["outcomes"])
        <= 3  # There is a bug in the API that returns more than 3 outcomes for some basketball games
    ]
    if not bet_list:
        logger.error("No data processed.")
    return pd.DataFrame([bet.__dict__ for bet in bet_list])


def create_bet(market: dict) -> Bet:
    """Create a Bet object from market data."""
    bet_type = SID_MAP.get(market["sId"], "unknown")
    if bet_type == "unknown":
        logger.warning(f"Unknown sport_id: {market['sId']}")

    event_date = datetime.datetime.strptime(str(market["e_date"]), "%y%m%d").date()
    time = str(market["m_hour"])[:2] + ":" + str(market["m_hour"])[2:]
    event = market["mp"]
    league = market["league"]
    option1 = remove_bidirectional_control_chars(market["outcomes"][0]["desc"])
    ratio1 = market["outcomes"][0]["price"]
    option2 = remove_bidirectional_control_chars(market["outcomes"][1]["desc"])
    ratio2 = market["outcomes"][1]["price"]
    option3, ratio3 = None, None
    if len(market["outcomes"]) == 3:
        option3 = remove_bidirectional_control_chars(market["outcomes"][2]["desc"])
        ratio3 = market["outcomes"][2]["price"]
    return Bet(
        bet_type,
        event_date,
        time,
        league,
        event,
        option1,
        ratio1,
        option2,
        ratio2,
        option3,
        ratio3,
    )


def save_to_s3(df, path, database, table, partition_cols):
    """Save dataframe to S3."""
    try:
        wr.s3.to_parquet(
            df,
            path=path,
            dataset=True,
            database=database,
            table=table,
            partition_cols=partition_cols,
            mode="append",
        )
    except Exception as e:
        logger.error(f"Failed to save to S3: {e}")
        raise


def save_raw_data(data: dict, date, run_time):
    data_string = json.dumps(data, indent=2, default=str)
    s3_client = boto3.client("s3")
    try:
        s3_client.put_object(
            Body=data_string, Bucket="winner-raw-data", Key=f"{date}/{run_time}.json"
        )
        logger.info(f"Successfully saved raw data.")
    except boto3.exceptions.S3UploadFailedError as e:
        logger.error("Failed to save raw data to S3: %s", e)
        raise


def main(event, context):
    logger.info("Environment: " + ENV)

    # lineChecksum = fetch_lineChecksum(HASH_CHECKSUM_URL)
    data = fetch_data(API_URL)

    try:
        bet_df = process_data(data)
    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        raise

    israel_timezone = pytz.timezone("Israel")
    current_time = datetime.datetime.now(israel_timezone)
    cur_date = current_time.strftime("%Y-%m-%d")
    cur_time = current_time.strftime("%H:%M:%S")
    run_time = f"{cur_date} {cur_time}"

    bet_df["date_parsed"] = cur_date
    bet_df["run_time"] = run_time

    hash_gen = HashGenerator()
    bet_df["result_id"] = bet_df.apply(hash_gen.generate_result_id, axis=1)
    bet_df["unique_id"] = bet_df.apply(hash_gen.generate_unique_id, axis=1)

    if ENV == "local":
        bet_df.to_csv("bets.csv", index=False)
    else:
        save_to_s3(
            bet_df,
            "s3://boaz-winner-api/bets",
            "winner-db",
            "api-odds",
            ["date_parsed", "type"],
        )
        save_raw_data(
            data,
            cur_date,
            run_time,
        )

    rows_processed = bet_df.shape[0]
    if rows_processed == 0:
        raise Exception("No data processed.")
    response = {
        "statusCode": 200,
        "body": f"Successfully scraped {rows_processed} rows.",
    }
    logger.info(f"Successfully wrote {rows_processed} rows.")
    return response


if __name__ == "__main__":
    main(None, None)
