import datetime
import hashlib
import json
import logging
import os

import awswrangler as wr
import boto3
import pandas as pd
import pytz
import requests
import random

from Bet import Bet

ENV = os.environ.get("ENV", "local")

RECORDING_BETS = [
    "הימור יתרון - תוצאת סיום (ללא הארכות)",  # Soccer
    "‮1X2‬ - תוצאת סיום (ללא הארכות)",  # Soccer
    "הימור יתרון - ללא הארכות",  # Basketball
    "הימור יתרון - כולל הארכות אם יהיו",  # Football
    "מחצית/סיום - ללא הארכות",  # Football
    "המנצח/ת - משחק",  # Tennis
]
SID_MAP = {
    240: "Soccer",
    227: "Basketball",
    1100: "Handball",
    1: "Football",
    239: "Tennis",
    226: "Baseball",
}
HASH_CHECKSUM_URL = "https://api.winner.co.il/v2/publicapi/GetCMobileHashes"
API_URL = "https://api.winner.co.il/v2/publicapi/GetCMobileLine"

headers = {
    "Deviceid": f"2e7f{random.randint(0,9)}66a5ff1{random.randint(0,9)}9d4a122e3d{random.randint(0,9)}5b35a0{random.randint(0,9)}",
    "Hashesmessage": '{"reason":"Initiated"}',
    "Host": "api.winner.co.il",
    "Origin": "https://www.winner.co.il",
    "Referer": "https://www.winner.co.il/",
    "Requestid": f"2bddbb1d6{random.randint(0,9)}275b23ecea8bf0{random.randint(0,9)}09d{random.randint(0,9)}6d",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Useragentdatresa": '{"devicemodel":"Macintosh","deviceos":"mac os","deviceosversion":"10.15.7","appversion":"1.8.1","apptype":"mobileweb","originId":"3","isAccessibility":false}',
    "X-Csrf-Token": "null",
}
PROXY_URL = "http://129.159.134.66:8080"
boto3.setup_default_session(region_name="il-central-1")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def clean_team_name(team_name):
    """
    This function cleans a team name by removing any parentheses and numbers at the end.
    """
    # Find the last index of an opening parenthesis
    opening_bracket_index = team_name.rfind("(")
    # If an opening parenthesis is found, remove everything after it (including the parenthesis)
    if opening_bracket_index != -1:
        return team_name[:opening_bracket_index].strip()
    else:
        return team_name


def create_hash(type, date, league, team1, team2):
    data_to_hash = f"{type}_{date}_{league}_{team1}_{team2}"
    return hashlib.sha1(data_to_hash.encode()).hexdigest()[:8]


def generate_unique_id(row):
    if not pd.isnull(row["option3"]):
        second_team = row["option3"]
    else:
        second_team = row["option2"]

    return create_hash(
        row["type"], row["event_date"], row["league"], row["option1"], second_team
    )


def generate_result_id(row):
    first_team = clean_team_name(row["option1"])
    # Don't use ternary operator here, as it will always evaluate both sides
    if not pd.isnull(row["option3"]):
        second_team = clean_team_name(row["option3"])
    else:
        second_team = clean_team_name(row["option2"])

    return create_hash(
        row["type"], row["event_date"], row["league"], first_team, second_team
    )


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
        # Make a POST request to the proxy server with the API URL
        response = requests.post(
            PROXY_URL,
            json={"url": url, "headers": headers},
        )
        response.raise_for_status()
        return response.json()["lineChecksum"]
    except Exception as e:
        logger.error("Failed to fetch lineChecksum from the API.")
        raise Exception(f"Failed to fetch lineChecksum from the API: {e}")


def fetch_data(url: str, lineChecksum: str) -> dict:
    """Fetch data from the API."""
    complete_url = f"{url}?lineChecksum={lineChecksum}"
    try:
        response = requests.post(
            PROXY_URL, json={"url": complete_url, "headers": headers}
        )
        return response.json()
    except:
        logger.error("Failed to fetch data from the API.")
        raise Exception(
            f"Failed to fetch data from the API: {response.status_code} , {response.content}"
        )


def process_data(data: dict) -> pd.DataFrame:
    """Process API data and return a list of bets."""
    markets = data["markets"]
    bet_list = []
    for market in markets:
        if (
            market["mp"] in RECORDING_BETS and len(market["outcomes"]) <= 3
        ):  # There is a bug in the API that returns more than 3 outcomes for some basketball games
            bet_list.append(create_bet(market))
    if not bet_list:
        logger.error("No data processed.")
    return pd.DataFrame([bet.__dict__ for bet in bet_list])


def create_bet(market: dict) -> Bet:
    """Create a Bet object from market data."""

    bet_type = SID_MAP.get(market["sId"], "unknown")
    if bet_type == "unknown":
        logger.error(f"Unknown sport_id: {market['sId']}")
    event_date = datetime.datetime.strptime(str(market["e_date"]), "%y%m%d").date()
    time = str(market["m_hour"])[:2] + ":" + str(market["m_hour"])[2:]
    event = market["mp"]
    league = market["league"]
    option1 = remove_bidirectional_control_chars(market["outcomes"][0]["desc"])
    ratio1 = market["outcomes"][0]["price"]
    option2 = remove_bidirectional_control_chars(market["outcomes"][1]["desc"])
    ratio2 = market["outcomes"][1]["price"]
    option3 = ratio3 = None
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
        raise Exception("Failed to save to S3")


def save_raw_data(data: dict, date, run_time):
    data_string = json.dumps(data, indent=2, default=str)
    s3_client = boto3.client("s3")
    s3_client.put_object(
        Body=data_string, Bucket="winner-raw-data", Key=f"{date}/{run_time}.json"
    )
    logger.info(f"Successfully saved raw data.")


def main(event, context):
    print("enviroment: ", ENV)
    lineChecksum = fetch_lineChecksum(HASH_CHECKSUM_URL)
    data = fetch_data(API_URL, lineChecksum)
    try:
        bet_df = process_data(data)
    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        raise Exception("Failed to process data")

    israel_timezone = pytz.timezone("Israel")
    current_time = datetime.datetime.now(israel_timezone)
    cur_date = current_time.strftime("%Y-%m-%d")
    cur_time = current_time.strftime("%H:%M:%S")
    run_time = f"{cur_date} {cur_time}"
    bet_df["date_parsed"] = cur_date
    bet_df["run_time"] = run_time
    bet_df["result_id"] = bet_df.apply(generate_result_id, axis=1)
    bet_df["unique_id"] = bet_df.apply(generate_unique_id, axis=1)
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
