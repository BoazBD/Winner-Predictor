import datetime
import hashlib
import logging
import os

import awswrangler as wr
import boto3
import pandas as pd
import pytz
import requests

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
API_URL = "https://api.winner.co.il/v2/publicapi/GetCMobileLine"

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


def generate_id(row):
    first_team = clean_team_name(row["option1"])
    second_team = (
        clean_team_name(row["option3"])
        if row["option3"] is not None
        else clean_team_name(row["option2"])
    )

    return create_hash(
        row["type"], row["date_parsed"], row["league"], first_team, second_team
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


def fetch_data(url: str) -> dict:
    """Fetch data from the API."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        logger.error("Failed to fetch data from the API.")
        raise Exception("Failed to fetch data from the API")


def process_data(data: dict) -> pd.DataFrame:
    """Process API data and return a list of bets."""
    markets = data["markets"]
    bet_list = []
    for market in markets:
        if market["mp"] in RECORDING_BETS:
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
        boto3.setup_default_session(region_name="il-central-1")
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


def main(event, context):
    print("enviroment: ", ENV)
    data = fetch_data(API_URL)
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
    bet_df["id"] = bet_df.apply(generate_id, axis=1)
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
    rows_processed = bet_df.shape[0]
    logger.info(f"Successfully wrote {rows_processed} rows.")
    if rows_processed == 0:
        raise Exception("No data processed.")
    response = {
        "statusCode": 200,
        "body": f"Successfully scraped {rows_processed} rows.",
    }
    return response


if __name__ == "__main__":
    main(None, None)
