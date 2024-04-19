import logging

import awswrangler as wr
import boto3
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

boto3.setup_default_session(region_name="il-central-1")
database = "winner-db"


def get_tables():
    odds = wr.athena.read_sql_query("SELECT * FROM api_odds", database=database)
    results = wr.athena.read_sql_query("SELECT * FROM results", database=database)
    return odds, results


def clean_tables(odds: pd.DataFrame) -> pd.DataFrame:
    odds = odds[odds["event_date"] < pd.to_datetime("2024-04-16").date()]
    odds = odds[~odds["option1"].str.contains(r".+ - .+")]
    odds = odds[odds.groupby("id")["id"].transform("count") >= 50]
    return odds


def save_results_to_s3(table: pd.DataFrame):
    try:
        wr.s3.to_parquet(
            table,
            path="s3://boaz-winner-api/gold",
            dataset=True,
            database=database,
            table="gold",
            partition_cols=["event_date", "type"],
            mode="overwrite_partitions",
        )
        logger.info(f"Successfully loaded to S3 {table.shape[0]} complete bets")
    except Exception as e:
        logger.error("An error occurred while saving to S3:", str(e))


if __name__ == "__main__":
    odds, results = get_tables()
    odds = clean_tables(odds)
    complete_table = odds.merge(
        results[["id", "scorea", "scoreb"]], on="id", how="inner"
    )
    save_results_to_s3(complete_table)
