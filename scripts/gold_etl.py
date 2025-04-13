import datetime
import logging

import awswrangler as wr
import boto3
import pandas as pd

from scripts.get_results import main as get_latest_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

boto3.setup_default_session(region_name="il-central-1")
database = "winner-db"


def get_tables():
    odds = wr.athena.read_sql_query("SELECT * FROM api_odds", database=database)
    results = wr.athena.read_sql_query("SELECT * FROM results", database=database)
    logger.info("Tables loaded successfully")
    return odds, results


def clean_tables(
    odds: pd.DataFrame, results: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    today = datetime.date.today()
    odds = odds[odds["event_date"] < today]
    odds = odds[~odds["option1"].str.contains(r".+ - .+")]  # Buggy odds handler

    results = results.drop_duplicates(subset=["id"], keep="first")
    logger.info("Tables cleaned successfully")
    return odds, results


def calculate_final_score(option: str, score: int) -> float:
    """
    Gets option and score and returns the final score after applying the constraint.
    For example, if the option is "Barcelona (+2)" and the score is 1, the final score will be 3.0
    Return value is float, since some constraints are not integers.
    """
    try:
        opening_bracket_index = option.rfind("(")
        closing_bracket_index = option.rfind(")")
        if opening_bracket_index == -1 and closing_bracket_index == -1:
            return score
        elif opening_bracket_index == -1 or closing_bracket_index == -1:
            raise ValueError(f"Invalid constraint for option {option}")

        constraint = option[opening_bracket_index + 1 : closing_bracket_index]
        if constraint == "0":
            return score
        sign = constraint[0]
        if sign == "-":
            # The negative sign is disregarded to prevent doubling the constraint
            return score
        value = float(constraint[1:])
        value = -value if sign == "-" else value
        return score + value
    except:
        raise Exception(
            f"Error while calculating final score for option {option} with score {score}"
        )


def validate_gold(gold: pd.DataFrame):
    if gold["result_id"].isnull().any() or gold["unique_id"].isnull().any():
        raise ValueError("Null values found in result_id or unique_id columns")

    invalid_rows = gold[(gold["final_scorea"] < 0) | (gold["final_scoreb"] < 0)]
    if invalid_rows.shape[0] > 0:
        raise ValueError(
            f"Invalid final scores found in gold table: {invalid_rows.shape[0]} rows"
        )
    only_one_true = gold[["bet1_won", "bet2_won", "tie_won"]].sum(axis=1) == 1
    if not only_one_true.all():
        raise ValueError(
            "Multiple True values found in bet1_won, bet2_won, tie_won columns"
        )
    duplicate_run_times = gold[
        gold.duplicated(subset=["unique_id", "run_time"], keep=False)
    ]
    if not duplicate_run_times.empty:
        pass
        # TODO - Fix duplications
    logger.info("Gold table validated successfully")


def save_results_to_s3(table: pd.DataFrame):
    try:
        wr.s3.to_parquet(
            table,
            path="s3://boaz-winner-api/processed_data",
            dataset=True,
            database=database,
            table="processed_data",
            partition_cols=["event_date", "type"],
            mode="overwrite_partitions",
        )
        logger.info(f"Successfully loaded to S3 {table.shape[0]} complete bets")
    except Exception as e:
        logger.error("An error occurred while saving to S3:", str(e))


def main():
    get_latest_results()
    odds, results = get_tables()
    odds, results = clean_tables(odds, results)
    gold = odds.merge(
        results[["id", "scorea", "scoreb"]],
        left_on="result_id",
        right_on="id",
        how="left",
        validate="m:1",
    ).drop(columns=["id"])
    logger.info("Merged tables successfully")
    logger.info(f"Unmatched games: {gold['scorea'].isna().sum()}")
    if gold["scorea"].isna().sum() / gold.shape[0] > 0.08:
        raise ValueError("Too many unmatched games")
    gold = gold[gold["scorea"].notnull()]
    gold["final_scorea"] = gold.apply(
        lambda row: calculate_final_score(row["option1"], int(row["scorea"])), axis=1
    )
    gold["final_scoreb"] = gold.apply(
        lambda row: (
            calculate_final_score(row["option2"], int(row["scoreb"]))
            if pd.isnull(row["option3"])
            else calculate_final_score(row["option3"], int(row["scoreb"]))
        ),
        axis=1,
    )
    gold[["final_scorea", "final_scoreb"]] = gold[
        ["final_scorea", "final_scoreb"]
    ].astype(float)
    logger.info("Final scores calculated successfully")
    gold["bet1_won"] = gold["final_scorea"] > gold["final_scoreb"]
    gold["bet2_won"] = gold["final_scoreb"] > gold["final_scorea"]
    gold["tie_won"] = gold["final_scorea"] == gold["final_scoreb"]
    validate_gold(gold)
    save_results_to_s3(gold)


if __name__ == "__main__":
    main()
