import os

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from scripts.train_model import BIG_GAMES, MAX_SEQ_LENGTH, prepare_features

boto3.setup_default_session(region_name="il-central-1")
MODEL = os.environ.get("MODEL")
loaded_model = load_model(f"trained_models/{MODEL}")

database = "winner-db"
THRESHOLD = 0.2

predictions_df = pd.DataFrame()


def create_prediction(bet, predicted_winner, prediction, latest_odds):
    new_row = pd.DataFrame(
        {
            "result_id": [bet["result_id"]],
            "unique_id": [bet["unique_id"]],
            "predicted_winner": [
                (
                    "1"
                    if predicted_winner == 1
                    else "2" if predicted_winner == 3 else "tie"
                )
            ],
            "prediction_precent": [prediction[0][predicted_winner - 1]],
            "odds": [latest_odds["ratio" + str(predicted_winner)].values[0]],
        }
    )
    return new_row


def determine_prediction_result(row):
    if row["predicted_winner"] == "1":
        return row["bet1_won"]
    elif row["predicted_winner"] == "2":
        return row["bet2_won"]
    else:
        return row["tie_won"]


def predict_model(bet, all_odds):
    global predictions_df
    predictions_df = pd.DataFrame()
    game_odds = all_odds[all_odds["unique_id"] == bet["unique_id"]]
    if game_odds.shape[0] < 400:
        return
    game_odds = game_odds.sort_values("run_time")
    x = game_odds[["ratio1", "ratio2", "ratio3"]].astype(float)
    x = prepare_features(x)

    prediction = loaded_model.predict(x.reshape(1, MAX_SEQ_LENGTH, 3))
    latest_odd = game_odds[game_odds["run_time"] == game_odds["run_time"].max()]
    for outcome in [1, 2, 3]:
        threshold = 1 / float(latest_odd["ratio" + str(outcome)].values[0]) + THRESHOLD
        if prediction[0][outcome - 1] > threshold:
            new_prediction = create_prediction(bet, outcome, prediction, latest_odd)
            predictions_df = pd.concat(
                [predictions_df, new_prediction], ignore_index=True
            )
            print(
                f"{outcome} - {bet['option1']} -  {bet['option2']} - {bet['option3']} - : {prediction[0]} and ratios {latest_odd['ratio1'].values[0]} - {latest_odd['ratio2'].values[0]} - {latest_odd['ratio3'].values[0]}"
            )

    return predictions_df


def main():
    betted_games = set()
    processed = wr.athena.read_sql_query(
        "SELECT * FROM processed_data", database=database
    )
    all_odds = wr.athena.read_sql_query(
        "SELECT * FROM api_odds where type='Soccer'", database=database
    )
    total_earnings = 0
    total_bets = 0
    for date in pd.date_range(
        end=pd.Timestamp.now().date() - pd.Timedelta(days=1), periods=7
    ):
        predictions_df = pd.DataFrame()
        relevant_odds = all_odds[pd.to_datetime(all_odds["date_parsed"]) <= date]
        latest_odds = relevant_odds[
            (relevant_odds["run_time"] == relevant_odds["run_time"].max())
        ]
        if MODEL.startswith("big_games"):
            latest_odds = latest_odds[latest_odds["league"].isin(BIG_GAMES)]
        predictions = latest_odds.apply(predict_model, args=(all_odds,), axis=1)
        predictions_df = pd.concat(predictions.tolist(), ignore_index=True)

        predictions_and_results = predictions_df.merge(
            processed.groupby("unique_id").head(1)[
                ["result_id", "unique_id", "bet1_won", "bet2_won", "tie_won"]
            ],
            on="unique_id",
            how="left",
            validate="m:1",
        )
        predictions_and_results = predictions_and_results[
            predictions_and_results["bet1_won"].notnull()
        ]  # remove games without results yet

        predictions_and_results = predictions_and_results[
            ~predictions_and_results["unique_id"].isin(betted_games)
        ]
        betted_games.update(predictions_and_results["unique_id"].tolist())

        predictions_and_results["pred_won"] = predictions_and_results.apply(
            determine_prediction_result, axis=1
        )
        expected_earnings = (
            predictions_and_results[predictions_and_results["pred_won"] == True]["odds"]
            .astype(float)
            .sum()
            - predictions_and_results.shape[0]
        )
        total_earnings += expected_earnings
        num_predictions = predictions_and_results.shape[0]
        total_bets += num_predictions
        print(
            f"for date {date}, found {num_predictions} bets and expected earnings are {expected_earnings}"
        )
    print(
        f"Model {MODEL} - total bets: {total_bets},total earnings are {total_earnings}"
    )


if __name__ == "__main__":
    main()
