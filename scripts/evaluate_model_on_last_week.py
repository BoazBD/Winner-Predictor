import os

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Import centralized data processing functions
from data_processing import (
    BIG_GAMES, RATIOS, FIXED_STRIDE,
    prepare_features_single, 
    process_ratios
)

MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ"))

boto3.setup_default_session(region_name="il-central-1")
MODEL_TYPE = os.environ.get("MODEL_TYPE")
EPOCHS = int(os.environ.get("EPOCHS"))
THRESHOLD = float(os.environ.get("THRESHOLD"))
MODEL_VERSION = int(os.environ.get("MODEL_VERSION"))
MODEL = f"big_games_standard_{EPOCHS}_{MAX_SEQ_LENGTH}_{MODEL_VERSION}.h5"
MODEL = "lstm_100_12_v1.h5"

loaded_model = load_model(f"trained_models/{MODEL}")

database = "winner-db"
print(f"evaluating model {MODEL} with threshold {THRESHOLD}")
print(f"Using fixed stride: {FIXED_STRIDE}")


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


def determine_prediction_result(row, coloumn="predicted_winner"):
    if row[coloumn] == "1":
        return row["bet1_won"]
    elif row[coloumn] == "2":
        return row["bet2_won"]
    else:
        return row["tie_won"]


def predict_model(bet, processed):
    bet_predictions_df = pd.DataFrame()
    game_odds = processed[processed["unique_id"] == bet["unique_id"]]
    game_odds = game_odds.sort_values("run_time")
    x = process_ratios(game_odds)[RATIOS]
    if x.shape[0] < MAX_SEQ_LENGTH:
        return bet_predictions_df
    x = prepare_features_single(x, MAX_SEQ_LENGTH, scaling_method="standard")

    prediction = loaded_model.predict(x.reshape(1, MAX_SEQ_LENGTH, 3), verbose=0)
    latest_odd = game_odds[game_odds["run_time"] == game_odds["run_time"].max()]
    # print(
    #     f"prediction for {bet['unique_id']} , ratios {latest_odd[RATIOS].values}, prediction {prediction[0]}"
    # )
    for outcome in [1, 2, 3]:
        threshold = 1 / float(latest_odd["ratio" + str(outcome)].values[0]) + THRESHOLD
        if prediction[0][outcome - 1] > threshold:
            new_prediction = create_prediction(bet, outcome, prediction, latest_odd)
            bet_predictions_df = pd.concat(
                [bet_predictions_df, new_prediction], ignore_index=True
            )
            print(
                f"{outcome} - {bet['option1']} -  {bet['option2']} - {bet['option3']} - : {prediction[0]} and ratios {latest_odd['ratio1'].values[0]} - {latest_odd['ratio2'].values[0]} - {latest_odd['ratio3'].values[0]}"
            )

    return bet_predictions_df

def main():
    all_results = []
    betted_games = set()
    # processed = wr.athena.read_sql_query(
    #     "SELECT * FROM processed_data", database=database
    # )
    print(f"Evaluating model {MODEL}")
    processed = pd.read_parquet("latest_processed_winner.parquet")
    processed = processed[processed["type"] == "Soccer"]

    total_earnings = 0

    total_bets = 0
    for date in pd.date_range(
        end=pd.Timestamp.now().date() - pd.Timedelta(days=2), periods=150
    ):
        predictions_df = pd.DataFrame()
        relevant_odds = processed[(pd.to_datetime(processed["date_parsed"]) <= date)]
        latest_run_time = relevant_odds["run_time"].max()
        
        # Get unique_ids that have data for the latest run time
        valid_games = relevant_odds[relevant_odds["run_time"] == latest_run_time]["unique_id"].unique()
        
        # Filter relevant_odds to only include these games
        relevant_odds = relevant_odds[relevant_odds["unique_id"].isin(valid_games)]
        
        # Get latest odds for these games
        latest_odds = relevant_odds[
            (relevant_odds["run_time"] == latest_run_time)
        ]
        
        latest_odds = latest_odds[latest_odds["league"].isin(BIG_GAMES)]
            
        latest_odds = latest_odds[
            ~latest_odds["unique_id"].isin(betted_games)
        ]
        predictions = latest_odds.apply(predict_model, args=(processed,), axis=1)
        
        # Filter out empty DataFrames before concatenating
        non_empty_predictions = [pred for pred in predictions.tolist() if not pred.empty]
        
        if not non_empty_predictions:
            print(f"no predictions for date {date}")
            continue
            
        predictions_df = pd.concat(non_empty_predictions, ignore_index=True)
        
        predictions_and_results = predictions_df.merge(
            processed.groupby("unique_id").head(1)[
                ["unique_id", "bet1_won", "bet2_won", "tie_won"]
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

        if predictions_and_results.empty:
            print(f"no predictions for date {date}")
            continue
        predictions_and_results["pred_won"] = predictions_and_results.apply(
            determine_prediction_result, axis=1
        )

        expected_earnings = (
            predictions_and_results[predictions_and_results["pred_won"] == True]["odds"]
            .astype(float)
            .sum()
            - predictions_and_results.shape[0]
        )
        expected_earnings = round(expected_earnings, 2)
        total_earnings += expected_earnings

        num_predictions = predictions_and_results.shape[0]
        total_bets += num_predictions
        print(
            f"for date {date}, found {num_predictions} bets, expected earnings:{expected_earnings}"
        )
        print(
            "-----------------------------------------------------------------------------------------------------------------------------"
        )
    print(
        f"SEQ {MAX_SEQ_LENGTH}  - total bets: {total_bets},total earnings:{total_earnings}"
    )
    total_bets = total_bets if total_bets > 0 else -1
    all_results.append(
        {
            "THRESHOLD": THRESHOLD,
            "MODEL": 'o_' + MODEL,
            "EPOCHS": EPOCHS,
            "MAX_SEQ_LENGTH": MAX_SEQ_LENGTH,
            "total_bets": total_bets,
            "total_earnings": round(total_earnings, 2),
            "expected_returns": round(total_earnings / total_bets, 2),
        }
    )
    # Check if evaluation_results.csv exists, if not create it
    if os.path.exists("evaluation_results.csv"):
        prev_results = pd.read_csv("evaluation_results.csv")
        all_results = pd.concat(
            [prev_results, pd.DataFrame(all_results)], ignore_index=True
        )
    else:
        # File doesn't exist, create new DataFrame
        all_results = pd.DataFrame(all_results)
    
    all_results.to_csv("evaluation_results.csv", index=False)


if __name__ == "__main__":
    main()
