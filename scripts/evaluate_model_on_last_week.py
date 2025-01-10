import os

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from scripts.train_model import BIG_GAMES, RATIOS, prepare_features, process_ratios

MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ"))

boto3.setup_default_session(region_name="il-central-1")
# MODEL = os.environ.get("MODEL")
EPOCHS = int(os.environ.get("EPOCHS"))
MODEL = f"model_{EPOCHS}_{MAX_SEQ_LENGTH}.h5"
loaded_model = load_model(f"trained_models/{MODEL}")

database = "winner-db"
THRESHOLD = 0.3

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


def determine_prediction_result(row, coloumn="predicted_winner"):
    if row[coloumn] == "1":
        return row["bet1_won"]
    elif row[coloumn] == "2":
        return row["bet2_won"]
    else:
        return row["tie_won"]


def predict_model(bet, processed):
    global predictions_df
    predictions_df = pd.DataFrame()
    game_odds = processed[processed["unique_id"] == bet["unique_id"]]
    game_odds = game_odds.sort_values("run_time")
    x = process_ratios(game_odds)[RATIOS]
    if x.shape[0] < MAX_SEQ_LENGTH:
        return predictions_df
    x = prepare_features(x, MAX_SEQ_LENGTH)

    prediction = loaded_model.predict(x.reshape(1, MAX_SEQ_LENGTH, 3), verbose=0)
    latest_odd = game_odds[game_odds["run_time"] == game_odds["run_time"].max()]
    # print(
    #     f"prediction for {bet['unique_id']} , ratios {latest_odd[RATIOS].values}, prediction {prediction[0]}"
    # )
    for outcome in [1, 2, 3]:
        threshold = 1 / float(latest_odd["ratio" + str(outcome)].values[0]) + THRESHOLD
        if prediction[0][outcome - 1] > threshold:
            new_prediction = create_prediction(bet, outcome, prediction, latest_odd)
            predictions_df = pd.concat(
                [predictions_df, new_prediction], ignore_index=True
            )
            # print(
            #     f"{outcome} - {bet['option1']} -  {bet['option2']} - {bet['option3']} - : {prediction[0]} and ratios {latest_odd['ratio1'].values[0]} - {latest_odd['ratio2'].values[0]} - {latest_odd['ratio3'].values[0]}"
            # )

    return predictions_df


def calculate_weighted_probabilities_and_guess(row: pd.Series, choices: list) -> str:
    inv_ratio1 = 1 / float(row["ratio1"])
    inv_ratio2 = 1 / float(row["ratio2"])
    inv_ratio3 = 1 / float(row["ratio3"])

    total_inv = inv_ratio1 + inv_ratio2 + inv_ratio3

    prob1 = inv_ratio1 / total_inv
    prob2 = inv_ratio2 / total_inv
    prob3 = inv_ratio3 / total_inv

    choices = ["1", "tie", "2"]
    probabilities = [prob1, prob2, prob3]

    return np.random.choice(choices, p=probabilities)


def add_guess_columns(predictions, latest_odds):
    predictions = predictions.merge(
        latest_odds[["unique_id", "ratio1", "ratio2", "ratio3"]], on="unique_id"
    )
    guess_to_ratio = {"1": "ratio1", "tie": "ratio2", "2": "ratio3"}
    choices = ["1", "tie", "2"]

    predictions["random_guess"] = np.random.choice(choices, size=len(predictions))
    predictions["random_guess_won"] = predictions.apply(
        determine_prediction_result, axis=1, args=("random_guess",)
    )
    predictions["random_guess_odds"] = predictions.apply(
        lambda row: row[guess_to_ratio[row["random_guess"]]], axis=1
    )
    predictions["weighted_guess"] = predictions.apply(
        calculate_weighted_probabilities_and_guess, axis=1, args=(choices,)
    )
    predictions["weighted_guess_won"] = predictions.apply(
        determine_prediction_result, axis=1, args=("weighted_guess",)
    )
    predictions["weighted_guess_odds"] = predictions.apply(
        lambda row: row[guess_to_ratio[row["weighted_guess"]]], axis=1
    )
    predictions = predictions.drop(columns=["ratio1", "ratio2", "ratio3"])
    return predictions


def main():
    all_results = []
    betted_games = set()
    # processed = wr.athena.read_sql_query(
    #     "SELECT * FROM processed_data", database=database
    # )
    # all_odds = wr.athena.read_sql_query(
    #     "SELECT * FROM api_odds where type='Soccer'", database=database
    # )
    # all_odds = pd.read_csv('all_odds.csv')
    processed = pd.read_parquet("processed_winner.parquet")
    processed = processed[processed["type"] == "Soccer"]

    total_earnings = 0
    total_random_guess_earnings = 0
    total_weighted_guess_earnings = 0

    total_bets = 0
    for date in pd.date_range(
        end=pd.Timestamp.now().date() - pd.Timedelta(days=2), periods=14
    ):
        predictions_df = pd.DataFrame()
        relevant_odds = processed[(pd.to_datetime(processed["date_parsed"]) <= date)]
        latest_odds = relevant_odds[
            (relevant_odds["run_time"] == relevant_odds["run_time"].max())
        ]
        if MODEL.startswith("big_games"):
            latest_odds = latest_odds[latest_odds["league"].isin(BIG_GAMES)]
        predictions = latest_odds.apply(predict_model, args=(processed,), axis=1)
        predictions_df = pd.concat(predictions.tolist(), ignore_index=True)
        if predictions_df.empty:
            continue
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
            continue
        predictions_and_results["pred_won"] = predictions_and_results.apply(
            determine_prediction_result, axis=1
        )
        predictions_and_results = add_guess_columns(
            predictions_and_results, latest_odds
        )

        expected_earnings = (
            predictions_and_results[predictions_and_results["pred_won"] == True]["odds"]
            .astype(float)
            .sum()
            - predictions_and_results.shape[0]
        )
        expected_earnings = round(expected_earnings, 2)
        total_earnings += expected_earnings
        random_guess_earnings = (
            predictions_and_results[
                predictions_and_results["random_guess_won"] == True
            ]["random_guess_odds"]
            .astype(float)
            .sum()
            - predictions_and_results.shape[0]
        )
        random_guess_earnings = round(random_guess_earnings, 2)
        total_random_guess_earnings += random_guess_earnings
        weighted_guess_earnings = (
            predictions_and_results[
                predictions_and_results["weighted_guess_won"] == True
            ]["weighted_guess_odds"]
            .astype(float)
            .sum()
            - predictions_and_results.shape[0]
        )
        weighted_guess_earnings = round(weighted_guess_earnings, 2)
        total_weighted_guess_earnings += weighted_guess_earnings
        num_predictions = predictions_and_results.shape[0]
        total_bets += num_predictions
        print(
            f"for date {date}, found {num_predictions} bets, expected earnings:{expected_earnings} ,random guess earnings:{random_guess_earnings}, weighted guess earnings:{weighted_guess_earnings}"
        )
        print(
            "-----------------------------------------------------------------------------------------------------------------------------"
        )
    print(
        f"SEQ {MAX_SEQ_LENGTH}  - total bets: {total_bets},total earnings:{total_earnings}, total random guess earnings:{total_random_guess_earnings}, total weighted guess earnings:{total_weighted_guess_earnings}"
    )
    print(
        f"expected returns:{total_earnings/total_bets}, random guess returns:{total_random_guess_earnings/total_bets}, weighted guess returns:{total_weighted_guess_earnings/total_bets}"
    )
    all_results.append(
        {
            "EPOCHS": EPOCHS,
            "MAX_SEQ_LENGTH": MAX_SEQ_LENGTH,
            "total_bets": total_bets,
            "total_earnings": round(total_earnings, 2),
            "expected_returns": round(total_earnings / total_bets, 2),
            "total_random_guess_earnings": round(total_random_guess_earnings, 2),
            "random_guessed_returns": round(
                total_random_guess_earnings / total_bets, 2
            ),
            "total_weighted_guess_earnings": round(total_weighted_guess_earnings, 2),
            "weighted_guess_returns": round(
                total_weighted_guess_earnings / total_bets, 2
            ),
        }
    )
    prev_results = pd.read_csv("evaluation_results.csv")
    all_results = pd.concat(
        [prev_results, pd.DataFrame(all_results)], ignore_index=True
    )
    all_results.to_csv("evaluation_results.csv", index=False)


if __name__ == "__main__":
    main()
