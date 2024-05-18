import os
from typing import Tuple

import awswrangler as wr
import boto3
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential

MAX_SEQ_LENGTH = 15
MIN_BET = 1
MAX_BET = 10
DAYS_TO_DISCARD = 10
EPOCHS = 10
database = "winner-db"

boto3.setup_default_session(region_name="il-central-1")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "all_games")
if MODEL_TYPE not in ["all_games", "big_games"]:
    raise ValueError("MODEL_TYPE must be either 'all_games' or 'big_games'")

BIG_GAMES = [
    "פרמייר ליג",
    "גביע ספרדי",
    "צרפתית ראשונה",
    "איטלקית ראשונה",
    "ליגת האלופות האסיאתית",
    "ליגת האלופות האפריקאית",
    "גביע אנגלי",
    "גביע המדינה",
    "קונפרנס ליג",
    "מוקדמות מונדיאל, אסיה",
    "מוקדמות אליפות אירופה",
    "גרמנית ראשונה",
    "ליגת העל",
    "סופר קאפ",
    "ספרדית ראשונה",
    "ליגת האלופות",
    "הליגה האירופית",
    "גביע איטלקי",
    "ליגת האומות",
    "גביע המדינה Winner",
    "גביע הליגה האנגלי",
    "גביע אסיה",
]
RATIOS = ["ratio1", "ratio2", "ratio3"]


def scale_features(features: pd.DataFrame) -> pd.DataFrame:
    return (features - MIN_BET) / (MAX_BET - 1)


def prepare_features(features: pd.DataFrame) -> np.array:
    features_scaled = scale_features(features)
    features_scaled = features_scaled.to_numpy()
    features_scaled = features_scaled[-MAX_SEQ_LENGTH:, :]
    return features_scaled


def remove_consecutive_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    shifted = df[RATIOS].shift()
    mask = ~((df[RATIOS] == shifted).all(axis=1))
    return df[mask]


def process_ratios(group_df: pd.DataFrame) -> pd.DataFrame:
    group_df[RATIOS] = group_df[RATIOS].astype(float)
    group_df = remove_consecutive_duplicates(group_df)
    return group_df


def prepare_data_for_model(grouped_data: pd.DataFrame) -> Tuple[np.array, np.array]:
    x, y = [], []
    for _, group_df in grouped_data:
        group_df = process_ratios(group_df)
        if group_df.shape[0] < MAX_SEQ_LENGTH:
            continue
        features = group_df[RATIOS]
        target = group_df[["bet1_won", "bet2_won", "tie_won"]].values[-1].astype(float)

        prepared_features = prepare_features(features)
        x.append(prepared_features)
        y.append(target)
    x = np.array(x)
    y = np.array(y)
    return x, y


def build_model(
    learning_rate: float, dropout_rate: float, units: int, activation: str
) -> Sequential:
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH, 3)))
    model.add(Bidirectional(LSTM(units, return_sequences=True)))
    model.add(LSTM(units // 2, activation=activation, return_sequences=True))
    model.add(LSTM(units // 2, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


def objective(trial: optuna.trial.Trial) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
    units = trial.suggest_categorical("units", [32, 64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    activation = trial.suggest_categorical("activation", ["sigmoid", "tanh"])

    last_week_start = pd.Timestamp.now().date() - pd.Timedelta(days=DAYS_TO_DISCARD)
    query = f"""
    SELECT league, ratio1, ratio2, ratio3, bet1_won, bet2_won, tie_won, run_time, unique_id 
    FROM processed_data 
    WHERE type='Soccer' 
    AND date_parsed < '{last_week_start}' 
    """
    processed = wr.athena.read_sql_query(query, database=database)

    if MODEL_TYPE == "big_games":
        processed = processed[processed["league"].isin(BIG_GAMES)]
    grouped_processed = list(
        processed.sort_values(by="run_time").groupby(["unique_id"])
    )
    x, y = prepare_data_for_model(grouped_processed)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracy = []
    for train_index, val_index in kfold.split(x):
        X_train, X_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = build_model(learning_rate, dropout_rate, units, activation)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=batch_size, verbose=1)
        loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracy.append(val_accuracy)

    avg_accuracy = np.mean(fold_accuracy)
    return avg_accuracy


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print(f"Best trial: {study.best_trial.value}")
    print(f"Best hyperparameters: {study.best_trial.params}")

    # Train the final model with the best hyperparameters
    best_params = study.best_trial.params
    best_model = build_model(
        best_params["learning_rate"],
        best_params["dropout_rate"],
        best_params["units"],
        best_params["activation"],
    )

    last_week_start = pd.Timestamp.now().date() - pd.Timedelta(days=DAYS_TO_DISCARD)
    query = f"""
    SELECT league, ratio1, ratio2, ratio3, bet1_won, bet2_won, tie_won, run_time, unique_id 
    FROM processed_data 
    WHERE type='Soccer' 
    AND date_parsed < '{last_week_start}' 
    """
    processed = wr.athena.read_sql_query(query, database=database)

    if MODEL_TYPE == "big_games":
        processed = processed[processed["league"].isin(BIG_GAMES)]
    grouped_processed = list(
        processed.sort_values(by="run_time").groupby(["unique_id"])
    )
    x, y = prepare_data_for_model(grouped_processed)

    best_model.fit(x, y, epochs=10, batch_size=best_params["batch_size"], verbose=1)
    best_model.save(f"trained_models/{MODEL_TYPE}_model_v1.h5")


if __name__ == "__main__":
    main()
