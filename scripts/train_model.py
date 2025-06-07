import logging
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
import random

boto3.setup_default_session(region_name="il-central-1")

# Define constants and configurations
MIN_BET = 1
MAX_BET = 10
DAYS_TO_DISCARD = 155
COMPETITION_LEVEL = os.environ.get("COMPETITION_LEVEL", "big_games")
if COMPETITION_LEVEL not in ["all_games", "big_games"]:
    raise ValueError("COMPETITION_LEVEL must be either 'all_games' or 'big_games'")


BIG_GAMES = [
    "פרמייר ליג",
    "גביע ספרדי",
    "צרפתית ראשונה",
    "איטלקית ראשונה",
    "גביע אנגלי",
    "קונפרנס ליג",
    "מוקדמות אליפות אירופה",
    "מוקדמות מונדיאל, אירופה",
    "גרמנית ראשונה",
    "ליגת העל",
    "סופר קאפ",
    "ספרדית ראשונה",
    "ליגת האלופות",
    "הליגה האירופית",
    "גביע איטלקי",
    "ליגת האומות",
    "גביע המדינה Winner",
    "ליגת Winner",
    "גביע הליגה האנגלי",
    "גביע אסיה",
    "גביע גרמני",
]
RATIOS = ["ratio1", "ratio2", "ratio3"]

logger = logging.getLogger(__name__)


def scale_features(features: pd.DataFrame) -> pd.DataFrame:
    return (features - MIN_BET) / (MAX_BET - 1)


def prepare_features_sliding_window(features: pd.DataFrame, max_seq_len: int, stride: int) -> list:
    """Generate sliding windows of features."""
    features_scaled = scale_features(features)
    features_scaled = features_scaled.to_numpy()
    
    windows = []
    for start_idx in range(0, len(features_scaled) - max_seq_len + 1, stride):
        end_idx = start_idx + max_seq_len
        window = features_scaled[start_idx:end_idx, :]
        windows.append(window)
    
    return windows


def prepare_features(features: pd.DataFrame, max_seq_len) -> np.array:
    features_scaled = scale_features(features)
    features_scaled = features_scaled.to_numpy()
    features_scaled = features_scaled[-max_seq_len:, :]
    return features_scaled


def remove_consecutive_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    shifted = df[RATIOS].shift()
    mask = ~((df[RATIOS] == shifted).all(axis=1))
    return df[mask]


def process_ratios(group_df: pd.DataFrame) -> pd.DataFrame:
    group_df[RATIOS] = group_df[RATIOS].astype(float)
    group_df = remove_consecutive_duplicates(group_df)
    return group_df


def prepare_data_for_model(
    grouped_data: pd.DataFrame, max_seq_length: int, stride: int = None
) -> Tuple[np.array, np.array]:
    if stride is None:
        stride = max_seq_length // 2
    
    x, y = [], []
    total_games = 0
    total_windows = 0
    
    for _, group_df in grouped_data:
        group_df = process_ratios(group_df)
        if group_df.shape[0] < max_seq_length:
            continue
        
        total_games += 1
        features = group_df[RATIOS]
        target = group_df[["bet1_won", "tie_won", "bet2_won"]].values[-1].astype(float)

        # Generate sliding windows for this game
        sliding_windows = prepare_features_sliding_window(features, max_seq_length, stride)
        total_windows += len(sliding_windows)
        
        # Add each window as a separate training sample with the same target
        for window in sliding_windows:
            x.append(window)
            y.append(target)
    
    x = np.array(x)
    y = np.array(y)
    
    print(f"Generated {total_windows} sliding windows from {total_games} games")
    print(f"Average windows per game: {total_windows / total_games if total_games > 0 else 0:.2f}")
    print(f"Training data shape: X={x.shape}, y={y.shape}")
    
    return x, y


def build_model(
    learning_rate: float,
    dropout_rate: float,
    units: int,
    activation: str,
    max_seq_length: int,
) -> Sequential:
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(max_seq_length, 3)))
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


def objective(trial: optuna.trial.Trial, epochs: int, max_seq_length: int) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    units = trial.suggest_categorical("units", [32, 64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    activation = trial.suggest_categorical("activation", ["sigmoid", "tanh"])
    stride = max_seq_length // 2  # Default stride

    last_week_start = pd.Timestamp.now().date() - pd.Timedelta(days=DAYS_TO_DISCARD)
    
    processed = pd.read_parquet("processed_winner_070625.parquet")
    processed = processed[processed["type"] == "Soccer"]
    processed = processed[pd.to_datetime(processed["date_parsed"]).dt.date < last_week_start]
    if COMPETITION_LEVEL == "big_games":
        processed = processed[processed["league"].isin(BIG_GAMES)]
    grouped_processed = list(
        processed.sort_values(by="run_time").groupby(["unique_id"])
    )
    x, y = prepare_data_for_model(grouped_processed, max_seq_length, stride)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracy = []
    for train_index, val_index in kfold.split(x):
        X_train, X_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = build_model(
            learning_rate, dropout_rate, units, activation, max_seq_length
        )
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracy.append(val_accuracy)

    avg_accuracy = np.mean(fold_accuracy)
    return avg_accuracy


def train_and_evaluate_model(epochs: int, max_seq_length: int, stride: int = None):
    if stride is None:
        stride = max_seq_length // 2
        
    study = optuna.create_study(direction="maximize")
    # Create a modified objective function that uses the specified stride
    def objective_with_stride(trial):
        return objective_with_custom_stride(trial, epochs, max_seq_length, stride)
    
    study.optimize(objective_with_stride, n_trials=1)

    print(f"Best trial: {study.best_trial.value}")
    print(f"Best hyperparameters: {study.best_trial.params}")

    best_params = study.best_trial.params
    best_model = build_model(
        best_params["learning_rate"],
        best_params["dropout_rate"],
        best_params["units"],
        best_params["activation"],
        max_seq_length,
    )

    last_week_start = pd.Timestamp.now().date() - pd.Timedelta(days=DAYS_TO_DISCARD)

    processed = pd.read_parquet("processed_winner_070625.parquet")
    processed = processed[processed["type"] == "Soccer"]
    processed = processed[pd.to_datetime(processed["date_parsed"]).dt.date < last_week_start]
    if COMPETITION_LEVEL == "big_games":
        processed = processed[processed["league"].isin(BIG_GAMES)]
    grouped_processed = list(
        processed.sort_values(by="run_time").groupby(["unique_id"])
    )
    x, y = prepare_data_for_model(grouped_processed, max_seq_length, stride)

    best_model.fit(x, y, epochs=epochs, batch_size=best_params["batch_size"], verbose=0)
    best_model.save(f"trained_models/{COMPETITION_LEVEL}_{epochs}_{max_seq_length}_n1.h5")


def objective_with_custom_stride(trial: optuna.trial.Trial, epochs: int, max_seq_length: int, stride: int) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    units = trial.suggest_categorical("units", [32, 64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    activation = trial.suggest_categorical("activation", ["sigmoid", "tanh"])

    last_week_start = pd.Timestamp.now().date() - pd.Timedelta(days=DAYS_TO_DISCARD)
    
    processed = pd.read_parquet("processed_winner_070625.parquet")
    processed = processed[processed["type"] == "Soccer"]
    processed = processed[pd.to_datetime(processed["date_parsed"]).dt.date < last_week_start]
    if COMPETITION_LEVEL == "big_games":
        processed = processed[processed["league"].isin(BIG_GAMES)]
    grouped_processed = list(
        processed.sort_values(by="run_time").groupby(["unique_id"])
    )
    x, y = prepare_data_for_model(grouped_processed, max_seq_length, stride)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracy = []
    for train_index, val_index in kfold.split(x):
        X_train, X_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = build_model(
            learning_rate, dropout_rate, units, activation, max_seq_length
        )
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracy.append(val_accuracy)

    avg_accuracy = np.mean(fold_accuracy)
    return avg_accuracy


def main():
    for epochs in [100]:
        for max_seq_length in range(10, 21):
            print(
                f"Training model with EPOCHS={epochs} and MAX_SEQ_LENGTH={max_seq_length}"
            )
            logger.info(
                f"Training model with EPOCHS={epochs} and MAX_SEQ_LENGTH={max_seq_length}"
            )
            tf.random.set_seed(42)
            np.random.seed(42)
            random.seed(42)
            train_and_evaluate_model(epochs, max_seq_length)


if __name__ == "__main__":
    main()
