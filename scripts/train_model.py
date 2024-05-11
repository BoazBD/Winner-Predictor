import os
from typing import Tuple

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

MAX_SEQ_LENGTH = 500
MIN_BET = 1
MAX_BET = 10
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


def scale_features(features: pd.DataFrame) -> pd.DataFrame:
    return (features - MIN_BET) / (MAX_BET - 1)


def pad_features(features: np.array) -> np.array:
    padding_length = MAX_SEQ_LENGTH - features.shape[0]
    if padding_length > 0:
        padded_features = np.zeros((MAX_SEQ_LENGTH, features.shape[1]))
        padded_features[padding_length:, :] = features
        features = padded_features
    else:
        features = features[-MAX_SEQ_LENGTH:, :]
    return features


def prepare_features(features: pd.DataFrame) -> np.array:
    features_scaled = scale_features(features)
    features_scaled = features_scaled.to_numpy()
    features_scaled_padded = pad_features(features_scaled)
    return features_scaled_padded


def prepare_data_for_model(grouped_data: pd.DataFrame) -> Tuple[np.array, np.array]:
    x, y = [], []
    for _, group_df in grouped_data:
        features = group_df[["ratio1", "ratio2", "ratio3"]].astype(float)
        target = group_df[["bet1_won", "bet2_won", "tie_won"]].values[-1].astype(float)

        prepared_features = prepare_features(features)
        x.append(prepared_features)
        y.append(target)

    x = np.array(x)
    y = np.array(y)
    return x, y


def train_model(
    x: pd.DataFrame,
    y: pd.DataFrame,
    learning_rate: float,
    dropout_rate: float,
    units: int,
):
    model = Sequential()
    model.add(
        LSTM(
            units,
            return_sequences=True,
            input_shape=(x.shape[1], x.shape[2]),
        )
    )
    model.add(LSTM(units // 2, activation="sigmoid", return_sequences=True))
    model.add(LSTM(units // 2, activation="sigmoid"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    model.fit(x, y, epochs=1, batch_size=1)
    return model


def main():
    learning_rates = [0.001, 0.005, 0.01]
    dropout_rates = [0.2, 0.3]
    layer_units = [32, 64]
    batch_sizes = [1]  # [32, 64, 128]

    max_accuracy = 0
    best_lr = None
    best_dropout_rate = None
    best_units = None
    best_model = None

    last_week_start = pd.Timestamp.now().date() - pd.Timedelta(days=7)

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

    for learning_rate in learning_rates:
        for dropout_rate in dropout_rates:
            for units in layer_units:
                for batch_size in batch_sizes:
                    print(
                        f"Training model for {MODEL_TYPE} with hyperparameters: "
                        f"learning_rate={learning_rate}, dropout_rate={dropout_rate}, "
                        f"units={units}, batch_size={batch_size}"
                    )
                    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                    fold_accuracy = []
                    for train_index, val_index in kfold.split(x):
                        X_train, X_val = x[train_index], x[val_index]
                        y_train, y_val = y[train_index], y[val_index]

                        model = train_model(
                            X_train, y_train, learning_rate, dropout_rate, units
                        )
                        loss, val_accuracy = model.evaluate(X_val, y_val)
                        fold_accuracy.append(val_accuracy)

                    avg_accuracy = np.mean(fold_accuracy)
                    print(
                        f"Average Accuracy for {learning_rate}, {dropout_rate}, {units}, {batch_size}: {avg_accuracy:.4f}"
                    )
                    if avg_accuracy > max_accuracy:
                        max_accuracy = avg_accuracy
                        best_lr = learning_rate
                        best_dropout_rate = dropout_rate
                        best_units = units
                        best_model = model

    print(
        f"Best model has accuracy of {max_accuracy} with learning rate {best_lr}, dropout rate {best_dropout_rate} and units {best_units}"
    )
    # Train best model on all data
    best_model = train_model(x, y, best_lr, best_dropout_rate, best_units)
    best_model.save(f"trained_models/{MODEL_TYPE}_model.h5")


if __name__ == "__main__":
    main()
