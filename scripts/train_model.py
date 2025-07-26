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

# Import centralized data processing functions
from data_processing import (
    FIXED_STRIDE,
    load_processed_features,
)

boto3.setup_default_session(region_name="il-central-1")

# Define constants and configurations
DAYS_TO_DISCARD = 150
COMPETITION_LEVEL = os.environ.get("COMPETITION_LEVEL", "big_games")
if COMPETITION_LEVEL not in ["all_games", "big_games"]:
    raise ValueError("COMPETITION_LEVEL must be either 'all_games' or 'big_games'")

logger = logging.getLogger(__name__)


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


def load_or_process_data(max_seq_length: int,
                        scaling_method: str = "standard") -> Tuple[np.array, np.array]:
    """
    Load pre-processed data if available, otherwise process raw data.
    
    Args:
        max_seq_length: Maximum sequence length
        use_preprocessed: Whether to try loading pre-processed data first
        scaling_method: "standard" or "roi"
    
    Returns:
        Tuple of (X, y) arrays
    """
    # Try to load pre-processed data first
    processed_file = f"processed_features/processed_features_{scaling_method}_{max_seq_length}_{FIXED_STRIDE}.pkl"
    
    print(f"Loading pre-processed features from {processed_file}")
    x, y, metadata = load_processed_features(processed_file)
    
    # Verify metadata matches current requirements
    if (metadata.get('max_seq_length') == max_seq_length and 
        metadata.get('stride') == FIXED_STRIDE and
        metadata.get('scaling_method') == scaling_method and
        metadata.get('competition_level') == COMPETITION_LEVEL):
        print("Pre-processed data matches requirements, using cached features")
        return x, y
    else:
        print("Pre-processed data metadata doesn't match, reprocessing...")
    

def objective(trial: optuna.trial.Trial, epochs: int, max_seq_length: int, 
             scaling_method: str = "standard") -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    units = trial.suggest_categorical("units", [32, 64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    activation = trial.suggest_categorical("activation", ["sigmoid", "tanh"])

    x, y = load_or_process_data(max_seq_length, scaling_method=scaling_method)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracy = []
    for train_index, val_index in kfold.split(x):
        X_train, X_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = build_model(
            learning_rate, dropout_rate, units, activation, max_seq_length
        )
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracy.append(val_accuracy)

    avg_accuracy = np.mean(fold_accuracy)
    return avg_accuracy


def train_and_evaluate_model(epochs: int, max_seq_length: int, scaling_method: str = "standard"):
    study = optuna.create_study(direction="maximize")
    # Create a modified objective function
    def objective_with_params(trial):
        return objective_with_custom_params(trial, epochs, max_seq_length, scaling_method)
    
    study.optimize(objective_with_params, n_trials=20)

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

    x, y = load_or_process_data(max_seq_length, scaling_method=scaling_method)

    best_model.fit(x, y, epochs=epochs, batch_size=best_params["batch_size"], verbose=0)
    
    # Create model filename based on scaling method
    model_suffix = "roi" if scaling_method == "roi" else "standard"
    model_filename = f"{COMPETITION_LEVEL}_{model_suffix}_{epochs}_{max_seq_length}_n1.h5"
    best_model.save(f"trained_models/{model_filename}")
    print(f"Saved model to trained_models/{model_filename}")


def objective_with_custom_params(trial: optuna.trial.Trial, epochs: int, max_seq_length: int, 
                                scaling_method: str = "standard") -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    units = trial.suggest_categorical("units", [32, 64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    activation = trial.suggest_categorical("activation", ["sigmoid", "tanh"])

    x, y = load_or_process_data(max_seq_length, scaling_method=scaling_method)

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
    # Get scaling method from environment variable
    scaling_method = os.environ.get("SCALING_METHOD", "standard")
    if scaling_method not in ["standard", "roi"]:
        raise ValueError("SCALING_METHOD must be either 'standard' or 'roi'")
    
    print(f"Training with scaling method: {scaling_method}")
    print(f"Using fixed stride: {FIXED_STRIDE}")
    
    for epochs in [50]:
        for max_seq_length in [12, 18]:
            print(
                f"Training model with EPOCHS={epochs}, MAX_SEQ_LENGTH={max_seq_length}, SCALING={scaling_method}, STRIDE={FIXED_STRIDE}"
            )
            logger.info(
                f"Training model with EPOCHS={epochs}, MAX_SEQ_LENGTH={max_seq_length}, SCALING={scaling_method}, STRIDE={FIXED_STRIDE}"
            )
            tf.random.set_seed(42)
            np.random.seed(42)
            random.seed(42)
            train_and_evaluate_model(epochs, max_seq_length, scaling_method=scaling_method)


if __name__ == "__main__":
    main()
