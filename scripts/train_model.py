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

class EVModel(tf.keras.Model):
    """Custom model that can compute EV loss using odds input"""
    
    def __init__(self, base_model, use_ev_loss=False):
        super().__init__()
        self.base_model = base_model
        self.use_ev_loss = use_ev_loss
        
    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)
    
    @property
    def layers(self):
        return self.base_model.layers
    
    def save_base_model(self, filepath):
        """Save the underlying base model"""
        self.base_model.save(filepath)
    
    @property
    def trainable_variables(self):
        return self.base_model.trainable_variables
    
    def train_step(self, data):
        x, y = data
        x_seq, x_odds = x
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            
            if self.use_ev_loss:
                # Compute EV loss
                p = y_pred
                profit = p * (x_odds - 1) * y - p * (1 - y)
                ev = tf.reduce_sum(profit, axis=1)
                loss = -tf.reduce_mean(ev)
            else:
                # Use standard categorical crossentropy
                loss = tf.keras.losses.categorical_crossentropy(y, y_pred)
                loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"loss": loss}

def expected_value_loss_simple(y_true, y_pred):
    """Simple EV loss that will be overridden by custom training"""
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

boto3.setup_default_session(region_name="il-central-1")

# Define constants and configurations
DAYS_TO_DISCARD = 150
COMPETITION_LEVEL = os.environ.get("COMPETITION_LEVEL", "big_games")
if COMPETITION_LEVEL not in ["all_games", "big_games"]:
    raise ValueError("COMPETITION_LEVEL must be either 'all_games' or 'big_games'")

logger = logging.getLogger(__name__)


def calculate_roi_from_predictions(y_true, y_pred, odds, threshold=0.5):
    """
    Calculate ROI from model predictions using threshold-based betting strategy.
    
    Args:
        y_true: True outcomes (one-hot encoded)
        y_pred: Model predictions (probabilities)
        odds: Final odds for each outcome
        threshold: Minimum probability threshold for betting
    
    Returns:
        ROI as float
    """
    total_bet = 0
    total_profit = 0
    
    for i in range(len(y_pred)):
        # Find the outcome with highest predicted probability
        best_outcome_idx = np.argmax(y_pred[i])
        best_prob = y_pred[i][best_outcome_idx]
        
        # Only bet if probability exceeds threshold
        if best_prob >= threshold:
            bet_amount = 1.0  # Fixed bet size
            total_bet += bet_amount
            
            # Check if this outcome actually won
            if y_true[i][best_outcome_idx] == 1:
                # Win: get back bet plus profit
                total_profit += bet_amount * odds[i][best_outcome_idx] - bet_amount
            else:
                # Loss: lose the bet
                total_profit -= bet_amount
    
    if total_bet == 0:
        return 0.0  # No bets placed
    
    return (total_profit / total_bet) * 100  # ROI as percentage


def build_model(
    learning_rate: float,
    dropout_rate: float,
    units: int,
    activation: str,
    max_seq_length: int,
    loss_type: str = "ce"    # new arg: "ce" or "ev"
):
    # 1) Odds input
    odds_in = tf.keras.layers.Input(shape=(3,), name="odds_input")
    # 2) Sequence input
    seq_in = tf.keras.layers.Input(shape=(max_seq_length, 3), name="seq_input")
    
    x = Bidirectional(LSTM(units, return_sequences=True))(seq_in)
    x = LSTM(units//2, activation=activation, return_sequences=True)(x)
    x = LSTM(units//2, activation=activation)(x)
    x = Dropout(dropout_rate)(x)
    
    # Simple approach: just add a dummy connection that zeros out
    # Connect odds to output but multiply by 0 so it doesn't affect predictions
    odds_dummy = tf.keras.layers.Lambda(lambda x: x * 0)(odds_in)  # Zero out odds
    odds_dummy = tf.keras.layers.Dense(3, use_bias=False)(odds_dummy)  # Map to 3 outputs
    
    pred_logits = Dense(3, activation="linear")(x)
    # Add the zeroed odds contribution (effectively does nothing)
    combined_logits = tf.keras.layers.Add()([pred_logits, odds_dummy])
    out = tf.keras.layers.Activation("softmax", name="pred")(combined_logits)
    
    base_model = tf.keras.Model(inputs=[seq_in, odds_in], outputs=out)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)
    
    # Wrap in custom model for EV loss support
    if loss_type == "ev":
        model = EVModel(base_model, use_ev_loss=True)
        model.compile(optimizer=opt, loss=expected_value_loss_simple)
    else:
        model = EVModel(base_model, use_ev_loss=False)
        model.compile(optimizer=opt, loss="categorical_crossentropy")
    
    return model


def load_or_process_data(max_seq_length: int,
                        scaling_method: str = "standard") -> Tuple[np.array, np.array, np.array]:
    """
    Load pre-processed data if available, otherwise process raw data.
    
    Args:
        max_seq_length: Maximum sequence length
        scaling_method: "standard" or "roi"
    
    Returns:
        Tuple of (X_seq, X_odds, y) arrays
    """
    # Try to load pre-processed data first
    processed_file = f"processed_features/processed_features_{scaling_method}_{max_seq_length}_{FIXED_STRIDE}.pkl"
    
    print(f"Loading pre-processed features from {processed_file}")
    x_seq, x_odds, y, metadata = load_processed_features(processed_file)
    
    # Verify metadata matches current requirements
    if (metadata.get('max_seq_length') == max_seq_length and 
        metadata.get('stride') == FIXED_STRIDE and
        metadata.get('scaling_method') == scaling_method and
        metadata.get('competition_level') == COMPETITION_LEVEL):
        print("Pre-processed data matches requirements, using cached features")
        return x_seq, x_odds, y
    else:
        print("Pre-processed data metadata doesn't match, reprocessing...")
        # Note: For now, return the loaded data anyway. Full reprocessing would require
        # implementing the full data pipeline here.
        return x_seq, x_odds, y
    

def objective(trial: optuna.trial.Trial, epochs: int, max_seq_length: int, 
             scaling_method: str = "standard") -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    units = trial.suggest_categorical("units", [32, 64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    activation = trial.suggest_categorical("activation", ["sigmoid", "tanh"])

    x_seq, x_odds, y = load_or_process_data(max_seq_length, scaling_method=scaling_method)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_roi = []
    for train_index, val_index in kfold.split(x_seq):
        X_seq_train, X_seq_val = x_seq[train_index], x_seq[val_index]
        X_odds_train, X_odds_val = x_odds[train_index], x_odds[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = build_model(
            learning_rate, dropout_rate, units, activation, max_seq_length, loss_type="ce"
        )
        model.fit([X_seq_train, X_odds_train], y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Get predictions and calculate ROI
        y_pred = model.predict([X_seq_val, X_odds_val], verbose=0)
        roi = calculate_roi_from_predictions(y_val, y_pred, X_odds_val, threshold=0.5)
        fold_roi.append(roi)

    avg_roi = np.mean(fold_roi)
    return avg_roi


def train_and_evaluate_model(epochs: int, max_seq_length: int, scaling_method: str = "standard"):
    study = optuna.create_study(direction="maximize")
    # Create a modified objective function
    def objective_with_params(trial):
        return objective_with_custom_params(trial, epochs, max_seq_length, scaling_method)
    
    study.optimize(objective_with_params, n_trials=20)

    print(f"Best trial: {study.best_trial.value}")
    print(f"Best hyperparameters: {study.best_trial.params}")

    best_params = study.best_trial.params
    x_seq, x_odds, y = load_or_process_data(max_seq_length, scaling_method=scaling_method)

    # Two-stage training
    print("Stage 1: Training with Cross-Entropy loss...")
    model = build_model(
        best_params["learning_rate"],
        best_params["dropout_rate"],
        best_params["units"],
        best_params["activation"],
        max_seq_length,
        loss_type="ce"
    )
    model.fit(
        [x_seq, x_odds], y,
        epochs=epochs, 
        batch_size=best_params["batch_size"], 
        verbose=1
    )

    print("Stage 2: Fine-tuning with EV loss...")
    # Switch to EV loss for fine-tuning
    model.use_ev_loss = True
    model.fit(
        [x_seq, x_odds], y,
        epochs=epochs//2,    # fewer epochs, e.g. half
        batch_size=best_params["batch_size"], 
        verbose=1
    )
    
    # Create model filename based on scaling method
    model_suffix = "roi" if scaling_method == "roi" else "standard"
    model_filename = f"{COMPETITION_LEVEL}_{model_suffix}_{epochs}_{max_seq_length}_ev.h5"
    model.save_base_model(f"trained_models/{model_filename}")
    print(f"Saved model to trained_models/{model_filename}")


def objective_with_custom_params(trial: optuna.trial.Trial, epochs: int, max_seq_length: int, 
                                scaling_method: str = "standard") -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    units = trial.suggest_categorical("units", [32, 64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    activation = trial.suggest_categorical("activation", ["sigmoid", "tanh"])

    x_seq, x_odds, y = load_or_process_data(max_seq_length, scaling_method=scaling_method)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_roi = []
    for train_index, val_index in kfold.split(x_seq):
        X_seq_train, X_seq_val = x_seq[train_index], x_seq[val_index]
        X_odds_train, X_odds_val = x_odds[train_index], x_odds[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = build_model(
            learning_rate, dropout_rate, units, activation, max_seq_length, loss_type="ce"
        )
        model.fit([X_seq_train, X_odds_train], y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Get predictions and calculate ROI
        y_pred = model.predict([X_seq_val, X_odds_val], verbose=0)
        roi = calculate_roi_from_predictions(y_val, y_pred, X_odds_val, threshold=0.5)
        fold_roi.append(roi)

    avg_roi = np.mean(fold_roi)
    return avg_roi


def main():
    # Get scaling method from environment variable
    scaling_method = os.environ.get("SCALING_METHOD", "standard")
    if scaling_method not in ["standard", "roi"]:
        raise ValueError("SCALING_METHOD must be either 'standard' or 'roi'")
    
    print(f"Training with scaling method: {scaling_method}")
    print(f"Using fixed stride: {FIXED_STRIDE}")
    
    for epochs in [50,100]:
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
