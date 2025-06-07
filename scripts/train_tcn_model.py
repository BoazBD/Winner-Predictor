import logging
import os
from typing import Tuple

import awswrangler as wr
import boto3
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import (
    Conv1D, Dense, Dropout, GlobalAveragePooling1D, 
    Input, Add, BatchNormalization, Activation,
    LayerNormalization, GlobalMaxPooling1D, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

boto3.setup_default_session(region_name="il-central-1")

# Define constants and configurations
MIN_BET = 1
MAX_BET = 10
DAYS_TO_DISCARD = 60
COMPETITION_LEVEL = os.environ.get("COMPETITION_LEVEL", "all_games")
if COMPETITION_LEVEL not in ["all_games", "big_games"]:
    raise ValueError("COMPETITION_LEVEL must be either 'all_games' or 'big_games'")

BIG_GAMES = [
    "פרמייר ליג",
    "גביע ספרדי",
    "צרפתית ראשונה",
    "איטלקית ראשונה",
    "גביע אנגלי",
    "גביע המדינה",
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


def robust_feature_engineering(features: pd.DataFrame) -> np.ndarray:
    """Enhanced feature engineering with multiple representations."""
    # Convert to numpy for easier manipulation
    odds_array = features.values.astype(float)
    
    # 1. Normalized probabilities (original method)
    probs = 1.0 / odds_array
    prob_sums = probs.sum(axis=1, keepdims=True)
    normalized_probs = probs / prob_sums
    
    # 2. Log odds ratios (relative to most likely outcome)
    min_odds = np.min(odds_array, axis=1, keepdims=True)
    log_odds_ratios = np.log(odds_array / min_odds)
    
    # 3. Changes/momentum features
    odds_changes = np.diff(odds_array, axis=0, prepend=odds_array[0:1])
    prob_changes = np.diff(normalized_probs, axis=0, prepend=normalized_probs[0:1])
    
    # 4. Volatility features (rolling std)
    window = min(5, odds_array.shape[0])
    if odds_array.shape[0] >= window:
        odds_volatility = np.array([
            np.std(odds_array[max(0, i-window+1):i+1], axis=0) 
            for i in range(odds_array.shape[0])
        ])
    else:
        odds_volatility = np.zeros_like(odds_array)
    
    # Combine all features
    combined_features = np.concatenate([
        normalized_probs,
        log_odds_ratios,
        odds_changes,
        prob_changes,
        odds_volatility
    ], axis=1)
    
    # Handle any NaN/inf values
    combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return combined_features


def prepare_features(features: pd.DataFrame, max_seq_len: int) -> np.array:
    """Prepare features with robust engineering and proper padding."""
    features_engineered = robust_feature_engineering(features)
    
    # If sequence is shorter than max_seq_len, pad with earliest values
    if features_engineered.shape[0] < max_seq_len:
        padding_needed = max_seq_len - features_engineered.shape[0]
        # Use first value for padding (more stable than zeros)
        earliest_value = features_engineered[0:1, :]
        padding = np.repeat(earliest_value, padding_needed, axis=0)
        features_engineered = np.vstack([padding, features_engineered])
    else:
        # Take the last max_seq_len values
        features_engineered = features_engineered[-max_seq_len:, :]
    
    return features_engineered


def remove_consecutive_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove consecutive duplicate odds to reduce noise."""
    shifted = df[RATIOS].shift()
    mask = ~((df[RATIOS] == shifted).all(axis=1))
    return df[mask]


def process_ratios(group_df: pd.DataFrame) -> pd.DataFrame:
    """Process ratios with additional cleaning."""
    group_df = group_df.copy()
    group_df[RATIOS] = group_df[RATIOS].astype(float)
    
    # Remove obvious outliers (odds < 1.01 or > 100)
    for ratio in RATIOS:
        group_df = group_df[(group_df[ratio] >= 1.01) & (group_df[ratio] <= 100)]
    
    group_df = remove_consecutive_duplicates(group_df)
    return group_df


def prepare_temporal_data(
    grouped_data: pd.DataFrame, max_seq_length: int
) -> Tuple[np.array, np.array, np.array]:
    """Prepare data preserving temporal order."""
    x, y, timestamps = [], [], []
    
    for unique_id, group_df in grouped_data:
        group_df = process_ratios(group_df)
        if group_df.shape[0] < 5:  # Need minimum data points
            continue
            
        features = group_df[RATIOS]
        target = group_df[["bet1_won", "tie_won", "bet2_won"]].values[-1].astype(float)
        
        # Get the latest timestamp for temporal ordering
        latest_timestamp = group_df['run_time'].max()
        
        prepared_features = prepare_features(features, max_seq_length)
        x.append(prepared_features)
        y.append(target)
        timestamps.append(latest_timestamp)
    
    x = np.array(x)
    y = np.array(y)
    timestamps = np.array(timestamps)
    
    return x, y, timestamps


def improved_tcn_residual_block(x, filters: int, kernel_size: int, dilation_rate: int, 
                               dropout_rate: float, l1_reg: float, l2_reg: float):
    """Improved TCN residual block with better regularization."""
    # First conv layer with stronger regularization
    conv1 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
        use_bias=False  # Remove bias since we use batch norm
    )(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(dropout_rate)(conv1)
    
    # Second conv layer
    conv2 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
        use_bias=False
    )(conv1)
    conv2 = BatchNormalization()(conv2)
    
    # Residual connection with dimension matching
    if x.shape[-1] != filters:
        x = Conv1D(
            filters, 1, padding='same',
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
        )(x)
    
    # Add residual connection and final activation
    output = Add()([x, conv2])
    output = Activation('relu')(output)
    output = Dropout(dropout_rate * 0.5)(output)  # Lighter dropout after residual
    
    return output


def build_improved_tcn_model(
    learning_rate: float,
    dropout_rate: float,
    filters: int,
    kernel_size: int,
    l1_reg: float,
    l2_reg: float,
    max_seq_length: int,
    num_features: int = 15,  # 3 original + 12 engineered features
) -> Model:
    """Build improved TCN model with better architecture and regularization."""
    
    # Input layer
    inputs = Input(shape=(max_seq_length, num_features))
    
    # Initial feature transformation
    x = Conv1D(filters//2, 1, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate * 0.5)(x)
    
    # Gradually increase filters through TCN blocks
    filter_progression = [filters//2, filters//2, filters, filters]
    dilation_rates = [1, 2, 4, 8]
    
    for i, (dilation_rate, block_filters) in enumerate(zip(dilation_rates, filter_progression)):
        x = improved_tcn_residual_block(
            x, block_filters, kernel_size, dilation_rate, 
            dropout_rate, l1_reg, l2_reg
        )
    
    # Multiple pooling strategies for better representation
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    # Combine pooling strategies
    combined = Concatenate()([avg_pool, max_pool])
    
    # Dense layers with aggressive regularization
    dense1 = Dense(
        64, 
        activation='relu',
        kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
    )(combined)
    dense1 = LayerNormalization()(dense1)
    dense1 = Dropout(dropout_rate)(dense1)
    
    dense2 = Dense(
        32, 
        activation='relu',
        kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
    )(dense1)
    dense2 = LayerNormalization()(dense2)
    dense2 = Dropout(dropout_rate)(dense2)
    
    # Output layer
    outputs = Dense(3, activation='softmax')(dense2)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use custom optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,  # Gradient clipping
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'categorical_crossentropy']
    )
    
    return model


def temporal_cross_validation(X, y, timestamps, n_splits=5):
    """Time-based cross-validation that respects temporal order."""
    # Sort by timestamp
    sort_idx = np.argsort(timestamps)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    timestamps_sorted = timestamps[sort_idx]
    
    # Use TimeSeriesSplit for temporal validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for train_idx, val_idx in tscv.split(X_sorted):
        yield train_idx, val_idx, X_sorted, y_sorted


def objective(trial: optuna.trial.Trial, epochs: int, max_seq_length: int) -> float:
    """Improved Optuna objective with temporal validation."""
    
    # More conservative hyperparameter ranges
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.6)
    filters = trial.suggest_categorical("filters", [16, 32, 48])  # Smaller models
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    l1_reg = trial.suggest_float("l1_reg", 1e-6, 1e-3, log=True)
    l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True)

    # Load and prepare data
    last_week_start = pd.Timestamp.now().date() - pd.Timedelta(days=DAYS_TO_DISCARD)
    processed = pd.read_parquet("processed_winner.parquet")
    processed = processed[processed["type"] == "Soccer"]
    processed = processed[pd.to_datetime(processed["date_parsed"]).dt.date < last_week_start]

    if COMPETITION_LEVEL == "big_games":
        processed = processed[processed["league"].isin(BIG_GAMES)]
    
    # Sort by run_time for temporal consistency
    processed = processed.sort_values('run_time')
    grouped_processed = list(processed.groupby(["unique_id"]))
    
    X, y, timestamps = prepare_temporal_data(grouped_processed, max_seq_length)
    
    if len(X) < 50:  # Need minimum samples
        return 0.0

    # Temporal cross-validation
    fold_scores = []
    fold_count = 0
    
    for train_idx, val_idx, X_sorted, y_sorted in temporal_cross_validation(X, y, timestamps, n_splits=3):
        if len(val_idx) < 10:  # Skip folds with too few validation samples
            continue
            
        X_train, X_val = X_sorted[train_idx], X_sorted[val_idx]
        y_train, y_val = y_sorted[train_idx], y_sorted[val_idx]

        # Build model
        model = build_improved_tcn_model(
            learning_rate, dropout_rate, filters, kernel_size, 
            l1_reg, l2_reg, max_seq_length
        )
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=min(epochs, 30),  # Cap epochs for hyperparameter search
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        # Get best validation score
        best_val_loss = min(history.history['val_loss'])
        fold_scores.append(-best_val_loss)  # Negative because we minimize loss
        fold_count += 1
        
        if fold_count >= 3:  # Limit folds for faster optimization
            break

    if not fold_scores:
        return 0.0
        
    avg_score = np.mean(fold_scores)
    return avg_score


def train_and_evaluate_improved_tcn(epochs: int, max_seq_length: int):
    """Train improved TCN with temporal validation and ensemble."""
    
    print(f"Starting improved TCN training with EPOCHS={epochs} and MAX_SEQ_LENGTH={max_seq_length}")
    logger.info(f"Starting improved TCN training with EPOCHS={epochs} and MAX_SEQ_LENGTH={max_seq_length}")
    
    # Hyperparameter optimization with fewer trials for faster iteration
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: objective(trial, epochs, max_seq_length), 
        n_trials=20,  # Increased from 10 for better optimization
        timeout=3600  # 1 hour timeout
    )

    print(f"Best trial score: {study.best_trial.value:.4f}")
    print(f"Best hyperparameters: {study.best_trial.params}")

    # Load data for final training
    best_params = study.best_trial.params
    
    last_week_start = pd.Timestamp.now().date() - pd.Timedelta(days=DAYS_TO_DISCARD)
    processed = pd.read_parquet("processed_winner.parquet")
    processed = processed[processed["type"] == "Soccer"]
    processed = processed[pd.to_datetime(processed["date_parsed"]).dt.date < last_week_start]
    
    if COMPETITION_LEVEL == "big_games":
        processed = processed[processed["league"].isin(BIG_GAMES)]
    
    processed = processed.sort_values('run_time')
    grouped_processed = list(processed.groupby(["unique_id"]))
    X, y, timestamps = prepare_temporal_data(grouped_processed, max_seq_length)

    # Train ensemble of models for better generalization
    ensemble_models = []
    n_ensemble = 3
    
    for i in range(n_ensemble):
        print(f"Training ensemble model {i+1}/{n_ensemble}")
        
        # Add slight randomization to hyperparameters for diversity
        varied_params = best_params.copy()
        if i > 0:
            varied_params["dropout_rate"] = min(0.8, varied_params["dropout_rate"] * (1 + 0.1 * np.random.randn()))
            varied_params["learning_rate"] = varied_params["learning_rate"] * (1 + 0.1 * np.random.randn())
        
        model = build_improved_tcn_model(
            varied_params["learning_rate"],
            varied_params["dropout_rate"],
            varied_params["filters"],
            varied_params["kernel_size"],
            varied_params["l1_reg"],
            varied_params["l2_reg"],
            max_seq_length,
        )
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='loss',
                patience=15,
                restore_best_weights=True,
                min_delta=1e-5
            ),
            ReduceLROnPlateau(
                monitor='loss',
                factor=0.3,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train with more sophisticated approach
        model.fit(
            X, y,
            epochs=epochs,
            batch_size=varied_params["batch_size"],
            callbacks=callbacks,
            validation_split=0.15,  # Hold out some data for monitoring
            verbose=1 if i == 0 else 0
        )
        
        ensemble_models.append(model)
    
    # Save ensemble models
    os.makedirs("trained_models", exist_ok=True)
    for i, model in enumerate(ensemble_models):
        model_filename = f"trained_models/improved_tcn_{epochs}_{max_seq_length}_v2_ensemble_{i}.h5"
        model.save(model_filename)
        print(f"Ensemble model {i} saved as: {model_filename}")
    
    # Save hyperparameters and ensemble info
    params_filename = f"trained_models/improved_tcn_{epochs}_{max_seq_length}_v2_params.txt"
    with open(params_filename, 'w') as f:
        f.write(f"Best optimization score: {study.best_trial.value:.4f}\n")
        f.write(f"Ensemble size: {n_ensemble}\n")
        f.write("Best hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\nImprovements made:\n")
        f.write("- Temporal cross-validation\n")
        f.write("- Enhanced feature engineering\n")
        f.write("- L1+L2 regularization\n")
        f.write("- Ensemble modeling\n")
        f.write("- Gradient clipping\n")
        f.write("- Layer normalization\n")
        f.write("- Multiple pooling strategies\n")
    print(f"Parameters and info saved as: {params_filename}")


def main():
    """Main training loop with improved methodology."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # More conservative training configurations
    epochs_list = [50, 100]  # Increased epochs since we have better regularization
    max_seq_lengths = [8, 12]  # Test different sequence lengths
    
    for epochs in epochs_list:
        for max_seq_length in max_seq_lengths:
            print(f"\n{'='*80}")
            print(f"Training IMPROVED TCN model with EPOCHS={epochs} and MAX_SEQ_LENGTH={max_seq_length}")
            print(f"Competition level: {COMPETITION_LEVEL}")
            print(f"Key improvements:")
            print(f"- Temporal cross-validation")
            print(f"- Enhanced feature engineering (15 features)")
            print(f"- L1+L2 regularization")
            print(f"- Ensemble modeling")
            print(f"- Better architecture")
            print(f"{'='*80}")
            
            logger.info(
                f"Training improved TCN model with EPOCHS={epochs} and MAX_SEQ_LENGTH={max_seq_length}"
            )
            
            try:
                train_and_evaluate_improved_tcn(epochs, max_seq_length)
                print(f"Successfully completed training for epochs={epochs}, seq_length={max_seq_length}")
            except Exception as e:
                print(f"Error training model with epochs={epochs}, seq_length={max_seq_length}: {str(e)}")
                logger.error(f"Error training model: {str(e)}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main() 
