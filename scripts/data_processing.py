import logging
import os
from typing import Tuple, Optional, Dict, Any
import pickle

import numpy as np
import pandas as pd

# Define constants
MIN_BET = 1
MAX_BET = 10
RATIOS = ["ratio1", "ratio2", "ratio3"]
FIXED_STRIDE = 8  # Fixed stride for all operations

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
    "אליפות העולם לקבוצות",
    "גביע הטוטו",
]

logger = logging.getLogger(__name__)


def scale_features_standard(features: pd.DataFrame) -> pd.DataFrame:
    """Standard scaling used in original training."""
    return (features - MIN_BET) / (MAX_BET - 1)


def scale_features_roi(features: pd.DataFrame) -> pd.DataFrame:
    """ROI scaling: convert odds to implied probabilities (1/odds)."""
    return 1.0 / features.astype(float)


def remove_consecutive_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove consecutive rows with identical ratios."""
    shifted = df[RATIOS].shift()
    mask = ~((df[RATIOS] == shifted).all(axis=1))
    return df[mask]


def process_ratios(group_df: pd.DataFrame) -> pd.DataFrame:
    """Process ratios by converting to float and removing consecutive duplicates."""
    group_df[RATIOS] = group_df[RATIOS].astype(float)
    group_df = remove_consecutive_duplicates(group_df)
    return group_df


def prepare_features_sliding_window(features: pd.DataFrame, max_seq_len: int, 
                                   scaling_method: str = "standard") -> list:
    """Generate sliding windows of features with fixed stride."""
    if scaling_method == "standard":
        features_scaled = scale_features_standard(features)
    elif scaling_method == "roi":
        features_scaled = scale_features_roi(features)
    else:
        raise ValueError("scaling_method must be 'standard' or 'roi'")
    
    features_scaled = features_scaled.to_numpy()
    
    windows = []
    for start_idx in range(0, len(features_scaled) - max_seq_len + 1, FIXED_STRIDE):
        end_idx = start_idx + max_seq_len
        window = features_scaled[start_idx:end_idx, :]
        windows.append(window)
    
    return windows


def prepare_features_single(features: pd.DataFrame, max_seq_len: int, 
                           scaling_method: str = "standard") -> np.array:
    """Prepare features for a single sequence (used in prediction)."""
    if scaling_method == "standard":
        features_scaled = scale_features_standard(features)
    elif scaling_method == "roi":
        features_scaled = scale_features_roi(features)
    else:
        raise ValueError("scaling_method must be 'standard' or 'roi'")
    
    features_scaled = features_scaled.to_numpy()
    features_scaled = features_scaled[-max_seq_len:, :]
    return features_scaled


def prepare_data_for_model(grouped_data: pd.DataFrame, max_seq_length: int, 
                          scaling_method: str = "standard") -> Tuple[np.array, np.array, np.array]:
    """
    Prepare data for training/validation with fixed stride.
    
    Args:
        grouped_data: Grouped DataFrame (result of groupby)
        max_seq_length: Maximum sequence length
        scaling_method: "standard" or "roi"
    
    Returns:
        Tuple of (X_seq, X_odds, y) arrays
    """
    x_seq, x_odds, y = [], [], []
    total_games = 0
    total_windows = 0
    
    for _, group_df in grouped_data:
        group_df = process_ratios(group_df)
        if group_df.shape[0] < max_seq_length:
            continue
        
        total_games += 1
        features = group_df[RATIOS]
        target = group_df[["bet1_won", "tie_won", "bet2_won"]].values[-1].astype(float)
        
        # Get final odds (last row of original ratios, not scaled)
        final_odds = features.iloc[-1].values.astype(float)  # Shape: (3,)

        # Generate sliding windows for this game with fixed stride
        sliding_windows = prepare_features_sliding_window(features, max_seq_length, scaling_method)
        total_windows += len(sliding_windows)
        
        # Add each window as a separate training sample with the same target and final odds
        for window in sliding_windows:
            x_seq.append(window)
            x_odds.append(final_odds)
            y.append(target)
    
    x_seq = np.array(x_seq)
    x_odds = np.array(x_odds)
    y = np.array(y)
    
    print(f"Generated {total_windows} sliding windows from {total_games} games")
    print(f"Average windows per game: {total_windows / total_games if total_games > 0 else 0:.2f}")
    print(f"Training data shape: X_seq={x_seq.shape}, X_odds={x_odds.shape}, y={y.shape}")
    print(f"Using fixed stride: {FIXED_STRIDE}")
    
    return x_seq, x_odds, y


def save_processed_features(x_seq: np.array, x_odds: np.array, y: np.array, output_path: str, metadata: Dict[str, Any] = None):
    """Save processed features and metadata to file."""
    data = {
        'X_seq': x_seq,
        'X_odds': x_odds,
        'y': y,
        'metadata': metadata or {}
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved processed features to {output_path}")
    print(f"X_seq shape: {x_seq.shape}, X_odds shape: {x_odds.shape}, y shape: {y.shape}")


def load_processed_features(input_path: str) -> Tuple[np.array, np.array, np.array, Dict[str, Any]]:
    """Load processed features and metadata from file."""
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded processed features from {input_path}")
    
    # Handle backward compatibility for old format
    if 'X_seq' in data and 'X_odds' in data:
        print(f"X_seq shape: {data['X_seq'].shape}, X_odds shape: {data['X_odds'].shape}, y shape: {data['y'].shape}")
        return data['X_seq'], data['X_odds'], data['y'], data['metadata']
    else:
        # Old format - create dummy odds (will need reprocessing)
        print("Old format detected - X_odds not available, returning zeros")
        print(f"X shape: {data['X'].shape}, y shape: {data['y'].shape}")
        x_odds_dummy = np.ones((data['X'].shape[0], 3))  # Dummy odds
        return data['X'], x_odds_dummy, data['y'], data['metadata']


def process_and_save_features(input_parquet_path: str, output_path: str, 
                             max_seq_length: int,
                             scaling_method: str = "standard",
                             competition_level: str = "big_games",
                             days_to_discard: int = 120):
    """
    Main function to process raw data and save features with fixed stride.
    
    Args:
        input_parquet_path: Path to the processed parquet file
        output_path: Path to save the processed features
        max_seq_length: Maximum sequence length
        scaling_method: "standard" or "roi"
        competition_level: "all_games" or "big_games"
        days_to_discard: Days to discard from the end for validation
    """
    print(f"Processing data from {input_parquet_path}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Fixed stride: {FIXED_STRIDE}")
    print(f"Scaling method: {scaling_method}")
    print(f"Competition level: {competition_level}")
    
    # Load and filter data
    last_week_start = pd.Timestamp.now().date() - pd.Timedelta(days=days_to_discard)
    
    processed = pd.read_parquet(input_parquet_path)
    processed = processed[processed["type"] == "Soccer"]
    processed = processed[pd.to_datetime(processed["date_parsed"]).dt.date < last_week_start]
    
    if competition_level == "big_games":
        processed = processed[processed["league"].isin(BIG_GAMES)]
    
    print(f"Loaded {len(processed)} rows after filtering")
    
    # Group and process
    grouped_processed = list(processed.sort_values(by="run_time").groupby(["unique_id"]))
    print(f"Found {len(grouped_processed)} unique games")
    
    # Prepare features
    x_seq, x_odds, y = prepare_data_for_model(grouped_processed, max_seq_length, scaling_method)
    
    # Create metadata
    metadata = {
        'max_seq_length': max_seq_length,
        'stride': FIXED_STRIDE,
        'scaling_method': scaling_method,
        'competition_level': competition_level,
        'days_to_discard': days_to_discard,
        'input_file': input_parquet_path,
        'total_games': len(grouped_processed),
        'total_windows': len(x_seq)
    }
    
    # Save processed features
    save_processed_features(x_seq, x_odds, y, output_path, metadata)


def main():
    """Process data with different configurations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process betting data into features')
    parser.add_argument('--input', required=True, help='Input parquet file path')
    parser.add_argument('--max_seq_length', type=int, default=12, help='Maximum sequence length')
    parser.add_argument('--scaling_method', choices=['standard', 'roi'], default='standard', help='Scaling method')
    parser.add_argument('--competition_level', choices=['all_games', 'big_games'], default='big_games', help='Competition level')
    parser.add_argument('--days_to_discard', type=int, default=120, help='Days to discard from end')
    
    args = parser.parse_args()
    output_path = f"processed_features/processed_features_{args.scaling_method}_{args.max_seq_length}_{FIXED_STRIDE}.pkl"
    
    process_and_save_features(
        input_parquet_path=args.input,
        output_path=output_path,
        max_seq_length=args.max_seq_length,
        scaling_method=args.scaling_method,
        competition_level=args.competition_level,
        days_to_discard=args.days_to_discard
    )


if __name__ == "__main__":
    main() 