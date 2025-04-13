#!/bin/bash
set -e

echo "Setting up local testing environment..."

# Build the Docker image locally
echo "Building Docker image..."
docker build -t profitable-games-predictor:local .

# Create a temporary directory for test data
mkdir -p ./tmp

# Create a test script
cat > ./tmp/test_lambda.py << 'EOL'
import json
import sys
import os

# Add Lambda function code path
sys.path.append('/var/task')
import lambda_function

# Replace Lambda function AWS clients with simple mocks
def mock_download_model():
    print('Creating test model...')
    
    # Create a simple model
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, 9), return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Save model to a temp location
    model_path = "/tmp/model.h5"
    model.save(model_path)
    print(f"Saved test model to {model_path}")
    return model_path

def mock_fetch_games():
    print('Mocking game data fetching')
    import pandas as pd
    return pd.DataFrame({
        'game_id': ['game1', 'game2', 'game3'],
        'home_team': ['Team A', 'Team B', 'Team C'],
        'away_team': ['Team D', 'Team E', 'Team F'],
        'game_date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'home_win_odds': [2.0, 1.5, 3.0],
        'draw_odds': [3.5, 3.2, 3.8],
        'away_win_odds': [2.8, 5.0, 1.8],
        'home_team_rank': [5, 2, 10],
        'away_team_rank': [8, 15, 3],
        'home_goals_scored_avg': [1.5, 2.2, 0.8],
        'home_goals_conceded_avg': [0.8, 0.5, 1.2],
        'away_goals_scored_avg': [1.2, 0.7, 2.0],
        'away_goals_conceded_avg': [1.0, 1.8, 0.6]
    })

def mock_save_to_dynamodb(games):
    print(f'Mocking save to DynamoDB: {len(games)} games')
    for game in games:
        print(f"  Game: {game['home_team']} vs {game['away_team']}, Prediction: {game['prediction']}")
    return True

# Create a mock prediction model that will return high probabilities to trigger profitable predictions
def mock_make_predictions(model, X, games_df):
    print('Mocking predictions with profitable bets')
    # Create predictions with high probabilities for home win
    import numpy as np
    predictions = np.zeros((len(games_df), 3))
    predictions[:, 0] = 0.8  # High home win probability
    predictions[:, 1] = 0.1  # Low draw probability
    predictions[:, 2] = 0.1  # Low away win probability
    
    # Calculate expected value for each outcome
    games_df['home_win_ev'] = predictions[:, 0] * games_df['home_win_odds'] - 1
    games_df['draw_ev'] = predictions[:, 1] * games_df['draw_odds'] - 1
    games_df['away_win_ev'] = predictions[:, 2] * games_df['away_win_odds'] - 1
    
    # Find profitable bets (expected value > threshold)
    threshold = float(os.environ.get('THRESHOLD', '0.02'))
    profitable_games = []
    
    for i, row in games_df.iterrows():
        game_id = row.get('game_id', str(i))
        home_team = row.get('home_team', 'Unknown')
        away_team = row.get('away_team', 'Unknown')
        game_date = row.get('game_date', 'Unknown')
        
        # Check each outcome for profitability
        if row['home_win_ev'] > threshold:
            profitable_games.append({
                'id': game_id,
                'prediction': 'Home Win',
                'home_team': home_team,
                'away_team': away_team,
                'game_date': game_date,
                'odds': float(row['home_win_odds']),
                'probability': float(predictions[i, 0]),
                'expected_value': float(row['home_win_ev'])
            })
    
    print(f"Found {len(profitable_games)} profitable predictions")
    return profitable_games

# Replace the AWS-dependent functions with our mocks
lambda_function.download_model_from_s3 = mock_download_model
lambda_function.fetch_games_from_athena = mock_fetch_games
lambda_function.save_to_dynamodb = mock_save_to_dynamodb
lambda_function.make_predictions = mock_make_predictions

# Run the handler
print('\nRunning lambda handler...')
event = {}
context = {}
result = lambda_function.lambda_handler(event, context)
print(f'\nResult: {result}')
EOL

# Run the local test
echo "Running local test with Docker..."
docker run -v "$(pwd)/tmp:/tmp" \
  -e AWS_REGION=us-east-1 \
  -e S3_BUCKET=test-bucket \
  -e MODEL_TYPE=lstm \
  -e EPOCHS=100 \
  -e MAX_SEQ=12 \
  -e THRESHOLD=0.02 \
  -e ATHENA_DATABASE=test-db \
  -e DYNAMODB_TABLE=test-table \
  --entrypoint python \
  profitable-games-predictor:local /tmp/test_lambda.py

echo "Local test completed. Check the output above for results."

# Clean up
rm -rf ./tmp
echo "Cleaned up temporary files." 