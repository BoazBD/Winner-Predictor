import os
import json
import boto3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import awswrangler as wr
from datetime import datetime, timedelta, date
import logging
from typing import Tuple, List, Dict, Any
from decimal import Decimal

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
    "ליגת Winner",
]

# Custom InputLayer class to handle batch_shape parameter
class CustomInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            # Convert batch_shape to input_shape for compatibility
            kwargs['input_shape'] = kwargs.pop('batch_shape')[1:]
        super(CustomInputLayer, self).__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            # Convert batch_shape to input_shape for compatibility
            config['input_shape'] = config.pop('batch_shape')[1:]
        return cls(**config)

# Register the custom layer for model loading
tf.keras.utils.get_custom_objects().update({'CustomInputLayer': CustomInputLayer})

# Check if running locally or in Lambda
IS_LOCAL = os.environ.get('IS_LOCAL', 'False').lower() == 'true'

# Environment variables with defaults
AWS_REGION = os.environ.get('AWS_REGION', 'il-central-1')
ATHENA_DATABASE = os.environ.get('ATHENA_DATABASE', 'winner-db')
ALL_PREDICTIONS_TABLE = os.environ.get('ALL_PREDICTIONS_TABLE', 'all-predicted-games')
PROFITABLE_GAMES_TABLE = os.environ.get('PROFITABLE_GAMES_TABLE', 'profitable-games')
S3_BUCKET = os.environ.get('S3_BUCKET', 'winner-site-data')
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'lstm')
EPOCHS = os.environ.get('EPOCHS', '100')
MAX_SEQ = os.environ.get('MAX_SEQ', '12')
THRESHOLD = float(os.environ.get('THRESHOLD', '0.00'))

# Local configuration
LOCAL_MODEL_DIR = os.environ.get('LOCAL_MODEL_DIR', 'trained_models')
LOCAL_DATA_FILE = os.environ.get('LOCAL_DATA_FILE', 'latest_odds.parquet')
LOCAL_OUTPUT_FILE = os.environ.get('LOCAL_OUTPUT_FILE', 'predicted_games.csv')

# Define constants for features processing - same as in predict_games.py
MIN_BET = 1
MAX_BET = 10
RATIOS = ["ratio1", "ratio2", "ratio3"]

# Initialize AWS clients if not running locally
if not IS_LOCAL:
    s3 = boto3.client('s3', region_name=AWS_REGION)
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    athena = boto3.client('athena', region_name=AWS_REGION)

def download_model_from_s3():
    """Download the trained model from S3."""
    if IS_LOCAL:
        # When running locally, use the local model file
        local_model_path = f"{LOCAL_MODEL_DIR}/{MODEL_TYPE}_{EPOCHS}_{MAX_SEQ}_v2.h5"
        logger.info(f"Running locally, using model from {local_model_path}")
        return local_model_path
    else:
        # When running in Lambda, download from S3
        model_key = f"models/{MODEL_TYPE}_{EPOCHS}_{MAX_SEQ}_v1.h5"
        local_model_path = f"/tmp/{MODEL_TYPE}_{EPOCHS}_{MAX_SEQ}_v1.h5"
        
        logger.info(f"Downloading model from s3://{S3_BUCKET}/{model_key}")
        try:
            s3.download_file(S3_BUCKET, model_key, local_model_path)
            logger.info(f"Model downloaded to {local_model_path}")
            return local_model_path
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise

def fetch_games_from_athena():
    """Fetch game data from the past 7 days from Athena, including historical odds."""

    try:
        # First, get the latest run_time in the dataset
        logger.info("Fetching games from Athena")
        latest_run_time_query = f'SELECT MAX(run_time) as latest_run_time FROM "{ATHENA_DATABASE}"."api_odds"'
        logger.info(f"Latest run_time query: {latest_run_time_query}")

        # Setup boto3 session
        boto3.setup_default_session(region_name=AWS_REGION)
        session = boto3.Session(region_name=AWS_REGION)

        latest_run_time_df = wr.athena.read_sql_query(
            sql=latest_run_time_query,
            database=ATHENA_DATABASE,
            boto3_session=session
        )

        latest_run_time = latest_run_time_df['latest_run_time'].iloc[0]

        # Convert to datetime if it's a string
        if isinstance(latest_run_time, str):
            latest_run_time = pd.to_datetime(latest_run_time)

        logger.info(f"Latest run_time in Athena: {latest_run_time}")

        # Calculate date 7 days before the latest run_time
        seven_days_before = latest_run_time - timedelta(days=7)

        # Query to get ALL odds history from the last 7 days, not just the latest
        query = f"""
        SELECT 
            *
        FROM 
            "{ATHENA_DATABASE}"."api_odds"
        WHERE 
            date_parsed >= '{seven_days_before.strftime("%Y-%m-%d")}'
            AND type = 'Soccer'
            AND league IN ({', '.join(f"'{league}'" for league in BIG_GAMES)})
        ORDER BY 
            unique_id, run_time
        """

        logger.info(f"Executing Athena query for games since {seven_days_before.strftime('%Y-%m-%d')}")

        # Execute query using AWS Wrangler
        df = wr.athena.read_sql_query(
            sql=query,
            database=ATHENA_DATABASE,
            boto3_session=session
        )

        # Convert run_time to datetime if it's not already
        df['run_time'] = pd.to_datetime(df['run_time'])

        # First, calculate the global maximum run_time in the processed DataFrame
        max_run_time = df["run_time"].max()

        # Identify the unique_ids (game IDs) that have at least one row with the max_run_time
        games_with_latest_run_time = df.loc[
            df["run_time"] == max_run_time, "unique_id"
        ].unique()

        # Keep only those games in the DataFrame (i.e. rows where unique_id is in the list above)
        df = df[df["unique_id"].isin(games_with_latest_run_time)].copy()

        logger.info(f"Fetched {len(df)} odds records for {df['unique_id'].nunique()} unique games from Athena")
        return df

    except Exception as e:
        logger.error(f"Error fetching games from Athena: {str(e)}")

        # If there's an error and we're in local mode, fall back to the parquet file
        if IS_LOCAL:
            logger.warning(f"Falling back to local parquet file: {LOCAL_DATA_FILE}")
            try:
                df = pd.read_parquet(LOCAL_DATA_FILE)
                logger.info(f"Loaded {len(df)} records for {df['unique_id'].nunique()} unique games from {LOCAL_DATA_FILE}")
                return df
            except Exception as e2:
                logger.error(f"Error loading data from {LOCAL_DATA_FILE}: {str(e2)}")
                raise
        else:
            raise

# Functions from predict_games.py
def scale_features(features: pd.DataFrame) -> pd.DataFrame:
    """Scale features to a range between 0 and 1"""
    return (features - MIN_BET) / (MAX_BET - 1)

def prepare_features(features: pd.DataFrame, max_seq_len: int) -> np.array:
    """Prepare features for the LSTM model"""
    features_scaled = scale_features(features)
    features_scaled = features_scaled.to_numpy()
    features_scaled = features_scaled[-max_seq_len:, :]
    return features_scaled

def remove_consecutive_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove consecutive duplicate rows"""
    shifted = df[RATIOS].shift()
    mask = ~((df[RATIOS] == shifted).all(axis=1))
    return df[mask]

def process_ratios(group_df: pd.DataFrame) -> pd.DataFrame:
    """Process odds ratios for prediction"""
    group_df[RATIOS] = group_df[RATIOS].astype(float)
    group_df = remove_consecutive_duplicates(group_df)
    return group_df

def group_games_by_id(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group games by their ID and create DataFrames with odds history"""
    grouped_games = {}
    
    for _, row in df.iterrows():
        game_id = row.get('unique_id', str(_))
        if game_id not in grouped_games:
            # Initialize with current odds
            grouped_games[game_id] = pd.DataFrame({
                'ratio1': [row.get('home_win_odds', 0)],
                'ratio2': [row.get('draw_odds', 0)],
                'ratio3': [row.get('away_win_odds', 0)],
                'game_info': [row]  # Store the full game info for later use
            })
        else:
            # Add current odds to the history
            grouped_games[game_id] = pd.concat([
                grouped_games[game_id],
                pd.DataFrame({
                    'ratio1': [row.get('home_win_odds', 0)],
                    'ratio2': [row.get('draw_odds', 0)],
                    'ratio3': [row.get('away_win_odds', 0)],
                    'game_info': [row]
                })
            ], ignore_index=True)
    
    return grouped_games

def preprocess_data(df):
    """Prepare the game data for prediction by processing historical odds for each game."""
    logger.info("Preprocessing game data")
    
    # Get unique game IDs
    game_ids = df['unique_id'].unique()
    logger.info(f"Processing {len(game_ids)} unique games")
    
    # List to store predictions that have enough data
    valid_game_data = []
    
    # Process each game individually
    for game_id in game_ids:
        # Get all odds history for this game
        game_odds = df[df['unique_id'] == game_id].sort_values('run_time')
        
        # Prepare data for this game
        game_df = pd.DataFrame({
            'ratio1': game_odds['ratio1'],
            'ratio2': game_odds['ratio2'],
            'ratio3': game_odds['ratio3'],
        })
        
        # Process the ratios to remove duplicates
        game_df = process_ratios(game_df)
        
        # Check if we have enough data points
        if game_df.shape[0] < int(MAX_SEQ):
            continue
        
        # Prepare features
        x = prepare_features(game_df, int(MAX_SEQ))
        
        # Get latest game info
        latest_game_info = game_odds.iloc[-1]
        
        # Add to valid games
        valid_game_data.append({
            'game_id': game_id,
            'features': x,
            'info': latest_game_info
        })
    
    logger.info(f"Found {len(valid_game_data)} games with enough data for prediction")
    return valid_game_data

def make_predictions(model, valid_game_data, model_name):
    """Make predictions for all games with enough historical data."""
    logger.info("Making predictions")
    
    # Initialize list for game predictions
    all_game_predictions = []
    
    try:
        # Process each game
        for game_data in valid_game_data:
            game_id = game_data['game_id']
            features = game_data['features']
            info = game_data['info']
            
            # Make prediction
            prediction = model.predict(features.reshape(1, int(MAX_SEQ), 3), verbose=0)
            logger.info(f"Prediction for game {game_id}: {prediction}")
            
            # Extract game information
            home_team = info.get('option1')
            away_team = info.get('option3')
            event_date = info.get('event_date')
            game_time = info.get('time')
            league = info.get('league')
            result_id = game_id  # Set result_id to be the same as the game's unique_id
            
            # Get odds
            home_odds = float(info.get('ratio1'))
            draw_odds = float(info.get('ratio2'))
            away_odds = float(info.get('ratio3'))
            
            # Store predictions, probabilities, and calculated expected values
            outcomes = []
            for outcome_idx, (outcome_name, odds_value) in enumerate(
                zip(['Home Win', 'Draw', 'Away Win'], [home_odds, draw_odds, away_odds])
            ):
                prob = float(prediction[0][outcome_idx])
                threshold = 1 / odds_value + THRESHOLD
                ev = prob * odds_value - 1
                
                outcomes.append({
                    'name': outcome_name,
                    'probability': prob,
                    'odds': odds_value,
                    'expected_value': ev,
                    'is_profitable': prob > threshold
                })
            
            # Find best outcome by expected value
            best_outcome = max(outcomes, key=lambda x: x['expected_value'])
            
            # Create game prediction record with timestamp
            prediction_timestamp = datetime.now().isoformat()
            
            # Convert date objects to strings
            if isinstance(event_date, date):
                event_date = event_date.isoformat()  # Convert to ISO format string
            
            # Create a valid match_time_str
            if event_date and game_time:
                match_time_str = f"{event_date} {game_time}"
            else:
                match_time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
                
            game_prediction = {
                'id': game_id,
                'result_id': result_id,  # Include result_id in prediction
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'event_date': event_date,
                'game_time': game_time,
                'match_time_str': match_time_str,
                'prediction_timestamp': prediction_timestamp,
                # Model metadata
                'model_name': model_name,
                # All odds
                'home_odds': home_odds,
                'draw_odds': draw_odds,
                'away_odds': away_odds,
                # All probabilities
                'home_win_prob': outcomes[0]['probability'],
                'draw_prob': outcomes[1]['probability'],
                'away_win_prob': outcomes[2]['probability'],
                # All EVs
                'home_win_ev': outcomes[0]['expected_value'],
                'draw_ev': outcomes[1]['expected_value'],
                'away_win_ev': outcomes[2]['expected_value'],
                # Is profitable flags for each outcome
                'home_win_is_profitable': outcomes[0]['is_profitable'],
                'draw_is_profitable': outcomes[1]['is_profitable'],
                'away_win_is_profitable': outcomes[2]['is_profitable'],
                # Status for tracking
                'status': 'pending',
                # Add overall expected value (maximum of all outcomes) for sorting
                'expected_value': max(outcomes[0]['expected_value'], outcomes[1]['expected_value'], outcomes[2]['expected_value']),
                # Add is_profitable for overall game (if any outcome is profitable)
                'is_profitable': outcomes[0]['is_profitable'] or outcomes[1]['is_profitable'] or outcomes[2]['is_profitable']
            }
            
            all_game_predictions.append(game_prediction)
            
    except Exception as e:
        logger.error(f"Error during model prediction: {str(e)}")
    
    # Sort by expected value (highest first)
    all_game_predictions.sort(key=lambda x: x['expected_value'], reverse=True)
    
    # Count profitable predictions
    profitable_count = len([p for p in all_game_predictions if p['is_profitable']])
    logger.info(f"Found {len(all_game_predictions)} games, of which {profitable_count} have profitable predictions")
    
    return all_game_predictions

def save_predictions(all_game_predictions):
    """Save predictions either to DynamoDB (in Lambda) or CSV (locally)."""
    if IS_LOCAL:
        # When running locally, save to CSV
        logger.info(f"Saving {len(all_game_predictions)} game predictions to {LOCAL_OUTPUT_FILE}")
        
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(all_game_predictions)
        
        # Save to CSV
        df.to_csv(LOCAL_OUTPUT_FILE, index=False)
        logger.info(f"Successfully saved predictions to {LOCAL_OUTPUT_FILE}")
    else:
        # When running in Lambda, save to both DynamoDB tables
        logger.info(f"Saving {len(all_game_predictions)} game predictions to DynamoDB tables")
        
        # Get DynamoDB tables
        all_predictions_table = dynamodb.Table(ALL_PREDICTIONS_TABLE)
        profitable_games_table = dynamodb.Table(PROFITABLE_GAMES_TABLE)
        
        timestamp = datetime.now().isoformat()
        
        # Helper function to convert floats to Decimal
        def convert_floats_to_decimal(obj):
            if isinstance(obj, float):
                return Decimal(str(obj))
            elif isinstance(obj, date):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats_to_decimal(i) for i in obj]
            else:
                return obj
        
        # Save all predictions to the all-predicted-games table
        logger.info(f"Saving all {len(all_game_predictions)} predictions to {ALL_PREDICTIONS_TABLE}")
        with all_predictions_table.batch_writer() as batch:
            for game in all_game_predictions:
                # Add timestamp and make sure all values are serializable
                game['timestamp'] = timestamp
                
                # Convert all float values to Decimal
                game_dict = convert_floats_to_decimal(game)
                
                # Add to DynamoDB
                try:
                    batch.put_item(Item=game_dict)
                except Exception as e:
                    logger.error(f"Error saving game prediction for {game['id']} to {ALL_PREDICTIONS_TABLE}: {str(e)}")
        
        # Filter for profitable predictions only - games with at least one profitable outcome
        profitable_predictions = [game for game in all_game_predictions if 
                                 (game['home_win_is_profitable'] or 
                                  game['draw_is_profitable'] or 
                                  game['away_win_is_profitable'])]
        
        # Save only profitable predictions to the profitable-games table
        logger.info(f"Saving {len(profitable_predictions)} profitable predictions to {PROFITABLE_GAMES_TABLE}")
        with profitable_games_table.batch_writer() as batch:
            for game in profitable_predictions:
                # Add timestamp and make sure all values are serializable
                game['timestamp'] = timestamp
                
                # Convert all float values to Decimal
                game_dict = convert_floats_to_decimal(game)
                
                # Add to DynamoDB
                try:
                    batch.put_item(Item=game_dict)
                except Exception as e:
                    logger.error(f"Error saving game prediction for {game['id']} to {PROFITABLE_GAMES_TABLE}: {str(e)}")
        
        logger.info(f"Successfully saved all game predictions to DynamoDB tables")

def check_existing_games(unique_ids):
    """Check if games already exist in the profitable games DynamoDB table."""
    if IS_LOCAL:
        logger.info("Running locally, skipping DynamoDB check")
        return []
    
    logger.info(f"Checking for existing games in {PROFITABLE_GAMES_TABLE}")
    
    # Get DynamoDB client for batch operations
    dynamodb_client = boto3.client('dynamodb', region_name=AWS_REGION)
    
    # Initialize list for existing games
    existing_profitable_games = []
    
    # Process in batches of 100 (DynamoDB batch operation limit)
    batch_size = 100
    for i in range(0, len(unique_ids), batch_size):
        batch_ids = unique_ids[i:i+batch_size]
        
        # Prepare batch get request for profitable-games table
        profitable_games_request = {
            PROFITABLE_GAMES_TABLE: {
                'Keys': [{'id': {'S': game_id}} for game_id in batch_ids],
                'ProjectionExpression': 'id'
            }
        }
        
        try:
            # Execute batch get for profitable-games table
            profitable_games_response = dynamodb_client.batch_get_item(
                RequestItems=profitable_games_request
            )
            
            # Process profitable-games response
            if PROFITABLE_GAMES_TABLE in profitable_games_response.get('Responses', {}):
                items = profitable_games_response['Responses'][PROFITABLE_GAMES_TABLE]
                for item in items:
                    game_id = item['id']['S']
                    existing_profitable_games.append(game_id)
                    
        except Exception as e:
            logger.error(f"Error in batch checking games in DynamoDB: {str(e)}")
    
    logger.info(f"Found {len(existing_profitable_games)} existing profitable games in {PROFITABLE_GAMES_TABLE}")
    
    return existing_profitable_games

def clear_dynamodb_tables():
    """Clear all items from both DynamoDB tables."""
    if IS_LOCAL:
        logger.info("Running locally, cannot clear DynamoDB tables")
        return False
    
    try:
        logger.info(f"Clearing DynamoDB tables: {ALL_PREDICTIONS_TABLE} and {PROFITABLE_GAMES_TABLE}")
        
        # Get DynamoDB resource
        dynamo_resource = boto3.resource('dynamodb', region_name=AWS_REGION)
        
        # Clear all-predicted-games table
        all_predictions_table = dynamo_resource.Table(ALL_PREDICTIONS_TABLE)
        
        # Scan for all items
        scan_response = all_predictions_table.scan(
            ProjectionExpression='id, prediction_timestamp'
        )
        items = scan_response.get('Items', [])
        
        logger.info(f"Found {len(items)} items in {ALL_PREDICTIONS_TABLE} table")
        
        # Delete all items in batches
        with all_predictions_table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={'id': item['id']})
        
        logger.info(f"Successfully cleared {len(items)} items from {ALL_PREDICTIONS_TABLE} table")
        
        # Clear profitable-games table
        profitable_games_table = dynamo_resource.Table(PROFITABLE_GAMES_TABLE)
        
        # Scan for all items
        scan_response = profitable_games_table.scan(
            ProjectionExpression='id, prediction_timestamp'
        )
        items = scan_response.get('Items', [])
        
        logger.info(f"Found {len(items)} items in {PROFITABLE_GAMES_TABLE} table")
        
        # Delete all items in batches
        with profitable_games_table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={'id': item['id']})
        
        logger.info(f"Successfully cleared {len(items)} items from {PROFITABLE_GAMES_TABLE} table")
        
        return True
        
    except Exception as e:
        logger.error(f"Error clearing DynamoDB tables: {str(e)}")
        return False

def lambda_handler(event, context):
    """Main Lambda function handler."""
    logger.info("Starting profitable games prediction Lambda")
    logger.info(f"Running in {'local' if IS_LOCAL else 'AWS Lambda'} mode")

    # Check if we need to clear the tables
    should_clear_tables = event.get('clear_tables', False) if event else False
    
    if should_clear_tables:
        success = clear_dynamodb_tables()
        status_code = 200 if success else 500
        return {
            'statusCode': status_code,
            'body': json.dumps('Successfully cleared DynamoDB tables' if success else 'Failed to clear DynamoDB tables')
        }

    try:
        # Step 1: Fetch game data (from Athena in Lambda, from local file locally)
        games_df = fetch_games_from_athena()

        # Get unique game IDs
        unique_game_ids = games_df['unique_id'].unique().tolist()
        logger.info(f"Found {len(unique_game_ids)} unique games in the dataset")

        # Step 2: Check which games already exist in profitable games DynamoDB table
        existing_profitable_games = check_existing_games(unique_game_ids)

        # Filter out games that already exist in the profitable games table
        games_to_process = []
        for game_id in unique_game_ids:
            if game_id in existing_profitable_games:
                # Skip games that exist in the profitable games table
                logger.info(f"Skipping game {game_id} as it already exists in {PROFITABLE_GAMES_TABLE}")
                continue
            else:
                games_to_process.append(game_id)

        logger.info(f"Filtered to {len(games_to_process)} games that need prediction")

        # If all games already exist in profitable games table, exit early
        if len(games_to_process) == 0:
            logger.info(f"All games already exist in {PROFITABLE_GAMES_TABLE}. Exiting early.")
            return {
                'statusCode': 200,
                'body': json.dumps(f'All {len(unique_game_ids)} games already exist in {PROFITABLE_GAMES_TABLE}. No new predictions needed.')
            }

        # Filter the dataframe to only include games that need processing
        filtered_games_df = games_df[games_df['unique_id'].isin(games_to_process)]
        logger.info(f"Filtered dataset contains {filtered_games_df['unique_id'].nunique()} unique games")

        # Step 3: Load the model
        logger.info(f"Loading model {MODEL_TYPE}_{EPOCHS}_{MAX_SEQ}")
        local_model_path = download_model_from_s3()
        
        # Extract model version from the model path
        if IS_LOCAL:
            model_version = "v2"  # Local version
        else:
            model_version = "v1"  # Default AWS version
            # Try to extract version from filename if possible
            if '_v' in local_model_path:
                try:
                    model_version = local_model_path.split('_v')[1].split('.')[0]
                    model_version = f"v{model_version}"
                except:
                    pass
        
        # Create full model name
        model_name = f"{MODEL_TYPE}_{EPOCHS}_{MAX_SEQ}_{model_version}"
        logger.info(f"Using model: {model_name}")

        # Load model using Keras load_model function
        loaded_model = load_model(local_model_path)
        logger.info(f"Successfully loaded model from {local_model_path}")

        # Step 4: Preprocess the data
        valid_game_data = preprocess_data(filtered_games_df)

        # Step 5: Check if there are any games with enough data points
        if len(valid_game_data) == 0:
            logger.warning("No games with enough data points found. Exiting early.")
            return {
                'statusCode': 200,
                'body': json.dumps(f'Processed {filtered_games_df["unique_id"].nunique()} games but none had enough data points for prediction.')
            }

        # Step 6: Make predictions for all games
        all_game_predictions = make_predictions(loaded_model, valid_game_data, model_name)

        # Step 7: Save predictions (to DynamoDB in Lambda, to CSV locally)
        if all_game_predictions:
            save_predictions(all_game_predictions)
        else:
            logger.info("No predictions made")

        # Count profitable predictions for the response
        profitable_count = len([p for p in all_game_predictions if p['is_profitable']])

        return {
            'statusCode': 200,
            'body': json.dumps(f'Successfully processed {len(games_to_process)} new games and saved {len(all_game_predictions)} game predictions to {ALL_PREDICTIONS_TABLE}, of which {profitable_count} profitable predictions were saved to {PROFITABLE_GAMES_TABLE}')
        }

    except Exception as e:
        logger.error(f"Error in Lambda execution: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }

# Make it callable for local debugging
if __name__ == "__main__" and IS_LOCAL:
    # Set IS_LOCAL=True for local execution
    os.environ['IS_LOCAL'] = 'True'
    lambda_handler(None, None)
