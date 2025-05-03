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
from tensorflow.keras.initializers import Orthogonal
from googletrans import Translator
from team_translations import TEAM_TRANSLATIONS, get_translation
from zoneinfo import ZoneInfo

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a translator instance for use throughout the function
translator = Translator()

# Define Israel Timezone using zoneinfo
israel_tz = ZoneInfo("Asia/Jerusalem")

# Initialize translation cache to avoid repeated API calls for the same text
_translation_cache = {}

BIG_GAMES = [
    "פרמייר ליג",
    "גביע ספרדי",
    "צרפתית ראשונה",
    "איטלקית ראשונה",
    "גביע אנגלי",
    "גביע המדינה",
    "קונפרנס ליג",
    "מוקדמות אליפות אירופה",
    "מוקדמות מונדיאל, אירופה" "גרמנית ראשונה",
    "ליגת העל",
    "סופר קאפ",
    "ספרדית ראשונה",
    "ליגת האלופות",
    "הליגה האירופית",
    "גביע איטלקי",
    "ליגת האומות",
    "גביע המדינה Winner",
    "ליגת Winner" "גביע הליגה האנגלי",
    "גביע אסיה",
    "גביע גרמני",
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

# Register custom initializers for model loading
tf.keras.utils.get_custom_objects().update({
    'CustomInputLayer': CustomInputLayer,
    'Orthogonal': Orthogonal
})

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

def list_models_in_s3():
    """List all models available in the S3 bucket models directory."""
    if IS_LOCAL:
        # When running locally, scan the local models directory
        local_models = []
        try:
            for file in os.listdir(LOCAL_MODEL_DIR):
                if file.endswith('.h5'):
                    local_models.append({
                        'key': file,
                        'local_path': f"{LOCAL_MODEL_DIR}/{file}"
                    })
            logger.info(f"Found {len(local_models)} models in local directory")
            return local_models
        except Exception as e:
            logger.error(f"Error listing local models: {str(e)}")
            # Fall back to default model
            return [{
                'key': f"{MODEL_TYPE}_{EPOCHS}_{MAX_SEQ}_v2.h5",
                'local_path': f"{LOCAL_MODEL_DIR}/{MODEL_TYPE}_{EPOCHS}_{MAX_SEQ}_v2.h5"
            }]
    else:
        # When running in Lambda, list models from S3
        models = []
        try:
            # List objects in the models prefix
            response = s3.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix='models/'
            )
            
            # Filter for .h5 files only
            for item in response.get('Contents', []):
                if item['Key'].endswith('.h5'):
                    # Extract the filename
                    filename = os.path.basename(item['Key'])
                    # Create a local path in /tmp
                    local_path = f"/tmp/{filename}"
                    
                    models.append({
                        'key': item['Key'],
                        'local_path': local_path
                    })
            
            logger.info(f"Found {len(models)} models in S3 bucket")
            return models
        except Exception as e:
            logger.error(f"Error listing models in S3: {str(e)}")
            # Fall back to default model
            return [{
                'key': f"models/{MODEL_TYPE}_{EPOCHS}_{MAX_SEQ}_v1.h5",
                'local_path': f"/tmp/{MODEL_TYPE}_{EPOCHS}_{MAX_SEQ}_v1.h5"
            }]

def download_model(model_info):
    """Download a specific model from S3."""
    if IS_LOCAL:
        # When running locally, just return the local path
        logger.info(f"Using local model at {model_info['local_path']}")
        return model_info['local_path']
    else:
        # When running in Lambda, download from S3
        try:
            logger.info(f"Downloading model from s3://{S3_BUCKET}/{model_info['key']}")
            s3.download_file(S3_BUCKET, model_info['key'], model_info['local_path'])
            logger.info(f"Model downloaded to {model_info['local_path']}")
            return model_info['local_path']
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

def translate_to_english(text, context=""):
    """
    Automatically translate Hebrew text to English using Google Translate API.
    Includes caching and fallback to known translations.
    
    Args:
        text (str): The Hebrew text to translate
        context (str): Optional context (e.g., "football team" or "football league")
    
    Returns:
        str: The translated English text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Check if already in English or contains only non-Hebrew characters
    if all(ord(c) < 1200 for c in text):  # Hebrew Unicode range starts around 1200
        return text
        
    # Check cache first
    cache_key = f"{text}:{context}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]
    
    # Check fallback dictionary for common terms
    if text in TEAM_TRANSLATIONS:
        translated_text = TEAM_TRANSLATIONS[text]
        logger.info(f"Using fallback translation for '{text}': '{translated_text}'")
        _translation_cache[cache_key] = translated_text
        return translated_text
    
    try:
        # Use Google Translate API
        translation = translator.translate(text, src='iw', dest='en')
        translated_text = translation.text
        
        # Cache the result
        _translation_cache[cache_key] = translated_text
        
        logger.info(f"Translated '{text}' to '{translated_text}'")
        return translated_text
    
    except Exception as e:
        logger.error(f"Translation error for text '{text}': {str(e)}")
        
        # If automatic translation fails, try to clean and return the original text
        # or use a fallback
        return text

def make_predictions(model, valid_game_data, model_name):
    """Make predictions for all games with enough historical data."""
    logger.info(f"Making predictions with model {model_name}")
    
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
            logger.info(f"Prediction for game {game_id} with model {model_name}: {prediction}")
            
            # Extract game information
            home_team = info.get('option1')
            away_team = info.get('option3')
            event_date = info.get('event_date')
            game_time = info.get('time')
            league = info.get('league')
            # Extract the result_id directly from the api_odds data (info object)
            result_id = info.get('result_id') 
            
            # Translate team names and league to English with context
            english_home_team = translate_to_english(home_team, "football team")
            english_away_team = translate_to_english(away_team, "football team")
            english_league = translate_to_english(league, "football league")
            
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
            
            # Create game prediction record with timestamp in Israel Time
            prediction_timestamp = datetime.now(israel_tz).isoformat()
            
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
                # Store English translations
                'english_home_team': english_home_team,
                'english_away_team': english_away_team,
                'english_league': english_league,
                'event_date': event_date,
                'game_time': game_time,
                'match_time_str': match_time_str,
                'prediction_timestamp': prediction_timestamp,
                # Set the prediction field to the best outcome name
                'prediction': best_outcome['name'],
                # Model metadata
                'model_name': model_name,  # Store the specific model name used for prediction
                'model_type': model_name.split('_')[0],  # Extract the model type (lstm, bert, etc.)
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
        logger.error(f"Error during model prediction with model {model_name}: {str(e)}")
    
    # Sort by expected value (highest first)
    all_game_predictions.sort(key=lambda x: x['expected_value'], reverse=True)
    
    # Count profitable predictions
    profitable_count = len([p for p in all_game_predictions if p['is_profitable']])
    logger.info(f"Model {model_name} found {len(all_game_predictions)} games, of which {profitable_count} have profitable predictions")
    
    return all_game_predictions

def save_predictions(all_game_predictions, existing_predictions_by_model={}):
    """Save all predictions to DynamoDB, allowing multiple profitable outcomes for the same game."""
    if IS_LOCAL:
        # When running locally, save to CSV
        try:
            logger.info(f"Saving predictions to {LOCAL_OUTPUT_FILE}")
            
            # Convert predictions to DataFrame
            predictions_df = pd.DataFrame(all_game_predictions)
            
            # Save to CSV
            predictions_df.to_csv(LOCAL_OUTPUT_FILE, index=False)
            
            logger.info(f"Successfully saved predictions to {LOCAL_OUTPUT_FILE}")
            
        except Exception as e:
            logger.error(f"Error saving predictions to CSV: {e}")
    else:
        # When running in Lambda, save to DynamoDB
        try:
            logger.info(f"Saving predictions to DynamoDB tables: {ALL_PREDICTIONS_TABLE} and {PROFITABLE_GAMES_TABLE}")
            
            # Create DynamoDB resource and tables
            dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
            all_predictions_table = dynamodb.Table(ALL_PREDICTIONS_TABLE)
            profitable_games_table = dynamodb.Table(PROFITABLE_GAMES_TABLE)
            
            # Track success and error counts
            saved_count = 0
            profitable_saved_count = 0
            skipped_count = 0
            error_count = 0
            
            # Save each prediction individually
            for i, game in enumerate(all_game_predictions):
                try:
                    # Get required fields
                    game_id = game['id']
                    model_name = game.get('model_name', MODEL_TYPE)
                    
                    # Generate timestamp if not present
                    if 'timestamp' not in game:
                        game['timestamp'] = datetime.now().isoformat()
                    
                    # Create model_timestamp composite sort key
                    game['model_timestamp'] = f"{model_name}#{game['timestamp']}"
                    
                    # Set game_id to be consistent
                    game['game_id'] = game_id
                    
                    # Add is_profitable as a numeric attribute for GSI
                    is_profitable = (
                        game.get('home_win_is_profitable', False) or
                        game.get('draw_is_profitable', False) or
                        game.get('away_win_is_profitable', False)
                    )
                    game['is_profitable'] = 1 if is_profitable else 0
                    
                    # Convert all float values to Decimal
                    game_dict = convert_floats_to_decimal(game)
                    
                    # Always save to all_predictions_table
                    all_predictions_table.put_item(Item=game_dict)
                    saved_count += 1
                    
                    # Check if we should save to profitable_games_table
                    if is_profitable:
                        # Check which outcomes are profitable
                        profitable_outcomes = []
                        
                        if game.get('home_win_is_profitable', False):
                            profitable_outcomes.append({
                                'outcome': 'Home Win',
                                'probability': game.get('home_win_prob', 0),
                                'ev': game.get('home_win_ev', 0),
                                'odds': game.get('home_odds', 0)
                            })
                            
                        if game.get('draw_is_profitable', False):
                            profitable_outcomes.append({
                                'outcome': 'Draw',
                                'probability': game.get('draw_prob', 0),
                                'ev': game.get('draw_ev', 0),
                                'odds': game.get('draw_odds', 0)
                            })
                            
                        if game.get('away_win_is_profitable', False):
                            profitable_outcomes.append({
                                'outcome': 'Away Win',
                                'probability': game.get('away_win_prob', 0),
                                'ev': game.get('away_win_ev', 0),
                                'odds': game.get('away_odds', 0)
                            })
                        
                        # Sort outcomes by expected value (highest first)
                        profitable_outcomes.sort(key=lambda x: x['ev'], reverse=True)
                        
                        # Add a profitable entry for each profitable outcome
                        for profitable_idx, profitable_outcome in enumerate(profitable_outcomes):
                            # Create a copy of the game dict for this specific outcome
                            outcome_game_dict = game_dict.copy()
                            
                            # Set the prediction to this outcome
                            outcome_game_dict['prediction'] = profitable_outcome['outcome']
                            
                            # Add unique ID suffix for additional profitable outcomes (after the first one)
                            if profitable_idx > 0:
                                outcome_game_dict['id'] = f"{game_id}_{profitable_outcome['outcome'].replace(' ', '_')}"
                                outcome_game_dict['game_id'] = outcome_game_dict['id']
                                
                                # Update model_timestamp to avoid overwriting
                                outcome_game_dict['model_timestamp'] = f"{model_name}#{profitable_outcome['outcome']}#{game_dict['timestamp']}"
                            
                            # Save each profitable outcome
                            profitable_games_table.put_item(Item=outcome_game_dict)
                            profitable_saved_count += 1
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error saving prediction {i} for game {game['id']} with model {game.get('model_name', MODEL_TYPE)}: {str(e)}")
            
            logger.info(f"Successfully saved {saved_count} predictions to {ALL_PREDICTIONS_TABLE}")
            logger.info(f"Added {profitable_saved_count} profitable predictions to {PROFITABLE_GAMES_TABLE}")
            logger.info(f"Encountered {error_count} errors during saving")

        except Exception as e:
            logger.error(f"Error saving predictions to DynamoDB: {e}")
            raise

def check_existing_predictions(model_name):
    """Check for existing profitable predictions for this model in DynamoDB."""
    if IS_LOCAL:
        logger.info("Running locally, skipping DynamoDB check for existing predictions")
        return {}
    
    # Get the profitable predictions table
    table = dynamodb.Table(PROFITABLE_GAMES_TABLE)
    existing_predictions = {}
    
    try:
        # Query for all profitable predictions for this model
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key('model_name').eq(model_name)
        )
        
        # Process items and add to dictionary
        for item in response.get('Items', []):
            existing_predictions[item['game_id']] = item
        
        # Handle pagination if needed
        while 'LastEvaluatedKey' in response:
            response = table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('model_name').eq(model_name),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            
            # Add newly retrieved items
            for item in response.get('Items', []):
                existing_predictions[item['game_id']] = item
                
        logger.info(f"Found {len(existing_predictions)} existing profitable predictions for model {model_name}")
        
        return existing_predictions
    
    except Exception as e:
        logger.error(f"Error checking for existing predictions for model {model_name}: {str(e)}")
        return {}

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
        
        # For our new schema, we need to get both game_id and model_timestamp
        # Scan for all items
        scan_response = all_predictions_table.scan(
            ProjectionExpression='game_id, model_timestamp'
        )
        items = scan_response.get('Items', [])
        
        # Handle pagination if needed
        while 'LastEvaluatedKey' in scan_response:
            scan_response = all_predictions_table.scan(
                ProjectionExpression='game_id, model_timestamp',
                ExclusiveStartKey=scan_response['LastEvaluatedKey']
            )
            items.extend(scan_response.get('Items', []))
        
        logger.info(f"Found {len(items)} items in {ALL_PREDICTIONS_TABLE} table")
        
        # Delete all items in batches
        with all_predictions_table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={
                    'game_id': item['game_id'],
                    'model_timestamp': item['model_timestamp']
                })
        
        logger.info(f"Successfully cleared {len(items)} items from {ALL_PREDICTIONS_TABLE} table")
        
        # Clear profitable-games table
        profitable_games_table = dynamo_resource.Table(PROFITABLE_GAMES_TABLE)
        
        # For our new schema, we need to get both model_name and game_id
        # Scan for all items
        scan_response = profitable_games_table.scan(
            ProjectionExpression='model_name, game_id'
        )
        items = scan_response.get('Items', [])
        
        # Handle pagination if needed
        while 'LastEvaluatedKey' in scan_response:
            scan_response = profitable_games_table.scan(
                ProjectionExpression='model_name, game_id',
                ExclusiveStartKey=scan_response['LastEvaluatedKey']
            )
            items.extend(scan_response.get('Items', []))
        
        logger.info(f"Found {len(items)} items in {PROFITABLE_GAMES_TABLE} table")
        
        # Delete all items in batches
        with profitable_games_table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={
                    'model_name': item['model_name'],
                    'game_id': item['game_id']
                })
        
        logger.info(f"Successfully cleared {len(items)} items from {PROFITABLE_GAMES_TABLE} table")
        
        return True
        
    except Exception as e:
        logger.error(f"Error clearing DynamoDB tables: {str(e)}")
        return False

def ensure_dynamodb_tables_exist():
    """Ensure DynamoDB tables exist with proper indexes."""
    if IS_LOCAL:
        logger.info("Running locally, skipping DynamoDB table creation")
        return
    
    try:
        # Create DynamoDB client and resource
        dynamodb_client = boto3.client('dynamodb', region_name=AWS_REGION)
        dynamodb_resource = boto3.resource('dynamodb', region_name=AWS_REGION)
        
        # Check if all-predictions table exists
        all_predictions_exists = False
        try:
            dynamodb_resource.Table(ALL_PREDICTIONS_TABLE).table_status
            all_predictions_exists = True
            logger.info(f"Table {ALL_PREDICTIONS_TABLE} exists")
        except:
            all_predictions_exists = False
            logger.info(f"Table {ALL_PREDICTIONS_TABLE} does not exist")
        
        # Create all-predictions table if it doesn't exist
        if not all_predictions_exists:
            logger.info(f"Creating {ALL_PREDICTIONS_TABLE} table with GSIs")
            dynamodb_client.create_table(
                TableName=ALL_PREDICTIONS_TABLE,
                KeySchema=[
                    {'AttributeName': 'game_id', 'KeyType': 'HASH'},  # Partition key
                    {'AttributeName': 'model_timestamp', 'KeyType': 'RANGE'}  # Sort key
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'game_id', 'AttributeType': 'S'},
                    {'AttributeName': 'model_timestamp', 'AttributeType': 'S'},
                    {'AttributeName': 'model_name', 'AttributeType': 'S'},
                    {'AttributeName': 'timestamp', 'AttributeType': 'S'},
                    {'AttributeName': 'is_profitable', 'AttributeType': 'N'}
                ],
                GlobalSecondaryIndexes=[
                    {
                        'IndexName': 'ModelTimeIndex',
                        'KeySchema': [
                            {'AttributeName': 'model_name', 'KeyType': 'HASH'},
                            {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                        ],
                        'Projection': {
                            'ProjectionType': 'ALL'
                        },
                        'ProvisionedThroughput': {
                            'ReadCapacityUnits': 5,
                            'WriteCapacityUnits': 5
                        }
                    },
                    {
                        'IndexName': 'ModelIsProfitableIndex',
                        'KeySchema': [
                            {'AttributeName': 'model_name', 'KeyType': 'HASH'},
                            {'AttributeName': 'is_profitable', 'KeyType': 'RANGE'}
                        ],
                        'Projection': {
                            'ProjectionType': 'ALL'
                        },
                        'ProvisionedThroughput': {
                            'ReadCapacityUnits': 5,
                            'WriteCapacityUnits': 5
                        }
                    }
                ],
                BillingMode='PROVISIONED',
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )
            logger.info(f"Waiting for {ALL_PREDICTIONS_TABLE} table to be created...")
            waiter = dynamodb_client.get_waiter('table_exists')
            waiter.wait(TableName=ALL_PREDICTIONS_TABLE)
            logger.info(f"Table {ALL_PREDICTIONS_TABLE} created successfully")
        
        # Check if profitable-predictions table exists
        profitable_games_exists = False
        try:
            dynamodb_resource.Table(PROFITABLE_GAMES_TABLE).table_status
            profitable_games_exists = True
            logger.info(f"Table {PROFITABLE_GAMES_TABLE} exists")
        except:
            profitable_games_exists = False
            logger.info(f"Table {PROFITABLE_GAMES_TABLE} does not exist")
            
        # Create profitable-predictions table if it doesn't exist
        if not profitable_games_exists:
            logger.info(f"Creating {PROFITABLE_GAMES_TABLE} table")
            dynamodb_client.create_table(
                TableName=PROFITABLE_GAMES_TABLE,
                KeySchema=[
                    {'AttributeName': 'model_name', 'KeyType': 'HASH'},  # Partition key
                    {'AttributeName': 'game_id', 'KeyType': 'RANGE'}  # Sort key
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'model_name', 'AttributeType': 'S'},
                    {'AttributeName': 'game_id', 'AttributeType': 'S'}
                ],
                BillingMode='PROVISIONED',
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )
            logger.info(f"Waiting for {PROFITABLE_GAMES_TABLE} table to be created...")
            waiter = dynamodb_client.get_waiter('table_exists')
            waiter.wait(TableName=PROFITABLE_GAMES_TABLE)
            logger.info(f"Table {PROFITABLE_GAMES_TABLE} created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error ensuring DynamoDB tables exist: {e}")
        return False

def convert_floats_to_decimal(obj):
    """
    Convert all float values in a nested object to Decimal for DynamoDB.
    DynamoDB doesn't support float values directly, so we need to convert them to Decimal.
    """
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
        # Ensure DynamoDB tables exist with proper indexes
        if not IS_LOCAL:
            ensure_dynamodb_tables_exist()
        
        # Step 1: Fetch game data (from Athena in Lambda, from local file locally)
        games_df = fetch_games_from_athena()

        # Get unique game IDs
        unique_game_ids = games_df['unique_id'].unique().tolist()
        logger.info(f"Found {len(unique_game_ids)} unique games in the dataset")

        # Step 2: Preprocess data once for all models
        valid_game_data = preprocess_data(games_df)
        
        # Check if there are any games with enough data points
        if len(valid_game_data) == 0:
            logger.warning("No games with enough data points found. Exiting early.")
            return {
                'statusCode': 200,
                'body': json.dumps(f'Processed {games_df["unique_id"].nunique()} games but none had enough data points for prediction.')
            }
        
        # Step 3: Get list of all available models
        models_list = list_models_in_s3()
        logger.info(f"Found {len(models_list)} models to process")
        
        # Track all predictions across all models
        all_predictions_across_models = []
        
        # Stats tracking dictionaries for each model
        model_prediction_counts = {}
        model_profitable_counts = {}
        model_skipped_counts = {}
        
        # Keep track of all existing predictions by model
        all_existing_predictions = {}
        
        # Step 4: Process each model
        for model_info in models_list:
            try:
                # Get model name from the key
                model_filename = os.path.basename(model_info['key'])
                
                # Extract model details from filename
                model_parts = model_filename.split('.')[0].split('_')
                if len(model_parts) >= 3:
                    model_type = model_parts[0]
                    epochs = model_parts[1]
                    max_seq = model_parts[2]
                    # Extract version if available (v1, v2, etc.)
                    model_version = "v1"  # Default
                    if len(model_parts) > 3 and model_parts[3].startswith('v'):
                        model_version = model_parts[3]
                else:
                    # If filename doesn't match expected pattern, use defaults
                    model_type = MODEL_TYPE
                    epochs = EPOCHS
                    max_seq = MAX_SEQ
                    model_version = "v1"
                
                model_name = f"{model_type}_{epochs}_{max_seq}_{model_version}"
                logger.info(f"Processing model: {model_name}")

                # Check which games already have predictions from this model
                existing_predictions = check_existing_predictions(model_name)
                
                # Store for later use
                all_existing_predictions[model_name] = existing_predictions
                
                # Initialize stats tracking for this model if not already done
                if model_name not in model_prediction_counts:
                    model_prediction_counts[model_name] = 0
                    model_profitable_counts[model_name] = 0
                    model_skipped_counts[model_name] = 0
                
                # Track skipped predictions for reporting
                model_skipped_counts[model_name] = len(existing_predictions)
                
                # Filter valid_game_data to only include games that don't already have profitable predictions
                # from this model. Games with non-profitable predictions will be processed again.
                model_games_to_process = []
                for game in valid_game_data:
                    game_id = game['game_id']
                    # Check if this game_id exists in the predictions AND is profitable
                    if game_id not in existing_predictions:
                        # No prediction for this game, process it
                        model_games_to_process.append(game)
                    elif existing_predictions[game_id].get('is_profitable', 0) != 1:
                        # Prediction exists but it's not profitable, process it again
                        model_games_to_process.append(game)
                        logger.info(f"Re-processing game {game_id} for model {model_name} - has prediction but not profitable")
                    else:
                        # Game already has a profitable prediction from this model, skip it
                        logger.info(f"Skipping game {game_id} for model {model_name} - already has profitable prediction")
                
                logger.info(f"Model {model_name} will process {len(model_games_to_process)} games out of {len(valid_game_data)} valid games")
                
                # If all games already have predictions from this model, skip to the next model
                if len(model_games_to_process) == 0:
                    logger.info(f"All games already have predictions from model {model_name}. Skipping to next model.")
                    continue
                
                # Download and load model
                local_model_path = download_model(model_info)
                loaded_model = load_model(local_model_path)
                logger.info(f"Successfully loaded model from {local_model_path}")
                
                # Make predictions with this model
                model_predictions = make_predictions(loaded_model, model_games_to_process, model_name)
                
                # Track statistics for this model
                model_prediction_counts[model_name] = len(model_predictions)
                model_profitable_counts[model_name] = len([p for p in model_predictions if 
                    p.get('home_win_is_profitable', False) or 
                    p.get('draw_is_profitable', False) or 
                    p.get('away_win_is_profitable', False)])
                
                # Add predictions to the complete list
                all_predictions_across_models.extend(model_predictions)
                
                # Clean up to free memory
                del loaded_model
                import gc
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing model {model_info['key']}: {str(e)}")
                # Initialize stats for failed models
                model_name = os.path.basename(model_info['key']).split('.')[0]
                model_prediction_counts[model_name] = 0
                model_profitable_counts[model_name] = 0
                model_skipped_counts[model_name] = 0
                # Continue to next model
                continue
        
        # Step 5: Save all predictions (to DynamoDB in Lambda, to CSV locally)
        if all_predictions_across_models:
            save_predictions(all_predictions_across_models, all_existing_predictions)
            # Count profitable predictions for the response
            profitable_count = len([p for p in all_predictions_across_models if p['is_profitable']])
            logger.info(f"Saved predictions from {len(models_list)} models with {profitable_count} profitable predictions")
        else:
            logger.info("No predictions made from any model")
        
        # Step 6: Generate summary report
        logger.info("\n" + "="*95)
        logger.info(" "*25 + "PREDICTION SUMMARY REPORT")
        logger.info("="*95)
        
        # Overall stats
        total_predictions = len(all_predictions_across_models)
        total_profitable = sum(model_profitable_counts.values())
        total_skipped = sum(model_skipped_counts.values())
        logger.info(f"Total predictions made: {total_predictions}")
        logger.info(f"Total predictions skipped (already in DB): {total_skipped}")
        
        if total_predictions > 0:
            profit_percentage = (total_profitable/total_predictions)*100
            logger.info(f"Total profitable predictions: {total_profitable} ({profit_percentage:.1f}% profitable)")
        else:
            logger.info(f"Total profitable predictions: 0 (0.0% profitable)")
        
        # Per-model stats
        logger.info("\nPredictions by model:")
        logger.info("-"*95)
        
        # Format as a table with headers
        logger.info(f"{'MODEL NAME':<30} {'PREDICTIONS':<15} {'PROFITABLE':<15} {'SKIPPED':<15} {'PERCENTAGE':<15}")
        logger.info("-"*95)
        
        # Sort models by number of profitable predictions (descending)
        sorted_models = sorted(model_prediction_counts.keys(), 
                              key=lambda x: model_profitable_counts.get(x, 0) + model_skipped_counts.get(x, 0), 
                              reverse=True)
        
        # Track total skipped predictions
        total_skipped = sum(model_skipped_counts.values())
        
        for model_name in sorted_models:
            pred_count = model_prediction_counts.get(model_name, 0)
            profitable_count = model_profitable_counts.get(model_name, 0)
            skipped_count = model_skipped_counts.get(model_name, 0)
            
            if pred_count > 0:
                profit_percent = (profitable_count / pred_count) * 100
                logger.info(f"{model_name:<30} {pred_count:<15d} {profitable_count:<15d} {skipped_count:<15d} {profit_percent:.<14.1f}%")
            else:
                if skipped_count > 0:
                    # If we only skipped games and didn't make any new predictions
                    logger.info(f"{model_name:<30} {'0':<15} {'0':<15} {skipped_count:<15d} {'N/A':<15}")
                else:
                    # No predictions and no skipped games
                    logger.info(f"{model_name:<30} {'0':<15} {'0':<15} {'0':<15} {'0.0':<14}%")
        
        logger.info("-"*95)
        # Add summary line for skipped predictions
        logger.info(f"{'TOTALS':<30} {total_predictions:<15d} {total_profitable:<15d} {total_skipped:<15d}")
        logger.info("="*95)

        return {
            'statusCode': 200,
            'body': json.dumps(f'Successfully processed predictions using {len(models_list)} models and saved {len(all_predictions_across_models)} predictions to {ALL_PREDICTIONS_TABLE}, of which {len([p for p in all_predictions_across_models if p["is_profitable"]])} profitable predictions were saved to {PROFITABLE_GAMES_TABLE}')
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
