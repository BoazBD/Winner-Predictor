import os
import json
import boto3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import awswrangler as wr
from datetime import datetime, timezone, timedelta, date
from zoneinfo import ZoneInfo
from googletrans import Translator
from google.cloud import firestore
import logging
from decimal import Decimal
import time
import traceback
from tensorflow.keras.initializers import Orthogonal
from team_translations import TEAM_TRANSLATIONS, get_translation

print("LAMBDA_FUNCTION.PY: Script loaded")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if running locally or in Lambda (simplified for Firestore Lambda, assuming not local for now)
IS_LOCAL = os.environ.get('IS_LOCAL', 'False').lower() == 'true'

# Environment variables with defaults (some will be from Lambda env, others are for consistency)
AWS_REGION = os.environ.get('AWS_REGION', 'il-central-1')
ATHENA_DATABASE = os.environ.get('ATHENA_DATABASE', 'winner-db')
S3_BUCKET = os.environ.get('S3_BUCKET', 'winner-site-data')
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'lstm') # Default model type if not in filename
EPOCHS = os.environ.get('EPOCHS', '100')     # Default epochs if not in filename
MAX_SEQ = os.environ.get('MAX_SEQ', '12')      # Default max_seq if not in filename
THRESHOLD = float(os.environ.get('THRESHOLD', '0.00')) # Profitability threshold

# Define constants for features processing
MIN_BET = 1.0 # Using float for consistency with calculations
MAX_BET = 10.0 # Using float
RATIOS = ["ratio1", "ratio2", "ratio3"]

# Initialize Firestore client
db = firestore.Client()

# Initialize translator
translator = Translator()

# Define Israel Timezone using zoneinfo
israel_tz = ZoneInfo("Asia/Jerusalem")

# Initialize translation cache to avoid repeated API calls for the same text
_translation_cache = {}

# Firestore Collection Names
FIRESTORE_PREDICTIONS_COLLECTION = "predictions"
FIRESTORE_PROFITABLE_PREDICTIONS_COLLECTION = "profitable_predictions"

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
    "אליפות העולם לקבוצות",
    "גביע הטוטו",
]

# Custom InputLayer class to handle batch_shape parameter
class CustomInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['input_shape'] = kwargs.pop('batch_shape')[1:]
        super(CustomInputLayer, self).__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config: # Ensure this check is also here
            config['input_shape'] = config.pop('batch_shape')[1:]
        return cls(**config)

tf.keras.utils.get_custom_objects().update({
    'CustomInputLayer': CustomInputLayer,
    'Orthogonal': Orthogonal
})

def list_models_in_s3():
    models = []
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix='models/')
        for item in response.get('Contents', []):
            if item['Key'].endswith('.h5'):
                filename = os.path.basename(item['Key'])
                local_path = f"/tmp/{filename}"
                models.append({'key': item['Key'], 'local_path': local_path})
        logger.info(f"Found {len(models)} models in S3 bucket s3://{S3_BUCKET}/models/")
        if not models:
            logger.warning("No .h5 models found in S3. Predictions will likely fail or be empty.")
    except Exception as e:
        print(f"LIST_MODELS_IN_S3: Exception: {str(e)}")
        logger.error(f"Error listing models in S3: {str(e)}")
        # Fallback or raise? For consistency, DynamoDB lambda has a fallback but it's for IS_LOCAL.
        # Here, if S3 listing fails, it's a bigger issue for Lambda.
    return models

def download_model(model_info):
    print(f"DOWNLOAD_MODEL: Entered for {model_info['key']}")
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    try:
        s3_client.download_file(S3_BUCKET, model_info['key'], model_info['local_path'])
        return model_info['local_path']
    except Exception as e:
        print(f"DOWNLOAD_MODEL: Exception for {model_info['key']}: {str(e)}")
        raise

def fetch_games_from_athena():
    """Fetch game data from the past 7 days from Athena, including historical odds."""

    try:
        # First, get the latest run_time in the dataset
        latest_run_time_query = f'SELECT MAX(run_time) as latest_run_time FROM "{ATHENA_DATABASE}"."api_odds"'
        logger.info(f"Latest run_time query: {latest_run_time_query}")

        # Setup boto3 session
        boto3.setup_default_session(region_name=AWS_REGION)
        session = boto3.Session(region_name=AWS_REGION)

        latest_run_time_df = wr.athena.read_sql_query(
            sql=latest_run_time_query, database=ATHENA_DATABASE, boto3_session=session
        )

        latest_run_time = latest_run_time_df["latest_run_time"].iloc[0]

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
            AND unique_id IN (
                SELECT unique_id 
                FROM "{ATHENA_DATABASE}"."api_odds"
                WHERE run_time = '{latest_run_time}'
            )
        ORDER BY 
            unique_id, run_time
        """

        # Execute query using AWS Wrangler
        df = wr.athena.read_sql_query(
            sql=query, database=ATHENA_DATABASE, boto3_session=session
        )

        # Convert run_time to datetime if it's not already
        df["run_time"] = pd.to_datetime(df["run_time"])

        # First, calculate the global maximum run_time in the processed DataFrame
        max_run_time = df["run_time"].max()

        # Identify the unique_ids (game IDs) that have at least one row with the max_run_time
        games_with_latest_run_time = df.loc[
            df["run_time"] == max_run_time, "unique_id"
        ].unique()

        # Keep only those games in the DataFrame (i.e. rows where unique_id is in the list above)
        df = df[df["unique_id"].isin(games_with_latest_run_time)].copy()

        logger.info(
            f"Fetched {len(df)} odds records for {df['unique_id'].nunique()} unique games from Athena"
        )
        return df

    except Exception as e:
        logger.error(f"Error fetching games from Athena: {str(e)}")


def scale_features(features: pd.DataFrame) -> pd.DataFrame:
    # Assuming MIN_BET and MAX_BET are defined globally (e.g., 1.0 and 10.0)
    return (features - MIN_BET) / (MAX_BET - MIN_BET) # Corrected scaling denominator

def prepare_features(features: pd.DataFrame, max_seq_len: int) -> np.array:
    features_scaled = scale_features(features) # features is a DataFrame
    features_scaled_np = features_scaled.to_numpy() # Convert to NumPy array
    # Take the last max_seq_len rows
    if features_scaled_np.shape[0] < max_seq_len:
        # This case should ideally be filtered out before calling prepare_features
        # or handled by padding if the model expects fixed size input
        logger.warning(f"Not enough data for max_seq_len {max_seq_len}, found {features_scaled_np.shape[0]}. Padding or error might occur.")
        # For now, to avoid error, let's attempt to reshape what we have, but this is not ideal.
        # The DynamoDB version filters this out in preprocess_data.
        # This function expects features_scaled_np to have at least max_seq_len rows after process_ratios.
    features_final = features_scaled_np[-max_seq_len:, :] 
    return features_final.reshape(1, max_seq_len, features_final.shape[1]) # Reshape for single prediction (batch size 1)

def remove_consecutive_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # Assuming RATIOS is defined globally (e.g., ["ratio1", "ratio2", "ratio3"])
    shifted = df[RATIOS].shift()
    mask = ~((df[RATIOS] == shifted).all(axis=1))
    return df[mask]

def process_ratios(group_df: pd.DataFrame) -> pd.DataFrame: # CHANGED: Returns DataFrame
    # Assuming RATIOS is defined globally
    group_df[RATIOS] = group_df[RATIOS].astype(float)
    group_df = remove_consecutive_duplicates(group_df)
    return group_df # Returns DataFrame, not .values

def preprocess_data(df: pd.DataFrame) -> list:
    game_ids = df['unique_id'].unique()
    logger.info(f"Processing {len(game_ids)} unique games for preprocessing.")
    valid_game_data = []
    max_seq = int(MAX_SEQ) # Ensure MAX_SEQ from env is int

    for game_id in game_ids:
        game_odds_history = df[df['unique_id'] == game_id].sort_values('run_time')
        
        # Create game_df with just the ratio columns for processing
        game_df_ratios = pd.DataFrame({
            'ratio1': game_odds_history['ratio1'],
            'ratio2': game_odds_history['ratio2'],
            'ratio3': game_odds_history['ratio3'],
        })
        
        processed_game_df_ratios = process_ratios(game_df_ratios) # This is a DataFrame
        
        if processed_game_df_ratios.shape[0] < max_seq:
            continue
            
        # features_for_model is a 3D NumPy array (1, max_seq, num_features)
        features_for_model = prepare_features(processed_game_df_ratios, max_seq) 
        
        latest_game_info = game_odds_history.iloc[-1].to_dict()
        
        valid_game_data.append({
            'game_id': game_id,
            'features': features_for_model,
            'info': latest_game_info # Changed from 'latest_game' to 'info' for consistency
        })
   
    logger.info(f"Found {len(valid_game_data)} games with enough data for prediction after preprocessing.")
    return valid_game_data

def translate_to_english(text, context=""):
    print(f"TRANSLATE_TO_ENGLISH: Text='{text}', Context='{context}'")
    if not text or not isinstance(text, str):
        return ""
    if all(ord(c) < 1200 for c in text): # Basic check if already English-like
        return text
        
    cache_key = f"{text}:{context}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]
    
    # Fallback to TEAM_TRANSLATIONS dictionary (from team_translations.py)
    if text in TEAM_TRANSLATIONS:
        translated_text = TEAM_TRANSLATIONS[text]
        _translation_cache[cache_key] = translated_text
        return translated_text
    
    try:
        time.sleep(0.2) # Reduced delay from 0.5 to 0.2 for potentially many translations
        translation = translator.translate(text, src='iw', dest='en')
        translated_text = translation.text
        _translation_cache[cache_key] = translated_text
        return translated_text
    except Exception as e:
        print(f"TRANSLATE_TO_ENGLISH: Exception: {str(e)}")
        logger.error(f"Translation error for text '{text}' with context '{context}': {str(e)}")
        return text # Fallback to original text on error

def make_predictions(model, valid_game_data, model_name_from_handler):
    all_game_predictions = []
    
    for game_data in valid_game_data:
        game_id = game_data['game_id']
        features = game_data['features'] # This is already shaped (1, max_seq, 3)
        info = game_data['info'] # Changed from 'latest_game'
        
        # Make prediction (features should already be correctly shaped by prepare_features)
        # If the model name ends with 'ev.h5', pass both sequence and final odds as inputs
        if model_name_from_handler.endswith("ev"):
            # features is (1, max_seq, 3)
            # Extract the latest odds from info
            final_odds = np.array([[float(info.get('ratio1', 0.0)), float(info.get('ratio2', 0.0)), float(info.get('ratio3', 0.0))]]).astype(float).reshape(1, 3)
            print(f"Final odds: {final_odds}")  
            prediction_probs = model.predict([features, final_odds], verbose=0)
        else:
            prediction_probs = model.predict(features, verbose=0)
        logger.info(f"Raw prediction for game {game_id} with model {model_name_from_handler}: {prediction_probs}")
        print(f"Raw prediction for game {game_id} with model {model_name_from_handler}: {prediction_probs}")
        
        home_team = info.get('option1')
        away_team = info.get('option3')
        event_date_val = info.get('event_date') # This is pd.Timestamp
        game_time = info.get('time')
        league = info.get('league')
        result_id = info.get('result_id')

        english_home_team = translate_to_english(home_team, "football team")
        english_away_team = translate_to_english(away_team, "football team")
        english_league = translate_to_english(league, "football league")

        home_odds = float(info.get('ratio1', 0.0))
        draw_odds = float(info.get('ratio2', 0.0))
        away_odds = float(info.get('ratio3', 0.0))

        outcomes_data = []
        prob_names = ['home_win_prob', 'draw_prob', 'away_win_prob']
        odds_values = [home_odds, draw_odds, away_odds]
        outcome_names_display = ['Home Win', 'Draw', 'Away Win']
        ev_names = ['home_win_ev', 'draw_ev', 'away_win_ev']
        profitable_names = ['home_win_is_profitable', 'draw_is_profitable', 'away_win_is_profitable']

        current_prediction_details = {}

        for i, name_display in enumerate(outcome_names_display):
            prob = float(prediction_probs[0][i])
            odds = odds_values[i]
            # Ensure odds are not zero to prevent DivisionByZeroError
            profit_threshold = (1 / odds) + THRESHOLD if odds > 0 else float('inf') 
            ev = (prob * odds) - 1 if odds > 0 else -1.0 # EV is -1 if odds are 0
            is_prof = prob > profit_threshold if odds > 0 else False

            current_prediction_details[prob_names[i]] = prob
            current_prediction_details[ev_names[i]] = ev
            current_prediction_details[profitable_names[i]] = is_prof
            
            outcomes_data.append({
                'name': name_display,
                'probability': prob,
                'odds': odds,
                'expected_value': ev,
                'is_profitable': is_prof
            })

        best_outcome_obj = max(outcomes_data, key=lambda x: x['expected_value'])
        
        prediction_timestamp_dt = datetime.now(israel_tz)

        event_date_str_formatted = ""
        if isinstance(event_date_val, pd.Timestamp):
            event_date_str_formatted = event_date_val.strftime('%Y-%m-%d')
        elif isinstance(event_date_val, date): # Handle if it's a date object from somewhere
            event_date_str_formatted = event_date_val.isoformat()
        elif isinstance(event_date_val, str): # If it's already a string
            event_date_str_formatted = event_date_val 

        match_time_str_val = f"{event_date_str_formatted} {game_time}" if event_date_str_formatted and game_time else datetime.now(israel_tz).strftime('%Y-%m-%d %H:%M')

        game_pred_record = {
            'game_id': game_id,
            'result_id': result_id,
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'english_home_team': english_home_team,
            'english_away_team': english_away_team,
            'english_league': english_league,
            'event_date': event_date_str_formatted, # Store formatted date string
            'game_time': game_time,
            'match_time': match_time_str_val, # Renamed from match_time_str for Firestore consistency, but content is same
            'prediction_timestamp': prediction_timestamp_dt.isoformat(),
            'timestamp': datetime.now(israel_tz).isoformat(), # Add timestamp field using Israeli timezone
            'prediction': best_outcome_obj['name'],
            'model_name': model_name_from_handler, # Use full model name passed from handler
            'model_type': model_name_from_handler.split('_')[0] if '_' in model_name_from_handler else model_name_from_handler,
            'odds': {
                'home': home_odds,
                'draw': draw_odds,
                'away': away_odds
            },
            **current_prediction_details, # Unpack all probs, EVs, is_profitable flags
            'status': 'pending',
            'expected_value': best_outcome_obj['expected_value'], # Overall best EV for sorting
            'is_profitable': any(o['is_profitable'] for o in outcomes_data) # Overall profitability
        }
        all_game_predictions.append(game_pred_record)

    all_game_predictions.sort(key=lambda x: x['expected_value'], reverse=True)
    logger.info(f"Model {model_name_from_handler} produced {len(all_game_predictions)} predictions.")
    return all_game_predictions

def get_already_predicted_profitable_games(model_name):
    """Get list of game IDs that have already been predicted as profitable for this model"""
    if not db:
        logger.error("Firestore client not initialized")
        return set()
    
    try:
        # Query profitable predictions for this specific model
        profitable_ref = db.collection(FIRESTORE_PROFITABLE_PREDICTIONS_COLLECTION)
        query = profitable_ref.where('model_name', '==', model_name)
        
        profitable_game_ids = set()
        docs = query.stream()
        
        for doc in docs:
            doc_data = doc.to_dict()
            game_id = doc_data.get('game_id')
            if game_id:
                profitable_game_ids.add(game_id)
        
        logger.info(f"Found {len(profitable_game_ids)} games already predicted as profitable for model {model_name}")
        return profitable_game_ids
        
    except Exception as e:
        logger.error(f"Error querying already predicted profitable games for model {model_name}: {e}")
        return set()

def filter_already_predicted_games(valid_game_data, model_name):
    """Filter out games that have already been predicted as profitable for this model"""
    already_predicted = get_already_predicted_profitable_games(model_name)
    
    if not already_predicted:
        logger.info(f"No games found as already predicted profitable for model {model_name}")
        return valid_game_data
    
    # Filter out games that have already been predicted as profitable
    filtered_games = []
    for game_data in valid_game_data:
        game_id = game_data['game_id']
        if game_id not in already_predicted:
            filtered_games.append(game_data)
        else:
            logger.info(f"Skipping game {game_id} - already predicted as profitable for model {model_name}")
    
    logger.info(f"Filtered from {len(valid_game_data)} to {len(filtered_games)} games for model {model_name}")
    return filtered_games

def save_predictions(predictions):
    """Save predictions to Firestore"""
    if not db:
        logger.error("Firestore client not initialized")
        return
        
    try:
        # Get references to collections
        predictions_ref = db.collection(FIRESTORE_PREDICTIONS_COLLECTION)
        profitable_ref = db.collection(FIRESTORE_PROFITABLE_PREDICTIONS_COLLECTION)
        
        # Track success and error counts
        success_count = 0
        error_count = 0
        profitable_count = 0
        
        # Save each prediction
        for pred in predictions:
            try:
                # Create a unique document ID using game_id, model_name, prediction, and timestamp
                prediction_outcome_for_id = pred['prediction'].replace(' ', '_')
                # Include timestamp to make each prediction run unique
                timestamp_for_id = datetime.fromisoformat(pred['prediction_timestamp']).strftime('%Y%m%d_%H%M%S')
                doc_id = f"{pred['game_id']}_{pred['model_name']}_{prediction_outcome_for_id}_{timestamp_for_id}"

                # Construct model_timestamp in the desired human-readable format
                dt_object = datetime.fromisoformat(pred['prediction_timestamp'])

                # Since pred['prediction_timestamp'] is generated using israel_tz, 
                # we can be explicit about the timezone display.
                display_timezone_str = "Israel Time" 
                
                pred['model_timestamp'] = dt_object.strftime(f"%b %d, %Y at %I:%M:%S %p ({display_timezone_str})")
                
                # Save to predictions collection
                if not IS_LOCAL:
                    predictions_ref.document(doc_id).set(pred)
                success_count += 1
                
                # If profitable, save to profitable_predictions collection
                # Use the same doc_id for consistency if a prediction is also profitable
                if pred['is_profitable']:
                    if not IS_LOCAL:
                        profitable_ref.document(doc_id).set(pred)
                    print(f"BBBBB: Saving profitable prediction: {pred}")
                    profitable_count += 1
                    
            except Exception as e:
                logger.error(f"Error saving prediction for game {pred.get('game_id')}: {e}")
                error_count += 1
        
        logger.info(f"Saved {success_count} predictions ({profitable_count} profitable) with {error_count} errors")
        
    except Exception as e:
        logger.error(f"Error saving predictions to Firestore: {e}")
        raise

def lambda_handler(event, context):
    
    all_predictions_for_all_models = []
    models_processed_count = 0

    try:
        print("LAMBDA_HANDLER: Listing models from S3")
        models_list = list_models_in_s3()
        if not models_list:
            logger.warning("No models found in S3. Exiting.")
            return {'statusCode': 200, 'body': json.dumps('No models found in S3. 0 predictions made.')}

        print(f"LAMBDA_HANDLER: Found {len(models_list)} models. Fetching Athena data.")
        df_athena = fetch_games_from_athena()
        print(f"LAMBDA_HANDLER: Fetched {len(df_athena)} rows from Athena.")

        if df_athena.empty:
            logger.warning("DataFrame from Athena is empty. No games to process for any model.")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'No games fetched from Athena. 0 predictions made across all models.',
                    'predictions_count': 0
                })
            }

        print("LAMBDA_HANDLER: Preprocessing Athena data once for all models.")
        valid_game_data_all = preprocess_data(df_athena) # Preprocess once
        logger.info(f"Number of valid games after one-time preprocessing: {len(valid_game_data_all)}")
        print(f"LAMBDA_HANDLER: Valid games after one-time preprocessing: {len(valid_game_data_all)}")

        if not valid_game_data_all:
            logger.warning("No valid games after preprocessing. Exiting.")
            return {'statusCode': 200, 'body': json.dumps('No valid games after preprocessing. 0 predictions made.')}

        for model_info in models_list:
            model_key = model_info['key']
            model_filename = os.path.basename(model_key)
            # Derive model_name from filename, e.g., lstm_100_12_v1 from models/lstm_100_12_v1.h5
            derived_model_name = model_filename.split('.h5')[0]
            if derived_model_name.startswith("models/"): # handle if prefix still there, though basename should remove it
                derived_model_name = derived_model_name.split("models/")[-1]
            
            print(f"LAMBDA_HANDLER: Processing model: {derived_model_name} from {model_key}")
            try:
                # Filter out games that have already been predicted as profitable for this model
                model_specific_games = filter_already_predicted_games(valid_game_data_all, derived_model_name)
                
                if not model_specific_games:
                    logger.info(f"No new games to predict for model {derived_model_name} - all games already predicted as profitable")
                    print(f"LAMBDA_HANDLER: No new games for model {derived_model_name}")
                    continue
                
                local_model_path = download_model(model_info)
                loaded_model = load_model(local_model_path) # Uses global custom objects
                print(f"LAMBDA_HANDLER: Model {derived_model_name} loaded.")

                # Pass the filtered games for this specific model
                model_specific_predictions = make_predictions(loaded_model, model_specific_games, derived_model_name)
                logger.info(f"Model {derived_model_name} made {len(model_specific_predictions)} predictions.")
                print(f"LAMBDA_HANDLER: Model {derived_model_name} made {len(model_specific_predictions)} predictions.")
                
                all_predictions_for_all_models.extend(model_specific_predictions)
                models_processed_count += 1

                del loaded_model # Clean up memory
                if os.path.exists(local_model_path):
                    os.remove(local_model_path)
                import gc
                gc.collect()

            except Exception as model_e:
                print(f"LAMBDA_HANDLER: Error processing model {derived_model_name}: {str(model_e)}")
                logger.error(f"Error processing model {derived_model_name} from {model_key}: {str(model_e)}")
                traceback.print_exc()
                # Continue to the next model
        
        if all_predictions_for_all_models:
            print(f"LAMBDA_HANDLER: Saving {len(all_predictions_for_all_models)} total predictions to Firestore.")
            print(f"BBBBB: All predictions: {all_predictions_for_all_models}")

            save_predictions(all_predictions_for_all_models) # Save all collected predictions
            print("LAMBDA_HANDLER: All predictions saved.")
        else:
            logger.info("No pßredictions made from any model after processing all models.")
            print("LAMBDA_HANDLER: No predictions made from any model.")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Predictions completed. Processed {models_processed_count}/{len(models_list)} models. Total predictions: {len(all_predictions_for_all_models)}',
                'total_predictions_generated': len(all_predictions_for_all_models)
            })
        }
        
    except Exception as e:
        print(f"LAMBDA_HANDLER: Top-level Exception: {str(e)}")
        logger.error(f"Error in lambda_handler: {e}")
        traceback.print_exc()
        print("LAMBDA_HANDLER: Full traceback printed via traceback.print_exc() for top-level error")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        } 

# Debug section for local testing
if __name__ == "__main__":
    # Mock event and context for local testing
    mock_event = {}
    mock_context = type('MockContext', (), {
        'function_name': 'predict_model_firestore',
        'memory_limit_in_mb': 1024,
        'invoked_function_arn': 'arn:aws:lambda:local:123456789012:function:predict_model_firestore',
        'aws_request_id': 'local-request-id'
    })()
    
    print("Starting local lambda execution...")
    result = lambda_handler(mock_event, mock_context)
    print(f"Lambda execution completed. Result: {result}") 
