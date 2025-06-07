import os
import json
import boto3
import pandas as pd
import numpy as np
import awswrangler as wr
from datetime import datetime, timezone, timedelta, date
from zoneinfo import ZoneInfo
from google.cloud import firestore
import logging
from decimal import Decimal
import time
import traceback

print("UPDATE_RESULTS_FIRESTORE.PY: Script loaded")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables with defaults
AWS_REGION = os.environ.get('CUSTOM_AWS_REGION', 'il-central-1')
ATHENA_DATABASE = os.environ.get('ATHENA_DATABASE', 'winner-db')
RESULTS_TABLE_NAME = os.environ.get('RESULTS_TABLE', 'results')
ATHENA_OUTPUT_BUCKET = os.environ.get('ATHENA_OUTPUT_BUCKET', 'winner-athena-output')

# Initialize Firestore client
try:
    db = firestore.Client()
    logger.info("Firestore client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Firestore client: {e}")
    db = None

# Define Israel Timezone using zoneinfo
israel_tz = ZoneInfo("Asia/Jerusalem")

# Firestore Collection Names
FIRESTORE_PREDICTIONS_COLLECTION = "predictions"
FIRESTORE_PROFITABLE_PREDICTIONS_COLLECTION = "profitable_predictions"

def fetch_results_from_athena():
    """Fetch game results for the past week from Athena."""

    try:
        # Calculate date 7 days ago
        seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching results since {seven_days_ago} from Athena")
        
        # Query to get results from the last 7 days
        query = f"""
        SELECT 
            id, time, league, teama, teamb, scorea, scoreb, date, type
        FROM 
            "{ATHENA_DATABASE}"."{RESULTS_TABLE_NAME}"
        WHERE 
            date >= '{seven_days_ago}'
            AND type = 'Soccer'
        """
        
        logger.info(f"Executing Athena query: {query}")
        
        # Setup boto3 session
        boto3.setup_default_session(region_name=AWS_REGION)
        session = boto3.Session(region_name=AWS_REGION)
        
        df = wr.athena.read_sql_query(
            sql=query,
            database=ATHENA_DATABASE,
            boto3_session=session,
            s3_output=f"s3://{ATHENA_OUTPUT_BUCKET}/"
        )
        
        # Drop duplicate results (keep first occurrence)
        df = df.drop_duplicates(subset=["id"], keep="first")
        
        logger.info(f"Fetched {len(df)} results from Athena")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching results from Athena: {str(e)}")
        raise

def get_pending_profitable_predictions_from_firestore():
    """Get profitable predictions from Firestore that need result updates."""
    if not db:
        logger.error("Firestore client not initialized")
        return []
        
    try:
        logger.info("Fetching pending profitable predictions from Firestore")
        
        collection_ref = db.collection(FIRESTORE_PROFITABLE_PREDICTIONS_COLLECTION)
        pending_predictions = []
        processed_doc_ids = set()
        
        # Get predictions where status != 'completed'
        query_not_completed = collection_ref.where('status', '!=', 'completed')
        for doc in query_not_completed.stream():
            prediction_data = doc.to_dict()
            prediction_data['doc_id'] = doc.id
            prediction_data['collection'] = FIRESTORE_PROFITABLE_PREDICTIONS_COLLECTION
            pending_predictions.append(prediction_data)
            processed_doc_ids.add(doc.id)
        
        # Get predictions where status field doesn't exist
        for doc in collection_ref.stream():
            if doc.id in processed_doc_ids:
                continue
                
            prediction_data = doc.to_dict()
            # Only include if status field doesn't exist
            if 'status' not in prediction_data:
                prediction_data['doc_id'] = doc.id
                prediction_data['collection'] = FIRESTORE_PROFITABLE_PREDICTIONS_COLLECTION
                pending_predictions.append(prediction_data)
        
        logger.info(f"Found {len(pending_predictions)} pending profitable predictions")
        return pending_predictions
        
    except Exception as e:
        logger.error(f"Error fetching pending profitable predictions from Firestore: {e}")
        return []

def calculate_final_score(option: str, score: int) -> float:
    """
    Gets option and score and returns the final score after applying the constraint.
    For example, if the option is "Barcelona (+2)" and the score is 1, the final score will be 3.0
    Return value is float, since some constraints are not integers.
    """
    try:
        opening_bracket_index = option.rfind("(")
        closing_bracket_index = option.rfind(")")
        if opening_bracket_index == -1 and closing_bracket_index == -1:
            return score
        elif opening_bracket_index == -1 or closing_bracket_index == -1:
            raise ValueError(f"Invalid constraint for option {option}")

        constraint = option[opening_bracket_index + 1 : closing_bracket_index]
        if constraint == "0":
            return score
        sign = constraint[0]
        if sign == "-":
            # The negative sign is disregarded to prevent doubling the constraint
            return score
        value = float(constraint[1:])
        value = -value if sign == "-" else value
        return score + value
    except Exception as e:
        logger.error(f"Error calculating final score for option {option} with score {score}: {str(e)}")
        # Return the original score if we can't parse the constraint
        return score

def determine_prediction_result(prediction: dict, results_df: pd.DataFrame):
    """
    Determine if the prediction was correct based on the actual result.
    Uses result_id from the prediction to match results in the Athena DataFrame.
    
    Args:
        prediction: Prediction dict from Firestore
        results_df: DataFrame with game results from Athena
    
    Returns:
        Dictionary with result information or None if no result found
    """
    try:
        result_id = prediction.get('result_id')
        
        if not result_id:
            logger.warning(f"Skipping prediction {prediction.get('game_id', 'unknown')} as it lacks a result_id.")
            return None

        # Find the result row in results_df where the 'id' column matches the prediction's result_id
        result_row = results_df[results_df['id'] == result_id]
        if result_row.empty:
            logger.info(f"No result found using result_id {result_id} for prediction {prediction.get('game_id', 'unknown')}")
            return None

        # Extract scores
        scorea = int(result_row['scorea'].iloc[0]) if 'scorea' in result_row.columns and not pd.isna(result_row['scorea'].iloc[0]) else 0
        scoreb = int(result_row['scoreb'].iloc[0]) if 'scoreb' in result_row.columns and not pd.isna(result_row['scoreb'].iloc[0]) else 0

        # Calculate final scores
        home_team = prediction.get('home_team', '')
        away_team = prediction.get('away_team', '')
        final_scorea = calculate_final_score(home_team, scorea)
        final_scoreb = calculate_final_score(away_team, scoreb)
        
        # Determine actual result
        home_win = final_scorea > final_scoreb
        away_win = final_scoreb > final_scorea
        draw = final_scorea == final_scoreb
        actual_outcome = 'Home Win' if home_win else 'Draw' if draw else 'Away Win'

        # Determine if the specific prediction was correct
        predicted_outcome = prediction.get('prediction', '')
        prediction_correct = predicted_outcome == actual_outcome
        
        # Check if any of the profitable predictions were correct
        home_pred_correct = prediction.get('home_win_is_profitable', False) and home_win
        draw_pred_correct = prediction.get('draw_is_profitable', False) and draw
        away_pred_correct = prediction.get('away_win_is_profitable', False) and away_win
        profitable_prediction_correct = home_pred_correct or draw_pred_correct or away_pred_correct

        result = {
            'home_score': scorea,
            'away_score': scoreb,
            'final_home_score': final_scorea,
            'final_away_score': final_scoreb,
            'actual_result': actual_outcome,
            'prediction_result': prediction_correct,
            'profitable_prediction_correct': profitable_prediction_correct,
            'home_prediction_correct': home_pred_correct,
            'draw_prediction_correct': draw_pred_correct,
            'away_prediction_correct': away_pred_correct,
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error determining prediction result for game {prediction.get('game_id', 'unknown')}: {str(e)}", exc_info=True)
        return None

def update_prediction_with_result(prediction: dict, result: dict) -> dict:
    """
    Update the prediction dict with the result information.
    
    Args:
        prediction: Prediction dict from Firestore
        result: Result dict from determine_prediction_result
    
    Returns:
        Updated prediction dict
    """
    updated_prediction = prediction.copy()
    
    updated_prediction['prediction_result'] = result.get('prediction_result')
    updated_prediction['profitable_prediction_correct'] = result.get('profitable_prediction_correct')
    updated_prediction['actual_result'] = result.get('actual_result')
    updated_prediction['home_score'] = result.get('home_score')
    updated_prediction['away_score'] = result.get('away_score')
    updated_prediction['final_home_score'] = result.get('final_home_score')
    updated_prediction['final_away_score'] = result.get('final_away_score')
    updated_prediction['home_prediction_correct'] = result.get('home_prediction_correct')
    updated_prediction['draw_prediction_correct'] = result.get('draw_prediction_correct')
    updated_prediction['away_prediction_correct'] = result.get('away_prediction_correct')
    updated_prediction['result_updated_at'] = datetime.now(israel_tz).isoformat()
    updated_prediction['status'] = 'completed'
    
    return updated_prediction

def update_firestore_with_results(updated_predictions: list):
    """
    Update Firestore with the prediction results.
    
    Args:
        updated_predictions: List of prediction dicts with result information
    """
    if not db:
        logger.error("Firestore client not initialized")
        return
        
    try:
        predictions_ref = db.collection(FIRESTORE_PREDICTIONS_COLLECTION)
        profitable_ref = db.collection(FIRESTORE_PROFITABLE_PREDICTIONS_COLLECTION)
        
        updated_count = 0
        failed_count = 0
        main_updated_count = 0
        profitable_updated_count = 0
        
        logger.info(f"Starting batch update for {len(updated_predictions)} predictions in Firestore")
        
        for prediction in updated_predictions:
            try:
                doc_id = prediction.get('doc_id')
                collection_name = prediction.get('collection')
                
                if not doc_id:
                    logger.error(f"Skipping prediction update due to missing doc_id: {prediction.get('game_id', 'unknown')}")
                    failed_count += 1
                    continue
                
                # Remove metadata fields from the data before saving
                prediction_data = {k: v for k, v in prediction.items() if k not in ['doc_id', 'collection']}
                
                # Update in the appropriate collection
                if collection_name == FIRESTORE_PREDICTIONS_COLLECTION:
                    predictions_ref.document(doc_id).update(prediction_data)
                    main_updated_count += 1
                elif collection_name == FIRESTORE_PROFITABLE_PREDICTIONS_COLLECTION:
                    profitable_ref.document(doc_id).update(prediction_data)
                    profitable_updated_count += 1
                else:
                    # Default to main collection if collection not specified
                    predictions_ref.document(doc_id).update(prediction_data)
                    main_updated_count += 1
                
                updated_count += 1
                logger.info(f"Updated prediction {doc_id} in {collection_name} - Correct: {prediction.get('prediction_result', False)}")
                
            except Exception as e:
                logger.error(f"Failed to update prediction {prediction.get('game_id', 'unknown')}: {str(e)}")
                failed_count += 1
        
        logger.info(f"Firestore batch update complete. Main collection updated: {main_updated_count}, "
                   f"Profitable collection updated: {profitable_updated_count}, Failed: {failed_count}")
        
    except Exception as e:
        logger.error(f"Error updating predictions in Firestore: {e}")
        raise

def lambda_handler(event, context):
    """Main Lambda function handler."""
    logger.info("Starting update results Firestore Lambda")
    
    try:
        # Fetch results from Athena
        results_df = fetch_results_from_athena()
        
        # Get pending profitable predictions from Firestore
        pending_predictions = get_pending_profitable_predictions_from_firestore()
        
        if not pending_predictions:
            logger.info("No pending profitable predictions found that need result updates")
            return {
                'statusCode': 200,
                'body': json.dumps('No pending profitable predictions found')
            }
        
        updated_predictions = []
        processed_ids = set()  # Keep track of IDs processed in this run

        for prediction in pending_predictions:
            game_id = prediction.get('game_id')
            model_name = prediction.get('model_name')
            doc_id = prediction.get('doc_id')
            collection_name = prediction.get('collection')
            
            if not game_id or not model_name or not doc_id:
                logger.warning(f"Skipping prediction due to missing required fields: {prediction}")
                continue

            # Double-check status 
            if prediction.get('status') == 'completed':
                logger.info(f"Skipping {doc_id} from {collection_name} as its status is already 'completed'.")
                continue

            # Create a unique key for this document in this collection
            unique_key = f"{collection_name}:{doc_id}"
            
            # Prevent processing the same document multiple times within the same run
            if unique_key in processed_ids:
                logger.info(f"Skipping {doc_id} from {collection_name} as it was already processed in this run.")
                continue
                 
            result = determine_prediction_result(prediction, results_df)
            if result:
                updated_prediction = update_prediction_with_result(prediction, result)
                updated_predictions.append(updated_prediction)
                processed_ids.add(unique_key)

                # Log the result
                home_team = prediction.get('home_team', 'unknown')
                away_team = prediction.get('away_team', 'unknown')
                prediction_correct = result.get('prediction_result', False)
                actual_result = result.get('actual_result', 'unknown')
                predicted_outcome = prediction.get('prediction', 'unknown')
                
                logger.info(f"Result for {doc_id} ({home_team} vs {away_team}): "
                           f"Predicted {predicted_outcome}, Actual {actual_result}, Correct: {prediction_correct}")
            else:
                logger.info(f"No result found in Athena for prediction {doc_id}, game_id {game_id}")

        logger.info(f"Processed {len(pending_predictions)} pending profitable predictions, found results for {len(updated_predictions)} predictions")
        
        if updated_predictions:
            update_firestore_with_results(updated_predictions)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Successfully processed {len(pending_predictions)} pending profitable predictions, updated {len(updated_predictions)} with results',
                'predictions_processed': len(pending_predictions),
                'predictions_updated': len(updated_predictions)
            })
        }
    
    except Exception as e:
        logger.error(f"Error in Lambda execution: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }

# For local testing
if __name__ == "__main__":
    lambda_handler(None, None) 