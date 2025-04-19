import json
import boto3
import awswrangler as wr
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Dict, List, Any, Tuple, Optional
import time
from boto3.dynamodb.conditions import Attr, Key

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables with defaults
AWS_REGION = os.environ.get("CUSTOM_AWS_REGION", "il-central-1")
ATHENA_DATABASE = os.environ.get("ATHENA_DATABASE", "winner-db")
ALL_PREDICTIONS_TABLE = os.environ.get("ALL_PREDICTIONS_TABLE", "all-predicted-games")
PROFITABLE_GAMES_TABLE = os.environ.get("PROFITABLE_GAMES_TABLE", "profitable-games")
RESULTS_TABLE_NAME = os.environ.get("RESULTS_TABLE", "results")
ATHENA_OUTPUT_BUCKET = os.environ.get("ATHENA_OUTPUT_BUCKET", "winner-athena-output")

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
dynamodb_client = boto3.client('dynamodb', region_name=AWS_REGION)
athena = boto3.client('athena', region_name=AWS_REGION)

def fetch_results_from_athena() -> pd.DataFrame:
    """
    Fetch game results for the past week from Athena.
    """
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
        
        # Execute query using AWS Wrangler with specific output location
        boto3.setup_default_session(region_name=AWS_REGION)
        session = boto3.Session(region_name=AWS_REGION)
        
        # Use the same approach as in predict_model_lambda
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

def get_pending_profitable_games_from_dynamodb() -> List[Dict]:
    """
    Get games from the profitable games table that have finished but are not marked as completed.
    """
    try:
        profitable_games_table = dynamodb.Table(PROFITABLE_GAMES_TABLE)
        logger.info(f"Fetching past, unprocessed games from DynamoDB table: {PROFITABLE_GAMES_TABLE}")

        three_hours_ago = datetime.now() - timedelta(hours=3)

        # Scan for games that are NOT marked as completed
        filter_expression = Attr('status').ne('completed') | Attr('status').not_exists()
        
        response = profitable_games_table.scan(FilterExpression=filter_expression)
        items = response.get('Items', [])
        
        while 'LastEvaluatedKey' in response:
            response = profitable_games_table.scan(
                FilterExpression=filter_expression,
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get('Items', []))
        
        logger.info(f"Retrieved {len(items)} non-completed games from {PROFITABLE_GAMES_TABLE}")
        
        # Filter games whose match_time is older than the threshold
        pending_profitable_games = []
        for item in items:
            match_time_str = item.get('match_time_str') 
            if not match_time_str:
                 event_date = item.get('event_date')
                 game_time = item.get('game_time')
                 if event_date and game_time:
                     match_time_str = f"{event_date} {game_time}"
                 else:
                     logger.warning(f"Skipping profitable game {item.get('prediction_id', 'unknown')} due to missing time information")
                     continue

            try:
                match_time = datetime.strptime(match_time_str, '%Y-%m-%d %H:%M')
                if match_time < three_hours_ago:
                    pending_profitable_games.append(item)
            except ValueError:
                logger.warning(f"Could not parse match_time_str '{match_time_str}' for prediction {item.get('prediction_id', 'unknown')} in {PROFITABLE_GAMES_TABLE}")
            except Exception as e:
                 logger.error(f"Error processing time for prediction {item.get('prediction_id', 'unknown')} in {PROFITABLE_GAMES_TABLE}: {str(e)}")

        logger.info(f"Found {len(pending_profitable_games)} past profitable games that need result updates")
        return pending_profitable_games
    
    except Exception as e:
        logger.error(f"Error retrieving games from {PROFITABLE_GAMES_TABLE}: {str(e)}")
        raise

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

def determine_bet_result(game: Dict, results_df: pd.DataFrame) -> Optional[Dict]:
    """
    Determine if the prediction for a game was correct based on the actual result.
    Uses game_id from the profitable game item to match results.
    
    Args:
        game: Game dict from DynamoDB (profitable-games table)
        results_df: DataFrame with game results from Athena
    
    Returns:
        Dictionary with result information or None if no result found
    """
    try:
        result_row = pd.DataFrame() 
        game_id = game.get('game_id') # Use game_id from the profitable game item

        if not game_id:
            logger.warning(f"Skipping profitable game prediction {game.get('prediction_id')} as it lacks a game_id.")
            return None

        result_row = results_df[results_df['id'] == game_id]
        if result_row.empty:
             logger.info(f"No result found using game_id {game_id} for prediction {game.get('prediction_id')}")
             return None

        # Extract scores
        scorea = int(result_row['scorea'].iloc[0]) if 'scorea' in result_row.columns and not pd.isna(result_row['scorea'].iloc[0]) else 0
        scoreb = int(result_row['scoreb'].iloc[0]) if 'scoreb' in result_row.columns and not pd.isna(result_row['scoreb'].iloc[0]) else 0

        # Calculate final scores
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        final_scorea = calculate_final_score(home_team, scorea)
        final_scoreb = calculate_final_score(away_team, scoreb)
        
        # Determine actual result
        home_win = final_scorea > final_scoreb
        away_win = final_scoreb > final_scorea
        draw = final_scorea == final_scoreb
        actual_outcome = 'Home Win' if home_win else 'Draw' if draw else 'Away Win'

        # Determine if the prediction associated with this profitable entry was correct
        # Use the specific profitable flags from the profitable game item
        home_pred_correct = game.get('home_win_is_profitable', False) and home_win
        draw_pred_correct = game.get('draw_is_profitable', False) and draw
        away_pred_correct = game.get('away_win_is_profitable', False) and away_win
        prediction_correct = home_pred_correct or draw_pred_correct or away_pred_correct

        result = {
            'home_score': scorea,
            'away_score': scoreb,
            'final_home_score': final_scorea,
            'final_away_score': final_scoreb,
            'actual_result': actual_outcome,
            'prediction_result': prediction_correct,
            'home_prediction_correct': home_pred_correct,
            'draw_prediction_correct': draw_pred_correct,
            'away_prediction_correct': away_pred_correct,
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error determining bet result for profitable game prediction {game.get('prediction_id', 'unknown')}: {str(e)}", exc_info=True)
        return None

def update_game_with_result(game: Dict, result: Dict) -> Dict:
    """
    Update the game dict with the result information.
    
    Args:
        game: Game dict from DynamoDB
        result: Result dict from determine_bet_result
    
    Returns:
        Updated game dict
    """
    updated_game = game.copy()
    
    updated_game['prediction_result'] = result.get('prediction_result') # Overall correctness
    updated_game['actual_result'] = result.get('actual_result')
    updated_game['home_score'] = result.get('home_score')
    updated_game['away_score'] = result.get('away_score')
    updated_game['final_home_score'] = result.get('final_home_score')
    updated_game['final_away_score'] = result.get('final_away_score')
    # Add individual results if needed
    # updated_game['home_prediction_correct'] = result.get('home_prediction_correct')
    # updated_game['draw_prediction_correct'] = result.get('draw_prediction_correct')
    # updated_game['away_prediction_correct'] = result.get('away_prediction_correct')
    updated_game['result_updated_at'] = datetime.now().isoformat()
    updated_game['status'] = 'completed' # Mark as completed
    
    return updated_game

def convert_floats_to_decimal(obj):
    """Convert float values to Decimal for DynamoDB compatibility."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimal(i) for i in obj]
    elif isinstance(obj, date):
        return obj.isoformat()
    else:
        return obj

def update_dynamodb_with_results(updated_games: List[Dict]):
    """
    Update the profitable-games DynamoDB table with the game results.
    
    Args:
        updated_games: List of profitable game dicts with result information
    """
    table = dynamodb.Table(PROFITABLE_GAMES_TABLE) # Target the profitable games table
    updated_count = 0
    failed_count = 0
    
    logger.info(f"Starting batch update for {len(updated_games)} games in {PROFITABLE_GAMES_TABLE}")

    with table.batch_writer() as batch:
        for game in updated_games:
            try:
                processed_game = convert_floats_to_decimal(game)
                prediction_id = processed_game.get('prediction_id') 
                
                if not prediction_id:
                     logger.error(f"Skipping profitable game update due to missing prediction_id: {game}")
                     failed_count += 1
                     continue
                
                # Use PutItem to update the item in profitable-games table
                batch.put_item(Item=processed_game)
                updated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to update profitable game {game.get('prediction_id', 'unknown')} in DynamoDB: {str(e)}", exc_info=True)
                failed_count += 1

    logger.info(f"DynamoDB batch update complete for {PROFITABLE_GAMES_TABLE}. Successfully updated: {updated_count}, Failed: {failed_count}")

def lambda_handler(event, context):
    """Main Lambda function handler."""
    logger.info("Starting update results Lambda (profitable games only)")
    
    try:
        results_df = fetch_results_from_athena()
        pending_profitable_games = get_pending_profitable_games_from_dynamodb() # Updated function call
        
        if not pending_profitable_games:
            logger.info("No pending profitable games found that need result updates")
            return {
                'statusCode': 200,
                'body': json.dumps('No pending profitable games found')
            }
        
        updated_games = []
        for game in pending_profitable_games:
            result = determine_bet_result(game, results_df)
            if result:
                updated_game = update_game_with_result(game, result)
                updated_games.append(updated_game)
                
                prediction_id = game.get('prediction_id', 'unknown')
                home_team = game.get('home_team', 'unknown')
                away_team = game.get('away_team', 'unknown')
                result_correct = result.get('prediction_result', False)
                actual_result = result.get('actual_result', 'unknown')
                
                logger.info(f"Profitable Prediction {prediction_id} ({home_team} vs {away_team}): " 
                           f"Actual {actual_result}, Correct: {result_correct}")
            else:
                 logger.info(f"No result found in Athena for profitable prediction {game.get('prediction_id')}, game_id {game.get('game_id')}")

        logger.info(f"Processed {len(pending_profitable_games)} pending profitable games, found results for {len(updated_games)} games")
        
        if updated_games:
            update_dynamodb_with_results(updated_games)
        
        return {
            'statusCode': 200,
            'body': json.dumps(f'Successfully processed {len(pending_profitable_games)} profitable games, updated {len(updated_games)} with results')
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