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

def get_pending_games_from_dynamodb() -> List[Dict]:
    """
    Get games from the profitable games table that don't have a result yet.
    """
    try:
        # Get DynamoDB table
        profitable_games_table = dynamodb.Table(PROFITABLE_GAMES_TABLE)
        
        logger.info(f"Fetching games without results from DynamoDB table: {PROFITABLE_GAMES_TABLE}")
        
        # Get all games from the table (scan operation)
        response = profitable_games_table.scan()
        items = response.get('Items', [])
        
        # Continue scanning if we haven't got all items
        while 'LastEvaluatedKey' in response:
            response = profitable_games_table.scan(
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get('Items', []))
        
        logger.info(f"Retrieved {len(items)} games from DynamoDB")
        
        # Filter games that have a result_id but don't have a prediction_result
        pending_games = []
        for item in items:
            # Check if the game has a result_id but doesn't have a prediction_result
            if (item.get('result_id') and 
                'prediction_result' not in item):
                pending_games.append(item)
        
        logger.info(f"Found {len(pending_games)} pending games that need result updates")
        return pending_games
    
    except Exception as e:
        logger.error(f"Error retrieving games from DynamoDB: {str(e)}")
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
    
    Args:
        game: Game dict from DynamoDB
        results_df: DataFrame with game results
    
    Returns:
        Dictionary with result information or None if no result found
    """
    try:
        result_id = game.get('result_id')
        if not result_id:
            logger.warning(f"No result_id for game {game.get('id')}")
            return None
        
        # Find the result in the results DataFrame
        result_row = results_df[results_df['id'] == result_id]
        if result_row.empty:
            logger.info(f"No result found for game {game.get('id')} with result_id {result_id}")
            return None
        
        # Extract scores
        scorea = int(result_row['scorea'].iloc[0])
        scoreb = int(result_row['scoreb'].iloc[0])
        
        # Calculate final scores with any constraints
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        final_scorea = calculate_final_score(home_team, scorea)
        final_scoreb = calculate_final_score(away_team, scoreb)
        
        # Determine actual result
        home_win = final_scorea > final_scoreb
        away_win = final_scoreb > final_scorea
        draw = final_scorea == final_scoreb
        
        # Determine which prediction was made
        prediction = game.get('prediction', '')
        
        # Get which bet type had the highest EV
        home_win_ev = float(game.get('home_win_ev', 0))
        draw_ev = float(game.get('draw_ev', 0))
        away_win_ev = float(game.get('away_win_ev', 0))
        
        highest_ev = max(home_win_ev, draw_ev, away_win_ev)
        
        bet_home = home_win_ev == highest_ev
        bet_draw = draw_ev == highest_ev
        bet_away = away_win_ev == highest_ev
        
        # If there's a tie in EV, use the explicit prediction field
        if prediction == 'Home Win':
            bet_home = True
            bet_draw = bet_away = False
        elif prediction == 'Draw':
            bet_draw = True
            bet_home = bet_away = False
        elif prediction == 'Away Win':
            bet_away = True
            bet_home = bet_draw = False
        
        # Determine if prediction was correct
        prediction_correct = (bet_home and home_win) or (bet_draw and draw) or (bet_away and away_win)
        
        result = {
            'home_score': scorea,
            'away_score': scoreb,
            'final_home_score': final_scorea,
            'final_away_score': final_scoreb,
            'actual_result': 'Home Win' if home_win else 'Draw' if draw else 'Away Win',
            'prediction_correct': prediction_correct,
            'bet_home': bet_home,
            'bet_draw': bet_draw,
            'bet_away': bet_away,
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error determining bet result for game {game.get('id')}: {str(e)}")
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
    # Make a copy of the game dict to avoid modifying the original
    updated_game = game.copy()
    
    # Add result information
    updated_game['prediction_result'] = result.get('prediction_correct')
    updated_game['actual_result'] = result.get('actual_result')
    updated_game['home_score'] = result.get('home_score')
    updated_game['away_score'] = result.get('away_score')
    updated_game['final_home_score'] = result.get('final_home_score')
    updated_game['final_away_score'] = result.get('final_away_score')
    updated_game['result_updated_at'] = datetime.now().isoformat()
    updated_game['status'] = 'completed'
    
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
    Update the DynamoDB table with the game results.
    
    Args:
        updated_games: List of game dicts with result information
    """
    # Get DynamoDB tables
    profitable_games_table = dynamodb.Table(PROFITABLE_GAMES_TABLE)
    all_predictions_table = dynamodb.Table(ALL_PREDICTIONS_TABLE)
    
    try:
        logger.info(f"Updating {len(updated_games)} games with results in DynamoDB")
        
        # Update profitable-games table
        with profitable_games_table.batch_writer() as batch:
            for game in updated_games:
                # Convert floats to Decimal
                game_dict = convert_floats_to_decimal(game)
                
                try:
                    batch.put_item(Item=game_dict)
                    logger.info(f"Updated game {game.get('id')} in {PROFITABLE_GAMES_TABLE} with result")
                except Exception as e:
                    logger.error(f"Error updating game {game.get('id')} in {PROFITABLE_GAMES_TABLE}: {str(e)}")
        
        # Also update all-predictions table
        with all_predictions_table.batch_writer() as batch:
            for game in updated_games:
                # Get the game from all-predictions table first
                try:
                    response = all_predictions_table.get_item(Key={'id': game.get('id')})
                    if 'Item' in response:
                        # Update with result information
                        item = response['Item']
                        item['prediction_result'] = game.get('prediction_result')
                        item['actual_result'] = game.get('actual_result')
                        item['home_score'] = game.get('home_score')
                        item['away_score'] = game.get('away_score')
                        item['final_home_score'] = game.get('final_home_score')
                        item['final_away_score'] = game.get('final_away_score')
                        item['result_updated_at'] = game.get('result_updated_at')
                        item['status'] = 'completed'
                        
                        # Convert floats to Decimal
                        item_dict = convert_floats_to_decimal(item)
                        
                        # Update in DynamoDB
                        batch.put_item(Item=item_dict)
                        logger.info(f"Updated game {game.get('id')} in {ALL_PREDICTIONS_TABLE} with result")
                except Exception as e:
                    logger.error(f"Error updating game {game.get('id')} in {ALL_PREDICTIONS_TABLE}: {str(e)}")
        
        logger.info("Successfully updated games with results in DynamoDB")
    
    except Exception as e:
        logger.error(f"Error updating games with results in DynamoDB: {str(e)}")
        raise

def lambda_handler(event, context):
    """Main Lambda function handler."""
    logger.info("Starting update results Lambda")
    
    try:
        # Step 1: Fetch game results from Athena
        results_df = fetch_results_from_athena()
        
        # Step 2: Get pending games from DynamoDB
        pending_games = get_pending_games_from_dynamodb()
        
        if not pending_games:
            logger.info("No pending games found that need result updates")
            return {
                'statusCode': 200,
                'body': json.dumps('No pending games found that need result updates')
            }
        
        # Step 3: Process each pending game and determine the result
        updated_games = []
        for game in pending_games:
            result = determine_bet_result(game, results_df)
            if result:
                updated_game = update_game_with_result(game, result)
                updated_games.append(updated_game)
                
                # Log the result
                game_id = game.get('id', 'unknown')
                home_team = game.get('home_team', 'unknown')
                away_team = game.get('away_team', 'unknown')
                prediction = game.get('prediction', 'unknown')
                result_correct = result.get('prediction_correct', False)
                actual_result = result.get('actual_result', 'unknown')
                
                logger.info(f"Game {game_id} ({home_team} vs {away_team}): Prediction {prediction}, " 
                           f"Actual {actual_result}, Correct: {result_correct}")
        
        logger.info(f"Processed {len(pending_games)} pending games, found results for {len(updated_games)} games")
        
        # Step 4: Update DynamoDB with the results
        if updated_games:
            update_dynamodb_with_results(updated_games)
        
        return {
            'statusCode': 200,
            'body': json.dumps(f'Successfully updated {len(updated_games)} games with results')
        }
    
    except Exception as e:
        logger.error(f"Error in Lambda execution: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }

# For local testing
if __name__ == "__main__":
    lambda_handler(None, None) 