from flask import Flask, render_template, send_from_directory, render_template_string, jsonify, request
from db import (
    get_profitable_games_from_db, 
    get_all_predictions_from_db, 
    get_profitable_games_from_dynamodb, 
    get_all_predictions_from_dynamodb,
    get_prediction_metadata_from_dynamodb,
    get_unique_leagues_from_db,
    get_unique_models_from_db,
    get_paginated_predictions_from_db,
    get_paginated_predictions_from_dynamodb
)
import logging
from datetime import datetime, timedelta
import os
import boto3
from boto3.dynamodb.conditions import Attr
from botocore.config import Config
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment from environment variables
FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
PORT = int(os.environ.get('PORT', 8081))
HOST = os.environ.get('HOST', '0.0.0.0')
DEBUG = os.environ.get('FLASK_DEBUG', '1') == '1'
USE_DYNAMODB = os.environ.get('USE_DYNAMODB', '1') == '1'
# DynamoDB tables
ALL_PREDICTIONS_TABLE = "all-predicted-games"
PROFITABLE_GAMES_TABLE = os.environ.get('PROFITABLE_GAMES_TABLE', 'profitable-games')

app = Flask(__name__, static_folder='static')

@app.route('/health')
def health_check():
    """Health check endpoint for Elastic Beanstalk"""
    return jsonify({'status': 'healthy', 'environment': FLASK_ENV}), 200

# Sample games as fallback
SAMPLE_GAMES = [
    {
        'id': '1',
        'result_id': 'sample_result_1',
        'home_team': 'Team A',
        'away_team': 'Team B',
        'league': 'Premier League',
        'match_time': datetime.strptime('2024-04-05 20:00:00', '%Y-%m-%d %H:%M:%S'),
        'prediction': 'Home Win',
        'confidence': 0.85,
        'status': 'completed',
        'odds': {'home': 2.0, 'draw': 3.5, 'away': 4.0},
        'is_profitable': True,
        'expected_value': 0.35,
        'model_type': 'lstm_v2',
        'home_win_prob': 0.65,
        'draw_prob': 0.20,
        'away_win_prob': 0.15,
        'home_win_ev': 0.35,
        'draw_ev': -0.1,
        'away_win_ev': -0.2,
        'home_win_is_profitable': True,
        'draw_is_profitable': False,
        'away_win_is_profitable': False,
        'prediction_result': True,
        'actual_result': 'Home Win',
        'home_score': 2,
        'away_score': 0,
        'final_home_score': 2,
        'final_away_score': 0,
        'result_updated_at': datetime.now().isoformat()
    }
]

def ensure_game_has_required_fields(game):
    """Ensure a game object has all required fields for template rendering"""
    # Required fields for basic rendering
    required_fields = {
        'id': 'unknown',
        'result_id': None,
        'home_team': 'Unknown Team',
        'away_team': 'Unknown Team',
        'league': 'Unknown League',
        'match_time': datetime.now(),
        'prediction': 'No prediction available',
        'confidence': None,
        'status': 'upcoming',
        'odds': {'home': None, 'draw': None, 'away': None},
        'expected_value': None,
        'is_profitable': None,
        'model_type': 'Not available',
        'home_win_prob': None,
        'draw_prob': None,
        'away_win_prob': None,
        'home_win_ev': None,
        'draw_ev': None,
        'away_win_ev': None,
        'home_win_is_profitable': None,
        'draw_is_profitable': None,
        'away_win_is_profitable': None,
        'prediction_result': None,
        'actual_result': None,
        'home_score': None,
        'away_score': None,
        'final_home_score': None,
        'final_away_score': None,
        'result_updated_at': None
    }
    
    # Add missing fields with default values (only for required structure)
    for field, default in required_fields.items():
        if field not in game:
            game[field] = default
    
    # Ensure match_time is a valid datetime object
    if not isinstance(game['match_time'], datetime):
        try:
            # If it's a string, try to parse it
            if isinstance(game['match_time'], str):
                if ' ' in game['match_time']:  # Has both date and time
                    game['match_time'] = datetime.strptime(game['match_time'], '%Y-%m-%d %H:%M')
                else:  # Only has date
                    game['match_time'] = datetime.strptime(game['match_time'], '%Y-%m-%d')
            else:
                # If parsing fails or it's not a string, use current time
                game['match_time'] = datetime.now()
                logger.warning(f"Invalid match_time for game {game.get('id', 'unknown')}, using current time")
        except Exception as e:
            # If parsing fails, use current time
            game['match_time'] = datetime.now()
            logger.warning(f"Failed to parse match_time for game {game.get('id', 'unknown')}: {e}")
    
    # Add one minute to the match time to give users a buffer
    game['match_time'] = game['match_time'] + timedelta(minutes=1)
    
    # Make sure odds is a dictionary with the right keys, but don't add fictional odds
    if not isinstance(game['odds'], dict):
        game['odds'] = {'home': None, 'draw': None, 'away': None}
    else:
        for key in ['home', 'draw', 'away']:
            if key not in game['odds']:
                game['odds'][key] = None
    
    # Only normalize probabilities if all three probabilities are present and not None
    if game['home_win_prob'] is not None and game['draw_prob'] is not None and game['away_win_prob'] is not None:
        normalize_probabilities(game)
    
    return game

def normalize_probabilities(game):
    """Ensure that home_win_prob, draw_prob, and away_win_prob sum to exactly 1.0"""
    home_prob = game.get('home_win_prob', 0)
    draw_prob = game.get('draw_prob', 0)
    away_prob = game.get('away_win_prob', 0)
    
    # Calculate the sum of probabilities
    total_prob = home_prob + draw_prob + away_prob
    
    # Only normalize if the sum is greater than 0
    if total_prob > 0:
        game['home_win_prob'] = home_prob / total_prob
        game['draw_prob'] = draw_prob / total_prob
        game['away_win_prob'] = away_prob / total_prob

def calculate_prediction_stats(games):
    """Calculate prediction statistics from a list of games."""
    total_predictions = len(games)
    completed_games = [g for g in games if g.get('status') == 'completed']
    correct_predictions = [g for g in completed_games if g.get('prediction_result') == True]
    incorrect_predictions = [g for g in completed_games if g.get('prediction_result') == False]
    
    # Calculate win rate
    win_rate = len(correct_predictions) / len(completed_games) * 100 if completed_games else 0
    
    # Calculate average ROI
    total_return = 0
    total_investment = len(completed_games)  # Assuming 1 unit stake per game
    
    for game in completed_games:
        # Get the odds for the predicted outcome
        prediction = game.get('prediction', '')
        if prediction == 'Home Win':
            odds = game.get('odds', {}).get('home', 0)
        elif prediction == 'Draw':
            odds = game.get('odds', {}).get('draw', 0)
        elif prediction == 'Away Win':
            odds = game.get('odds', {}).get('away', 0)
        else:
            odds = 0
        
        if game.get('prediction_result') == True and odds > 0:
            # For winning bets, add the return (odds * stake)
            total_return += odds  # Assuming 1 unit stake
        # For losing bets, return is 0 (stake is lost)
    
    # ROI = (Total Return - Total Investment) / Total Investment * 100
    avg_roi = ((total_return - total_investment) / total_investment) * 100 if total_investment > 0 else 0
    
    return {
        'total_predictions': total_predictions,
        'win_rate': round(win_rate, 1),
        'successful_bets': len(correct_predictions),
        'unsuccessful_bets': len(incorrect_predictions),
        'avg_roi': round(avg_roi, 1)
    }

@app.route('/')
def index():
    try:
        # Get the selected model filter, default to 'lstm_100_12_v1'
        DEFAULT_MODEL = 'lstm_100_12_v1'
        selected_model = request.args.get('model_type', DEFAULT_MODEL) 
        logger.info(f"Selected model (default '{DEFAULT_MODEL}'): '{selected_model}'")
        
        # Get language preference, default to 'english'
        selected_language = request.args.get('lang', 'english')
        logger.info(f"Selected language: '{selected_language}'")
        
        # Get profitable predictions
        if USE_DYNAMODB:
            # Revert to fetching directly from the profitable games table
            logger.info(f"Fetching profitable games directly from DynamoDB table: {PROFITABLE_GAMES_TABLE}")
            
            # Optional: Keep some debug info if needed, but target the correct table
            debug_info = {}
            try:
                dynamodb = boto3.resource('dynamodb', region_name="il-central-1")
                table = dynamodb.Table(PROFITABLE_GAMES_TABLE) 
                count_response = table.scan(Select='COUNT')
                debug_info['profitable_table_total_items'] = count_response.get('Count', 0)
                logger.info(f"DynamoDB debug info for profitable table: {debug_info}")
            except Exception as debug_e:
                logger.warning(f"Error collecting debug info for profitable table: {debug_e}")
            
            # Use the function that reads from the profitable-games table
            games = get_profitable_games_from_dynamodb() 
            logger.info(f"Raw profitable games data count from DynamoDB: {len(games)}")
            
            # Debug info for first few profitable games
            if games and len(games) > 0:
                for i, game in enumerate(games[:3]):
                    # Log status to verify
                    logger.info(f"Sample profitable game {i+1}: id={game.get('id')}, model={game.get('model_name')}, status={game.get('status')}, match_time={game.get('match_time')}") 
            else:
                logger.warning("No profitable games found in profitable-games table")
        else:
            logger.info("Fetching profitable games from SQLite")
            games = get_profitable_games_from_db()
        
        if not games:
            logger.warning("No profitable games found, using sample data")
            games = SAMPLE_GAMES.copy()
        else:
            logger.info(f"Successfully retrieved {len(games)} profitable games for display")
        
        # Ensure all games have required fields
        processed_games = [ensure_game_has_required_fields(game) for game in games]
        
        # Apply language preference if set to English
        if selected_language == 'english':
            for game in processed_games:
                if 'english_home_team' in game and game['english_home_team']:
                    game['display_home_team'] = game['english_home_team']
                else:
                    game['display_home_team'] = game['home_team']
                    
                if 'english_away_team' in game and game['english_away_team']:
                    game['display_away_team'] = game['english_away_team']
                else:
                    game['display_away_team'] = game['away_team']
                    
                if 'english_league' in game and game['english_league']:
                    game['display_league'] = game['english_league']
                else:
                    game['display_league'] = game['league']
        else:
            # Use original team and league names
            for game in processed_games:
                game['display_home_team'] = game['home_team']
                game['display_away_team'] = game['away_team']
                game['display_league'] = game['league']
        
        # Get unique model names for the toggles
        model_types = sorted(list(set(game.get('model_name', 'Unknown Model') for game in processed_games)))
        logger.info(f"Available models: {model_types}")
        
        # Apply model filter if selected
        if selected_model:
            before_filter = len(processed_games)
            processed_games = [game for game in processed_games if 
                              game.get('model_name') and selected_model.lower() in game.get('model_name', '').lower()]
            logger.info(f"Filtered to {len(processed_games)} games from {before_filter} for model: {selected_model}")
            
            if len(processed_games) == 0:
                logger.warning(f"Filter '{selected_model}' resulted in zero games. Available models were: {model_types}")
        
        # Count upcoming games for info
        current_time = datetime.now()
        upcoming_games = [g for g in processed_games if g.get('match_time', current_time) > current_time]
        logger.info(f"Found {len(upcoming_games)} upcoming games out of {len(processed_games)} total profitable games displayed")
        
        # Calculate statistics based on the displayed profitable games
        stats = calculate_prediction_stats(processed_games)
        
        return render_template('index.html', 
                              games=processed_games, 
                              stats=stats,
                              model_types=model_types,
                              selected_model=selected_model,
                              selected_language=selected_language)
    
    except Exception as e:
        logger.error(f"Error fetching games for index: {str(e)}", exc_info=True)
        sample_games = [ensure_game_has_required_fields(game) for game in SAMPLE_GAMES]
        stats = calculate_prediction_stats(sample_games)
        return render_template('index.html', 
                              games=sample_games, 
                              stats=stats,
                              model_types=['lstm_100_12_v1', 'lstm_100_12_v2'],
                              selected_model='',
                              selected_language='english')

@app.route('/all')
def all_predictions():
    try:
        # Get filter parameters from request
        selected_league = request.args.get('league', '')
        selected_model = request.args.get('model_type', 'lstm_100_12_v1')
        selected_prediction = request.args.get('prediction_type', '')
        selected_status = request.args.get('status', '')
        selected_result = request.args.get('result', '')
        selected_ev = request.args.get('ev', '')
        
        # Get pagination parameters - increase default to show more predictions
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))  # Increased from 40 to 50
        
        # Get language preference, default to 'english'
        selected_language = request.args.get('lang', 'english')
        logger.info(f"Selected language: '{selected_language}'")
        
        # Get leagues and models for dropdowns - these should load quickly
        if USE_DYNAMODB:
            # Get just the metadata instead of all predictions
            leagues, model_types = get_prediction_metadata_from_dynamodb()
        else:
            # If using SQLite, fetch minimal data for dropdowns
            leagues = get_unique_leagues_from_db()
            model_types = get_unique_models_from_db()
        
        # Begin with an empty stats object
        stats = {
            'total_predictions': 0,
            'win_rate': 0,
            'successful_bets': 0,
            'unsuccessful_bets': 0,
            'avg_roi': 0
        }
        
        # Create a loading state template with filters but no predictions
        if request.args.get('init') == '1':
            # First load - return just the structure without predictions
            return render_template('all_predictions.html',
                                games=[],
                                stats=stats,
                                leagues=leagues,
                                model_types=model_types,
                                selected_league=selected_league,
                                selected_model=selected_model,
                                selected_prediction=selected_prediction,
                                selected_status=selected_status,
                                selected_result=selected_result,
                                selected_ev=selected_ev,
                                selected_language=selected_language,
                                page=page,
                                per_page=per_page,
                                total_pages=1,
                                total_count=0,
                                has_prev=False,
                                has_next=False,
                                loading=True)
        
        # Use a more efficient query if possible
        filtered_games = []
        total_count = 0
        logger.info(f"Fetching predictions with filters: league={selected_league}, model={selected_model}")
        
        if USE_DYNAMODB:
            filtered_games, total_count = get_paginated_predictions_from_dynamodb(
                page=page,
                per_page=per_page * 5,  # Get enough to account for grouping
                league=selected_league,
                model=selected_model,
                prediction=selected_prediction,
                status=selected_status,
                result=selected_result,
                ev=selected_ev
            )
            
            # --- Add detailed logging for fetched games ---
            logger.info(f"[DEBUG] Fetched {len(filtered_games)} games from DB function. Total count reported: {total_count}")
            if filtered_games:
                 logger.info(f"[DEBUG] First fetched game raw: {filtered_games[0]}")
            # --- End added logging ---
            
            # If we're using model filtering, also fetch the latest predictions regardless of model
            # to ensure we have a mix of filtered + latest predictions
            if selected_model and (len(filtered_games) < per_page):
                logger.info(f"Also fetching latest predictions regardless of model to ensure freshness")
                latest_games, _ = get_paginated_predictions_from_dynamodb(
                    page=1,
                    per_page=per_page * 2,
                    league=selected_league,
                    model='',  # No model filter
                    prediction=selected_prediction,
                    status=selected_status,
                    result=selected_result,
                    ev=selected_ev
                )
                
                # Add them to our filtered games
                for game in latest_games:
                    if not any(g.get('id') == game.get('id') for g in filtered_games):
                        filtered_games.append(game)
                
                logger.info(f"After adding latest games, now have {len(filtered_games)} total games")
                
        else:
            # For SQLite, fetch with pagination
            filtered_games, total_count = get_paginated_predictions_from_db(
                page=page, 
                per_page=per_page * 5,  # Get enough to account for grouping
                league=selected_league,
                model=selected_model,
                prediction=selected_prediction,
                status=selected_status,
                result=selected_result,
                ev=selected_ev
            )
            
            # --- Add similar detailed logging here if using SQLite ---
            logger.info(f"[DEBUG-SQLite] Fetched {len(filtered_games)} games from DB function. Total count reported: {total_count}")
            if filtered_games:
                logger.info(f"[DEBUG-SQLite] First fetched game raw: {filtered_games[0]}")
            # --- End added logging ---
            
            # If we're using model filtering, also fetch the latest predictions regardless of model
            if selected_model and (len(filtered_games) < per_page):
                logger.info(f"Also fetching latest predictions regardless of model to ensure freshness")
                latest_games, _ = get_paginated_predictions_from_db(
                    page=1,
                    per_page=per_page * 2,
                    league=selected_league,
                    model='',  # No model filter
                    prediction=selected_prediction,
                    status=selected_status,
                    result=selected_result,
                    ev=selected_ev
                )
                
                # Add them to our filtered games
                for game in latest_games:
                    if not any(g.get('id') == game.get('id') for g in filtered_games):
                        filtered_games.append(game)
                
                logger.info(f"After adding latest games, now have {len(filtered_games)} total games")
        
        if not filtered_games:
            logger.warning("No predictions found after filtering, using sample data")
            filtered_games = SAMPLE_GAMES[:per_page]
            total_count = len(SAMPLE_GAMES)
        else:
            logger.info(f"Retrieved {len(filtered_games)} predictions for page {page} (potential total in DB: {total_count})")
            
            # --- Add detailed logging before sorting ---
            logger.info(f"[DEBUG] Attempting to sort {len(filtered_games)} fetched games.")
            # Log sorting keys for first few games
            for i, game in enumerate(filtered_games[:3]):
                sort_key = get_sort_key_timestamp(game)
                logger.info(f"[DEBUG] Game {i+1} (ID: {game.get('id')}) - Raw Timestamp Fields: prediction_timestamp={game.get('prediction_timestamp')}, timestamp={game.get('timestamp')} -> Parsed Sort Key: {sort_key} (type: {type(sort_key)})")
            # --- End added logging ---
            
            # Sort filtered games by prediction timestamp to get the latest predictions first
            filtered_games.sort(
                key=lambda x: get_sort_key_timestamp(x),
                reverse=True  # newest first
            )
            
            # --- Add detailed logging after sorting ---
            logger.info(f"[DEBUG] Sorting complete. First 3 games after sorting:")
            for i, game in enumerate(filtered_games[:3]):
                 logger.info(f"[DEBUG] Sorted Game {i+1} (ID: {game.get('id')}) Timestamp: {get_sort_key_timestamp(game)}")
            # --- End added logging ---
        
        # Process the retrieved games
        processed_games = []
        for game in filtered_games:
            # Ensure game has all required fields
            game = ensure_game_has_required_fields(game)
            
            # Apply language preference
            if selected_language == 'english':
                if 'english_home_team' in game and game['english_home_team']:
                    game['display_home_team'] = game['english_home_team']
                else:
                    game['display_home_team'] = game['home_team']
                
                if 'english_away_team' in game and game['english_away_team']:
                    game['display_away_team'] = game['english_away_team']
                else:
                    game['display_away_team'] = game['away_team']
                
                if 'english_league' in game and game['english_league']:
                    game['display_league'] = game['english_league']
                else:
                    game['display_league'] = game['league']
            else:
                game['display_home_team'] = game['home_team']
                game['display_away_team'] = game['away_team']
                game['display_league'] = game['league']
            
            processed_games.append(game)
        
        # Group predictions by game ID and model
        game_model_predictions = {}
        for game in processed_games:
            game_id = game.get('id', '')
            model_name = game.get('model_name', 'Unknown Model')
            key = f"{game_id}_{model_name}"
            
            # Initialize list for this game-model combination if not yet in dictionary
            if key not in game_model_predictions:
                game_model_predictions[key] = []
                
            # Append this prediction to the appropriate list
            game_model_predictions[key].append(game)
        
        # --- Add detailed logging after grouping ---
        logger.info(f"[DEBUG] Grouped {len(processed_games)} processed games into {len(game_model_predictions)} unique game-model groups.")
        # --- End added logging ---
        
        # Process each group to find primary and history games
        primary_games = []
        total_primary_games = 0
        
        # Sort the keys by the newest prediction timestamp first
        sorted_keys = sorted(
            game_model_predictions.keys(),
            key=lambda k: max([get_sort_key_timestamp(x) for x in game_model_predictions[k]]),
            reverse=True  # newest first
        )
        
        for key in sorted_keys:
            predictions = game_model_predictions[key]
            # Sort by prediction timestamp (newest first)
            predictions.sort(
                key=lambda x: get_sort_key_timestamp(x),
                reverse=True
            )
            
            if predictions:
                # Mark the newest prediction as primary
                primary_game = predictions[0]
                primary_game['is_primary'] = True
                
                # Get historical games and ensure they're sorted newest first
                if len(predictions) > 1:
                    history_games = predictions[1:]
                    # Make sure historical games are also sorted with most recent first
                    history_games.sort(
                        key=lambda x: get_sort_key_timestamp(x),
                        reverse=True
                    )
                    primary_game['history_games'] = history_games
                else:
                    primary_game['history_games'] = []
                    
                primary_games.append(primary_game)
                total_primary_games += 1
                
                # Only include enough primary games to fill the page
                if len(primary_games) >= per_page:
                    break
        
        # --- Add detailed logging after selecting primary games ---
        logger.info(f"[DEBUG] Selected {len(primary_games)} primary games for display on page {page} (per_page={per_page}).")
        # --- End added logging ---
        
        # Calculate stats if we have data
        if primary_games:
            stats = calculate_prediction_stats(primary_games)
        
        # Get the total count returned by the database function
        # Renaming the variable for clarity
        db_total_count = total_count 
        final_total_count = db_total_count

        # Calculate count based on fetched/grouped games for logging/debugging
        unique_game_group_count = len(game_model_predictions)
        total_primary_games_on_page = len(primary_games)
        logger.info(f"DB returned count: {db_total_count}, Unique groups fetched: {unique_game_group_count}, Primary games on page: {total_primary_games_on_page}")

        # Sanity check log: If DB count is 0 but we processed games for the page.
        if final_total_count == 0 and total_primary_games_on_page > 0:
            logger.warning(f"Database function returned total_count=0, but {total_primary_games_on_page} primary games were processed for display.")
            # The template will handle showing "X games" instead of "X of 0 games"

        logger.info(f"Using final display count for pagination/display: {final_total_count}")
        
        # Calculate pagination metadata using the final_total_count
        total_pages = (final_total_count + per_page - 1) // per_page if final_total_count > 0 else 1
        has_prev = page > 1
        has_next = page < total_pages
        
        logger.info(f"Rendering page with {len(primary_games)} games.")
        return render_template('all_predictions.html',
                            games=primary_games,
                            stats=stats,
                            leagues=leagues,
                            model_types=model_types,
                            selected_league=selected_league,
                            selected_model=selected_model,
                            selected_prediction=selected_prediction,
                            selected_status=selected_status,
                            selected_result=selected_result,
                            selected_ev=selected_ev,
                            selected_language=selected_language,
                            page=page,
                            per_page=per_page,
                            total_pages=total_pages,
                            total_count=final_total_count,
                            has_prev=has_prev,
                            has_next=has_next,
                            loading=False)
    
    except Exception as e:
        logger.error(f"Error fetching all predictions: {str(e)}", exc_info=True)
        # Return a simple error response
        return render_template('all_predictions.html',
                            games=[],
                            stats={
                                'total_predictions': 0,
                                'win_rate': 0,
                                'successful_bets': 0,
                                'unsuccessful_bets': 0,
                                'avg_roi': 0
                            },
                            leagues=['Premier League'],
                            model_types=['lstm_100_12_v1', 'lstm_100_12_v2'],
                            selected_league='',
                            selected_model='lstm_100_12_v1',
                            selected_prediction='',
                            selected_status='',
                            selected_result='',
                            selected_ev='',
                            selected_language='english',
                            page=1,
                            per_page=50,
                            total_pages=1,
                            total_count=0,
                            has_prev=False,
                            has_next=False,
                            error=str(e))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# API endpoints for the React frontend
@app.route('/api/profitable-predictions')
def api_profitable_predictions():
    try:
        # Get language preference, default to 'english'
        selected_language = request.args.get('lang', 'english')
        logger.info(f"API: Selected language: '{selected_language}'")
        
        if USE_DYNAMODB:
            logger.info("Fetching profitable games from DynamoDB for API")
            games = get_profitable_games_from_dynamodb()
        else:
            logger.info("Fetching profitable games from SQLite for API")
            games = get_profitable_games_from_db()
        
        if not games:
            logger.warning("No profitable games found for API, using sample data")
            games = SAMPLE_GAMES.copy()
        else:
            logger.info(f"Successfully retrieved {len(games)} profitable games for API")
            # Sort by match time (newest first)
            games.sort(key=lambda x: x.get('match_time', datetime.min), reverse=True)
        
        # Ensure all games have required fields
        processed_games = [ensure_game_has_required_fields(game) for game in games]
        
        # Apply language preference if set to English
        if selected_language == 'english':
            for game in processed_games:
                if 'english_home_team' in game and game['english_home_team']:
                    game['display_home_team'] = game['english_home_team']
                else:
                    game['display_home_team'] = game['home_team']
                    
                if 'english_away_team' in game and game['english_away_team']:
                    game['display_away_team'] = game['english_away_team']
                else:
                    game['display_away_team'] = game['away_team']
                    
                if 'english_league' in game and game['english_league']:
                    game['display_league'] = game['english_league']
                else:
                    game['display_league'] = game['league']
        else:
            # Use original team and league names
            for game in processed_games:
                game['display_home_team'] = game['home_team']
                game['display_away_team'] = game['away_team']
                game['display_league'] = game['league']
        
        # Convert datetime objects to ISO format strings for JSON serialization
        for game in processed_games:
            if isinstance(game['match_time'], datetime):
                game['match_time'] = game['match_time'].isoformat()
        
        return jsonify({'predictions': processed_games, 'language': selected_language})
    
    except Exception as e:
        logger.error(f"Error fetching games for API: {str(e)}", exc_info=True)
        # Make sure sample games have all required fields
        sample_games = [ensure_game_has_required_fields(game) for game in SAMPLE_GAMES]
        
        # Convert datetime objects to ISO format strings
        for game in sample_games:
            if isinstance(game['match_time'], datetime):
                game['match_time'] = game['match_time'].isoformat()
        
        return jsonify({'predictions': sample_games, 'language': 'english'})

@app.route('/api/all-predictions')
def api_all_predictions():
    try:
        # Get language preference, default to 'english'
        selected_language = request.args.get('lang', 'english')
        logger.info(f"API: Selected language: '{selected_language}'")
        
        if USE_DYNAMODB:
            logger.info("Fetching all predictions from DynamoDB for API")
            games = get_all_predictions_from_dynamodb()
        else:
            logger.info("Fetching all predictions from SQLite for API")
            games = get_all_predictions_from_db()
        
        if not games:
            logger.warning("No predictions found for API, using sample data")
            games = SAMPLE_GAMES.copy()
        else:
            logger.info(f"Successfully retrieved {len(games)} predictions for API")
            # Sort by match time (newest first)
            games.sort(key=lambda x: x.get('match_time', datetime.min), reverse=True)
        
        # Ensure all games have required fields
        processed_games = [ensure_game_has_required_fields(game) for game in games]
        
        # Apply language preference if set to English
        if selected_language == 'english':
            for game in processed_games:
                if 'english_home_team' in game and game['english_home_team']:
                    game['display_home_team'] = game['english_home_team']
                else:
                    game['display_home_team'] = game['home_team']
                    
                if 'english_away_team' in game and game['english_away_team']:
                    game['display_away_team'] = game['english_away_team']
                else:
                    game['display_away_team'] = game['away_team']
                    
                if 'english_league' in game and game['english_league']:
                    game['display_league'] = game['english_league']
                else:
                    game['display_league'] = game['league']
        else:
            # Use original team and league names
            for game in processed_games:
                game['display_home_team'] = game['home_team']
                game['display_away_team'] = game['away_team']
                game['display_league'] = game['league']
        
        # Convert datetime objects to ISO format strings for JSON serialization
        for game in processed_games:
            if isinstance(game['match_time'], datetime):
                game['match_time'] = game['match_time'].isoformat()
        
        return jsonify({'predictions': processed_games, 'language': selected_language})
    
    except Exception as e:
        logger.error(f"Error fetching all predictions for API: {str(e)}", exc_info=True)
        # Make sure sample games have all required fields
        sample_games = [ensure_game_has_required_fields(game) for game in SAMPLE_GAMES]
        
        # Convert datetime objects to ISO format strings
        for game in sample_games:
            if isinstance(game['match_time'], datetime):
                game['match_time'] = game['match_time'].isoformat()
        
        return jsonify({'predictions': sample_games, 'language': 'english'})

@app.route('/api/models')
def api_models():
    """API endpoint to list all available models"""
    try:
        if USE_DYNAMODB:
            logger.info("Fetching all predictions from DynamoDB")
            games = get_all_predictions_from_dynamodb()
        else:
            logger.info("Fetching all predictions from SQLite")
            games = get_all_predictions_from_db()
        
        # Get unique model names
        model_names = sorted(list(set(game.get('model_name', 'Unknown Model') for game in games)))
        
        # Count games by model
        model_counts = {}
        for model in model_names:
            model_counts[model] = len([game for game in games if game.get('model_name') == model])
        
        # Format response
        response = {
            'models': model_names,
            'model_counts': model_counts,
            'total_games': len(games)
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error fetching models for API: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def safely_parse_timestamp(timestamp_value):
    """Safely parse a timestamp value which might be a string or datetime object.
    Ensures the returned datetime is timezone-naive.
    """
    logger.info(f"Parsing timestamp: {timestamp_value} (type: {type(timestamp_value).__name__})")
    
    parsed_dt = None # Initialize variable to store the parsed datetime
    
    if isinstance(timestamp_value, datetime):
        logger.info(f"  -> Already a datetime object: {timestamp_value}")
        parsed_dt = timestamp_value
        
    elif isinstance(timestamp_value, str):
        try:
            # Handle ISO format timestamps like '2025-05-01T09:00:38.636543' or with timezone
            if 'T' in timestamp_value:
                # Replace Z with +00:00 for timezone handling if needed
                clean_timestamp = timestamp_value.replace('Z', '+00:00')
                parsed_dt = datetime.fromisoformat(clean_timestamp)
                logger.info(f"  -> Parsed ISO format: {parsed_dt}")
                
            # Handle simple date string formats
            elif ' ' in timestamp_value:  # Has date and time
                parsed_dt = datetime.strptime(timestamp_value, '%Y-%m-%d %H:%M:%S')
                logger.info(f"  -> Parsed datetime format: {parsed_dt}")
                
            else:  # Only date
                parsed_dt = datetime.strptime(timestamp_value, '%Y-%m-%d')
                logger.info(f"  -> Parsed date format: {parsed_dt}")
                
        except Exception as e:
            logger.warning(f"Error parsing timestamp '{timestamp_value}': {e}")
            parsed_dt = None # Ensure it's None on error
    
    # If parsing failed or input was not datetime/string, return fallback
    if parsed_dt is None:
        logger.warning(f"  -> Failed to parse timestamp or invalid type, using fallback: {datetime.min}")
        return datetime.min
        
    # --- Make the datetime timezone-naive --- 
    if parsed_dt.tzinfo is not None:
        logger.info(f"  -> Converting timezone-aware datetime {parsed_dt} to naive.")
        parsed_dt = parsed_dt.replace(tzinfo=None)
        logger.info(f"  -> Resulting naive datetime: {parsed_dt}")
    # --- End timezone conversion ---
    
    return parsed_dt

def get_sort_key_timestamp(game):
    """Extract the timestamp to use for sorting a game."""
    # --- Add logging inside this function --- 
    logger.debug(f"get_sort_key_timestamp called for game ID: {game.get('id')}, prediction_id: {game.get('prediction_id')}")
    val_to_parse = None
    source = "none"
    
    if 'prediction_timestamp' in game and game['prediction_timestamp']:
        val_to_parse = game['prediction_timestamp']
        source = "prediction_timestamp"
    elif 'timestamp' in game and game['timestamp']:
        val_to_parse = game['timestamp']
        source = "timestamp"
    elif 'match_time' in game and game['match_time']:
        val_to_parse = game['match_time']
        source = "match_time"
        
    if val_to_parse:
        logger.debug(f"  Attempting to parse '{source}': {val_to_parse} (type: {type(val_to_parse).__name__})")
        parsed_time = safely_parse_timestamp(val_to_parse) # Use the existing safe parser
        logger.debug(f"  Parsed result: {parsed_time} (type: {type(parsed_time).__name__})")
        return parsed_time
    else:
        logger.warning(f"  No suitable timestamp found (prediction_timestamp, timestamp, match_time), returning datetime.min")
        return datetime.min
    # --- End added logging ---

def get_all_predictions_from_dynamodb():
    """Fetch all predictions from DynamoDB."""
    try:
        # Connect to DynamoDB
        dynamodb = boto3.resource('dynamodb', region_name="il-central-1")
        table = dynamodb.Table(ALL_PREDICTIONS_TABLE)
        
        logger.info(f"Fetching all predictions from DynamoDB table: {ALL_PREDICTIONS_TABLE}")
        
        # We'll use an index to fetch predictions efficiently
        # Using ModelTimeIndex for chronological order per model
        all_items = []
        
        # Configure retry logic to handle ProvisionedThroughputExceededException
        config = Config(
            retries = {
                'max_attempts': 10,
                'mode': 'adaptive'
            }
        )
        client = boto3.client('dynamodb', region_name="il-central-1", config=config)
        resource = boto3.resource('dynamodb', region_name="il-central-1", config=config)
        table = resource.Table(ALL_PREDICTIONS_TABLE)
        
        # First, get a list of unique model names by scanning the ModelTimeIndex
        # This is more efficient than scanning the entire table
        unique_models = set()
        try:
            # Use a smaller batch size to avoid throughput issues
            scan_params = {
                'ProjectionExpression': "model_name",
                'FilterExpression': boto3.dynamodb.conditions.Attr('model_name').exists(),
                'Limit': 25  # Use smaller batches
            }
            scan_response = table.scan(**scan_params)
            for item in scan_response.get('Items', []):
                if 'model_name' in item:
                    unique_models.add(item['model_name'])
            
            # Handle pagination if needed
            while 'LastEvaluatedKey' in scan_response:
                try:
                    # Add a small delay to avoid hitting throughput limits
                    time.sleep(0.5)
                    scan_params['ExclusiveStartKey'] = scan_response['LastEvaluatedKey']
                    scan_response = table.scan(**scan_params)
                    for item in scan_response.get('Items', []):
                        if 'model_name' in item:
                            unique_models.add(item['model_name'])
                except Exception as page_error:
                    logger.error(f"Error during pagination for model names: {str(page_error)}")
                    break  # Break the loop if we hit an error
        except Exception as e:
            logger.error(f"Error getting unique model names: {str(e)}")
            # Fall back to basic scan
            unique_models = []
        
        logger.info(f"Found {len(unique_models)} unique models in DynamoDB")
        
        # If we couldn't get unique models, fall back to a simple scan
        if not unique_models:
            logger.info("Falling back to simple scan for all items")
            try:
                # Use a smaller batch size and limit the number of items
                scan_params = {
                    'Limit': 50  # Get fewer items at a time
                }
                scan_response = table.scan(**scan_params)
                all_items = scan_response.get('Items', [])
                
                # Handle pagination if needed - limit to 5 pages to avoid throughput issues
                page_count = 1
                max_pages = 5
                while 'LastEvaluatedKey' in scan_response and page_count < max_pages:
                    try:
                        # Add a delay between scans
                        time.sleep(0.5)
                        scan_params['ExclusiveStartKey'] = scan_response['LastEvaluatedKey']
                        scan_response = table.scan(**scan_params)
                        all_items.extend(scan_response.get('Items', []))
                        page_count += 1
                    except Exception as page_error:
                        logger.error(f"Error during pagination for items: {str(page_error)}")
                        break
                
                logger.info(f"Fetched {len(all_items)} items in {page_count} pages")
            except Exception as scan_error:
                logger.error(f"Error during simple scan: {str(scan_error)}")
                all_items = []
        else:
            # Query each model separately using the GSI
            # Limit to first 3 models to avoid throughput issues
            for model_name in list(unique_models)[:3]:
                try:
                    # We're using the ModelTimeIndex GSI to get all predictions for a specific model
                    query_params = {
                        'IndexName': 'ModelTimeIndex',
                        'KeyConditionExpression': boto3.dynamodb.conditions.Key('model_name').eq(model_name),
                        'Limit': 50  # Limit items per query
                    }
                    query_response = table.query(**query_params)
                    
                    model_items = query_response.get('Items', [])
                    
                    # Handle pagination if needed - limit to 3 pages per model
                    page_count = 1
                    max_pages = 3
                    while 'LastEvaluatedKey' in query_response and page_count < max_pages:
                        try:
                            # Add a delay between queries
                            time.sleep(0.5)
                            query_params['ExclusiveStartKey'] = query_response['LastEvaluatedKey']
                            query_response = table.query(**query_params)
                            model_items.extend(query_response.get('Items', []))
                            page_count += 1
                        except Exception as page_error:
                            logger.error(f"Error during pagination for model {model_name}: {str(page_error)}")
                            break
                    
                    logger.info(f"Fetched {len(model_items)} items for model {model_name} in {page_count} pages")
                    all_items.extend(model_items)
                    
                except Exception as model_error:
                    logger.error(f"Error querying model {model_name}: {str(model_error)}")
                    # Continue with the next model
                    continue
        
        logger.info(f"Found {len(all_items)} total predictions in DynamoDB")
        
        # Process the items and format them for the template
        games = []
        for item in all_items:
            # Parse the prediction timestamp
            prediction_timestamp = item.get('timestamp', datetime.now().isoformat())
            if isinstance(prediction_timestamp, str):
                try:
                    prediction_timestamp = datetime.fromisoformat(prediction_timestamp)
                except:
                    prediction_timestamp = datetime.now()
                
            # Parse match time
            match_time_str = item.get('match_time_str')
            match_time = None
            if match_time_str:
                try:
                    match_time = datetime.strptime(match_time_str, '%Y-%m-%d %H:%M')
                except:
                    try:
                        match_time = datetime.fromisoformat(match_time_str)
                    except:
                        match_time = datetime.now()
            else:
                event_date = item.get('event_date')
                game_time = item.get('game_time')
                if event_date and game_time:
                    try:
                        match_time = datetime.strptime(f"{event_date} {game_time}", '%Y-%m-%d %H:%M')
                    except:
                        match_time = datetime.now()
                else:
                    match_time = datetime.now()
            
            # Determine the prediction based on EV or is_profitable flags
            home_win_ev = float(item.get('home_win_ev', 0))
            draw_ev = float(item.get('draw_ev', 0))
            away_win_ev = float(item.get('away_win_ev', 0))
            
            home_win_profitable = item.get('home_win_is_profitable', False)
            draw_profitable = item.get('draw_is_profitable', False)
            away_profitable = item.get('away_win_is_profitable', False)
            
            # Choose prediction based on expected value
            prediction = 'No prediction'
            highest_ev = max(home_win_ev, draw_ev, away_win_ev)
            if highest_ev == home_win_ev:
                prediction = 'Home Win'
            elif highest_ev == draw_ev:
                prediction = 'Draw'
            elif highest_ev == away_win_ev:
                prediction = 'Away Win'
            
            # Format the game dict
            game = {
                'id': item.get('game_id', item.get('id', 'unknown')),  # Use game_id or fall back to id
                'home_team': item.get('home_team', 'Unknown Team'),
                'away_team': item.get('away_team', 'Unknown Team'),
                'league': item.get('league', 'Unknown League'),
                # Add English versions of team names and league
                'english_home_team': item.get('english_home_team', ''),
                'english_away_team': item.get('english_away_team', ''),
                'english_league': item.get('english_league', ''),
                'match_time': match_time,
                'prediction': prediction,
                'confidence': max(float(item.get('home_win_prob', 0)), 
                                float(item.get('draw_prob', 0)), 
                                float(item.get('away_win_prob', 0))),
                'status': item.get('status', 'upcoming'),
                'odds': {
                    'home': float(item.get('home_odds', 0)),
                    'draw': float(item.get('draw_odds', 0)),
                    'away': float(item.get('away_odds', 0))
                },
                'is_profitable': item.get('is_profitable', 0) == 1,
                'expected_value': highest_ev,
                'model_name': item.get('model_name', 'Unknown Model'),
                'model_type': item.get('model_type', 'Unknown Type'),
                'home_win_prob': float(item.get('home_win_prob', 0)),
                'draw_prob': float(item.get('draw_prob', 0)),
                'away_win_prob': float(item.get('away_win_prob', 0)),
                'home_win_ev': home_win_ev,
                'draw_ev': draw_ev,
                'away_win_ev': away_win_ev,
                'home_win_is_profitable': home_win_profitable,
                'draw_is_profitable': draw_profitable,
                'away_win_is_profitable': away_profitable,
                'prediction_timestamp': prediction_timestamp,
                # Additional fields for completed games
                'prediction_result': item.get('prediction_result'),
                'actual_result': item.get('actual_result'),
                'home_score': item.get('home_score'),
                'away_score': item.get('away_score'),
                'final_home_score': item.get('final_home_score'),
                'final_away_score': item.get('final_away_score'),
                'result_updated_at': item.get('result_updated_at')
            }
            
            games.append(game)
        
        # Sort games by match time (upcoming first, then by confidence for same day)
        games.sort(key=lambda x: (x['match_time'], -x['confidence']))
        
        return games
    
    except Exception as e:
        logger.error(f"Error fetching all predictions from DynamoDB: {str(e)}", exc_info=True)
        return []

if __name__ == '__main__':
    app.run(debug=DEBUG, host=HOST, port=PORT) 