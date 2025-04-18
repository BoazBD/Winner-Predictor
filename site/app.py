from flask import Flask, render_template, send_from_directory, render_template_string, jsonify, request
from db import get_profitable_games_from_db, get_all_predictions_from_db, get_profitable_games_from_dynamodb, get_all_predictions_from_dynamodb
import logging
from datetime import datetime, timedelta
import os
import boto3
from boto3.dynamodb.conditions import Attr

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
        # Get the selected model filter
        selected_model = request.args.get('model_type', '')
        logger.info(f"Model filter selected: '{selected_model}'")
        
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
                              selected_model=selected_model)
    
    except Exception as e:
        logger.error(f"Error fetching games for index: {str(e)}", exc_info=True)
        sample_games = [ensure_game_has_required_fields(game) for game in SAMPLE_GAMES]
        stats = calculate_prediction_stats(sample_games)
        return render_template('index.html', 
                              games=sample_games, 
                              stats=stats,
                              model_types=['lstm_100_12_v1', 'lstm_100_12_v2'],
                              selected_model='')

@app.route('/all')
def all_predictions():
    try:
        # Get filter parameters from request
        selected_league = request.args.get('league', '')
        selected_model = request.args.get('model_type', '')
        selected_prediction = request.args.get('prediction_type', '')
        selected_status = request.args.get('status', '')
        selected_result = request.args.get('result', '')
        selected_ev = request.args.get('ev', '')
        
        # Get all predictions
        if USE_DYNAMODB:
            logger.info("Fetching all predictions from DynamoDB")
            games = get_all_predictions_from_dynamodb()
        else:
            logger.info("Fetching all predictions from SQLite")
            games = get_all_predictions_from_db()
        
        if not games:
            logger.warning("No predictions found, using sample data")
            games = SAMPLE_GAMES.copy()
        else:
            logger.info(f"Successfully retrieved {len(games)} predictions")
        
        # Ensure all games have required fields
        processed_games = [ensure_game_has_required_fields(game) for game in games]
        
        # Group predictions by game ID and model name
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
        
        # Sort each group by prediction timestamp (newest first) and mark primary prediction
        ordered_games = []
        for key, predictions in game_model_predictions.items():
            # Check if prediction_timestamp exists and use it to sort
            has_timestamps = all(game.get('prediction_timestamp') for game in predictions)
            
            if has_timestamps:
                # Sort by prediction_timestamp (newest first)
                predictions.sort(
                    key=lambda x: x['prediction_timestamp'] if isinstance(x['prediction_timestamp'], datetime) 
                    else datetime.fromisoformat(x['prediction_timestamp']) if isinstance(x['prediction_timestamp'], str) 
                    else datetime.min, 
                    reverse=True
                )
            
            # Mark the newest (first) prediction as primary, the rest as historical
            for i, pred in enumerate(predictions):
                pred['is_primary'] = (i == 0)
                pred['history_index'] = i
                
            # Add all predictions to the ordered list
            ordered_games.extend(predictions)
        
        logger.info(f"After organizing: {len(ordered_games)} total predictions, {sum(1 for g in ordered_games if g.get('is_primary'))} primary")
        
        # Apply filters (we filter after organizing to maintain prediction groups)
        filtered_games = ordered_games
        
        if selected_league:
            filtered_games = [game for game in filtered_games if game.get('league') == selected_league]
            
        if selected_model:
            # Use model_name instead of model_type for filtering - use case-insensitive contains
            before_filter = len(filtered_games)
            filtered_games = [game for game in filtered_games if 
                             game.get('model_name') and selected_model.lower() in game.get('model_name', '').lower()]
            logger.info(f"Filtered to {len(filtered_games)} games from {before_filter} for model: {selected_model}")
            
        if selected_prediction:
            filtered_games = [game for game in filtered_games if game.get('prediction') == selected_prediction]
            
        if selected_status:
            current_time = datetime.now()
            if selected_status == 'upcoming':
                filtered_games = [game for game in filtered_games if 
                                  game.get('status') in ['pending', 'upcoming'] or
                                  game.get('match_time', current_time) > current_time]
            elif selected_status == 'completed':
                filtered_games = [game for game in filtered_games if 
                                  game.get('status') == 'completed' or
                                  game.get('match_time', current_time) < current_time]
        
        if selected_result:
            if selected_result == 'correct':
                filtered_games = [game for game in filtered_games if game.get('prediction_result') == True]
            elif selected_result == 'incorrect':
                filtered_games = [game for game in filtered_games if game.get('prediction_result') == False]
                
        if selected_ev:
            if selected_ev == 'high':
                filtered_games = [game for game in filtered_games if game.get('expected_value', 0) > 0.1]
            elif selected_ev == 'medium':
                filtered_games = [game for game in filtered_games if 0.05 <= game.get('expected_value', 0) <= 0.1]
            elif selected_ev == 'low':
                filtered_games = [game for game in filtered_games if 0 < game.get('expected_value', 0) < 0.05]
        
        # Get unique leagues for filter dropdown
        leagues = sorted(list(set(game.get('league', 'Unknown League') for game in processed_games)))
        
        # Get unique model names for filter dropdown - use model_name instead of model_type
        model_types = sorted(list(set(game.get('model_name', 'Unknown Model') for game in processed_games)))
        logger.info(f"Available models: {model_types}")
        
        # Count upcoming games for info
        current_time = datetime.now()
        upcoming_games = [g for g in filtered_games if g.get('match_time', current_time) > current_time]
        logger.info(f"Found {len(upcoming_games)} upcoming games out of {len(filtered_games)} filtered games")
        
        # Calculate statistics
        stats = calculate_prediction_stats(processed_games)
        
        return render_template('all_predictions.html', 
                              games=filtered_games, 
                              stats=stats,
                              leagues=leagues,
                              model_types=model_types,
                              selected_league=selected_league,
                              selected_model=selected_model,
                              selected_prediction=selected_prediction,
                              selected_status=selected_status,
                              selected_result=selected_result,
                              selected_ev=selected_ev)
    
    except Exception as e:
        logger.error(f"Error fetching all predictions: {str(e)}", exc_info=True)
        # Make sure sample games have all required fields
        sample_games = [ensure_game_has_required_fields(game) for game in SAMPLE_GAMES]
        stats = calculate_prediction_stats(sample_games)
        return render_template('all_predictions.html', 
                              games=sample_games, 
                              stats=stats,
                              leagues=['Premier League'],
                              model_types=['lstm_100_12_v1', 'lstm_100_12_v2'],
                              selected_league='',
                              selected_model='',
                              selected_prediction='',
                              selected_status='',
                              selected_result='',
                              selected_ev='')

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
        
        # Convert datetime objects to ISO format strings for JSON serialization
        for game in processed_games:
            if isinstance(game['match_time'], datetime):
                game['match_time'] = game['match_time'].isoformat()
        
        return jsonify({'predictions': processed_games})
    
    except Exception as e:
        logger.error(f"Error fetching games for API: {str(e)}", exc_info=True)
        # Make sure sample games have all required fields
        sample_games = [ensure_game_has_required_fields(game) for game in SAMPLE_GAMES]
        
        # Convert datetime objects to ISO format strings
        for game in sample_games:
            if isinstance(game['match_time'], datetime):
                game['match_time'] = game['match_time'].isoformat()
        
        return jsonify({'predictions': sample_games})

@app.route('/api/all-predictions')
def api_all_predictions():
    try:
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
        
        # Convert datetime objects to ISO format strings for JSON serialization
        for game in processed_games:
            if isinstance(game['match_time'], datetime):
                game['match_time'] = game['match_time'].isoformat()
        
        return jsonify({'predictions': processed_games})
    
    except Exception as e:
        logger.error(f"Error fetching all predictions for API: {str(e)}", exc_info=True)
        # Make sure sample games have all required fields
        sample_games = [ensure_game_has_required_fields(game) for game in SAMPLE_GAMES]
        
        # Convert datetime objects to ISO format strings
        for game in sample_games:
            if isinstance(game['match_time'], datetime):
                game['match_time'] = game['match_time'].isoformat()
        
        return jsonify({'predictions': sample_games})

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
        
        # First, get a list of unique model names by scanning the ModelTimeIndex
        # This is more efficient than scanning the entire table
        unique_models = set()
        try:
            scan_response = table.scan(
                ProjectionExpression="model_name",
                FilterExpression=boto3.dynamodb.conditions.Attr('model_name').exists()
            )
            for item in scan_response.get('Items', []):
                if 'model_name' in item:
                    unique_models.add(item['model_name'])
            
            # Handle pagination if needed
            while 'LastEvaluatedKey' in scan_response:
                scan_response = table.scan(
                    ProjectionExpression="model_name",
                    FilterExpression=boto3.dynamodb.conditions.Attr('model_name').exists(),
                    ExclusiveStartKey=scan_response['LastEvaluatedKey']
                )
                for item in scan_response.get('Items', []):
                    if 'model_name' in item:
                        unique_models.add(item['model_name'])
        except Exception as e:
            logger.error(f"Error getting unique model names: {str(e)}")
            # Fall back to basic scan
            unique_models = []
        
        logger.info(f"Found {len(unique_models)} unique models in DynamoDB")
        
        # If we couldn't get unique models, fall back to a simple scan
        if not unique_models:
            logger.info("Falling back to simple scan for all items")
            scan_response = table.scan()
            all_items = scan_response.get('Items', [])
            
            # Handle pagination if needed
            while 'LastEvaluatedKey' in scan_response:
                scan_response = table.scan(
                    ExclusiveStartKey=scan_response['LastEvaluatedKey']
                )
                all_items.extend(scan_response.get('Items', []))
        else:
            # Query each model separately using the GSI
            for model_name in unique_models:
                try:
                    # We're using the ModelTimeIndex GSI to get all predictions for a specific model
                    query_response = table.query(
                        IndexName='ModelTimeIndex',
                        KeyConditionExpression=boto3.dynamodb.conditions.Key('model_name').eq(model_name)
                    )
                    
                    model_items = query_response.get('Items', [])
                    
                    # Handle pagination if needed
                    while 'LastEvaluatedKey' in query_response:
                        query_response = table.query(
                            IndexName='ModelTimeIndex',
                            KeyConditionExpression=boto3.dynamodb.conditions.Key('model_name').eq(model_name),
                            ExclusiveStartKey=query_response['LastEvaluatedKey']
                        )
                        model_items.extend(query_response.get('Items', []))
                    
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