from flask import Flask, render_template, send_from_directory, render_template_string, jsonify, request
from db import get_profitable_games_from_db, get_all_predictions_from_db, get_profitable_games_from_dynamodb, get_all_predictions_from_dynamodb
import logging
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment from environment variables
FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
PORT = int(os.environ.get('PORT', 8080))
HOST = os.environ.get('HOST', '0.0.0.0')
DEBUG = os.environ.get('FLASK_DEBUG', '1') == '1'
USE_DYNAMODB = os.environ.get('USE_DYNAMODB', '1') == '1'
# DynamoDB tables
ALL_PREDICTIONS_TABLE = os.environ.get('ALL_PREDICTIONS_TABLE', 'all-predicted-games')
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
    total_roi = 0
    for game in completed_games:
        if game.get('prediction_result') == True:
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
            total_roi += (odds - 1) if odds > 0 else 0
    
    avg_roi = total_roi / len(completed_games) * 100 if completed_games else 0
    
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
        # Get profitable predictions
        if USE_DYNAMODB:
            logger.info("Fetching profitable games from DynamoDB")
            games = get_profitable_games_from_dynamodb()
        else:
            logger.info("Fetching profitable games from SQLite")
            games = get_profitable_games_from_db()
        
        if not games:
            logger.warning("No profitable games found, using sample data")
            games = SAMPLE_GAMES.copy()
        else:
            logger.info(f"Successfully retrieved {len(games)} profitable games")
            # Sort by match time (newest first)
            games.sort(key=lambda x: x.get('match_time', datetime.min), reverse=True)
        
        # Ensure all games have required fields
        processed_games = [ensure_game_has_required_fields(game) for game in games]
        
        # Calculate statistics
        stats = calculate_prediction_stats(processed_games)
        
        return render_template('index.html', games=processed_games, stats=stats)
    
    except Exception as e:
        logger.error(f"Error fetching games: {str(e)}", exc_info=True)
        # Make sure sample games have all required fields
        sample_games = [ensure_game_has_required_fields(game) for game in SAMPLE_GAMES]
        stats = calculate_prediction_stats(sample_games)
        return render_template('index.html', games=sample_games, stats=stats)

@app.route('/all')
def all_predictions():
    try:
        # Get filter parameters from request
        selected_league = request.args.get('league', '')
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
            # Sort by match time (newest first)
            games.sort(key=lambda x: x.get('match_time', datetime.min), reverse=True)
        
        # Ensure all games have required fields
        processed_games = [ensure_game_has_required_fields(game) for game in games]
        
        # Apply filters
        filtered_games = processed_games
        
        if selected_league:
            filtered_games = [game for game in filtered_games if game.get('league') == selected_league]
            
        if selected_prediction:
            filtered_games = [game for game in filtered_games if game.get('prediction') == selected_prediction]
            
        if selected_status:
            if selected_status == 'upcoming':
                filtered_games = [game for game in filtered_games if game.get('status') in ['pending', 'upcoming']]
            elif selected_status == 'completed':
                filtered_games = [game for game in filtered_games if game.get('status') == 'completed']
        
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
        
        # Calculate statistics
        stats = calculate_prediction_stats(processed_games)
        
        return render_template('all_predictions.html', 
                              games=filtered_games, 
                              stats=stats,
                              leagues=leagues,
                              selected_league=selected_league,
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
                              selected_league='',
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

if __name__ == '__main__':
    app.run(debug=DEBUG, host=HOST, port=PORT) 